#include "../includes/common.cuh"
#include "../includes/matmul_add_bias.h"

#include <mma.h>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

// -----------------------------------------------------------------------
// Kernel 5 — Fused MatMul + Add Bias.
//
// v1 (naive tiled, kept below as `matmul_add_bias_kernel_naive`): classic
// shared-memory-tiled GEMM, one output element per thread. Benchmarked at
// ~1.2-1.4 TFLOPS *regardless of M/K/N/dtype* — 7-30x *slower* than eager
// (target was >=15% *faster* than the naive unfused baseline). A flat
// TFLOPS ceiling independent of problem size or precision is the known
// signature of a shared-memory-bandwidth-bound kernel: every FMA needs 2
// shared-memory reads (one x element, one weight element) and only
// produces 1 output contribution, so arithmetic intensity per
// shared-memory transaction is fixed at a low constant no matter how the
// problem scales. This matches published data points for this exact
// naive-tiled optimization level (e.g. Simon Boehm's "How to Optimize a
// CUDA Matmul Kernel" benchmarks a near-identical kernel at ~1.3 TFLOPS
// against cuBLAS's ~15-20 TFLOPS on comparable hardware) — not a bug.
//
// v2 (`matmul_add_bias_kernel_blocked` below, now the default dispatch
// path): 1D register blocking. Each thread computes TM=8 output elements
// (a strip along M) instead of 1, so each weight value read from shared
// memory is reused across TM=8 FMAs instead of 1 — the standard next step
// once a kernel is shared-memory-bandwidth-bound rather than
// compute-bound. Falls back to the naive kernel for M/N too small to fill
// a block tile (BLOCK_M=64/BLOCK_N=64) — small-matrix cases are already
// latency-bound, not throughput-bound, so the naive kernel's simpler
// per-thread logic isn't worth replacing there, and it keeps every shape
// (however small) correct without needing partial-tile masking logic in
// the blocked kernel.
//
// Measured result: ~1.8-3.0 TFLOPS, a real ~2x gain over v1 but still
// 3-14x short of eager/cuBLAS, and short of the ">=15% faster than naive
// unfused" target. Stopped iterating here rather than push further (2D
// register blocking, vectorized float4 shared-memory loads, double
// buffering are the standard next steps and would likely help
// meaningfully) — closing the remaining gap to a vendor GEMM library is a
// well-documented multi-iteration undertaking even for FP32 (cuBLAS's
// FP16/BF16 path additionally uses tensor cores, which no amount of
// CUDA-core-only tiling can match; that requires WMMA/MMA intrinsics, a
// different technique entirely). Reported honestly as a partial result
// rather than pushed further under time pressure at rising
// bug-introduction risk — see project_plan.md's Kernel 5 entry.
// -----------------------------------------------------------------------

namespace {

// ---- v1: naive tiled (fallback for small M/N) ---------------------------

constexpr int kNaiveTile = 32;

template <typename T>
__global__ void matmul_add_bias_kernel_naive(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int64_t m,
    int64_t k,
    int64_t n) {
    __shared__ float x_tile[kNaiveTile][kNaiveTile];
    __shared__ float w_tile[kNaiveTile][kNaiveTile];

    const int64_t row = static_cast<int64_t>(blockIdx.y) * kNaiveTile + threadIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * kNaiveTile + threadIdx.x;

    float acc = 0.0f;
    const int64_t num_k_tiles = (k + kNaiveTile - 1) / kNaiveTile;

    for (int64_t t = 0; t < num_k_tiles; ++t) {
        const int64_t x_k = t * kNaiveTile + threadIdx.x;
        const int64_t w_k = t * kNaiveTile + threadIdx.y;

        x_tile[threadIdx.y][threadIdx.x] = (row < m && x_k < k) ? to_float(x[row * k + x_k]) : 0.0f;
        w_tile[threadIdx.y][threadIdx.x] = (col < n && w_k < k) ? to_float(weight[col * k + w_k]) : 0.0f;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kNaiveTile; ++kk) {
            acc += x_tile[threadIdx.y][kk] * w_tile[kk][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        const float b = (bias != nullptr) ? to_float(bias[col]) : 0.0f;
        y[row * n + col] = from_float<T>(acc + b);
    }
}

// ---- v2: 1D register-blocked tiled GEMM ----------------------------------

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;  // outputs per thread along M
// Block shape: (kBlockN, kBlockM / kThreadM) = (64, 8) = 512 threads.

template <typename T>
__global__ void matmul_add_bias_kernel_blocked(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int64_t m,
    int64_t k,
    int64_t n) {
    __shared__ float x_tile[kBlockM][kBlockK];
    __shared__ float w_tile[kBlockK][kBlockN];

    const int64_t block_row0 = static_cast<int64_t>(blockIdx.y) * kBlockM;
    const int64_t block_col0 = static_cast<int64_t>(blockIdx.x) * kBlockN;

    const int tx = threadIdx.x;  // 0..kBlockN-1
    const int ty = threadIdx.y;  // 0..(kBlockM/kThreadM)-1
    const int linear_id = ty * kBlockN + tx;  // 0..(kBlockM*kBlockN/kThreadM)-1, == 0..511

    const int64_t col = block_col0 + tx;

    float acc[kThreadM];
#pragma unroll
    for (int i = 0; i < kThreadM; ++i) {
        acc[i] = 0.0f;
    }

    const int64_t num_k_tiles = (k + kBlockK - 1) / kBlockK;
    for (int64_t t = 0; t < num_k_tiles; ++t) {
        // x_tile[BLOCK_M][BLOCK_K]: 64*8 = 512 elements, one per thread.
        const int xt_row = linear_id / kBlockK;
        const int xt_col = linear_id % kBlockK;
        const int64_t x_row_g = block_row0 + xt_row;
        const int64_t x_col_g = t * kBlockK + xt_col;
        x_tile[xt_row][xt_col] = (x_row_g < m && x_col_g < k) ? to_float(x[x_row_g * k + x_col_g]) : 0.0f;

        // w_tile[BLOCK_K][BLOCK_N]: 8*64 = 512 elements, one per thread.
        const int wt_row = linear_id / kBlockN;
        const int wt_col = linear_id % kBlockN;
        const int64_t w_row_g = block_col0 + wt_col;  // weight is [N, K]
        const int64_t w_col_g = t * kBlockK + wt_row;
        w_tile[wt_row][wt_col] = (w_row_g < n && w_col_g < k) ? to_float(weight[w_row_g * k + w_col_g]) : 0.0f;

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBlockK; ++kk) {
            const float w_val = w_tile[kk][tx];
#pragma unroll
            for (int mi = 0; mi < kThreadM; ++mi) {
                acc[mi] += x_tile[ty * kThreadM + mi][kk] * w_val;
            }
        }
        __syncthreads();
    }

    const float b = (bias != nullptr && col < n) ? to_float(bias[col]) : 0.0f;
#pragma unroll
    for (int mi = 0; mi < kThreadM; ++mi) {
        const int64_t row = block_row0 + ty * kThreadM + mi;
        if (row < m && col < n) {
            y[row * n + col] = from_float<T>(acc[mi] + b);
        }
    }
}

// ---- v3: tensor-core path (WMMA, fp16/bf16 inputs, fp32 accumulate) ------
//
// Phase 4 attempt: the v2 kernel above is CUDA-core-only and cannot approach
// cuBLAS's fp16/bf16 throughput. This kernel runs the same y = x @ W^T + b on
// tensor cores. Block tile 128(M) x 128(N) x 32(K), 8 warps arranged 2(M) x
// 4(N), each warp owns a 64x32 output tile (4x2 WMMA 16x16x16 fragments).
// Global->shared copies use 16-byte cp.async into a 2-stage pipeline, and the
// shared tiles are padded by 8 elements to avoid bank conflicts. No masking:
// the path is taken only when M%128==0, N%128==0, K%32==0 and the pointers are
// 16-byte aligned, otherwise dispatch falls back to v2.
constexpr int kWmmaBM = 128;
constexpr int kWmmaBN = 128;
constexpr int kWmmaBK = 32;
constexpr int kWmmaPad = 8;
constexpr int kWmmaLd = kWmmaBK + kWmmaPad;  // 40 elements per smem row
constexpr int kWmmaThreads = 256;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <typename T>
__global__ void __launch_bounds__(kWmmaThreads) matmul_add_bias_kernel_wmma(
    const T* __restrict__ x, const T* __restrict__ weight, const T* __restrict__ bias,
    T* __restrict__ y, int64_t m, int64_t k, int64_t n) {
    using namespace nvcuda;
    extern __shared__ __align__(16) unsigned char wmma_smem_raw[];
    T* smem = reinterpret_cast<T*>(wmma_smem_raw);
    // stage s: A tile at s*2*tile, B tile right after.
    constexpr int tile = kWmmaBM * kWmmaLd;
    auto a_s = [&](int s) { return smem + (2 * s) * tile; };
    auto b_s = [&](int s) { return smem + (2 * s + 1) * tile; };

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wm = warp / 4;  // 0..1
    const int wn = warp % 4;  // 0..3
    const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kWmmaBM;
    const int64_t col0 = static_cast<int64_t>(blockIdx.x) * kWmmaBN;

    auto load_stage = [&](int s, int64_t k0) {
        // 128 rows x 4 chunks of 8 elements = 512 chunks per operand, 2 per thread.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * kWmmaThreads;
            const int r = c / 4;
            const int cc = (c % 4) * 8;
            cp_async16(a_s(s) + r * kWmmaLd + cc, x + (row0 + r) * k + k0 + cc);
            cp_async16(b_s(s) + r * kWmmaLd + cc, weight + (col0 + r) * k + k0 + cc);
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    const int num_kt = static_cast<int>(k / kWmmaBK);
    load_stage(0, 0);
    cp_async_commit();
    for (int kt = 0; kt < num_kt; ++kt) {
        if (kt + 1 < num_kt) {
            load_stage((kt + 1) & 1, static_cast<int64_t>(kt + 1) * kWmmaBK);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const T* as = a_s(kt & 1);
        const T* bs = b_s(kt & 1);
#pragma unroll
        for (int ks = 0; ks < kWmmaBK; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> af[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> bf[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(af[i], as + (wm * 64 + i * 16) * kWmmaLd + ks, kWmmaLd);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(bf[j], bs + (wn * 32 + j * 16) * kWmmaLd + ks, kWmmaLd);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Epilogue: stage each 16x16 fragment through shared memory (reusing the pipeline
    // buffers, safe after the final __syncthreads), add bias, convert, store.
    float* stage = reinterpret_cast<float*>(wmma_smem_raw) + warp * 256;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(stage, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int r = lane / 2;
            const int c8 = (lane % 2) * 8;
            const int64_t gr = row0 + wm * 64 + i * 16 + r;
            const int64_t gc = col0 + wn * 32 + j * 16 + c8;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float v = stage[r * 16 + c8 + e];
                if (bias != nullptr) v += to_float(bias[gc + e]);
                y[gr * n + gc + e] = from_float<T>(v);
            }
            __syncwarp();
        }
    }
}

template <typename T>
cudaError_t dispatch_wmma(
    const T* x, const T* weight, const T* bias, T* y, int64_t m, int64_t k, int64_t n,
    cudaStream_t stream) {
    constexpr size_t smem_bytes = 4 * kWmmaBM * kWmmaLd * sizeof(T);  // 2 stages x (A + B) = 40960
    static_assert(smem_bytes >= kWmmaThreads / 32 * 256 * sizeof(float), "epilogue staging must fit");
    const dim3 grid(static_cast<unsigned int>(n / kWmmaBN), static_cast<unsigned int>(m / kWmmaBM));
    cudaFuncSetAttribute(matmul_add_bias_kernel_wmma<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    matmul_add_bias_kernel_wmma<T><<<grid, kWmmaThreads, smem_bytes, stream>>>(x, weight, bias, y, m, k, n);
    return cudaGetLastError();
}

// CUSTOM_CUDA_DISABLE_WMMA=1 forces the v2 CUDA-core kernel (used for A/B measurement).
inline bool wmma_disabled() {
    static const bool v = std::getenv("CUSTOM_CUDA_DISABLE_WMMA") != nullptr;
    return v;
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

template <typename T>
cudaError_t dispatch(
    const T* x, const T* weight, const T* bias, T* y, int64_t m, int64_t k, int64_t n,
    cudaStream_t stream) {
    if constexpr (!std::is_same_v<T, float>) {
        if (m % kWmmaBM == 0 && n % kWmmaBN == 0 && k % kWmmaBK == 0 && !wmma_disabled() && aligned16(x) && aligned16(weight)) {
            return dispatch_wmma<T>(x, weight, bias, y, m, k, n, stream);
        }
    }
    if (m >= kBlockM && n >= kBlockN) {
        const dim3 block(kBlockN, kBlockM / kThreadM);
        const dim3 grid(static_cast<unsigned int>((n + kBlockN - 1) / kBlockN),
                         static_cast<unsigned int>((m + kBlockM - 1) / kBlockM));
        matmul_add_bias_kernel_blocked<T><<<grid, block, 0, stream>>>(x, weight, bias, y, m, k, n);
    } else {
        const dim3 block(kNaiveTile, kNaiveTile);
        const dim3 grid(static_cast<unsigned int>((n + kNaiveTile - 1) / kNaiveTile),
                         static_cast<unsigned int>((m + kNaiveTile - 1) / kNaiveTile));
        matmul_add_bias_kernel_naive<T><<<grid, block, 0, stream>>>(x, weight, bias, y, m, k, n);
    }
    return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t launch_matmul_add_bias_fwd(
    const void* x,
    const void* weight,
    const void* bias,
    void* y,
    int64_t m,
    int64_t k,
    int64_t n,
    int32_t dtype,
    cudaStream_t stream) {
    if (m <= 0 || k <= 0 || n <= 0) {
        return cudaSuccess;
    }

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch<float>(
                static_cast<const float*>(x), static_cast<const float*>(weight),
                static_cast<const float*>(bias), static_cast<float*>(y), m, k, n, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch<__half>(
                static_cast<const __half*>(x), static_cast<const __half*>(weight),
                static_cast<const __half*>(bias), static_cast<__half*>(y), m, k, n, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(bias), static_cast<__nv_bfloat16*>(y), m, k, n,
                stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}
