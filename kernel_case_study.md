# Case study: Custom CUDA kernels for efficient model execution

## Problem

Model execution can spend substantial time moving data, allocating temporary tensors, and launching separate GPU operations. I built a library of twelve custom operation families to explore reducing those costs for transformer, MoE, retrieval, graph, and sequence workloads.

## Approach

I implemented the computations in **CUDA C++**, used **Rust/PyO3** for tensor validation and native bindings, and provided **Python APIs** for PyTorch integration. Kernels access existing GPU tensor storage through device pointers and run on PyTorch's current CUDA stream. This avoids copying tensor contents across the language boundary; outputs and scratch buffers still allocate memory.

The experiment pipeline was: establish a reference implementation → test numerical correctness → benchmark → investigate bottlenecks → optimize → keep or reject the change. A developer CLI supports verification, benchmarking, plotting, environment checks, and profiling.

## The twelve kernels

| Kernel | What it does | Important implementation detail |
|---|---|---|
| RMSNorm + residual | Adds a residual and normalizes each row | One fused launch, FP32 reduction, two input-reading passes |
| SwiGLU | Computes `SiLU(gate) * up` | Fuses activation/multiplication with vectorized access and scalar fallbacks |
| RoPE | Adds positional information by rotating Q/K features | Processes both tensors in one launch using half-split pairs |
| Linear cross-entropy | Computes vocabulary projection loss | Chunked logits and stable online log-sum-exp avoid a full logits matrix |
| Matmul + bias | Computes `x @ weight.T + bias` | Shared-memory tiling, register reuse, bias in the final write |
| MoE top-k router | Selects experts and routing weights | Warp-level softmax and repeated top-k selection; up to 256 experts |
| Token gather/combine | Rearranges tokens and combines expert outputs | Fuses weighted combination and avoids competing atomic writes |
| Cosine top-k | Finds the most similar candidate embeddings | Exhaustive partitioned search plus merge; avoids the full score matrix |
| Pairwise distance | Computes squared Euclidean distances | Precomputed row norms plus tiled dot products |
| Graph message passing | Aggregates current/previous neighbor features | Destination-indexed CSR gives each output a single owner |
| Viterbi | Finds the most likely hidden-state sequence | One persistent launch; timesteps remain sequential inside the kernel |
| FP8 quantization | Calculates scales and casts values to FP8 | Supports E4M3/E5M2 with tile-level or whole-tensor scaling |

## Evaluation and results

Benchmarks use CUDA events, warmup, cache-displacement work, and median/IQR reporting. Tests cover numerical comparisons, multiple dtypes, shape variations, and edge cases. This checkout successfully collected **1,665 tests** during documentation preparation; collection does not establish that all tests passed.

The README records these historical results on an **RTX 4070 Laptop GPU**:

- **SwiGLU:** approximately 2.4–6.4× faster than its eager reference.
- **Linear cross-entropy:** approximately 12.5–31.3× lower measured peak memory than its reference.
- **Viterbi:** approximately 20–120× faster than a Python/PyTorch timestep loop.
- **Integration example:** replacing RMSNorm, RoPE, and SwiGLU in a Llama-3-8B-shaped decoder block reduced reported latency from **65.401 ms to 50.836 ms**: **1.287× speedup**, or **22.3% lower latency**.

These are workload-specific historical reports, not fresh benchmark results. The integration uses one randomly initialized decoder block, not a pretrained model or production serving service. Raw benchmark CSVs are absent from this checkout.

## Key lessons and limitations

Fusion helps when it removes meaningful memory traffic or launch overhead. It does not automatically outperform optimized libraries: the custom GEMM and cosine search remain slower than cuBLAS-backed baselines.

Work distribution matters as much as arithmetic. Candidate partitioning fixed poor cosine-search utilization, while smaller blocks corrected a RoPE vectorization regression. Shared-memory staging for RMSNorm was rejected when measurements showed no benefit.

The operations are **forward-only**, and the repository contains no Kubernetes/vLLM deployment or classifier-quality evaluation. For Netcraft, the transferable strengths are inference efficiency, controlled experiments, numerical validation, and bottleneck investigation. Production adoption would additionally require realistic serving benchmarks and checks that precision, recall, and false-positive rates remain acceptable.
