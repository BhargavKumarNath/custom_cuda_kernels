"""Phase 4: tensor-core MatMul+Bias vs the CUDA-core v2 kernel vs cuBLAS (F.linear).

Run the script twice: once normally (WMMA path active) and once with CUSTOM_CUDA_DISABLE_WMMA=1 (v2 path).
Method as in the other benchmarks: CUDA events, 256 MB L2 flush, 10 warm-ups, 100 measured, median + IQR.
Usage: python benchmarks/phase4_matmul_tc.py <label> -> results/phase4/matmul_tc_<label>.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from custom_cuda.kernels.matmul_add_bias import matmul_add_bias  # noqa: E402

SHAPES = {"qkv_proj": (2048, 4096, 4096), "mlp_up": (2048, 4096, 11008), "mlp_down": (2048, 11008, 4096),
          "m512": (512, 4096, 4096), "m128": (128, 4096, 4096), "square_4096": (4096, 4096, 4096)}


def time_fn(fn, flush, warm=10, iters=100):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(iters):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    q = statistics.quantiles(ts, n=4)
    return statistics.median(ts), q[2] - q[0]


def main() -> None:
    label = sys.argv[1]
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    out = []
    for dt in (torch.float16, torch.bfloat16):
        for name, (m, k, n) in SHAPES.items():
            x = torch.randn(m, k, device="cuda", dtype=dt); w = torch.randn(n, k, device="cuda", dtype=dt) * 0.05
            b = torch.randn(n, device="cuda", dtype=dt)
            med_c, iqr_c = time_fn(lambda: matmul_add_bias(x, w, b), flush)
            med_b, iqr_b = time_fn(lambda: F.linear(x, w, b), flush)
            flops = 2 * m * k * n
            err = (matmul_add_bias(x, w, b).float() - F.linear(x.float(), w.float(), b.float())).abs().max().item()
            out.append({"dtype": str(dt).replace("torch.", ""), "shape": name, "m": m, "k": k, "n": n,
                        "custom_ms": med_c, "custom_iqr_ms": iqr_c, "cublas_ms": med_b, "cublas_iqr_ms": iqr_b,
                        "custom_tflops": flops / med_c / 1e9, "cublas_tflops": flops / med_b / 1e9,
                        "custom_over_cublas": med_b / med_c, "max_abs_err_vs_fp32": err})
            print(out[-1])
    Path(f"results/phase4/matmul_tc_{label}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
