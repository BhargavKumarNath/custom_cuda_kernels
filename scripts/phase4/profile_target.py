"""Run one below-target kernel at one shape for Nsight Compute (Phase 4).

Usage: ncu ... python scripts/phase4/profile_target.py <kernel> <case>
Runs 2 warm-up calls then 1 call to profile; ncu is told to skip the warm-ups.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CASES = {
    "matmul": {"qkv_bf16": (2048, 4096, 4096, torch.bfloat16), "qkv_fp16": (2048, 4096, 4096, torch.float16),
               "qkv_fp32": (2048, 4096, 4096, torch.float32)},
    "cosine": {"large_pool_fp16": (8, 50000, 768, 10, torch.float16), "small_pool_fp16": (8, 2000, 384, 10, torch.float16),
               "nsweep_500000_fp16": (8, 500000, 384, 10, torch.float16)},
    "pairwise": {"dim1536_fp16": (2048, 2048, 1536, torch.float16), "dim384_fp16": (2048, 2048, 384, torch.float16),
                 "dim1536_fp32": (2048, 2048, 1536, torch.float32)},
    "graph": {"n2000_fp16": (2000, 64, 40_000, torch.float16), "n20000_fp16": (20000, 64, 400_000, torch.float16),
              "n100000_fp16": (100000, 64, 2_000_000, torch.float16)},
}


def main() -> None:
    kernel, case = sys.argv[1], sys.argv[2]
    p = CASES[kernel][case]
    if kernel == "matmul":
        from custom_cuda.kernels.matmul_add_bias import matmul_add_bias
        m, k, n, dt = p
        x = torch.randn(m, k, device="cuda", dtype=dt); w = torch.randn(n, k, device="cuda", dtype=dt)
        b = torch.randn(n, device="cuda", dtype=dt)
        fn = lambda: matmul_add_bias(x, w, b)  # noqa: E731
    elif kernel == "cosine":
        from custom_cuda.kernels.cosine_topk import cosine_topk
        q, c, d, kk, dt = p
        qs = torch.randn(q, d, device="cuda", dtype=dt); cs = torch.randn(c, d, device="cuda", dtype=dt)
        fn = lambda: cosine_topk(qs, cs, kk)  # noqa: E731
    elif kernel == "pairwise":
        from custom_cuda.kernels.pairwise_distance import pairwise_distance_sq
        m, n, d, dt = p
        a = torch.randn(m, d, device="cuda", dtype=dt); b = torch.randn(n, d, device="cuda", dtype=dt)
        fn = lambda: pairwise_distance_sq(a, b)  # noqa: E731
    else:
        from custom_cuda.kernels.graph_message_passing import spatiotemporal_message_passing
        n, d, e, dt = p
        g = torch.Generator(device="cuda").manual_seed(0)
        xc = torch.randn(n, d, device="cuda", dtype=dt, generator=g); xp = torch.randn(n, d, device="cuda", dtype=dt, generator=g)
        ss = torch.randint(0, n, (e,), device="cuda", generator=g); sd = torch.randint(0, n, (e,), device="cuda", generator=g)
        sw = torch.rand(e, device="cuda", generator=g)
        ts = torch.arange(n, device="cuda"); tw = torch.rand(n, device="cuda", generator=g)
        fn = lambda: spatiotemporal_message_passing(xc, xp, ss, sd, sw, ts, ts, tw)  # noqa: E731
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
