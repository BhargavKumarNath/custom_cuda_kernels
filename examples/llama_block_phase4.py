"""Phase 4: the Llama-3-8B block with more than three kernels swapped in, plus run-to-run variance.

Variants (all bf16, batch 1, seq 4096, identical weights):
  eager     PyTorch eager block
  k123      Kernels 1+2+3 (RMSNorm+residual, SwiGLU, RoPE): the existing integration proof
  k1235     k123 + Kernel 5 (tensor-core MatMul+bias, bias=None) for all 7 projections (4 kernels)
  eager+ce  eager block + LM head + eager cross-entropy (vocab 32,000)
  k123+ce   k123 block + eager head/CE
  k123+k4   k123 block + Kernel 4 (fused linear cross-entropy)
  k1235+k4  k1235 block + Kernel 4 (5 kernels)
Each invocation measures every variant once (interleaved order, 50 measured iterations, L2 flush) and appends a
record to results/phase4/llama_runs.jsonl; run it N times for run-to-run variance (scripts/phase4/run_llama_repeats.sh).
Correctness of every variant vs eager is checked before timing.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
import llama_block as lb  # noqa: E402
from custom_cuda.kernels.linear_cross_entropy import linear_cross_entropy  # noqa: E402
from custom_cuda.kernels.matmul_add_bias import matmul_add_bias  # noqa: E402
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers  # noqa: E402

VOCAB = 32_000  # Llama-2 vocabulary: the 128,256 Llama-3 head needs ~5 GB of eager logits and does not fit next to the block on 8 GB
ITERS, WARM = 50, 10


class K5Linear(nn.Module):
    """nn.Linear drop-in that runs Kernel 5 (no bias)."""

    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        self.weight = lin.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        y = matmul_add_bias(x.reshape(-1, shp[-1]).contiguous(), self.weight, None)
        return y.view(*shp[:-1], -1)


class Block123(lb.CustomCUDA_LlamaBlock):
    pass


class Block1235(lb.CustomCUDA_LlamaBlock):
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            setattr(self, n, K5Linear(getattr(self, n)))


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, lb.SEQ_LEN, lb.HIDDEN_SIZE, dtype=lb.DTYPE, device="cuda")
    targets = torch.randint(0, VOCAB, (lb.SEQ_LEN,), device="cuda")
    head_w = (torch.randn(VOCAB, lb.HIDDEN_SIZE, dtype=lb.DTYPE, device="cuda") * 0.02)

    eager = lb.BaselineLlamaBlock().eval()
    b123 = Block123().eval()
    b1235 = Block1235().eval()
    for blk in (b123, b1235):  # share the eager block's parameters: identical weights, no duplicate memory
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            getattr(blk, n).weight = getattr(eager, n).weight
        blk.input_layernorm_weight = eager.input_layernorm_weight
        blk.post_attention_layernorm_weight = eager.post_attention_layernorm_weight

    def eager_ce(h):
        return F.cross_entropy(F.linear(h.view(-1, lb.HIDDEN_SIZE), head_w).float(), targets)

    def k4_ce(h):
        return linear_cross_entropy(h.view(-1, lb.HIDDEN_SIZE).contiguous(), head_w, targets)

    variants = {
        "eager": lambda: eager(x), "k123": lambda: b123(x), "k1235": lambda: b1235(x),
        "eager+ce": lambda: eager_ce(eager(x)), "k123+ce": lambda: eager_ce(b123(x)),
        "k123+k4": lambda: k4_ce(b123(x)), "k1235+k4": lambda: k4_ce(b1235(x)),
    }
    # --- correctness before timing ---
    with torch.no_grad():
        ref = eager(x)
        for n, blk in (("k123", b123), ("k1235", b1235)):
            assert_close_with_outliers(blk(x), ref, rtol=1e-2, atol=2e-2, max_outlier_fraction=DEFAULT_BF16_OUTLIER_FRACTION)
        ref_loss = eager_ce(ref).float()
        for n in ("eager+ce", "k123+ce", "k123+k4", "k1235+k4"):
            loss = variants[n]().float()
            assert abs(loss.item() - ref_loss.item()) < 5e-2 * max(1.0, abs(ref_loss.item())), (n, loss.item(), ref_loss.item())

    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    order = list(variants)
    random.Random(time.time_ns()).shuffle(order)
    rec = {"t": time.time(), "order": order, "median_ms": {}, "iqr_ms": {}, "peak_mb": {}}
    with torch.no_grad():
        for name in order:
            fn = variants[name]
            for _ in range(WARM):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(ITERS):
                flush.fill_(1.0); torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); fn(); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            q = statistics.quantiles(ts, n=4)
            rec["median_ms"][name] = statistics.median(ts); rec["iqr_ms"][name] = q[2] - q[0]
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fn(); torch.cuda.synchronize()
            rec["peak_mb"][name] = (torch.cuda.max_memory_allocated() - base) / 2**20
    with open(ROOT / "results/phase4/llama_runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print({k: round(v, 2) for k, v in rec["median_ms"].items()})


if __name__ == "__main__":
    main()
