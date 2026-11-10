#!/usr/bin/env bash
# Profile every (kernel, case) with Nsight Compute (needs GPU counter permission, e.g. sudo).
set -u
cd "$(dirname "$0")/../.."
PY=$PWD/.venv/bin/python
NCU=${NCU:-/usr/local/cuda-13.1/bin/ncu}
declare -A KREG=( [matmul]=matmul [cosine]=cosine_topk [pairwise]=pairwise_distance [graph]=graph_message )
while read -r k c; do
  out=results/phase4/ncu_${k}_${c}
  sudo -n "$NCU" --set full --launch-skip 2 --launch-count 3 -k regex:${KREG[$k]} -f -o "$out" "$PY" scripts/phase4/profile_target.py "$k" "$c" > "$out.log" 2>&1
  sudo -n chown "$USER" "$out.ncu-rep"
done <<'EOF'
matmul qkv_bf16
matmul qkv_fp16
matmul qkv_fp32
cosine large_pool_fp16
cosine small_pool_fp16
cosine nsweep_500000_fp16
pairwise dim1536_fp16
pairwise dim384_fp16
pairwise dim1536_fp32
graph n2000_fp16
graph n20000_fp16
graph n100000_fp16
EOF
