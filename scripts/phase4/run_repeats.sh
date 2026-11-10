#!/usr/bin/env bash
# Run-to-run variance: repeat the Llama block comparison and every kernel benchmark in separate processes.
set -u
cd "$(dirname "$0")/../.."
PY=$PWD/.venv/bin/python
N_LLAMA=${N_LLAMA:-8}
N_BENCH=${N_BENCH:-3}
for i in $(seq 1 $N_LLAMA); do $PY examples/llama_block_phase4.py > /dev/null 2>&1; echo "llama $i done $(date +%T)"; done
mkdir -p results/phase4/bench_repeats
for b in benchmarks/*_bench.py; do
  name=$(basename $b _bench.py)
  for i in $(seq 1 $N_BENCH); do
    $PY $b > results/phase4/bench_repeats/${name}_run${i}.log 2>&1
    cp benchmarks/${name}_results.csv results/phase4/bench_repeats/${name}_run${i}.csv 2>/dev/null
    echo "$name $i done $(date +%T)"
  done
done
