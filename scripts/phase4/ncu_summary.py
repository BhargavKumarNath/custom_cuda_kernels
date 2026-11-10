"""Pull the metrics that explain each below-target kernel out of the Nsight Compute reports."""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

NCU = "/usr/local/cuda-13.1/bin/ncu"
M = {
    "duration_us": "gpu__time_duration.sum",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "registers_per_thread": "launch__registers_per_thread",
    "smem_per_block_B": "launch__shared_mem_per_block_static",
    "grid_size": "launch__grid_size",
    "block_size": "launch__block_size",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "fma_pipe_pct": "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "tensor_pipe_pct": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "issue_active_pct": "sm__issue_active.avg.pct_of_peak_sustained_active",
    "l1_hit_pct": "l1tex__t_sector_hit_rate.pct",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "smem_bank_conflicts_ld": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem_ld_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "stall_long_scoreboard": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "stall_short_scoreboard": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "stall_barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "stall_mio_throttle": "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "stall_math_pipe_throttle": "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "stall_lg_throttle": "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "stall_wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "stall_not_selected": "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "stall_no_instruction": "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "stall_selected": "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
    "dram_bytes": "dram__bytes.sum",
    "ipc": "sm__inst_executed.avg.per_cycle_elapsed",
}


def main() -> None:
    out = {}
    for rep in sorted(Path("results/phase4").glob("ncu_*.ncu-rep")):
        txt = subprocess.run([NCU, "--import", str(rep), "--page", "raw", "--csv", "--metrics", ",".join(M.values())],
                             capture_output=True, text=True).stdout
        rows = list(csv.reader(io.StringIO(txt)))
        hdr_i = next(i for i, r in enumerate(rows) if "Kernel Name" in r)
        hdr = rows[hdr_i]
        data = [r for r in rows[hdr_i + 2:] if len(r) == len(hdr)]
        entries = []
        for r in data:
            d = dict(zip(hdr, r))
            e = {"kernel": d["Kernel Name"][:60]}
            for k, v in M.items():
                x = d.get(v)
                if x is not None:
                    try:
                        e[k] = float(x.replace(",", ""))
                    except ValueError:
                        e[k] = x
            e["_units_duration"] = rows[hdr_i + 1][hdr.index(M["duration_us"])] if M["duration_us"] in hdr else ""
            entries.append(e)
        out[rep.stem.replace("ncu_", "")] = entries
    Path("results/phase4/ncu_summary.json").write_text(json.dumps(out, indent=1))
    for case, es in out.items():
        for e in es[:2]:
            print(case, "|", e["kernel"][:38], "| dur", e.get("duration_us"), e.get("_units_duration"), "| SM%", e.get("sm_throughput_pct"),
                  "DRAM%", e.get("dram_throughput_pct"), "L1%", e.get("l1_throughput_pct"), "occ", e.get("achieved_occupancy_pct"),
                  "/", e.get("theoretical_occupancy_pct"), "regs", e.get("registers_per_thread"), "tensor%", e.get("tensor_pipe_pct"))


if __name__ == "__main__":
    main()
