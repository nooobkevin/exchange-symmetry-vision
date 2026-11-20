#!/usr/bin/env python3
"""Throughput benchmark: 8x single-GPU parallel vs 4x 2-GPU DDP on /dev/shm/in1k_raw_cache_ar320.

Directly measures steady-state aggregate node throughput (images/sec across all 8 H800s),
data-wait fraction, and effective runs completed per hour so we schedule the 48-hour
ImageNet-1k study with zero I/O or DDP bottlenecks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = "/project/itscspod/chtaihei/cliffordnet/CliffordNet/.venv/bin/python"
TORCHRUN = "/project/itscspod/chtaihei/cliffordnet/CliffordNet/.venv/bin/torchrun"
TRAIN = "/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/scripts/train_in1k.py"
SRC = "/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/src"
OUT_ROOT = "/scratch/itscspod/chtaihei/exsym_runs/bench_throughput"


def _parse_bench(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("BENCH_RESULT:"):
            return json.loads(line[len("BENCH_RESULT:"):])
    return {"error": stdout[-500:]}


def run_8x1gpu(batch_size: int = 256, num_workers: int = 6, max_steps: int = 80) -> dict:
    print(f"\n[bench] === Testing 8x 1-GPU parallel (bs={batch_size}/gpu, workers={num_workers}) ===", flush=True)
    t0 = time.monotonic()

    def _one(g: int) -> dict:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        env["PYTHONPATH"] = SRC
        cmd = [
            PY, TRAIN,
            "--name", f"bench_1gpu_bs{batch_size}_g{g}",
            "--output-dir", OUT_ROOT,
            "--mode", "free_anti",
            "--batch-size", str(batch_size),
            "--num-workers", str(num_workers),
            "--max-steps", str(max_steps),
        ]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return _parse_bench(p.stdout + "\n" + p.stderr)

    with ThreadPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(_one, range(8)))
    wall = time.monotonic() - t0
    ok = [r for r in res if "img_per_sec" in r]
    agg_ips = sum(r["img_per_sec"] for r in ok)
    mean_wait = sum(r["data_wait_frac"] for r in ok) / max(len(ok), 1)
    # 1 epoch = 1,281,167 images per single-GPU run -> wall time per epoch for 8 concurrent runs
    ep_sec_per_run = 1281167.0 / max(agg_ips / 8.0, 1.0)
    summary = {
        "layout": f"8x1gpu_bs{batch_size}",
        "wall_sec_incl_compile": round(wall, 1),
        "aggregate_img_per_sec": round(agg_ips, 1),
        "per_gpu_img_per_sec": round(agg_ips / 8.0, 1),
        "mean_data_wait_frac": round(mean_wait, 4),
        "est_epoch_min_per_run": round(ep_sec_per_run / 60.0, 2),
        "est_30ep_runs_per_hour": round(8.0 / (30.0 * ep_sec_per_run / 3600.0), 2),
        "per_worker": res,
    }
    print(f"[bench] 8x1gpu_bs{batch_size}: agg={agg_ips:.0f} img/s | wait={100*mean_wait:.1f}% | est_ep={summary['est_epoch_min_per_run']}m | 30ep_runs/hr={summary['est_30ep_runs_per_hour']}", flush=True)
    return summary


def run_4x2gpu(batch_size: int = 256, num_workers: int = 6, max_steps: int = 80) -> dict:
    print(f"\n[bench] === Testing 4x 2-GPU DDP parallel (bs={batch_size}/gpu, workers={num_workers}) ===", flush=True)
    t0 = time.monotonic()
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

    def _one(idx: int) -> dict:
        g0, g1 = pairs[idx]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = f"{g0},{g1}"
        env["PYTHONPATH"] = SRC
        cmd = [
            TORCHRUN,
            "--nproc_per_node=2",
            f"--master_port={29600 + idx}",
            TRAIN,
            "--name", f"bench_2gpu_bs{batch_size}_p{idx}",
            "--output-dir", OUT_ROOT,
            "--mode", "free_anti",
            "--batch-size", str(batch_size),
            "--num-workers", str(num_workers),
            "--max-steps", str(max_steps),
        ]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return _parse_bench(p.stdout + "\n" + p.stderr)

    with ThreadPoolExecutor(max_workers=4) as pool:
        res = list(pool.map(_one, range(4)))
    wall = time.monotonic() - t0
    ok = [r for r in res if "img_per_sec" in r]
    agg_ips = sum(r["img_per_sec"] for r in ok)
    mean_wait = sum(r["data_wait_frac"] for r in ok) / max(len(ok), 1)
    ep_sec_per_run = 1281167.0 / max(agg_ips / 4.0, 1.0)
    summary = {
        "layout": f"4x2gpu_ddp_bs{batch_size}",
        "wall_sec_incl_compile": round(wall, 1),
        "aggregate_img_per_sec": round(agg_ips, 1),
        "per_gpu_img_per_sec": round(agg_ips / 8.0, 1),
        "mean_data_wait_frac": round(mean_wait, 4),
        "est_epoch_min_per_run": round(ep_sec_per_run / 60.0, 2),
        "est_30ep_runs_per_hour": round(4.0 / (30.0 * ep_sec_per_run / 3600.0), 2),
        "per_worker": res,
    }
    print(f"[bench] 4x2gpu_bs{batch_size}: agg={agg_ips:.0f} img/s | wait={100*mean_wait:.1f}% | est_ep={summary['est_epoch_min_per_run']}m | 30ep_runs/hr={summary['est_30ep_runs_per_hour']}", flush=True)
    return summary


def main() -> None:
    results = [
        run_8x1gpu(batch_size=256, num_workers=6, max_steps=100),
        run_8x1gpu(batch_size=512, num_workers=8, max_steps=60),
        run_4x2gpu(batch_size=256, num_workers=6, max_steps=100),
    ]
    out_p = Path("/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/results/throughput_benchmark_h800.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[bench] Saved {out_p}", flush=True)


if __name__ == "__main__":
    main()
