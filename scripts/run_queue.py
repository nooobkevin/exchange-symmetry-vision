#!/usr/bin/env python3
"""Multi-GPU priority job queue for exchange-symmetry-vision ImageNet-1k waves.

Supports arbitrary GPU group allocations (e.g., 8x 1-GPU workers OR 4x 2-GPU DDP workers),
deadline checks, automatic skipping of already-finished runs, and structured JSON/MD summaries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any

PY = "/project/itscspod/chtaihei/cliffordnet/CliffordNet/.venv/bin/python"
TORCHRUN = "/project/itscspod/chtaihei/cliffordnet/CliffordNet/.venv/bin/torchrun"
TRAIN = "/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/scripts/train_in1k.py"
SRC = "/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/src"


def _run_job(
    job: dict[str, Any],
    slot_queue: Queue[tuple[list[int], int]],
    output_dir: Path,
    deadline_iso: str,
    deadline_ts: float | None,
    est_job_min: float,
) -> dict[str, Any]:
    run_dir = output_dir / job["name"]
    summary_file = run_dir / "summary.json"
    if summary_file.exists():
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        print(f"[SKIP] {job['name']} already complete (best_ema={data.get('best_ema_top1')}%)", flush=True)
        return {**data, "status": "skipped_done"}

    gpus, port = slot_queue.get()
    try:
        if deadline_ts is not None and (time.time() + est_job_min * 60.0) > deadline_ts:
            print(f"[DEADLINE] Skipping {job['name']} (est {est_job_min:.0f}m exceeds deadline {deadline_iso})", flush=True)
            return {"name": job["name"], "status": "skipped_deadline"}

        run_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        env["PYTHONPATH"] = SRC

        args = [
            "--name", job["name"],
            "--output-dir", str(output_dir),
            "--resume",
        ]
        if deadline_iso:
            args.extend(["--deadline-iso", deadline_iso])
        for k, v in job.get("args", {}).items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    args.append(flag)
            else:
                args.extend([flag, str(v)])

        if len(gpus) > 1:
            cmd = [TORCHRUN, f"--nproc_per_node={len(gpus)}", f"--master_port={port}", TRAIN] + args
        else:
            cmd = [PY, TRAIN] + args

        gpu_tag = ",".join(str(g) for g in gpus)
        print(f"[GPU {gpu_tag}] START {job['name']}", flush=True)
        t0 = time.monotonic()
        log_path = run_dir / "stdout.log"
        with open(log_path, "a", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        wall_min = (time.monotonic() - t0) / 60.0

        if proc.returncode == 0 and summary_file.exists():
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            print(f"[GPU {gpu_tag}] DONE  {job['name']} ({wall_min:.1f}m) | best_ema={data.get('best_ema_top1'):.2f}%", flush=True)
            return {**data, "status": "ok"}
        else:
            print(f"[GPU {gpu_tag}] FAIL  {job['name']} (rc={proc.returncode}, {wall_min:.1f}m)", flush=True)
            return {"name": job["name"], "status": f"failed_rc_{proc.returncode}", "wall_time_min": round(wall_min, 2)}
    finally:
        slot_queue.put((gpus, port))


def main() -> None:
    p = argparse.ArgumentParser(description="Priority GPU queue for exchange-symmetry-vision")
    p.add_argument("--spec", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--slots", type=str, default="0;1;2;3;4;5;6;7", help="Semicolon-separated GPU slots, e.g. '0,1;2,3;4,5;6,7'")
    p.add_argument("--deadline-iso", type=str, default="2026-11-23T09:45:00+08:00")
    p.add_argument("--est-job-min", type=float, default=45.0)
    p.add_argument("--summary-out", type=str, required=True)
    args = p.parse_args()

    jobs = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline_ts = dt.datetime.fromisoformat(args.deadline_iso).timestamp() if args.deadline_iso else None

    slot_groups = [[int(x) for x in s.split(",") if x.strip()] for s in args.slots.split(";") if s.strip()]
    slot_queue: Queue[tuple[list[int], int]] = Queue()
    for idx, grp in enumerate(slot_groups):
        slot_queue.put((grp, 29700 + idx))

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(slot_groups)) as pool:
        futs = {
            pool.submit(_run_job, job, slot_queue, out_dir, args.deadline_iso, deadline_ts, args.est_job_min): job["name"]
            for job in jobs
        }
        for fut in as_completed(futs):
            res = fut.result()
            results[res["name"]] = res

    ordered = [results[j["name"]] for j in jobs]
    out_p = Path(args.summary_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"\n[queue] Summary saved to {out_p}", flush=True)


if __name__ == "__main__":
    main()
