#!/usr/bin/env python3
"""Experiment 1C: Rank-Bottleneck & Noise Phase Diagram (CPU-parallel across 64 cores).

Tests rotated bilinear teachers (d=16, exact rank=16) as student rank R varies across
{4, 8, 12, 16, 24} and label noise sigma varies across {0.05, 0.25, 0.50} at N=512.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from exsym.adapter import ExchangeBilinearCore
from exp1_synthetic_oracle import _make_teacher_matrix


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    d = 16
    rank = task["rank"]
    teacher_type = task["teacher_type"]
    noise_std = task["noise_std"]
    seed = task["seed"]
    n_train = 512
    steps = 600

    M = _make_teacher_matrix(d, teacher_type, "rotated", seed=9000 + seed)
    rng = np.random.default_rng(9500 + seed * 53 + rank * 7 + int(noise_std * 100))

    u_tr = rng.standard_normal((n_train, d)).astype(np.float32)
    v_tr = rng.standard_normal((n_train, d)).astype(np.float32)
    y_tr = (np.einsum("ni,ij,nj->n", u_tr, M, v_tr) + rng.normal(0.0, noise_std, size=n_train)).astype(np.float32)

    u_va = rng.standard_normal((4096, d)).astype(np.float32)
    v_va = rng.standard_normal((4096, d)).astype(np.float32)
    y_va = np.einsum("ni,ij,nj->n", u_va, M, v_va).astype(np.float32)
    y_var = float(np.var(y_va))

    u_tr_t, v_tr_t, y_tr_t = torch.from_numpy(u_tr), torch.from_numpy(v_tr), torch.from_numpy(y_tr)
    u_va_t, v_va_t, y_va_t = torch.from_numpy(u_va), torch.from_numpy(v_va), torch.from_numpy(y_va)

    res = {}
    for mode in ("anti", "sym", "star", "free_anti", "free_star"):
        torch.manual_seed(9800 + seed)
        core = ExchangeBilinearCore(d, rank, mode=mode, bias=False)
        readout = nn.Linear(rank, 1, bias=False)
        opt = torch.optim.AdamW(list(core.parameters()) + list(readout.parameters()), lr=2.5e-2, weight_decay=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

        best_r2 = -1e9
        best_eff_a = None
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = (readout(core(u_tr_t, v_tr_t)).squeeze(-1) - y_tr_t).pow(2).mean()
            loss.backward()
            opt.step()
            sched.step()
            if (step + 1) % 60 == 0 or step == steps - 1:
                with torch.no_grad():
                    fv = core(u_va_t, v_va_t).float()
                    vp = readout(fv).squeeze(-1)
                    mse = float((vp - y_va_t).pow(2).mean().item())
                    r2 = float(1.0 - mse / max(y_var, 1e-12))
                    if r2 > best_r2:
                        best_r2 = r2
                        ce = readout.weight.detach().float().squeeze(0).pow(2) * fv.var(dim=0, unbiased=False)
                        _, ea = core.get_energy_split()
                        best_eff_a = float((ce * ea).sum().item() / max(ce.sum().item(), 1e-12))
        res[mode] = {"val_r2": round(best_r2, 4), "eff_anti": round(best_eff_a, 4) if best_eff_a is not None else None}
    return {**task, "results": res}


def main() -> None:
    t0 = time.monotonic()
    tasks = []
    # Rank sweep at noise=0.05
    for tt in ("antisymmetric", "symmetric", "general"):
        for r in (4, 8, 12, 16, 24):
            for s in (42, 101, 202):
                tasks.append({"sweep": "rank", "teacher_type": tt, "rank": r, "noise_std": 0.05, "seed": s})
    # Noise sweep at rank=16
    for tt in ("antisymmetric", "symmetric", "general"):
        for ns in (0.25, 0.50):
            for s in (42, 101, 202):
                tasks.append({"sweep": "noise", "teacher_type": tt, "rank": 16, "noise_std": ns, "seed": s})

    rows = []
    with ProcessPoolExecutor(max_workers=64) as pool:
        for res in as_completed([pool.submit(_run_one, t) for t in tasks]):
            rows.append(res.result())

    # Aggregate
    agg: dict[str, Any] = {"rank_sweep": {}, "noise_sweep": {}}
    modes = ["anti", "sym", "star", "free_anti", "free_star"]
    print("\n=== EXP 1C: RANK-BOTTLENECK SWEEP (Rotated, d=16, N=512, noise=0.05) ===")
    for tt in ("antisymmetric", "symmetric", "general"):
        agg["rank_sweep"][tt] = {}
        print(f"\nTeacher={tt}")
        print(f"  {'Rank':>4s} | {'anti':>8s} | {'sym':>8s} | {'star':>8s} | {'free_anti':>9s} | {'free_star':>9s}")
        for r in (4, 8, 12, 16, 24):
            sub = [x for x in rows if x["sweep"] == "rank" and x["teacher_type"] == tt and x["rank"] == r]
            agg["rank_sweep"][tt][str(r)] = {
                m: round(float(np.mean([x["results"][m]["val_r2"] for x in sub])), 4) for m in modes
            }
            d_r = agg["rank_sweep"][tt][str(r)]
            print(f"  {r:4d} | {d_r['anti']:8.4f} | {d_r['sym']:8.4f} | {d_r['star']:8.4f} | {d_r['free_anti']:9.4f} | {d_r['free_star']:9.4f}")

    print("\n=== EXP 1C: NOISE SWEEP (Rotated, d=16, rank=16, N=512) ===")
    for tt in ("antisymmetric", "symmetric", "general"):
        agg["noise_sweep"][tt] = {}
        for ns in (0.05, 0.25, 0.50):
            sub = [x for x in rows if x["teacher_type"] == tt and x["rank"] == 16 and abs(x["noise_std"] - ns) < 1e-6]
            agg["noise_sweep"][tt][str(ns)] = {
                m: round(float(np.mean([x["results"][m]["val_r2"] for x in sub])), 4) for m in modes
            }
            d_r = agg["noise_sweep"][tt][str(ns)]
            print(f"  {tt:13s} noise={ns:.2f} | anti={d_r['anti']:.4f} sym={d_r['sym']:.4f} star={d_r['star']:.4f} free_s={d_r['free_star']:.4f}")

    out_p = Path("/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/results/exp1c_rank_and_noise.json")
    out_p.write_text(json.dumps({"wall_time_sec": round(time.monotonic() - t0, 2), "summary": agg}, indent=2), encoding="utf-8")
    print(f"\nSaved {out_p} in {time.monotonic() - t0:.1f}s!")


if __name__ == "__main__":
    main()
