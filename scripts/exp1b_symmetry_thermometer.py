#!/usr/bin/env python3
"""Experiment 1B: Symmetry Thermometer Calibration Curve (CPU-parallel across 64 cores).

Tests ground-truth bilinear teachers across a continuous sweep of antisymmetric energy
fraction E_anti_true in {0.0, 0.25, 0.50, 0.75, 1.0} (rotated coordinates, d=24, rank=24)
to verify whether Free(theta) initialized from Anti, Star, or Sym recovers the exact
ground-truth exchange symmetry fraction via readout-weighted EffAnti.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from exsym.adapter import ExchangeBilinearCore


def _make_mixed_teacher(d: int, frac_anti: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    M_sym = 0.5 * (A + A.T)
    B = rng.standard_normal((d, d))
    M_anti = 0.5 * (B - B.T)
    M_sym /= np.linalg.norm(M_sym) + 1e-12
    M_anti /= np.linalg.norm(M_anti) + 1e-12
    M = math.sqrt(max(0.0, 1.0 - frac_anti)) * M_sym + math.sqrt(max(0.0, frac_anti)) * M_anti
    M /= np.linalg.norm(M) + 1e-12
    return M


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    d = task["d"]
    rank = task["rank"]
    frac_anti = task["frac_anti"]
    n_train = task["n_train"]
    seed = task["seed"]
    steps = task["steps"]

    M = _make_mixed_teacher(d, frac_anti, seed=5000 + seed)
    rng = np.random.default_rng(6000 + seed * 31 + int(frac_anti * 100))

    u_tr = rng.standard_normal((n_train, d)).astype(np.float32)
    v_tr = rng.standard_normal((n_train, d)).astype(np.float32)
    y_tr = (np.einsum("ni,ij,nj->n", u_tr, M, v_tr) + rng.normal(0.0, 0.05, size=n_train)).astype(np.float32)

    u_va = rng.standard_normal((4096, d)).astype(np.float32)
    v_va = rng.standard_normal((4096, d)).astype(np.float32)
    y_va = np.einsum("ni,ij,nj->n", u_va, M, v_va).astype(np.float32)
    y_var = float(np.var(y_va))

    u_tr_t, v_tr_t, y_tr_t = torch.from_numpy(u_tr), torch.from_numpy(v_tr), torch.from_numpy(y_tr)
    u_va_t, v_va_t, y_va_t = torch.from_numpy(u_va), torch.from_numpy(v_va), torch.from_numpy(y_va)

    out: dict[str, Any] = {}
    for mode in ("anti", "sym", "star", "free_anti", "free_star", "free_sym"):
        torch.manual_seed(7000 + seed)
        core = ExchangeBilinearCore(d, rank, mode=mode, bias=False)
        readout = nn.Linear(rank, 1, bias=False)
        opt = torch.optim.AdamW(list(core.parameters()) + list(readout.parameters()), lr=2.5e-2, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            pred = readout(core(u_tr_t, v_tr_t)).squeeze(-1)
            loss = (pred - y_tr_t).pow(2).mean()
            loss.backward()
            opt.step()
            sched.step()

        with torch.no_grad():
            f_va = core(u_va_t, v_va_t).float()
            pred_va = readout(f_va).squeeze(-1)
            mse = float((pred_va - y_va_t).pow(2).mean().item())
            r2 = float(1.0 - mse / max(y_var, 1e-12))
            f_var = f_va.var(dim=0, unbiased=False)
            w_sq = readout.weight.detach().float().squeeze(0).pow(2)
            ce = w_sq * f_var
            _, e_anti = core.get_energy_split()
            eff_anti = float((ce * e_anti).sum().item() / max(ce.sum().item(), 1e-12))
            mean_anti = float(e_anti.mean().item())
            # Also compute exact bilinear matrix M_hat recovered by student to check true antisym fraction of M_hat!
            A_w = core.proj_a.weight.detach().float()  # (R, d)
            B_w = core.proj_b.weight.detach().float()  # (R, d)
            c_w = readout.weight.detach().float().squeeze(0)  # (R,)
            th = core.theta.detach().float()
            M_p = (A_w.T * c_w * torch.cos(th)) @ B_w
            M_q = (B_w.T * c_w * torch.sin(th)) @ A_w
            M_hat = M_p + M_q
            M_hat_anti = 0.5 * (M_hat - M_hat.T)
            true_func_anti = float(M_hat_anti.pow(2).sum().item() / max(M_hat.pow(2).sum().item(), 1e-12))

        out[mode] = {
            "val_r2": round(r2, 4),
            "eff_anti": round(eff_anti, 4),
            "mean_e_anti": round(mean_anti, 4),
            "func_anti_frac": round(true_func_anti, 4),
        }
    return {"frac_anti": frac_anti, "seed": seed, "results": out}


def main() -> None:
    t0 = time.monotonic()
    fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
    seeds = [42, 101, 202, 303, 404]
    tasks = [{"d": 24, "rank": 24, "frac_anti": fa, "n_train": 2048, "seed": s, "steps": 700} for fa in fracs for s in seeds]

    rows = []
    with ProcessPoolExecutor(max_workers=25) as pool:
        for res in as_completed([pool.submit(_run_one, t) for t in tasks]):
            rows.append(res.result())

    modes = ["anti", "sym", "star", "free_anti", "free_star", "free_sym"]
    summary: dict[str, Any] = {}
    print("\n=== EXP 1B: SYMMETRY THERMOMETER CALIBRATION (d=24, rank=24, N=2048, 5 seeds) ===")
    print(f"{'True E_anti':>11s} | {'free_anti EffA (FuncA)':>24s} | {'free_star EffA (FuncA)':>24s} | {'free_sym EffA (FuncA)':>24s} | {'R2(star/free_s)':>16s}")
    for fa in fracs:
        sub = [r for r in rows if r["frac_anti"] == fa]
        summary[str(fa)] = {}
        for m in modes:
            summary[str(fa)][m] = {
                "r2_mean": round(float(np.mean([r["results"][m]["val_r2"] for r in sub])), 4),
                "eff_anti_mean": round(float(np.mean([r["results"][m]["eff_anti"] for r in sub])), 4),
                "eff_anti_std": round(float(np.std([r["results"][m]["eff_anti"] for r in sub], ddof=1)), 4),
                "func_anti_mean": round(float(np.mean([r["results"][m]["func_anti_frac"] for r in sub])), 4),
            }
        fa_m = summary[str(fa)]["free_anti"]
        fs_m = summary[str(fa)]["free_star"]
        fy_m = summary[str(fa)]["free_sym"]
        st_m = summary[str(fa)]["star"]
        print(
            f"{fa:11.2f} | "
            f"{fa_m['eff_anti_mean']:.3f}±{fa_m['eff_anti_std']:.3f} ({fa_m['func_anti_mean']:.3f}) | "
            f"{fs_m['eff_anti_mean']:.3f}±{fs_m['eff_anti_std']:.3f} ({fs_m['func_anti_mean']:.3f}) | "
            f"{fy_m['eff_anti_mean']:.3f}±{fy_m['eff_anti_std']:.3f} ({fy_m['func_anti_mean']:.3f}) | "
            f"{st_m['r2_mean']:.4f} / {fs_m['r2_mean']:.4f}"
        )

    out_p = Path("/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/results/exp1b_symmetry_thermometer.json")
    out_p.write_text(json.dumps({"wall_time_sec": round(time.monotonic() - t0, 2), "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nSaved {out_p} in {time.monotonic() - t0:.1f}s!")


if __name__ == "__main__":
    main()
