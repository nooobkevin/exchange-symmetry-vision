#!/usr/bin/env python3
"""Experiment 1: Synthetic Bilinear Oracle (CPU-parallel across 64 cores, zero GPU interference).

Tests the fundamental theoretical question cleanly with known ground truth:
    y = u^T M v + eps,   u, v ~ N(0, I_d)

Three teacher symmetries (matched Frobenius norm ||M||_F = 1 and rank r):
  1. antisymmetric : M^T = -M (block-aligned 4x4 skew blocks OR randomly rotated Q M Q^T)
  2. symmetric     : M^T = +M (block-aligned 4x4 sym blocks OR randomly rotated Q M Q^T)
  3. general       : M = (M_sym + M_anti) / sqrt(2) (50/50 energy split)

Two coordinate systems:
  - aligned : non-zero interaction lives cleanly inside 4-channel groups (favors grouped k=4 wedge)
  - rotated : inputs u' = Q u, v' = Q v via a fixed random orthogonal matrix Q in O(d)
              (preserves exact exchange symmetry of M', but spreads interactions across groups!)

Students (all matched capacity to represent the ground-truth teacher):
  - ols_bilinear        : closed-form ridge/OLS on vec(u v^T) (sanity-check upper bound at large N)
  - grouped_wedge_fixed : Clifford k=4 grouped wedge + dot WITHOUT input projection (pure coordinate-locked)
  - grouped_wedge_proj  : Clifford k=4 grouped wedge + dot WITH learnable input projection
  - anti                : ExchangeBilinearCore(mode='anti')  -> (p - q)/sqrt(2), rotation-invariant!
  - sym                 : ExchangeBilinearCore(mode='sym')   -> (p + q)/sqrt(2)
  - star                : ExchangeBilinearCore(mode='star')  -> p = (Au)*(Bv)
  - free_anti           : ExchangeBilinearCore(mode='free_anti') -> learnable theta from -pi/4
  - free_star           : ExchangeBilinearCore(mode='free_star') -> learnable theta from 0
  - free_sym            : ExchangeBilinearCore(mode='free_sym')  -> learnable theta from +pi/4

Measures:
  - Held-out R^2 and normalized MSE across sample budgets N in {64, 128, 256, 512, 1024, 2048, 4096}
  - Learned readout-weighted effective antisymmetry EffAnti for free_anti / free_star / free_sym
    (Does gradient descent recover EffAnti -> 1.0 on antisym, 0.0 on sym, ~0.5 on general
     regardless of initialization?!)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from exsym.adapter import ExchangeBilinearCore, GroupedWedgeCore


def _make_teacher_matrix(d: int, teacher_type: str, coord: str, seed: int) -> np.ndarray:
    """Create a d x d ground-truth bilinear matrix M with ||M||_F = 1."""
    rng = np.random.default_rng(seed)
    assert d % 4 == 0
    G = d // 4
    M = np.zeros((d, d), dtype=np.float64)

    for g in range(G):
        s = 4 * g
        blk = rng.standard_normal((4, 4))
        if teacher_type == "antisymmetric":
            b = 0.5 * (blk - blk.T)
        elif teacher_type == "symmetric":
            b = 0.5 * (blk + blk.T)
        elif teacher_type == "general":
            b_sym = 0.5 * (blk + blk.T)
            b_anti = 0.5 * (blk - blk.T)
            b_sym /= np.linalg.norm(b_sym) + 1e-12
            b_anti /= np.linalg.norm(b_anti) + 1e-12
            b = (b_sym + b_anti) / math.sqrt(2.0)
        else:
            raise ValueError(teacher_type)
        M[s : s + 4, s : s + 4] = b

    if coord == "rotated":
        # Fixed random orthogonal matrix Q in O(d)
        G_rand = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(G_rand)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        M = Q @ M @ Q.T
    elif coord != "aligned":
        raise ValueError(coord)

    M = M / (np.linalg.norm(M) + 1e-12)
    return M


class BilinearStudent(nn.Module):
    def __init__(self, d: int, rank: int, student_name: str) -> None:
        super().__init__()
        self.student_name = student_name
        if student_name == "grouped_wedge_fixed":
            self.core = GroupedWedgeCore(d, rank, subspace_k=4, include_dot=True, learnable_in_proj=False)
        elif student_name == "grouped_wedge_proj":
            self.core = GroupedWedgeCore(d, rank, subspace_k=4, include_dot=True, learnable_in_proj=True)
        else:
            self.core = ExchangeBilinearCore(d, rank, mode=student_name, bias=False)
        self.readout = nn.Linear(rank, 1, bias=False)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.readout(self.core(u, v)).squeeze(-1)

    @torch.no_grad()
    def eff_anti(self, u: torch.Tensor, v: torch.Tensor) -> float | None:
        if not isinstance(self.core, ExchangeBilinearCore):
            return None
        f = self.core(u, v).float()
        f_var = f.var(dim=0, unbiased=False)
        w_sq = self.readout.weight.detach().float().squeeze(0).pow(2)
        ce = w_sq * f_var
        _, e_anti = self.core.get_energy_split()
        return float((ce * e_anti).sum().item() / max(ce.sum().item(), 1e-12))

    @torch.no_grad()
    def mean_e_anti(self) -> float | None:
        if not isinstance(self.core, ExchangeBilinearCore):
            return None
        _, e_anti = self.core.get_energy_split()
        return float(e_anti.mean().item())


def _run_one_config(task: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    d = task["d"]
    rank = task["rank"]
    teacher_type = task["teacher_type"]
    coord = task["coord"]
    n_train = task["n_train"]
    seed = task["seed"]
    noise_std = task["noise_std"]
    steps = task["steps"]

    M = _make_teacher_matrix(d, teacher_type, coord, seed=1000 + seed)
    rng = np.random.default_rng(2000 + seed * 97 + n_train)

    u_tr = rng.standard_normal((n_train, d)).astype(np.float32)
    v_tr = rng.standard_normal((n_train, d)).astype(np.float32)
    y_tr_clean = np.einsum("ni,ij,nj->n", u_tr, M, v_tr).astype(np.float32)
    y_tr = y_tr_clean + rng.normal(0.0, noise_std, size=n_train).astype(np.float32)

    n_val = 4096
    u_va = rng.standard_normal((n_val, d)).astype(np.float32)
    v_va = rng.standard_normal((n_val, d)).astype(np.float32)
    y_va = np.einsum("ni,ij,nj->n", u_va, M, v_va).astype(np.float32)
    y_va_var = float(np.var(y_va))

    results: dict[str, Any] = {}

    # 1. Exact ridge-regularized OLS on outer product vec(u v^T) (dim = d^2)
    X_tr = (u_tr[:, :, None] * v_tr[:, None, :]).reshape(n_train, d * d).astype(np.float64)
    X_va = (u_va[:, :, None] * v_va[:, None, :]).reshape(n_val, d * d).astype(np.float64)
    lam = 1e-2
    A = X_tr.T @ X_tr + lam * np.eye(d * d)
    w_ols = np.linalg.solve(A, X_tr.T @ y_tr.astype(np.float64))
    pred_va = X_va @ w_ols
    mse_ols = float(np.mean((pred_va - y_va) ** 2))
    r2_ols = float(1.0 - mse_ols / max(y_va_var, 1e-12))
    results["ols_bilinear"] = {
        "val_mse": mse_ols,
        "val_r2": r2_ols,
        "n_params": d * d,
        "eff_anti": None,
        "mean_e_anti": None,
    }

    u_tr_t = torch.from_numpy(u_tr)
    v_tr_t = torch.from_numpy(v_tr)
    y_tr_t = torch.from_numpy(y_tr)
    u_va_t = torch.from_numpy(u_va)
    v_va_t = torch.from_numpy(v_va)
    y_va_t = torch.from_numpy(y_va)

    students = [
        "grouped_wedge_fixed",
        "grouped_wedge_proj",
        "anti",
        "sym",
        "star",
        "free_anti",
        "free_star",
        "free_sym",
    ]
    for sname in students:
        torch.manual_seed(3000 + seed)
        model = BilinearStudent(d=d, rank=rank, student_name=sname)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

        best_mse = float("inf")
        best_r2 = -1e9
        best_eff_anti = None
        best_mean_e_anti = None

        for step in range(steps):
            model.train()
            opt.zero_grad(set_to_none=True)
            pred = model(u_tr_t, v_tr_t)
            loss = (pred - y_tr_t).pow(2).mean()
            loss.backward()
            opt.step()
            sched.step()

            if (step + 1) % 50 == 0 or step == steps - 1:
                model.eval()
                with torch.no_grad():
                    vp = model(u_va_t, v_va_t)
                    vmse = float((vp - y_va_t).pow(2).mean().item())
                    if vmse < best_mse:
                        best_mse = vmse
                        best_r2 = float(1.0 - vmse / max(y_va_var, 1e-12))
                        best_eff_anti = model.eff_anti(u_va_t, v_va_t)
                        best_mean_e_anti = model.mean_e_anti()

        results[sname] = {
            "val_mse": best_mse,
            "val_r2": best_r2,
            "n_params": n_params,
            "eff_anti": best_eff_anti,
            "mean_e_anti": best_mean_e_anti,
        }

    return {
        "teacher_type": teacher_type,
        "coord": coord,
        "n_train": n_train,
        "seed": seed,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: CPU-parallel Synthetic Bilinear Oracle")
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--out",
        type=str,
        default="/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/results/exp1_synthetic_oracle.json",
    )
    args = parser.parse_args()

    t0 = time.monotonic()
    teachers = ["antisymmetric", "symmetric", "general"]
    coords = ["aligned", "rotated"]
    n_trains = [64, 128, 256, 512, 1024, 2048]
    seeds = [42, 101, 202, 303, 404]

    tasks = [
        {
            "d": args.d,
            "rank": args.rank,
            "teacher_type": tt,
            "coord": co,
            "n_train": nt,
            "seed": sd,
            "noise_std": args.noise_std,
            "steps": args.steps,
        }
        for tt in teachers
        for co in coords
        for nt in n_trains
        for sd in seeds
    ]
    print(f"[exp1] Launching {len(tasks)} synthetic oracle tasks across {args.workers} CPU workers...", flush=True)

    raw_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run_one_config, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            raw_rows.append(fut.result())
            done += 1
            if done % 30 == 0 or done == len(tasks):
                print(f"  [exp1] {done}/{len(tasks)} done ({time.monotonic() - t0:.1f}s)", flush=True)

    # Aggregate mean +/- std across seeds
    students = list(raw_rows[0]["results"].keys())
    summary: dict[str, Any] = {}
    for tt in teachers:
        summary[tt] = {}
        for co in coords:
            summary[tt][co] = {}
            for nt in n_trains:
                cell_rows = [r for r in raw_rows if r["teacher_type"] == tt and r["coord"] == co and r["n_train"] == nt]
                summary[tt][co][str(nt)] = {}
                for s in students:
                    r2s = [r["results"][s]["val_r2"] for r in cell_rows]
                    mses = [r["results"][s]["val_mse"] for r in cell_rows]
                    eas = [r["results"][s]["eff_anti"] for r in cell_rows if r["results"][s]["eff_anti"] is not None]
                    summary[tt][co][str(nt)][s] = {
                        "r2_mean": round(float(np.mean(r2s)), 4),
                        "r2_std": round(float(np.std(r2s, ddof=1)), 4),
                        "mse_mean": round(float(np.mean(mses)), 5),
                        "eff_anti_mean": round(float(np.mean(eas)), 4) if eas else None,
                        "eff_anti_std": round(float(np.std(eas, ddof=1)), 4) if len(eas) > 1 else None,
                        "n_params": cell_rows[0]["results"][s]["n_params"],
                    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "wall_time_sec": round(time.monotonic() - t0, 2),
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[exp1] Saved {out_path} in {time.monotonic() - t0:.1f}s!", flush=True)

    # Print compact headline table at N=256 and N=1024
    print("\n=== EXP 1 HEADLINE (Val R^2, d=16, rank=16, 5 seeds) ===")
    for tt in teachers:
        for co in coords:
            row256 = summary[tt][co]["256"]
            row1024 = summary[tt][co]["1024"]
            print(f"\nTeacher={tt:13s} | Coord={co:7s}")
            print(f"  {'Student':20s} | {'N=256 R2':>14s} | {'N=1024 R2':>14s} | {'EffAnti(1024)':>14s}")
            for s in students:
                a = row256[s]
                b = row1024[s]
                ea = f"{b['eff_anti_mean']:.3f}±{b['eff_anti_std']:.3f}" if b["eff_anti_mean"] is not None else "    N/A"
                print(f"  {s:20s} | {a['r2_mean']:7.4f}±{a['r2_std']:.4f} | {b['r2_mean']:7.4f}±{b['r2_std']:.4f} | {ea:>14s}")


if __name__ == "__main__":
    main()
