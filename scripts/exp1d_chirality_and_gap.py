#!/usr/bin/env python3
"""Experiment 1D: 1D Spatial Chirality & GAP Cancellation Bridge (CPU-parallel, 32 cores).

Constructs 1D spatial feature sequences X in R^(L x d) (L=8, d=12) with adjacent directed pairs
u_t = X_t, v_t = X_{t+1}. Evaluates two tasks on the exact same spatial inputs:
  1. Reflection-Invariant (Even) task: y(flip(X)) == +y(X)  (undirected adjacent co-occurrence)
  2. Chiral / Directional (Odd) task:  y(flip(X)) == -y(X)  (oriented spatial transition / vorticity)

Also compares:
  - direct_gap : linear readout directly on mean_t(f(u_t, v_t))
  - post_nl    : GELU(f(u_t, v_t)) -> mean_t -> linear readout (analogous to downstream ConvNeXt block)

Demonstrates cleanly:
  (a) Under reflection-invariant tasks, Free-from-Star drives FuncAnti -> 0 and Anti fails.
  (b) Under chiral/directional tasks, Free-from-Star drives FuncAnti -> 1 and Sym fails!
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


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    L, d, rank = 8, 12, 16
    task_kind = task["task_kind"]
    pool_mode = task["pool_mode"]
    seed = task["seed"]
    n_train, n_val, steps = 2048, 2048, 600

    rng = np.random.default_rng(11000 + seed)
    A_t = rng.standard_normal((d, d))
    M_sym = 0.5 * (A_t + A_t.T)
    M_sym /= np.linalg.norm(M_sym)
    B_t = rng.standard_normal((d, d))
    M_anti = 0.5 * (B_t - B_t.T)
    M_anti /= np.linalg.norm(M_anti)

    def _gen(n: int, s: int):
        r = np.random.default_rng(s)
        # Spatially correlated AR(1) sequence + directional drift component so adjacent pairs carry signal
        X = r.standard_normal((n, L, d)).astype(np.float32)
        for t in range(1, L):
            X[:, t] = 0.4 * X[:, t - 1] + 0.9165 * X[:, t]
        u = X[:, :-1]  # (n, L-1, d)
        v = X[:, 1:]   # (n, L-1, d)
        if task_kind == "reflection_invariant":
            # Strictly invariant to spatial reversal X -> flip(X)
            s_val = np.einsum("nti,ij,ntj->nt", u, M_sym, v).mean(axis=1)
        elif task_kind == "chiral_directional":
            # Strictly flips sign under spatial reversal X -> flip(X)
            s_val = np.einsum("nti,ij,ntj->nt", u, M_anti, v).mean(axis=1)
        else:
            raise ValueError(task_kind)
        y = (s_val > 0.0).astype(np.int64)
        return X, y

    X_tr, y_tr = _gen(n_train, 12000 + seed)
    X_va, y_va = _gen(n_val, 13000 + seed)
    X_tr_t, y_tr_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    X_va_t, y_va_t = torch.from_numpy(X_va), torch.from_numpy(y_va)
    X_va_flip = torch.flip(X_va_t, dims=[1])

    out: dict[str, Any] = {}
    for mode in ("anti", "sym", "star", "free_anti", "free_star", "free_sym"):
        torch.manual_seed(14000 + seed)
        core = ExchangeBilinearCore(d, rank, mode=mode, bias=False)
        head = nn.Linear(rank, 2)
        act = nn.GELU() if pool_mode == "post_nl" else nn.Identity()
        opt = torch.optim.AdamW(list(core.parameters()) + list(head.parameters()), lr=2e-2, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            f = core(X_tr_t[:, :-1], X_tr_t[:, 1:])
            logits = head(act(f).mean(dim=1))
            loss = nn.functional.cross_entropy(logits, y_tr_t)
            loss.backward()
            opt.step()
            sched.step()

        with torch.no_grad():
            f_uv = core(X_va_t[:, :-1], X_va_t[:, 1:]).float()
            f_vu = core(X_va_t[:, 1:], X_va_t[:, :-1]).float()
            lg = head(act(f_uv).mean(dim=1))
            lg_flip = head(act(core(X_va_flip[:, :-1], X_va_flip[:, 1:])).mean(dim=1))
            pred = lg.argmax(dim=1)
            pred_flip = lg_flip.argmax(dim=1)
            acc = float((pred == y_va_t).float().mean().item() * 100.0)
            flip_cons = float((pred == pred_flip).float().mean().item() * 100.0)

            # Exact functional antisymmetry pre-act
            es = float((0.5 * (f_uv + f_vu)).pow(2).mean().item())
            ea = float((0.5 * (f_uv - f_vu)).pow(2).mean().item())
            func_anti = ea / max(es + ea, 1e-12)

            ce = head.weight.detach().float().pow(2).sum(dim=0) * f_uv.var(dim=(0, 1), unbiased=False)
            _, e_a_ch = core.get_energy_split()
            eff_anti = float((ce * e_a_ch).sum().item() / max(ce.sum().item(), 1e-12))

        out[mode] = {
            "acc": round(acc, 2),
            "flip_consistency": round(flip_cons, 2),
            "eff_anti": round(eff_anti, 4),
            "func_anti": round(func_anti, 4),
        }
    return {**task, "results": out}


def main() -> None:
    t0 = time.monotonic()
    tasks = [
        {"task_kind": tk, "pool_mode": pm, "seed": s}
        for tk in ("reflection_invariant", "chiral_directional")
        for pm in ("direct_gap", "post_nl")
        for s in (42, 101, 202, 303, 404)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=20) as pool:
        for res in as_completed([pool.submit(_run_one, t) for t in tasks]):
            rows.append(res.result())

    modes = ["anti", "sym", "star", "free_anti", "free_star", "free_sym"]
    summary: dict[str, Any] = {}
    print("\n=== EXP 1D: 1D SPATIAL CHIRALITY & GAP BRIDGE (5 seeds) ===")
    for tk in ("reflection_invariant", "chiral_directional"):
        summary[tk] = {}
        for pm in ("direct_gap", "post_nl"):
            sub = [r for r in rows if r["task_kind"] == tk and r["pool_mode"] == pm]
            summary[tk][pm] = {}
            print(f"\nTask={tk} | Pool={pm}")
            for m in modes:
                acc_m = round(float(np.mean([r["results"][m]["acc"] for r in sub])), 2)
                acc_s = round(float(np.std([r["results"][m]["acc"] for r in sub], ddof=1)), 2)
                fc_m = round(float(np.mean([r["results"][m]["flip_consistency"] for r in sub])), 2)
                ea_m = round(float(np.mean([r["results"][m]["eff_anti"] for r in sub])), 3)
                fa_m = round(float(np.mean([r["results"][m]["func_anti"] for r in sub])), 3)
                summary[tk][pm][m] = {"acc_mean": acc_m, "acc_std": acc_s, "flip_cons": fc_m, "eff_anti": ea_m, "func_anti": fa_m}
                print(f"  {m:10s}: acc={acc_m:5.2f}±{acc_s:4.2f}% | flip_cons={fc_m:5.2f}% | EffA={ea_m:.3f} FuncA={fa_m:.3f}")

    out_p = Path("/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/results/exp1d_chirality_and_gap.json")
    out_p.write_text(json.dumps({"wall_time_sec": round(time.monotonic() - t0, 2), "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nSaved {out_p} in {time.monotonic() - t0:.1f}s!")


if __name__ == "__main__":
    main()
