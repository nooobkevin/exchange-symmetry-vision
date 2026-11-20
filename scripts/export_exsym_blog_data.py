#!/usr/bin/env python3
"""Aggregate all verified exchange-symmetry-vision experiment JSONs into docs/data/exsym_data.{json,js}."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision")
RES = REPO / "results"
RUNS_ROOT = Path("/scratch/itscspod/chtaihei/exsym_runs")
DOCS_DATA = REPO / "docs/data"


def _load_if_exists(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _collect_wave_trajectories(wave_dir: Path) -> dict[str, Any]:
    """Read metrics.jsonl + summary.json from all runs in a wave directory."""
    if not wave_dir.exists():
        return {"status": "pending", "runs": {}}
    runs: dict[str, Any] = {}
    for d in sorted(wave_dir.iterdir()):
        if not d.is_dir():
            continue
        mpath = d / "metrics.jsonl"
        spath = d / "summary.json"
        if not mpath.exists():
            continue
        traj = []
        for line in mpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ev = r.get("val_ema") or r.get("val_raw")
            if ev is None:
                continue
            item: dict[str, Any] = {
                "epoch": r["epoch"],
                "top1": round(float(ev["top1"]), 2),
                "top5": round(float(ev["top5"]), 2),
                "loss": round(float(ev["loss"]), 4),
            }
            if ev.get("symmetry"):
                item["sym"] = {
                    k: {
                        "eff_anti": round(float(v["eff_anti"]), 4) if v.get("eff_anti") is not None else None,
                        "func_anti": round(float(v["func_anti"]), 4) if v.get("func_anti") is not None else None,
                        "mean_e_anti": round(float(v["mean_e_anti"]), 4) if v.get("mean_e_anti") is not None else None,
                        "rms_ratio": round(float(v["rms_ratio"]), 4) if v.get("rms_ratio") is not None else None,
                    }
                    for k, v in ev["symmetry"].items()
                }
            traj.append(item)
        runs[d.name] = {
            "summary": _load_if_exists(spath),
            "manifest": _load_if_exists(d / "manifest.json"),
            "trajectory": traj,
        }
    return {"status": "complete" if runs else "pending", "runs": runs}


def main() -> None:
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "exp1_oracle": _load_if_exists(RES / "exp1_synthetic_oracle.json"),
        "exp1b_thermometer": _load_if_exists(RES / "exp1b_symmetry_thermometer.json"),
        "exp1c_rank_noise": _load_if_exists(RES / "exp1c_rank_and_noise.json"),
        "exp1d_chirality": _load_if_exists(RES / "exp1d_chirality_and_gap.json"),
        "exp2_frozen": {
            "s4_directed": _load_if_exists(RES / "exp2_frozen_s4_directed.json"),
            "s4_context": _load_if_exists(RES / "exp2_frozen_s4_context.json"),
            "s3_directed": _load_if_exists(RES / "exp2_frozen_s3_directed.json"),
            "s3_context": _load_if_exists(RES / "exp2_frozen_s3_context.json"),
        },
        "throughput_h800": _load_if_exists(RES / "throughput_benchmark_h800.json"),
        "waves": {
            "waveA_ft_s4_directed": _collect_wave_trajectories(RUNS_ROOT / "waveA_ft_s4_directed"),
            "waveC_rot4_s4_directed": _collect_wave_trajectories(RUNS_ROOT / "waveC_rot4_s4_directed"),
            "waveB_ft_pair_and_stage": _collect_wave_trajectories(RUNS_ROOT / "waveB_ft_pair_and_stage"),
            "waveD_scratch_100ep": _collect_wave_trajectories(RUNS_ROOT / "waveD_scratch_100ep"),
            "waveE_vit_tiny_scratch": _collect_wave_trajectories(RUNS_ROOT / "waveE_vit_tiny_scratch"),
        },
        "forensics": {
            "waveA": _load_if_exists(RES / "forensics_waveA.json"),
            "waveC": _load_if_exists(RES / "forensics_waveC.json"),
            "waveB": _load_if_exists(RES / "forensics_waveB.json"),
            "waveD": _load_if_exists(RES / "forensics_waveD.json"),
            "waveE": _load_if_exists(RES / "forensics_waveE.json"),
        },
    }

    json_str = json.dumps(payload, separators=(",", ":"))
    (DOCS_DATA / "exsym_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (DOCS_DATA / "exsym_data.js").write_text(f"window.__EXSYM_DATA__ = {json_str};\n", encoding="utf-8")
    print(f"[export] Wrote docs/data/exsym_data.json ({len(json_str)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
