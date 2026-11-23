#!/usr/bin/env python3
"""Export all verified exchange-symmetry-vision JSONs & trajectories into docs/data/exsym_data.{json,js}."""

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


def _summarize_forensics(f_json: dict | None, prefix: str, modes: list[str], seeds: list[int], sym_keys: list[str]) -> dict[str, Any]:
    if not f_json or "runs" not in f_json:
        return {"status": "pending", "modes": {}}
    runs = f_json["runs"]
    out: dict[str, Any] = {}
    none_vals = [runs[f"{prefix}none_s{s}"]["full_eval"]["clean"] for s in seeds if f"{prefix}none_s{s}" in runs]

    for m in modes:
        rs = [runs[f"{prefix}{m}_s{s}"] for s in seeds if f"{prefix}{m}_s{s}" in runs]
        if not rs:
            continue
        cl = [r["full_eval"]["clean"] for r in rs]
        entry: dict[str, Any] = {
            "n_seeds": len(rs),
            "clean_mean": round(float(np.mean(cl)), 2),
            "clean_std": round(float(np.std(cl, ddof=1)), 2) if len(cl) > 1 else 0.0,
            "clean_seeds": cl,
        }
        if none_vals and len(none_vals) >= len(cl) and m != "none":
            diffs = [a - b for a, b in zip(cl, none_vals[: len(cl)])]
            entry["delta_vs_none_mean"] = round(float(np.mean(diffs)), 2)
            entry["delta_vs_none_std"] = round(float(np.std(diffs, ddof=1)), 2) if len(diffs) > 1 else 0.0
            entry["delta_vs_none_seeds"] = [round(x, 2) for x in diffs]

        for k in ("clean_top5", "contrast_012", "gaussian_noise", "fft_phase", "hflip_consistency"):
            vals = [r["full_eval"][k] for r in rs if k in r["full_eval"]]
            if vals:
                entry[k] = round(float(np.mean(vals)), 2)

        # Symmetry per stage
        entry["symmetry"] = {}
        for sk in sym_keys:
            s_dicts = [r.get("symmetry_final", {}).get(sk, {}) for r in rs]
            s_dicts = [x for x in s_dicts if x]
            if s_dicts:
                entry["symmetry"][sk] = {
                    metric: round(float(np.mean([x[metric] for x in s_dicts if x.get(metric) is not None])), 4)
                    for metric in ("eff_anti", "func_anti", "mean_e_anti", "rms_ratio")
                    if any(x.get(metric) is not None for x in s_dicts)
                }

        # Causal decomposed interventions
        if rs[0].get("intervention"):
            inv_out: dict[str, Any] = {}
            base_c = np.mean([r["intervention"]["anti_sweep"]["s1.0_a1.0"]["clean"] for r in rs])
            inv_out["bypass_delta"] = round(float(np.mean([r["intervention"]["bypass"]["s0.0_a0.0"]["clean"] for r in rs]) - base_c), 2)
            inv_out["sym0_delta"] = round(float(np.mean([r["intervention"]["sym_sweep"]["s0.0_a1.0"]["clean"] for r in rs]) - base_c), 2)
            inv_out["anti0_delta"] = round(float(np.mean([r["intervention"]["anti_sweep"]["s1.0_a0.0"]["clean"] for r in rs]) - base_c), 2)
            inv_out["anti_curve"] = {
                g: round(float(np.mean([r["intervention"]["anti_sweep"][f"s1.0_a{g}"]["clean"] for r in rs]) - base_c), 2)
                for g in ("1.0", "0.75", "0.5", "0.0")
            }
            inv_out["sym_curve"] = {
                g: round(float(np.mean([r["intervention"]["sym_sweep"][f"s{g}_a1.0"]["clean"] for r in rs]) - base_c), 2)
                for g in ("1.0", "0.75", "0.5", "0.0")
            }
            entry["intervention"] = inv_out

        # Mean trajectory across seeds
        trajs = [r.get("trajectory", []) for r in rs if r.get("trajectory")]
        if trajs:
            min_len = min(len(t) for t in trajs)
            mean_traj = []
            for idx_t in range(min_len):
                ep = trajs[0][idx_t]["epoch"]
                t1s = [t[idx_t]["top1"] for t in trajs if t[idx_t].get("top1") is not None]
                pt: dict[str, Any] = {"epoch": ep, "top1": round(float(np.mean(t1s)), 2) if t1s else None}
                for sk in sym_keys:
                    fas = [t[idx_t].get("symmetry", {}).get(sk, {}).get("func_anti") for t in trajs]
                    fas = [x for x in fas if x is not None]
                    rrs = [t[idx_t].get("symmetry", {}).get(sk, {}).get("rms_ratio") for t in trajs]
                    rrs = [x for x in rrs if x is not None]
                    if fas or rrs:
                        pt[sk] = {
                            "func_anti": round(float(np.mean(fas)), 4) if fas else None,
                            "rms_ratio": round(float(np.mean(rrs)), 4) if rrs else None,
                        }
                mean_traj.append(pt)
            entry["trajectory"] = mean_traj

        out[m] = entry
    return {"status": "complete", "modes": out}


def main() -> None:
    DOCS_DATA.mkdir(parents=True, exist_ok=True)

    f_wa = _load_if_exists(RES / "forensics_waveA_matched_aa0.json")
    f_wb_ctx = _load_if_exists(RES / "forensics_waveB_center_context.json")
    f_wb_s3 = _load_if_exists(RES / "forensics_waveB2_stage3_directed.json")
    f_wc = _load_if_exists(RES / "forensics_waveC_rot4_matched_aa0.json")
    f_wd = _load_if_exists(RES / "forensics_waveD_scratch_100ep.json")
    f_we = _load_if_exists(RES / "forensics_waveE_vit_tiny_3seeds.json") or _load_if_exists(RES / "forensics_waveE_vit_tiny_scratch.json")
    f_wf = _load_if_exists(RES / "forensics_waveF_vit_tiny_rot4.json")

    # Inject matched Wave A `none` baseline into Wave B summaries so delta_vs_none is computed cleanly
    if f_wa and f_wb_ctx and "runs" in f_wb_ctx:
        for s in (42, 101, 202):
            if f"ft_s4_dir_none_s{s}" in f_wa["runs"]:
                f_wb_ctx["runs"][f"ft_s4_ctx_none_s{s}"] = f_wa["runs"][f"ft_s4_dir_none_s{s}"]
    if f_wa and f_wb_s3 and "runs" in f_wb_s3:
        for s in (42, 101, 202):
            if f"ft_s4_dir_none_s{s}" in f_wa["runs"]:
                f_wb_s3["runs"][f"ft_s3_dir_none_s{s}"] = f_wa["runs"][f"ft_s4_dir_none_s{s}"]

    payload: dict[str, Any] = {
        "exp1_oracle": _load_if_exists(RES / "exp1_synthetic_oracle.json"),
        "exp1b_thermometer": _load_if_exists(RES / "exp1b_symmetry_thermometer.json"),
        "exp1c_rank_noise": _load_if_exists(RES / "exp1c_rank_and_noise.json"),
        "exp1d_chirality": _load_if_exists(RES / "exp1d_chirality_and_gap.json"),
        "exp1e_saddle": _load_if_exists(RES / "exp1e_saddle_point_coadaptation.json"),
        "exp2_frozen": {
            "s4_directed": _load_if_exists(RES / "exp2_frozen_s4_directed.json"),
            "s4_context": _load_if_exists(RES / "exp2_frozen_s4_context.json"),
            "s3_directed": _load_if_exists(RES / "exp2_frozen_s3_directed.json"),
            "s3_context": _load_if_exists(RES / "exp2_frozen_s3_context.json"),
        },
        "diag_self_avg": _load_if_exists(RES / "diag_self_avg_and_init_scale.json"),
        "diag_cross_regime_swap": _load_if_exists(RES / "diag_cross_regime_selfavg_swap.json"),
        "diag_demod": _load_if_exists(RES / "diag_demodulation_coadaptation.json"),
        "throughput_h800": _load_if_exists(RES / "throughput_benchmark_h800.json"),
        "waves": {
            "waveA_ft_s4_dir": _summarize_forensics(
                f_wa, "ft_s4_dir_", ["none", "anti_rmsmatched", "sym", "self_avg", "free_star", "star", "free_anti"], [42, 101, 202, 303], ["stage4_mid"]
            ),
            "waveC_rot4_s4_dir": _summarize_forensics(
                f_wc, "rot4_s4_dir_", ["none", "anti", "sym", "star", "self_avg", "free_anti", "free_star", "free_sym"], [42, 101, 202], ["stage4_mid"]
            ),
            "waveB_ft_s4_ctx": _summarize_forensics(
                f_wb_ctx, "ft_s4_ctx_", ["none", "anti", "sym", "star", "self_avg", "free_anti", "free_star", "grouped_wedge"], [42, 101, 202], ["stage4_mid"]
            ),
            "waveB2_ft_s3_dir": _summarize_forensics(
                f_wb_s3, "ft_s3_dir_", ["none", "anti", "sym", "star", "self_avg", "free_anti", "free_star"], [42, 101, 202], ["stage3_mid"]
            ),
            "waveD_scratch_convnext": _summarize_forensics(
                f_wd, "scratch_ms_dir_", ["none", "anti", "sym", "star", "self_avg", "free_anti", "free_star", "grouped_wedge"], [42, 101, 202], ["stage3_mid", "stage4_mid"]
            ),
            "waveE_scratch_vittiny": _summarize_forensics(
                f_we, "scratch_vittiny_dir_", ["none", "anti", "sym", "star", "self_avg", "free_star"], [42, 101, 202], ["mid_vit"]
            ),
            "waveF_rot4_vittiny": _summarize_forensics(
                f_wf, "rot4_vittiny_dir_", ["anti", "sym", "self_avg", "free_star"], [42, 101], ["mid_vit"]
            ),
        },
    }

    json_str = json.dumps(payload, separators=(",", ":"))
    (DOCS_DATA / "exsym_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (DOCS_DATA / "exsym_data.js").write_text(f"window.__EXSYM_DATA__ = {json_str};\n", encoding="utf-8")
    print(f"[export] Wrote docs/data/exsym_data.json & exsym_data.js ({len(json_str)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
