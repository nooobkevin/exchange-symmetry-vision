#!/usr/bin/env python3
"""Astra Diagnostics:
1. Zero-retraining Sym -> SelfAvg checkpoint replacement test on all 4 Wave A sym checkpoints
   (plus free_star and free_sym checkpoints):
   - Post-W_out RMS of cross-token difference term delta_cross = SelfAvg - Sym = W_out(A(u-v)*B(u-v)/sqrt2)*gamma
   - Exact 50k val NLL loss, Top-1, Top-5, logit L2, symmetric KL, and top-1 prediction agreement
2. Step-0 Initialization Scale Diagnostic (Sym vs Anti vs Star vs SelfAvg) with identical A, B, W_out, gamma:
   - Measures RMS after product, after W_out*gamma, and after spatial GAP, plus adjacent-token cosine similarity.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exsym.adapter import SQRT2_INV
from exsym.data import IMAGENET_MEAN, IMAGENET_STD, ShmImageNetDataset, normalize_batch_gpu
from exsym.models import ConvNeXtWithAdapter


@torch.no_grad()
def run_init_scale_diagnostic(ckpt_path: str, val_loader: DataLoader, device: torch.device, mean_t: torch.Tensor, std_t: torch.Tensor) -> dict[str, Any]:
    torch.manual_seed(1042)
    base_model = ConvNeXtWithAdapter(
        timm_name="convnext_atto", num_classes=1000, adapter_mode="sym",
        pair_type="directed_right", placement="stage4_mid", pretrained_ckpt=ckpt_path,
    ).to(device).eval()

    ad = base_model.adapters["stage4_mid"]
    A = ad.core.proj_a
    B = ad.core.proj_b
    W = ad.out_proj
    g = ad.gamma.view(1, -1, 1, 1)

    stats: dict[str, list[float]] = {k: [] for k in (
        "cos_uv",
        "prod_rms_sym", "prod_rms_anti", "prod_rms_star", "prod_rms_selfavg", "prod_rms_cross",
        "wout_rms_sym", "wout_rms_anti", "wout_rms_star", "wout_rms_selfavg", "wout_rms_cross",
        "gap_rms_sym", "gap_rms_anti", "gap_rms_star", "gap_rms_selfavg", "gap_rms_cross",
    )}

    for bi, (imgs_u8, _) in enumerate(val_loader):
        if bi >= 16:
            break
        x = normalize_batch_gpu(imgs_u8.to(device), mean_t, std_t)
        # Extract input to stage4_mid adapter
        h = base_model.backbone.stem(x)
        h = base_model.backbone.stages[0](h)
        h = base_model.backbone.stages[1](h)
        h = base_model.backbone.stages[2](h)
        s3 = base_model.backbone.stages[3]
        h = s3.downsample(h)
        h = s3.blocks[0](h)

        u_nchw, v_nchw = ad._make_pair(h)
        u = u_nchw.permute(0, 2, 3, 1).float()
        v = v_nchw.permute(0, 2, 3, 1).float()
        # Cosine similarity on valid interior columns (exclude zero-padded last column)
        cos_uv = F.cosine_similarity(u[:, :, :-1, :], v[:, :, :-1, :], dim=-1).mean().item()
        stats["cos_uv"].append(cos_uv)

        au, bv = A(u), B(v)
        av, bu = A(v), B(u)
        p = au * bv
        q = av * bu
        f_sym = (p + q) * SQRT2_INV
        f_anti = (p - q) * SQRT2_INV
        f_star = p
        f_sa = (au * bu + av * bv) * SQRT2_INV
        f_cross = f_sa - f_sym

        for name, ft in [("sym", f_sym), ("anti", f_anti), ("star", f_star), ("selfavg", f_sa), ("cross", f_cross)]:
            stats[f"prod_rms_{name}"].append(float(ft.pow(2).mean().sqrt().item()))
            dw = W(ft).permute(0, 3, 1, 2) * g
            stats[f"wout_rms_{name}"].append(float(dw.pow(2).mean().sqrt().item()))
            stats[f"gap_rms_{name}"].append(float(dw.mean(dim=(2, 3)).pow(2).mean().sqrt().item()))

    res = {k: round(float(np.mean(v)), 5) for k, v in stats.items()}
    res["ratio_anti_to_sym_prod"] = round(res["prod_rms_anti"] / res["prod_rms_sym"], 4)
    res["ratio_anti_to_sym_wout"] = round(res["wout_rms_anti"] / res["wout_rms_sym"], 4)
    res["ratio_anti_to_sym_gap"] = round(res["gap_rms_anti"] / res["gap_rms_sym"], 4)
    res["ratio_cross_to_sym_wout"] = round(res["wout_rms_cross"] / res["wout_rms_sym"], 4)
    return res


@torch.no_grad()
def eval_checkpoint_sym_vs_selfavg(run_dir: Path, val_loader: DataLoader, device: torch.device, mean_t: torch.Tensor, std_t: torch.Tensor) -> dict[str, Any]:
    ck = torch.load(str(run_dir / "best_ema.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    model = ConvNeXtWithAdapter(
        timm_name=args["backbone"], num_classes=1000, drop_path_rate=0.0,
        adapter_mode=args["mode"], pair_type=args["pair_type"], placement=args["placement"],
        rank_ratio=args.get("rank_ratio", 1.0), gamma_init=args.get("gamma_init", 0.1),
    )
    model.load_state_dict(ck["ema_model"], strict=True)
    model = model.to(device).to(memory_format=torch.channels_last).eval()

    ad = model.adapters["stage4_mid"]
    orig_mode = ad.core.mode

    # 1. Measure post-W_out delta_cross / delta_sym on first 16 batches
    rms_sym_l, rms_sa_l, rms_cross_l, cos_l = [], [], [], []
    for bi, (imgs_u8, _) in enumerate(val_loader):
        if bi >= 16:
            break
        x = normalize_batch_gpu(imgs_u8.to(device), mean_t, std_t).to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model.backbone.stem(x)
            h = model.backbone.stages[0](h)
            h = model.backbone.stages[1](h)
            h = model.backbone.stages[2](h)
            s3 = model.backbone.stages[3]
            h = s3.blocks[0](s3.downsample(h))
            u, v = ad._make_pair(h)
            ad.core.mode = orig_mode
            d_orig = ad.forward_delta_from_pair(u, v).float()
            ad.core.mode = "self_avg"
            d_sa = ad.forward_delta_from_pair(u, v).float()
            ad.core.mode = orig_mode
        d_cross = d_sa - d_orig
        rms_sym_l.append(float(d_orig.pow(2).mean().sqrt().item()))
        rms_sa_l.append(float(d_sa.pow(2).mean().sqrt().item()))
        rms_cross_l.append(float(d_cross.pow(2).mean().sqrt().item()))
        cos_l.append(float(F.cosine_similarity(d_orig.flatten(1), d_sa.flatten(1), dim=1).mean().item()))

    # 2. Exact full 50k validation comparison: orig vs self_avg vs bypass
    def _pass(mode_override: str | None, bypass: bool = False):
        if bypass:
            ad.set_intervention(0.0, 0.0)
        else:
            ad.set_intervention(1.0, 1.0)
            ad.core.mode = mode_override or orig_mode
        logits_all, c1, c5, nll, tot = [], 0, 0, 0.0, 0
        for imgs_u8, targets in val_loader:
            imgs_u8 = imgs_u8.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            x = normalize_batch_gpu(imgs_u8, mean_t, std_t).to(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x).float()
            nll += float(F.cross_entropy(lg, targets, reduction="sum").item())
            _, topk = lg.topk(5, dim=1)
            c1 += int(topk[:, 0].eq(targets).sum().item())
            c5 += int(topk.eq(targets.view(-1, 1)).any(dim=1).sum().item())
            tot += int(targets.size(0))
            logits_all.append(lg.cpu())
        ad.core.mode = orig_mode
        ad.set_intervention(1.0, 1.0)
        return torch.cat(logits_all, dim=0), {
            "top1": round(100.0 * c1 / tot, 3),
            "top5": round(100.0 * c5 / tot, 3),
            "nll": round(nll / tot, 5),
        }

    lg_orig, m_orig = _pass(orig_mode)
    lg_sa, m_sa = _pass("self_avg")
    lg_byp, m_byp = _pass(orig_mode, bypass=True)

    p_orig = F.softmax(lg_orig, dim=-1)
    log_p_orig = F.log_softmax(lg_orig, dim=-1)
    log_p_sa = F.log_softmax(lg_sa, dim=-1)
    log_p_byp = F.log_softmax(lg_byp, dim=-1)

    kl_orig_sa = float(F.kl_div(log_p_sa, p_orig, reduction="batchmean").item())
    kl_orig_byp = float(F.kl_div(log_p_byp, p_orig, reduction="batchmean").item())
    l2_orig_sa = float((lg_orig - lg_sa).norm(dim=-1).mean().item())
    l2_orig_byp = float((lg_orig - lg_byp).norm(dim=-1).mean().item())
    agree_sa = float((lg_orig.argmax(1) == lg_sa.argmax(1)).float().mean().item() * 100.0)
    agree_byp = float((lg_orig.argmax(1) == lg_byp.argmax(1)).float().mean().item() * 100.0)

    return {
        "run": run_dir.name,
        "orig_mode": orig_mode,
        "post_wout": {
            "rms_orig": round(float(np.mean(rms_sym_l)), 4),
            "rms_selfavg": round(float(np.mean(rms_sa_l)), 4),
            "rms_cross_diff": round(float(np.mean(rms_cross_l)), 4),
            "cross_over_orig_ratio": round(float(np.mean(rms_cross_l) / max(np.mean(rms_sym_l), 1e-8)), 4),
            "cos_orig_vs_selfavg": round(float(np.mean(cos_l)), 4),
        },
        "orig": m_orig,
        "self_avg_swap": m_sa,
        "bypass": m_byp,
        "deltas_selfavg_vs_orig": {
            "top1_diff": round(m_sa["top1"] - m_orig["top1"], 3),
            "nll_diff": round(m_sa["nll"] - m_orig["nll"], 5),
            "kl_div": round(kl_orig_sa, 5),
            "logit_l2": round(l2_orig_sa, 4),
            "pred_agree_pct": round(agree_sa, 2),
        },
        "deltas_bypass_vs_orig": {
            "top1_diff": round(m_byp["top1"] - m_orig["top1"], 3),
            "nll_diff": round(m_byp["nll"] - m_orig["nll"], 5),
            "kl_div": round(kl_orig_byp, 5),
            "logit_l2": round(l2_orig_byp, 4),
            "pred_agree_pct": round(agree_byp, 2),
        },
    }


def main() -> None:
    torch.set_num_threads(4)
    t0 = time.monotonic()
    device = torch.device("cuda:0")
    val_ds = ShmImageNetDataset(cache_dir="/dev/shm/in1k_raw_cache_ar320", split="val", image_size=224, hflip=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    ckpt_base = "/scratch/itscspod/chtaihei/cliffordnet_runs/in1k_waveC_100ep/in1k_convnext_atto_100ep/checkpoints/last.ckpt"
    print("=== 1. STEP-0 INITIALIZATION SCALE DIAGNOSTIC ===", flush=True)
    init_diag = run_init_scale_diagnostic(ckpt_base, val_loader, device, mean_t, std_t)
    print(json.dumps(init_diag, indent=2), flush=True)

    print("\n=== 2. ZERO-RETRAINING Sym -> SelfAvg CHECKPOINT REPLACEMENT (50k IN1K Val) ===", flush=True)
    wave_a_root = Path("/scratch/itscspod/chtaihei/exsym_runs/waveA_ft_s4_directed")
    targets = (
        [f"ft_s4_dir_sym_s{s}" for s in (42, 101, 202, 303)]
        + [f"ft_s4_dir_free_star_s{s}" for s in (42, 101)]
        + [f"ft_s4_dir_free_sym_s{s}" for s in (42, 101)]
    )
    swaps = {}
    for name in targets:
        r = eval_checkpoint_sym_vs_selfavg(wave_a_root / name, val_loader, device, mean_t, std_t)
        swaps[name] = r
        pw = r["post_wout"]
        ds = r["deltas_selfavg_vs_orig"]
        db = r["deltas_bypass_vs_orig"]
        print(
            f"  {name:24s} | cross/orig={pw['cross_over_orig_ratio']:.3f} cos={pw['cos_orig_vs_selfavg']:.3f} | "
            f"SelfAvgΔ: top1={ds['top1_diff']:+.3f}% nll={ds['nll_diff']:+.4f} KL={ds['kl_div']:.4f} agree={ds['pred_agree_pct']:.2f}% | "
            f"BypassΔ: top1={db['top1_diff']:+.3f}% nll={db['nll_diff']:+.4f} KL={db['kl_div']:.4f}",
            flush=True,
        )

    out_p = Path("/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/results/diag_self_avg_and_init_scale.json")
    out_p.write_text(json.dumps({"wall_time_sec": round(time.monotonic() - t0, 1), "init_scale": init_diag, "sym_to_selfavg_swaps": swaps}, indent=2), encoding="utf-8")
    print(f"\nSaved {out_p} in {time.monotonic() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
