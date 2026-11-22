#!/usr/bin/env python3
"""Mechanistic Forensics (CPU-only, nice-friendly):
How does 100-epoch from-scratch training (Wave D) unlock directed_right Anti (RMS=0.744, BypassΔ=-5.13%)
whereas 30-epoch fine-tuning on a pretrained backbone (Wave A) suppresses it 10x (RMS=0.037, BypassΔ=-0.01%)?

Tests two exact mathematical demodulation mechanisms in the downstream ConvNeXt block Phi(x + delta)
immediately following adapters['stage4_mid'] (s3.blocks[1]):
  Mechanism 1 (Linear Spatial Derivative Demodulation: odd * odd -> even):
    Does s3.blocks[1].conv_dw (7x7 depthwise conv) shift its horizontal kernel asymmetry
    K_odd / K_total = ||0.5*(K - flip_x(K))||_2^2 / ||K||_2^2 (weighted by adapter channel energy)
    when trained with Anti vs Sym vs SelfAvg vs None?
  Mechanism 2 (Quadratic Nonlinearity Rectification: GELU(x + delta) + GELU(x - delta) - 2*GELU(x)):
    On real validation feature maps (CPU, 256 images), decompose the downstream block output shift
    Delta_Phi(delta) = Phi(x + delta) - Phi(x) into:
      - Odd/Linear response   L(delta) = 0.5 * (Phi(x + delta) - Phi(x - delta))
      - Even/Quadratic rectif Q(delta) = 0.5 * (Phi(x + delta) + Phi(x - delta)) - Phi(x)
    and measure their RMS BEFORE and AFTER Global Average Pooling (GAP) and classifier head!
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from exsym.data import IMAGENET_MEAN, IMAGENET_STD, ShmImageNetDataset, normalize_batch_gpu
from exsym.models import ConvNeXtWithAdapter


@torch.no_grad()
def analyze_checkpoint(ckpt_path: Path, x_val: torch.Tensor) -> dict[str, Any]:
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    args = ck["args"]
    model = ConvNeXtWithAdapter(
        timm_name=args["backbone"],
        num_classes=1000,
        drop_path_rate=0.0,
        adapter_mode=args["mode"],
        pair_type=args["pair_type"],
        placement=args["placement"],
        rank_ratio=args.get("rank_ratio", 1.0),
        gamma_init=args.get("gamma_init", 0.1),
    )
    model.load_state_dict(ck["ema_model"], strict=True)
    model.eval()

    # Downstream block immediately after stage4_mid is stages[3].blocks[1]
    blk_next = model.backbone.stages[3].blocks[1]
    K = blk_next.conv_dw.weight.detach().float()  # (320, 1, 7, 7)
    K_flip_h = torch.flip(K, dims=[3])
    K_even = 0.5 * (K + K_flip_h)
    K_odd = 0.5 * (K - K_flip_h)
    ch_odd_frac = K_odd.pow(2).sum(dim=(1, 2, 3)) / K.pow(2).sum(dim=(1, 2, 3)).clamp(min=1e-12)

    out: dict[str, Any] = {
        "dwconv_horiz_odd_energy_frac_mean": round(float(ch_odd_frac.mean().item()), 4),
    }

    if "stage4_mid" not in model.adapters:
        return out

    ad = model.adapters["stage4_mid"]
    # Weight channel oddness by adapter output channel energy ||gamma_c * W_out[c, :]||^2
    w_ch_energy = (ad.out_proj.weight.detach().float() * ad.gamma.detach().float().view(-1, 1)).pow(2).sum(dim=1)
    out["dwconv_horiz_odd_energy_frac_weighted"] = round(
        float((ch_odd_frac * w_ch_energy).sum().item() / w_ch_energy.sum().clamp(min=1e-12).item()), 4
    )

    # Forward x_val up to stage4 block 0
    h = model.backbone.stem(x_val)
    h = model.backbone.stages[0](h)
    h = model.backbone.stages[1](h)
    h = model.backbone.stages[2](h)
    if "stage3_mid" in model.adapters:
        s2 = model.backbone.stages[2]
        # Recompute stage 2 with stage3_mid properly
        h_prev = model.backbone.stages[1](model.backbone.stages[0](model.backbone.stem(x_val)))
        h = s2.downsample(h_prev)
        mid = len(s2.blocks) // 2
        for b in s2.blocks[:mid]:
            h = b(h)
        h = model.adapters["stage3_mid"](h)
        for b in s2.blocks[mid:]:
            h = b(h)

    s3 = model.backbone.stages[3]
    x_pre = s3.blocks[0](s3.downsample(h))  # (B, 320, 7, 7)
    u, v = ad._make_pair(x_pre)
    delta = ad.forward_delta_from_pair(u, v)

    # Evaluate downstream block Phi = s3.blocks[1] at x_pre, x_pre + delta, x_pre - delta
    phi_0 = blk_next(x_pre)
    phi_pos = blk_next(x_pre + delta)
    phi_neg = blk_next(x_pre - delta)

    total_shift = phi_pos - phi_0
    lin_odd = 0.5 * (phi_pos - phi_neg)          # First-order (odd in delta)
    quad_even = 0.5 * (phi_pos + phi_neg) - phi_0  # Second-order rectification (even in delta!)

    # Spatial RMS (before GAP) vs Pooled RMS (after GAP) vs Logit RMS (after classifier head)
    head = model.backbone.head
    lg_0 = head(phi_0)
    lg_pos = head(phi_pos)
    lg_neg = head(phi_neg)
    lg_total = lg_pos - lg_0
    lg_lin = 0.5 * (lg_pos - lg_neg)
    lg_quad = 0.5 * (lg_pos + lg_neg) - lg_0

    def _rms(t: torch.Tensor) -> float:
        return float(t.pow(2).mean().sqrt().item())

    out.update({
        "delta_rms": round(_rms(delta), 4),
        "spatial_shift_rms": round(_rms(total_shift), 4),
        "spatial_lin_rms": round(_rms(lin_odd), 4),
        "spatial_quad_rms": round(_rms(quad_even), 4),
        "gap_shift_rms": round(_rms(total_shift.mean(dim=(2, 3))), 4),
        "gap_lin_rms": round(_rms(lin_odd.mean(dim=(2, 3))), 4),
        "gap_quad_rms": round(_rms(quad_even.mean(dim=(2, 3))), 4),
        "gap_quad_energy_share": round(
            float(quad_even.mean(dim=(2, 3)).pow(2).mean().item() /
                  max(lin_odd.mean(dim=(2, 3)).pow(2).mean().item() + quad_even.mean(dim=(2, 3)).pow(2).mean().item(), 1e-12)),
            4,
        ),
        "logit_shift_rms": round(_rms(lg_total), 4),
        "logit_lin_rms": round(_rms(lg_lin), 4),
        "logit_quad_rms": round(_rms(lg_quad), 4),
        "logit_quad_energy_share": round(
            float(lg_quad.pow(2).mean().item() / max(lg_lin.pow(2).mean().item() + lg_quad.pow(2).mean().item(), 1e-12)),
            4,
        ),
    })
    return out


def main() -> None:
    torch.set_num_threads(12)
    t0 = time.monotonic()

    val_ds = ShmImageNetDataset(cache_dir="/dev/shm/in1k_raw_cache_ar320", split="val", image_size=224, hflip=False)
    # Sample 128 evenly spaced val images across classes
    idx = np.linspace(0, len(val_ds) - 1, 128, dtype=int)
    imgs = torch.stack([val_ds[int(i)][0] for i in idx])
    mean_t = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x_val = normalize_batch_gpu(imgs, mean_t, std_t)

    results: dict[str, Any] = {"waveA_finetuned": {}, "waveD_scratch": {}}

    root_a = Path("/scratch/itscspod/chtaihei/exsym_runs/waveA_matched_aa0")
    for m in ("none", "anti_rmsmatched", "sym", "self_avg", "free_star"):
        per_s = [analyze_checkpoint(root_a / f"ft_s4_dir_{m}_s{s}/best_ema.pt", x_val) for s in (42, 101, 202)]
        keys = per_s[0].keys()
        results["waveA_finetuned"][m] = {k: round(float(np.mean([p[k] for p in per_s])), 4) for k in keys}

    root_d = Path("/scratch/itscspod/chtaihei/exsym_runs/waveD_scratch_100ep")
    for m in ("none", "anti", "sym", "star", "self_avg", "free_anti", "free_star"):
        per_s = [analyze_checkpoint(root_d / f"scratch_ms_dir_{m}_s{s}/best_ema.pt", x_val) for s in (42, 101, 202)]
        keys = per_s[0].keys()
        results["waveD_scratch"][m] = {k: round(float(np.mean([p[k] for p in per_s])), 4) for k in keys}

    out_p = Path("/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/results/diag_demodulation_coadaptation.json")
    out_p.write_text(json.dumps({"wall_time_sec": round(time.monotonic() - t0, 1), **results}, indent=2), encoding="utf-8")

    print("\n=== DOWNSTREAM DEMODULATION FORENSICS (3-seed mean, CPU) ===")
    for grp in ("waveA_finetuned", "waveD_scratch"):
        print(f"\n--- {grp} ---")
        print(f"{'Mode':16s} | {'DW_Odd%':>7s} | {'d_RMS':>6s} | {'GAP_Lin':>7s} | {'GAP_Quad':>8s} | {'QuadShare':>9s} | {'Logit_Lin':>9s} | {'Logit_Quad':>10s}")
        for m, v in results[grp].items():
            if "delta_rms" not in v:
                print(f"{m:16s} | {100*v['dwconv_horiz_odd_energy_frac_mean']:6.2f}% |    N/A |     N/A |      N/A |       N/A |       N/A |        N/A")
            else:
                print(
                    f"{m:16s} | {100*v['dwconv_horiz_odd_energy_frac_weighted']:6.2f}% | "
                    f"{v['delta_rms']:6.3f} | {v['gap_lin_rms']:7.4f} | {v['gap_quad_rms']:8.4f} | "
                    f"{100*v['gap_quad_energy_share']:8.1f}% | {v['logit_lin_rms']:9.4f} | {v['logit_quad_rms']:10.4f}"
                )
    print(f"\nSaved {out_p} in {time.monotonic() - t0:.1f}s!")


if __name__ == "__main__":
    main()
