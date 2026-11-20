#!/usr/bin/env python3
"""Experiment 2: Frozen ConvNeXt-Atto Multi-Shot & Directional-Task Probes.

Extracts fixed spatial feature maps from our verified 100-epoch ImageNet-1k ConvNeXt-Atto
checkpoint (EMA weights, 71.1% Top-1) at Stage 3 (14x14, C=160) and Stage 4 block 0 (7x7, C=320),
then trains 9 matched heads across 3 paired seeds on two contrasting tasks:
  1. ImageNet-1k Classification (reflection-invariant): 10 shots/class (10k) and 50 shots/class (50k)
     Evaluated on Clean, Low-Contrast (0.12x), Gaussian Noise, FFT Phase-Only, and HFlip Consistency.
  2. 4-Way Rotation Prediction {0, 90, 180, 270} (direction-sensitive!): 10k and 50k samples
     Evaluated on 20k held-out validation images.

All heads share the exact same pre-extracted GPU/RAM feature tensors and identical batch orderings.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from exsym.adapter import ExchangeBilinearCore
from exsym.data import IMAGENET_MEAN, IMAGENET_STD, apply_corruption_gpu, make_stratified_indices
from exsym.models import ConvNeXtWithAdapter, FrozenPairHead

HEAD_MODES = [
    "linear_u",
    "linear_uv",
    "grouped_wedge",
    "anti",
    "sym",
    "star",
    "free_anti",
    "free_star",
    "free_sym",
]


@torch.no_grad()
def extract_features_for_indices(
    backbone_model: ConvNeXtWithAdapter,
    imgs_mm: np.ndarray,
    shps_mm: np.ndarray | None,
    indices: np.ndarray,
    stage: int,
    device: torch.device,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    batch_size: int = 256,
    corruption: str = "clean",
    rotations: np.ndarray | None = None,
) -> torch.Tensor:
    """Extract (N, C, H, W) fp16 spatial feature maps on CPU pinned/plain tensor."""
    backbone_model.eval()
    n = len(indices)
    feats_out = None

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        imgs_list = []
        for ii, idx in enumerate(batch_idx):
            if shps_mm is not None:
                nh, nw = int(shps_mm[idx, 0]), int(shps_mm[idx, 1])
                arr = np.array(imgs_mm[idx, :, :nh, :nw], copy=True)
                t = torch.from_numpy(arr).float().unsqueeze(0)
                # Center-crop/resize to 224x224 cleanly for fixed feature extraction
                t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False, antialias=True).squeeze(0)
            else:
                arr = np.array(imgs_mm[idx], copy=True)
                t = torch.from_numpy(arr).float()
            if rotations is not None:
                k = int(rotations[start + ii])
                if k > 0:
                    t = torch.rot90(t, k=k, dims=[1, 2])
            imgs_list.append(t)

        x = torch.stack(imgs_list, dim=0).to(device, non_blocking=True).div_(255.0)
        x = (x - mean_t) / std_t
        if corruption != "clean":
            x = apply_corruption_gpu(x, corruption, mean_t, std_t, seed=start)
        x = x.to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f = backbone_model.extract_stage_map(x, stage=stage)
        f_fp16 = f.to(torch.float16).cpu()
        if feats_out is None:
            feats_out = torch.empty((n, *f_fp16.shape[1:]), dtype=torch.float16)
        feats_out[start:end] = f_fp16

    assert feats_out is not None
    return feats_out


def train_and_eval_head(
    mode: str,
    pair_type: str,
    dim: int,
    num_classes: int,
    rank: int,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    val_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    wd: float,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    head = FrozenPairHead(
        dim=dim,
        num_classes=num_classes,
        mode=mode,
        rank=rank,
        pair_type=pair_type,
        add_linear_skip=True,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())

    decay, nodecay = [], []
    for pname, p in head.named_parameters():
        if p.ndim <= 1 or pname.endswith(".bias") or "theta" in pname:
            nodecay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}],
        lr=lr,
        fused=True,
    )
    n_tr = x_tr.shape[0]
    steps_per_ep = max(1, math.ceil(n_tr / batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep, eta_min=1e-5)

    # Keep training tensors on GPU (50k x 320 x 7 x 7 fp16 = only ~1.5 GB on 80GB H800!)
    x_tr_gpu = x_tr.to(device)
    y_tr_gpu = y_tr.to(device)
    g = torch.Generator(device=device).manual_seed(seed * 1009 + 7)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n_tr, generator=g, device=device)
        for s in range(0, n_tr, batch_size):
            idx = perm[s : s + batch_size]
            xb = x_tr_gpu[idx]
            yb = y_tr_gpu[idx]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = head(xb)
                loss = F.cross_entropy(logits.float(), yb, label_smoothing=0.05 if num_classes > 10 else 0.0)
            loss.backward()
            opt.step()
            sched.step()

    head.eval()
    metrics: dict[str, Any] = {"n_params": n_params}
    preds_clean = None
    with torch.no_grad():
        for vname, (xv, yv) in val_sets.items():
            correct = 0
            total = 0
            all_preds = []
            for s in range(0, len(yv), 512):
                xb = xv[s : s + 512].to(device)
                yb = yv[s : s + 512].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = head(xb)
                pr = logits.argmax(dim=1)
                correct += int((pr == yb).sum().item())
                total += int(yb.size(0))
                if vname in ("clean", "hflip"):
                    all_preds.append(pr.cpu())
            metrics[vname] = round(100.0 * correct / max(total, 1), 2)
            if vname == "clean":
                preds_clean = torch.cat(all_preds)
            elif vname == "hflip" and preds_clean is not None:
                preds_flip = torch.cat(all_preds)
                metrics["hflip_consistency"] = round(100.0 * float((preds_clean == preds_flip).float().mean().item()), 2)

        # Symmetry telemetry (readout-weighted EffAnti AND exact swapped-pair FuncAnti!)
        if head.core is not None and head.mode != "linear_uv":
            xb = val_sets["clean"][0][:1024].to(device)
            u_hwc, v_hwc = head._make_uv(xb.float())
            f_uv = head.core(u_hwc, v_hwc).float()
            f_vu = head.core(v_hwc, u_hwc).float()
            # Functional antisymmetry BEFORE GELU (exact bilinear interaction energy)
            e_sym_pre = float((0.5 * (f_uv + f_vu)).pow(2).mean().item())
            e_anti_pre = float((0.5 * (f_uv - f_vu)).pow(2).mean().item())
            metrics["func_anti_pre"] = round(e_anti_pre / max(e_sym_pre + e_anti_pre, 1e-12), 4)
            # Functional antisymmetry of actual pooled readout logits contributed by bilinear branch!
            h_uv = head.act(f_uv).mean(dim=(1, 2))
            h_vu = head.act(f_vu).mean(dim=(1, 2))
            w_bilin = head.head.weight.detach().float()[:, -rank:]
            lg_uv = h_uv @ w_bilin.T
            lg_vu = h_vu @ w_bilin.T
            e_sym_lg = float((0.5 * (lg_uv + lg_vu)).pow(2).mean().item())
            e_anti_lg = float((0.5 * (lg_uv - lg_vu)).pow(2).mean().item())
            metrics["func_anti_logit"] = round(e_anti_lg / max(e_sym_lg + e_anti_lg, 1e-12), 4)

            if isinstance(head.core, ExchangeBilinearCore):
                _, e_anti = head.core.get_energy_split()
                e_anti = e_anti.detach().float().cpu()
                f_var = f_uv.var(dim=(0, 1, 2), unbiased=False).cpu()
                w_sq = w_bilin.detach().cpu().pow(2).sum(dim=0)
                ce = w_sq * f_var
                metrics["mean_e_anti"] = round(float(e_anti.mean().item()), 4)
                metrics["eff_anti"] = round(float((ce * e_anti).sum().item() / max(ce.sum().item(), 1e-12)), 4)
            w_lin = head.head.weight.detach().float()[:, :dim]
            metrics["bilin_weight_frac"] = round(
                float(w_bilin.pow(2).sum().item() / max(w_bilin.pow(2).sum().item() + w_lin.pow(2).sum().item(), 1e-12)),
                4,
            )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: Frozen ConvNeXt-Atto Multi-Shot & Directional Probes")
    parser.add_argument("--cache-dir", type=str, default="/dev/shm/in1k_raw_cache_ar320")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/scratch/itscspod/chtaihei/cliffordnet_runs/in1k_waveC_100ep/in1k_convnext_atto_100ep/checkpoints/last.ckpt",
    )
    parser.add_argument("--stage", type=int, default=4, choices=[3, 4])
    parser.add_argument("--pair-type", type=str, default="directed_right", choices=["directed_right", "center_context"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    t0 = time.monotonic()
    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cuda.matmul.allow_tf32 = True

    cache_dir = Path(args.cache_dir)
    tr_imgs = np.load(str(cache_dir / "train_images.npy"), mmap_mode="r")
    tr_lbls = np.load(str(cache_dir / "train_labels.npy"), mmap_mode="r")
    tr_shps = np.load(str(cache_dir / "train_shapes.npy"), mmap_mode="r")
    va_imgs = np.load(str(cache_dir / "val_images.npy"), mmap_mode="r")
    va_lbls = np.load(str(cache_dir / "val_labels.npy"), mmap_mode="r")

    backbone = ConvNeXtWithAdapter(
        timm_name="convnext_atto",
        num_classes=1000,
        adapter_mode="none",
        pretrained_ckpt=args.ckpt,
    ).to(device).to(memory_format=torch.channels_last).eval()

    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    dim = 320 if args.stage == 4 else 160
    rank = 256 if args.stage == 4 else 160
    seeds = [42, 101, 202]

    all_results: dict[str, Any] = {"config": vars(args), "tasks": {}}

    # ---- TASK A: Reflection-invariant ImageNet-1k few-shot (10 & 50 shots/class) ----
    print(f"[exp2 stage={args.stage} pair={args.pair_type}] Extracting ImageNet val features (20k stratified)...", flush=True)
    va_idx_in1k = make_stratified_indices(np.asarray(va_lbls), shots_per_class=20, seed=999)
    va_y_in1k = torch.from_numpy(np.asarray(va_lbls)[va_idx_in1k].copy()).long()
    val_sets_in1k: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for corr in ("clean", "contrast_012", "gaussian_noise", "fft_phase", "hflip"):
        fv = extract_features_for_indices(backbone, va_imgs, None, va_idx_in1k, args.stage, device, mean_t, std_t, corruption=corr)
        val_sets_in1k[corr] = (fv, va_y_in1k)

    for shots, epochs, lr in [(10, 40, 2e-3), (50, 30, 1.5e-3)]:
        tname = f"in1k_{shots}shot"
        print(f"[exp2] Extracting {tname} train features ({shots*1000} images)...", flush=True)
        tr_idx = make_stratified_indices(np.asarray(tr_lbls), shots_per_class=shots, seed=42)
        tr_y = torch.from_numpy(np.asarray(tr_lbls)[tr_idx].copy()).long()
        tr_f = extract_features_for_indices(backbone, tr_imgs, tr_shps, tr_idx, args.stage, device, mean_t, std_t)

        task_res: dict[str, Any] = {}
        for mode in HEAD_MODES:
            per_seed = {}
            for sd in seeds:
                per_seed[str(sd)] = train_and_eval_head(
                    mode=mode, pair_type=args.pair_type, dim=dim, num_classes=1000, rank=rank,
                    x_tr=tr_f, y_tr=tr_y, val_sets=val_sets_in1k,
                    seed=sd, epochs=epochs, batch_size=256, lr=lr, wd=1e-2, device=device,
                )
            # Aggregate mean/std
            agg: dict[str, Any] = {"per_seed": per_seed, "n_params": per_seed["42"]["n_params"]}
            for k in ("clean", "contrast_012", "gaussian_noise", "fft_phase", "hflip_consistency", "eff_anti", "mean_e_anti", "func_anti_pre", "func_anti_logit", "bilin_weight_frac"):
                vals = [per_seed[str(sd)][k] for sd in seeds if k in per_seed[str(sd)]]
                if vals:
                    agg[f"{k}_mean"] = round(float(np.mean(vals)), 3)
                    agg[f"{k}_std"] = round(float(np.std(vals, ddof=1)), 3)
            task_res[mode] = agg
            ea_s = f"EffAnti={agg['eff_anti_mean']:.3f}" if "eff_anti_mean" in agg else "EffAnti=N/A"
            print(
                f"  [{tname}] {mode:14s}: clean={agg['clean_mean']:.2f}±{agg['clean_std']:.2f} "
                f"fft={agg['fft_phase_mean']:.2f} contrast={agg['contrast_012_mean']:.2f} {ea_s}",
                flush=True,
            )
        all_results["tasks"][tname] = task_res
        del tr_f

    # ---- TASK B: Direction-sensitive 4-way Rotation Classification {0, 90, 180, 270} ----
    print(f"[exp2 stage={args.stage} pair={args.pair_type}] Extracting 4-way rotation task features...", flush=True)
    rng = np.random.default_rng(777)
    va_rot_idx = np.arange(20000)
    va_rot_lbl = (va_rot_idx % 4).astype(np.int64)
    va_rot_f = extract_features_for_indices(backbone, va_imgs, None, va_rot_idx, args.stage, device, mean_t, std_t, rotations=va_rot_lbl)
    val_sets_rot = {"clean": (va_rot_f, torch.from_numpy(va_rot_lbl))}

    for n_samp, epochs in [(10000, 40), (40000, 30)]:
        tname = f"rot4_{n_samp//1000}k"
        tr_rot_idx = rng.choice(len(tr_lbls), size=n_samp, replace=False)
        tr_rot_lbl = rng.integers(0, 4, size=n_samp, dtype=np.int64)
        tr_rot_f = extract_features_for_indices(backbone, tr_imgs, tr_shps, tr_rot_idx, args.stage, device, mean_t, std_t, rotations=tr_rot_lbl)
        tr_rot_y = torch.from_numpy(tr_rot_lbl)

        task_res = {}
        for mode in HEAD_MODES:
            per_seed = {}
            for sd in seeds:
                per_seed[str(sd)] = train_and_eval_head(
                    mode=mode, pair_type=args.pair_type, dim=dim, num_classes=4, rank=rank,
                    x_tr=tr_rot_f, y_tr=tr_rot_y, val_sets=val_sets_rot,
                    seed=sd, epochs=epochs, batch_size=256, lr=1.5e-3, wd=1e-2, device=device,
                )
            agg = {"per_seed": per_seed, "n_params": per_seed["42"]["n_params"]}
            for k in ("clean", "eff_anti", "mean_e_anti", "func_anti_pre", "func_anti_logit", "bilin_weight_frac"):
                vals = [per_seed[str(sd)][k] for sd in seeds if k in per_seed[str(sd)]]
                if vals:
                    agg[f"{k}_mean"] = round(float(np.mean(vals)), 3)
                    agg[f"{k}_std"] = round(float(np.std(vals, ddof=1)), 3)
            task_res[mode] = agg
            ea_s = f"EffAnti={agg['eff_anti_mean']:.3f}" if "eff_anti_mean" in agg else "EffAnti=N/A"
            print(f"  [{tname}] {mode:14s}: rot_acc={agg['clean_mean']:.2f}±{agg['clean_std']:.2f} {ea_s}", flush=True)
        all_results["tasks"][tname] = task_res
        del tr_rot_f

    all_results["wall_time_sec"] = round(time.monotonic() - t0, 2)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"[exp2] DONE -> {out_p} in {time.monotonic() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
