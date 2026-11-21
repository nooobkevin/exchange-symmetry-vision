#!/usr/bin/env python3
"""Deep forensics & decomposed symmetric/antisymmetric inference intervention evaluator.

Evaluates trained ConvNeXtWithAdapter / ViTWithAdapter checkpoints on:
  1. Clean Top-1 / Top-5 (full 50k val), Low-Contrast (0.12x), Gaussian Noise, FFT Phase-Only,
     and Horizontal-Flip Consistency (%)
  2. Exact decomposed inference intervention on learned adapters:
     - anti_gain sweep in {1.0, 0.75, 0.5, 0.0} (with sym_gain = 1.0)
     - sym_gain  sweep in {1.0, 0.75, 0.5, 0.0} (with anti_gain = 1.0)
     - full adapter bypass (0.0, 0.0)
     Measured on Clean, Low-Contrast, FFT Phase-Only, and Horizontal-Flip Consistency!
  3. Readout-weighted EffAnti, adapter residual RMS ratio, and theta trajectory from metrics.jsonl!
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from exsym.data import IMAGENET_MEAN, IMAGENET_STD, ShmImageNetDataset, apply_corruption_gpu, normalize_batch_gpu
from exsym.models import ConvNeXtWithAdapter, ViTWithAdapter


def _load_ckpt_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt_path = run_dir / "best_ema.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last.pt"
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    args = ck["args"]
    num_classes = 4 if args.get("rotation_task", False) else 1000

    if args["backbone"].startswith("vit_"):
        model = ViTWithAdapter(
            timm_name=args["backbone"],
            num_classes=num_classes,
            drop_path_rate=0.0,
            adapter_mode=args["mode"],
            pair_type=args["pair_type"],
            insert_after_block=args.get("vit_insert_block", 8),
            rank_ratio=args.get("rank_ratio", 1.0),
            gamma_init=args.get("gamma_init", 0.1),
        )
    else:
        model = ConvNeXtWithAdapter(
            timm_name=args["backbone"],
            num_classes=num_classes,
            drop_path_rate=0.0,
            adapter_mode=args["mode"],
            pair_type=args["pair_type"],
            placement=args["placement"],
            rank_ratio=args.get("rank_ratio", 1.0),
            gamma_init=args.get("gamma_init", 0.1),
            pretrained_ckpt=None,
        )
    model.load_state_dict(ck["ema_model"], strict=True)
    model = model.to(device).eval()
    if not args["backbone"].startswith("vit_"):
        model = model.to(memory_format=torch.channels_last)
    return model, args


@torch.no_grad()
def _eval_corruptions(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    channels_last: bool,
    corruptions: list[str],
    max_batches: int | None = None,
) -> dict[str, float]:
    res: dict[str, float] = {}
    preds_clean = None
    for corr in corruptions:
        c1, c5, tot = 0, 0, 0
        preds_all = []
        for bi, (imgs_u8, targets) in enumerate(val_loader):
            if max_batches is not None and bi >= max_batches:
                break
            imgs_u8 = imgs_u8.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            x = normalize_batch_gpu(imgs_u8, mean_t, std_t)
            x = apply_corruption_gpu(x, corr, mean_t, std_t, seed=bi)
            if channels_last:
                x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            k = min(5, logits.size(-1))
            _, topk = logits.topk(k, dim=1)
            c1 += int(topk[:, 0].eq(targets).sum().item())
            c5 += int(topk.eq(targets.view(-1, 1)).any(dim=1).sum().item())
            tot += int(targets.size(0))
            if corr in ("clean", "hflip"):
                preds_all.append(topk[:, 0].cpu())
        res[corr] = round(100.0 * c1 / max(tot, 1), 2)
        if corr == "clean":
            res["clean_top5"] = round(100.0 * c5 / max(tot, 1), 2)
            preds_clean = torch.cat(preds_all)
        elif corr == "hflip" and preds_clean is not None:
            preds_flip = torch.cat(preds_all)
            res["hflip_consistency"] = round(100.0 * float((preds_clean == preds_flip).float().mean().item()), 2)
    return res


def eval_run(run_dir: Path, device: torch.device, cache_dir: str, fast_intervention_batches: int = 25) -> dict[str, Any]:
    model, args = _load_ckpt_model(run_dir, device)
    channels_last = not args["backbone"].startswith("vit_")
    rot_task = bool(args.get("rotation_task", False))

    val_ds = ShmImageNetDataset(cache_dir=cache_dir, split="val", image_size=224, hflip=False, rotation_task=rot_task)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=6, pin_memory=True)
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # Full 50k corruption evaluation
    corrs = ["clean"] if rot_task else ["clean", "contrast_012", "gaussian_noise", "fft_phase", "hflip"]
    model.set_capture_stats(True)
    full_eval = _eval_corruptions(model, val_loader, device, mean_t, std_t, channels_last, corrs)
    sym_summary = model.summarize_symmetry()
    model.set_capture_stats(False)

    # Decomposed symmetric / antisymmetric inference intervention (on 12,800 val images = 25 batches of 512)
    intervention: dict[str, Any] = {}
    if args["mode"] not in ("none", "grouped_wedge"):
        inv_corrs = ["clean"] if rot_task else ["clean", "contrast_012", "fft_phase", "hflip"]
        for tag, pairs in [
            ("anti_sweep", [(1.0, 1.0), (1.0, 0.75), (1.0, 0.5), (1.0, 0.0)]),
            ("sym_sweep", [(1.0, 1.0), (0.75, 1.0), (0.5, 1.0), (0.0, 1.0)]),
            ("bypass", [(0.0, 0.0)]),
        ]:
            intervention[tag] = {}
            for sg, ag in pairs:
                model.set_intervention(sym_gain=sg, anti_gain=ag)
                key = f"s{sg}_a{ag}"
                intervention[tag][key] = _eval_corruptions(
                    model, val_loader, device, mean_t, std_t, channels_last, inv_corrs, max_batches=fast_intervention_batches
                )
        model.set_intervention(1.0, 1.0)

    # Extract trajectory from metrics.jsonl
    traj = []
    mpath = run_dir / "metrics.jsonl"
    if mpath.exists():
        for line in mpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ev = r.get("val_ema") or r.get("val_raw") or {}
            t_item = {"epoch": r["epoch"], "top1": ev.get("top1"), "theta_grad_norm": r.get("theta_grad_norm")}
            if ev.get("symmetry"):
                t_item["symmetry"] = {
                    k: {
                        "eff_anti": v.get("eff_anti"),
                        "func_anti": v.get("func_anti"),
                        "mean_e_anti": v.get("mean_e_anti"),
                        "rms_ratio": v.get("rms_ratio"),
                    }
                    for k, v in ev["symmetry"].items()
                }
            traj.append(t_item)

    return {
        "name": run_dir.name,
        "args": args,
        "full_eval": full_eval,
        "symmetry_final": sym_summary,
        "intervention": intervention,
        "trajectory": traj,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate forensics & decomposed inference intervention across runs")
    p.add_argument("--runs-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="/dev/shm/in1k_raw_cache_ar320")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    torch.set_num_threads(4)
    t0 = time.monotonic()
    device = torch.device(f"cuda:{args.gpu}")
    runs_root = Path(args.runs_dir)
    all_dirs = sorted([d for d in runs_root.iterdir() if d.is_dir() and ((d / "best_ema.pt").exists() or (d / "last.pt").exists())])
    run_dirs = [d for i, d in enumerate(all_dirs) if i % args.num_shards == args.shard_idx]

    results = {}
    for rdir in run_dirs:
        print(f"[forensics] Evaluating {rdir.name}...", flush=True)
        res = eval_run(rdir, device, args.cache_dir)
        results[rdir.name] = res
        fe = res["full_eval"]
        print(
            f"  {rdir.name:36s}: clean={fe['clean']:.2f}% fft={fe.get('fft_phase', 0):.2f}% "
            f"contrast={fe.get('contrast_012', 0):.2f}% hflip_cons={fe.get('hflip_consistency', 0):.2f}%",
            flush=True,
        )

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"wall_time_sec": round(time.monotonic() - t0, 2), "runs": results}, indent=2), encoding="utf-8")
    print(f"[forensics] Saved {out_p} in {time.monotonic() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
