#!/usr/bin/env python3
"""Fast, paired-seed ImageNet-1k trainer for exchange-symmetry-vision.

Features:
- Zero-JPEG-decode /dev/shm/in1k_raw_cache_ar320 memmap loader
- Exact paired-seed data order & GPU-side Mixup/CutMix so Anti/Sym/Star/Free see identical batches
- BF16 AMP + channels_last + torch.compile + fused AdamW + Cosine LR + EMA
- Supports both single-GPU and multi-GPU DDP (torchrun)
- Per-epoch JSONL telemetry: val_top1, val_top5, ema_top1, ema_top5, adapter rms_ratio,
  readout-weighted EffAnti, unweighted mean_e_anti, theta quantiles, grad_norm_theta, img/s
- Per-epoch resumable checkpoints & deadline guard
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

from exsym.data import IMAGENET_MEAN, IMAGENET_STD, ShmImageNetDataset, normalize_batch_gpu
from exsym.models import ConvNeXtWithAdapter, ViTWithAdapter


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


def build_model(args: argparse.Namespace) -> nn.Module:
    num_classes = 4 if args.rotation_task else 1000
    if args.backbone.startswith("vit_"):
        model = ViTWithAdapter(
            timm_name=args.backbone,
            num_classes=num_classes,
            drop_path_rate=args.drop_path,
            adapter_mode=args.mode,
            pair_type=args.pair_type,
            insert_after_block=args.vit_insert_block,
            rank_ratio=args.rank_ratio,
            gamma_init=args.gamma_init,
        )
    else:
        model = ConvNeXtWithAdapter(
            timm_name=args.backbone,
            num_classes=num_classes,
            drop_path_rate=args.drop_path,
            adapter_mode=args.mode,
            pair_type=args.pair_type,
            placement=args.placement,
            rank_ratio=args.rank_ratio,
            gamma_init=args.gamma_init,
            pretrained_ckpt=args.pretrained_ckpt if args.pretrained_ckpt else None,
        )
    return model


@torch.no_grad()
def update_ema(ema_model: nn.Module, raw_model: nn.Module, decay: float) -> None:
    for p_ema, p_raw in zip(ema_model.parameters(), raw_model.parameters()):
        p_ema.mul_(decay).add_(p_raw.detach(), alpha=1.0 - decay)
    for b_ema, b_raw in zip(ema_model.buffers(), raw_model.buffers()):
        b_ema.copy_(b_raw)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    channels_last: bool,
    capture_sym: bool = False,
    world_size: int = 1,
) -> dict[str, Any]:
    model.eval()
    raw = model.module if isinstance(model, DDP) else model
    raw_inner = getattr(raw, "_orig_mod", raw)
    if capture_sym and hasattr(raw_inner, "set_capture_stats"):
        raw_inner.set_capture_stats(True)

    correct1 = torch.zeros(1, device=device, dtype=torch.float64)
    correct5 = torch.zeros(1, device=device, dtype=torch.float64)
    loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    total = torch.zeros(1, device=device, dtype=torch.float64)

    sym_accum: dict[str, dict[str, list[float]]] = {}
    num_batches = 0

    for imgs_u8, targets in val_loader:
        imgs_u8 = imgs_u8.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        x = normalize_batch_gpu(imgs_u8, mean_t, std_t)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.float(), targets, reduction="sum")

        bs = targets.size(0)
        total += bs
        loss_sum += loss.double()
        k = min(5, logits.size(-1))
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(targets.view(-1, 1))
        correct1 += correct[:, :1].sum().double()
        correct5 += correct[:, :k].sum().double()

        if capture_sym and num_batches < 16 and hasattr(raw_inner, "summarize_symmetry"):
            s = raw_inner.summarize_symmetry()
            for aname, adict in s.items():
                sym_accum.setdefault(aname, {})
                for key in ("eff_anti", "func_anti", "rms_ratio", "delta_rms", "x_rms", "top25_energy_frac"):
                    if key in adict:
                        sym_accum[aname].setdefault(key, []).append(float(adict[key]))
        num_batches += 1

    if capture_sym and hasattr(raw_inner, "set_capture_stats"):
        raw_inner.set_capture_stats(False)

    if world_size > 1:
        for t in (correct1, correct5, loss_sum, total):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    n = max(float(total.item()), 1.0)
    out: dict[str, Any] = {
        "loss": float(loss_sum.item() / n),
        "top1": float(100.0 * correct1.item() / n),
        "top5": float(100.0 * correct5.item() / n),
    }
    if capture_sym and hasattr(raw_inner, "summarize_symmetry"):
        base_sym = raw_inner.summarize_symmetry()
        for aname, adict in base_sym.items():
            if aname in sym_accum:
                for key, vals in sym_accum[aname].items():
                    adict[key] = float(np.mean(vals))
        out["symmetry"] = base_sym
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast ImageNet-1k trainer for exchange-symmetry-vision")
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="/dev/shm/in1k_raw_cache_ar320")
    p.add_argument("--backbone", type=str, default="convnext_atto")
    p.add_argument("--mode", type=str, default="free_anti")
    p.add_argument("--pair-type", type=str, default="directed_right")
    p.add_argument("--placement", type=str, default="stage4_mid")
    p.add_argument("--vit-insert-block", type=int, default=8)
    p.add_argument("--rank-ratio", type=float, default=1.0)
    p.add_argument("--gamma-init", type=float, default=0.1)
    p.add_argument("--pretrained-ckpt", type=str, default="")
    p.add_argument("--rotation-task", action="store_true")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--adapter-lr-mult", type=float, default=2.0)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=5.0)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--init-seed", type=int, default=42)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--save-best-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="Optional cap for micro-benchmarking")
    p.add_argument("--deadline-iso", type=str, default="")
    return p.parse_args()


def main() -> None:
    torch.set_num_threads(4)
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    run_dir = Path(args.output_dir) / args.name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Separate init_seed for adapter initialization so paired runs share data_seed
    seed_everything(args.init_seed)
    raw_model = build_model(args).to(device)
    channels_last = not args.backbone.startswith("vit_")
    if channels_last:
        raw_model = raw_model.to(memory_format=torch.channels_last)

    n_total = sum(p.numel() for p in raw_model.parameters())
    n_adapter = sum(p.numel() for p in raw_model.adapters.parameters()) if hasattr(raw_model, "adapters") else 0

    ema_model = copy.deepcopy(raw_model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Separate parameter groups: no weight decay on 1D/bias/theta/gamma; optional lr mult for adapter
    backbone_decay, backbone_nodecay = [], []
    adapter_decay, adapter_nodecay = [], []
    for pname, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        is_adapter = pname.startswith("adapters.")
        no_wd = (p.ndim <= 1) or pname.endswith(".bias") or ("gamma" in pname) or ("theta" in pname)
        if is_adapter:
            (adapter_nodecay if no_wd else adapter_decay).append(p)
        else:
            (backbone_nodecay if no_wd else backbone_decay).append(p)

    param_groups = []
    if backbone_decay:
        param_groups.append({"params": backbone_decay, "lr": args.lr, "weight_decay": args.weight_decay, "lr_mult": 1.0})
    if backbone_nodecay:
        param_groups.append({"params": backbone_nodecay, "lr": args.lr, "weight_decay": 0.0, "lr_mult": 1.0})
    if adapter_decay:
        param_groups.append({"params": adapter_decay, "lr": args.lr * args.adapter_lr_mult, "weight_decay": args.weight_decay, "lr_mult": args.adapter_lr_mult})
    if adapter_nodecay:
        param_groups.append({"params": adapter_nodecay, "lr": args.lr * args.adapter_lr_mult, "weight_decay": 0.0, "lr_mult": args.adapter_lr_mult})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8, fused=True)

    # Datasets
    train_ds = ShmImageNetDataset(
        cache_dir=args.cache_dir,
        split="train",
        image_size=224,
        hflip=not args.rotation_task,
        rotation_task=args.rotation_task,
    )
    val_ds = ShmImageNetDataset(
        cache_dir=args.cache_dir,
        split="val",
        image_size=224,
        hflip=False,
        rotation_task=args.rotation_task,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.data_seed)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    def _worker_init(wid: int) -> None:
        seed_everything(args.data_seed + wid * 9973 + rank * 100003)

    g_loader = torch.Generator().manual_seed(args.data_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=_worker_init,
        generator=g_loader if train_sampler is None else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(2, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Mixup/CutMix (disabled for 4-way rotation task)
    use_mixup = (not args.rotation_task) and (args.mixup > 0.0 or args.cutmix > 0.0)
    mixup_fn = (
        Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.label_smoothing,
            num_classes=1000,
        )
        if use_mixup
        else None
    )
    criterion = SoftTargetCrossEntropy() if use_mixup else nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Resume if requested
    start_epoch = 0
    best_ema_top1 = 0.0
    best_raw_top1 = 0.0
    last_ckpt_path = run_dir / "last.pt"
    if args.resume and last_ckpt_path.exists():
        ck = torch.load(str(last_ckpt_path), map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ck["model"])
        ema_model.load_state_dict(ck["ema_model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck["epoch"]) + 1
        best_ema_top1 = float(ck.get("best_ema_top1", 0.0))
        best_raw_top1 = float(ck.get("best_raw_top1", 0.0))
        if is_main:
            print(f"[{args.name}] Resumed from epoch {start_epoch}, best_ema={best_ema_top1:.2f}%", flush=True)

    # Wrap compile & DDP
    train_model: nn.Module = raw_model
    if not args.no_compile:
        train_model = torch.compile(train_model)
    if world_size > 1:
        train_model = DDP(train_model, device_ids=[local_rank], broadcast_buffers=False)

    mean_t = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_opt_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def set_lr(opt_step: int) -> float:
        if opt_step < warmup_steps:
            base = args.lr * float(opt_step + 1) / float(max(1, warmup_steps))
        else:
            prog = float(opt_step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
            base = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))
        for pg in optimizer.param_groups:
            pg["lr"] = base * pg["lr_mult"]
        return base

    deadline_ts = dt.datetime.fromisoformat(args.deadline_iso).timestamp() if args.deadline_iso else None
    metrics_path = run_dir / "metrics.jsonl"

    # Epoch-0 baseline eval before any training step (verifies identical step-0 start!)
    if start_epoch == 0 and args.max_steps == 0:
        ev0 = evaluate(raw_model, val_loader, device, mean_t, std_t, channels_last, capture_sym=True, world_size=world_size)
        if is_main:
            rec0 = {"epoch": -1, "step": 0, "val_raw": ev0, "n_params": n_total, "n_adapter_params": n_adapter}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec0) + "\n")
            print(
                f"[{args.name}] Step-0 | params={n_total/1e6:.3f}M (adapter={n_adapter/1e6:.3f}M) | "
                f"val_top1={ev0['top1']:.2f}% | sym={ev0.get('symmetry', {})}",
                flush=True,
            )

    global_opt_step = start_epoch * steps_per_epoch
    t_run_start = time.monotonic()

    for epoch in range(start_epoch, args.epochs):
        if deadline_ts is not None and time.time() > deadline_ts:
            if is_main:
                print(f"[{args.name}] Reached deadline before epoch {epoch}; stopping cleanly.", flush=True)
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        else:
            # Deterministic paired epoch shuffling for single-GPU runs
            g_loader.manual_seed(args.data_seed + epoch * 1009)

        # Deterministic paired mixup RNG per epoch
        seed_everything(args.data_seed + epoch * 31337 + rank * 17)

        train_model.train()
        t_ep0 = time.monotonic()
        t_data_wait = 0.0
        t_prev = time.monotonic()
        loss_accum = 0.0
        n_steps_ep = 0
        n_imgs_ep = 0
        warmup_bench_steps = 10 if args.max_steps > 0 else 0
        theta_grad_norms: list[float] = []
        optimizer.zero_grad(set_to_none=True)

        for it, (imgs_u8, targets) in enumerate(train_loader):
            t_data_wait += time.monotonic() - t_prev
            imgs_u8 = imgs_u8.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            x = normalize_batch_gpu(imgs_u8, mean_t, std_t)
            if mixup_fn is not None:
                x, targets_mix = mixup_fn(x, targets)
            else:
                targets_mix = targets
            if channels_last:
                x = x.to(memory_format=torch.channels_last)

            is_sync_step = ((it + 1) % args.grad_accum == 0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = train_model(x)
                loss = criterion(logits, targets_mix) / args.grad_accum

            loss.backward()
            loss_accum += float(loss.detach().item()) * args.grad_accum
            n_imgs_ep += int(imgs_u8.size(0)) * world_size

            if is_sync_step:
                lr_now = set_lr(global_opt_step)
                if args.clip_grad > 0.0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip_grad)
                # Record theta gradient norm periodically
                if n_steps_ep % 50 == 0 and hasattr(raw_model, "adapters"):
                    gn = 0.0
                    for ad in raw_model.adapters.values():
                        if hasattr(ad.core, "theta") and isinstance(ad.core.theta, nn.Parameter) and ad.core.theta.grad is not None:
                            gn += float(ad.core.theta.grad.detach().float().norm().item())
                    theta_grad_norms.append(gn)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema_model, raw_model, args.ema_decay)
                global_opt_step += 1
                n_steps_ep += 1

                # Reset benchmark timer after torch.compile warmup steps
                if warmup_bench_steps > 0 and n_steps_ep == warmup_bench_steps:
                    torch.cuda.synchronize()
                    t_ep0 = time.monotonic()
                    t_data_wait = 0.0
                    n_imgs_ep = 0

            t_prev = time.monotonic()
            if args.max_steps > 0 and n_steps_ep >= args.max_steps:
                break

        torch.cuda.synchronize()
        ep_sec = time.monotonic() - t_ep0
        img_per_sec = n_imgs_ep / max(ep_sec, 1e-6)
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        if args.max_steps > 0:
            if is_main:
                bench_rec = {
                    "name": args.name,
                    "world_size": world_size,
                    "batch_size_per_gpu": args.batch_size,
                    "steps": n_steps_ep,
                    "wall_sec": round(ep_sec, 2),
                    "img_per_sec": round(img_per_sec, 1),
                    "data_wait_frac": round(t_data_wait / max(ep_sec, 1e-6), 4),
                    "peak_vram_gb": round(peak_vram_gb, 2),
                }
                print("BENCH_RESULT:" + json.dumps(bench_rec), flush=True)
            break

        do_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch == args.epochs - 1)
        ev_raw = None
        ev_ema = None
        if do_eval:
            ev_raw = evaluate(raw_model, val_loader, device, mean_t, std_t, channels_last, capture_sym=True, world_size=world_size)
            ev_ema = evaluate(ema_model, val_loader, device, mean_t, std_t, channels_last, capture_sym=True, world_size=world_size)

        if is_main:
            rec: dict[str, Any] = {
                "epoch": epoch,
                "step": global_opt_step,
                "train_loss": round(loss_accum / max(len(train_loader), 1), 4),
                "lr": lr_now,
                "ep_sec": round(ep_sec, 2),
                "img_per_sec": round(img_per_sec, 1),
                "data_wait_frac": round(t_data_wait / max(ep_sec, 1e-6), 4),
                "peak_vram_gb": round(peak_vram_gb, 2),
                "theta_grad_norm": round(float(np.mean(theta_grad_norms)), 6) if theta_grad_norms else 0.0,
            }
            if ev_raw is not None:
                rec["val_raw"] = ev_raw
                rec["val_ema"] = ev_ema
                if ev_raw["top1"] > best_raw_top1:
                    best_raw_top1 = ev_raw["top1"]
                if ev_ema["top1"] > best_ema_top1:
                    best_ema_top1 = ev_ema["top1"]
                    torch.save(
                        {"epoch": epoch, "model": raw_model.state_dict(), "ema_model": ema_model.state_dict(), "args": vars(args), "val_ema": ev_ema},
                        str(run_dir / "best_ema.pt"),
                    )
                # Compact status line
                sym_str = ""
                if ev_ema.get("symmetry"):
                    parts = []
                    for aname, ad in ev_ema["symmetry"].items():
                        ea = ad.get("eff_anti", ad.get("weight_weighted_e_anti"))
                        fa = ad.get("func_anti")
                        rr = ad.get("rms_ratio", 0.0)
                        if ea is not None and fa is not None:
                            parts.append(f"{aname}:EffA={ea:.3f},FuncA={fa:.3f},rms={rr:.3f}")
                        elif fa is not None:
                            parts.append(f"{aname}:FuncA={fa:.3f},rms={rr:.3f}")
                        else:
                            parts.append(f"{aname}:rms={rr:.3f}")
                    sym_str = " | " + " ".join(parts)
                print(
                    f"[{args.name}] Ep {epoch:03d}/{args.epochs:03d} | loss={rec['train_loss']:.3f} | "
                    f"raw={ev_raw['top1']:.2f}% ema={ev_ema['top1']:.2f}% (best={best_ema_top1:.2f}%) | "
                    f"{img_per_sec:.0f} img/s ({ep_sec:.0f}s, wait={100*rec['data_wait_frac']:.1f}%){sym_str}",
                    flush=True,
                )

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            # Save resumable checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model": raw_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_ema_top1": best_ema_top1,
                    "best_raw_top1": best_raw_top1,
                    "args": vars(args),
                },
                str(last_ckpt_path),
            )

    if is_main and args.max_steps == 0:
        summary = {
            "name": args.name,
            "mode": args.mode,
            "pair_type": args.pair_type,
            "placement": args.placement,
            "backbone": args.backbone,
            "data_seed": args.data_seed,
            "init_seed": args.init_seed,
            "n_params": n_total,
            "n_adapter_params": n_adapter,
            "best_ema_top1": round(best_ema_top1, 4),
            "best_raw_top1": round(best_raw_top1, 4),
            "wall_time_min": round((time.monotonic() - t_run_start) / 60.0, 2),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{args.name}] FINISHED: best_ema={best_ema_top1:.2f}%, wall={summary['wall_time_min']:.2f}m", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
