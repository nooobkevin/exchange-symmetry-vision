#!/usr/bin/env python3
"""Generate paired-seed JSON wave manifests for all Stage 3 ImageNet-1k experiments."""

from __future__ import annotations

import json
from pathlib import Path

CKPT = "/scratch/itscspod/chtaihei/cliffordnet_runs/in1k_waveC_100ep/in1k_convnext_atto_100ep/checkpoints/last.ckpt"
CFG_DIR = Path("/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/configs")
CFG_DIR.mkdir(parents=True, exist_ok=True)


def write_wave(filename: str, jobs: list[dict]) -> None:
    p = CFG_DIR / filename
    p.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"Wrote {p.name}: {len(jobs)} jobs")


def main() -> None:
    # Wave A: Core ImageNet-1k Fine-Tuning (ConvNeXt-Atto Stage-4 mid, directed_right, 30 ep, 4 paired seeds)
    # Ordered seed-by-seed so every 7-job block finishes a complete paired seed!
    wave_a = []
    for seed in (42, 101, 202, 303):
        for mode in ("none", "anti", "sym", "star", "free_anti", "free_star", "free_sym"):
            wave_a.append({
                "name": f"ft_s4_dir_{mode}_s{seed}",
                "args": {
                    "backbone": "convnext_atto",
                    "mode": mode,
                    "pair_type": "directed_right",
                    "placement": "stage4_mid",
                    "pretrained_ckpt": CKPT,
                    "epochs": 30,
                    "batch_size": 512,
                    "grad_accum": 2,
                    "num_workers": 12,
                    "lr": 4e-4,
                    "adapter_lr_mult": 3.0,
                    "warmup_epochs": 2,
                    "weight_decay": 0.05,
                    "drop_path": 0.1,
                    "ema_decay": 0.999,
                    "data_seed": seed,
                    "init_seed": seed + 1000,
                    "eval_interval": 2,
                },
            })
    write_wave("waveA_in1k_ft_stage4_directed.json", wave_a)

    # Wave B: Generalization across Pair Type (center_context) & Stage Placement (stage3_mid), 30 ep, 3 seeds
    wave_b = []
    for seed in (42, 101, 202):
        # Stage-4 center_context (direct bridge to CliffordNet!)
        for mode in ("anti", "sym", "star", "free_anti", "free_star", "grouped_wedge"):
            wave_b.append({
                "name": f"ft_s4_ctx_{mode}_s{seed}",
                "args": {
                    "backbone": "convnext_atto",
                    "mode": mode,
                    "pair_type": "center_context",
                    "placement": "stage4_mid",
                    "pretrained_ckpt": CKPT,
                    "epochs": 30,
                    "batch_size": 512,
                    "grad_accum": 2,
                    "num_workers": 12,
                    "lr": 4e-4,
                    "adapter_lr_mult": 3.0,
                    "warmup_epochs": 2,
                    "weight_decay": 0.05,
                    "drop_path": 0.1,
                    "ema_decay": 0.999,
                    "data_seed": seed,
                    "init_seed": seed + 2000,
                    "eval_interval": 3,
                },
            })
        # Stage-3 mid directed_right (14x14 mid-level spatial features)
        for mode in ("anti", "sym", "star", "free_anti", "free_star"):
            wave_b.append({
                "name": f"ft_s3_dir_{mode}_s{seed}",
                "args": {
                    "backbone": "convnext_atto",
                    "mode": mode,
                    "pair_type": "directed_right",
                    "placement": "stage3_mid",
                    "pretrained_ckpt": CKPT,
                    "epochs": 30,
                    "batch_size": 512,
                    "grad_accum": 2,
                    "num_workers": 12,
                    "lr": 4e-4,
                    "adapter_lr_mult": 3.0,
                    "warmup_epochs": 2,
                    "weight_decay": 0.05,
                    "drop_path": 0.1,
                    "ema_decay": 0.999,
                    "data_seed": seed,
                    "init_seed": seed + 3000,
                    "eval_interval": 3,
                },
            })
    write_wave("waveB_in1k_ft_pair_and_stage.json", wave_b)

    # Wave C: Direction-Sensitive 4-Way Rotation Task on Full ImageNet-1k (20 ep, 3 paired seeds)
    # Falsifiable hypothesis: On a task that truly depends on orientation, does Free-from-Star
    # rotate toward Antisymmetry and does Anti outperform Sym?
    wave_c = []
    for seed in (42, 101, 202):
        for mode in ("none", "anti", "sym", "star", "free_anti", "free_star", "free_sym"):
            wave_c.append({
                "name": f"rot4_s4_dir_{mode}_s{seed}",
                "args": {
                    "backbone": "convnext_atto",
                    "mode": mode,
                    "pair_type": "directed_right",
                    "placement": "stage4_mid",
                    "pretrained_ckpt": CKPT,
                    "rotation_task": True,
                    "epochs": 15,
                    "batch_size": 512,
                    "grad_accum": 2,
                    "num_workers": 12,
                    "lr": 5e-4,
                    "adapter_lr_mult": 3.0,
                    "warmup_epochs": 1,
                    "weight_decay": 0.05,
                    "drop_path": 0.1,
                    "ema_decay": 0.999,
                    "mixup": 0.0,
                    "cutmix": 0.0,
                    "label_smoothing": 0.0,
                    "data_seed": seed,
                    "init_seed": seed + 4000,
                    "eval_interval": 1,
                },
            })
    write_wave("waveC_in1k_rotation_4way.json", wave_c)

    # Wave D: Full 100-Epoch FROM-SCRATCH ImageNet-1k (Multi-stage ConvNeXt-Atto, 3 paired seeds)
    wave_d = []
    for seed in (42, 101, 202):
        for mode in ("none", "anti", "sym", "star", "free_anti", "free_star", "free_sym", "grouped_wedge"):
            wave_d.append({
                "name": f"scratch_ms_dir_{mode}_s{seed}",
                "args": {
                    "backbone": "convnext_atto",
                    "mode": mode,
                    "pair_type": "directed_right",
                    "placement": "multi_stage",
                    "epochs": 100,
                    "batch_size": 512,
                    "grad_accum": 2,
                    "num_workers": 12,
                    "lr": 4e-3,
                    "adapter_lr_mult": 1.0,
                    "warmup_epochs": 5,
                    "weight_decay": 0.05,
                    "drop_path": 0.1,
                    "ema_decay": 0.9995,
                    "data_seed": seed,
                    "init_seed": seed + 5000,
                    "eval_interval": 5,
                },
            })
    write_wave("waveD_in1k_scratch_100ep.json", wave_d)

    # Wave E: Cross-Backbone Generalization — ViT-Tiny (100-ep ImageNet-1k from scratch, 2 paired seeds)
    wave_e = []
    for seed in (42, 101):
        for mode in ("none", "anti", "sym", "star", "free_anti", "free_star"):
            wave_e.append({
                "name": f"scratch_vittiny_dir_{mode}_s{seed}",
                "args": {
                    "backbone": "vit_tiny_patch16_224",
                    "mode": mode,
                    "pair_type": "directed_right",
                    "vit_insert_block": 8,
                    "epochs": 100,
                    "batch_size": 512,
                    "grad_accum": 2,
                    "num_workers": 12,
                    "lr": 1e-3,
                    "adapter_lr_mult": 1.0,
                    "warmup_epochs": 10,
                    "weight_decay": 0.05,
                    "drop_path": 0.1,
                    "ema_decay": 0.9995,
                    "data_seed": seed,
                    "init_seed": seed + 6000,
                    "eval_interval": 5,
                },
            })
    write_wave("waveE_in1k_vit_tiny_scratch.json", wave_e)


if __name__ == "__main__":
    main()
