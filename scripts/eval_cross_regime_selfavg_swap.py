#!/usr/bin/env python3
"""Cross-Regime Sym -> SelfAvg Zero-Retraining Swap Audit (8-GPU parallel):
Compares how much trained `sym` and `free_star` models actually rely on the cross-token
difference term Delta_cross = SelfAvg(u,v) - Sym(u,v) = A(u-v)*B(u-v)/sqrt(2) across:
  1. Wave A-Matched : 30-epoch Fine-Tuned ConvNeXt-Atto (Stage 4)
  2. Wave D         : 100-epoch From-Scratch ConvNeXt-Atto (Multi-Stage: Stage 3 + Stage 4)
  3. Wave E         : 100-epoch From-Scratch ViT-Tiny (Block 8, 14x14 patch grid)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exsym.data import IMAGENET_MEAN, IMAGENET_STD, ShmImageNetDataset, normalize_batch_gpu
from exsym.models import ConvNeXtWithAdapter, ViTWithAdapter


_VAL_IMGS = None
_VAL_LBLS = None


@torch.no_grad()
def _eval_one_ckpt(task: tuple[int, str, str]) -> dict[str, Any]:
    gpu_id, regime, ckpt_str = task
    device = torch.device(f"cuda:{gpu_id}")
    run_dir = Path(ckpt_str)
    ck = torch.load(str(run_dir / "best_ema.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    is_vit = args["backbone"].startswith("vit_")

    if is_vit:
        model = ViTWithAdapter(
            timm_name=args["backbone"], num_classes=1000, drop_path_rate=0.0,
            adapter_mode=args["mode"], pair_type=args["pair_type"],
            insert_after_block=args.get("vit_insert_block", 8),
            rank_ratio=args.get("rank_ratio", 1.0), gamma_init=args.get("gamma_init", 0.1),
        )
    else:
        model = ConvNeXtWithAdapter(
            timm_name=args["backbone"], num_classes=1000, drop_path_rate=0.0,
            adapter_mode=args["mode"], pair_type=args["pair_type"], placement=args["placement"],
            rank_ratio=args.get("rank_ratio", 1.0), gamma_init=args.get("gamma_init", 0.1),
        )
    model.load_state_dict(ck["ema_model"], strict=True)
    model = model.to(device).eval()
    if not is_vit:
        model = model.to(memory_format=torch.channels_last)

    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    orig_modes = {k: ad.core.mode for k, ad in model.adapters.items()}
    n_total = len(_VAL_LBLS)

    def _run_pass(swap_to_selfavg: bool = False, bypass: bool = False):
        for k, ad in model.adapters.items():
            if bypass:
                ad.set_intervention(0.0, 0.0)
                ad.core.mode = orig_modes[k]
            elif swap_to_selfavg:
                ad.set_intervention(1.0, 1.0)
                ad.core.mode = "self_avg"
            else:
                ad.set_intervention(1.0, 1.0)
                ad.core.mode = orig_modes[k]

        c1, nll, tot = 0, 0.0, 0
        lg_list = []
        for s in range(0, n_total, 512):
            e = min(s + 512, n_total)
            imgs_u8 = _VAL_IMGS[s:e].to(device, non_blocking=True)
            targets = _VAL_LBLS[s:e].to(device, non_blocking=True)
            x = normalize_batch_gpu(imgs_u8, mean_t, std_t)
            if not is_vit:
                x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x).float()
            nll += float(F.cross_entropy(lg, targets, reduction="sum").item())
            c1 += int(lg.argmax(1).eq(targets).sum().item())
            tot += int(targets.size(0))
            lg_list.append(lg.cpu())

        for k, ad in model.adapters.items():
            ad.core.mode = orig_modes[k]
            ad.set_intervention(1.0, 1.0)
        return torch.cat(lg_list, dim=0), round(100.0 * c1 / tot, 3), round(nll / tot, 5)

    lg_o, t1_o, nll_o = _run_pass(False, False)
    lg_s, t1_s, nll_s = _run_pass(True, False)
    lg_b, t1_b, nll_b = _run_pass(False, True)

    p_o = F.softmax(lg_o, dim=-1)
    kl_sa = float(F.kl_div(F.log_softmax(lg_s, dim=-1), p_o, reduction="batchmean").item())
    kl_byp = float(F.kl_div(F.log_softmax(lg_b, dim=-1), p_o, reduction="batchmean").item())
    agree_sa = float((lg_o.argmax(1) == lg_s.argmax(1)).float().mean().item() * 100.0)

    return {
        "regime": regime,
        "run": run_dir.name,
        "mode": args["mode"],
        "orig_top1": t1_o,
        "selfavg_swap_top1": t1_s,
        "bypass_top1": t1_b,
        "selfavg_top1_delta": round(t1_s - t1_o, 3),
        "bypass_top1_delta": round(t1_b - t1_o, 3),
        "selfavg_nll_delta": round(nll_s - nll_o, 5),
        "bypass_nll_delta": round(nll_b - nll_o, 5),
        "selfavg_kl": round(kl_sa, 5),
        "bypass_kl": round(kl_byp, 5),
        "selfavg_agree_pct": round(agree_sa, 2),
    }


def main() -> None:
    global _VAL_IMGS, _VAL_LBLS
    t0 = time.monotonic()
    _VAL_IMGS = torch.from_numpy(np.load("/dev/shm/in1k_raw_cache_ar320/val_images.npy", mmap_mode="r"))
    _VAL_LBLS = torch.from_numpy(np.load("/dev/shm/in1k_raw_cache_ar320/val_labels.npy", mmap_mode="r"))
    root = Path("/scratch/itscspod/chtaihei/exsym_runs")
    raw_tasks = []
    for s in (42, 101, 202):
        for m in ("sym", "free_star"):
            raw_tasks.append(("waveA_ft_convnext", str(root / f"waveA_matched_aa0/ft_s4_dir_{m}_s{s}")))
            raw_tasks.append(("waveD_scratch_convnext", str(root / f"waveD_scratch_100ep/scratch_ms_dir_{m}_s{s}")))
            raw_tasks.append(("waveE_scratch_vittiny", str(root / f"waveE_vit_tiny_scratch/scratch_vittiny_dir_{m}_s{s}")))

    # Dispatch in batches of 8 across GPUs 0..7
    rows: list[dict[str, Any]] = []
    for start in range(0, len(raw_tasks), 8):
        chunk = [(i, raw_tasks[start + i][0], raw_tasks[start + i][1]) for i in range(min(8, len(raw_tasks) - start))]
        with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            rows.extend(list(pool.map(_eval_one_ckpt, chunk)))

    summary: dict[str, Any] = {}
    print("\n=== CROSS-REGIME Sym/FreeStar -> SelfAvg ZERO-RETRAINING SWAP AUDIT (50k Val, 3 seeds) ===")
    print(f"{'Regime':24s} | {'Mode':10s} | {'Orig Top1':>9s} | {'Swap->SA Δ':>10s} | {'Bypass Δ':>9s} | {'SA/Byp Share':>12s} | {'KL(SA)':>8s} | {'Agree%':>7s}")
    for reg in ("waveA_ft_convnext", "waveD_scratch_convnext", "waveE_scratch_vittiny"):
        summary[reg] = {}
        for m in ("sym", "free_star"):
            sub = [r for r in rows if r["regime"] == reg and r["mode"] == m]
            o_m = float(np.mean([r["orig_top1"] for r in sub]))
            sa_d = float(np.mean([r["selfavg_top1_delta"] for r in sub]))
            by_d = float(np.mean([r["bypass_top1_delta"] for r in sub]))
            kl_s = float(np.mean([r["selfavg_kl"] for r in sub]))
            ag_s = float(np.mean([r["selfavg_agree_pct"] for r in sub]))
            cross_loss_share = round(100.0 * sa_d / min(by_d, -1e-6), 1)
            summary[reg][m] = {
                "orig_top1": round(o_m, 2),
                "selfavg_top1_delta": round(sa_d, 2),
                "bypass_top1_delta": round(by_d, 2),
                "cross_token_loss_share_pct": cross_loss_share,
                "selfavg_kl": round(kl_s, 4),
                "selfavg_agree_pct": round(ag_s, 2),
            }
            print(
                f"{reg:24s} | {m:10s} | {o_m:8.2f}% | {sa_d:+9.2f}% | {by_d:+8.2f}% | "
                f"{cross_loss_share:11.1f}% | {kl_s:8.4f} | {ag_s:6.2f}%"
            )

    out_p = Path("/project/itscspod/chtaihei/cliffordnet/exchange-symmetry-vision/results/diag_cross_regime_selfavg_swap.json")
    out_p.write_text(json.dumps({"wall_time_sec": round(time.monotonic() - t0, 1), "summary": summary, "per_run": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved {out_p} in {time.monotonic() - t0:.1f}s!")


if __name__ == "__main__":
    main()
