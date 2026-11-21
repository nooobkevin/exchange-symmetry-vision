"""Vision backbones with exchange-symmetry bilinear adapters and frozen-feature heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import (
    ExchangeBilinearCore,
    GroupedWedgeCore,
    RMSNorm1d,
    SpatialExchangeAdapter,
)


def load_convnext_atto_ema_weights(backbone: nn.Module, ckpt_path: str | Path) -> dict[str, Any]:
    """Load EMA backbone weights from a verified CliffordNet Lightning ConvNeXt-Atto checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    prefix = "_ema_model.backbone."
    matched = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not matched:
        prefix = "_raw_model.backbone."
        matched = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not matched:
        matched = sd

    missing, unexpected = backbone.load_state_dict(matched, strict=True)
    return {"loaded_keys": len(matched), "prefix": prefix, "missing": missing, "unexpected": unexpected}


class ConvNeXtWithAdapter(nn.Module):
    """ConvNeXt-Atto (or Pico) with optional SpatialExchangeAdapter(s) placed inside stages.

    Placement options:
    - "none"       : plain ConvNeXt baseline (no adapter)
    - "stage4_mid" : inserted between Stage 4 block 0 and block 1 (7x7, C=320),
                     so block 1 (DWConv7x7 + LayerNorm + 4x GELU MLP) follows before GAP.
    - "stage3_mid" : inserted between Stage 3 block 2 and block 3 (14x14, C=160).
    - "multi_stage": inserted at both stage3_mid and stage4_mid (for from-scratch studies).
    """

    def __init__(
        self,
        timm_name: str = "convnext_atto",
        num_classes: int = 1000,
        drop_path_rate: float = 0.1,
        adapter_mode: str = "none",
        pair_type: str = "directed_right",
        placement: str = "stage4_mid",
        rank_ratio: float = 1.0,
        gamma_init: float = 0.1,
        pretrained_ckpt: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.timm_name = timm_name
        self.adapter_mode = adapter_mode
        self.pair_type = pair_type
        self.placement = placement if adapter_mode != "none" else "none"

        init_num_classes = 1000 if pretrained_ckpt is not None else num_classes
        self.backbone = timm.create_model(
            timm_name,
            num_classes=init_num_classes,
            drop_path_rate=drop_path_rate,
        )
        if pretrained_ckpt is not None:
            load_convnext_atto_ema_weights(self.backbone, pretrained_ckpt)
            if num_classes != 1000:
                in_f = self.backbone.head.fc.in_features
                self.backbone.head.fc = nn.Linear(in_f, num_classes)

        # Inspect stage channel dimensions
        stage3_dim = self.backbone.stages[2].blocks[0].conv_dw.in_channels
        stage4_dim = self.backbone.stages[3].blocks[0].conv_dw.in_channels

        self.adapters = nn.ModuleDict()
        if adapter_mode != "none":
            if placement in ("stage3_mid", "multi_stage"):
                r3 = max(16, int(round(stage3_dim * rank_ratio)))
                self.adapters["stage3_mid"] = SpatialExchangeAdapter(
                    dim=stage3_dim,
                    rank=r3,
                    mode=adapter_mode,
                    pair_type=pair_type,
                    gamma_init=gamma_init,
                )
            if placement in ("stage4_mid", "multi_stage"):
                r4 = max(16, int(round(stage4_dim * rank_ratio)))
                self.adapters["stage4_mid"] = SpatialExchangeAdapter(
                    dim=stage4_dim,
                    rank=r4,
                    mode=adapter_mode,
                    pair_type=pair_type,
                    gamma_init=gamma_init,
                )

    def set_intervention(self, sym_gain: float = 1.0, anti_gain: float = 1.0) -> None:
        for ad in self.adapters.values():
            ad.set_intervention(sym_gain=sym_gain, anti_gain=anti_gain)

    def set_capture_stats(self, enabled: bool) -> None:
        for ad in self.adapters.values():
            ad._capture_stats = enabled

    def summarize_symmetry(self) -> dict[str, Any]:
        return {name: ad.summarize_symmetry() for name, ad in self.adapters.items()}

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.stem(x)
        x = self.backbone.stages[0](x)
        x = self.backbone.stages[1](x)

        # Stage 3 (index 2)
        s2 = self.backbone.stages[2]
        x = s2.downsample(x)
        if "stage3_mid" in self.adapters:
            mid = len(s2.blocks) // 2
            for b in s2.blocks[:mid]:
                x = b(x)
            x = self.adapters["stage3_mid"](x)
            for b in s2.blocks[mid:]:
                x = b(x)
        else:
            x = s2.blocks(x)

        # Stage 4 (index 3)
        s3 = self.backbone.stages[3]
        x = s3.downsample(x)
        if "stage4_mid" in self.adapters:
            x = s3.blocks[0](x)
            x = self.adapters["stage4_mid"](x)
            for b in s3.blocks[1:]:
                x = b(x)
        else:
            x = s3.blocks(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features_map(x)
        return self.backbone.head(feat)

    @torch.no_grad()
    def extract_stage_map(self, x: torch.Tensor, stage: int = 4) -> torch.Tensor:
        """Extract frozen spatial feature map after stage 3 (14x14, 160) or stage 4 block 0 (7x7, 320)."""
        x = self.backbone.stem(x)
        x = self.backbone.stages[0](x)
        x = self.backbone.stages[1](x)
        if stage == 3:
            return self.backbone.stages[2](x)
        x = self.backbone.stages[2](x)
        s3 = self.backbone.stages[3]
        x = s3.downsample(x)
        x = s3.blocks[0](x)
        return x


class FrozenPairHead(nn.Module):
    """Lightweight classification head on frozen spatial feature maps X in (B, C, H, W).

    Constructs ordered pairs (u, v) at every spatial location, applies either:
    - "linear_u"     : linear readout on GAP(u) only
    - "linear_uv"    : linear readout on GAP(MLP/Linear([u, v])) so reading v is controlled
    - "anti"/"sym"/"star"/"free_anti"/"free_star"/"free_sym"/"grouped_wedge":
      bilinear interaction on (norm(u), norm(v)) -> GELU (breaks odd cancellation under GAP) -> GAP -> Linear!
      Also includes a residual linear skip from GAP(norm(u)) if add_linear_skip=True so all heads
      have a strong linear base and we directly measure what the multiplicative interaction adds!
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        mode: str = "free_anti",
        rank: int = 256,
        pair_type: str = "directed_right",
        add_linear_skip: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.mode = mode
        self.rank = rank
        self.pair_type = pair_type
        self.add_linear_skip = add_linear_skip

        self.norm_u = RMSNorm1d(dim)
        if pair_type == "center_context":
            self.ctx_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
            self.norm_v = RMSNorm1d(dim)
        else:
            self.ctx_dw = None
            self.norm_v = self.norm_u

        if mode == "linear_u":
            self.core = None
            self.head = nn.Linear(dim, num_classes)
        elif mode == "linear_uv":
            self.core = nn.Linear(2 * dim, rank)
            self.act = nn.GELU()
            self.head = nn.Linear(rank + (dim if add_linear_skip else 0), num_classes)
        elif mode == "grouped_wedge":
            self.core = GroupedWedgeCore(dim, rank, subspace_k=4, include_dot=False, learnable_in_proj=True)
            self.act = nn.GELU()
            self.head = nn.Linear(rank + (dim if add_linear_skip else 0), num_classes)
        else:
            self.core = ExchangeBilinearCore(dim, rank, mode=mode, bias=False)
            self.act = nn.GELU()
            self.head = nn.Linear(rank + (dim if add_linear_skip else 0), num_classes)

    def _make_uv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        u_hwc = self.norm_u(x.permute(0, 2, 3, 1))
        if self.pair_type == "directed_right":
            v_raw = F.pad(x[:, :, :, 1:], (0, 1, 0, 0))
            v_hwc = self.norm_v(v_raw.permute(0, 2, 3, 1))
        elif self.pair_type == "center_context":
            v_raw = self.ctx_dw(x)
            v_hwc = self.norm_v(v_raw.permute(0, 2, 3, 1))
        else:
            raise ValueError(self.pair_type)
        return u_hwc, v_hwc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u_hwc, v_hwc = self._make_uv(x)
        u_pool = u_hwc.mean(dim=(1, 2))
        if self.mode == "linear_u":
            return self.head(u_pool)
        if self.mode == "linear_uv":
            h = self.act(self.core(torch.cat([u_hwc, v_hwc], dim=-1))).mean(dim=(1, 2))
        else:
            f = self.core(u_hwc, v_hwc)
            h = self.act(f).mean(dim=(1, 2))
        if self.add_linear_skip:
            return self.head(torch.cat([u_pool, h], dim=-1))
        return self.head(h)


class ViTWithAdapter(nn.Module):
    """Cross-backbone generalization: ViT-Tiny with SpatialExchangeAdapter inserted mid-transformer."""

    def __init__(
        self,
        timm_name: str = "vit_tiny_patch16_224",
        num_classes: int = 1000,
        drop_path_rate: float = 0.1,
        adapter_mode: str = "none",
        pair_type: str = "directed_right",
        insert_after_block: int = 8,
        rank_ratio: float = 1.0,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.timm_name = timm_name
        self.adapter_mode = adapter_mode
        self.insert_after_block = insert_after_block
        self.backbone = timm.create_model(
            timm_name,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
        dim = self.backbone.embed_dim
        self.adapters = nn.ModuleDict()
        if adapter_mode != "none":
            r = max(16, int(round(dim * rank_ratio)))
            self.adapters["mid_vit"] = SpatialExchangeAdapter(
                dim=dim,
                rank=r,
                mode=adapter_mode,
                pair_type=pair_type,
                gamma_init=gamma_init,
            )

    def set_intervention(self, sym_gain: float = 1.0, anti_gain: float = 1.0) -> None:
        for ad in self.adapters.values():
            ad.set_intervention(sym_gain=sym_gain, anti_gain=anti_gain)

    def set_capture_stats(self, enabled: bool) -> None:
        for ad in self.adapters.values():
            ad._capture_stats = enabled

    def summarize_symmetry(self) -> dict[str, Any]:
        return {name: ad.summarize_symmetry() for name, ad in self.adapters.items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)

        for idx, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if idx == self.insert_after_block and "mid_vit" in self.adapters:
                # Separate CLS token (if present) and apply spatial 2D adapter on 14x14 patch grid
                n_prefix = self.backbone.num_prefix_tokens
                prefix, patches = x[:, :n_prefix, :], x[:, n_prefix:, :]
                B, N, C = patches.shape
                H = W = int(round(N ** 0.5))
                p_2d = patches.transpose(1, 2).reshape(B, C, H, W)
                p_2d = self.adapters["mid_vit"](p_2d)
                patches = p_2d.flatten(2).transpose(1, 2)
                x = torch.cat([prefix, patches], dim=1)

        x = self.backbone.norm(x)
        return self.backbone.forward_head(x)
