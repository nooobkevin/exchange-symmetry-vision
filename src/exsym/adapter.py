"""Core exchange-symmetry bilinear modules and spatial residual adapters.

For an ordered pair of feature vectors (u, v) in R^C, shared projections A, B: R^C -> R^R
compute the swapped Hadamard products:
    p = (A u) * (B v)
    q = (A v) * (B u)

The exchange-symmetry family parameterizes each interaction channel r by an angle theta_r:
    f_r(u, v) = cos(theta_r) * p_r + sin(theta_r) * q_r
              = alpha_sym_r * s_r + alpha_anti_r * a_r
where:
    s = (p + q) / sqrt(2)   (exchange-symmetric:    s(v, u) = +s(u, v))
    a = (p - q) / sqrt(2)   (exchange-antisymmetric: a(v, u) = -a(u, v))
    alpha_sym  = (cos(theta) + sin(theta)) / sqrt(2) = cos(theta - pi/4)
    alpha_anti = (cos(theta) - sin(theta)) / sqrt(2) = cos(theta + pi/4)
    E_anti = alpha_anti^2 = (1 - sin(2 * theta)) / 2
    E_sym  = alpha_sym^2  = (1 + sin(2 * theta)) / 2  (E_sym + E_anti == 1)

Special cases:
    - anti      : theta = -pi/4 fixed  -> f = (p - q) / sqrt(2)   (E_anti = 1.0)
    - sym       : theta = +pi/4 fixed  -> f = (p + q) / sqrt(2)   (E_anti = 0.0)
    - star      : theta = 0     fixed  -> f = p = (A u) * (B v)   (E_anti = 0.5)
    - free_anti : theta learnable, initialized at -pi/4 (step-0 identical to anti)
    - free_star : theta learnable, initialized at 0     (step-0 identical to star)
    - free_sym  : theta learnable, initialized at +pi/4 (step-0 identical to sym)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT2_INV = 1.0 / math.sqrt(2.0)

VALID_MODES = (
    "anti",
    "sym",
    "star",
    "self_avg",
    "free_anti",
    "free_star",
    "free_sym",
    "grouped_wedge",
)

MODE_INIT_THETA: dict[str, float] = {
    "anti": -math.pi / 4.0,
    "sym": math.pi / 4.0,
    "star": 0.0,
    "self_avg": math.pi / 4.0,
    "free_anti": -math.pi / 4.0,
    "free_star": 0.0,
    "free_sym": math.pi / 4.0,
}


class RMSNorm2d(nn.Module):
    """Channel-wise RMSNorm for 2D (B, C, H, W) feature maps, computed in fp32 for stability."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(dim=1, keepdim=True) + self.eps)
        out = (x_f * rms).to(x.dtype)
        return out * self.weight.view(1, -1, 1, 1)


class RMSNorm1d(nn.Module):
    """Last-dimension RMSNorm for (..., C) tensors, computed in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight


class ExchangeBilinearCore(nn.Module):
    """Vector-pair bilinear interaction with fixed or learnable exchange symmetry.

    Operates on tensors u, v of shape (..., C) and produces (..., R).
    Supports exact inference-time intervention via sym_gain and anti_gain.
    """

    def __init__(
        self,
        dim: int,
        rank: int,
        mode: str = "free_anti",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if mode not in MODE_INIT_THETA:
            raise ValueError(f"Unsupported ExchangeBilinearCore mode: {mode!r}")
        self.dim = dim
        self.rank = rank
        self.mode = mode
        self.learnable_theta = mode.startswith("free_")

        self.proj_a = nn.Linear(dim, rank, bias=bias)
        self.proj_b = nn.Linear(dim, rank, bias=bias)

        init_val = MODE_INIT_THETA[mode]
        theta_tensor = torch.full((rank,), init_val, dtype=torch.float32)
        if self.learnable_theta:
            self.theta = nn.Parameter(theta_tensor)
        else:
            self.register_buffer("theta", theta_tensor)

        # Inference-time intervention gains (default 1.0 = exact unperturbed forward)
        self.sym_gain: float = 1.0
        self.anti_gain: float = 1.0

    def set_intervention(self, sym_gain: float = 1.0, anti_gain: float = 1.0) -> None:
        self.sym_gain = float(sym_gain)
        self.anti_gain = float(anti_gain)

    def get_theta(self) -> torch.Tensor:
        return self.theta

    def get_energy_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (E_sym, E_anti) per channel in [0, 1] with E_sym + E_anti == 1."""
        sin2 = torch.sin(2.0 * self.theta.float())
        e_sym = 0.5 * (1.0 + sin2)
        e_anti = 0.5 * (1.0 - sin2)
        return e_sym, e_anti

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.mode == "self_avg":
            # Same-token quadratic features averaged across both tokens: ((Au)*(Bu) + (Av)*(Bv)) / sqrt(2)
            # Exact algebraic relation: SelfAvg(u,v) - Sym(u,v) == (A(u-v) * B(u-v)) / sqrt(2)
            hu = self.proj_a(u) * self.proj_b(u)
            hv = self.proj_a(v) * self.proj_b(v)
            out_f = (hu.float() + hv.float()) * SQRT2_INV
            if self.sym_gain != 1.0:
                out_f = out_f * self.sym_gain
            return out_f.to(u.dtype)

        au = self.proj_a(u)
        bv = self.proj_b(v)
        p = au * bv

        # Fast path when exact Star with no intervention
        if self.mode == "star" and self.sym_gain == 1.0 and self.anti_gain == 1.0:
            return p

        av = self.proj_a(v)
        bu = self.proj_b(u)
        q = av * bu

        # Compute symmetric/antisymmetric combination in fp32 to avoid bf16 cancellation
        p_f = p.float()
        q_f = q.float()
        th = self.theta.float()
        c = torch.cos(th)
        s = torch.sin(th)

        if self.sym_gain == 1.0 and self.anti_gain == 1.0:
            out_f = c * p_f + s * q_f
        else:
            sym_part = (p_f + q_f) * SQRT2_INV
            anti_part = (p_f - q_f) * SQRT2_INV
            alpha_sym = (c + s) * SQRT2_INV
            alpha_anti = (c - s) * SQRT2_INV
            out_f = (self.sym_gain * alpha_sym) * sym_part + (self.anti_gain * alpha_anti) * anti_part

        return out_f.to(u.dtype)


class GroupedWedgeCore(nn.Module):
    """Clifford-style grouped k=4 subspace wedge product (with optional input linear proj).

    Maps (u, v) of shape (..., C) -> grouped k=4 wedge bivectors (6 per 4 channels) +
    optional dot terms, then linearly maps to (..., R) to match downstream readout shape.
    Useful for testing coordinate-alignment dependence against rotation-invariant Anti.
    """

    def __init__(
        self,
        dim: int,
        rank: int,
        subspace_k: int = 4,
        include_dot: bool = False,
        learnable_in_proj: bool = True,
    ) -> None:
        super().__init__()
        if dim % subspace_k != 0:
            raise ValueError(f"dim={dim} must be divisible by subspace_k={subspace_k}")
        self.dim = dim
        self.rank = rank
        self.subspace_k = subspace_k
        self.include_dot = include_dot
        self.learnable_in_proj = learnable_in_proj

        # Precompute upper-triangular pair indices (i < j) inside each k-dim group
        idx_i, idx_j = [], []
        for i in range(subspace_k):
            for j in range(i + 1, subspace_k):
                idx_i.append(i)
                idx_j.append(j)
        self.register_buffer("_idx_i", torch.tensor(idx_i, dtype=torch.long))
        self.register_buffer("_idx_j", torch.tensor(idx_j, dtype=torch.long))

        n_groups = dim // subspace_k
        n_biv = len(idx_i)
        raw_out = n_groups * (n_biv + (subspace_k if include_dot else 0))

        # Parameter-matched input projection if requested
        if learnable_in_proj:
            self.in_proj_u = nn.Linear(dim, dim, bias=False)
            self.in_proj_v = nn.Linear(dim, dim, bias=False)
        else:
            self.in_proj_u = None
            self.in_proj_v = None

        self.to_rank = nn.Linear(raw_out, rank, bias=False)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.in_proj_u is not None:
            u = self.in_proj_u(u)
            v = self.in_proj_v(v)
        orig_shape = u.shape[:-1]
        G = self.dim // self.subspace_k
        K = self.subspace_k
        ug = u.reshape(*orig_shape, G, K).float()
        vg = v.reshape(*orig_shape, G, K).float()

        ui = ug[..., self._idx_i]
        uj = ug[..., self._idx_j]
        vi = vg[..., self._idx_i]
        vj = vg[..., self._idx_j]
        wedge = (ui * vj - uj * vi).reshape(*orig_shape, -1)

        if self.include_dot:
            dot = (ug * vg).reshape(*orig_shape, -1)
            feats = torch.cat([dot, wedge], dim=-1)
        else:
            feats = wedge
        return self.to_rank(feats.to(u.dtype))


class SpatialExchangeAdapter(nn.Module):
    """Pure bilinear residual adapter inserted into a 2D vision backbone stage.

    Design invariants:
    1. No pointwise nonlinearity inside the adapter (pure bilinear residual):
       - Eliminates SiLU dead-zone confounds completely.
       - Preserves exact exchange symmetry/antisymmetry all the way through out_proj:
         delta(v, u) == -delta(u, v) for Anti, +delta(u, v) for Sym.
    2. Shared RMSNorm2d on u and v when pair_type == "directed_right":
       - Guarantees exact exchange symmetry of (norm(u), norm(v)).
       - Prevents bf16 quadratic blow-up (no BatchNorm variance spikes).
    3. Non-circular single-direction spatial shift for "directed_right":
       - v[:, :, :, :-1] = x[:, :, :, 1:], v[:, :, :, -1] = 0.
       - Never cancels (u, v) with (v, u) inside the same pair construction.
    4. Small nonzero out_proj init + LayerScale gamma (default 0.1):
       - Starts near identity (small perturbation to pretrained backbone),
         while keeping d(loss)/d(theta) strictly nonzero from step 0!
    """

    def __init__(
        self,
        dim: int,
        rank: int | None = None,
        mode: str = "free_anti",
        pair_type: str = "directed_right",
        gamma_init: float = 0.1,
        out_init_std: float = 0.02,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown adapter mode: {mode!r}; expected one of {VALID_MODES}")
        if pair_type not in ("directed_right", "directed_down", "center_context"):
            raise ValueError(f"Unknown pair_type: {pair_type!r}")

        self.dim = dim
        self.rank = rank if rank is not None else dim
        self.mode = mode
        self.pair_type = pair_type

        self.norm_u = RMSNorm2d(dim)
        if pair_type == "center_context":
            self.ctx_dw = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
            self.norm_v = RMSNorm2d(dim)
        else:
            self.ctx_dw = None
            # Share normalization parameters so swapping (u, v) is exact
            self.norm_v = self.norm_u

        if mode == "grouped_wedge":
            self.core: nn.Module = GroupedWedgeCore(dim, self.rank, subspace_k=4, include_dot=False, learnable_in_proj=True)
        else:
            self.core = ExchangeBilinearCore(dim, self.rank, mode=mode, bias=False)

        self.out_proj = nn.Linear(self.rank, dim, bias=False)
        nn.init.trunc_normal_(self.out_proj.weight, std=out_init_std)
        # 1D parameter avoids PyTorch Inductor channels_last backward shape mismatch on [1, C, 1, 1]
        self.gamma = nn.Parameter(torch.full((dim,), gamma_init))

        if drop_path > 0.0:
            from timm.layers import DropPath
            self.drop_path: nn.Module = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        # Optional lightweight telemetry capture (disabled by default during torch.compile)
        self._capture_stats: bool = False
        self._last_stats: dict[str, Any] = {}

    def set_intervention(self, sym_gain: float = 1.0, anti_gain: float = 1.0) -> None:
        if isinstance(self.core, ExchangeBilinearCore):
            self.core.set_intervention(sym_gain=sym_gain, anti_gain=anti_gain)

    def _make_pair(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pair_type == "directed_right":
            u = self.norm_u(x)
            # Non-circular right neighbor shift (last column zero-padded)
            v_raw = F.pad(x[:, :, :, 1:], (0, 1, 0, 0))
            v = self.norm_v(v_raw)
            return u, v
        elif self.pair_type == "directed_down":
            u = self.norm_u(x)
            v_raw = F.pad(x[:, :, 1:, :], (0, 0, 0, 1))
            v = self.norm_v(v_raw)
            return u, v
        else:  # center_context
            u = self.norm_u(x)
            v = self.norm_v(self.ctx_dw(x))
            return u, v

    def forward_delta_from_pair(self, u_nchw: torch.Tensor, v_nchw: torch.Tensor) -> torch.Tensor:
        """Compute residual update delta(u, v) from already-normalized NCHW pairs."""
        # Channels-last linear ops: (B, C, H, W) -> (B, H, W, C)
        u_hwc = u_nchw.permute(0, 2, 3, 1)
        v_hwc = v_nchw.permute(0, 2, 3, 1)
        f = self.core(u_hwc, v_hwc)
        out_hwc = self.out_proj(f)
        gamma_4d = self.gamma.view(1, -1, 1, 1)
        delta = out_hwc.permute(0, 3, 1, 2) * gamma_4d
        if self._capture_stats:
            with torch.no_grad():
                f_f = f.float()
                f_var = f_f.var(dim=(0, 1, 2), unbiased=False)  # (R,)
                w_col_sq = (self.out_proj.weight.float() * self.gamma.float().view(-1, 1)).pow(2).sum(dim=0)  # (R,)
                channel_energy = w_col_sq * f_var
                # Exact post-W_out functional antisymmetry via swapped input pair (v, u):
                # delta_sym = 0.5*(delta(u,v) + delta(v,u)), delta_anti = 0.5*(delta(u,v) - delta(v,u))
                f_swapped = self.core(v_hwc, u_hwc)
                delta_swapped = self.out_proj(f_swapped).permute(0, 3, 1, 2) * gamma_4d
                d_f = delta.float()
                ds_f = delta_swapped.float()
                e_sym_func = float((0.5 * (d_f + ds_f)).pow(2).mean().item())
                e_anti_func = float((0.5 * (d_f - ds_f)).pow(2).mean().item())
                func_anti = e_anti_func / max(e_sym_func + e_anti_func, 1e-12)
                self._last_stats = {
                    "f_var": f_var.detach().cpu(),
                    "channel_energy": channel_energy.detach().cpu(),
                    "delta_rms": float(d_f.pow(2).mean().sqrt().item()),
                    "func_anti": func_anti,
                }
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self._make_pair(x)
        delta = self.forward_delta_from_pair(u, v)
        if self._capture_stats:
            with torch.no_grad():
                x_rms = float(x.float().pow(2).mean().sqrt().item())
                self._last_stats["x_rms"] = x_rms
                self._last_stats["rms_ratio"] = self._last_stats["delta_rms"] / max(x_rms, 1e-8)
        return x + self.drop_path(delta)

    @torch.no_grad()
    def summarize_symmetry(self) -> dict[str, Any]:
        """Return unweighted and readout-energy-weighted antisymmetry statistics."""
        info: dict[str, Any] = {"mode": self.mode, "pair_type": self.pair_type, "dim": self.dim, "rank": self.rank}
        if isinstance(self.core, ExchangeBilinearCore):
            th = self.core.theta.detach().float().cpu()
            e_sym, e_anti = self.core.get_energy_split()
            e_sym = e_sym.detach().cpu()
            e_anti = e_anti.detach().cpu()
            info["mean_e_anti"] = float(e_anti.mean().item())
            info["std_e_anti"] = float(e_anti.std().item())
            info["theta_mean"] = float(th.mean().item())
            info["theta_std"] = float(th.std().item())
            info["theta_quantiles"] = [float(x) for x in torch.quantile(th, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])).tolist()]
            # Weight by ||gamma * W_out[:, r]||_2^2 (always available) and by variance if captured
            w_col_sq = (self.out_proj.weight.detach().float().cpu() * self.gamma.detach().float().cpu().view(-1, 1)).pow(2).sum(dim=0)
            info["weight_weighted_e_anti"] = float((w_col_sq * e_anti).sum().item() / max(w_col_sq.sum().item(), 1e-12))
            if "channel_energy" in self._last_stats:
                ce = self._last_stats["channel_energy"]
                info["eff_anti"] = float((ce * e_anti).sum().item() / max(ce.sum().item(), 1e-12))
                info["func_anti"] = float(self._last_stats.get("func_anti", 0.0))
                info["delta_rms"] = self._last_stats.get("delta_rms", 0.0)
                info["x_rms"] = self._last_stats.get("x_rms", 0.0)
                info["rms_ratio"] = self._last_stats.get("rms_ratio", 0.0)
                # Fraction of readout energy carried by top-50% channels
                ce_sorted = torch.sort(ce, descending=True).values
                info["top25_energy_frac"] = float(ce_sorted[: max(1, len(ce_sorted) // 4)].sum().item() / max(ce.sum().item(), 1e-12))
        else:
            if "delta_rms" in self._last_stats:
                info["func_anti"] = float(self._last_stats.get("func_anti", 0.0))
                info["delta_rms"] = self._last_stats.get("delta_rms", 0.0)
                info["x_rms"] = self._last_stats.get("x_rms", 0.0)
                info["rms_ratio"] = self._last_stats.get("rms_ratio", 0.0)
        return info
