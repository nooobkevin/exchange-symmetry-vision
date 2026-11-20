"""Rigorous mathematical and gradient unit tests for ExchangeBilinearAdapter."""

from __future__ import annotations

import copy
import math
import torch

from exsym.adapter import ExchangeBilinearCore, GroupedWedgeCore, SpatialExchangeAdapter
from exsym.models import ConvNeXtWithAdapter


def test_exact_exchange_symmetry_core() -> None:
    torch.manual_seed(0)
    u = torch.randn(4, 7, 7, 64)
    v = torch.randn(4, 7, 7, 64)

    anti = ExchangeBilinearCore(64, 64, mode="anti")
    assert torch.allclose(anti(u, v), -anti(v, u), atol=1e-6), "Anti must flip sign on swap!"
    assert torch.allclose(anti(u, u), torch.zeros_like(anti(u, u)), atol=1e-6), "Anti(u, u) must vanish!"

    sym = ExchangeBilinearCore(64, 64, mode="sym")
    assert torch.allclose(sym(u, v), sym(v, u), atol=1e-6), "Sym must be invariant on swap!"


def test_step0_equivalence_and_nonzero_theta_grad() -> None:
    """Verify Anti == Free-from-Anti (and Star == Free-from-Star, Sym == Free-from-Sym)
    at step 0 for BOTH forward output AND all parameter/input gradients, while dL/dtheta != 0!
    """
    for fixed_mode, free_mode in [("anti", "free_anti"), ("star", "free_star"), ("sym", "free_sym")]:
        torch.manual_seed(42)
        m_fixed = SpatialExchangeAdapter(64, rank=64, mode=fixed_mode, pair_type="directed_right", gamma_init=0.1)
        m_free = SpatialExchangeAdapter(64, rank=64, mode=free_mode, pair_type="directed_right", gamma_init=0.1)
        # Copy exact weights
        with torch.no_grad():
            m_free.norm_u.weight.copy_(m_fixed.norm_u.weight)
            m_free.core.proj_a.weight.copy_(m_fixed.core.proj_a.weight)
            m_free.core.proj_b.weight.copy_(m_fixed.core.proj_b.weight)
            m_free.out_proj.weight.copy_(m_fixed.out_proj.weight)
            m_free.gamma.copy_(m_fixed.gamma)

        x1 = torch.randn(2, 64, 7, 7, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        target = torch.randn(2, 64, 7, 7)

        y1 = m_fixed(x1)
        y2 = m_free(x2)
        assert torch.allclose(y1, y2, atol=1e-6), f"Forward mismatch {fixed_mode} vs {free_mode}"

        loss1 = (y1 - target).pow(2).mean()
        loss2 = (y2 - target).pow(2).mean()
        loss1.backward()
        loss2.backward()

        assert torch.allclose(x1.grad, x2.grad, atol=1e-6), f"Input grad mismatch {fixed_mode} vs {free_mode}"
        assert torch.allclose(m_fixed.core.proj_a.weight.grad, m_free.core.proj_a.weight.grad, atol=1e-6)
        assert torch.allclose(m_fixed.out_proj.weight.grad, m_free.out_proj.weight.grad, atol=1e-6)

        theta_grad = m_free.core.theta.grad
        assert theta_grad is not None and theta_grad.abs().mean().item() > 1e-6, (
            f"theta grad must be nonzero at step 0 for {free_mode}!"
        )


def test_exact_adapter_swap_through_wout() -> None:
    """Because SpatialExchangeAdapter has no internal activation, delta(v, u) == -delta(u, v)
    holds strictly AFTER W_out and gamma!
    """
    torch.manual_seed(7)
    ad_anti = SpatialExchangeAdapter(64, rank=64, mode="anti", pair_type="directed_right")
    ad_sym = SpatialExchangeAdapter(64, rank=64, mode="sym", pair_type="directed_right")

    u = torch.randn(2, 64, 7, 7)
    v = torch.randn(2, 64, 7, 7)
    assert torch.allclose(ad_anti.forward_delta_from_pair(u, v), -ad_anti.forward_delta_from_pair(v, u), atol=1e-6)
    assert torch.allclose(ad_sym.forward_delta_from_pair(u, v), ad_sym.forward_delta_from_pair(v, u), atol=1e-6)


def test_exact_decomposition_intervention() -> None:
    """Verify f(sym_gain=1, anti_gain=1) == f_sym + f_anti == unperturbed forward for any theta."""
    torch.manual_seed(13)
    core = ExchangeBilinearCore(48, 32, mode="free_star")
    with torch.no_grad():
        core.theta.uniform_(-math.pi, math.pi)
    u = torch.randn(3, 5, 48)
    v = torch.randn(3, 5, 48)

    core.set_intervention(1.0, 1.0)
    f_full = core(u, v)

    core.set_intervention(1.0, 0.0)
    f_sym = core(u, v)
    assert torch.allclose(f_sym, core(v, u), atol=1e-6), "Sym component must be even!"

    core.set_intervention(0.0, 1.0)
    f_anti = core(u, v)
    assert torch.allclose(f_anti, -core(v, u), atol=1e-6), "Anti component must be odd!"

    assert torch.allclose(f_full, f_sym + f_anti, atol=1e-6), "Decomposition must sum identically to full!"


def test_non_cancellation_in_convnext_stage4() -> None:
    """Verify Anti adapter in ConvNeXt stage4_mid produces non-trivial residual RMS and logit change."""
    torch.manual_seed(99)
    m_base = ConvNeXtWithAdapter("convnext_atto", num_classes=10, adapter_mode="none")
    m_anti = ConvNeXtWithAdapter("convnext_atto", num_classes=10, adapter_mode="anti", placement="stage4_mid")
    m_anti.backbone.load_state_dict(copy.deepcopy(m_base.backbone.state_dict()))
    m_base.eval()
    m_anti.eval()
    m_anti.set_capture_stats(True)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_base = m_base(x)
        out_anti = m_anti(x)
    diff = (out_anti - out_base).abs().mean().item()
    stats = m_anti.summarize_symmetry()["stage4_mid"]
    assert diff > 1e-4, f"Anti adapter output should not cancel before classifier (diff={diff})"
    assert stats["rms_ratio"] > 1e-3, f"Adapter residual RMS too small: {stats}"
    assert abs(stats["eff_anti"] - 1.0) < 1e-5, f"Anti eff_anti should be 1.0, got {stats['eff_anti']}"


def test_grouped_wedge_runs() -> None:
    torch.manual_seed(5)
    gw = GroupedWedgeCore(64, 64, subspace_k=4, include_dot=False, learnable_in_proj=True)
    u = torch.randn(2, 7, 7, 64)
    v = torch.randn(2, 7, 7, 64)
    out = gw(u, v)
    assert out.shape == (2, 7, 7, 64)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    test_exact_exchange_symmetry_core()
    test_step0_equivalence_and_nonzero_theta_grad()
    test_exact_adapter_swap_through_wout()
    test_exact_decomposition_intervention()
    test_non_cancellation_in_convnext_stage4()
    test_grouped_wedge_runs()
    print("ALL 6 EXCHANGE-SYMMETRY UNIT TESTS PASSED!")
