"""Unit tests for the analytical 2D HO quench reference (T1.2 prototype).

The arithmetic in :mod:`realtime_quench_ho` is short and the test surface is
correspondingly small: we verify the analytical formulas against
self-consistency checks and known textbook limits, plus a residual-vanishing
check on the *exact* solution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch.set_default_dtype(torch.float64)

from realtime_pinn import compute_realtime_residual  # noqa: E402
from realtime_quench_ho import (  # noqa: E402
    HOQuenchConfig,
    analytical_breathing_period,
    analytical_x2_aggregate,
    analytical_x2_per_electron,
    build_ho_quench_pool,
)


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


def test_pool_shapes_and_e_l0_constant() -> None:
    cfg = HOQuenchConfig(n_particles=2, dim=2, omega_0=1.0, omega_1=2.0)
    pool = build_ho_quench_pool(cfg, n_samples=128, seed=0)
    assert pool["x"].shape == (128, 2, 2)
    assert pool["E_L0"].shape == (128,)
    assert pool["grad_log_psi0"].shape == (128, 2, 2)
    assert pool["deltaV"].shape == (128,)
    # E_L^{(0)}(x) ≡ N·ω₀ = 2·1 = 2 for ψ₀ being the exact GS of H_0.
    assert torch.allclose(pool["E_L0"], torch.full_like(pool["E_L0"], 2.0), atol=1e-12)
    # ∇log ψ_0 = -ω_0 x (component-wise).
    expected = -1.0 * pool["x"]
    assert torch.allclose(pool["grad_log_psi0"], expected, atol=1e-12)
    # ΔV(x) = ½(ω_1²-ω_0²) Σ_i |x_i|² = 1.5 r²
    r2 = (pool["x"] ** 2).sum(dim=(1, 2))
    assert torch.allclose(pool["deltaV"], 1.5 * r2, atol=1e-12)


def test_pool_sample_density_matches_psi0() -> None:
    """Samples come from |ψ₀|² ⇒ second moment ⟨|x|²⟩ ≈ N · 1/ω₀."""
    cfg = HOQuenchConfig(n_particles=2, dim=2, omega_0=1.5, omega_1=3.0)
    pool = build_ho_quench_pool(cfg, n_samples=20_000, seed=42)
    r2_mean = (pool["x"] ** 2).sum(dim=(1, 2)).mean().item()
    expected = cfg.n_particles * (1.0 / cfg.omega_0)
    assert abs(r2_mean - expected) < 0.05  # ≈3% MC noise at 20k samples
    # And per-axis ⟨x²⟩ ≈ 1/(2ω_0) ≈ 1/3 for ω₀=1.5.
    x2_axis_mean = pool["x"][..., 0].pow(2).mean().item()
    assert abs(x2_axis_mean - 1.0 / (2 * cfg.omega_0)) < 0.02


# ---------------------------------------------------------------------------
# Analytical formulas
# ---------------------------------------------------------------------------


def test_analytical_x2_at_t0_returns_psi0_value() -> None:
    """At t=0, ⟨|x|²⟩_{1e}(0) = 1/ω₀ for 2D HO GS."""
    for omega_0 in (0.5, 1.0, 1.5, 2.0):
        cfg = HOQuenchConfig(omega_0=omega_0, omega_1=omega_0 + 0.5)
        v = analytical_x2_per_electron(cfg, 0.0)
        assert abs(v - 1.0 / omega_0) < 1e-12


def test_analytical_no_quench_constant_in_time() -> None:
    """ω_0 = ω_1 ⇒ no actual quench ⇒ ⟨|x|²⟩(t) constant in t."""
    cfg = HOQuenchConfig(omega_0=1.3, omega_1=1.3)
    ts = np.linspace(0, 5.0, 50)
    vals = analytical_x2_per_electron(cfg, ts)
    expected = 1.0 / cfg.omega_0
    assert np.allclose(vals, expected, atol=1e-12)


def test_analytical_period_predictability() -> None:
    """⟨|x|²⟩(t + T) = ⟨|x|²⟩(t) for T = π/ω_1."""
    cfg = HOQuenchConfig(omega_0=1.0, omega_1=2.5)
    T = analytical_breathing_period(cfg)
    assert abs(T - np.pi / cfg.omega_1) < 1e-15
    ts = np.linspace(0, 1.0, 17)
    a = analytical_x2_per_electron(cfg, ts)
    b = analytical_x2_per_electron(cfg, ts + T)
    assert np.allclose(a, b, atol=1e-12)


def test_analytical_x2_at_quarter_period_matches_textbook() -> None:
    """At t = π/(2 ω_1) (the sin² = 1, cos² = 0 quarter-period extreme),
    ``⟨|x|²⟩_{1e} = ω₀ / ω₁²``.

    For ω_0=1, ω_1=2 ⇒ value = 1/4 = 0.25 (per electron, 2D).
    """
    cfg = HOQuenchConfig(omega_0=1.0, omega_1=2.0)
    t_quarter = np.pi / (2.0 * cfg.omega_1)
    v = analytical_x2_per_electron(cfg, t_quarter)
    expected = cfg.omega_0 / cfg.omega_1**2
    assert abs(v - expected) < 1e-12
    # Check the N-particle aggregate scales with N.
    cfg2 = HOQuenchConfig(n_particles=2, omega_0=1.0, omega_1=2.0)
    v_agg = analytical_x2_aggregate(cfg2, t_quarter)
    assert abs(v_agg - 2.0 * expected) < 1e-12


# ---------------------------------------------------------------------------
# Residual identity at t=0 in the quench setting
# ---------------------------------------------------------------------------


def test_residual_at_t0_returns_E0_real_and_zero_imag_with_deltaV() -> None:
    """At t=0, hard-IC g(x,0)=0 ⇒ Re(E_L) = E_L^{(0)} + ΔV; Im(E_L) = 0.

    This verifies that the quench-pool ΔV plumbs into the residual correctly.
    """
    from realtime_pinn import RealTimeNet

    cfg = HOQuenchConfig(n_particles=2, dim=2, omega_0=1.0, omega_1=2.0)
    pool = build_ho_quench_pool(cfg, n_samples=32, seed=7)
    net = RealTimeNet(n_particles=2, dim=2, hidden=16, n_layers=2,
                      t_embed=8, n_freq=2)
    t = torch.zeros(32, dtype=torch.float64)
    res = compute_realtime_residual(
        net, pool["x"], t,
        E_L0=pool["E_L0"],
        grad_log_psi0=pool["grad_log_psi0"],
        deltaV=pool["deltaV"],
    )
    expected_E_L_real = pool["E_L0"] + pool["deltaV"]
    assert torch.allclose(res.E_L_real, expected_E_L_real, atol=1e-12)
    assert torch.allclose(res.E_L_imag, torch.zeros(32, dtype=torch.float64), atol=1e-12)
