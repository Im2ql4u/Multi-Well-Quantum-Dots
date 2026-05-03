"""Unit tests for the HO+softcore-Coulomb quench reference (T1.2b).

These tests verify the *bookkeeping* of :mod:`realtime_quench_coulomb`:
shapes, the softcore pair-potential identity, the Coulomb-off limit
matching the existing :mod:`realtime_quench_ho` reference, and the
expected sign / magnitude of the pair interaction at ``t=0``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch.set_default_dtype(torch.float64)

from realtime_quench_coulomb import (  # noqa: E402
    CoulombQuenchConfig,
    analytical_breathing_period_coulomb,
    analytical_x2_aggregate_ohne_coulomb,
    build_coulomb_quench_pool,
    coulomb_quench_to_ho_quench,
    initial_pair_interaction_value,
    softcore_coulomb_pair_potential,
)
from realtime_quench_ho import (  # noqa: E402
    HOQuenchConfig,
    analytical_breathing_period,
    analytical_x2_aggregate,
    build_ho_quench_pool,
)


# ---------------------------------------------------------------------------
# Softcore pair potential
# ---------------------------------------------------------------------------


def test_softcore_pair_potential_two_particles_matches_closed_form() -> None:
    """For N=2, ``∑_{i<j} 1/√(r²+ε²)`` is just one term: 1/√(|r₁-r₂|² + ε²)."""
    x = torch.tensor(
        [
            [[0.3, 0.0], [0.7, 0.0]],   # |r₁-r₂|=0.4
            [[0.0, 0.0], [0.0, 1.0]],   # |r₁-r₂|=1.0
        ],
        dtype=torch.float64,
    )
    eps = 0.05
    v = softcore_coulomb_pair_potential(x, epsilon=eps)
    expected = torch.tensor(
        [
            1.0 / float(np.sqrt(0.4**2 + eps**2)),
            1.0 / float(np.sqrt(1.0**2 + eps**2)),
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(v, expected, atol=1e-12)


def test_softcore_pair_potential_three_particles_sum() -> None:
    """For N=3, ``∑_{i<j} 1/√(r_ij²+ε²)`` has 3 terms."""
    x = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float64,
    )
    eps = 0.1
    v = softcore_coulomb_pair_potential(x, epsilon=eps)
    r12 = 1.0
    r13 = 1.0
    r23 = float(np.sqrt(2.0))
    expected = (
        1.0 / float(np.sqrt(r12**2 + eps**2))
        + 1.0 / float(np.sqrt(r13**2 + eps**2))
        + 1.0 / float(np.sqrt(r23**2 + eps**2))
    )
    assert abs(v.item() - expected) < 1e-12


def test_softcore_pair_potential_no_singularity_at_origin() -> None:
    """At zero separation, the softcore caps at ``1/ε``, not infinity."""
    x = torch.zeros(1, 2, 2, dtype=torch.float64)
    v = softcore_coulomb_pair_potential(x, epsilon=0.07)
    assert abs(v.item() - 1.0 / 0.07) < 1e-12


def test_softcore_pair_potential_one_particle_returns_zero() -> None:
    """``N=1`` has no pairs ⇒ pair sum is identically zero."""
    x = torch.randn(5, 1, 2, dtype=torch.float64)
    v = softcore_coulomb_pair_potential(x, epsilon=0.05)
    assert torch.allclose(v, torch.zeros(5, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


def test_pool_shapes_and_keys() -> None:
    cfg = CoulombQuenchConfig(omega_0=1.0, omega_1=2.0, lambda_coul=0.4, epsilon_coul=0.1)
    pool = build_coulomb_quench_pool(cfg, n_samples=64, seed=0)
    for key in (
        "x",
        "E_L0",
        "grad_log_psi0",
        "deltaV",
        "deltaV_omega_piece",
        "deltaV_coulomb_piece",
        "E_0",
        "sigma",
    ):
        assert key in pool, f"missing key {key}"
    assert pool["x"].shape == (64, 2, 2)
    assert pool["E_L0"].shape == (64,)
    assert pool["grad_log_psi0"].shape == (64, 2, 2)
    assert pool["deltaV"].shape == (64,)
    assert pool["deltaV_omega_piece"].shape == (64,)
    assert pool["deltaV_coulomb_piece"].shape == (64,)


def test_pool_deltaV_is_omega_plus_coulomb_pieces() -> None:
    """``ΔV`` must equal the sum of the ω-piece and the Coulomb piece."""
    cfg = CoulombQuenchConfig(omega_0=1.2, omega_1=2.5, lambda_coul=0.7, epsilon_coul=0.08)
    pool = build_coulomb_quench_pool(cfg, n_samples=32, seed=11)
    expected = pool["deltaV_omega_piece"] + pool["deltaV_coulomb_piece"]
    assert torch.allclose(pool["deltaV"], expected, atol=1e-12)


def test_pool_omega_piece_matches_ho_pool_when_lambda_zero() -> None:
    """With ``λ=0`` the Coulomb piece vanishes and ``ΔV`` reduces to the ω-piece,
    which must agree with :func:`realtime_quench_ho.build_ho_quench_pool`."""
    cfg_c = CoulombQuenchConfig(omega_0=1.0, omega_1=2.0, lambda_coul=0.0, epsilon_coul=0.1)
    pool_c = build_coulomb_quench_pool(cfg_c, n_samples=64, seed=0)
    pool_h = build_ho_quench_pool(
        HOQuenchConfig(omega_0=1.0, omega_1=2.0), n_samples=64, seed=0
    )
    # Same seed ⇒ same x ⇒ same per-axis variance, same ω-piece, same E_L0,
    # and Coulomb piece is zero.
    assert torch.allclose(pool_c["x"], pool_h["x"], atol=1e-12)
    assert torch.allclose(pool_c["deltaV"], pool_h["deltaV"], atol=1e-12)
    assert torch.allclose(pool_c["E_L0"], pool_h["E_L0"], atol=1e-12)
    assert torch.allclose(
        pool_c["deltaV_coulomb_piece"],
        torch.zeros_like(pool_c["deltaV_coulomb_piece"]),
        atol=1e-12,
    )


def test_pool_coulomb_piece_is_lambda_times_pair_potential() -> None:
    """``deltaV_coulomb_piece`` = λ · ∑_{i<j} u(r_ij)."""
    cfg = CoulombQuenchConfig(omega_0=1.0, omega_1=1.5, lambda_coul=0.4, epsilon_coul=0.07)
    pool = build_coulomb_quench_pool(cfg, n_samples=32, seed=5)
    pair = softcore_coulomb_pair_potential(pool["x"], epsilon=cfg.epsilon_coul)
    assert torch.allclose(pool["deltaV_coulomb_piece"], cfg.lambda_coul * pair, atol=1e-12)


def test_pool_initial_pair_interaction_matches_psi0_average() -> None:
    """``initial_pair_interaction_value`` returns the |ψ₀|²-pool average of
    ``λ · ∑_{i<j} u(r_ij)`` — i.e. the t=0 expectation value."""
    cfg = CoulombQuenchConfig(
        omega_0=1.0, omega_1=1.0, lambda_coul=0.5, epsilon_coul=0.1
    )
    pool = build_coulomb_quench_pool(cfg, n_samples=4096, seed=7)
    v = initial_pair_interaction_value(pool)
    # Sanity: λ=0.5, ε=0.1 ⇒ V_int(0) ∼ a few tenths (positive, finite).
    assert v > 0.0
    assert np.isfinite(v)


# ---------------------------------------------------------------------------
# Coulomb-off limit reproduces the omega-quench reference exactly
# ---------------------------------------------------------------------------


def test_coulomb_quench_to_ho_quench_round_trip() -> None:
    cfg_c = CoulombQuenchConfig(
        n_particles=3, dim=2, omega_0=0.7, omega_1=1.4, lambda_coul=0.3, epsilon_coul=0.05
    )
    cfg_h = coulomb_quench_to_ho_quench(cfg_c)
    assert cfg_h.n_particles == 3
    assert cfg_h.dim == 2
    assert cfg_h.omega_0 == 0.7
    assert cfg_h.omega_1 == 1.4


def test_analytical_x2_ohne_coulomb_matches_ho_module() -> None:
    cfg = CoulombQuenchConfig(omega_0=1.0, omega_1=2.0, lambda_coul=0.5, epsilon_coul=0.1)
    cfg_h = HOQuenchConfig(omega_0=1.0, omega_1=2.0)
    ts = np.linspace(0, 1.5, 17)
    a = analytical_x2_aggregate_ohne_coulomb(cfg, ts)
    b = analytical_x2_aggregate(cfg_h, ts)
    assert np.allclose(a, b, atol=1e-15)


def test_analytical_breathing_period_coulomb_matches_ho_period() -> None:
    cfg = CoulombQuenchConfig(omega_0=1.0, omega_1=2.5, lambda_coul=0.5, epsilon_coul=0.1)
    cfg_h = HOQuenchConfig(omega_0=1.0, omega_1=2.5)
    assert abs(analytical_breathing_period_coulomb(cfg) - analytical_breathing_period(cfg_h)) < 1e-15
