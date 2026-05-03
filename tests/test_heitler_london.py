"""Unit tests for the Heitler-London N=2 anchor (Phase 0.4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observables.heitler_london import (
    HLConfig,
    HLResult,
    heitler_london,
    heitler_london_density,
    overlap_S,
    spin_correlator_S1S2,
    two_body_LRLR,
    two_body_LRRL,
)


@pytest.fixture
def cfg_typical() -> HLConfig:
    return HLConfig(sep=4.0, omega=1.0, kappa=1.0, epsilon=0.01, smooth_t=0.5)


def test_overlap_matches_closed_form(cfg_typical: HLConfig) -> None:
    s = overlap_S(cfg_typical)
    expected = float(np.exp(-cfg_typical.omega * cfg_typical.sep**2 / 4.0))
    assert s == pytest.approx(expected, rel=1e-12)


def test_overlap_decays_with_separation() -> None:
    s_small = overlap_S(HLConfig(sep=2.0, omega=1.0, kappa=1.0, epsilon=0.01))
    s_large = overlap_S(HLConfig(sep=10.0, omega=1.0, kappa=1.0, epsilon=0.01))
    assert s_small > s_large
    assert s_large < 1e-10


def test_singlet_spin_correlator() -> None:
    assert spin_correlator_S1S2("singlet") == pytest.approx(-0.75, rel=1e-12)


def test_triplet_spin_correlator() -> None:
    assert spin_correlator_S1S2("triplet") == pytest.approx(0.25, rel=1e-12)


def test_invalid_sector_raises() -> None:
    with pytest.raises(ValueError, match="unknown sector"):
        spin_correlator_S1S2("doublet")  # type: ignore[arg-type]


def test_two_body_direct_integral_finite_and_positive(cfg_typical: HLConfig) -> None:
    qv = two_body_LRLR(cfg_typical)
    assert np.isfinite(qv)
    assert qv > 0


def test_two_body_exchange_smaller_than_direct(cfg_typical: HLConfig) -> None:
    qv = two_body_LRLR(cfg_typical)
    kv = two_body_LRRL(cfg_typical)
    assert abs(kv) < qv


def test_heitler_london_basic_smoke(cfg_typical: HLConfig) -> None:
    res = heitler_london(cfg_typical)
    assert isinstance(res, HLResult)
    assert np.isfinite(res.E_singlet)
    assert np.isfinite(res.E_triplet)
    assert np.isfinite(res.J_HL)


def test_singlet_lower_than_triplet_at_moderate_sep() -> None:
    """At sep=4, the antiferromagnetic exchange wins → singlet should be lower."""
    cfg = HLConfig(sep=4.0, omega=1.0, kappa=1.0, epsilon=0.01)
    res = heitler_london(cfg)
    assert res.E_singlet < res.E_triplet
    assert res.J_HL > 0  # Heisenberg J = E_T - E_S > 0 for AFM


def test_J_decays_with_separation() -> None:
    """Larger d → exponentially weaker exchange → smaller J."""
    res_close = heitler_london(HLConfig(sep=3.0, omega=1.0, kappa=1.0, epsilon=0.01))
    res_far = heitler_london(HLConfig(sep=8.0, omega=1.0, kappa=1.0, epsilon=0.01))
    assert res_close.J_HL > res_far.J_HL
    assert res_far.J_HL >= 0  # still non-negative


def test_energy_two_omega_in_far_separated_limit() -> None:
    """In the d → ∞ limit the wells decouple and E ≈ 2ω + small Coulomb tail."""
    cfg = HLConfig(sep=14.0, omega=1.0, kappa=1.0, epsilon=0.01)
    res = heitler_london(cfg)
    # Two non-interacting 2D HOs at separated wells → 2*ω = 2 Ha.
    # Coulomb tail at d=14 is kappa/d ≈ 0.071. Total ≈ 2.071.
    assert 2.0 < res.E_singlet < 2.2
    assert 2.0 < res.E_triplet < 2.2
    assert abs(res.E_singlet - res.E_triplet) < 1e-4


def test_density_normalisation_singlet(cfg_typical: HLConfig) -> None:
    grid = np.linspace(-8.0, 8.0, 161)
    n = heitler_london_density(cfg_typical, grid, grid, sector="singlet")
    dx = grid[1] - grid[0]
    total = float(n.sum() * dx * dx)
    # Integrated density of an N=2 wavefunction is 2 (for singlet/triplet HL,
    # which is normalised so that <Psi|Psi>=1 → <n>=2).
    assert total == pytest.approx(2.0, rel=2e-2)


def test_density_normalisation_triplet(cfg_typical: HLConfig) -> None:
    grid = np.linspace(-8.0, 8.0, 161)
    n = heitler_london_density(cfg_typical, grid, grid, sector="triplet")
    dx = grid[1] - grid[0]
    total = float(n.sum() * dx * dx)
    assert total == pytest.approx(2.0, rel=2e-2)


def test_density_two_peaks_at_well_centres(cfg_typical: HLConfig) -> None:
    grid = np.linspace(-6.0, 6.0, 121)
    n = heitler_london_density(cfg_typical, grid, grid, sector="singlet")
    profile = n[:, len(grid) // 2]  # y = 0 slice
    half = cfg_typical.half_separation()
    idx_l = int(np.argmin(np.abs(grid + half)))
    idx_r = int(np.argmin(np.abs(grid - half)))
    idx_mid = int(np.argmin(np.abs(grid - 0.0)))
    assert profile[idx_l] > profile[idx_mid]
    assert profile[idx_r] > profile[idx_mid]


def test_high_omega_quad_converges() -> None:
    """Increasing the quadrature node count must not change the answer materially."""
    base = heitler_london(HLConfig(sep=4.0, omega=1.0, kappa=1.0, epsilon=0.01, n_quad_per_axis=24))
    high = heitler_london(HLConfig(sep=4.0, omega=1.0, kappa=1.0, epsilon=0.01, n_quad_per_axis=40))
    assert abs(base.E_singlet - high.E_singlet) < 1e-3
    assert abs(base.J_HL - high.J_HL) < 1e-3
