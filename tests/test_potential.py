from __future__ import annotations

from types import SimpleNamespace

import torch
from config import SystemConfig
from imaginary_time_pinn import compute_potential as legacy_compute_potential
from potential import compute_potential, compute_potential_legacy_compatible


def _sample_inputs():
    x = torch.tensor(
        [[[-1, 0.5], [0.25, -0.75]], [[0.2, 1.1], [-0.6, 0.4]]], dtype=torch.float64
    )
    spin = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    return (x, spin)


def test_generalized_potential_matches_legacy_single_well():
    (x, spin) = _sample_inputs()
    legacy = SimpleNamespace(
        dim=2, omega=1.25, n_particles=2, well_sep=0, smooth_T=0.2,
        coulomb=True, magnetic_B=0.35, g_factor=2, mu_B=1, zeeman_electron1_only=False,
    )
    expected = legacy_compute_potential(
        x, omega=legacy.omega, well_sep=legacy.well_sep, smooth_T=legacy.smooth_T,
        coulomb=legacy.coulomb, magnetic_B=legacy.magnetic_B, spin=spin,
        g_factor=legacy.g_factor, mu_B=legacy.mu_B,
        zeeman_electron1_only=legacy.zeeman_electron1_only,
    )
    actual = compute_potential(x, system=SystemConfig.from_legacy(legacy), spin=spin)
    torch.testing.assert_close(actual, expected)


def test_generalized_potential_matches_legacy_double_well():
    (x, spin) = _sample_inputs()
    legacy = SimpleNamespace(
        dim=2, omega=0.9, n_particles=2, well_sep=3.5, smooth_T=0.15,
        coulomb=True, magnetic_B=-0.2, g_factor=2.5, mu_B=0.8, zeeman_electron1_only=True,
    )
    expected = legacy_compute_potential(
        x, omega=legacy.omega, well_sep=legacy.well_sep, smooth_T=legacy.smooth_T,
        coulomb=legacy.coulomb, magnetic_B=legacy.magnetic_B, spin=spin,
        g_factor=legacy.g_factor, mu_B=legacy.mu_B,
        zeeman_electron1_only=legacy.zeeman_electron1_only,
    )
    actual = compute_potential_legacy_compatible(
        x, omega=legacy.omega, well_sep=legacy.well_sep, smooth_T=legacy.smooth_T,
        coulomb=legacy.coulomb, magnetic_B=legacy.magnetic_B, spin=spin,
        g_factor=legacy.g_factor, mu_B=legacy.mu_B,
        zeeman_electron1_only=legacy.zeeman_electron1_only,
    )
    torch.testing.assert_close(actual, expected)
