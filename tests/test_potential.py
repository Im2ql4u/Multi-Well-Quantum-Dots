from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from config import SystemConfig
from imaginary_time_pinn import (
    PINNConfig,
    _compute_delta_potential_for_cfg,
    _with_updated_well_separation,
    compute_potential as legacy_compute_potential,
)
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


def test_zeeman_particle_indices_apply_subset_only():
    x = torch.zeros((1, 3, 2), dtype=torch.float64)
    spin = torch.tensor([[0, 1, 0]], dtype=torch.long)  # spin_z = [+1, -1, +1]

    base = compute_potential_legacy_compatible(
        x,
        omega=1.0,
        well_sep=0.0,
        smooth_T=0.2,
        coulomb=False,
        magnetic_B=0.0,
        spin=spin,
        g_factor=2.0,
        mu_B=1.0,
    )
    masked = compute_potential_legacy_compatible(
        x,
        omega=1.0,
        well_sep=0.0,
        smooth_T=0.2,
        coulomb=False,
        magnetic_B=0.3,
        spin=spin,
        g_factor=2.0,
        mu_B=1.0,
        zeeman_particle_indices=(0, 2),
    )

    # 0.5 * g * mu_B * B * (s0 + s2) = 0.5 * 2 * 1 * 0.3 * (1 + 1) = 0.6
    torch.testing.assert_close(masked - base, torch.tensor([0.6], dtype=torch.float64))


def test_zeeman_subset_and_electron1_mode_conflict():
    x = torch.zeros((1, 2, 2), dtype=torch.float64)
    spin = torch.tensor([[0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_potential_legacy_compatible(
            x,
            omega=1.0,
            well_sep=0.0,
            smooth_T=0.2,
            coulomb=False,
            magnetic_B=0.2,
            spin=spin,
            g_factor=2.0,
            mu_B=1.0,
            zeeman_electron1_only=True,
            zeeman_particle_indices=(0,),
        )


def test_update_well_separation_preserves_two_well_occupancies():
    system = SystemConfig.double_dot(N_L=1, N_R=2, sep=4.0, omega=1.0, dim=2)

    updated = _with_updated_well_separation(system, 8.0)

    assert updated.wells[0].n_particles == 1
    assert updated.wells[1].n_particles == 2
    assert abs(updated.wells[0].center[0] + 4.0) < 1e-12
    assert abs(updated.wells[1].center[0] - 4.0) < 1e-12
    assert abs(updated.wells[0].center[1]) < 1e-12
    assert abs(updated.wells[1].center[1]) < 1e-12


def test_delta_potential_includes_well_separation_quench():
    x = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    spin = torch.tensor([[0, 1]], dtype=torch.long)
    cfg = PINNConfig(
        n_particles=2,
        dim=2,
        omega=1.0,
        well_sep=8.0,
        well_sep_initial=4.0,
        well_sep_final=8.0,
        coulomb=False,
        magnetic_B_initial=0.0,
        magnetic_B=0.0,
    )

    deltaV = _compute_delta_potential_for_cfg(
        x,
        cfg,
        spin_batch=spin,
        system_override=None,
    )
    expected = compute_potential_legacy_compatible(
        x,
        omega=1.0,
        well_sep=8.0,
        smooth_T=0.2,
        coulomb=False,
        magnetic_B=0.0,
        spin=spin,
        g_factor=2.0,
        mu_B=1.0,
    ) - compute_potential_legacy_compatible(
        x,
        omega=1.0,
        well_sep=4.0,
        smooth_T=0.2,
        coulomb=False,
        magnetic_B=0.0,
        spin=spin,
        g_factor=2.0,
        mu_B=1.0,
    )

    torch.testing.assert_close(deltaV, expected)
