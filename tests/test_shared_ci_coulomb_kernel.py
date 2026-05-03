"""Regression test documenting the shared-CI Coulomb-kernel pathology.

This test pins down what the SHARED-DVR-CI path actually returns vs. the
reliable ONE-PER-WELL CI path at d=8 for ω=1, κ=1, ε=0.01.

The shared-DVR-CI path is mathematically broken because the singular Coulomb
diagonal V_{ii} = κ/ε at production ε=0.01 dominates the two-electron integral
and produces unphysical ground-state energies (below the non-interacting
2.0 Ha limit). This is *not* fixed by toggling the
``include_quadrature_weights`` flag — both conventions yield wrong answers,
just at different magnitudes. The one-per-well path doesn't suffer this issue
because its CI bra/ket orbitals live on disjoint wells, so the diagonal
``V_{ii}`` term contributes negligibly.

This test simply pins down the *current* behaviour so any future changes to
``precompute_coulomb_kernel`` / ``build_ci_hamiltonian`` are caught.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.exact_diag_double_dot import (  # noqa: E402
    DiagConfig,
    build_2d_dvr,
    build_ci_hamiltonian,
    build_potential_matrix,
    build_slater_basis_sorted,
    infer_box_half_widths,
    precompute_coulomb_kernel,
    run_exact_diagonalization_one_per_well,
    single_particle_eigenstates,
)
from scipy.linalg import eigh


def _shared_e0(*, sep: float, omega: float, kappa: float, epsilon: float, include_quadrature_weights: bool) -> float:
    """Return the SHARED-DVR-CI ground-state energy for the given setup."""
    x_half, y_half = infer_box_half_widths(sep, omega=omega, n_wells=2)
    cfg = DiagConfig(
        nx=24,
        ny=18,
        sep=sep,
        omega=omega,
        smooth_t=0.5,
        kappa=kappa,
        epsilon=epsilon,
        n_sp_states=6,
        n_ci_compute=36,
        b_field=0.0,
        g_factor=2.0,
        mu_b=0.5,
        x_half_width=x_half,
        y_half_width=y_half,
        n_wells=2,
        model_mode="shared",
        kinetic_prefactor=0.5,
    )
    x_grid, y_grid, w_x, w_y, t2d = build_2d_dvr(
        nx=cfg.nx, ny=cfg.ny, x_half_width=cfg.x_half_width, y_half_width=cfg.y_half_width,
    )
    v2d = build_potential_matrix(
        x_grid=x_grid, y_grid=y_grid, sep=cfg.sep, omega=cfg.omega, smooth_t=cfg.smooth_t,
    )
    t2d = cfg.kinetic_prefactor * t2d
    sp_e, sp_v = single_particle_eigenstates(t2d=t2d, v2d=v2d, n_sp_states=cfg.n_sp_states)
    slater_basis = build_slater_basis_sorted(cfg.n_sp_states, sp_e)
    kernel = precompute_coulomb_kernel(
        x_grid=x_grid,
        y_grid=y_grid,
        w_x=w_x,
        w_y=w_y,
        kappa=cfg.kappa,
        epsilon=cfg.epsilon,
        include_quadrature_weights=include_quadrature_weights,
    )
    h_ci = build_ci_hamiltonian(
        slater_basis=slater_basis,
        single_energies=sp_e,
        single_vecs=sp_v,
        kernel=kernel,
        n_ci_compute=cfg.n_ci_compute,
        b_field=cfg.b_field,
        g_factor=cfg.g_factor,
        mu_b=cfg.mu_b,
    )
    eigvals, _ = eigh(h_ci)
    return float(eigvals[0])


def _opw_e0(*, sep: float, omega: float, kappa: float, epsilon: float) -> float:
    """Return the ONE-PER-WELL CI ground-state energy (the reliable reference)."""
    x_half, y_half = infer_box_half_widths(sep, omega=omega, n_wells=2)
    cfg = DiagConfig(
        nx=24,
        ny=18,
        sep=sep,
        omega=omega,
        smooth_t=0.5,
        kappa=kappa,
        epsilon=epsilon,
        n_sp_states=6,
        n_ci_compute=36,
        b_field=0.0,
        g_factor=2.0,
        mu_b=0.5,
        x_half_width=x_half,
        y_half_width=y_half,
        n_wells=2,
        model_mode="one_per_well",
        kinetic_prefactor=0.5,
    )
    eigs, _, _, _, _ = run_exact_diagonalization_one_per_well(cfg)
    return float(eigs[0])


def test_one_per_well_reference_is_physical_at_d8():
    """Sanity: the one-per-well reference gives E_GS ≈ 2 + 1/d at d=8."""
    e0 = _opw_e0(sep=8.0, omega=1.0, kappa=1.0, epsilon=0.01)
    assert e0 > 2.0, (
        f"one_per_well E_GS={e0} should be > 2.0 (non-interacting limit) for repulsive Coulomb."
    )
    expected = 2.0 + 1.0 / 8.0
    assert abs(e0 - expected) < 0.05, (
        f"one_per_well E_GS={e0} should be ≈ 2 + 1/d = {expected} at d=8."
    )


def test_shared_dvr_ci_is_unphysical_at_eps_0p01():
    """The shared-DVR-CI path returns unphysical energies at production eps=0.01.

    Pin the current behaviour so any future code change that *fixes* this is
    flagged (and the test can be updated).
    """
    e_w_true = _shared_e0(sep=8.0, omega=1.0, kappa=1.0, epsilon=0.01, include_quadrature_weights=True)
    e_w_false = _shared_e0(sep=8.0, omega=1.0, kappa=1.0, epsilon=0.01, include_quadrature_weights=False)

    # Both shared paths give *wrong* answers at production ε.
    # W=True: undercounts Coulomb (E < 2.0, below non-interacting limit)
    assert e_w_true < 2.0, (
        f"Expected SHARED-DVR-CI(W=True) to undershoot the non-interacting limit "
        f"of 2.0 Ha at d=8, eps=0.01 — got {e_w_true:.4f}. If this test now fails, "
        f"the bug may have been fixed; verify the fix and update the assertion."
    )

    # W=False: massively negative due to the singular DVR-Coulomb diagonal V_{ii}=κ/ε=100
    assert e_w_false < 0.0, (
        f"Expected SHARED-DVR-CI(W=False) to be unphysically negative at "
        f"d=8, eps=0.01 (singular DVR diagonal) — got {e_w_false:.4f}."
    )


def test_shared_dvr_ci_improves_at_larger_epsilon():
    """At eps=1.0 the singular diagonal is suppressed and W=False roughly matches OPW."""
    e_w_false = _shared_e0(sep=8.0, omega=1.0, kappa=1.0, epsilon=1.0, include_quadrature_weights=False)
    e_opw = _opw_e0(sep=8.0, omega=1.0, kappa=1.0, epsilon=1.0)

    # At eps=1.0, the shared path with W=False is much closer to the OPW reference
    # (both feel a softened Coulomb)
    assert e_w_false > 1.5, (
        f"At eps=1.0 the SHARED-DVR-CI(W=False) should be in the physical range, got {e_w_false:.4f}."
    )
    # Rough cross-check: shared(W=False) and OPW differ by < 1.0 Ha at eps=1.0
    assert abs(e_w_false - e_opw) < 1.0, (
        f"Shared(W=False)={e_w_false:.4f} vs OPW={e_opw:.4f} should agree to within 1 Ha at eps=1.0."
    )
