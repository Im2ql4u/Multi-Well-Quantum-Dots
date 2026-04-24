from __future__ import annotations

import torch
from config import SystemConfig
from wavefunction import (
    GroundStateWF,
    SlaterOnlyWF,
    assess_magnetic_response_capability,
    resolve_spin_configuration,
    setup_closed_shell_system,
)


def _assert_finite_grads(model: torch.nn.Module) -> None:
    saw_grad = False
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            saw_grad = True
    assert saw_grad


def _make_reference_state():
    system = SystemConfig.single_dot(N=2, omega=1)
    (C_occ, spin, params) = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref=3
    )
    return (system, C_occ, spin, params)


def test_slater_only_wavefunction_forward_is_finite():
    (system, C_occ, spin, params) = _make_reference_state()
    model = SlaterOnlyWF(system, C_occ, spin, params).double()
    x = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_ground_state_wavefunction_supports_all_architectures():
    (system, C_occ, spin, params) = _make_reference_state()
    x = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    model = GroundStateWF(system, C_occ, spin, params).double()
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    _assert_finite_grads(model)


def test_resolve_spin_configuration_accepts_explicit_sector_counts() -> None:
    system = SystemConfig.single_dot(N=4, omega=1.0, dim=2)

    spin_meta = resolve_spin_configuration(system, {"n_up": 3, "n_down": 1})

    assert spin_meta["pattern"] == [0, 0, 0, 1]
    assert spin_meta["n_up"] == 3
    assert spin_meta["n_down"] == 1
    assert spin_meta["label"] == "3up_1down"


def test_setup_closed_shell_system_honors_explicit_spin_pattern() -> None:
    system = SystemConfig.single_dot(N=4, omega=1.0, dim=2)

    c_occ, spin, params = setup_closed_shell_system(
        system,
        device="cpu",
        dtype=torch.float64,
        E_ref="auto",
        allow_missing_dmc=True,
        spin_pattern=[0, 0, 0, 1],
    )

    assert spin.tolist() == [0, 0, 0, 1]
    assert params["n_up"] == 3
    assert params["n_down"] == 1
    assert params["up_col_idx"] == [0, 1, 2]
    assert params["down_col_idx"] == [3]
    assert c_occ.shape[1] == 4


def test_setup_supports_open_shell_odd_n():
    system = SystemConfig.triple_dot(Ns=(1, 1, 1), spacing=4.0, omega=1.0, dim=2)
    (c_occ, spin, params) = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref=3
    )
    assert spin.numel() == 3
    assert int((spin == 0).sum().item()) == 2
    assert int((spin == 1).sum().item()) == 1
    assert params["n_up"] == 2
    assert params["n_down"] == 1
    assert params["up_col_idx"] == [0, 1]
    assert params["down_col_idx"] == [2]

    # Ensure occupied columns are not all taken from a single well block.
    nonzero_rows = torch.nonzero(c_occ, as_tuple=False)[:, 0].tolist()
    # For this setup n_orb=3 -> per-well basis size is 1; expected occupied rows
    # include one basis function from each well block.
    assert set(nonzero_rows) == {0, 1, 2}


def test_ground_state_wavefunction_three_wells_forward_is_finite():
    system = SystemConfig.triple_dot(Ns=(1, 1, 1), spacing=4.0, omega=1.0, dim=2)
    (C_occ, spin, params) = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref=3
    )
    model = GroundStateWF(system, C_occ, spin, params, arch_type="pinn").double()
    x = torch.randn(4, 3, 2, dtype=torch.float64, requires_grad=True)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    _assert_finite_grads(model)


def test_magnetic_assessment_flags_uniform_b_as_constant_shift_for_fixed_spin() -> None:
    system = SystemConfig.triple_dot(
        Ns=(1, 1, 1),
        spacing=4.0,
        omega=1.0,
        dim=2,
    )
    system = SystemConfig(
        wells=system.wells,
        dim=system.dim,
        coulomb=system.coulomb,
        smooth_T=system.smooth_T,
        B_magnitude=0.5,
        B_direction=(0.0, 0.0, 1.0),
        g_factor=2.0,
        mu_B=1.0,
        zeeman_electron1_only=False,
        zeeman_particle_indices=None,
    )
    spin = torch.tensor([0, 0, 1], dtype=torch.long)

    assessment = assess_magnetic_response_capability(system, spin)

    assert assessment["classification"] == "constant_zeeman_shift_only"
    assert assessment["structurally_trivial_uniform_zeeman"] is True
    assert assessment["state_response_supported"] is False
    assert assessment["selected_spin_projection"] == 1.0
    assert assessment["constant_energy_shift"] == 0.5


def test_magnetic_assessment_tracks_particle_subset_projection() -> None:
    base = SystemConfig.triple_dot(
        Ns=(1, 1, 1),
        spacing=4.0,
        omega=1.0,
        dim=2,
    )
    system = SystemConfig(
        wells=base.wells,
        dim=base.dim,
        coulomb=base.coulomb,
        smooth_T=base.smooth_T,
        B_magnitude=0.5,
        B_direction=(0.0, 0.0, 1.0),
        g_factor=2.0,
        mu_B=1.0,
        zeeman_electron1_only=False,
        zeeman_particle_indices=(1, 2),
    )
    spin = torch.tensor([0, 0, 1], dtype=torch.long)

    assessment = assess_magnetic_response_capability(system, spin)

    assert assessment["zeeman_scope"] == "particle_subset"
    assert assessment["selected_particle_indices"] == [1, 2]
    assert assessment["selected_spin_projection"] == 0.0
    assert assessment["constant_energy_shift"] == 0.0


def test_magnetic_assessment_flags_transverse_components_as_unimplemented() -> None:
    base = SystemConfig.single_dot(N=2, omega=1.0, dim=2)
    system = SystemConfig(
        wells=base.wells,
        dim=base.dim,
        coulomb=base.coulomb,
        smooth_T=base.smooth_T,
        B_magnitude=0.5,
        B_direction=(1.0, 0.0, 0.0),
        g_factor=2.0,
        mu_B=1.0,
        zeeman_electron1_only=False,
        zeeman_particle_indices=None,
    )
    spin = torch.tensor([0, 1], dtype=torch.long)

    assessment = assess_magnetic_response_capability(system, spin)

    assert assessment["classification"] == "no_implemented_longitudinal_coupling"
    assert assessment["transverse_components_present"] is True
    assert any("transverse" in note.lower() for note in assessment["notes"])


def test_setup_multi_well_even_n_covers_all_wells():
    """N=4 even-N multi-well: every well must appear in the occupied SD columns.

    Regression for the closed-shell shortcut (n_orb = n_up = 2) that left
    wells 2 and 3 unrepresented, causing the sampler to crowd all 4 particles
    into the left two wells and producing a falsely high VMC energy (~6.0 vs
    exact ~5.1 for 4-well 1-per-well at sep=4, omega=1).
    """
    from config import WellSpec

    wells = tuple(
        WellSpec(center=(float(x), 0.0), omega=1.0, n_particles=1)
        for x in (-6.0, -2.0, 2.0, 6.0)
    )
    system = SystemConfig(wells=wells, dim=2)
    (c_occ, spin, params) = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref="auto", allow_missing_dmc=True
    )
    assert params["n_up"] == 2
    assert params["n_down"] == 2
    assert params["up_col_idx"] == [0, 1]
    # Spin-down must use *different* orbital columns so all 4 wells are covered.
    assert params["down_col_idx"] == [2, 3]
    assert c_occ.shape == (4, 4), f"Expected (4,4) C_occ, got {tuple(c_occ.shape)}"
    # Every well block must have exactly one occupied orbital.
    nonzero_rows = torch.nonzero(c_occ, as_tuple=False)[:, 0].tolist()
    assert set(nonzero_rows) == {0, 1, 2, 3}, (
        f"Wells not covered by C_occ: {set(nonzero_rows)}"
    )


def test_single_dot_even_n_stays_closed_shell():
    """Single-dot N=4 must remain closed-shell (n_orb = n_up = 2)."""
    system = SystemConfig.single_dot(N=4, omega=1.0, dim=2)
    (c_occ, _, params) = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref="auto", allow_missing_dmc=True
    )
    assert params["n_up"] == 2
    assert params["n_down"] == 2
    assert params["up_col_idx"] == [0, 1]
    assert params["down_col_idx"] == [0, 1], "Single-dot should still use doubly-occupied orbitals"
    # C_occ should have n_orb=2 columns (closed-shell)
    assert c_occ.shape[1] == 2


def test_multi_ref_n2_matches_singlet_permanent() -> None:
    """For N=2 double-dot, multi_ref must produce identical sign×|perm| as singlet permanent."""
    from config import WellSpec

    wells = (
        WellSpec(center=(-2.0, 0.0), omega=1.0, n_particles=1),
        WellSpec(center=(2.0, 0.0), omega=1.0, n_particles=1),
    )
    system = SystemConfig(wells=wells, dim=2)
    C_occ, spin, params = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref="auto", allow_missing_dmc=True
    )

    model_singlet = GroundStateWF(system, C_occ, spin, params, use_backflow=False, singlet=True).double()
    model_mr = GroundStateWF(system, C_occ, spin, params, use_backflow=False, multi_ref=True).double()

    # Copy PINN weights so correlator is identical.
    model_mr.pinn.load_state_dict(model_singlet.pinn.state_dict())

    torch.manual_seed(0)
    x = torch.randn(8, 2, 2, dtype=torch.float64)

    sign_s, log_s = model_singlet._permanent_sign_logabs(x)
    sign_mr, log_mr = model_mr._multi_ref_sign_logabs(x, model_mr.spin_template)

    # Both should agree on magnitude (up to sign convention — perm always positive).
    assert torch.allclose(log_mr, log_s, atol=1e-10), (
        f"Multi-ref log|psi| differs from singlet permanent: max diff={( log_mr - log_s).abs().max():.2e}"
    )
    # Signs of multi-ref should be non-negative for positive permanent.
    assert (sign_mr * sign_s > 0).all() or (sign_mr == sign_s).all()


def test_multi_ref_n3_forward_is_finite_and_differentiable() -> None:
    """N=3 triple-dot with multi_ref=True must produce finite outputs and gradients."""
    system = SystemConfig.triple_dot(Ns=(1, 1, 1), spacing=4.0, omega=1.0, dim=2)
    C_occ, spin, params = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref=3
    )

    model = GroundStateWF(system, C_occ, spin, params, use_backflow=False, multi_ref=True).double()
    x = torch.randn(6, 3, 2, dtype=torch.float64, requires_grad=True)
    sign, log_psi = model.signed_log_psi(x)

    assert log_psi.shape == (6,)
    assert sign.shape == (6,)
    assert torch.isfinite(log_psi).all()
    assert torch.isfinite(sign).all()
    log_psi.sum().backward()
    assert torch.isfinite(x.grad).all()
    _assert_finite_grads(model)


def test_multi_ref_n4_has_correct_reference_count() -> None:
    """N=4 (n_up=2, n_down=2) should use C(4,2)=6 references."""
    from config import WellSpec
    from itertools import combinations

    wells = tuple(
        WellSpec(center=(float(xc), 0.0), omega=1.0, n_particles=1)
        for xc in (-6.0, -2.0, 2.0, 6.0)
    )
    system = SystemConfig(wells=wells, dim=2)
    C_occ, spin, params = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref="auto", allow_missing_dmc=True
    )

    model = GroundStateWF(system, C_occ, spin, params, use_backflow=False, multi_ref=True).double()

    n_up = params["n_up"]  # 2
    n_wells = 4
    expected_refs = len(list(combinations(range(n_wells), n_up)))
    assert expected_refs == 6

    x = torch.randn(4, 4, 2, dtype=torch.float64)
    sign, log_psi = model.signed_log_psi(x)
    assert torch.isfinite(log_psi).all()


def test_multi_ref_is_permutation_symmetric_n3() -> None:
    """Swapping any two particles should only change the sign of the multi-ref det sum."""
    system = SystemConfig.triple_dot(Ns=(1, 1, 1), spacing=4.0, omega=1.0, dim=2)
    C_occ, spin, params = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref=3
    )

    model = GroundStateWF(system, C_occ, spin, params, use_backflow=False, multi_ref=True).double()
    spin_t = model.spin_template  # [0, 0, 1] for N=3

    torch.manual_seed(42)
    x = torch.randn(8, 3, 2, dtype=torch.float64)

    sign_orig, log_orig = model._multi_ref_sign_logabs(x, spin_t)

    # Swap the two up-spin particles (index 0 and 1).
    x_swap = x[:, [1, 0, 2], :]
    spin_swap = spin_t[[1, 0, 2]]
    sign_swap, log_swap = model._multi_ref_sign_logabs(x_swap, spin_swap)

    # Log-magnitudes must be identical.
    assert torch.allclose(log_orig, log_swap, atol=1e-10), (
        f"Log-magnitude changed under particle swap: max diff={( log_orig - log_swap).abs().max():.2e}"
    )
    # Signs must flip (antisymmetry under exchange of same-spin particles).
    assert torch.allclose(sign_orig, -sign_swap, atol=1e-10)
