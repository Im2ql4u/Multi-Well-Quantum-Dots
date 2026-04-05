from __future__ import annotations

import pytest
import torch

from observables.validation import compute_virial_metrics
from training import (
    adapt_sigma_fs,
    colloc_fd_loss,
    importance_resample,
    mcmc_resample,
    rayleigh_hybrid_loss,
    weak_form_local_energy,
)
from training.vmc_colloc import GroundStateTrainingConfig, lr_schedule_factor, train_ground_state
from config import SystemConfig
from wavefunction import GroundStateWF, setup_closed_shell_system

_DEFAULT_SIGMA_FS = (0.8, 1.3, 2.0)


def _gaussian_logpsi(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * (x**2).sum(dim=(1, 2))


@pytest.mark.skip(reason="sample_mixture was refactored out; no longer exists in codebase")
def test_sample_mixture_returns_finite_samples_and_logq() -> None:
    pass


def test_importance_resample_returns_requested_batch_without_mcmc() -> None:
    system = SystemConfig.single_dot(N=2, omega=1.0)
    x, ess = importance_resample(
        _gaussian_logpsi,
        n_keep=24,
        n_elec=2,
        dim=2,
        omega=1.0,
        device="cpu",
        dtype=torch.float64,
        n_cand_mult=4,
        sigma_fs=_DEFAULT_SIGMA_FS,
        min_pair_cutoff=0.0,
        weight_temp=1.0,
        logw_clip_q=0.0,
        langevin_steps=0,
        langevin_step_size=0.01,
        system=system,
    )
    assert x.shape == (24, 2, 2)
    assert torch.isfinite(x).all()
    assert ess > 0.0


def test_importance_resample_can_return_normalized_weights() -> None:
    system = SystemConfig.single_dot(N=2, omega=1.0)
    x, ess, w = importance_resample(
        _gaussian_logpsi,
        n_keep=32,
        n_elec=2,
        dim=2,
        omega=1.0,
        device="cpu",
        dtype=torch.float64,
        n_cand_mult=4,
        sigma_fs=_DEFAULT_SIGMA_FS,
        min_pair_cutoff=0.0,
        weight_temp=1.0,
        logw_clip_q=0.0,
        langevin_steps=0,
        langevin_step_size=0.01,
        system=system,
        return_weights=True,
    )
    assert x.shape == (32, 2, 2)
    assert w.shape == (32,)
    assert torch.isfinite(w).all()
    assert torch.all(w > 0)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0, dtype=w.dtype), atol=1e-10, rtol=1e-10)
    assert ess > 0.0


@pytest.mark.skip(reason="sample_multiwell was refactored out; no longer exists in codebase")
def test_sample_multiwell_returns_finite_samples_and_logq() -> None:
    pass


@pytest.mark.skip(reason="langevin validation was removed in refactoring")
def test_importance_resample_rejects_langevin_without_proposal_correction() -> None:
    pass


def test_adapt_sigma_fs_widens_for_small_omega() -> None:
    assert adapt_sigma_fs(1.0, _DEFAULT_SIGMA_FS) == (0.8, 1.3, 2.0)
    # Small omega → large scale → values should be wider than base
    wide = adapt_sigma_fs(0.01, _DEFAULT_SIGMA_FS)
    assert all(w > b for w, b in zip(wide, _DEFAULT_SIGMA_FS))


@pytest.mark.skip(reason="compute_grad_logpsi was refactored out; no longer exists in codebase")
def test_compute_grad_and_weak_form_match_single_particle_gaussian_case() -> None:
    pass


def test_collocation_losses_are_finite_for_single_particle_gaussian_case() -> None:
    system = SystemConfig.single_dot(N=1, omega=1.0)
    x = torch.randn(8, 1, 2, dtype=torch.float64)
    params = {"omega": 1.0}

    fd_loss, e_mean, E_L, _ = colloc_fd_loss(
        _gaussian_logpsi,
        x,
        omega=1.0,
        params=params,
        system=system,
        h=0.01,
    )
    hybrid_loss, reward, E_eff, _ = rayleigh_hybrid_loss(
        _gaussian_logpsi,
        x,
        omega=1.0,
        params=params,
        system=system,
        direct_weight=0.1,
    )

    assert torch.isfinite(fd_loss)
    assert torch.isfinite(E_L).all()
    assert torch.isfinite(hybrid_loss)
    assert torch.isfinite(E_eff).all()
    assert isinstance(e_mean, float)
    assert isinstance(reward, float)


def test_lr_schedule_factor_hits_expected_endpoints() -> None:
    factors = [
        lr_schedule_factor(epoch, total_epochs=8, warmup_epochs=2, min_factor=0.1)
        for epoch in range(8)
    ]

    assert factors[0] == 0.1
    assert factors[2] == 1.0
    assert factors[-1] == 0.1
    assert max(factors) == 1.0


def test_lr_schedule_factor_is_flat_when_disabled() -> None:
    factors = [
        lr_schedule_factor(epoch, total_epochs=5, warmup_epochs=0, min_factor=1.0)
        for epoch in range(5)
    ]

    assert factors == [1.0] * 5


def test_compute_virial_metrics_uses_coulomb_minus_sign() -> None:
    virial_lhs, virial_rhs, virial_residual, virial_relative = compute_virial_metrics(
        T_mean=0.5,
        V_trap_mean=1.0,
        V_int_mean=1.0,
        E_mean=2.0,
    )

    assert virial_lhs == 1.0
    assert virial_rhs == 1.0
    assert virial_residual == 0.0
    assert virial_relative == 0.0


def test_mcmc_resample_gaussian_target_mean_r2() -> None:
    """2D harmonic ground state: log ψ = -ω r²/2  ⇒  ⟨r²⟩ = 1/ω (ℏ=m=1)."""
    omega = 1.0
    system = SystemConfig.single_dot(N=1, omega=omega)

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return -0.5 * omega * (x**2).sum(dim=(1, 2))

    torch.manual_seed(0)
    x, accept_rate, _ = mcmc_resample(
        psi_log_fn,
        None,
        2048,
        n_elec=1,
        dim=2,
        omega=omega,
        device="cpu",
        dtype=torch.float64,
        system=system,
        sigma_fs=_DEFAULT_SIGMA_FS,
        mh_steps=400,
        mh_step_scale=0.25,
        mh_decorrelation=1,
    )
    r2 = (x**2).sum(dim=(1, 2))
    mean_r2 = float(r2.mean().item())
    assert abs(mean_r2 - 1.0) < 0.05 * 1.0 + 1e-2
    assert 0.2 <= accept_rate <= 0.85


def test_mcmc_resample_default_accept_rate_reasonable() -> None:
    omega = 1.0
    system = SystemConfig.single_dot(N=1, omega=omega)

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return -0.5 * omega * (x**2).sum(dim=(1, 2))

    torch.manual_seed(1)
    _, accept_rate, _ = mcmc_resample(
        psi_log_fn,
        None,
        512,
        n_elec=1,
        dim=2,
        omega=omega,
        device="cpu",
        dtype=torch.float64,
        system=system,
        sigma_fs=_DEFAULT_SIGMA_FS,
        mh_steps=50,
        mh_step_scale=0.25,
        mh_decorrelation=1,
    )
    assert 0.15 <= accept_rate <= 0.85


def test_mcmc_resample_multiwell_gaussian_target_tracks_well_centres() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=4.0, omega=1.0)

    centres = torch.tensor([[-2.0, 0.0], [2.0, 0.0]], dtype=torch.float64)

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        diff = x - centres.unsqueeze(0)
        return -0.5 * diff.pow(2).sum(dim=(1, 2))

    torch.manual_seed(2)
    x, accept_rate, _ = mcmc_resample(
        psi_log_fn,
        None,
        1024,
        n_elec=2,
        dim=2,
        omega=1.0,
        device="cpu",
        dtype=torch.float64,
        system=system,
        sigma_fs=_DEFAULT_SIGMA_FS,
        mh_steps=80,
        mh_step_scale=0.25,
        mh_decorrelation=1,
    )

    mean_pos = x.mean(dim=0)
    assert abs(float(mean_pos[0, 0].item()) + 2.0) < 0.25
    assert abs(float(mean_pos[1, 0].item()) - 2.0) < 0.25
    assert abs(float(mean_pos[:, 1].mean().item())) < 0.2
    assert 0.15 <= accept_rate <= 0.85


def test_train_ground_state_smoke_runs_on_cpu() -> None:
    system = SystemConfig.single_dot(N=2, omega=1.0)
    C_occ, spin, params = setup_closed_shell_system(
        system,
        device="cpu",
        dtype=torch.float64,
        E_ref=3.0,
    )
    model = GroundStateWF(
        system,
        C_occ,
        spin,
        params,
        pinn_hidden=16,
        bf_hidden=8,
    ).double()
    train_cfg = GroundStateTrainingConfig(
        epochs=2,
        n_coll=16,
        n_cand_mult=2,
        device="cpu",
        dtype="float64",
        print_every=1,
        seed=3,
    )

    result = train_ground_state(model, system, params, train_cfg)
    assert len(result["history"]["loss"]) == 2
    assert len(result["history"]["energy_var"]) == 2
    assert all(torch.isfinite(torch.tensor(result["history"][key])).all() for key in ("loss", "energy", "ess"))
    assert torch.isfinite(torch.tensor(result["history"]["energy_var"])).all()
    assert "diagnostics" in result
    assert "final_energy_var" in result


def test_train_ground_state_mh_sampler_smoke_runs_on_cpu() -> None:
    system = SystemConfig.single_dot(N=2, omega=1.0)
    C_occ, spin, params = setup_closed_shell_system(
        system,
        device="cpu",
        dtype=torch.float64,
        E_ref=3.0,
    )
    model = GroundStateWF(
        system,
        C_occ,
        spin,
        params,
        pinn_hidden=16,
        bf_hidden=8,
    ).double()
    train_cfg = GroundStateTrainingConfig(
        epochs=5,
        n_coll=32,
        n_cand_mult=2,
        sampler="mh",
        mh_steps=4,
        mh_step_scale=0.25,
        device="cpu",
        dtype="float64",
        print_every=100,
        seed=7,
    )

    result = train_ground_state(model, system, params, train_cfg)
    assert len(result["history"]["loss"]) == 5
    assert all(0.0 <= v <= 1.0 for v in result["history"]["ess"])


# ---------------------------------------------------------------------------
# Per-well GMM sampler tests
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="sample_multiwell was refactored out; no longer exists in codebase")
def test_sample_multiwell_single_dot_returns_finite_samples() -> None:
    pass


@pytest.mark.skip(reason="sample_multiwell was refactored out; no longer exists in codebase")
def test_sample_multiwell_double_dot_concentrates_near_well_centers() -> None:
    pass


def test_importance_resample_multiwell_ess_above_threshold() -> None:
    """Check that per-well IS gives healthy ESS for a double-well Gaussian wavefunction."""
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=6.0, omega=1.0, dim=2)
    c1 = torch.tensor([[-3.0, 0.0]], dtype=torch.float64)
    c2 = torch.tensor([[3.0, 0.0]], dtype=torch.float64)

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        e1 = -0.5 * ((x[:, 0:1, :] - c1) ** 2).sum(-1).squeeze(-1)
        e2 = -0.5 * ((x[:, 1:2, :] - c2) ** 2).sum(-1).squeeze(-1)
        return e1 + e2

    torch.manual_seed(42)
    _, ess_multiwell = importance_resample(
        psi_log_fn,
        n_keep=128,
        n_elec=2,
        dim=2,
        omega=1.0,
        device="cpu",
        dtype=torch.float64,
        n_cand_mult=8,
        sigma_fs=_DEFAULT_SIGMA_FS,
        min_pair_cutoff=0.0,
        weight_temp=1.0,
        logw_clip_q=0.0,
        langevin_steps=0,
        langevin_step_size=0.01,
        system=system,
    )

    # Without multiwell proposal, ESS typically ≪ 10 for sep=6
    assert ess_multiwell > 20.0, (
        f"ESS={ess_multiwell:.1f} is too low — per-well proposal not improving coverage"
    )
