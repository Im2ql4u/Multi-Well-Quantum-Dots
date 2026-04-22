#!/usr/bin/env python
"""
Imaginary-Time Spectroscopy — Neural PDE Solver (Full PINN)
============================================================

A FiLM-conditioned neural network g_θ(x, τ) learns the full imaginary-time
dynamics from the PDE residual, discovering excited-state gaps autonomously.

Architecture:
  log ψ(x, τ) = log ψ_0(x) + g_θ(x, τ)

  ψ_0: frozen VMC ground state (SD × PINN × Backflow, ~25K params)
  g_θ:  FiLM-conditioned MLP (~3K params) trained on PDE residual

PDE (imaginary-time Schrödinger equation):
  ∂_τ log ψ = -(E_L - E_0)

Expanding:
  ∂_τ g = -(E_L^(0) - E_0) + ½∇²g + ∇logψ_0·∇g + ½|∇g|²

The network learns the spectral decomposition:
  g(x, τ) ≈ Σ_k a_k(x) exp(-(E_k - E_0)τ)

and E(τ) = E_0 + Σ_k ΔE_k exp(-2(E_k-E_0)τ) reveals the gaps.

Training:
  Phase 1: VMC ground state (SD × PINN × BF)
  Phase 2: Pre-compute E_L^(0), ∇logψ_0 on cached samples
  Phase 3: Train g_θ on PDE residual + IC/BC losses
  Phase 4: Evaluate E(τ), fit exponential gaps

Usage:
  python imaginary_time_pinn.py --test_free   # N=2 non-interacting validation
  python imaginary_time_pinn.py --tiny        # N=2 interacting, quick test
  python imaginary_time_pinn.py --full        # N=2 interacting, converged
  python imaginary_time_pinn.py --sweep       # distance sweep
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUDA_MANUAL_DEVICE", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import SystemConfig, WellSpec
from functions.Slater_Determinant import slater_determinant_closed_shell
from PINN import PINN, CTNNBackflowNet
from potential import compute_potential as compute_potential_generalized
from potential import compute_potential_legacy_compatible
from run_ground_state import _build_system
from training.sampling import stratified_logpdf, stratified_resample
from training.vmc_colloc import clip_local_energy_by_mad
from wavefunction import GroundStateWF as GeneralizedGroundStateWF
from wavefunction import resolve_reference_energy, resolve_spin_configuration, setup_closed_shell_system

_manual = os.environ.get("CUDA_MANUAL_DEVICE")
if _manual is not None and torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{_manual}" if _manual.isdigit() else _manual)
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
torch.set_num_threads(4)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "imag_time_pinn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================
@dataclass
class PINNConfig:
    n_particles: int = 2
    dim: int = 2
    omega: float = 1.0
    well_sep: float = 0.0
    well_sep_initial: float | None = None
    well_sep_final: float | None = None
    smooth_T: float = 0.2
    E_ref: float = 3.0
    coulomb: bool = True  # set False for non-interacting test
    magnetic_B_initial: float = 0.0  # magnetic field for VMC ground state prep (usually 0)
    magnetic_B: float = 0.0  # magnetic field for imaginary-time evolution (sudden quench)
    g_factor: float = 2.0
    mu_B: float = 1.0
    zeeman_electron1_only: bool = False
    zeeman_particle_indices: tuple[int, ...] | None = None
    tau_max: float = 5.0
    # Phase 1: VMC
    n_epochs_vmc: int = 800
    n_samples_vmc: int = 512
    lr_vmc: float = 3e-3
    # Phase 2: pre-computation
    n_precompute: int = 4096
    # Phase 3: PDE training
    n_epochs_pde: int = 8000
    batch_pde: int = 256
    lr_pde: float = 1e-3
    lambda_ic: float = 50.0  # IC loss weight
    lambda_bc: float = 1.0  # BC loss weight
    lambda_reg: float = 0.01  # output regularisation
    ic_amplitude: float = 0.5  # amplitude of initial perturbation
    ic_type: str = "dipole"  # "dipole", "quadrupole", or "random"
    # Phase 4: evaluation
    n_tau_eval: int = 40
    n_samples_eval: int = 6000
    eval_sampler: str = "precomputed"  # precomputed, mcmc, stratified
    eval_local_energy_clip_width: float = 0.0
    eval_sampler_mix_weights: tuple[float, float, float, float, float] = (
        0.55,
        0.05,
        0.20,
        0.20,
        0.0,
    )
    eval_sampler_sigma_center: float = 0.15
    eval_sampler_sigma_tails: float = 0.80
    eval_sampler_sigma_mixed_in: float = 0.20
    eval_sampler_sigma_mixed_out: float = 0.50
    eval_sampler_shell_radius: float = 1.00
    eval_sampler_shell_radius_sigma: float = 0.05
    eval_sampler_dimer_pairs: int = 0
    eval_sampler_dimer_eps_max: float = 0.0
    # Architecture
    pinn_hidden: int = 64
    pinn_layers: int = 2
    bf_hidden: int = 32
    bf_layers: int = 2
    use_backflow: bool = True
    g_hidden: int = 64  # FiLM network hidden dim
    g_layers: int = 3  # FiLM network layers
    g_tau_embed: int = 32  # τ embedding dim
    g_n_freq: int = 6  # Fourier frequencies for τ
    use_spectral: bool = True  # use SpectralG (preferred) vs FiLM
    g_modes: int = 3  # number of spectral modes
    no_vmc_train: bool = False  # use untrained Slater-only reference instead of VMC phase
    # Optional path to a run_ground_state output directory containing model.pt
    # and config.yaml. If set, Phase 1 loads this frozen GS artifact instead of
    # training a legacy ground state in this script.
    ground_state_dir: str | None = None
    seed: int | None = None  # RNG seed for reproducibility; None = non-deterministic
    # PINN training variants
    loss_style: str = "curriculum_mse"  # curriculum_mse, mse, huber, logcosh
    tau_sampling: str = "small_bias"  # small_bias, uniform, two_stage
    x_sampling: str = "uniform"  # uniform, energy_weighted
    tau_small_bias_power: float = 1.5
    # Phase 2 precompute sampler
    precompute_sampler: str = "mcmc"  # mcmc, stratified
    precompute_local_energy_clip_width: float = 5.0
    precompute_sampler_mix_weights: tuple[float, float, float, float, float] = (
        0.25,
        0.2,
        0.25,
        0.2,
        0.1,
    )
    precompute_sampler_sigma_center: float = 0.20
    precompute_sampler_sigma_tails: float = 1.20
    precompute_sampler_sigma_mixed_in: float = 0.25
    precompute_sampler_sigma_mixed_out: float = 0.90
    precompute_sampler_shell_radius: float = 1.40
    precompute_sampler_shell_radius_sigma: float = 0.08
    precompute_sampler_dimer_pairs: int = 2
    precompute_sampler_dimer_eps_max: float = 0.08


# ============================================================
# Potential energy
# ============================================================
def compute_potential(
    x: torch.Tensor,
    omega: float,
    well_sep: float,
    smooth_T: float = 0.2,
    coulomb: bool = True,
    magnetic_B: float = 0.0,
    spin: torch.Tensor | None = None,
    g_factor: float = 2.0,
    mu_B: float = 1.0,
    zeeman_electron1_only: bool = False,
    zeeman_particle_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
    return compute_potential_legacy_compatible(
        x,
        omega=omega,
        well_sep=well_sep,
        smooth_T=smooth_T,
        coulomb=coulomb,
        magnetic_B=magnetic_B,
        spin=spin,
        g_factor=g_factor,
        mu_B=mu_B,
        zeeman_electron1_only=zeeman_electron1_only,
        zeeman_particle_indices=zeeman_particle_indices,
    )


def _spin_batch_for_model(model: nn.Module, batch_size: int) -> torch.Tensor:
    if hasattr(model, "spin"):
        spin = getattr(model, "spin")
    elif hasattr(model, "spin_template"):
        spin = getattr(model, "spin_template")
    else:
        raise AttributeError("Ground-state model has neither 'spin' nor 'spin_template'.")
    if not isinstance(spin, torch.Tensor):
        raise TypeError("Ground-state spin attribute must be a torch.Tensor.")
    return spin.to(device=DEVICE).expand(batch_size, -1)


def _compute_potential_for_cfg(
    x: torch.Tensor,
    cfg: PINNConfig,
    *,
    spin_batch: torch.Tensor,
    magnetic_B: float,
    system_override,
    well_sep_override: float | None = None,
) -> torch.Tensor:
    if system_override is None:
        return compute_potential(
            x,
            cfg.omega,
            cfg.well_sep if well_sep_override is None else float(well_sep_override),
            cfg.smooth_T,
            cfg.coulomb,
            magnetic_B=magnetic_B,
            spin=spin_batch,
            g_factor=cfg.g_factor,
            mu_B=cfg.mu_B,
            zeeman_electron1_only=cfg.zeeman_electron1_only,
            zeeman_particle_indices=cfg.zeeman_particle_indices,
        )

    system_at_B = system_override
    if well_sep_override is not None and system_override.n_wells == 2:
        system_at_B = _with_updated_well_separation(system_at_B, float(well_sep_override))

    system_at_B = replace(
        system_at_B,
        B_magnitude=float(magnetic_B),
        g_factor=float(cfg.g_factor),
        mu_B=float(cfg.mu_B),
        zeeman_electron1_only=bool(cfg.zeeman_electron1_only),
        zeeman_particle_indices=cfg.zeeman_particle_indices,
    )
    return compute_potential_generalized(x, system=system_at_B, spin=spin_batch)


def _system_max_well_separation(system: SystemConfig) -> float:
    if len(system.wells) < 2:
        return 0.0
    return max(
        math.sqrt(sum((a - b) ** 2 for a, b in zip(w1.center, w2.center)))
        for i, w1 in enumerate(system.wells)
        for w2 in system.wells[i + 1 :]
    )


def _with_updated_well_separation(system: SystemConfig, well_sep: float) -> SystemConfig:
    if len(system.wells) < 2:
        return system
    if len(system.wells) != 2:
        raise ValueError("well-separation quench currently supports two-well systems only.")

    left_well, right_well = system.wells
    midpoint = tuple(0.5 * (a + b) for a, b in zip(left_well.center, right_well.center))
    left_center = list(midpoint)
    right_center = list(midpoint)
    left_center[0] -= 0.5 * float(well_sep)
    right_center[0] += 0.5 * float(well_sep)

    return SystemConfig.custom(
        wells=(
            WellSpec(
                center=tuple(left_center),
                omega=float(left_well.omega),
                n_particles=int(left_well.n_particles),
            ),
            WellSpec(
                center=tuple(right_center),
                omega=float(right_well.omega),
                n_particles=int(right_well.n_particles),
            ),
        ),
        dim=int(system.dim),
        coulomb=bool(system.coulomb),
        smooth_T=float(system.smooth_T),
        B_magnitude=float(system.B_magnitude),
        B_direction=tuple(float(v) for v in system.B_direction),
        g_factor=float(system.g_factor),
        mu_B=float(system.mu_B),
        zeeman_electron1_only=bool(system.zeeman_electron1_only),
        zeeman_particle_indices=system.zeeman_particle_indices,
    )


def _resolved_well_sep_initial(cfg: PINNConfig, system_override) -> float:
    if cfg.well_sep_initial is not None:
        return float(cfg.well_sep_initial)
    if system_override is not None:
        return _system_max_well_separation(system_override)
    return float(cfg.well_sep)


def _resolved_well_sep_final(cfg: PINNConfig, system_override) -> float:
    if cfg.well_sep_final is not None:
        return float(cfg.well_sep_final)
    if cfg.well_sep_initial is not None:
        return float(cfg.well_sep)
    if system_override is not None:
        return _system_max_well_separation(system_override)
    return float(cfg.well_sep)


def _compute_delta_potential_for_cfg(
    x: torch.Tensor,
    cfg: PINNConfig,
    *,
    spin_batch: torch.Tensor,
    system_override,
) -> torch.Tensor:
    well_sep_initial = _resolved_well_sep_initial(cfg, system_override)
    well_sep_final = _resolved_well_sep_final(cfg, system_override)
    same_B = abs(cfg.magnetic_B - cfg.magnetic_B_initial) <= 1e-14
    same_sep = abs(well_sep_final - well_sep_initial) <= 1e-14
    if same_B and same_sep:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    V_initial = _compute_potential_for_cfg(
        x,
        cfg,
        spin_batch=spin_batch,
        magnetic_B=cfg.magnetic_B_initial,
        system_override=system_override,
        well_sep_override=well_sep_initial,
    )
    V_final = _compute_potential_for_cfg(
        x,
        cfg,
        spin_batch=spin_batch,
        magnetic_B=cfg.magnetic_B,
        system_override=system_override,
        well_sep_override=well_sep_final,
    )
    return (V_final - V_initial).detach()


def _load_locked_ground_state(cfg: PINNConfig):
    if not cfg.ground_state_dir:
        raise ValueError("ground_state_dir is required to load locked ground state.")
    gs_dir = Path(cfg.ground_state_dir).expanduser().resolve()
    model_path = gs_dir / "model.pt"
    config_path = gs_dir / "config.yaml"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ground-state model checkpoint: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing ground-state config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    system = _build_system(raw_cfg["system"])
    arch_cfg = raw_cfg.get("architecture", {})
    allow_missing_dmc = bool(raw_cfg.get("allow_missing_dmc", True))
    input_E_ref = raw_cfg.get("E_ref", "auto")
    spin_cfg = raw_cfg.get("spin")
    spin_meta = resolve_spin_configuration(system, spin_cfg)
    resolved_E_ref = resolve_reference_energy(
        system,
        input_E_ref,
        allow_missing_dmc=allow_missing_dmc,
    )
    C_occ, spin, params = setup_closed_shell_system(
        system,
        device=DEVICE,
        dtype=DTYPE,
        E_ref=resolved_E_ref,
        allow_missing_dmc=allow_missing_dmc,
        spin_pattern=spin_meta["pattern"],
    )

    wf = GeneralizedGroundStateWF(
        system,
        C_occ,
        spin,
        params,
        arch_type=arch_cfg.get("arch_type", "pinn"),
        pinn_hidden=int(arch_cfg.get("pinn_hidden", 64)),
        pinn_layers=int(arch_cfg.get("pinn_layers", 2)),
        bf_hidden=int(arch_cfg.get("bf_hidden", 32)),
        bf_layers=int(arch_cfg.get("bf_layers", 2)),
        use_well_features=bool(arch_cfg.get("use_well_features", False)),
        use_well_backflow=bool(arch_cfg.get("use_well_backflow", False)),
        use_backflow=bool(arch_cfg.get("use_backflow", True)),
    ).to(device=DEVICE, dtype=DTYPE)

    state = torch.load(model_path, map_location=DEVICE)
    wf.load_state_dict(state, strict=True)
    wf.eval()
    for p in wf.parameters():
        p.requires_grad_(False)
    return wf, system, raw_cfg


def _load_locked_ground_state_energy(gs_dir: Path) -> float | None:
    result_path = gs_dir / "result.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    for key in ("final_energy", "energy_mean_last_window"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            energy = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(energy):
            return energy
    return None


def _build_initial_walkers(
    n_samples: int,
    n_particles: int,
    dim: int,
    omega: float,
    well_sep: float = 0.0,
    system_override=None,
) -> torch.Tensor:
    sigma = 1.0 / math.sqrt(omega)
    x = torch.randn(n_samples, n_particles, dim, device=DEVICE, dtype=DTYPE) * sigma

    if system_override is None:
        if well_sep > 1e-10:
            x[:, 0, 0] -= well_sep / 2
            x[:, 1, 0] += well_sep / 2
        return x

    particle_centers: list[tuple[float, ...]] = []
    particle_sigmas: list[float] = []
    for well in system_override.wells:
        local_sigma = 1.0 / math.sqrt(float(well.omega))
        for _ in range(int(well.n_particles)):
            particle_centers.append(tuple(float(c) for c in well.center))
            particle_sigmas.append(local_sigma)

    if len(particle_centers) != n_particles:
        return x

    centers = torch.tensor(particle_centers, device=DEVICE, dtype=DTYPE)
    scales = torch.tensor(particle_sigmas, device=DEVICE, dtype=DTYPE).view(1, n_particles, 1)
    return centers.unsqueeze(0) + 0.5 * scales * torch.randn_like(x)


def _legacy_system_from_cfg(cfg: PINNConfig, *, well_sep_override: float | None = None) -> SystemConfig:
    well_sep = float(cfg.well_sep if well_sep_override is None else well_sep_override)
    if well_sep > 1e-10:
        return SystemConfig.double_dot(
            N_L=cfg.n_particles // 2,
            N_R=cfg.n_particles - cfg.n_particles // 2,
            sep=well_sep,
            omega=float(cfg.omega),
            dim=int(cfg.dim),
        )
    return SystemConfig.single_dot(
        N=int(cfg.n_particles),
        omega=float(cfg.omega),
        dim=int(cfg.dim),
    )


def _precompute_system(cfg: PINNConfig, system_override) -> SystemConfig:
    well_sep_initial = _resolved_well_sep_initial(cfg, system_override)
    if system_override is not None:
        legacy = _with_updated_well_separation(system_override, well_sep_initial)
    else:
        legacy = _legacy_system_from_cfg(cfg, well_sep_override=well_sep_initial)
    return replace(
        legacy,
        coulomb=bool(cfg.coulomb),
        smooth_T=float(cfg.smooth_T),
        B_magnitude=float(cfg.magnetic_B_initial),
        g_factor=float(cfg.g_factor),
        mu_B=float(cfg.mu_B),
        zeeman_electron1_only=bool(cfg.zeeman_electron1_only),
        zeeman_particle_indices=cfg.zeeman_particle_indices,
    )


def _stratified_sampler_kwargs(cfg: PINNConfig, *, eval_mode: bool) -> dict:
    if eval_mode:
        return {
            "component_weights": cfg.eval_sampler_mix_weights,
            "sigma_center": cfg.eval_sampler_sigma_center,
            "sigma_tails": cfg.eval_sampler_sigma_tails,
            "sigma_mixed_in": cfg.eval_sampler_sigma_mixed_in,
            "sigma_mixed_out": cfg.eval_sampler_sigma_mixed_out,
            "shell_radius": cfg.eval_sampler_shell_radius,
            "shell_radius_sigma": cfg.eval_sampler_shell_radius_sigma,
            "dimer_pairs": cfg.eval_sampler_dimer_pairs,
            "dimer_eps_max": cfg.eval_sampler_dimer_eps_max,
        }
    return {
        "component_weights": cfg.precompute_sampler_mix_weights,
        "sigma_center": cfg.precompute_sampler_sigma_center,
        "sigma_tails": cfg.precompute_sampler_sigma_tails,
        "sigma_mixed_in": cfg.precompute_sampler_sigma_mixed_in,
        "sigma_mixed_out": cfg.precompute_sampler_sigma_mixed_out,
        "shell_radius": cfg.precompute_sampler_shell_radius,
        "shell_radius_sigma": cfg.precompute_sampler_shell_radius_sigma,
        "dimer_pairs": cfg.precompute_sampler_dimer_pairs,
        "dimer_eps_max": cfg.precompute_sampler_dimer_eps_max,
    }


def _compute_precomputed_quantities(
    x: torch.Tensor,
    ground_wf: nn.Module,
    cfg: PINNConfig,
    *,
    system_override=None,
    acc: float | None = None,
    ess: float | None = None,
    sampler_name: str,
    clip_width: float,
    extra_fields: dict | None = None,
) -> dict:
    x_g = x.detach().requires_grad_(True)
    lp = ground_wf(x_g)
    grad_lp = torch.autograd.grad(lp.sum(), x_g, create_graph=True)[0]

    B, N, d = x_g.shape
    lap = torch.zeros(B, device=DEVICE, dtype=DTYPE)
    for i in range(N):
        for j in range(d):
            g2 = torch.autograd.grad(
                grad_lp[:, i, j].sum(), x_g, create_graph=True, retain_graph=True
            )[0]
            lap += g2[:, i, j]

    T = -0.5 * (lap + (grad_lp**2).sum(dim=(1, 2)))
    spin_batch = _spin_batch_for_model(ground_wf, B)
    well_sep_initial = _resolved_well_sep_initial(cfg, system_override)
    V = _compute_potential_for_cfg(
        x_g,
        cfg,
        spin_batch=spin_batch,
        magnetic_B=cfg.magnetic_B_initial,
        system_override=system_override,
        well_sep_override=well_sep_initial,
    )
    E_L0 = (T + V).detach()
    E_L0 = clip_local_energy_by_mad(E_L0, clip_width)

    deltaV = _compute_delta_potential_for_cfg(
        x_g,
        cfg,
        spin_batch=spin_batch,
        system_override=system_override,
    )

    result = {
        "x": x_g.detach(),
        "log_psi0": lp.detach(),
        "E_L0": E_L0,
        "deltaV": deltaV,
        "grad_log_psi0": grad_lp.detach(),
        "acc": acc,
        "ess": ess,
        "sampler": sampler_name,
    }
    if extra_fields:
        result.update(extra_fields)

    stats = (
        f"  <E_L^(0)> = {E_L0.mean():.5f} +/- {E_L0.std().item()/math.sqrt(B):.5f}"
    )
    if acc is not None:
        stats += f", acc = {acc:.2f}"
    if ess is not None:
        stats += f", ess = {ess:.1f}"
    print(stats)
    if torch.any(deltaV.abs() > 1e-14):
        print(
            f"  <ΔV_quench> = {deltaV.mean():.5f} +/- {deltaV.std().item()/math.sqrt(B):.5f}"
        )
        if abs(float(deltaV.mean().item())) < 1e-8:
            print(
                "  WARNING: ΔV_quench is ~0. The selected Zeeman subset likely cancels "
                "net spin-z and induces no effective quench."
            )
    return result


# ============================================================
# MCMC sampling
# ============================================================
@torch.no_grad()
def mcmc_sample(
    log_psi_fn,
    n_samples,
    n_particles,
    dim,
    omega,
    well_sep=0.0,
    n_warmup=300,
    step_size=0.5,
    init_x: torch.Tensor | None = None,
):
    sigma = 1.0 / math.sqrt(omega)
    if init_x is None:
        x = torch.randn(n_samples, n_particles, dim, device=DEVICE, dtype=DTYPE) * sigma
        if well_sep > 1e-10:
            x[:, 0, 0] -= well_sep / 2
            x[:, 1, 0] += well_sep / 2
    else:
        x = init_x.to(device=DEVICE, dtype=DTYPE)

    log_prob = 2 * log_psi_fn(x)
    n_accept = 0
    for step in range(n_warmup):
        x_new = x + step_size * sigma * torch.randn_like(x)
        log_prob_new = 2 * log_psi_fn(x_new)
        accept = torch.log(torch.rand(n_samples, device=DEVICE, dtype=DTYPE)) < (
            log_prob_new - log_prob
        )
        x = torch.where(accept.view(-1, 1, 1), x_new, x)
        log_prob = torch.where(accept, log_prob_new, log_prob)
        if step >= n_warmup - 50:
            n_accept += accept.float().mean().item()
    return x.detach(), n_accept / 50


# ============================================================
# Ground-state wavefunction: SD × PINN × Backflow
# ============================================================
class GroundStateWF(nn.Module):
    def __init__(
        self,
        n_particles,
        dim,
        omega,
        C_occ,
        spin,
        params,
        *,
        pinn_hidden=64,
        pinn_layers=2,
        bf_hidden=32,
        bf_layers=2,
        use_backflow=True,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.omega = omega
        self.register_buffer("C_occ", C_occ)
        self.register_buffer("spin", spin)
        self.params = params
        self.use_backflow = use_backflow

        self.pinn = PINN(
            n_particles=n_particles,
            d=dim,
            omega=omega,
            dL=5,
            hidden_dim=pinn_hidden,
            n_layers=pinn_layers,
            act="gelu",
            init="xavier",
            use_gate=True,
        )
        self.bf_net = None
        if use_backflow:
            self.bf_net = CTNNBackflowNet(
                d=dim,
                msg_hidden=bf_hidden,
                msg_layers=2,
                hidden=bf_hidden,
                layers=bf_layers,
                act="gelu",
                aggregation="mean",
                use_spin=True,
                same_spin_only=False,
                out_bound="tanh",
                bf_scale_init=0.3,
                zero_init_last=True,
                omega=omega,
            )

    def forward(self, x):
        B = x.shape[0]
        spin = self.spin.expand(B, -1)
        x_eff = x
        if self.use_backflow and self.bf_net is not None:
            dx = self.bf_net(x, spin=spin)
            x_eff = x + dx
        f_pinn = self.pinn(x, spin=spin)
        _, log_sd = slater_determinant_closed_shell(
            x_eff,
            self.C_occ,
            params=self.params,
            spin=spin,
        )
        return log_sd + f_pinn.squeeze(-1)


class SlaterOnlyWF(nn.Module):
    """Reference wavefunction using only the closed-shell Slater determinant."""

    def __init__(self, n_particles, dim, omega, C_occ, spin, params, well_sep: float = 0.0):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.omega = omega
        self.register_buffer("C_occ", C_occ)
        self.register_buffer("spin", spin)
        self.params = params
        self.well_sep = well_sep

    def forward(self, x):
        B = x.shape[0]
        spin = self.spin.expand(B, -1)
        x_eff = x
        # For separated dots, recenter each electron around its nearest well
        # so the reference determinant remains physically sensible without VMC.
        if self.well_sep > 1e-10 and self.n_particles >= 2 and self.dim >= 1:
            x_eff = x.clone()
            x_eff[:, 0, 0] = x_eff[:, 0, 0] + self.well_sep / 2
            x_eff[:, 1, 0] = x_eff[:, 1, 0] - self.well_sep / 2
        _, log_sd = slater_determinant_closed_shell(
            x_eff,
            self.C_occ,
            params=self.params,
            spin=spin,
        )
        return log_sd


def setup_sd(n_particles, dim, omega, E_ref):
    n_occ = n_particles // 2
    nx = max(2, int(math.ceil(math.sqrt(2 * n_occ))))
    ny = nx
    config.update(
        device=DEVICE,
        omega=omega,
        n_particles=n_particles,
        d=dim,
        dimensions=dim,
        basis="cart",
        nx=nx,
        ny=ny,
        E=E_ref,
    )
    params = config.get().as_dict()
    n_basis = nx * ny
    pairs = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    pairs.sort(key=lambda t: (t[0] + t[1], t[0]))
    sel = pairs[:n_occ]
    cols = [ix * ny + iy for (ix, iy) in sel]
    C_occ = torch.zeros(n_basis, n_occ, dtype=DTYPE, device=DEVICE)
    for j, c in enumerate(cols):
        C_occ[c, j] = 1.0
    spin = torch.cat(
        [
            torch.zeros(n_particles // 2, dtype=torch.long, device=DEVICE),
            torch.ones(n_particles - n_particles // 2, dtype=torch.long, device=DEVICE),
        ]
    )
    return C_occ, spin, params


# ============================================================
# VMC training (Phase 1)
# ============================================================
def train_vmc(cfg: PINNConfig, system_override=None):
    C_occ, spin, params = setup_sd(cfg.n_particles, cfg.dim, cfg.omega, cfg.E_ref)
    wf = GroundStateWF(
        cfg.n_particles,
        cfg.dim,
        cfg.omega,
        C_occ,
        spin,
        params,
        pinn_hidden=cfg.pinn_hidden,
        pinn_layers=cfg.pinn_layers,
        bf_hidden=cfg.bf_hidden,
        bf_layers=cfg.bf_layers,
        use_backflow=cfg.use_backflow,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in wf.parameters() if p.requires_grad)
    print(f"  Ground-state model: {n_params:,} parameters")

    optimizer = optim.Adam(wf.parameters(), lr=cfg.lr_vmc)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.n_epochs_vmc, eta_min=cfg.lr_vmc / 10
    )

    print_every = max(cfg.n_epochs_vmc // 8, 1)
    best_E, best_state = float("inf"), None

    for epoch in range(cfg.n_epochs_vmc):
        with torch.no_grad():
            x, acc = mcmc_sample(
                wf, cfg.n_samples_vmc, cfg.n_particles, cfg.dim, cfg.omega, cfg.well_sep, 200, 0.4
            )
        x_g = x.detach().requires_grad_(True)
        lp = wf(x_g)
        grad = torch.autograd.grad(lp.sum(), x_g, create_graph=True)[0]
        B, N, d = x_g.shape
        lap = torch.zeros(B, device=DEVICE, dtype=DTYPE)
        for i in range(N):
            for j in range(d):
                g2 = torch.autograd.grad(
                    grad[:, i, j].sum(), x_g, create_graph=True, retain_graph=True
                )[0]
                lap += g2[:, i, j]
        T = -0.5 * (lap + (grad**2).sum(dim=(1, 2)))
        spin_batch = _spin_batch_for_model(wf, B)
        V = _compute_potential_for_cfg(
            x_g,
            cfg,
            spin_batch=spin_batch,
            magnetic_B=cfg.magnetic_B_initial,
            system_override=system_override,
        )
        E_L = T + V
        E_mean = E_L.mean()
        loss = ((E_L - E_mean.detach()) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wf.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        E_val = E_mean.item()
        if E_val < best_E:
            best_E = E_val
            best_state = {k: v.clone() for k, v in wf.state_dict().items()}
        if epoch % print_every == 0:
            E_err = E_L.detach().std().item() / math.sqrt(cfg.n_samples_vmc)
            print(
                f"    [VMC] Ep {epoch:4d} | E={E_val:.5f} +/- {E_err:.5f} | "
                f"var={loss.item():.4f} | acc={acc:.2f}"
            )

    if best_state is not None:
        wf.load_state_dict(best_state)

    with torch.no_grad():
        x, _ = mcmc_sample(wf, 2000, cfg.n_particles, cfg.dim, cfg.omega, cfg.well_sep, 500, 0.4)
    x_g = x.detach().requires_grad_(True)
    lp = wf(x_g)
    grad = torch.autograd.grad(lp.sum(), x_g, create_graph=True)[0]
    B = x_g.shape[0]
    lap = torch.zeros(B, device=DEVICE, dtype=DTYPE)
    for i in range(cfg.n_particles):
        for j in range(cfg.dim):
            g2 = torch.autograd.grad(
                grad[:, i, j].sum(), x_g, create_graph=True, retain_graph=True
            )[0]
            lap += g2[:, i, j]
    T = -0.5 * (lap + (grad**2).sum(dim=(1, 2)))
    spin_batch = _spin_batch_for_model(wf, B)
    V = _compute_potential_for_cfg(
        x_g,
        cfg,
        spin_batch=spin_batch,
        magnetic_B=cfg.magnetic_B_initial,
        system_override=system_override,
    )
    E_L = T + V
    E_final = E_L.detach().mean().item()
    E_err = E_L.detach().std().item() / math.sqrt(2000)
    print(f"  VMC converged: E_0 = {E_final:.5f} +/- {E_err:.5f}  (ref={cfg.E_ref:.5f})")
    return wf, E_final


def estimate_energy(
    wf: nn.Module,
    cfg: PINNConfig,
    n_eval: int = 2000,
    system_override=None,
) -> tuple[float, float]:
    """Estimate local energy and error bar for a provided reference wf."""
    with torch.no_grad():
        init_x = _build_initial_walkers(
            n_eval,
            cfg.n_particles,
            cfg.dim,
            cfg.omega,
            cfg.well_sep,
            system_override=system_override,
        )
        x, _ = mcmc_sample(
            wf,
            n_eval,
            cfg.n_particles,
            cfg.dim,
            cfg.omega,
            cfg.well_sep,
            500,
            0.4,
            init_x=init_x,
        )
    x_g = x.detach().requires_grad_(True)
    lp = wf(x_g)
    grad = torch.autograd.grad(lp.sum(), x_g, create_graph=True)[0]
    B = x_g.shape[0]
    lap = torch.zeros(B, device=DEVICE, dtype=DTYPE)
    for i in range(cfg.n_particles):
        for j in range(cfg.dim):
            g2 = torch.autograd.grad(
                grad[:, i, j].sum(), x_g, create_graph=True, retain_graph=True
            )[0]
            lap += g2[:, i, j]
    T = -0.5 * (lap + (grad**2).sum(dim=(1, 2)))
    spin_batch = _spin_batch_for_model(wf, B)
    V = _compute_potential_for_cfg(
        x_g,
        cfg,
        spin_batch=spin_batch,
        magnetic_B=cfg.magnetic_B_initial,
        system_override=system_override,
    )
    E_L = T + V
    E_mean = E_L.detach().mean().item()
    E_err = E_L.detach().std().item() / math.sqrt(n_eval)
    return E_mean, E_err


# ============================================================
# Pre-compute ground-state quantities (Phase 2)
# ============================================================
def precompute_ground_state(ground_wf: nn.Module, cfg: PINNConfig, system_override=None) -> dict:
    """Sample x ~ |ψ_0|² and compute E_L^(0), ∇logψ_0 once."""
    n = cfg.n_precompute
    print(f"  Pre-computing on {n} walkers...")

    with torch.no_grad():
        init_x = _build_initial_walkers(
            n,
            cfg.n_particles,
            cfg.dim,
            cfg.omega,
            cfg.well_sep,
            system_override=system_override,
        )
        x, acc = mcmc_sample(
            ground_wf,
            n,
            cfg.n_particles,
            cfg.dim,
            cfg.omega,
            cfg.well_sep,
            500,
            0.4,
            init_x=init_x,
        )

    return _compute_precomputed_quantities(
        x,
        ground_wf,
        cfg,
        system_override=system_override,
        acc=acc,
        ess=None,
        sampler_name="mcmc",
        clip_width=cfg.precompute_local_energy_clip_width,
    )



def precompute_ground_state_stratified(
    ground_wf: nn.Module,
    cfg: PINNConfig,
    system_override=None,
    *,
    eval_mode: bool = False,
    store_sampling_density: bool = False,
) -> dict:
    """Sample i.i.d. collocation points and compute E_L^(0), ∇logψ_0 once."""
    n = cfg.n_precompute
    print(f"  Pre-computing on {n} walkers with stratified i.i.d. sampler...")

    system = _precompute_system(cfg, system_override)
    sampler_kwargs = _stratified_sampler_kwargs(cfg, eval_mode=eval_mode)
    with torch.no_grad():
        x, ess = stratified_resample(
            n_keep=n,
            omega=cfg.omega,
            system=system,
            device=DEVICE,
            dtype=DTYPE,
            **sampler_kwargs,
        )

        extra_fields = None
        if store_sampling_density:
            extra_fields = {
                "log_q": stratified_logpdf(
                    x,
                    omega=cfg.omega,
                    system=system,
                    **sampler_kwargs,
                ).detach()
            }

    return _compute_precomputed_quantities(
        x,
        ground_wf,
        cfg,
        system_override=system_override,
        acc=None,
        ess=ess,
        sampler_name="stratified",
        clip_width=(cfg.eval_local_energy_clip_width if eval_mode else cfg.precompute_local_energy_clip_width),
        extra_fields=extra_fields,
    )


# ============================================================
# FiLM-conditioned τ network  g_θ(x, τ)  (general, retained as option)
# ============================================================
class TauEmbedding(nn.Module):
    """Fourier feature embedding for τ."""

    def __init__(self, embed_dim=32, n_freq=6):
        super().__init__()
        freqs = torch.logspace(math.log10(0.5), math.log10(20.0), n_freq, dtype=DTYPE)
        self.register_buffer("freqs", freqs)
        input_dim = 2 * n_freq + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, tau):
        t = tau.unsqueeze(-1)
        phases = t * self.freqs
        embed = torch.cat([t, torch.sin(phases), torch.cos(phases)], dim=-1)
        return self.net(embed)


class TauConditionedG(nn.Module):
    """FiLM-conditioned spatial network (general τ-dependence)."""

    def __init__(self, n_particles, dim, hidden=64, n_layers=3, tau_embed=32, n_freq=6, **kw):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        input_dim = n_particles * dim
        self.spatial_proj = nn.Linear(input_dim, hidden)
        self.tau_embed = TauEmbedding(tau_embed, n_freq)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.film_heads = nn.ModuleList([nn.Linear(tau_embed, 2 * hidden) for _ in range(n_layers)])
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )
        nn.init.normal_(self.output[-1].weight, std=0.01)
        nn.init.zeros_(self.output[-1].bias)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  g_θ FiLM: {n_params:,} parameters")

    def forward(self, x, tau):
        B = x.shape[0]
        h = F.gelu(self.spatial_proj(x.reshape(B, -1)))
        tau_emb = self.tau_embed(tau)
        for linear, norm, film_head in zip(self.layers, self.norms, self.film_heads, strict=False):
            gb = film_head(tau_emb)
            gamma, beta = gb.chunk(2, dim=-1)
            gamma = 1.0 + 0.1 * gamma
            h = linear(h)
            h = norm(h)
            h = gamma * h + beta
            h = F.gelu(h)
        return self.output(h).squeeze(-1)


# ============================================================
# Spectral decomposition network  (physics-informed, preferred)
# ============================================================
class SpectralG(nn.Module):
    """g(x, τ) = f_0(x) + Σ_{k=1}^K f_k(x) · exp(-α_k τ)

    Physics-informed spectral decomposition:
      - f_0(x): constant mode (permanent ground-state correction)
      - f_k(x): decaying spatial MLPs (transient excited-state modes)
      - α_k: learnable decay rates → gap_k = α_k directly!
      - τ-derivative is analytical (no autograd for τ)
      - Ordered: α_1 < α_2 < ... by construction

    At τ→∞:  g → f_0(x), the steady-state correction to ψ₀
    At τ=0:   g = f_0(x) + Σ f_k(x), enforced ≈ 0 by IC loss
    """

    def __init__(self, n_particles, dim, n_modes=3, hidden=32, n_layers=2,
                 const_hidden=None, const_layers=None, **kw):
        super().__init__()
        self.n_modes = n_modes
        self.n_particles = n_particles
        self.dim = dim
        input_dim = n_particles * dim

        # Constant mode f_0(x) — the permanent ground-state correction
        # Intentionally small: the correction is smooth and we need to
        # average over E_L noise, not memorise it.
        ch = const_hidden or max(hidden // 3, 16)
        cl = const_layers if const_layers is not None else 1
        clayers = [nn.Linear(input_dim, ch), nn.GELU()]
        for _ in range(cl - 1):
            clayers.extend([nn.Linear(ch, ch), nn.GELU()])
        clayers.append(nn.Linear(ch, 1))
        nn.init.zeros_(clayers[-1].weight)
        nn.init.zeros_(clayers[-1].bias)
        self.const_net = nn.Sequential(*clayers)

        # Decaying modes f_k(x) · exp(-α_k τ)
        self.mode_nets = nn.ModuleList()
        for k in range(n_modes):
            layers_list = [nn.Linear(input_dim, hidden), nn.GELU()]
            for _ in range(n_layers - 1):
                layers_list.extend([nn.Linear(hidden, hidden), nn.GELU()])
            layers_list.append(nn.Linear(hidden, 1))
            net = nn.Sequential(*layers_list)
            # Small init for last layer
            nn.init.normal_(net[-1].weight, std=0.05)
            nn.init.zeros_(net[-1].bias)
            self.mode_nets.append(net)

        # Ordered rates: α_k = α_base + Σ_{j≤k} softplus(δ_j)
        self.alpha_base = nn.Parameter(torch.tensor(0.5, dtype=DTYPE))
        self.alpha_deltas = nn.Parameter(torch.zeros(n_modes, dtype=DTYPE))

    def _get_alphas(self):
        """Get ordered rates α_1 < α_2 < ..."""
        deltas = F.softplus(self.alpha_deltas) + 0.1  # min spacing 0.1
        alphas = torch.zeros(self.n_modes, dtype=DTYPE, device=self.alpha_base.device)
        alphas[0] = F.softplus(self.alpha_base) + 0.1
        for k in range(1, self.n_modes):
            alphas[k] = alphas[k - 1] + deltas[k]
        return alphas

    def forward(self, x, tau):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        alphas = self._get_alphas()
        g = self.const_net(x_flat).squeeze(-1)  # constant mode f_0(x)
        for k in range(self.n_modes):
            fk = self.mode_nets[k](x_flat).squeeze(-1)
            g = g + fk * torch.exp(-alphas[k] * tau)
        return g

    def forward_decomposed(self, x, tau):
        """Return per-mode contributions.

        Returns list of (fk, decay, alpha_k) where:
          - mode 0: (f_0, 1.0, 0.0) — constant mode
          - mode k>0: (f_k, exp(-α_k τ), α_k) — decaying modes
        """
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        alphas = self._get_alphas()
        modes = []
        # Constant mode: decay=1, alpha=0 (∂/∂τ contribution = 0)
        f0 = self.const_net(x_flat).squeeze(-1)
        ones = torch.ones(B, device=x.device, dtype=x.dtype)
        zero_alpha = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        modes.append((f0, ones, zero_alpha))
        # Decaying modes
        for k in range(self.n_modes):
            fk = self.mode_nets[k](x_flat).squeeze(-1)
            decay = torch.exp(-alphas[k] * tau)
            modes.append((fk, decay, alphas[k]))
        return modes

    def get_gaps(self):
        return self._get_alphas().detach()

    def get_const_mode_rms(self, x):
        """RMS of f_0(x) on given samples."""
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        with torch.no_grad():
            f0 = self.const_net(x_flat).squeeze(-1)
        return f0.pow(2).mean().sqrt().item()

    def count_params(self):
        n = sum(p.numel() for p in self.parameters())
        n_const = sum(p.numel() for p in self.const_net.parameters())
        print(f"  g_θ Spectral ({self.n_modes} modes + constant): "
              f"{n:,} parameters ({n_const:,} in f_0)")
        return n


# ============================================================
# Efficient PDE residual for SpectralG
# ============================================================
def compute_pde_residual_spectral(
    g_net: SpectralG, x, tau, E_L0, grad_log_psi0, E_ref, deltaV=None
):
    """PDE residual for spectral form. τ-derivative is analytical.

    For g = f_0(x) + Σ_k f_k · exp(-α_k τ):
      ∂_τ g = -Σ_k α_k f_k exp(-α_k τ)  (constant mode contributes 0)
      ∇g = ∇f_0 + Σ_k ∇f_k exp(-α_k τ)
      ∇²g = ∇²f_0 + Σ_k ∇²f_k exp(-α_k τ)
    """
    B, N, d = x.shape
    x = x.detach().requires_grad_(True)

    modes = g_net.forward_decomposed(x, tau)

    # Accumulate analytical τ-derivative and spatial derivatives
    dg_dtau = torch.zeros(B, device=x.device, dtype=x.dtype)
    grad_g = torch.zeros(B, N, d, device=x.device, dtype=x.dtype)
    lap_g = torch.zeros(B, device=x.device, dtype=x.dtype)
    g_total = torch.zeros(B, device=x.device, dtype=x.dtype)

    for fk, decay_k, alpha_k in modes:
        gk = fk * decay_k  # (B,)
        g_total = g_total + gk

        # τ-derivative: analytical
        dg_dtau = dg_dtau - alpha_k * gk

        # Spatial derivatives: autograd on f_k
        grad_fk = torch.autograd.grad(fk.sum(), x, create_graph=True, retain_graph=True)[
            0
        ]  # (B, N, d)
        grad_g = grad_g + grad_fk * decay_k.unsqueeze(-1).unsqueeze(-1)

        # Laplacian of f_k
        for i in range(N):
            for j in range(d):
                d2fk = torch.autograd.grad(
                    grad_fk[:, i, j].sum(), x, create_graph=True, retain_graph=True
                )[0]
                lap_g = lap_g + d2fk[:, i, j] * decay_k

    # Cross term: ∇logψ_0 · ∇g
    cross = (grad_log_psi0 * grad_g).sum(dim=(1, 2))
    # |∇g|²
    gradg_sq = (grad_g**2).sum(dim=(1, 2))

    if deltaV is None:
        deltaV = torch.zeros_like(E_L0)

    # E_L (post-quench H) = E_L^(0) + ΔV - ½∇²g - ∇logψ_0·∇g - ½|∇g|²
    E_L = E_L0 + deltaV - 0.5 * lap_g - cross - 0.5 * gradg_sq

    # Residual: ∂_τ g + (E_L - E_ref) = 0
    residual = dg_dtau + (E_L - E_ref)

    return residual, g_total, E_L


# ============================================================
# Initial-condition perturbation p(x)
# ============================================================
def make_perturbation_fn(cfg: PINNConfig):
    """Return p(x) target for g(x, 0)."""
    A = cfg.ic_amplitude
    if cfg.ic_type == "dipole":

        def p(x):
            return A * x[:, :, 0].mean(dim=1)  # X_cm

        return p, "dipole (X_cm)"
    elif cfg.ic_type == "quadrupole":

        def p(x):
            return A * (x**2).sum(dim=(1, 2)) / cfg.n_particles  # R²/N

        return p, "quadrupole (R²/N)"
    elif cfg.ic_type == "random":
        # Frozen random network as perturbation
        rand_net = (
            nn.Sequential(
                nn.Linear(cfg.n_particles * cfg.dim, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
            .to(DEVICE)
            .to(DTYPE)
        )
        for p_param in rand_net.parameters():
            p_param.requires_grad_(False)

        def p(x):
            return A * rand_net(x.reshape(x.shape[0], -1)).squeeze(-1)

        return p, "random network"
    else:
        raise ValueError(f"Unknown IC type: {cfg.ic_type}")


# ============================================================
# PDE residual computation
# ============================================================
def compute_pde_residual(g_net, x, tau, E_L0, grad_log_psi0, E_ref, deltaV=None):
    """Compute PDE residual, g values, and E_L for a batch.

    PDE: ∂_τ g + (E_L - E_0) = 0
    where E_L = E_L^(0) - ½∇²g - ∇logψ_0·∇g - ½|∇g|²

    Returns: residual (B,), g (B,), E_L (B,)
    """
    B, N, d = x.shape

    x = x.detach().requires_grad_(True)
    tau = tau.detach().requires_grad_(True)

    g = g_net(x, tau)

    # ∂_τ g
    dg_dtau = torch.autograd.grad(g.sum(), tau, create_graph=True, retain_graph=True)[0]

    # ∇_x g
    dg_dx = torch.autograd.grad(g.sum(), x, create_graph=True, retain_graph=True)[0]  # (B, N, d)

    # ∇²_x g  (exact Laplacian — 4 backward passes for N=2, d=2)
    lap_g = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            d2g = torch.autograd.grad(
                dg_dx[:, i, j].sum(), x, create_graph=True, retain_graph=True
            )[0]
            lap_g = lap_g + d2g[:, i, j]

    # Cross term: ∇logψ_0 · ∇g
    cross = (grad_log_psi0 * dg_dx).sum(dim=(1, 2))

    # |∇g|²
    gradg_sq = (dg_dx**2).sum(dim=(1, 2))

    if deltaV is None:
        deltaV = torch.zeros_like(E_L0)

    # E_L(x, τ; H_quench) = E_L^(0) + ΔV - ½∇²g - ∇logψ_0·∇g - ½|∇g|²
    E_L = E_L0 + deltaV - 0.5 * lap_g - cross - 0.5 * gradg_sq

    # Residual: ∂_τ g + (E_L - E_ref) = 0
    residual = dg_dtau + (E_L - E_ref)

    return residual, g, E_L


# ============================================================
# Phase 3: Train g_θ on PDE residual
# ============================================================
def train_pinn(cfg: PINNConfig, ground_wf: GroundStateWF, precomputed: dict, E_ref: float):
    """Train spectral g_θ(x, τ) = Σ_k f_k(x)·exp(-α_k τ) on PDE losses."""

    for p in ground_wf.parameters():
        p.requires_grad_(False)

    use_spectral = getattr(cfg, "use_spectral", True)

    if use_spectral:
        g_net = SpectralG(
            cfg.n_particles,
            cfg.dim,
            n_modes=getattr(cfg, "g_modes", 3),
            hidden=cfg.g_hidden,
            n_layers=cfg.g_layers,
        ).to(DEVICE)
        g_net.count_params()
        pde_fn = compute_pde_residual_spectral
    else:
        g_net = TauConditionedG(
            cfg.n_particles,
            cfg.dim,
            hidden=cfg.g_hidden,
            n_layers=cfg.g_layers,
            tau_embed=cfg.g_tau_embed,
            n_freq=cfg.g_n_freq,
        ).to(DEVICE)
        pde_fn = compute_pde_residual

    perturbation_fn, ic_desc = make_perturbation_fn(cfg)
    print(f"  Initial condition: {ic_desc}, amplitude={cfg.ic_amplitude}")
    print(
        "  PINN styles: "
        f"loss={cfg.loss_style}, tau_sampling={cfg.tau_sampling}, x_sampling={cfg.x_sampling}"
    )

    optimizer = optim.Adam(g_net.parameters(), lr=cfg.lr_pde)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.n_epochs_pde, eta_min=cfg.lr_pde / 30
    )

    x_pool = precomputed["x"]
    E_L0_pool = precomputed["E_L0"]
    deltaV_pool = precomputed.get("deltaV", torch.zeros_like(E_L0_pool))
    grad_pool = precomputed["grad_log_psi0"]
    pool_size = x_pool.shape[0]

    if cfg.x_sampling == "energy_weighted":
        with torch.no_grad():
            w = (E_L0_pool - E_L0_pool.mean()).abs() + 1e-8
            x_sample_probs = w / w.sum()
    elif cfg.x_sampling == "uniform":
        x_sample_probs = None
    else:
        raise ValueError(f"Unknown x_sampling: {cfg.x_sampling}")

    print_every = max(cfg.n_epochs_pde // 20, 1)
    history = {"pde": [], "ic": [], "bc": [], "total": [], "gaps": []}

    for epoch in range(cfg.n_epochs_pde):
        # --- Random batch ---
        if x_sample_probs is None:
            idx = torch.randint(pool_size, (cfg.batch_pde,))
        else:
            idx = torch.multinomial(x_sample_probs, cfg.batch_pde, replacement=True)
        x_batch = x_pool[idx]
        E_L0_batch = E_L0_pool[idx]
        deltaV_batch = deltaV_pool[idx]
        grad_batch = grad_pool[idx]

        # --- Random τ sampling ---
        if cfg.tau_sampling == "small_bias":
            tau_pde = (
                torch.rand(cfg.batch_pde, dtype=DTYPE, device=DEVICE)
                ** cfg.tau_small_bias_power
                * cfg.tau_max
            )
        elif cfg.tau_sampling == "uniform":
            tau_pde = torch.rand(cfg.batch_pde, dtype=DTYPE, device=DEVICE) * cfg.tau_max
        elif cfg.tau_sampling == "two_stage":
            tau_small = (
                torch.rand(cfg.batch_pde, dtype=DTYPE, device=DEVICE)
                ** cfg.tau_small_bias_power
                * cfg.tau_max
            )
            tau_uniform = torch.rand(cfg.batch_pde, dtype=DTYPE, device=DEVICE) * cfg.tau_max
            chooser = torch.rand(cfg.batch_pde, dtype=DTYPE, device=DEVICE)
            tau_pde = torch.where(chooser < 0.7, tau_small, tau_uniform)
        else:
            raise ValueError(f"Unknown tau_sampling: {cfg.tau_sampling}")

        # --- PDE loss ---
        residual, g_pde, E_L = pde_fn(
            g_net, x_batch, tau_pde, E_L0_batch, grad_batch, E_ref, deltaV_batch
        )
        if cfg.loss_style in {"curriculum_mse", "mse"}:
            L_pde = (residual**2).mean()
        elif cfg.loss_style == "huber":
            L_pde = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=0.1)
        elif cfg.loss_style == "logcosh":
            L_pde = (F.softplus(2.0 * residual) - residual - math.log(2.0)).mean()
        else:
            raise ValueError(f"Unknown loss_style: {cfg.loss_style}")

        # --- IC loss: g(x, 0) ≈ p(x) ---
        tau_zero = torch.zeros(cfg.batch_pde, dtype=DTYPE, device=DEVICE)
        g_ic = g_net(x_batch, tau_zero)
        p_target = perturbation_fn(x_batch)
        L_ic = ((g_ic - p_target) ** 2).mean()

        # --- BC is automatic for Spectral (exp(-α τ_max) ≈ 0) ---
        tau_end = torch.full((cfg.batch_pde,), cfg.tau_max, dtype=DTYPE, device=DEVICE)
        g_bc = g_net(x_batch, tau_end)
        L_bc = (g_bc**2).mean()

        # --- Regularisation ---
        L_reg = (g_pde**2).mean()

        # --- Optional curriculum ---
        if cfg.loss_style == "curriculum_mse":
            progress = epoch / max(cfg.n_epochs_pde - 1, 1)
            w_ic = cfg.lambda_ic * max(1.0 - 0.5 * progress, 0.3)
            w_pde = 1.0 + 9.0 * min(progress * 2, 1.0)  # ramp to 10x PDE weight
        else:
            w_ic = cfg.lambda_ic
            w_pde = 1.0

        loss = w_pde * L_pde + w_ic * L_ic + cfg.lambda_bc * L_bc + cfg.lambda_reg * L_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % print_every == 0:
            history["pde"].append(L_pde.item())
            history["ic"].append(L_ic.item())
            history["bc"].append(L_bc.item())
            history["total"].append(loss.item())
            g0_rms = g_ic.detach().pow(2).mean().sqrt().item()
            gT_rms = g_bc.detach().pow(2).mean().sqrt().item()
            gap_str = ""
            if use_spectral:
                gaps = g_net.get_gaps()
                history["gaps"].append(gaps.tolist())
                gap_str = f" | gaps=[{', '.join(f'{g:.4f}' for g in gaps)}]"
            print(
                f"    [PINN] Ep {epoch:5d} | PDE={L_pde.item():.6f} "
                f"IC={L_ic.item():.6f} BC={L_bc.item():.6f} | "
                f"g(0)_rms={g0_rms:.4f} g(T)_rms={gT_rms:.4f}{gap_str}"
            )

    if use_spectral:
        gaps = g_net.get_gaps()
        print(f"\n  *** LEARNED GAPS (direct) = [{', '.join(f'{g:.5f}' for g in gaps)}] ***")
    print(f"  Final PDE loss: {history['pde'][-1]:.6f}")
    return g_net, history


# ============================================================
# Phase 4: Evaluate E(τ) trajectory
# ============================================================
def evaluate_trajectory(g_net, precomputed: dict, cfg: PINNConfig, E_ref: float):
    """Evaluate E(τ) using proposal-aware self-normalized importance sampling."""
    g_net.eval()

    # Use evaluation samples (larger pool)
    with torch.no_grad():
        x_eval = precomputed["x"][: cfg.n_samples_eval]
        log_psi0_eval = precomputed.get("log_psi0")
        E_L0_eval = precomputed["E_L0"][: cfg.n_samples_eval]
        deltaV_eval = precomputed.get("deltaV", torch.zeros_like(precomputed["E_L0"]))[
            : cfg.n_samples_eval
        ]
        grad_eval = precomputed["grad_log_psi0"][: cfg.n_samples_eval]
        log_q_eval = precomputed.get("log_q")

    if log_psi0_eval is not None:
        log_psi0_eval = log_psi0_eval[: cfg.n_samples_eval]
    sampler_name = str(precomputed.get("sampler", "mcmc"))
    if sampler_name == "stratified" and (log_psi0_eval is None or log_q_eval is None):
        raise ValueError(
            "Stratified evaluation requires log_psi0 and log_q in the precomputed pool."
        )
    if log_q_eval is not None:
        log_q_eval = log_q_eval[: cfg.n_samples_eval]

    n = x_eval.shape[0]
    tau_values = np.concatenate([[0.0], np.geomspace(0.01, cfg.tau_max, cfg.n_tau_eval - 1)])
    results = []

    for tau_val in tau_values:
        tau_t = torch.full((n,), tau_val, dtype=DTYPE, device=DEVICE)

        # Need ∇g and ∇²g for E_L — use no_grad where possible
        x_e = x_eval.detach().requires_grad_(True)

        g = g_net(x_e, tau_t)
        dg_dx = torch.autograd.grad(g.sum(), x_e, create_graph=True)[0]

        N, d = cfg.n_particles, cfg.dim
        lap_g = torch.zeros(n, device=DEVICE, dtype=DTYPE)
        for i in range(N):
            for j in range(d):
                d2g = torch.autograd.grad(dg_dx[:, i, j].sum(), x_e, retain_graph=True)[0]
                lap_g += d2g[:, i, j]

        cross = (grad_eval * dg_dx.detach()).sum(dim=(1, 2))
        gradg_sq = (dg_dx.detach() ** 2).sum(dim=(1, 2))
        E_L = E_L0_eval + deltaV_eval - 0.5 * lap_g.detach() - cross - 0.5 * gradg_sq

        g_vals = g.detach()
        if sampler_name == "stratified":
            log_w = 2.0 * (log_psi0_eval + g_vals) - log_q_eval
        else:
            # For MCMC / |psi_0|^2 proposals, the Radon-Nikodym factor reduces to exp(2g).
            log_w = 2.0 * g_vals
        log_w = log_w - log_w.max()
        w = torch.exp(log_w)
        w = w / w.sum()
        n_eff = 1.0 / (w**2).sum().item()

        E_L_np = E_L.detach().cpu().numpy()
        w_np = w.detach().cpu().numpy()
        E_mean = float(np.sum(w_np * E_L_np))
        E_var = float(np.sum(w_np * (E_L_np - E_mean) ** 2))
        E_err = float(np.sqrt(E_var / max(n_eff, 1.0)))

        results.append(
            {
                "tau": float(tau_val),
                "E": E_mean,
                "E_std": np.sqrt(E_var),
                "E_err": E_err,
                "n_eff": n_eff,
                "g_rms": float(g_vals.pow(2).mean().sqrt()),
            }
        )

    g_net.train()
    return results


# ============================================================
# Fitting
# ============================================================
def fit_log_linear(traj, E0_fixed, tau_min=0.0, tau_max=None):
    """Robust gap extraction via log-linear fit: log(E-E0) = log(dE) - gamma*tau.
    Fixes E0 to remove one free parameter."""
    tau = np.array([r["tau"] for r in traj])
    E = np.array([r["E"] for r in traj])
    dE = E - E0_fixed
    mask = dE > 1e-6  # only positive deviations
    mask &= tau >= tau_min
    if tau_max is not None:
        mask &= tau <= tau_max
    if mask.sum() < 5:
        return {"success": False}
    t, y = tau[mask], np.log(dE[mask])
    # Weight by 1/E_err (propagated)
    E_err = np.clip(np.array([r["E_err"] for r in traj]), 1e-6, None)[mask]
    w = dE[mask] / E_err  # propagation: sigma_log = sigma_E / dE
    w = np.clip(w, 0.1, 100)
    # Weighted least squares: y = a + b*t
    W = np.diag(w**2)
    A = np.column_stack([np.ones_like(t), t])
    AtWA = A.T @ W @ A
    AtWy = A.T @ W @ y
    try:
        params = np.linalg.solve(AtWA, AtWy)
        cov = np.linalg.inv(AtWA)
    except np.linalg.LinAlgError:
        return {"success": False}
    gamma = -params[1]
    gamma_err = np.sqrt(cov[1, 1])
    return {
        "E0": E0_fixed,
        "gamma": gamma,
        "gamma_err": gamma_err,
        "gap": gamma / 2,
        "gap_err": gamma_err / 2,
        "log_dE": params[0],
        "method": "log-linear",
        "n_points": int(mask.sum()),
        "success": True,
    }


def fit_log_linear_windows(traj, E0_fixed, windows):
    """Run log-linear fits over multiple tau windows for stability diagnostics."""
    results = []
    for (tau_min, tau_max) in windows:
        fit = fit_log_linear(traj, E0_fixed, tau_min=tau_min, tau_max=tau_max)
        entry = {
            "tau_min": float(tau_min),
            "tau_max": float(tau_max),
            "success": bool(fit.get("success", False)),
        }
        if fit.get("success"):
            entry.update(
                {
                    "gap": float(fit["gap"]),
                    "gap_err": float(fit["gap_err"]),
                    "n_points": int(fit.get("n_points", 0)),
                    "gamma": float(fit.get("gamma", float("nan"))),
                }
            )
        results.append(entry)
    return results


def fit_single_exponential(traj, E_ref):
    tau = np.array([r["tau"] for r in traj])
    E = np.array([r["E"] for r in traj])
    E_err = np.clip(np.array([r["E_err"] for r in traj]), 1e-6, None)

    def model(t, E0, dE, gamma):
        return E0 + dE * np.exp(-gamma * t)

    try:
        dE_init = max(E[0] - E_ref, 0.01)
        p0 = [E_ref, dE_init, 2.0]
        bounds = ([E_ref - 0.5, 0, 0.01], [E_ref + 0.5, 20, 50])
        popt, pcov = curve_fit(model, tau, E, p0=p0, sigma=E_err, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return {
            "E0": popt[0],
            "E0_err": perr[0],
            "dE": popt[1],
            "dE_err": perr[1],
            "gamma": popt[2],
            "gamma_err": perr[2],
            "gap": popt[2] / 2,
            "gap_err": perr[2] / 2,
            "success": True,
        }
    except Exception as e:
        print(f"  Fit failed: {e}")
        return {"success": False}


def fit_restricted_exponential(traj, E_ref, tau_min=0.15, tau_max=3.0):
    """Nonlinear exponential fit restricted to medium-τ range.
    Avoids early-τ multi-exponential contamination and late-τ noise."""
    tau = np.array([r["tau"] for r in traj])
    E = np.array([r["E"] for r in traj])
    E_err = np.clip(np.array([r["E_err"] for r in traj]), 1e-6, None)
    mask = (tau >= tau_min) & (tau <= tau_max)
    if mask.sum() < 5:
        return {"success": False}
    t, e, sig = tau[mask], E[mask], E_err[mask]

    def model(t, E0, dE, gamma):
        return E0 + dE * np.exp(-gamma * t)

    try:
        dE_init = max(e[0] - E_ref, 0.01)
        p0 = [E_ref, dE_init, 2.0]
        bounds = ([E_ref - 0.5, 0, 0.01], [E_ref + 0.5, 20, 50])
        popt, pcov = curve_fit(model, t, e, p0=p0, sigma=sig, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return {
            "E0": popt[0],
            "E0_err": perr[0],
            "dE": popt[1],
            "dE_err": perr[1],
            "gamma": popt[2],
            "gamma_err": perr[2],
            "gap": popt[2] / 2,
            "gap_err": perr[2] / 2,
            "tau_range": (tau_min, tau_max),
            "n_points": int(mask.sum()),
            "success": True,
        }
    except Exception as e:
        return {"success": False}


def fit_best_estimate(fits_dict, expected=None):
    """Combine multiple gap estimates into a robust best estimate.
    Uses inverse-variance weighting of successful fits."""
    gaps, ivars = [], []
    for name, fit in fits_dict.items():
        if not fit.get("success") or "gap" not in fit or "gap_err" not in fit:
            continue
        g, ge = fit["gap"], fit["gap_err"]
        if ge > 0 and ge < 0.5:  # sanity check on error bar
            gaps.append(g)
            ivars.append(1.0 / ge**2)
    if len(gaps) < 2:
        return {"success": False}
    gaps, ivars = np.array(gaps), np.array(ivars)
    w = ivars / ivars.sum()
    gap_mean = np.sum(w * gaps)
    # Error: combine statistical + spread
    gap_stat_err = 1.0 / np.sqrt(np.sum(ivars))
    gap_spread = np.sqrt(np.sum(w * (gaps - gap_mean) ** 2))
    gap_err = max(gap_stat_err, gap_spread)
    return {
        "gap": gap_mean,
        "gap_err": gap_err,
        "n_methods": len(gaps),
        "individual_gaps": gaps.tolist(),
        "method": "ensemble",
        "success": True,
    }


def fit_optimal_E0(traj, E_ref):
    """Find E0 that gives best linear fit in log(E-E0) space.
    Scan over E0 values and pick the one minimizing curvature."""
    tau = np.array([r["tau"] for r in traj])
    E = np.array([r["E"] for r in traj])
    E_err = np.clip(np.array([r["E_err"] for r in traj]), 1e-6, None)
    E_min = E[-1]  # upper bound for E0
    best = {"success": False}
    best_residual = float("inf")

    for E0_trial in np.linspace(E_ref - 0.001, E_min - 1e-5, 200):
        dE = E - E0_trial
        mask = (dE > 1e-5) & (tau > 0.03) & (tau < 3.5)
        if mask.sum() < 8:
            continue
        t, y = tau[mask], np.log(dE[mask])
        w = dE[mask] / E_err[mask]
        w = np.clip(w, 0.1, 100)
        W = np.diag(w**2)
        A = np.column_stack([np.ones_like(t), t])
        try:
            params = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
            pred = A @ params
            res = np.sum(w**2 * (y - pred) ** 2) / mask.sum()
            if res < best_residual:
                best_residual = res
                cov = np.linalg.inv(A.T @ W @ A)
                gamma = -params[1]
                gamma_err = np.sqrt(cov[1, 1])
                best = {
                    "E0": E0_trial,
                    "gamma": gamma,
                    "gamma_err": gamma_err,
                    "gap": gamma / 2,
                    "gap_err": gamma_err / 2,
                    "residual": res,
                    "n_points": int(mask.sum()),
                    "method": "optimal_E0",
                    "success": True,
                }
        except np.linalg.LinAlgError:
            continue
    return best


def fit_double_exponential(traj, E_ref):
    tau = np.array([r["tau"] for r in traj])
    E = np.array([r["E"] for r in traj])
    E_err = np.clip(np.array([r["E_err"] for r in traj]), 1e-6, None)

    def model(t, E0, dE1, g1, dE2, g2):
        return E0 + dE1 * np.exp(-g1 * t) + dE2 * np.exp(-g2 * t)

    try:
        dE_init = max(E[0] - E_ref, 0.01)
        p0 = [E_ref, dE_init * 0.7, 2.0, dE_init * 0.3, 4.0]
        bounds = ([E_ref - 0.5, 0, 0.01, 0, 0.01], [E_ref + 0.5, 20, 50, 20, 50])
        popt, pcov = curve_fit(model, tau, E, p0=p0, sigma=E_err, bounds=bounds, maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        g1, g2 = sorted([popt[2], popt[4]])
        return {
            "E0": popt[0],
            "gap1": g1 / 2,
            "gap2": g2 / 2,
            "success": True,
        }
    except Exception:
        return {"success": False}


# ============================================================
# Plotting
# ============================================================
def plot_results(traj, fit_s, fit_d, cfg, E_ref, save_dir, tag=""):
    tau = np.array([r["tau"] for r in traj])
    E = np.array([r["E"] for r in traj])
    E_err = np.array([r["E_err"] for r in traj])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: E(τ)
    ax = axes[0]
    ax.errorbar(tau, E, yerr=E_err, fmt="o", markersize=3, color="C0", label="PINN E(τ)")
    if fit_s.get("success"):
        t_fit = np.linspace(0, tau.max(), 200)
        E_fit = fit_s["E0"] + fit_s["dE"] * np.exp(-fit_s["gamma"] * t_fit)
        ax.plot(
            t_fit,
            E_fit,
            "--",
            color="C3",
            lw=2,
            label=f"Fit: gap={fit_s['gap']:.4f}±{fit_s['gap_err']:.4f}",
        )
    ax.axhline(E_ref, color="gray", ls=":", alpha=0.5, label=f"E_ref={E_ref:.3f}")
    ax.set_xlabel("τ")
    ax.set_ylabel("E(τ)")
    ax.set_title(f"d={cfg.well_sep:.1f}, ω={cfg.omega:.1f}, coul={'on' if cfg.coulomb else 'off'}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: log(E-E0)
    ax = axes[1]
    if fit_s.get("success"):
        dE = E - fit_s["E0"]
        mask = dE > 0
        if mask.sum() > 2:
            ax.semilogy(tau[mask], dE[mask], "o", ms=3, color="C0")
            t_fit = np.linspace(0, tau.max(), 200)
            dE_fit = fit_s["dE"] * np.exp(-fit_s["gamma"] * t_fit)
            ax.semilogy(t_fit, dE_fit, "--", color="C3", lw=2, label=f"γ = {fit_s['gamma']:.4f}")
            ax.legend(fontsize=8)
    ax.set_xlabel("τ")
    ax.set_ylabel("E(τ) - E₀")
    ax.set_title("Exponential decay")
    ax.grid(True, alpha=0.3)

    # Panel 3: n_eff and g_rms
    ax = axes[2]
    n_eff = [r.get("n_eff", 0) for r in traj]
    g_rms = [r.get("g_rms", 0) for r in traj]
    ax.plot(tau, n_eff, "o-", ms=3, color="C2", label="n_eff")
    ax.set_xlabel("τ")
    ax.set_ylabel("n_eff", color="C2")
    ax2 = ax.twinx()
    ax2.plot(tau, g_rms, "s-", ms=3, color="C1", label="g_rms")
    ax2.set_ylabel("g_rms", color="C1")
    ax.set_title("Diagnostics")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    name = f"pinn_{tag}d{cfg.well_sep:.1f}_w{cfg.omega:.1f}.png"
    path = save_dir / name
    plt.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"  Figure: {path}")


# ============================================================
# Run single configuration
# ============================================================
def run_single(cfg: PINNConfig, tag="") -> dict:
    if cfg.zeeman_electron1_only and cfg.zeeman_particle_indices is not None:
        raise ValueError(
            "zeeman_electron1_only and zeeman_particle_indices are mutually exclusive."
        )
    if cfg.zeeman_particle_indices is not None and len(cfg.zeeman_particle_indices) == 0:
        raise ValueError("zeeman_particle_indices must not be empty.")

    if cfg.well_sep_initial is None:
        cfg.well_sep_initial = float(cfg.well_sep)
    if cfg.well_sep_final is None:
        cfg.well_sep_final = float(cfg.well_sep)
    cfg.well_sep = float(cfg.well_sep_initial)

    # Seed all RNGs if seed is specified
    if cfg.seed is not None:
        import random
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        print(f"  [RNG] Seeded all generators with seed={cfg.seed}")

    coul_str = "interacting" if cfg.coulomb else "NON-interacting"
    print(f"\n{'='*65}")
    print(
        f"  {coul_str}: d_initial={cfg.well_sep_initial:.1f}, "
        f"d_final={cfg.well_sep_final:.1f}, ω={cfg.omega:.1f}, E_ref={cfg.E_ref:.5f}"
    )
    if abs(cfg.magnetic_B_initial) > 1e-14 or abs(cfg.magnetic_B) > 1e-14:
        if cfg.zeeman_particle_indices is not None:
            zmode = f"particles={list(cfg.zeeman_particle_indices)}"
        elif cfg.zeeman_electron1_only:
            zmode = "electron1"
        else:
            zmode = "all-electrons"
        print(
            f"  magnetic: B_initial={cfg.magnetic_B_initial:.4f} (VMC), B_evolution={cfg.magnetic_B:.4f} (PINN)"
        )
        print(
            f"            g={cfg.g_factor:.3f}, mu_B={cfg.mu_B:.3f}, zeeman={zmode}"
        )
    print(f"{'='*65}")

    system_override = None

    # Phase 1: reference state (locked GS artifact, trained VMC, or Slater-only)
    t0 = time.time()
    if cfg.ground_state_dir is not None:
        print("\n  Phase 1: Loading locked generalized ground state artifact...")
        ground_wf, system_override, gs_cfg = _load_locked_ground_state(cfg)
        gs_dir = Path(cfg.ground_state_dir).expanduser().resolve()
        cfg.n_particles = int(system_override.n_particles)
        cfg.dim = int(system_override.dim)
        cfg.omega = float(system_override.omega)
        cfg.coulomb = bool(system_override.coulomb)
        cfg.smooth_T = float(system_override.smooth_T)
        locked_sep = _system_max_well_separation(system_override)
        if cfg.well_sep_initial is None:
            cfg.well_sep_initial = locked_sep
        if cfg.well_sep_final is None:
            cfg.well_sep_final = locked_sep
        cfg.well_sep = float(cfg.well_sep_initial)
        locked_energy = _load_locked_ground_state_energy(gs_dir)
        if locked_energy is not None:
            E_vmc = locked_energy
            E_err = None
            print(
                f"  Loaded GS from {cfg.ground_state_dir} | "
                f"run_name={gs_cfg.get('run_name', 'unknown')} | "
                f"saved E = {E_vmc:.5f}"
            )
        else:
            E_vmc, E_err = estimate_energy(
                ground_wf,
                cfg,
                n_eval=2000,
                system_override=system_override,
            )
            print(
                f"  Loaded GS from {cfg.ground_state_dir} | "
                f"run_name={gs_cfg.get('run_name', 'unknown')} | "
                f"E = {E_vmc:.5f} +/- {E_err:.5f}"
            )
    elif cfg.no_vmc_train:
        print("\n  Phase 1: Slater-only reference (no VMC training)...")
        C_occ, spin, params = setup_sd(cfg.n_particles, cfg.dim, cfg.omega, cfg.E_ref)
        ground_wf = SlaterOnlyWF(
            cfg.n_particles,
            cfg.dim,
            cfg.omega,
            C_occ,
            spin,
            params,
            well_sep=cfg.well_sep,
        ).to(DEVICE)
        E_vmc, E_err = estimate_energy(ground_wf, cfg, n_eval=2000, system_override=None)
        print(f"  Reference energy: E = {E_vmc:.5f} +/- {E_err:.5f}")
    else:
        print("\n  Phase 1: VMC ground state...")
        ground_wf, E_vmc = train_vmc(cfg, system_override=None)
    t_vmc = time.time() - t0
    print(f"  Phase 1 time: {t_vmc:.0f}s")

    if cfg.no_vmc_train:
        if cfg.coulomb:
            E_ref = 3.0 if cfg.well_sep <= 1e-10 else 2.0 + 1.0 / max(cfg.well_sep, 1e-10)
        else:
            E_ref = 2.0
    elif cfg.ground_state_dir is not None:
        # Generalized locked ground states do not encode geometry via well_sep.
        E_ref = E_vmc
    else:
        E_ref = E_vmc if cfg.well_sep > 0.01 else cfg.E_ref
    print(f"  Using E_ref = {E_ref:.5f}")

    # Phase 2: pre-compute
    t0 = time.time()
    print("\n  Phase 2: Pre-computing ground-state data...")
    if cfg.precompute_sampler == "mcmc":
        precomputed = precompute_ground_state(ground_wf, cfg, system_override=system_override)
    elif cfg.precompute_sampler == "stratified":
        precomputed = precompute_ground_state_stratified(
            ground_wf,
            cfg,
            system_override=system_override,
        )
    else:
        raise ValueError(
            f"Unknown precompute_sampler '{cfg.precompute_sampler}', expected 'mcmc' or 'stratified'."
        )
    t_pre = time.time() - t0
    print(f"  Phase 2 time: {t_pre:.0f}s")

    cfg.well_sep = float(cfg.well_sep_final)

    # Phase 3: PINN training
    t0 = time.time()
    print("\n  Phase 3: Training FiLM-conditioned PINN g_θ(x, τ)...")
    g_net, history = train_pinn(cfg, ground_wf, precomputed, E_ref)
    t_pinn = time.time() - t0
    print(f"  Phase 3 time: {t_pinn:.0f}s")

    # If pool is smaller than eval, re-compute a larger pool
    if cfg.eval_sampler == "precomputed":
        if cfg.precompute_sampler != "mcmc":
            raise ValueError(
                "eval_sampler='precomputed' requires precompute_sampler='mcmc' because "
                "evaluate_trajectory currently assumes samples are drawn from |psi_0|^2. "
                "Use eval_sampler='mcmc' when training with stratified precompute."
            )
        if cfg.n_precompute < cfg.n_samples_eval:
            print(f"  Re-computing {cfg.n_samples_eval} samples for evaluation...")
            eval_cfg = PINNConfig(
                **{
                    **cfg.__dict__,
                    "n_precompute": cfg.n_samples_eval,
                    "precompute_local_energy_clip_width": cfg.eval_local_energy_clip_width,
                }
            )
            precomputed_eval = precompute_ground_state(
                ground_wf,
                eval_cfg,
                system_override=system_override,
            )
        else:
            precomputed_eval = precomputed
    elif cfg.eval_sampler == "mcmc":
        print(f"  Re-computing {cfg.n_samples_eval} MCMC samples for evaluation...")
        eval_cfg = PINNConfig(
            **{
                **cfg.__dict__,
                "n_precompute": cfg.n_samples_eval,
                "precompute_local_energy_clip_width": cfg.eval_local_energy_clip_width,
            }
        )
        precomputed_eval = precompute_ground_state(
            ground_wf,
            eval_cfg,
            system_override=system_override,
        )
    elif cfg.eval_sampler == "stratified":
        print(f"  Re-computing {cfg.n_samples_eval} stratified samples for evaluation...")
        eval_cfg = PINNConfig(**{**cfg.__dict__, "n_precompute": cfg.n_samples_eval})
        precomputed_eval = precompute_ground_state_stratified(
            ground_wf,
            eval_cfg,
            system_override=system_override,
            eval_mode=True,
            store_sampling_density=True,
        )
    else:
        raise ValueError(
            f"Unknown eval_sampler '{cfg.eval_sampler}', expected 'precomputed', 'mcmc', or 'stratified'."
        )

    # Phase 4: Evaluate
    t0 = time.time()
    print("\n  Phase 4: Evaluating E(τ) trajectory...")
    traj = evaluate_trajectory(g_net, precomputed_eval, cfg, E_ref)
    t_eval = time.time() - t0
    print(f"  Phase 4 time: {t_eval:.0f}s")

    print(f"\n  {'tau':>8s}  {'E':>10s}  {'E_err':>10s}  {'g_rms':>8s}  {'n_eff':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for pt in traj:
        print(
            f"  {pt['tau']:8.4f}  {pt['E']:10.5f}  {pt['E_err']:10.5f}  "
            f"{pt['g_rms']:8.4f}  {pt['n_eff']:8.0f}"
        )

    # Fit
    print("\n  Fitting...")
    fit_s = fit_single_exponential(traj, E_ref)
    fit_d = fit_double_exponential(traj, E_ref)
    fit_r = fit_restricted_exponential(traj, E_ref, tau_min=0.15, tau_max=3.0)
    # Log-linear fit: fix E0 to last τ-point average
    E0_for_log = np.mean([r["E"] for r in traj[-5:]])
    fit_ll = fit_log_linear(traj, E0_for_log)
    # Log-linear with E0 = VMC energy (best independent estimate)
    fit_ll_vmc = fit_log_linear(traj, E_vmc, tau_min=0.1, tau_max=2.5)
    # Windowed log-linear: use middle τ range where single exponential dominates
    fit_ll_win = fit_log_linear(traj, E0_for_log, tau_min=0.15, tau_max=2.5)
    # Explicit window sweep to diagnose fit stability vs τ-range choices.
    fit_ll_windows = fit_log_linear_windows(
        traj,
        E0_for_log,
        windows=[
            (0.05, 1.0),
            (0.10, 1.5),
            (0.15, 2.0),
            (0.20, 2.5),
            (0.30, 3.0),
        ],
    )
    # Optimal E0 scan: finds E0 that gives best linear fit in log-space
    fit_opt = fit_optimal_E0(traj, E_ref)
    # Best estimate: ensemble of exp fit, restricted exp, and log-linear
    fit_best = fit_best_estimate(
        {
            "exp": fit_s,
            "restricted": fit_r,
            "loglin": fit_ll,
            "loglin_vmc": fit_ll_vmc,
        }
    )

    expected = cfg.omega if cfg.well_sep < 0.01 else None  # Kohn gap only for d=0
    print("\n  =============================================")
    print(
        f"  RESULT: d={cfg.well_sep:.1f}, ω={cfg.omega:.1f}, "
        f"coulomb={'on' if cfg.coulomb else 'off'}"
    )
    if hasattr(g_net, "get_gaps"):
        direct_gaps = g_net.get_gaps()
        print(f"  [DIRECT]   gaps = [{', '.join(f'{g:.5f}' for g in direct_gaps)}]")
    if fit_s.get("success"):
        err_str = f", err={abs(fit_s['gap']-expected)/expected*100:.2f}%" if expected else ""
        print(
            f"  [Exp fit]  E_0={fit_s['E0']:.5f}±{fit_s['E0_err']:.5f}, "
            f"gap={fit_s['gap']:.4f}±{fit_s['gap_err']:.4f}{err_str}"
        )
    if fit_r.get("success"):
        err_str = f", err={abs(fit_r['gap']-expected)/expected*100:.2f}%" if expected else ""
        print(
            f"  [RestExp]  E_0={fit_r['E0']:.5f}±{fit_r['E0_err']:.5f}, "
            f"gap={fit_r['gap']:.4f}±{fit_r['gap_err']:.4f}{err_str} "
            f"({fit_r['n_points']} pts, τ∈[0.15,3.0])"
        )
    if fit_ll.get("success"):
        err_str = f", err={abs(fit_ll['gap']-expected)/expected*100:.2f}%" if expected else ""
        print(
            f"  [Log-lin]  E_0={fit_ll['E0']:.5f}(fixed), "
            f"gap={fit_ll['gap']:.4f}±{fit_ll['gap_err']:.4f}{err_str} "
            f"({fit_ll['n_points']} pts)"
        )
    if fit_ll_vmc.get("success"):
        err_str = f", err={abs(fit_ll_vmc['gap']-expected)/expected*100:.2f}%" if expected else ""
        print(
            f"  [LL-VMC]   E_0={E_vmc:.5f}(vmc), "
            f"gap={fit_ll_vmc['gap']:.4f}±{fit_ll_vmc['gap_err']:.4f}{err_str}"
        )
    if fit_ll_win.get("success"):
        err_str = f", err={abs(fit_ll_win['gap']-expected)/expected*100:.2f}%" if expected else ""
        print(
            f"  [Windowed] E_0={fit_ll_win['E0']:.5f}(fixed), "
            f"gap={fit_ll_win['gap']:.4f}±{fit_ll_win['gap_err']:.4f}{err_str} "
            f"({fit_ll_win['n_points']} pts, τ∈[0.15,2.5])"
        )
    print("  [LL-WIN]   window diagnostics (E0 fixed to late-τ mean):")
    for w in fit_ll_windows:
        if w.get("success"):
            print(
                f"             τ∈[{w['tau_min']:.2f},{w['tau_max']:.2f}] "
                f"gap={w['gap']:.4f}±{w['gap_err']:.4f} (n={w['n_points']})"
            )
        else:
            print(
                f"             τ∈[{w['tau_min']:.2f},{w['tau_max']:.2f}] fit failed"
            )
    if fit_d.get("success"):
        print(f"  [Double]   gap1={fit_d['gap1']:.4f}, gap2={fit_d['gap2']:.4f}")
    if fit_opt.get("success"):
        err_str = f", err={abs(fit_opt['gap']-expected)/expected*100:.2f}%" if expected else ""
        print(
            f"  [OptE0]    E_0={fit_opt['E0']:.5f}, "
            f"gap={fit_opt['gap']:.4f}±{fit_opt['gap_err']:.4f}{err_str} "
            f"(res={fit_opt['residual']:.6f})"
        )
    if fit_best.get("success"):
        err_str = f", err={abs(fit_best['gap']-expected)/expected*100:.2f}%" if expected else ""
        print(
            f"  [BEST]     gap={fit_best['gap']:.4f}±{fit_best['gap_err']:.4f}{err_str} "
            f"(ensemble of {fit_best['n_methods']})"
        )
    if expected:
        print(f"  Expected = {expected:.4f} (Kohn mode)")
    else:
        print(f"  (No built-in exact reference for d={cfg.well_sep:.1f}; compare against external diag target if available)")
    print("  =============================================")

    plot_results(traj, fit_s, fit_d, cfg, E_ref, RESULTS_DIR, tag)

    # Save checkpoint for post-hoc density analysis
    from dataclasses import asdict

    ckpt_path = RESULTS_DIR / f"{tag}checkpoint.pt"
    torch.save(
        {
            "ground_wf_state": ground_wf.state_dict(),
            "g_net_state": g_net.state_dict(),
            "config": asdict(cfg),
            "E_ref": E_ref,
            "E_vmc": E_vmc,
            "precomputed_x": precomputed_eval["x"],
            "precomputed_E_L0": precomputed_eval["E_L0"],
            "precomputed_deltaV": precomputed_eval.get("deltaV"),
            "precomputed_grad": precomputed_eval["grad_log_psi0"],
        },
        ckpt_path,
    )
    print(f"  Checkpoint saved: {ckpt_path}")

    return {
        "d": cfg.well_sep,
        "well_sep_initial": cfg.well_sep_initial,
        "well_sep_final": cfg.well_sep_final,
        "omega": cfg.omega,
        "magnetic_B_initial": cfg.magnetic_B_initial,
        "magnetic_B": cfg.magnetic_B,
        "g_factor": cfg.g_factor,
        "mu_B": cfg.mu_B,
        "zeeman_electron1_only": cfg.zeeman_electron1_only,
        "zeeman_particle_indices": list(cfg.zeeman_particle_indices) if cfg.zeeman_particle_indices is not None else None,
        "E_ref": E_ref,
        "E_vmc": E_vmc,
        "coulomb": cfg.coulomb,
        "no_vmc_train": cfg.no_vmc_train,
        "loss_style": cfg.loss_style,
        "tau_sampling": cfg.tau_sampling,
        "x_sampling": cfg.x_sampling,
        "precompute_sampler": cfg.precompute_sampler,
        "eval_sampler": cfg.eval_sampler,
        "trajectory": traj,
        "fit_single": fit_s,
        "fit_double": fit_d,
        "fit_restricted": fit_r,
        "fit_log_linear": fit_ll,
        "fit_loglin_vmc": fit_ll_vmc,
        "fit_windowed": fit_ll_win,
        "fit_optimal_E0": fit_opt,
        "fit_best": fit_best,
        "fit_loglin_windows": fit_ll_windows,
        "direct_gaps": g_net.get_gaps().tolist() if hasattr(g_net, "get_gaps") else None,
        "t_vmc": t_vmc,
        "t_pinn": t_pinn,
        "t_eval": t_eval,
        "ic_type": cfg.ic_type,
        "checkpoint": str(ckpt_path),
    }


# ============================================================
# CLI modes
# ============================================================
def _save(data, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=conv)


def test_free():
    """Non-interacting N=2, ω=1: E_0=2.0, gap=1.0 (exact)."""
    print("\n" + "=" * 70)
    print("VALIDATION: Non-interacting N=2, ω=1.0")
    print("Expected: E_0=2.0, gap=ω=1.0 (exact)")
    print("=" * 70)
    cfg = PINNConfig(
        omega=1.0,
        well_sep=0.0,
        E_ref=2.0,
        coulomb=False,
        tau_max=4.0,
        n_epochs_vmc=600,
        n_samples_vmc=512,
        lr_vmc=3e-3,
        n_precompute=8192,
        n_epochs_pde=10000,
        batch_pde=256,
        lr_pde=1e-3,
        ic_amplitude=1.0,
        ic_type="dipole",
        lambda_ic=80.0,
        lambda_bc=0.5,
        lambda_reg=0.005,
        n_tau_eval=40,
        n_samples_eval=8000,
        use_backflow=False,
        use_spectral=True,
        g_modes=3,
        g_hidden=32,
        g_layers=2,
    )
    result = run_single(cfg, tag="free_")
    _save(result, RESULTS_DIR / "pinn_free.json")
    print(f"\nSaved: {RESULTS_DIR / 'pinn_free.json'}")


def tiny_test():
    """Interacting N=2, ω=1, d=0: E_0=3.0, gap=1.0 (Kohn)."""
    print("\n" + "=" * 70)
    print("TINY: Interacting N=2, ω=1.0, d=0")
    print("Expected: E_0=3.0, gap=ω=1.0 (Kohn theorem)")
    print("=" * 70)
    cfg = PINNConfig(
        omega=1.0,
        well_sep=0.0,
        E_ref=3.0,
        coulomb=True,
        tau_max=4.0,
        n_epochs_vmc=600,
        n_samples_vmc=512,
        lr_vmc=3e-3,
        n_precompute=8192,
        n_epochs_pde=10000,
        batch_pde=256,
        lr_pde=1e-3,
        ic_amplitude=1.0,
        ic_type="dipole",
        lambda_ic=80.0,
        lambda_bc=0.5,
        lambda_reg=0.005,
        n_tau_eval=40,
        n_samples_eval=8000,
        use_backflow=True,
        use_spectral=True,
        g_modes=3,
        g_hidden=32,
        g_layers=2,
    )
    result = run_single(cfg, tag="tiny_")
    _save(result, RESULTS_DIR / "pinn_tiny.json")
    print(f"\nSaved: {RESULTS_DIR / 'pinn_tiny.json'}")


def full_test():
    """Converged interacting test with stronger BC enforcement."""
    print("\n" + "=" * 70)
    print("FULL: Interacting N=2, ω=1.0, d=0")
    print("=" * 70)
    cfg = PINNConfig(
        omega=1.0,
        well_sep=0.0,
        E_ref=3.0,
        coulomb=True,
        tau_max=5.0,
        n_epochs_vmc=1000,
        n_samples_vmc=512,
        lr_vmc=3e-3,
        n_precompute=8192,
        n_epochs_pde=15000,
        batch_pde=256,
        lr_pde=1e-3,
        ic_amplitude=1.0,
        ic_type="dipole",
        lambda_ic=80.0,
        lambda_bc=1.5,
        lambda_reg=0.005,
        n_tau_eval=50,
        n_samples_eval=8000,
        use_backflow=True,
        use_spectral=True,
        g_modes=3,
        g_hidden=32,
        g_layers=2,
    )
    result = run_single(cfg, tag="full_")
    _save(result, RESULTS_DIR / "pinn_full.json")


def sudden_quench_B_sweep():
    """Sweep magnetic field with sudden quench: GS prepared without B, then evolved with B."""
    print("\n" + "=" * 70)
    print("SUDDEN QUENCH B-SWEEP: d=0, ω=1.0")
    print("Protocol: GS prep with B_initial=0, evolve with B_evolution=[0.0, 0.1, 0.2, 0.3]")
    print("=" * 70)
    all_results = []
    for B_evol in [0.0, 0.1, 0.2, 0.3]:
        cfg = PINNConfig(
            omega=1.0,
            well_sep=0.0,
            magnetic_B_initial=0.0,      # No field for VMC prep
            magnetic_B=B_evol,           # Field turned on during evolution
            E_ref=3.0,
            coulomb=True,
            tau_max=4.0,
            n_epochs_vmc=600,
            n_samples_vmc=512,
            lr_vmc=3e-3,
            n_precompute=8192,
            n_epochs_pde=10000,
            batch_pde=256,
            lr_pde=1e-3,
            ic_amplitude=1.0,
            ic_type="dipole",
            lambda_ic=80.0,
            lambda_bc=0.5,
            lambda_reg=0.005,
            n_tau_eval=40,
            n_samples_eval=8000,
            use_backflow=True,
            use_spectral=True,
            g_modes=3,
            g_hidden=32,
            g_layers=2,
        )
        result = run_single(cfg, tag=f"quench_B{B_evol:.1f}_")
        all_results.append(result)
    _save(all_results, RESULTS_DIR / "pinn_sudden_quench_B.json")

    # Summary table
    print("\n" + "=" * 70)
    print("SUDDEN QUENCH B-SWEEP SUMMARY")
    print("=" * 70)
    print(f"  {'B_evol':>8s}  {'E_vmc':>8s}  {'gap_exp':>8s}  {'gap_ll':>8s}  {'gap_best':>9s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")
    for r in all_results:
        B_val = r.get("magnetic_B", float("nan"))
        ge = r.get("fit_single", {}).get("gap", float("nan"))
        gl = r.get("fit_log_linear", {}).get("gap", float("nan"))
        gb = r.get("fit_best", {}).get("gap", float("nan"))
        print(f"  {B_val:8.4f}  {r['E_vmc']:8.4f}  {ge:8.4f}  {gl:8.4f}  {gb:9.4f}")
    print("=" * 70)


def build_quench_config(
    B_evol: float,
    profile: str = "full",
    *,
    zeeman_electron1_only: bool = False,
    zeeman_particle_indices: tuple[int, ...] | None = None,
) -> PINNConfig:
    """Build config for sudden-quench run at a single B value."""
    if profile == "fast":
        return PINNConfig(
            omega=1.0,
            well_sep=0.0,
            magnetic_B_initial=0.0,
            magnetic_B=float(B_evol),
            zeeman_electron1_only=zeeman_electron1_only,
            zeeman_particle_indices=zeeman_particle_indices,
            E_ref=3.0,
            coulomb=True,
            tau_max=2.0,
            n_epochs_vmc=250,
            n_samples_vmc=256,
            lr_vmc=3e-3,
            n_precompute=3072,
            n_epochs_pde=3000,
            batch_pde=192,
            lr_pde=1e-3,
            ic_amplitude=1.0,
            ic_type="dipole",
            lambda_ic=80.0,
            lambda_bc=0.5,
            lambda_reg=0.005,
            n_tau_eval=30,
            n_samples_eval=4000,
            use_backflow=True,
            use_spectral=True,
            g_modes=3,
            g_hidden=32,
            g_layers=2,
        )

    return PINNConfig(
        omega=1.0,
        well_sep=0.0,
        magnetic_B_initial=0.0,
        magnetic_B=float(B_evol),
        zeeman_electron1_only=zeeman_electron1_only,
        zeeman_particle_indices=zeeman_particle_indices,
        E_ref=3.0,
        coulomb=True,
        tau_max=4.0,
        n_epochs_vmc=600,
        n_samples_vmc=512,
        lr_vmc=3e-3,
        n_precompute=8192,
        n_epochs_pde=10000,
        batch_pde=256,
        lr_pde=1e-3,
        ic_amplitude=1.0,
        ic_type="dipole",
        lambda_ic=80.0,
        lambda_bc=0.5,
        lambda_reg=0.005,
        n_tau_eval=40,
        n_samples_eval=8000,
        use_backflow=True,
        use_spectral=True,
        g_modes=3,
        g_hidden=32,
        g_layers=2,
    )


def run_quench_single_B(
    B_evol: float,
    profile: str = "full",
    *,
    zeeman_electron1_only: bool = False,
    zeeman_particle_indices: tuple[int, ...] | None = None,
    ground_state_dir: str | None = None,
) -> dict:
    """Run sudden-quench pipeline for one B value and save dedicated JSON."""
    cfg = build_quench_config(
        B_evol,
        profile=profile,
        zeeman_electron1_only=zeeman_electron1_only,
        zeeman_particle_indices=zeeman_particle_indices,
    )
    if ground_state_dir:
        cfg.ground_state_dir = str(ground_state_dir)
    btag = f"{B_evol:.2f}".replace(".", "p")
    profile_tag = "fast" if profile == "fast" else "full"
    tag = f"quench_single_{profile_tag}_B{btag}_"
    result = run_single(cfg, tag=tag)
    out = RESULTS_DIR / f"pinn_quench_single_{profile_tag}_B{btag}.json"
    _save(result, out)
    print(f"Saved single-B quench result: {out}")
    return result


def build_quench_config_sep(
    well_sep_initial: float,
    well_sep_final: float,
    profile: str = "fast",
) -> PINNConfig:
    if profile == "fast":
        return PINNConfig(
            omega=1.0,
            well_sep=float(well_sep_final),
            well_sep_initial=float(well_sep_initial),
            well_sep_final=float(well_sep_final),
            magnetic_B_initial=0.0,
            magnetic_B=0.0,
            E_ref=3.0,
            coulomb=True,
            tau_max=2.0,
            n_precompute=3072,
            n_epochs_pde=3000,
            batch_pde=192,
            lr_pde=1e-3,
            ic_amplitude=1.0,
            ic_type="dipole",
            lambda_ic=80.0,
            lambda_bc=0.5,
            lambda_reg=0.005,
            n_tau_eval=30,
            n_samples_eval=4000,
            use_backflow=False,
            use_spectral=True,
            g_modes=3,
            g_hidden=32,
            g_layers=2,
            precompute_sampler="stratified",
            eval_sampler="stratified",
        )

    return PINNConfig(
        omega=1.0,
        well_sep=float(well_sep_final),
        well_sep_initial=float(well_sep_initial),
        well_sep_final=float(well_sep_final),
        magnetic_B_initial=0.0,
        magnetic_B=0.0,
        E_ref=3.0,
        coulomb=True,
        tau_max=4.0,
        n_precompute=8192,
        n_epochs_pde=10000,
        batch_pde=256,
        lr_pde=1e-3,
        ic_amplitude=1.0,
        ic_type="dipole",
        lambda_ic=80.0,
        lambda_bc=0.5,
        lambda_reg=0.005,
        n_tau_eval=40,
        n_samples_eval=8000,
        use_backflow=False,
        use_spectral=True,
        g_modes=3,
        g_hidden=32,
        g_layers=2,
        precompute_sampler="stratified",
        eval_sampler="stratified",
    )


def run_quench_single_sep(
    well_sep_initial: float,
    well_sep_final: float,
    profile: str = "fast",
    *,
    ground_state_dir: str | None = None,
) -> dict:
    cfg = build_quench_config_sep(
        well_sep_initial=well_sep_initial,
        well_sep_final=well_sep_final,
        profile=profile,
    )
    if ground_state_dir:
        cfg.ground_state_dir = str(ground_state_dir)
    itag = f"{well_sep_initial:.2f}".replace(".", "p")
    ftag = f"{well_sep_final:.2f}".replace(".", "p")
    profile_tag = "fast" if profile == "fast" else "full"
    tag = f"quench_sep_{profile_tag}_d{itag}_to_d{ftag}_"
    result = run_single(cfg, tag=tag)
    out = RESULTS_DIR / f"pinn_quench_sep_{profile_tag}_d{itag}_to_d{ftag}.json"
    _save(result, out)
    print(f"Saved single-separation quench result: {out}")
    return result


def sweep_distances():
    print("\n" + "=" * 70)
    print("SWEEP: ω=1.0, d=[0, 1, 2, 4]")
    print("=" * 70)
    all_results = []
    for d in [0.0, 1.0, 2.0, 4.0]:
        cfg = PINNConfig(
            omega=1.0,
            well_sep=d,
            E_ref=3.0,
            coulomb=True,
            tau_max=4.0,
            n_epochs_vmc=600,
            n_samples_vmc=512,
            lr_vmc=3e-3,
            n_precompute=8192,
            n_epochs_pde=10000,
            batch_pde=256,
            lr_pde=1e-3,
            ic_amplitude=1.0,
            ic_type="dipole",
            lambda_ic=80.0,
            lambda_bc=0.5,
            lambda_reg=0.005,
            n_tau_eval=40,
            n_samples_eval=8000,
            use_backflow=True,
            use_spectral=True,
            g_modes=3,
            g_hidden=32,
            g_layers=2,
        )
        result = run_single(cfg, tag=f"sweep_d{d:.0f}_")
        all_results.append(result)
    _save(all_results, RESULTS_DIR / "pinn_sweep.json")

    # Summary table
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"  {'d':>4s}  {'E_vmc':>8s}  {'gap_exp':>8s}  {'gap_ll':>8s}  {'gap_best':>9s}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")
    for r in all_results:
        ge = r.get("fit_single", {}).get("gap", float("nan"))
        gl = r.get("fit_log_linear", {}).get("gap", float("nan"))
        gb = r.get("fit_best", {}).get("gap", float("nan"))
        print(f"  {r['d']:4.1f}  {r['E_vmc']:8.4f}  {ge:8.4f}  {gl:8.4f}  {gb:9.4f}")
    print("=" * 70)


def rerun_d4():
    """Rerun SpectralG d=4 with beefed-up settings."""
    print("\n" + "=" * 70)
    print("RERUN: d=4 with more resources")
    print("=" * 70)
    cfg = PINNConfig(
        omega=1.0,
        well_sep=4.0,
        E_ref=3.0,
        coulomb=True,
        tau_max=4.0,
        n_epochs_vmc=1200,       # 2x more VMC
        n_samples_vmc=1024,      # 2x more samples
        lr_vmc=3e-3,
        n_precompute=16384,      # 2x more pool
        n_epochs_pde=20000,      # 2x more PDE epochs
        batch_pde=512,           # 2x batch
        lr_pde=5e-4,             # lower LR for stability
        ic_amplitude=2.0,        # stronger IC
        ic_type="dipole",
        lambda_ic=80.0,
        lambda_bc=0.5,
        lambda_reg=0.002,        # less reg
        n_tau_eval=60,           # more eval points
        n_samples_eval=16000,    # 2x eval samples
        use_backflow=True,
        use_spectral=True,
        g_modes=5,               # more modes
        g_hidden=64,             # bigger network
        g_layers=3,              # deeper
    )
    result = run_single(cfg, tag="rerun_d4_")
    _save(result, RESULTS_DIR / "pinn_rerun_d4.json")

    # Also update the sweep file
    sweep_path = RESULTS_DIR / "pinn_sweep.json"
    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep = json.load(f)
        # Replace the d=4 entry
        for i, r in enumerate(sweep):
            if abs(r["d"] - 4.0) < 0.01:
                sweep[i] = result
                print(f"  Updated d=4 entry in {sweep_path}")
                break
        _save(sweep, sweep_path)
    print(f"\nSaved: {RESULTS_DIR / 'pinn_rerun_d4.json'}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Imaginary-Time PINN Spectroscopy")
    p.add_argument("--test_free", action="store_true", help="Non-interacting validation")
    p.add_argument("--tiny", action="store_true", help="Quick interacting test")
    p.add_argument("--full", action="store_true", help="Converged interacting test")
    p.add_argument("--quench", action="store_true", help="Sudden quench B-field sweep")
    p.add_argument("--quench_B", type=float, default=None, help="Run sudden quench for one B")
    p.add_argument(
        "--zeeman_electron1_only",
        action="store_true",
        help="Apply Zeeman term only to particle 0 (mutually exclusive with --zeeman_particles).",
    )
    p.add_argument(
        "--zeeman_particles",
        type=str,
        default="",
        help="Comma-separated particle indices for Zeeman targeting, e.g. '0,2,3'.",
    )
    p.add_argument(
        "--quench_profile",
        choices=["full", "fast"],
        default="full",
        help="Config profile for --quench_B",
    )
    p.add_argument(
        "--ground_state_dir",
        type=str,
        default=None,
        help="Path to run_ground_state output directory (contains model.pt + config.yaml).",
    )
    p.add_argument("--quench_sep_initial", type=float, default=None, help="Initial well separation for separation quench.")
    p.add_argument("--quench_sep_final", type=float, default=None, help="Final well separation for separation quench.")
    p.add_argument(
        "--quench_sep_profile",
        choices=["full", "fast"],
        default="fast",
        help="Config profile for separation quench.",
    )
    p.add_argument("--sweep", action="store_true", help="Distance sweep")
    p.add_argument("--rerun_d4", action="store_true", help="Rerun d=4 with better settings")
    args = p.parse_args()

    zeeman_particle_indices: tuple[int, ...] | None = None
    if args.zeeman_particles.strip():
        zeeman_particle_indices = tuple(
            int(tok.strip()) for tok in args.zeeman_particles.split(",") if tok.strip()
        )

    if args.test_free:
        test_free()
    elif args.tiny:
        tiny_test()
    elif args.full:
        full_test()
    elif args.quench_B is not None:
        run_quench_single_B(
            args.quench_B,
            profile=args.quench_profile,
            zeeman_electron1_only=bool(args.zeeman_electron1_only),
            zeeman_particle_indices=zeeman_particle_indices,
            ground_state_dir=args.ground_state_dir,
        )
    elif args.quench_sep_initial is not None or args.quench_sep_final is not None:
        if args.quench_sep_initial is None or args.quench_sep_final is None:
            raise ValueError("Both --quench_sep_initial and --quench_sep_final are required.")
        run_quench_single_sep(
            args.quench_sep_initial,
            args.quench_sep_final,
            profile=args.quench_sep_profile,
            ground_state_dir=args.ground_state_dir,
        )
    elif args.quench:
        sudden_quench_B_sweep()
    elif args.sweep:
        sweep_distances()
    elif args.rerun_d4:
        rerun_d4()
    else:
        test_free()
