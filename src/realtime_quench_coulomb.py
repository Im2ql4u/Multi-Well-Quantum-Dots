"""Coulomb-perturbed harmonic-oscillator quench reference for the real-time PINN.

This is the **next stress test** of the real-time PINN framework
(``src/realtime_pinn.py``) after the closed-form ω-quench
(``src/realtime_quench_ho.py``). The idea: keep the analytical Gaussian
``ψ_0`` and the analytical ω-quench piece of ``ΔV`` so that the polynomial
ansatz remains a *good* approximation, but add a softcore-Coulomb pair
interaction at ``t = 0``, which breaks separability between particles and
generates genuinely non-Gaussian dynamics that no closed-form polynomial
ansatz can capture exactly.

Setup
-----
* Initial state. ``N`` non-interacting electrons in a 2D HO at frequency
  ``ω₀``: ``ψ_0(x) = ∏_i (ω₀/π)^{1/2} exp(-ω₀|x_i|²/2)``.
* Pre-quench Hamiltonian: ``H_0 = -½∑∇²_i + ½ω₀² ∑|x_i|²`` (no interaction).
* Post-quench Hamiltonian: ``H = H_0 + ½(ω₁² - ω₀²) ∑|x_i|² + λ ∑_{i<j} u(r_{ij})``
  with the softcore-Coulomb pair potential
  ``u(r) = 1/√(r² + ε²)`` (regularised so finite-batch deterministic samples
  never see the Coulomb singularity).
* PDE inputs:

  - ``E_L^{(0)}(x) ≡ N · ω₀`` (``ψ_0`` is exact GS of the non-interacting HO).
  - ``∇log ψ_0(x) = -ω₀ · x``.
  - ``ΔV(x) = ½(ω₁² - ω₀²) ∑_i|x_i|² + λ ∑_{i<j} u(r_{ij})``.

The polynomial backbone of :class:`HybridPolyMLPNet` exactly absorbs the
quadratic ``½(ω₁² - ω₀²) ∑|x_i|²`` piece (which in the *separable* case is
the entire ``ΔV``); the MLP residual is then in charge of representing the
particle-correlated correction induced by ``λ ∑_{i<j} u(r_{ij})``.

Module surface
--------------
* :class:`CoulombQuenchConfig` — dataclass with ``ω₀, ω₁, λ, ε`` plus
  geometry (``N, d``).
* :func:`build_coulomb_quench_pool` — analytical Gaussian samples + the
  precomputed ``E_L^{(0)}``, ``∇log ψ_0``, and ``ΔV`` (including the
  pair-interaction piece).
* :func:`pinn_pair_interaction_aggregate` — reweighted ``⟨V_int⟩(t)``
  observable at the post-quench Hamiltonian (zero at ``t=0`` if ``λ=0``,
  otherwise ``λ ∑_{i<j} u(r_{ij})`` averaged over the |ψ₀|² pool).
* :func:`analytical_x2_aggregate_ohne_coulomb` — convenience: closed-form
  ``⟨Σ|x_i|²⟩(t)`` for the *separable* (``λ=0``) limit, useful as a sanity
  check that the hybrid net agrees with :class:`PolynomialQuenchNet` when
  Coulomb is off.

The trained PINN's deviation from the ``λ=0`` analytical reference is
exactly the signal we want from this experiment: how much does Coulomb
correlation modify the breathing dynamics, and does the hybrid ansatz
capture it self-consistently (loss ↓, residual ↓, Z(t)→1)?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from realtime_quench_ho import (
    HOQuenchConfig,
    analytical_breathing_period,
    analytical_x2_aggregate,
    analytical_x2_per_electron,
)


@dataclass(frozen=True)
class CoulombQuenchConfig:
    """Parameters of an HO+softcore-Coulomb quench.

    A pre-quench non-interacting Gaussian GS (frequency ``ω₀``) is evolved
    under a post-quench Hamiltonian that mixes a frequency change and a
    softcore-Coulomb pair interaction.

    Attributes
    ----------
    n_particles, dim
        Geometry (``dim`` must be 2 for now — the analytical Gaussian and
        the reference observables assume 2D).
    omega_0, omega_1
        Pre- and post-quench HO frequencies. With ``omega_1 == omega_0`` the
        ``ω``-piece of ``ΔV`` vanishes and the *only* perturbation is the
        Coulomb interaction (a pure interaction quench).
    lambda_coul
        Strength of the pair interaction. ``0.0`` reduces to the separable
        ω-quench; ``> 0`` introduces non-trivial particle correlations.
    epsilon_coul
        Softcore regularisation length. Standard 2D Coulomb is
        ``1/r``; with ``ε > 0`` we use ``1/√(r² + ε²)`` so the
        deterministic Gaussian sampler never encounters a singularity.
        Practical default: ``ε = 0.05/√(2 ω₀)`` ≈ 5 % of the GS cloud
        width — small enough to be Coulomb-like at typical separations, big
        enough to suppress the high-energy tail at coincident points.
    """

    n_particles: int = 2
    dim: int = 2
    omega_0: float = 1.0
    omega_1: float = 1.0
    lambda_coul: float = 0.5
    epsilon_coul: float = 0.05


def softcore_coulomb_pair_potential(
    x: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Compute ``∑_{i<j} 1/√(|x_i - x_j|² + ε²)`` per batch sample.

    Parameters
    ----------
    x
        ``(B, N, d)`` particle positions.
    epsilon
        Softcore regularisation length (must be ``> 0``).

    Returns
    -------
    torch.Tensor
        ``(B,)`` softcore-Coulomb pair sum per sample.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be (B, N, d); got {x.shape}.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}.")
    b, n, _ = x.shape
    if n < 2:
        return torch.zeros(b, dtype=x.dtype, device=x.device)
    # Pairwise differences ``x_i - x_j`` over the upper triangle.
    iu, ju = torch.triu_indices(n, n, offset=1)
    diff = x[:, iu, :] - x[:, ju, :]  # (B, n_pairs, d)
    r2 = (diff**2).sum(dim=-1) + epsilon**2  # (B, n_pairs)
    return torch.rsqrt(r2).sum(dim=-1)


def build_coulomb_quench_pool(
    cfg: CoulombQuenchConfig,
    *,
    n_samples: int,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Sample (x, E_L^{(0)}, ∇log ψ_0, ΔV) for the HO+Coulomb quench.

    The non-interacting Gaussian density ``|ψ_0|²`` has per-axis variance
    ``1/(2 ω₀)`` (same as :func:`realtime_quench_ho.build_ho_quench_pool`).
    On top of the ω-quench piece, ``ΔV`` includes the softcore-Coulomb pair
    interaction with strength ``λ`` and softcore ``ε``.

    The per-sample ``ΔV(x)`` is also split out into ``omega_piece`` and
    ``coulomb_piece`` so callers can study them separately (e.g. to verify
    that turning ``λ → 0`` reproduces the closed-form ω-quench).
    """
    if cfg.dim != 2:
        raise NotImplementedError(
            "Coulomb quench reference is currently set up for dim=2 only."
        )
    if cfg.n_particles < 1:
        raise ValueError(f"n_particles must be >=1, got {cfg.n_particles}.")
    if cfg.omega_0 <= 0 or cfg.omega_1 <= 0:
        raise ValueError("omega_0 and omega_1 must be strictly positive.")
    if cfg.lambda_coul < 0:
        raise ValueError(f"lambda_coul must be >= 0, got {cfg.lambda_coul}.")
    if cfg.epsilon_coul <= 0:
        raise ValueError(f"epsilon_coul must be > 0, got {cfg.epsilon_coul}.")

    rng = np.random.default_rng(seed)
    sigma = 1.0 / np.sqrt(2.0 * cfg.omega_0)
    x_np = rng.normal(0.0, sigma, size=(n_samples, cfg.n_particles, cfg.dim))
    x = torch.tensor(x_np, dtype=dtype, device=device)

    grad_log_psi0 = (-cfg.omega_0 * x).clone()

    e_0 = float(cfg.n_particles * cfg.omega_0)
    e_l0 = torch.full((n_samples,), e_0, dtype=dtype, device=device)

    r2 = (x**2).sum(dim=(1, 2))  # (B,) total Σ_i |x_i|²
    omega_piece = 0.5 * (cfg.omega_1**2 - cfg.omega_0**2) * r2

    if cfg.lambda_coul > 0 and cfg.n_particles >= 2:
        v_int = softcore_coulomb_pair_potential(x, epsilon=cfg.epsilon_coul)
        coulomb_piece = cfg.lambda_coul * v_int
    else:
        coulomb_piece = torch.zeros_like(omega_piece)

    deltaV = omega_piece + coulomb_piece

    return {
        "x": x,
        "E_L0": e_l0,
        "grad_log_psi0": grad_log_psi0,
        "deltaV": deltaV,
        "deltaV_omega_piece": omega_piece,
        "deltaV_coulomb_piece": coulomb_piece,
        "E_0": e_0,
        "sigma": float(sigma),
    }


def coulomb_quench_to_ho_quench(cfg: CoulombQuenchConfig) -> HOQuenchConfig:
    """Project a Coulomb-quench config onto its underlying ``HOQuenchConfig``.

    This is convenient for re-using the closed-form ω-quench analytical
    reference (``analytical_x2_aggregate``, ``analytical_breathing_period``)
    in the ``λ → 0`` ablation limit.
    """
    return HOQuenchConfig(
        n_particles=cfg.n_particles,
        dim=cfg.dim,
        omega_0=cfg.omega_0,
        omega_1=cfg.omega_1,
    )


def analytical_x2_aggregate_ohne_coulomb(
    cfg: CoulombQuenchConfig, t: float | np.ndarray
) -> np.ndarray:
    """Closed-form ``⟨Σ|x_i|²⟩(t)`` for the **non-interacting** ω-quench.

    This is a useful baseline curve to overlay on the PINN's prediction in
    the Coulomb-on case: the *gap* between the two is the genuine effect of
    the pair interaction on the breathing observable.
    """
    return analytical_x2_aggregate(coulomb_quench_to_ho_quench(cfg), t)


def analytical_breathing_period_coulomb(cfg: CoulombQuenchConfig) -> float:
    """Period ``π / ω₁`` of the underlying ω-quench (Coulomb shifts it slightly)."""
    return analytical_breathing_period(coulomb_quench_to_ho_quench(cfg))


@torch.no_grad()
def pinn_pair_interaction_aggregate(
    net: nn.Module,
    pool: dict[str, torch.Tensor],
    t_values: list[float] | np.ndarray,
    *,
    epsilon: float,
    lambda_coul: float = 1.0,
) -> dict[str, np.ndarray]:
    """Reweighted ``⟨λ ∑_{i<j} u(r_ij)⟩(t)`` from a trained PINN.

    Returns a dict with ``t`` (T,), ``mean`` (T,), and ``Z`` (T,). The
    estimator follows the same |ψ_0|²-importance reweighting scheme as
    :func:`realtime_quench_ho.pinn_x2_aggregate`.
    """
    x = pool["x"]
    n = x.shape[0]
    device = x.device
    dtype = x.dtype
    O = lambda_coul * softcore_coulomb_pair_potential(x, epsilon=epsilon)  # (B,)

    means = []
    Z = []
    for t_val in t_values:
        t_t = torch.full((n,), float(t_val), dtype=dtype, device=device)
        g_R, _ = net(x, t_t)
        w = torch.exp(2.0 * g_R)
        z = w.mean().item()
        m = (O * w).sum().item() / w.sum().item()
        means.append(m)
        Z.append(z)
    return {
        "t": np.asarray(t_values, dtype=np.float64),
        "mean": np.asarray(means, dtype=np.float64),
        "Z": np.asarray(Z, dtype=np.float64),
    }


@torch.no_grad()
def initial_pair_interaction_value(pool: dict[str, torch.Tensor]) -> float:
    """``⟨λ ∑_{i<j} u(r_ij)⟩`` at ``t = 0`` (i.e. on the |ψ_0|² pool).

    Helper convenience: the Coulomb-piece of ``ΔV`` already encodes
    ``λ · u(r_ij)``, so we just average it. This is a useful sanity check
    that the PINN observable agrees with the t=0 analytical limit.
    """
    return float(pool["deltaV_coulomb_piece"].mean().item())
