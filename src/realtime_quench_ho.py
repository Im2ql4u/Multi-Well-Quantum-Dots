"""Closed-form harmonic-oscillator quench reference for the real-time PINN.

This is the *first non-trivial test* of the real-time PINN framework
(``src/realtime_pinn.py``): a sudden frequency quench of a 2D harmonic
oscillator. The analytical solution is known in closed form, so we have an
exact benchmark for every observable — no ED needed.

Setup
-----
N non-interacting (Coulomb-off) electrons in a 2D harmonic well with
frequency ``ω = ω₀``. The closed-shell ground state is the product of
single-particle 2D HO ground states:

.. math::

    \\psi_0(x_1,\\dots,x_N) = \\prod_i \\varphi_0(x_i),
    \\qquad
    \\varphi_0(x) = (\\omega_0/\\pi)^{1/2}\\,
                    \\exp\\!\\bigl(-\\omega_0 |x|^2/2\\bigr).

At ``t = 0`` we *suddenly* switch the frequency to ``ω = ω₁``. The PDE
quench potential is

.. math::

    \\Delta V(x_1,\\dots,x_N)
        = \\tfrac{1}{2}(\\omega_1^2 - \\omega_0^2)\\sum_i |x_i|^2.

The full Hamiltonian splits into single-particle pieces, so the analytical
solution per electron is the standard squeezed-state evolution. The
**observable** we benchmark against is

.. math::

    \\langle |x|^2\\rangle_{\\text{1e}}(t)
      = \\frac{1}{\\omega_0}\\cos^2(\\omega_1 t)
      + \\frac{\\omega_0}{\\omega_1^2}\\sin^2(\\omega_1 t),

with the N-particle aggregate ``⟨Σ_i |x_i|²⟩(t) = N · ⟨|x|²⟩_{1e}(t)``.

The PINN measurement is a deterministic reweighted average over the same
``|ψ₀|²``-distributed pool used in training:

.. math::

    \\langle O\\rangle_{\\text{PINN}}(t)
      = \\frac{\\langle O\\,e^{2 g_R(x,t)}\\rangle_{|\\psi_0|^2}}
             {\\langle e^{2 g_R(x,t)}\\rangle_{|\\psi_0|^2}}.

Module surface
--------------
* :func:`build_ho_quench_pool` — analytical ``ψ₀`` pool: ``x``, ``E_L^{(0)}
  = N·ω₀``, ``∇log ψ₀ = -ω₀ x``, ``ΔV(x)``.
* :func:`analytical_x2_per_electron` — closed-form ``⟨|x|²⟩_{1e}(t)``.
* :func:`pinn_x2_aggregate` — reweighted ``⟨Σ_i |x_i|²⟩_{PINN}(t)`` from a
  trained :class:`RealTimeNet`.
* :func:`pinn_norm_ratio` — the diagnostic ``Z(t) := ⟨e^{2 g_R}⟩``; should
  stay equal to 1 under exact unitary evolution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class HOQuenchConfig:
    """Parameters of a sudden-frequency 2D HO quench."""

    n_particles: int = 2
    dim: int = 2  # spatial dim per electron — keep at 2 for now
    omega_0: float = 1.0
    omega_1: float = 2.0


def build_ho_quench_pool(
    cfg: HOQuenchConfig,
    *,
    n_samples: int,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Sample (x, E_L^{(0)}, ∇log ψ_0, ΔV) for the HO quench.

    Samples are drawn from the analytical ``|ψ_0|²`` density (a product of
    Gaussians of width ``σ = 1/√ω₀``) using ``numpy.random.default_rng`` —
    deterministic, no MCMC, no acceptance/rejection.

    Returns a dict with keys ``x`` (B, N, d), ``E_L0`` (B,),
    ``grad_log_psi0`` (B, N, d), ``deltaV`` (B,), ``E_0`` (float).
    """
    if cfg.dim != 2:
        raise NotImplementedError(
            "HO quench reference is currently set up for dim=2 only "
            "(closed-form per-axis decomposition assumes 2D)."
        )
    if cfg.n_particles < 1:
        raise ValueError(f"n_particles must be >=1, got {cfg.n_particles}.")
    if cfg.omega_0 <= 0 or cfg.omega_1 <= 0:
        raise ValueError("omega_0 and omega_1 must be strictly positive.")

    rng = np.random.default_rng(seed)
    # |ψ₀|² ∝ exp(-ω₀ Σ|x_i|²) ⇒ per-axis variance is 1/(2 ω₀).
    sigma = 1.0 / np.sqrt(2.0 * cfg.omega_0)
    x_np = rng.normal(0.0, sigma, size=(n_samples, cfg.n_particles, cfg.dim))
    x = torch.tensor(x_np, dtype=dtype, device=device)

    grad_log_psi0 = (-cfg.omega_0 * x).clone()

    # ψ_0 is the exact GS of H_0 ⇒ E_L^{(0)} ≡ E_0 (per-particle ω₀).
    e_0 = float(cfg.n_particles * cfg.omega_0)
    e_l0 = torch.full((n_samples,), e_0, dtype=dtype, device=device)

    r2 = (x**2).sum(dim=(1, 2))  # (B,) total Σ_i |x_i|²
    deltaV = 0.5 * (cfg.omega_1**2 - cfg.omega_0**2) * r2

    return {
        "x": x,
        "E_L0": e_l0,
        "grad_log_psi0": grad_log_psi0,
        "deltaV": deltaV,
        "E_0": e_0,
        "sigma": float(sigma),
    }


def analytical_x2_per_electron(cfg: HOQuenchConfig, t: float | np.ndarray) -> np.ndarray:
    """Closed-form ``⟨|x|²⟩_{1e}(t)`` for the 2D HO ω-quench.

    Derivation (Heisenberg picture for ``H_1``):

    * ``x(t) = x cos(ω₁ t) + (p/ω₁) sin(ω₁ t)``,
    * ``⟨x²⟩₀ = 1/(2ω₀)`` (1D HO GS variance),
    * ``⟨p²⟩₀ = ω₀/2``,
    * ``⟨{x,p}⟩₀ = 0``.

    Hence (per axis)
    ``⟨x²⟩₁ₐₓᵢₛ(t) = (1/(2ω₀)) cos²(ω₁ t) + (ω₀/(2ω₁²)) sin²(ω₁ t)``.

    For 2D the x and y axes decouple identically, so
    ``⟨|x|²⟩_{1e}(t) = 2 × ⟨x²⟩₁ₐₓᵢₛ(t) =
        (1/ω₀) cos²(ω₁ t) + (ω₀/ω₁²) sin²(ω₁ t)``.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    c2 = np.cos(cfg.omega_1 * t_arr) ** 2
    s2 = np.sin(cfg.omega_1 * t_arr) ** 2
    return (1.0 / cfg.omega_0) * c2 + (cfg.omega_0 / cfg.omega_1**2) * s2


def analytical_x2_aggregate(cfg: HOQuenchConfig, t: float | np.ndarray) -> np.ndarray:
    """Closed-form ``⟨Σ_i |x_i|²⟩(t) = N · ⟨|x|²⟩_{1e}(t)``."""
    return cfg.n_particles * analytical_x2_per_electron(cfg, t)


def analytical_breathing_period(cfg: HOQuenchConfig) -> float:
    """Period of the breathing oscillation: ``T = π / ω_1``.

    The squared-position observable oscillates at ``2 ω_1`` (since both
    ``cos²`` and ``sin²`` have argument ``ω_1 t``), so the full breathing
    period is ``π / ω_1``.
    """
    return float(np.pi / cfg.omega_1)


@torch.no_grad()
def pinn_norm_ratio(
    net: nn.Module,
    pool: dict[str, torch.Tensor],
    t_values: list[float] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute ``Z(t) := ⟨e^{2 g_R(x, t)}⟩_{|ψ₀|²}`` on a held-out time grid.

    Under exact unitary evolution ``Z(t) ≡ 1`` (norm-conserved). Deviations
    diagnose how well the PINN respects unitarity.
    """
    x = pool["x"]
    n = x.shape[0]
    device = x.device
    dtype = x.dtype

    Z = []
    for t_val in t_values:
        t_t = torch.full((n,), float(t_val), dtype=dtype, device=device)
        g_R, _ = net(x, t_t)
        z = torch.exp(2.0 * g_R).mean().item()
        Z.append(z)
    return {"t": np.asarray(t_values, dtype=np.float64), "Z": np.asarray(Z, dtype=np.float64)}


@torch.no_grad()
def pinn_x2_aggregate(
    net: nn.Module,
    pool: dict[str, torch.Tensor],
    t_values: list[float] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Reweighted ``⟨Σ_i |x_i|²⟩(t)`` from the trained PINN.

    Returns dict with ``t`` (T,), ``mean`` (T,), and ``Z`` (T,) (the norm
    ratio diagnostic). The reweighted estimator is

    .. math::

        \\langle O\\rangle(t) = \\frac{\\sum_b O(x_b)\\,e^{2 g_R(x_b, t)}}
                                     {\\sum_b e^{2 g_R(x_b, t)}}.
    """
    x = pool["x"]
    n = x.shape[0]
    device = x.device
    dtype = x.dtype
    O = (x**2).sum(dim=(1, 2))  # (B,) Σ_i |x_i|²

    means = []
    Z = []
    for t_val in t_values:
        t_t = torch.full((n,), float(t_val), dtype=dtype, device=device)
        g_R, _ = net(x, t_t)
        w = torch.exp(2.0 * g_R)  # importance weight relative to |ψ₀|² sampling
        z = w.mean().item()
        m = (O * w).sum().item() / w.sum().item()
        means.append(m)
        Z.append(z)
    return {
        "t": np.asarray(t_values, dtype=np.float64),
        "mean": np.asarray(means, dtype=np.float64),
        "Z": np.asarray(Z, dtype=np.float64),
    }
