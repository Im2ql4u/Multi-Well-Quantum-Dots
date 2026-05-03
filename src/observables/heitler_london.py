"""Closed-form Heitler-London reference for N=2 in a 2D two-well dot.

This module provides an analytical (Heitler-London) anchor for the
two-electron / two-well system that we routinely train PINN ground states on.
It is the **first** element of the Phase 0 anchor inventory described in
``reports/2026-04-28_grand_plan_anchored.md`` — every more-elaborate result
must reduce to this one in the appropriate limit.

Setup
-----
Two 2D wells at :math:`\\mathbf R_L = (-d/2, 0)` and :math:`\\mathbf R_R = (+d/2, 0)`.
Each well contributes a localised harmonic potential
:math:`V_w(\\mathbf r) = \\tfrac12 \\omega^2 |\\mathbf r - \\mathbf R_w|^2`.
The total confining potential is the **soft-min** combination

.. math::

    V_{\\text{trap}}(\\mathbf r) = -t \\log \\sum_w \\exp(-V_w(\\mathbf r)/t),

matching the production trap in ``scripts/eval_ground_state_components.py``.
The two-electron Hamiltonian uses a regularised Coulomb interaction

.. math::

    V_{12}(\\mathbf r_1, \\mathbf r_2) = \\frac{\\kappa}
    {\\sqrt{|\\mathbf r_1 - \\mathbf r_2|^2 + \\epsilon^2}}.

Heitler-London ansatz
---------------------
Single-particle Gaussians :math:`\\phi_X(\\mathbf r) = \\sqrt{\\omega/\\pi}\\,
\\exp[-\\omega |\\mathbf r - \\mathbf R_X|^2 / 2]` are combined into

.. math::

    \\Psi_S = \\frac{\\phi_L(1)\\phi_R(2) + \\phi_R(1)\\phi_L(2)}
    {\\sqrt{2(1+S^2)}}, \\quad
    \\Psi_T = \\frac{\\phi_L(1)\\phi_R(2) - \\phi_R(1)\\phi_L(2)}
    {\\sqrt{2(1-S^2)}},

with :math:`S = \\langle\\phi_L|\\phi_R\\rangle = \\exp(-\\omega d^2 / 4)`.
Their energies follow the textbook formulas

.. math::

    E_{S/T} = \\frac{Q \\pm K}{1 \\pm S^2}, \\qquad
    Q = \\langle\\phi_L \\phi_R | H | \\phi_L \\phi_R\\rangle, \\quad
    K = \\langle\\phi_L \\phi_R | H | \\phi_R \\phi_L\\rangle.

All integrals are evaluated by tensor-product Gauss-Hermite quadrature, which
is exact for Gaussian-times-polynomial integrands and converges rapidly for
the regularised Coulomb kernel.

Spin correlator
---------------
:math:`\\langle\\mathbf S_1 \\cdot \\mathbf S_2\\rangle` is determined purely
by the spin sector: :math:`-3/4` for the singlet and :math:`+1/4` for any
triplet component (by direct application of
:math:`\\mathbf S_1 \\cdot \\mathbf S_2 = \\tfrac12(\\mathbf S_{tot}^2 - \\mathbf S_1^2 - \\mathbf S_2^2)`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import erfc


@dataclass(frozen=True)
class HLConfig:
    """Material-agnostic Heitler-London geometry / interaction parameters."""

    sep: float
    omega: float
    kappa: float
    epsilon: float
    smooth_t: float = 0.5
    n_quad_per_axis: int = 24

    def half_separation(self) -> float:
        return 0.5 * self.sep


# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------


def _hermite_grid_2d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return tensor-product Hermite-E nodes/weights on the 2D plane.

    Returns
    -------
    nodes
        Array of shape ``(n*n, 2)`` of integration nodes.
    weights
        Array of shape ``(n*n,)`` integrating against the standard 2D normal
        density :math:`(2\\pi)^{-1}\\exp(-(x^2+y^2)/2)`.
    """
    xs, ws = hermegauss(n)
    nodes = np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)
    weights = np.outer(ws, ws).reshape(-1) / (2.0 * np.pi)
    return nodes, weights


# ---------------------------------------------------------------------------
# Single-particle building blocks
# ---------------------------------------------------------------------------


def overlap_S(cfg: HLConfig) -> float:
    """:math:`\\langle\\phi_L | \\phi_R\\rangle = \\exp(-\\omega d^2 / 4)`."""
    return float(np.exp(-cfg.omega * cfg.sep**2 / 4.0))


def _gaussian_orbital(r: np.ndarray, center: np.ndarray, omega: float) -> np.ndarray:
    """Normalised 2D HO ground state at ``center`` with frequency ``omega``."""
    diff2 = (r - center) ** 2
    return np.sqrt(omega / np.pi) * np.exp(-0.5 * omega * diff2.sum(axis=-1))


def _trap_potential(r: np.ndarray, cfg: HLConfig) -> np.ndarray:
    """Soft-min two-well harmonic trap at points ``r`` (shape ``(..., 2)``)."""
    half = cfg.half_separation()
    rl = r.copy()
    rl[..., 0] = rl[..., 0] + half
    rr = r.copy()
    rr[..., 0] = rr[..., 0] - half
    v_l = 0.5 * cfg.omega**2 * (rl**2).sum(axis=-1)
    v_r = 0.5 * cfg.omega**2 * (rr**2).sum(axis=-1)
    t = cfg.smooth_t
    stack = np.stack([-v_l / t, -v_r / t], axis=-1)
    log_sum = np.log(np.exp(stack[..., 0]) + np.exp(stack[..., 1]))
    return -t * log_sum


def _quad_one_body(
    cfg: HLConfig,
    integrand_fn,
) -> float:
    """Two-dimensional Gauss-Hermite integral of ``f(r) phi_a(r) phi_b(r)``.

    The integrand is delivered through ``integrand_fn(r)`` (vectorised over the
    leading axis) **without** the orbital factors — those are absorbed by
    sampling ``r`` from the centred 2D normal :math:`(2\\pi)^{-1}\\exp(-r^2/2)`
    and reweighting by :math:`\\phi_a(r)\\phi_b(r) / \\mathcal N(r)`.

    Helper used internally by :func:`one_body_LL`, :func:`one_body_LR`, etc.
    """
    raise NotImplementedError("use the explicit one_body_* helpers instead")


def one_body_h_LX(cfg: HLConfig, *, off_diagonal: bool = False) -> float:
    """Compute :math:`\\langle\\phi_a | T + V_{\\text{trap}} | \\phi_b\\rangle`.

    For ``off_diagonal=False`` returns :math:`h_{LL} = h_{RR}`. For
    ``off_diagonal=True`` returns :math:`h_{LR} = h_{RL}`.

    The kinetic part is computed analytically — for the 2D HO ground state
    Gaussian, :math:`\\langle\\phi|T|\\phi\\rangle = \\omega/2` and the
    cross term :math:`\\langle\\phi_L | T | \\phi_R\\rangle = (\\omega/2) (1 -
    \\omega d^2/2) S` (standard ladder-operator result for displaced Gaussians).
    The potential part is computed by quadrature.
    """
    omega = cfg.omega
    half = cfg.half_separation()
    s = overlap_S(cfg)

    # Kinetic part (closed form for 2D Gaussian ground state).
    if off_diagonal:
        # <phi_L| -1/2 nabla^2 |phi_R> = (omega/2)(1 - omega d^2 / 2) * S.
        # Derivation: -1/2 nabla^2 phi_R = (omega - (omega^2/2)|r-R_R|^2) phi_R.
        # Then <phi_L|...|phi_R> involves <phi_L|(r-R_R)^2|phi_R>, which equals
        # (1/omega + d^2/4) * S for displaced 2D Gaussians.
        # => <phi_L|T|phi_R> = (omega - omega^2/2 * (1/omega + d^2/4)) * S
        #                    = (omega - omega/2 - omega^2 d^2/8) * S
        #                    = (omega/2 - omega^2 d^2 / 8) * S.
        t_off = (0.5 * omega - omega**2 * cfg.sep**2 / 8.0) * s
        kinetic = t_off
    else:
        # <phi|T|phi> = omega/2 (2D HO ground state) — actually omega for 2D
        # ground state since E_GS = omega and virial gives T = V_trap = omega/2
        # in a *single* HO. So kinetic = omega/2.
        kinetic = 0.5 * omega

    # Potential part by 2D Gauss-Hermite quadrature.
    # Sample r from N(0, sigma^2 I) where sigma^2 = 1/(2 omega) so that the
    # weight phi_a(r) phi_b(r) / pdf(r) is most concentrated. For diagonal
    # (a=b) elements we centre on R_L; for off-diagonal we centre on the
    # midpoint (origin) and absorb the relative phase.
    n = cfg.n_quad_per_axis
    xs, ws = hermegauss(n)
    # For HermiteE, integral is over exp(-x^2/2). Map x -> sigma * x to get
    # 2D Gaussian with stddev sigma.
    # For diagonal: pdf(r) = (omega/pi) exp(-omega r'^2) where r' = r - R_L.
    # We pick the sampling Gaussian with stddev sigma so that
    #   sigma_used = 1 / sqrt(2 omega) -> exp(-r^2 / (2 sigma^2)) = exp(-omega r^2)
    # which exactly matches |phi_L|^2 (no reweighting needed).
    sigma = 1.0 / np.sqrt(2.0 * omega)
    nodes_2d_x = sigma * xs
    nodes_2d_y = sigma * xs
    grid_x, grid_y = np.meshgrid(nodes_2d_x, nodes_2d_y, indexing="ij")
    weights = np.outer(ws, ws) / (2.0 * np.pi)
    # phi_a(r) phi_b(r) = (omega/pi) * exp(-(omega/2)|r-R_a|^2 - (omega/2)|r-R_b|^2)
    # For a=b=L: pdf(r) is exactly omega/pi exp(-omega |r-R_L|^2). So shift
    # the sample points to be centred at R_L.

    if off_diagonal:
        # phi_L(r) phi_R(r) = (omega/pi) exp(-omega d^2/4) exp(-omega r^2)
        # Sample r ~ N(0, 1/(2 omega) I). Then weight is
        # phi_L(r) phi_R(r) / pdf(r) where pdf(r) = (omega/pi) exp(-omega r^2).
        # => weight = exp(-omega d^2 / 4) = S.
        r_x = grid_x  # centred at origin
        r_y = grid_y
        v_at = _trap_potential(np.stack([r_x, r_y], axis=-1), cfg)
        pot = float((weights * v_at).sum() * s)
    else:
        # phi_L(r)^2 = (omega/pi) exp(-omega |r-R_L|^2). Shift sampling to R_L.
        r_x = grid_x - half  # centred at R_L = (-d/2, 0)
        r_y = grid_y
        v_at = _trap_potential(np.stack([r_x, r_y], axis=-1), cfg)
        pot = float((weights * v_at).sum())

    return kinetic + pot


# ---------------------------------------------------------------------------
# Two-body Coulomb integrals (4D Gauss-Hermite)
# ---------------------------------------------------------------------------


def two_body_LRLR(cfg: HLConfig) -> float:
    """Direct Coulomb integral :math:`Q_V = \\langle\\phi_L\\phi_R | V_{12} | \\phi_L\\phi_R\\rangle`.

    The 4D integral collapses to a single 2D quadrature on the relative
    coordinate. With :math:`\\tilde r_i = r_i - R_i` we have
    :math:`|\\phi_L(r_1)|^2 |\\phi_R(r_2)|^2 = (\\omega/\\pi)^2
    e^{-\\omega(\\tilde r_1^2 + \\tilde r_2^2)}` and the centre-of-mass /
    relative split :math:`\\tilde r_1^2 + \\tilde r_2^2 = 2\\tilde R^2 +
    \\tilde\\rho^2/2` exposes a Gaussian
    :math:`(\\omega/\\pi)^2 \\cdot \\pi/(2\\omega) = \\omega/(2\\pi)` after the
    centre-of-mass integration. The remaining 2D integral over the relative
    coordinate :math:`\\rho' = \\tilde\\rho + \\Delta` (with :math:`\\Delta =
    R_L - R_R`, magnitude :math:`d`) is

    .. math::

        Q_V = \\kappa \\, \\mathbb E_{\\rho' \\sim \\mathcal N(\\Delta, (1/\\omega) I)}
        \\big[1/\\sqrt{|\\rho'|^2 + \\epsilon^2}\\big].

    Because :math:`|\\rho'|` concentrates around :math:`d \\gg \\epsilon`, the
    integrand is smooth and standard 2D Gauss-Hermite converges fast.
    """
    omega = cfg.omega
    kappa = cfg.kappa
    eps = cfg.epsilon
    sep = cfg.sep
    n = cfg.n_quad_per_axis
    xs, ws = hermegauss(n)
    sigma_rel = 1.0 / np.sqrt(omega)
    delta = np.array([-sep, 0.0])  # R_L - R_R = (-d, 0)
    grid_x, grid_y = np.meshgrid(xs, xs, indexing="ij")
    rho_x = delta[0] + sigma_rel * grid_x
    rho_y = delta[1] + sigma_rel * grid_y
    r2 = rho_x**2 + rho_y**2
    integrand = kappa / np.sqrt(r2 + eps**2)
    w2 = np.outer(ws, ws) / (2.0 * np.pi)
    return float((w2 * integrand).sum())


def two_body_LRRL(cfg: HLConfig) -> float:
    """Exchange Coulomb :math:`K_V = \\langle\\phi_L\\phi_R | V_{12} | \\phi_R\\phi_L\\rangle`.

    Each orbital pair on a single electron, :math:`\\phi_L(r)\\phi_R(r)`,
    factorises into a centred 2D Gaussian times :math:`S`:

    .. math::

        \\phi_L(r)\\phi_R(r) = S \\cdot (\\omega/\\pi) e^{-\\omega r^2}.

    The 4D integral therefore reduces, after the centre-of-mass integration,
    to a 1D Bessel-free integral over the relative coordinate magnitude

    .. math::

        K_V = S^2 \\, \\omega \\kappa
        \\int_0^\\infty \\rho \\, e^{-\\omega \\rho^2/2}
        \\frac{d\\rho}{\\sqrt{\\rho^2 + \\epsilon^2}}.

    With the substitution :math:`v = \\rho^2 + \\epsilon^2` the integral has
    the closed form

    .. math::

        K_V = S^2 \\kappa \\sqrt{\\pi\\omega/2} \\,
        e^{\\omega \\epsilon^2 / 2} \\, \\text{erfc}(\\epsilon \\sqrt{\\omega/2}),

    which is what we evaluate (numerically stable for any :math:`\\epsilon \\ge 0`).
    """
    omega = cfg.omega
    kappa = cfg.kappa
    eps = cfg.epsilon
    s = overlap_S(cfg)
    arg = eps * np.sqrt(omega / 2.0)
    factor = float(np.sqrt(np.pi * omega / 2.0) * np.exp(omega * eps**2 / 2.0) * erfc(arg))
    return float(s**2 * kappa * factor)


# ---------------------------------------------------------------------------
# Heitler-London driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HLResult:
    """Heitler-London output bundle for one geometry."""

    sep: float
    omega: float
    kappa: float
    epsilon: float
    smooth_t: float
    overlap_S: float
    h_LL: float
    h_LR: float
    Q_V: float
    K_V: float
    E_singlet: float
    E_triplet: float
    J_HL: float
    SS_singlet: float = -0.75
    SS_triplet: float = 0.25

    def to_dict(self) -> dict:
        return {
            "sep": self.sep,
            "omega": self.omega,
            "kappa": self.kappa,
            "epsilon": self.epsilon,
            "smooth_t": self.smooth_t,
            "overlap_S": self.overlap_S,
            "h_LL": self.h_LL,
            "h_LR": self.h_LR,
            "Q_V": self.Q_V,
            "K_V": self.K_V,
            "E_singlet": self.E_singlet,
            "E_triplet": self.E_triplet,
            "J_HL": self.J_HL,
            "SS_singlet": self.SS_singlet,
            "SS_triplet": self.SS_triplet,
        }


def heitler_london(cfg: HLConfig) -> HLResult:
    """Evaluate Heitler-London singlet/triplet energies + Heisenberg J.

    All integrals via Gauss-Hermite quadrature with
    ``cfg.n_quad_per_axis`` nodes per axis. The default of 24 is more than
    enough for omega ~ 1, sep ~ 1-12, kappa ~ 1, epsilon ~ 1e-2.
    """
    s = overlap_S(cfg)
    h_ll = one_body_h_LX(cfg, off_diagonal=False)
    h_lr = one_body_h_LX(cfg, off_diagonal=True)
    q_v = two_body_LRLR(cfg)
    k_v = two_body_LRRL(cfg)

    Q = 2.0 * h_ll + q_v
    K = 2.0 * s * h_lr + k_v

    e_s = (Q + K) / (1.0 + s**2)
    e_t = (Q - K) / (1.0 - s**2)
    j_hl = e_t - e_s

    return HLResult(
        sep=cfg.sep,
        omega=cfg.omega,
        kappa=cfg.kappa,
        epsilon=cfg.epsilon,
        smooth_t=cfg.smooth_t,
        overlap_S=s,
        h_LL=h_ll,
        h_LR=h_lr,
        Q_V=q_v,
        K_V=k_v,
        E_singlet=e_s,
        E_triplet=e_t,
        J_HL=j_hl,
    )


def spin_correlator_S1S2(sector: Literal["singlet", "triplet"]) -> float:
    """Return :math:`\\langle\\mathbf S_1 \\cdot \\mathbf S_2\\rangle`.

    Determined purely by the total-spin sector — :math:`-3/4` for singlet,
    :math:`+1/4` for any triplet ``M = -1, 0, +1``.
    """
    if sector == "singlet":
        return -0.75
    if sector == "triplet":
        return 0.25
    raise ValueError(f"unknown sector '{sector}', expected 'singlet' or 'triplet'.")


def heitler_london_density(
    cfg: HLConfig,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    sector: Literal["singlet", "triplet"] = "singlet",
) -> np.ndarray:
    """One-body density :math:`n(\\mathbf r)` for the singlet/triplet HL state.

    Returns
    -------
    density
        Array of shape ``(len(grid_x), len(grid_y))`` evaluating
        :math:`n(\\mathbf r) = 2 \\int |\\Psi(\\mathbf r, \\mathbf r_2)|^2 d\\mathbf r_2`.
    """
    s = overlap_S(cfg)
    half = cfg.half_separation()
    omega = cfg.omega
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="ij")
    r = np.stack([gx, gy], axis=-1)
    r_l = r.copy()
    r_l[..., 0] = r_l[..., 0] + half
    r_r = r.copy()
    r_r[..., 0] = r_r[..., 0] - half
    phi_l = np.sqrt(omega / np.pi) * np.exp(-0.5 * omega * (r_l**2).sum(axis=-1))
    phi_r = np.sqrt(omega / np.pi) * np.exp(-0.5 * omega * (r_r**2).sum(axis=-1))
    phi_l_sq = phi_l**2
    phi_r_sq = phi_r**2
    cross = phi_l * phi_r
    if sector == "singlet":
        return (phi_l_sq + phi_r_sq + 2.0 * s * cross) / (1.0 + s**2)
    if sector == "triplet":
        return (phi_l_sq + phi_r_sq - 2.0 * s * cross) / (1.0 - s**2)
    raise ValueError(f"unknown sector '{sector}'")
