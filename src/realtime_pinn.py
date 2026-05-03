"""Real-time TDSE evolution of a fermionic NQS via PDE-residual PINN.

This is the **T1 moonshot** of the 2026-04-28 pivot
(``reports/2026-04-28_pivot_realtime_nqs.md``). It branches the physics from
``src/imaginary_time_pinn.py`` (where ``g(x, τ) ∈ ℝ`` solves
``∂_τ g + (E_L − E_ref) = 0``) into the **real-time** Schrödinger equation
``i ∂_t ψ = H ψ`` by promoting the network output to a complex correction
``g(x, t) = g_R(x, t) + i g_I(x, t)`` and solving the coupled PDE pair derived
below.

The non-MCMC, deterministic-sampling, PDE-residual approach is what makes
this distinctive: standard real-time NQS work uses MCMC + stochastic
reconfiguration / TDVP linear solves; here the network learns the *entire*
trajectory `t → ψ(·, t)` from collocation residuals on (x, t) — no
SR, no time-stepping, no MCMC.

Physics
-------
Setup. ``ψ(x, t) = exp(log ψ_0(x) + g(x, t))`` with ``ψ_0`` the trained
ground state of ``H_0`` and the system evolved under
``H = H_0 + ΔV(x)`` (sudden quench at ``t = 0``). The TDSE becomes
``∂_t g = -i E_L`` with

.. math::

    E_L = E_L^{(0)} + \\Delta V
        - \\tfrac{1}{2}\\nabla^2 g
        - (\\nabla \\log\\psi_0)\\cdot\\nabla g
        - \\tfrac{1}{2}(\\nabla g)^2.

Splitting ``g = g_R + i g_I`` and writing ``a := ∇log ψ_0``, the real and
imaginary parts give two coupled real PDEs (``Im[E_L] = ∂_t g_R`` and
``-Re[E_L] = ∂_t g_I``):

.. math::

    \\partial_t g_R &= -\\tfrac{1}{2}\\nabla^2 g_I - a\\cdot\\nabla g_I
                       - \\nabla g_R \\cdot \\nabla g_I, \\\\
    \\partial_t g_I &= -E_L^{(0)} - \\Delta V
                       + \\tfrac{1}{2}\\nabla^2 g_R + a\\cdot\\nabla g_R
                       + \\tfrac{1}{2}(|\\nabla g_R|^2 - |\\nabla g_I|^2).

Initial condition: ``g_R(x, 0) = g_I(x, 0) = 0``.

Sanity check (``ΔV = 0``, ``ψ_0`` is an exact eigenstate). Then
``E_L^{(0)} ≡ E_0`` and the analytical solution is ``g_R = 0``,
``g_I = -E_0 t`` (a global phase ``ψ(x, t) = e^{-i E_0 t}\\psi_0(x)``). This
is the smoke test wired into ``tests/test_realtime_pinn.py``.

Module surface
--------------
* :class:`RealTimeNet` — FiLM-conditioned ``(x, t) → (g_R, g_I)`` MLP.
* :func:`compute_realtime_residual` — autograd evaluation of both PDE
  residuals on a (x, t) batch, given precomputed ``E_L^{(0)}`` and
  ``∇log ψ_0``.
* :func:`train_realtime_pinn` — Adam loop with the IC penalty.
* :func:`evaluate_realtime_state` — returns ``log|ψ(x,t)|`` and ``arg ψ(x,t)``
  diagnostics on an (x, t) grid (for ED comparison).

This module is **self-contained** with respect to the PDE: the heavy
``ψ_0`` machinery (Slater, Backflow, PINN baseline) is loaded by callers from
``src.wavefunction`` and ``src.imaginary_time_pinn``. The split keeps this
file readable (≈400 LOC) and testable without GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small helper: time embedding shared with the imaginary-time PINN
# ---------------------------------------------------------------------------


class TimeEmbedding(nn.Module):
    """Sinusoidal + linear time embedding for FiLM conditioning."""

    def __init__(self, embed_dim: int = 32, n_freq: int = 6, t_scale: float = 1.0) -> None:
        super().__init__()
        if embed_dim < 1 + 2 * n_freq:
            raise ValueError(
                f"embed_dim={embed_dim} is too small for n_freq={n_freq} "
                f"(need >= {1 + 2 * n_freq})."
            )
        # Geometric frequency ladder, doubling each stage.
        freqs = 2.0 ** torch.arange(n_freq, dtype=torch.get_default_dtype())
        self.register_buffer("freqs", freqs * t_scale)
        self.proj = nn.Linear(1 + 2 * n_freq, embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)
        phases = t * self.freqs
        embed = torch.cat([t, torch.sin(phases), torch.cos(phases)], dim=-1)
        return self.proj(embed)


# ---------------------------------------------------------------------------
# Two-channel (real/imag) FiLM-conditioned network
# ---------------------------------------------------------------------------


class RealTimeNet(nn.Module):
    """Network ``g(x, t) = (g_R(x, t), g_I(x, t))`` with hard IC at t=0.

    The IC ``g_R(x, 0) = g_I(x, 0) = 0`` is enforced *exactly* by multiplying
    the raw network output by ``t`` (no IC penalty term needed). This trick
    massively improves trainability and removes one hyperparameter.

    Parameters
    ----------
    n_particles, dim
        Geometry of the configuration space.
    hidden, n_layers, t_embed, n_freq, t_scale
        FiLM-MLP hyperparameters.
    output_scale
        Scalar multiplied into the raw two-channel output. Default 1.0; lower
        values can help the optimiser when E_0 t spans a large range.
    """

    def __init__(
        self,
        n_particles: int,
        dim: int,
        hidden: int = 64,
        n_layers: int = 3,
        t_embed: int = 32,
        n_freq: int = 6,
        t_scale: float = 1.0,
        output_scale: float = 1.0,
        use_quadratic_features: bool = False,
    ) -> None:
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.use_quadratic_features = use_quadratic_features
        n_lin = n_particles * dim
        # Quadratic features: per-coordinate x_i², per-particle |x_i|², and
        # the cloud variance Σ_i |x_i|². These exact polynomial features make
        # the breathing/squeezing dynamics easy to represent — the analytical
        # solution for separable Gaussian quenches has g(x, t) quadratic in
        # x, so providing x² up front removes the depth needed to synthesise
        # it through nonlinearities.
        n_quad = n_particles * dim + n_particles + 1 if use_quadratic_features else 0
        input_dim = n_lin + n_quad
        self.spatial_proj = nn.Linear(input_dim, hidden)
        self.t_embed = TimeEmbedding(t_embed, n_freq, t_scale=t_scale)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.film_heads = nn.ModuleList(
            [nn.Linear(t_embed, 2 * hidden) for _ in range(n_layers)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 2)
        )
        # Final layer initialised small so the IC is well-respected before
        # the t-multiplier kicks in.
        nn.init.normal_(self.output[-1].weight, std=0.01)
        nn.init.zeros_(self.output[-1].bias)
        self.output_scale = output_scale

    def _featurise(self, x: torch.Tensor) -> torch.Tensor:
        """Build the spatial input features (linear + quadratic)."""
        b = x.shape[0]
        feats = [x.reshape(b, -1)]
        if self.use_quadratic_features:
            x_sq = x**2  # (B, N, d)
            feats.append(x_sq.reshape(b, -1))                 # (B, N·d)
            feats.append(x_sq.sum(dim=-1))                    # (B, N), per-particle |x|²
            feats.append(x_sq.sum(dim=(1, 2), keepdim=False).unsqueeze(-1))  # (B, 1)
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(g_R, g_I)`` of shape ``(B,)`` each."""
        if x.ndim != 3:
            raise ValueError(f"x must be (B, N, d); got {x.shape}.")
        if t.ndim != 1:
            raise ValueError(f"t must be (B,); got {t.shape}.")
        if x.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch mismatch: x has {x.shape[0]}, t has {t.shape[0]}."
            )

        h = F.gelu(self.spatial_proj(self._featurise(x)))
        emb = self.t_embed(t)
        for linear, norm, head in zip(self.layers, self.norms, self.film_heads, strict=True):
            gb = head(emb)
            gamma, beta = gb.chunk(2, dim=-1)
            gamma = 1.0 + 0.1 * gamma
            h = linear(h)
            h = norm(h)
            h = gamma * h + beta
            h = F.gelu(h)
        raw = self.output(h) * self.output_scale  # (B, 2)
        # Hard IC: multiply by t so g(x, 0) = 0 by construction.
        g = raw * t.unsqueeze(-1)
        return g[:, 0], g[:, 1]


class PolynomialQuenchNet(nn.Module):
    """Polynomial ansatz for separable Gaussian quenches.

    For Hamiltonians where the exact time evolution preserves a Gaussian
    structure (e.g. the harmonic-oscillator ω-quench, position quench, or
    general quadratic potentials), the analytical form is

    .. math::

        g(x, t) = c_R(t) + \\alpha_R(t)\\,r^2(x)
                + i \\,(c_I(t) + \\alpha_I(t)\\,r^2(x))

    with ``r^2(x) = \\sum_i |x_i|^2``. This network parameterises exactly that
    structure: a small time-embedding MLP outputs the four scalar coefficients
    ``(c_R, c_I, \\alpha_R, \\alpha_I)``, which are then assembled into
    ``(g_R, g_I)`` via the closed-form expression above. The hard initial
    condition ``g(x, 0) = 0`` is preserved by the same ``× t`` trick as
    :class:`RealTimeNet`.

    Use this for **validation** of the realtime PDE-residual machinery on
    problems where the analytical answer is polynomial in ``r^2``. For
    Coulomb-coupled systems the polynomial ansatz is no longer exact and a
    generic :class:`RealTimeNet` should be used instead (or the polynomial
    ansatz extended with an additive MLP residual).
    """

    def __init__(
        self,
        n_particles: int,
        dim: int,
        hidden: int = 32,
        t_embed: int = 24,
        n_freq: int = 4,
        t_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.t_embed = TimeEmbedding(t_embed, n_freq, t_scale=t_scale)
        # Tiny MLP from time embedding to four scalar coefficients.
        self.coeff_net = nn.Sequential(
            nn.Linear(t_embed, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.normal_(self.coeff_net[-1].weight, std=0.01)
        nn.init.zeros_(self.coeff_net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must be (B, N, d); got {x.shape}.")
        if t.ndim != 1:
            raise ValueError(f"t must be (B,); got {t.shape}.")
        if x.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch mismatch: x has {x.shape[0]}, t has {t.shape[0]}."
            )

        emb = self.t_embed(t)
        coeffs = self.coeff_net(emb)  # (B, 4)
        c_R, c_I, alpha_R, alpha_I = coeffs.unbind(dim=-1)
        r2 = (x**2).sum(dim=(1, 2))
        g_R = (c_R + alpha_R * r2) * t
        g_I = (c_I + alpha_I * r2) * t
        return g_R, g_I


class HybridPolyMLPNet(nn.Module):
    """Polynomial backbone + MLP residual for non-separable quenches.

    For Hamiltonians close to (but not exactly) separable Gaussian — the
    canonical example being a 2D HO with a softcore-Coulomb pair interaction
    suddenly switched on — most of the dynamics still tracks the breathing
    mode that :class:`PolynomialQuenchNet` captures exactly, while genuinely
    non-Gaussian, particle-correlated structure shows up as a small additive
    correction. This network parameterises that decomposition explicitly:

    .. math::

        g(x, t) = g^{\\text{poly}}(x, t) + s_\\text{res} \\cdot g^{\\text{MLP}}(x, t),

    where ``g^{\\text{poly}}`` is the :class:`PolynomialQuenchNet` output
    (with its own hard ``× t`` IC) and ``g^{\\text{MLP}}`` is a fresh
    :class:`RealTimeNet` (with the same ``× t`` hard IC). ``s_res`` is a
    fixed scalar that biases the optimiser toward leaving the polynomial
    backbone in charge unless the residual *has* to grow.

    Both sub-networks already enforce ``g(x, 0) = 0``, so the hybrid
    inherits that exactly. Setting ``residual_scale = 0`` (and freezing the
    MLP) reproduces :class:`PolynomialQuenchNet` bit-for-bit: this is the
    "ablation knob" used in the unit tests.
    """

    def __init__(
        self,
        n_particles: int,
        dim: int,
        *,
        poly_hidden: int = 32,
        poly_t_embed: int = 24,
        poly_n_freq: int = 4,
        mlp_hidden: int = 48,
        mlp_n_layers: int = 3,
        mlp_t_embed: int = 24,
        mlp_n_freq: int = 4,
        mlp_use_quadratic_features: bool = False,
        t_scale: float = 1.0,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.residual_scale = float(residual_scale)
        self.poly = PolynomialQuenchNet(
            n_particles=n_particles,
            dim=dim,
            hidden=poly_hidden,
            t_embed=poly_t_embed,
            n_freq=poly_n_freq,
            t_scale=t_scale,
        )
        self.mlp = RealTimeNet(
            n_particles=n_particles,
            dim=dim,
            hidden=mlp_hidden,
            n_layers=mlp_n_layers,
            t_embed=mlp_t_embed,
            n_freq=mlp_n_freq,
            t_scale=t_scale,
            output_scale=1.0,
            use_quadratic_features=mlp_use_quadratic_features,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g_R_p, g_I_p = self.poly(x, t)
        g_R_m, g_I_m = self.mlp(x, t)
        g_R = g_R_p + self.residual_scale * g_R_m
        g_I = g_I_p + self.residual_scale * g_I_m
        return g_R, g_I

    def freeze_residual(self) -> None:
        """Freeze the MLP residual (used for ablation / warm-start)."""
        for param in self.mlp.parameters():
            param.requires_grad_(False)

    def unfreeze_residual(self) -> None:
        """Unfreeze the MLP residual (used after a polynomial-only warm-up)."""
        for param in self.mlp.parameters():
            param.requires_grad_(True)


# ---------------------------------------------------------------------------
# PDE residual computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealTimeResidual:
    """Per-sample residuals + diagnostics on a batched (x, t) point cloud."""

    res_R: torch.Tensor  # ∂_t g_R - Im(E_L) residual, shape (B,)
    res_I: torch.Tensor  # ∂_t g_I + Re(E_L) residual, shape (B,)
    g_R: torch.Tensor
    g_I: torch.Tensor
    E_L_real: torch.Tensor
    E_L_imag: torch.Tensor


def compute_realtime_residual(
    net: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    E_L0: torch.Tensor,
    grad_log_psi0: torch.Tensor,
    deltaV: torch.Tensor | None = None,
) -> RealTimeResidual:
    """Evaluate the coupled real-time PDE residual on a (x, t) batch.

    See module docstring for the equations. The Laplacians ``∇²g_R``,
    ``∇²g_I`` are built by autograd over the per-particle, per-axis
    components: ``N · d`` second-derivative passes per residual.

    Parameters
    ----------
    net
        Any ``nn.Module`` whose ``forward(x, t)`` returns ``(g_R, g_I)`` —
        :class:`RealTimeNet`, :class:`PolynomialQuenchNet`, and
        :class:`HybridPolyMLPNet` are all supported.
    x
        ``(B, N, d)`` collocation points (real space).
    t
        ``(B,)`` collocation times.
    E_L0
        ``(B,)`` precomputed local energy ``E_L^{(0)}(x) := (H_0 ψ_0)/ψ_0``
        evaluated at the same ``x`` on the *baseline* Hamiltonian ``H_0``.
    grad_log_psi0
        ``(B, N, d)`` precomputed ``∇ log ψ_0(x)`` on the same x.
    deltaV
        Optional ``(B,)`` quench potential ``ΔV(x) = V_final(x) - V_0(x)``.
        ``None`` means free evolution under the same ``H_0`` (smoke test).
    """
    if deltaV is None:
        deltaV = torch.zeros_like(E_L0)

    if x.shape != grad_log_psi0.shape:
        raise ValueError(
            f"grad_log_psi0 shape {grad_log_psi0.shape} does not match x shape {x.shape}."
        )
    if E_L0.shape != t.shape:
        raise ValueError(
            f"E_L0 shape {E_L0.shape} must equal t shape {t.shape}."
        )

    x = x.detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)

    g_R, g_I = net(x, t)

    def _grad_or_zero(out: torch.Tensor, wrt: torch.Tensor) -> torch.Tensor:
        """``torch.autograd.grad`` that returns zeros when the graph is empty.

        Some ansatz components may be constant in ``x`` or ``t``; PyTorch
        raises in that case unless we set ``allow_unused=True``. We do, then
        fold the resulting ``None`` to a zero tensor of the right shape.
        """
        result = torch.autograd.grad(
            out.sum(), wrt, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if result is None:
            return torch.zeros_like(wrt)
        return result

    # ∂_t g_{R,I}: scalar derivatives of the per-batch sums.
    dgR_dt = _grad_or_zero(g_R, t)
    dgI_dt = _grad_or_zero(g_I, t)

    # Spatial gradients ∇_x g_{R,I}: (B, N, d).
    dgR_dx = _grad_or_zero(g_R, x)
    dgI_dx = _grad_or_zero(g_I, x)

    # Laplacians ∇²g_{R,I}: sum of N·d second derivatives. We can short-cut
    # the second derivative when the first derivative is *exactly* zero
    # (i.e. ``g`` does not depend on ``x``), since that branch is invariant
    # under further differentiation. This both speeds up the trivial case
    # and avoids the empty-graph autograd error for oracle networks.
    b, n_p, d = x.shape
    lap_R = torch.zeros(b, dtype=x.dtype, device=x.device)
    lap_I = torch.zeros(b, dtype=x.dtype, device=x.device)
    gR_has_x_grad = dgR_dx.requires_grad and dgR_dx.abs().sum() > 0
    gI_has_x_grad = dgI_dx.requires_grad and dgI_dx.abs().sum() > 0
    for i in range(n_p):
        for j in range(d):
            if gR_has_x_grad:
                d2R = torch.autograd.grad(
                    dgR_dx[:, i, j].sum(),
                    x,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if d2R is not None:
                    lap_R = lap_R + d2R[:, i, j]
            if gI_has_x_grad:
                d2I = torch.autograd.grad(
                    dgI_dx[:, i, j].sum(),
                    x,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if d2I is not None:
                    lap_I = lap_I + d2I[:, i, j]

    a_dot_gR = (grad_log_psi0 * dgR_dx).sum(dim=(1, 2))
    a_dot_gI = (grad_log_psi0 * dgI_dx).sum(dim=(1, 2))
    gR_sq = (dgR_dx**2).sum(dim=(1, 2))
    gI_sq = (dgI_dx**2).sum(dim=(1, 2))
    gR_dot_gI = (dgR_dx * dgI_dx).sum(dim=(1, 2))

    # Re(E_L) = E_L^{(0)} + ΔV - ½∇²g_R - a·∇g_R - ½(|∇g_R|² - |∇g_I|²)
    # Im(E_L) =                    -½∇²g_I - a·∇g_I -    ∇g_R·∇g_I
    E_L_real = E_L0 + deltaV - 0.5 * lap_R - a_dot_gR - 0.5 * (gR_sq - gI_sq)
    E_L_imag = -0.5 * lap_I - a_dot_gI - gR_dot_gI

    # ∂_t g_R = Im(E_L)  →  res_R = ∂_t g_R - Im(E_L) = 0
    # ∂_t g_I = -Re(E_L) →  res_I = ∂_t g_I + Re(E_L) = 0
    res_R = dgR_dt - E_L_imag
    res_I = dgI_dt + E_L_real

    return RealTimeResidual(
        res_R=res_R,
        res_I=res_I,
        g_R=g_R,
        g_I=g_I,
        E_L_real=E_L_real,
        E_L_imag=E_L_imag,
    )


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


@dataclass
class RealTimeTrainConfig:
    """Hyperparameters for :func:`train_realtime_pinn`.

    The optional ``norm_weight`` activates a *unitarity regularizer*:

        L_norm(t) = (⟨e^{2 g_R(x, t)}⟩_{|ψ_0|²} - 1)²

    which is an exact identity for unitary evolution (the time-evolved state
    is normalised in the same Born measure as ψ_0 because the imaginary part
    of ``log ψ`` doesn't contribute to ``|ψ|²``). Without this term the bare
    PDE residual leaves ``Z(t) := ⟨e^{2 g_R}⟩`` free to drift, which biases
    every reweighted observable. With ``norm_weight > 0`` we evaluate the
    constraint at a single random ``t`` per step using the current batch
    ``x_b``, contributing essentially zero overhead.
    """

    n_epochs: int = 2000
    batch_pde: int = 256
    t_max: float = 1.0
    lr: float = 3.0e-3
    lr_min_ratio: float = 1.0 / 30.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    norm_weight: float = 0.0
    anchor_weight: float = 0.0
    anchor_t_frac: float = 0.02
    print_every: int = 50
    seed: int | None = 0
    history_every: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


def train_realtime_pinn(
    net: nn.Module,
    *,
    x_pool: torch.Tensor,
    E_L0_pool: torch.Tensor,
    grad_log_psi0_pool: torch.Tensor,
    deltaV_pool: torch.Tensor | None = None,
    train_cfg: RealTimeTrainConfig,
    log_fn: Any = print,
) -> dict[str, Any]:
    """Train the network on the coupled real-time PDE residuals.

    The IC ``g(x, 0) = 0`` is hardcoded by :class:`RealTimeNet`'s
    ``output × t`` factor, so the loss is purely the PDE residual.

    Parameters
    ----------
    net
        :class:`RealTimeNet` instance, already on the desired device/dtype.
    x_pool
        ``(P, N, d)`` precomputed collocation points (typically VMC samples
        of the *initial* state ``ψ_0``).
    E_L0_pool
        ``(P,)`` precomputed local energies on the baseline ``H_0``.
    grad_log_psi0_pool
        ``(P, N, d)`` precomputed ``∇log ψ_0``.
    deltaV_pool
        Optional ``(P,)`` precomputed quench shifts. ``None`` ⇒ free evolution.
    train_cfg
        :class:`RealTimeTrainConfig` instance.
    log_fn
        Callable for status messages (default ``print``).

    Returns
    -------
    dict
        ``{'history': {'loss', 'res_R_rms', 'res_I_rms', 'epoch'}, 'final_loss': float}``.
    """
    pool_size = x_pool.shape[0]
    if pool_size == 0:
        raise ValueError("x_pool is empty.")
    if E_L0_pool.shape[0] != pool_size or grad_log_psi0_pool.shape[0] != pool_size:
        raise ValueError("Pool tensors must agree on the leading dim.")
    if deltaV_pool is not None and deltaV_pool.shape[0] != pool_size:
        raise ValueError("deltaV_pool leading dim must match x_pool.")

    device = x_pool.device
    dtype = x_pool.dtype

    if train_cfg.seed is not None:
        torch.manual_seed(train_cfg.seed)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.n_epochs,
        eta_min=train_cfg.lr * train_cfg.lr_min_ratio,
    )

    history: dict[str, list[float] | list[int]] = {
        "loss": [],
        "res_R_rms": [],
        "res_I_rms": [],
        "norm_loss": [],
        "Z_at_norm_t": [],
        "anchor_loss": [],
        "epoch": [],
    }

    use_norm = train_cfg.norm_weight > 0.0
    use_anchor = train_cfg.anchor_weight > 0.0
    t_anchor = float(train_cfg.anchor_t_frac * train_cfg.t_max)

    for epoch in range(train_cfg.n_epochs):
        idx = torch.randint(pool_size, (train_cfg.batch_pde,), device=device)
        x_b = x_pool[idx]
        E_L0_b = E_L0_pool[idx]
        grad_b = grad_log_psi0_pool[idx]
        dV_b = deltaV_pool[idx] if deltaV_pool is not None else None

        t_b = torch.rand(train_cfg.batch_pde, dtype=dtype, device=device) * train_cfg.t_max

        residual = compute_realtime_residual(
            net,
            x_b,
            t_b,
            E_L0=E_L0_b,
            grad_log_psi0=grad_b,
            deltaV=dV_b,
        )
        res_loss = (residual.res_R**2).mean() + (residual.res_I**2).mean()

        norm_loss_t = torch.zeros((), dtype=dtype, device=device)
        Z_t = torch.ones((), dtype=dtype, device=device)
        if use_norm:
            t_norm_scalar = torch.rand(1, dtype=dtype, device=device) * train_cfg.t_max
            t_norm_b = t_norm_scalar.expand(train_cfg.batch_pde)
            g_R_norm, _ = net(x_b, t_norm_b)
            # Use the *log-stabilised* form (log Z(t))² so the gradient is
            # ``∝ g_R · log Z`` rather than ``∝ exp(2 g_R) · (Z-1)``, which
            # is much steadier when g_R has a few positive outliers.
            Z_t = torch.exp(2.0 * g_R_norm).mean()
            log_Z = torch.log(Z_t.clamp_min(1e-12))
            norm_loss_t = log_Z**2

        anchor_loss_t = torch.zeros((), dtype=dtype, device=device)
        if use_anchor:
            # Linearised analytic solution at small t:
            #   g_R(x, t_a) ≈ 0
            #   g_I(x, t_a) ≈ -(E_L^{(0)}(x) + ΔV(x)) · t_a
            # We pin both via a least-squares term. This is *free knowledge*
            # — every quantity is precomputed — and dramatically straightens
            # the loss landscape near the IC where the bare PDE residual is
            # under-determined.
            t_a_b = torch.full(
                (train_cfg.batch_pde,), t_anchor, dtype=dtype, device=device
            )
            g_R_a, g_I_a = net(x_b, t_a_b)
            target_I = -(E_L0_b + (dV_b if dV_b is not None else 0.0)) * t_anchor
            anchor_loss_t = (g_R_a**2).mean() + ((g_I_a - target_I) ** 2).mean()

        loss = (
            res_loss
            + train_cfg.norm_weight * norm_loss_t
            + train_cfg.anchor_weight * anchor_loss_t
        )

        optimizer.zero_grad()
        loss.backward()
        if train_cfg.grad_clip is not None and math.isfinite(train_cfg.grad_clip):
            torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch % train_cfg.history_every == 0:
            history["loss"].append(float(loss.item()))
            history["res_R_rms"].append(float(residual.res_R.detach().pow(2).mean().sqrt().item()))
            history["res_I_rms"].append(float(residual.res_I.detach().pow(2).mean().sqrt().item()))
            history["norm_loss"].append(float(norm_loss_t.item()))
            history["Z_at_norm_t"].append(float(Z_t.item()))
            history["anchor_loss"].append(float(anchor_loss_t.item()))
            history["epoch"].append(int(epoch))

        if epoch % train_cfg.print_every == 0:
            extra_parts = []
            if use_norm:
                extra_parts.append(f"Z={Z_t.item():.4f}")
                extra_parts.append(f"L_norm={norm_loss_t.item():.3e}")
            if use_anchor:
                extra_parts.append(f"L_anch={anchor_loss_t.item():.3e}")
            extra = ("  " + "  ".join(extra_parts)) if extra_parts else ""
            log_fn(
                f"  [RT-PINN] epoch={epoch:5d}  loss={loss.item():.6e}  "
                f"|res_R|={residual.res_R.detach().pow(2).mean().sqrt().item():.4e}  "
                f"|res_I|={residual.res_I.detach().pow(2).mean().sqrt().item():.4e}  "
                f"lr={optimizer.param_groups[0]['lr']:.3e}{extra}"
            )

    final_loss = float(history["loss"][-1]) if history["loss"] else float("nan")
    return {"history": history, "final_loss": final_loss}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_realtime_state(
    net: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    log_psi0: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return ``log|ψ(x,t)|``, ``arg ψ(x,t)``, and the raw ``(g_R, g_I)``.

    With ``ψ(x,t) = exp(log ψ_0(x) + g_R + i g_I)`` we have
    ``log|ψ(x,t)| = log|ψ_0(x)| + g_R(x,t)``  and
    ``arg ψ(x,t) = arg ψ_0(x) + g_I(x,t) (mod 2π)``.
    """
    g_R, g_I = net(x, t)
    out: dict[str, torch.Tensor] = {"g_R": g_R, "g_I": g_I}
    if log_psi0 is not None:
        out["log_abs_psi"] = log_psi0 + g_R
        out["arg_psi"] = g_I  # caller may add arg ψ_0 if known.
    return out
