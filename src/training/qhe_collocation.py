"""Local energy computation for quantum Hall electrons with orbital B field.

For the Hamiltonian in symmetric gauge A = B/2(-y, x):

    H = -½ Σᵢ ∇ᵢ² - B/2 Σᵢ Lz_i + B²/8 Σᵢ rᵢ² + V_Coulomb

where Lz_i = xᵢ∂_{yᵢ} - yᵢ∂_{xᵢ}  (classical angular momentum derivative)

For complex log Ψ = F + iΦ (F = log|Ψ|, Φ = phase):

    Real part of local kinetic energy:
      T_L_re = -½(|∇F|² - |∇Φ|²) - ½∇²F - B/2 Lz F + B²r²/8

    Imaginary part (should → 0 for eigenstates):
      T_L_im = -(∇F·∇Φ + ½∇²Φ) - B/2 Lz Φ

    Full local energy (real part):
      E_L = T_L_re + V_Coulomb + V_trap

Training objective: minimise Var(E_L_re) + λ × ⟨E_L_im²⟩
The second term enforces that Ψ is a true eigenstate (not just variational).
"""
from __future__ import annotations

from typing import Callable

import torch


def _grad_and_lap(
    log_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ∇log_fn(x) and ∇²log_fn(x) via autograd.

    Returns:
        grad: (B, N, 2) first derivatives ∂log/∂x_ij
        lap:  (B,)       Laplacian Σᵢⱼ ∂²log/∂x_ij²
    """
    x_req = x.detach().clone().requires_grad_(True)
    f = log_fn(x_req)
    if f.ndim == 2:
        f = f.squeeze(-1)

    grad = torch.autograd.grad(f.sum(), x_req, create_graph=True)[0]  # (B, N, 2)

    B, N, d = x_req.shape
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            g2 = torch.autograd.grad(
                grad[:, i, j].sum(), x_req,
                create_graph=True, retain_graph=True
            )[0]
            lap = lap + g2[:, i, j]

    return grad, lap


def _lz_log(
    log_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute Σᵢ (xᵢ ∂_{yᵢ} - yᵢ ∂_{xᵢ}) log_fn(x).

    This is the classical angular momentum derivative of log Ψ.

    Returns:
        (B,) tensor
    """
    x_req = x.detach().clone().requires_grad_(True)
    f = log_fn(x_req)
    if f.ndim == 2:
        f = f.squeeze(-1)

    grad = torch.autograd.grad(f.sum(), x_req, create_graph=True)[0]  # (B, N, 2)
    # grad[:, i, 0] = ∂f/∂xᵢ,  grad[:, i, 1] = ∂f/∂yᵢ
    # Lz contribution from particle i: xᵢ(∂f/∂yᵢ) - yᵢ(∂f/∂xᵢ)
    lz = (x_req[:, :, 0] * grad[:, :, 1] - x_req[:, :, 1] * grad[:, :, 0]).sum(dim=1)  # (B,)
    return lz


def _coulomb_potential(x: torch.Tensor) -> torch.Tensor:
    """1/r Coulomb repulsion between all pairs. Returns (B,)."""
    B, N, d = x.shape
    xi = x.unsqueeze(2)
    xj = x.unsqueeze(1)
    r2 = ((xi - xj) ** 2).sum(dim=-1)  # (B, N, N)
    eps = 1e-10
    r = torch.sqrt(r2 + eps)
    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1)
    return (1.0 / r)[:, triu].sum(dim=-1)  # (B,)


def _trap_potential(x: torch.Tensor, omega: float = 1.0, B: float = 0.0) -> torch.Tensor:
    """Combined harmonic trap + diamagnetic (B²r²/8) potential.

    With orbital B field, the effective radial potential is:
      V_eff = ω²r²/2 + B²r²/8 = (ω² + B²/4)/2 × r²  (renormalised frequency)

    But we keep them separate to allow ω=0 (pure LLL limit).
    """
    r2 = (x * x).sum(dim=-1).sum(dim=-1)  # (B,)
    trap = 0.5 * omega ** 2 * r2
    diamagnetic = B ** 2 / 8.0 * r2
    return trap + diamagnetic


def qhe_local_energy(
    log_amplitude_fn: Callable[[torch.Tensor], torch.Tensor],
    phase_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    B: float = 1.0,
    omega: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the local energy for a QHE wavefunction Ψ = |Ψ|×exp(iΦ).

    Args:
        log_amplitude_fn: callable x → log|Ψ(x)|, shape (B,)
        phase_fn:         callable x → Φ(x) = m Σ arg(z_i-z_j), shape (B,)
        x:                (batch, N, 2) electron positions
        B:                perpendicular magnetic field strength
        omega:            harmonic trap frequency (set 0 for pure LLL)

    Returns:
        e_real: (B,) real part of local energy  [used as training target]
        e_imag: (B,) imaginary part             [should → 0 for eigenstates]
    """
    # Gradients and Laplacians of log|Ψ| and phase
    grad_F, lap_F = _grad_and_lap(log_amplitude_fn, x)    # (B,N,2), (B,)
    grad_Phi, lap_Phi = _grad_and_lap(phase_fn, x)        # (B,N,2), (B,)

    # |∇F|² and |∇Φ|²
    grad_F_sq = (grad_F * grad_F).sum(dim=(1, 2))          # (B,)
    grad_Phi_sq = (grad_Phi * grad_Phi).sum(dim=(1, 2))    # (B,)
    grad_cross = (grad_F * grad_Phi).sum(dim=(1, 2))       # (B,)  ∇F·∇Φ

    # Angular momentum Σᵢ (xᵢ ∂yᵢ - yᵢ ∂xᵢ) applied to F and Φ
    # Reuse grads already computed
    lz_F = (x[:, :, 0] * grad_F[:, :, 1] - x[:, :, 1] * grad_F[:, :, 0]).sum(dim=1)    # (B,)
    lz_Phi = (x[:, :, 0] * grad_Phi[:, :, 1] - x[:, :, 1] * grad_Phi[:, :, 0]).sum(dim=1)  # (B,)

    # Potential energy
    v_coulomb = _coulomb_potential(x)
    v_total = v_coulomb + _trap_potential(x, omega=omega, B=0.0)  # diamagnetic already in T_re

    # Real part of kinetic local energy
    # T_L_re = -½(|∇F|² - |∇Φ|²) - ½∇²F - B/2 Lz F + B²r²/8
    r2_sum = (x * x).sum(dim=(1, 2))  # (B,)  Σᵢ rᵢ²
    t_re = (
        -0.5 * (grad_F_sq - grad_Phi_sq)
        - 0.5 * lap_F
        - B / 2.0 * lz_F
        + B ** 2 / 8.0 * r2_sum
    )

    # Imaginary part (penalty term; should vanish for true eigenstates)
    # T_L_im = -(∇F·∇Φ + ½∇²Φ) - B/2 Lz Φ
    t_im = -(grad_cross + 0.5 * lap_Phi) - B / 2.0 * lz_Phi

    e_real = t_re + v_total
    e_imag = t_im

    return e_real, e_imag


def qhe_loss(
    log_amplitude_fn: Callable[[torch.Tensor], torch.Tensor],
    phase_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    B: float = 1.0,
    omega: float = 0.0,
    imag_penalty: float = 1.0,
    clip_width: float = 5.0,
) -> tuple[torch.Tensor, float, torch.Tensor, dict]:
    """Energy variance loss for QHE training.

    Returns (loss, mean_energy, local_energies, diagnostics).
    """
    e_real, e_imag = qhe_local_energy(log_amplitude_fn, phase_fn, x, B=B, omega=omega)

    # Clip by MAD for stability
    if clip_width > 0:
        med = e_real.median()
        mad = (e_real - med).abs().median()
        e_clipped = e_real.clamp(med - clip_width * mad, med + clip_width * mad)
    else:
        e_clipped = e_real

    E_mean = e_clipped.mean()
    loss_var = ((e_clipped - E_mean.detach()) ** 2).mean()

    # Penalise variance of E_L_im (not MSE): Lz_Phi = m*N(N-1)/2 is a
    # configuration-independent constant so the PINN (amplitude-only) can never
    # reduce the mean of e_imag.  Variance measures deviation from eigenstatehood.
    # Clip e_imag by MAD too — single bad sample can spike loss_imag to 1e6+.
    if clip_width > 0:
        med_i = e_imag.detach().median()
        mad_i = (e_imag.detach() - med_i).abs().median().clamp_min(1e-8)
        e_imag = e_imag.clamp(med_i - clip_width * mad_i, med_i + clip_width * mad_i)
    e_imag_centered = e_imag - e_imag.detach().mean()
    loss_imag = (e_imag_centered ** 2).mean()
    loss = loss_var + imag_penalty * loss_imag

    diag = {
        "energy": float(E_mean),
        "energy_var": float(loss_var),
        "imag_penalty": float(loss_imag),
        "imag_mean": float(e_imag.mean()),
    }
    return loss, float(E_mean), e_clipped, diag
