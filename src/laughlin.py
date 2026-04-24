"""Laughlin wavefunction base for fractional quantum Hall states.

The Laughlin state at filling ν = 1/m (m odd) for N electrons in 2D
with perpendicular magnetic field B (symmetric gauge A = B/2(-y, x)):

    Ψ_L(z₁,...,z_N) = ∏_{i<j} (z_i - z_j)^m × exp(-B/4 × Σ|z_i|²)

where z_i = x_i + i y_i are complex coordinates.

This module provides:
  - log|Ψ_L| (real, differentiable) — for variational training
  - phase_L (real, differentiable) — the Berry phase ∑ arg(z_i - z_j)
  - LaughlinJastrowWF: PINN-corrected Laughlin state Ψ = Ψ_L × exp(J_PINN)
  - qhe_local_energy: full local energy with orbital B field

Magnetic length: l_B = 1/√B  (sets the length scale for QHE physics)
Filling factor: ν = N / N_Φ = 1/m  where N_Φ = B × Area / (2π) flux quanta

Physics check: The Laughlin state has:
  - Total angular momentum L_z = -m × N(N-1)/2
  - Pair correlation g(r) ~ r^{2m} at short distances (Laughlin gap)
  - Energy gap ~ e²/(ε l_B) above the ground state (incompressible)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def laughlin_log_amplitude(
    x: torch.Tensor,
    *,
    m: int = 3,
    B: float = 1.0,
) -> torch.Tensor:
    """Compute log|Ψ_L(x)| for the Laughlin state.

    Args:
        x: (batch, N, 2) electron positions
        m: Laughlin exponent (m=3 for ν=1/3, m=5 for ν=1/5, m=1 for IQH)
        B: magnetic field strength

    Returns:
        (batch,) real tensor: log|Ψ_L|
    """
    B_t, N, d = x.shape
    assert d == 2, "Laughlin state requires 2D positions"

    # Pairwise vectors r_ij = r_i - r_j, shape (B, N, N, 2)
    xi = x.unsqueeze(2)   # (B, N, 1, 2)
    xj = x.unsqueeze(1)   # (B, 1, N, 2)
    dr = xi - xj           # (B, N, N, 2)

    # |z_i - z_j|² = (x_i-x_j)² + (y_i-y_j)²
    r2_ij = (dr * dr).sum(dim=-1)  # (B, N, N)

    # Take upper triangle i < j, add small ε to avoid log(0) at coincident points
    eps = 1e-20
    log_r_ij = 0.5 * torch.log(r2_ij + eps)   # (B, N, N)

    # Sum over i < j pairs
    triu_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1)
    log_jastrow = m * log_r_ij[:, triu_mask].sum(dim=-1)  # (B,)

    # Gaussian factor: -B/4 × Σ|r_i|²
    r2_i = (x * x).sum(dim=-1)   # (B, N)
    log_gaussian = -B / 4.0 * r2_i.sum(dim=-1)   # (B,)

    return log_jastrow + log_gaussian


def laughlin_phase(
    x: torch.Tensor,
    *,
    m: int = 3,
) -> torch.Tensor:
    """Compute the Laughlin phase φ(x) = m × Σ_{i<j} arg(z_i - z_j).

    For the full complex wavefunction: Ψ_L = |Ψ_L| × exp(i φ_L).

    Args:
        x: (batch, N, 2) electron positions [:,:,0]=x, [:,:,1]=y
        m: Laughlin exponent

    Returns:
        (batch,) real tensor: phase in radians (mod-2π not applied — raw sum)
    """
    B_t, N, d = x.shape

    # Only compute atan2 for i < j pairs to avoid atan2(0,0) at diagonal
    # whose gradient is undefined (NaN) in PyTorch.
    ii, jj = torch.triu_indices(N, N, offset=1, device=x.device)  # each (n_pairs,)
    xi = x[:, ii, :]   # (B, n_pairs, 2)
    xj = x[:, jj, :]   # (B, n_pairs, 2)
    dr = xi - xj        # (B, n_pairs, 2)

    # arg(z_i - z_j) = atan2(Δy, Δx)  — safe: r_ij > 0 for i≠j
    angles = torch.atan2(dr[..., 1], dr[..., 0])  # (B, n_pairs)
    phase = m * angles.sum(dim=-1)                  # (B,)
    return phase


def laughlin_angular_momentum(x: torch.Tensor, m: int = 3) -> float:
    """Expected total angular momentum L_z = -m × N(N-1)/2 for Laughlin state."""
    N = x.shape[1]
    return -m * N * (N - 1) / 2


class LaughlinJastrowWF(nn.Module):
    """Laughlin × PINN-Jastrow wavefunction for variational QHE.

    log|Ψ(x)| = log|Ψ_L(x)| + J_PINN(x)
    phase(x)  = phase_L(x)               (phase only from Laughlin factor)

    The PINN Jastrow J_PINN is trained to minimise E[Ψ] within the
    constraint that Ψ has the correct Laughlin structure.

    To use as a pure Laughlin test, set J_PINN=None (J=0 everywhere).
    """

    def __init__(
        self,
        n_particles: int,
        *,
        m: int = 3,
        B: float = 1.0,
        jastrow: nn.Module | None = None,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.m = m
        self.B_field = B
        self.jastrow = jastrow  # PINN Jastrow correction (optional)

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return log|Ψ(x)|, shape (batch,)."""
        logamp = laughlin_log_amplitude(x, m=self.m, B=self.B_field)
        if self.jastrow is not None:
            logamp = logamp + self.jastrow(x).squeeze(-1)
        return logamp

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Return phase φ(x) = m Σ_{i<j} arg(z_i-z_j), shape (batch,)."""
        return laughlin_phase(x, m=self.m)

    def log_psi_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Return complex log Ψ(x) = log|Ψ| + i φ, shape (batch,) complex."""
        log_amp = self.log_amplitude(x)
        phi = self.phase(x)
        return torch.complex(log_amp, phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log|Ψ(x)| for compatibility with existing training loop."""
        return self.log_amplitude(x)
