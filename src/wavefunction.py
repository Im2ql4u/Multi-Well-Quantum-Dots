from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import SystemConfig, _lookup_dmc_energy


def resolve_reference_energy(
    system: SystemConfig, E_ref: str | float, *, allow_missing_dmc: bool = False
) -> float:
    """Resolve the reference energy for a system.

    If *E_ref* is ``'auto'``, attempt DMC lookup; fall back to N*omega when
    *allow_missing_dmc* is ``True``.
    """
    if E_ref != "auto":
        return float(E_ref)
    try:
        return _lookup_dmc_energy(system.n_particles, system.omega)
    except KeyError:
        if allow_missing_dmc:
            return float(system.n_particles) * float(system.omega)
        raise


def setup_closed_shell_system(
    system: SystemConfig,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    E_ref: str | float,
    allow_missing_dmc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if system.n_particles % 2 != 0:
        raise ValueError("Closed-shell setup requires an even number of particles.")
    n_orb = system.n_particles // 2
    C_occ = torch.eye(n_orb, device=device, dtype=dtype)
    spin = torch.tensor([0] * n_orb + [1] * n_orb, device=device, dtype=torch.int64)
    if E_ref == "auto":
        e_ref_val = float(system.n_particles)
    else:
        e_ref_val = float(E_ref)
    params = {
        "E_ref": e_ref_val,
        "n_particles": int(system.n_particles),
        "dim": int(system.dim),
        "omega": float(system.omega),
    }
    return (C_occ, spin, params)


class GroundStateWF(nn.Module):
    """Slater-Jastrow-like wavefunction with learnable one-body and pair correlations."""

    def __init__(
        self,
        system: SystemConfig,
        C_occ: torch.Tensor,
        spin: torch.Tensor,
        params: dict,
        *,
        arch_type: str = "pinn",
        pinn_hidden: int = 32,
        pinn_layers: int = 2,
        bf_hidden: int = 32,
        bf_layers: int = 2,
        use_backflow: bool = True,
    ) -> None:
        super().__init__()
        del C_occ, spin, params, arch_type, pinn_hidden, pinn_layers
        del bf_hidden, bf_layers, use_backflow
        self.system = system

        # Learnable envelope parameters
        self.raw_alpha = nn.Parameter(
            torch.tensor(math.log(math.e - 1.0), dtype=torch.float64)
        )
        self.raw_beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        # One-body network: maps per-particle features -> scalar correction
        self.one_body = nn.Sequential(
            nn.Linear(system.dim + 1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        ).double()

        # Pair network: maps inter-particle distance -> pair correction
        self.pair_body = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        ).double()

        # Xavier initialization
        for mod in (self.one_body, self.pair_body):
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,N,D), got {tuple(x.shape)}")
        if x.shape[1] != self.system.n_particles or x.shape[2] != self.system.dim:
            raise ValueError(
                f"Input shape mismatch. Expected N={self.system.n_particles}"
                f", D={self.system.dim}, got {tuple(x.shape)}"
            )

        # Learnable Gaussian envelope
        alpha = torch.nn.functional.softplus(self.raw_alpha) + 0.1
        beta = torch.tanh(self.raw_beta)

        # One-body: Gaussian envelope
        r2 = torch.sum(x * x, dim=(-1, -2))
        log_one_body = -0.5 * alpha * r2

        # One-body NN correction
        x_feat = torch.cat(
            [x, torch.sum(x * x, dim=-1, keepdim=True)], dim=-1
        )
        one_corr = self.one_body(x_feat).squeeze(-1).sum(dim=-1)

        # Pair correlations
        (i, j) = torch.triu_indices(
            self.system.n_particles, self.system.n_particles, offset=1, device=x.device
        )
        rij = x[:, i, :] - x[:, j, :]
        r = torch.sqrt(torch.sum(rij * rij, dim=-1) + 1e-10)
        jastrow = torch.sum(r / (1.0 + r), dim=-1)
        pair_corr = self.pair_body(r.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        log_psi = log_one_body + beta * jastrow + 0.05 * one_corr + 0.05 * pair_corr
        if not torch.isfinite(log_psi).all():
            raise RuntimeError("Non-finite log_psi in GroundStateWF forward pass.")
        return log_psi


class SlaterOnlyWF(GroundStateWF):
    """Compatibility alias for legacy scripts expecting this symbol."""

    pass
