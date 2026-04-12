from __future__ import annotations

from typing import Sequence

import torch

from config import SystemConfig


def compute_potential(
    x: torch.Tensor,
    *,
    system: SystemConfig,
    spin: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the full potential for a SystemConfig."""
    # Reconstruct legacy parameters from SystemConfig.
    omega = system.omega
    if system.n_wells == 1:
        well_sep = 0.0
    else:
        well_sep = system.wells[1].center[0] - system.wells[0].center[0]
    magnetic_B = system.B_magnitude * system.B_direction[2]
    return compute_potential_legacy_compatible(
        x,
        omega=omega,
        well_sep=well_sep,
        smooth_T=system.smooth_T,
        coulomb=system.coulomb,
        magnetic_B=magnetic_B,
        spin=spin,
        g_factor=system.g_factor,
        mu_B=system.mu_B,
        zeeman_electron1_only=system.zeeman_electron1_only,
        zeeman_particle_indices=system.zeeman_particle_indices,
    )


def compute_potential_legacy_compatible(
    x: torch.Tensor,
    *,
    omega: float,
    well_sep: float,
    smooth_T: float = 0.2,
    coulomb: bool = True,
    magnetic_B: float = 0.0,
    spin: torch.Tensor | None = None,
    g_factor: float = 2.0,
    mu_B: float = 1.0,
    zeeman_electron1_only: bool = False,
    zeeman_particle_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Legacy-compatible external + Coulomb + optional Zeeman potential.

    This mirrors the behavior expected by `imaginary_time_pinn.py` and
    `imaginary_time_vmc.py` for N-electron 2D runs.
    """

    B, N, d = x.shape

    # External confinement potential (single well or soft-min double well).
    if well_sep <= 1e-10:
        v_ext = 0.5 * float(omega) ** 2 * (x * x).sum(dim=(1, 2))
    else:
        r_l = torch.zeros(d, dtype=x.dtype, device=x.device)
        r_r = torch.zeros(d, dtype=x.dtype, device=x.device)
        r_l[0] = -0.5 * float(well_sep)
        r_r[0] = 0.5 * float(well_sep)
        v_l = 0.5 * float(omega) ** 2 * ((x - r_l) ** 2).sum(dim=-1)
        v_r = 0.5 * float(omega) ** 2 * ((x - r_r) ** 2).sum(dim=-1)
        T = float(smooth_T)
        v_pp = -T * torch.logaddexp(-v_l / T, -v_r / T)
        v_ext = v_pp.sum(dim=1)

    v_coul = torch.zeros(B, dtype=x.dtype, device=x.device)
    if coulomb:
        eps_sc = 1e-6 / max(float(omega), 1e-6) ** 0.5
        for i in range(N):
            for j in range(i + 1, N):
                r2 = ((x[:, i, :] - x[:, j, :]) ** 2).sum(dim=-1)
                v_coul = v_coul + 1.0 / torch.sqrt(r2 + eps_sc * eps_sc)

    v = v_ext + v_coul

    # Optional Zeeman term used by quench sweeps.
    if abs(float(magnetic_B)) > 0.0:
        if spin is None:
            # Closed-shell default: first half up (+1), second half down (-1)
            up = N // 2
            spin_z = torch.ones(N, dtype=x.dtype, device=x.device)
            spin_z[up:] = -1.0
            spin_z = spin_z.unsqueeze(0).expand(B, -1)
        else:
            s = spin.to(device=x.device)
            if s.dim() == 1:
                s = s.unsqueeze(0).expand(B, -1)
            # Accept either {0,1} coding or +/-1 coding.
            if torch.all((s == 0) | (s == 1)):
                spin_z = 1.0 - 2.0 * s.to(x.dtype)
            else:
                spin_z = s.to(x.dtype)

        if zeeman_electron1_only and zeeman_particle_indices is not None:
            raise ValueError(
                "zeeman_electron1_only and zeeman_particle_indices are mutually exclusive."
            )

        if zeeman_particle_indices is not None:
            if len(zeeman_particle_indices) == 0:
                raise ValueError("zeeman_particle_indices must not be empty.")
            idx = torch.tensor(zeeman_particle_indices, device=x.device, dtype=torch.long)
            if int(idx.min().item()) < 0 or int(idx.max().item()) >= N:
                raise ValueError(
                    f"zeeman_particle_indices out of range for N={N}: {tuple(int(i) for i in zeeman_particle_indices)}"
                )
            zeeman = 0.5 * float(g_factor) * float(mu_B) * float(magnetic_B) * spin_z[:, idx].sum(dim=1)
        elif zeeman_electron1_only:
            zeeman = 0.5 * float(g_factor) * float(mu_B) * float(magnetic_B) * spin_z[:, 0]
        else:
            zeeman = 0.5 * float(g_factor) * float(mu_B) * float(magnetic_B) * spin_z.sum(dim=1)
        v = v + zeeman

    return v
