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
    """Compute full potential for an arbitrary multi-well SystemConfig."""
    (batch_size, n_particles, dim) = x.shape
    dtype = x.dtype
    device = x.device

    well_vals = []
    for well in system.wells:
        center = torch.tensor(well.center, device=device, dtype=dtype).view(1, 1, dim)
        dr = x - center
        v_well = 0.5 * float(well.omega) ** 2 * torch.sum(dr * dr, dim=-1)
        well_vals.append(v_well)
    v_stack = torch.stack(well_vals, dim=-1)
    T = float(system.smooth_T)
    v_conf = -T * torch.logsumexp(-v_stack / T, dim=-1)
    v_conf = torch.sum(v_conf, dim=-1)

    v_coul = torch.zeros(batch_size, dtype=dtype, device=device)
    if system.coulomb:
        eps = 1e-6 / max(float(system.omega), 1e-6) ** 0.5
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                r2 = ((x[:, i, :] - x[:, j, :]) ** 2).sum(dim=-1)
                v_coul = v_coul + 1.0 / torch.sqrt(r2 + eps * eps)
        v_coul = v_coul * float(system.coulomb_strength)

    v = v_conf + v_coul

    magnetic_B = float(system.B_magnitude) * float(system.B_direction[2])
    if abs(magnetic_B) > 0.0:
        if spin is None:
            up = n_particles // 2
            spin_z = torch.ones(n_particles, dtype=dtype, device=device)
            spin_z[up:] = -1.0
            spin_z = spin_z.unsqueeze(0).expand(batch_size, -1)
        else:
            s = spin.to(device=device)
            if s.dim() == 1:
                s = s.unsqueeze(0).expand(batch_size, -1)
            if torch.all((s == 0) | (s == 1)):
                spin_z = 1.0 - 2.0 * s.to(dtype)
            else:
                spin_z = s.to(dtype)

        if system.zeeman_electron1_only and system.zeeman_particle_indices is not None:
            raise ValueError(
                "zeeman_electron1_only and zeeman_particle_indices are mutually exclusive."
            )

        if system.zeeman_particle_indices is not None:
            if len(system.zeeman_particle_indices) == 0:
                raise ValueError("zeeman_particle_indices must not be empty.")
            idx = torch.tensor(system.zeeman_particle_indices, device=device, dtype=torch.long)
            if int(idx.min().item()) < 0 or int(idx.max().item()) >= n_particles:
                raise ValueError(
                    f"zeeman_particle_indices out of range for N={n_particles}: "
                    f"{tuple(int(i) for i in system.zeeman_particle_indices)}"
                )
            zeeman = (
                0.5
                * float(system.g_factor)
                * float(system.mu_B)
                * magnetic_B
                * spin_z[:, idx].sum(dim=1)
            )
        elif system.zeeman_electron1_only:
            zeeman = (
                0.5 * float(system.g_factor) * float(system.mu_B) * magnetic_B * spin_z[:, 0]
            )
        else:
            zeeman = (
                0.5 * float(system.g_factor) * float(system.mu_B) * magnetic_B * spin_z.sum(dim=1)
            )
        v = v + zeeman

    return v


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
