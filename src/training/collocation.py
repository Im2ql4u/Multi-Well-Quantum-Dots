from __future__ import annotations

from typing import Callable

import torch
from config import SystemConfig


def _potential_energy(x: torch.Tensor, *, omega: float, system: SystemConfig) -> torch.Tensor:
    (bsz, _, dim) = x.shape
    dtype = x.dtype
    device = x.device
    well_vals = []
    for well in system.wells:
        center = torch.tensor(well.center, device=device, dtype=dtype).view(1, 1, dim)
        dr = x - center
        v = 0.5 * float(well.omega) ** 2 * torch.sum(dr * dr, dim=-1)
        well_vals.append(v)
    v_stack = torch.stack(well_vals, dim=-1)
    T = float(system.smooth_T)
    v_conf = -T * torch.logsumexp(-v_stack / T, dim=-1)
    v_conf = torch.sum(v_conf, dim=-1)
    if not system.coulomb:
        return v_conf
    eps = 1.0 / max(float(omega), 1e-08) ** 0.5
    (i, j) = torch.triu_indices(system.n_particles, system.n_particles, offset=1, device=device)
    rij = x[:, i, :] - x[:, j, :]
    r2 = torch.sum(rij * rij, dim=-1)
    v_ee = torch.sum(1 / torch.sqrt(r2 + eps * eps), dim=-1)
    return v_conf + v_ee


def _laplacian_over_psi_fd(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, h: float
) -> torch.Tensor:
    (bsz, n, d) = x.shape
    log_psi0 = psi_log_fn(x)
    lap = torch.zeros(bsz, device=x.device, dtype=x.dtype)
    h2 = float(h) * float(h)
    for i in range(n):
        for j in range(d):
            xp = x.clone()
            xm = x.clone()
            xp[:, i, j] = xp[:, i, j] + h
            xm[:, i, j] = xm[:, i, j] - h
            lp = psi_log_fn(xp)
            lm = psi_log_fn(xm)
            ratio_p = torch.exp(lp - log_psi0)
            ratio_m = torch.exp(lm - log_psi0)
            lap = lap + (ratio_p - 2 + ratio_m) / h2
    return lap


def colloc_fd_loss(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    omega: float,
    params: dict,
    system: SystemConfig,
    h: float,
) -> tuple[torch.Tensor, float, torch.Tensor, dict]:
    del params
    lap_over_psi = _laplacian_over_psi_fd(psi_log_fn, x, h)
    v = _potential_energy(x, omega=omega, system=system)
    e_loc = -0.5 * lap_over_psi + v
    if not torch.isfinite(e_loc).all():
        raise RuntimeError("Non-finite local energy in colloc_fd_loss.")
    loss = torch.mean(e_loc)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss in colloc_fd_loss.")
    return (loss, float(loss.detach().item()), e_loc, {})


def weak_form_local_energy(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    omega: float,
    params: dict,
    system: SystemConfig,
) -> torch.Tensor:
    (_, _, e_loc, _) = colloc_fd_loss(
        psi_log_fn, x, omega=omega, params=params, system=system, h=0.01
    )
    return e_loc


def rayleigh_hybrid_loss(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    omega: float,
    params: dict,
    system: SystemConfig,
    direct_weight: float,
) -> tuple[torch.Tensor, float, torch.Tensor, dict]:
    del direct_weight
    return colloc_fd_loss(psi_log_fn, x, omega=omega, params=params, system=system, h=0.01)
