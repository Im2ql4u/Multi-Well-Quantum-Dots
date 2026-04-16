from __future__ import annotations

import math
from typing import Callable

import torch
from config import SystemConfig


def adapt_sigma_fs(omega: float, sigma_fs: tuple[float, ...]) -> tuple[float, ...]:
    scale = 1.0 / max(float(omega), 1e-08) ** 0.5
    return tuple(float(s) * scale for s in sigma_fs)


def _sample_multiwell_init(
    n_samples: int, *, system: SystemConfig, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    x = torch.empty(n_samples, system.n_particles, system.dim, device=device, dtype=dtype)
    idx = 0
    for well in system.wells:
        n_e = int(well.n_particles)
        if n_e == 0:
            continue
        center = torch.tensor(well.center, device=device, dtype=dtype).view(1, 1, system.dim)
        sigma = 1 / max(float(well.omega), 1e-08) ** 0.5
        chunk = center + sigma * torch.randn(n_samples, n_e, system.dim, device=device, dtype=dtype)
        x[:, idx : idx + n_e, :] = chunk
        idx += n_e
    return x


def sample_multiwell_init(
    n_samples: int, *, system: SystemConfig, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Public wrapper for the fixed Gaussian non-MCMC proposal."""
    return _sample_multiwell_init(n_samples, system=system, device=device, dtype=dtype)


def multiwell_init_logpdf(x: torch.Tensor, *, system: SystemConfig) -> torch.Tensor:
    """Log-density of the fixed Gaussian proposal used by sample_multiwell_init."""
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (batch, n_particles, dim), got {tuple(x.shape)}.")

    batch = x.shape[0]
    total = torch.zeros(batch, device=x.device, dtype=x.dtype)
    idx = 0
    for well in system.wells:
        n_e = int(well.n_particles)
        if n_e == 0:
            continue
        center = torch.tensor(well.center, device=x.device, dtype=x.dtype).view(1, 1, system.dim)
        sigma = 1.0 / max(float(well.omega), 1e-8) ** 0.5
        chunk = x[:, idx : idx + n_e, :]
        total = total + _log_isotropic_gaussian(chunk, center, sigma).sum(dim=1)
        idx += n_e
    return total


def _particle_well_centers(
    system: SystemConfig, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    centers = []
    for well in system.wells:
        for _ in range(int(well.n_particles)):
            centers.append(list(well.center))
    return torch.tensor(centers, device=device, dtype=dtype)


def _log_isotropic_gaussian(x: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma_t = torch.as_tensor(float(sigma), device=x.device, dtype=x.dtype).clamp_min(1e-12)
    diff = x - centers
    dim = x.shape[-1]
    log_norm = -0.5 * dim * math.log(2.0 * math.pi) - dim * torch.log(sigma_t)
    quad = -0.5 * diff.pow(2).sum(dim=-1) / sigma_t.pow(2)
    return log_norm + quad


def _log_shell_density_2d(
    x: torch.Tensor,
    centers: torch.Tensor,
    radius_mean: float,
    radius_sigma: float,
) -> torch.Tensor:
    sigma_t = torch.as_tensor(float(radius_sigma), device=x.device, dtype=x.dtype).clamp_min(1e-12)
    mean_t = torch.as_tensor(float(radius_mean), device=x.device, dtype=x.dtype)
    diff = x - centers
    radius = diff.pow(2).sum(dim=-1).sqrt().clamp_min(1e-12)

    z1 = (radius - mean_t) / sigma_t
    z2 = (radius + mean_t) / sigma_t
    log_phi1 = -0.5 * z1.pow(2) - torch.log(sigma_t) - 0.5 * math.log(2.0 * math.pi)
    log_phi2 = -0.5 * z2.pow(2) - torch.log(sigma_t) - 0.5 * math.log(2.0 * math.pi)
    log_radial = torch.logaddexp(log_phi1, log_phi2)
    return log_radial - math.log(2.0 * math.pi) - torch.log(radius)


def stratified_logpdf(
    x: torch.Tensor,
    *,
    omega: float,
    system: SystemConfig,
    component_weights: tuple[float, float, float, float, float] = (0.25, 0.2, 0.25, 0.2, 0.0),
    sigma_center: float = 0.2,
    sigma_tails: float = 1.2,
    sigma_mixed_in: float = 0.25,
    sigma_mixed_out: float = 0.9,
    shell_radius: float = 1.4,
    shell_radius_sigma: float = 0.08,
    dimer_pairs: int = 2,
    dimer_eps_max: float = 0.08,
) -> torch.Tensor:
    del dimer_pairs, dimer_eps_max

    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (batch, n_particles, dim), got {tuple(x.shape)}.")

    if float(component_weights[4]) > 0.0:
        raise ValueError(
            "stratified_logpdf does not support nonzero dimer weight yet. "
            "Use eval mixture weights with zero dimer component for non-MCMC evaluation."
        )

    a_ho = 1.0 / max(float(omega), 1e-8) ** 0.5
    centers = _particle_well_centers(system, device=x.device, dtype=x.dtype).view(1, system.n_particles, system.dim)

    w = torch.as_tensor(component_weights, device=x.device, dtype=x.dtype)
    w = w / w.sum().clamp_min(1e-12)
    log_w = torch.log(w.clamp_min(1e-12))

    log_center = _log_isotropic_gaussian(x, centers, sigma_center * a_ho).sum(dim=1)
    log_tails = _log_isotropic_gaussian(x, centers, sigma_tails * a_ho).sum(dim=1)

    log_mixed_in = _log_isotropic_gaussian(x, centers, sigma_mixed_in * a_ho)
    log_mixed_out = _log_isotropic_gaussian(x, centers, sigma_mixed_out * a_ho)
    log_mixed_particle = torch.logaddexp(
        math.log(0.5) + log_mixed_in,
        math.log(0.5) + log_mixed_out,
    )
    log_mixed = log_mixed_particle.sum(dim=1)

    if system.dim == 2:
        log_shell = _log_shell_density_2d(
            x,
            centers,
            shell_radius * a_ho,
            shell_radius_sigma * a_ho,
        ).sum(dim=1)
    else:
        log_shell = _log_isotropic_gaussian(x, centers, sigma_mixed_out * a_ho).sum(dim=1)

    component_logs = [
        log_w[0] + log_center,
        log_w[1] + log_tails,
        log_w[2] + log_mixed,
        log_w[3] + log_shell,
    ]
    stacked = torch.stack(component_logs, dim=0)
    return torch.logsumexp(stacked, dim=0)


def stratified_resample(
    *,
    n_keep: int,
    omega: float,
    system: SystemConfig,
    device: torch.device,
    dtype: torch.dtype,
    component_weights: tuple[float, float, float, float, float] = (0.25, 0.2, 0.25, 0.2, 0.1),
    sigma_center: float = 0.2,
    sigma_tails: float = 1.2,
    sigma_mixed_in: float = 0.25,
    sigma_mixed_out: float = 0.9,
    shell_radius: float = 1.4,
    shell_radius_sigma: float = 0.08,
    dimer_pairs: int = 2,
    dimer_eps_max: float = 0.08,
) -> tuple[torch.Tensor, float]:
    """Draw i.i.d. non-MCMC collocation points from a stratified mixture.

    Components are [center, tails, mixed, shells, dimers]. This is deliberately
    label-preserving for multi-well occupancy and does not use random permutations.
    """
    a_ho = 1.0 / max(float(omega), 1e-8) ** 0.5
    n_particles = int(system.n_particles)
    dim = int(system.dim)
    centers = _particle_well_centers(system, device=device, dtype=dtype).view(1, n_particles, dim)

    w = torch.tensor(component_weights, device=device, dtype=dtype)
    w = w / w.sum().clamp_min(1e-12)
    comp_idx = torch.multinomial(w, n_keep, replacement=True)

    x = torch.empty(n_keep, n_particles, dim, device=device, dtype=dtype)

    mask_center = comp_idx == 0
    if mask_center.any():
        n = int(mask_center.sum().item())
        x[mask_center] = centers + (sigma_center * a_ho) * torch.randn(n, n_particles, dim, device=device, dtype=dtype)

    mask_tails = comp_idx == 1
    if mask_tails.any():
        n = int(mask_tails.sum().item())
        x[mask_tails] = centers + (sigma_tails * a_ho) * torch.randn(n, n_particles, dim, device=device, dtype=dtype)

    mask_mixed = comp_idx == 2
    if mask_mixed.any():
        n = int(mask_mixed.sum().item())
        choose_inner = (torch.rand(n, n_particles, 1, device=device, dtype=dtype) < 0.5).to(dtype)
        sigma = choose_inner * (sigma_mixed_in * a_ho) + (1.0 - choose_inner) * (sigma_mixed_out * a_ho)
        x[mask_mixed] = centers + sigma * torch.randn(n, n_particles, dim, device=device, dtype=dtype)

    mask_shell = comp_idx == 3
    if mask_shell.any():
        n = int(mask_shell.sum().item())
        if dim == 2:
            theta = 2.0 * torch.pi * torch.rand(n, n_particles, 1, device=device, dtype=dtype)
            radius = (shell_radius + shell_radius_sigma * torch.randn(n, n_particles, 1, device=device, dtype=dtype)).abs() * a_ho
            offsets = torch.cat([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
            x[mask_shell] = centers + offsets
        else:
            x[mask_shell] = centers + (sigma_mixed_out * a_ho) * torch.randn(n, n_particles, dim, device=device, dtype=dtype)

    mask_dimer = comp_idx == 4
    if mask_dimer.any():
        n = int(mask_dimer.sum().item())
        xd = centers + (sigma_center * a_ho) * torch.randn(n, n_particles, dim, device=device, dtype=dtype)
        n_pairs = max(0, min(int(dimer_pairs), n_particles // 2))
        if n_pairs > 0:
            base = torch.arange(0, 2 * n_pairs, device=device)
            for b in range(n):
                perm = torch.randperm(n_particles, device=device)
                eps = dimer_eps_max * a_ho * torch.rand(n_pairs, 1, device=device, dtype=dtype)
                if dim == 2:
                    ang = 2.0 * torch.pi * torch.rand(n_pairs, 1, device=device, dtype=dtype)
                    dvec = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
                else:
                    signs = torch.where(
                        torch.rand(n_pairs, 1, device=device, dtype=dtype) < 0.5,
                        torch.full((n_pairs, 1), -1.0, device=device, dtype=dtype),
                        torch.full((n_pairs, 1), 1.0, device=device, dtype=dtype),
                    )
                    dvec = signs
                for p in range(n_pairs):
                    i = int(perm[base[2 * p]].item())
                    j = int(perm[base[2 * p + 1]].item())
                    xd[b, j, :] = xd[b, i, :] + eps[p] * dvec[p]
        x[mask_dimer] = xd

    return x, float(n_keep)


def mcmc_resample(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    x_prev: torch.Tensor | None,
    n_keep: int,
    *,
    n_elec: int,
    dim: int,
    omega: float,
    device: torch.device,
    dtype: torch.dtype,
    system: SystemConfig,
    sigma_fs: tuple[float, ...],
    mh_steps: int,
    mh_step_scale: float,
    mh_decorrelation: int,
) -> tuple[torch.Tensor, float, float]:
    del n_elec, dim, sigma_fs

    if x_prev is not None:
        x = x_prev.detach().clone()
    else:
        x = _sample_multiwell_init(n_keep, system=system, device=device, dtype=dtype)

    n_steps = max(1, int(mh_steps) * max(1, int(mh_decorrelation)))
    step = float(mh_step_scale) / max(float(omega), 1e-08) ** 0.5
    accepted = 0.0
    total = 0.0

    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(n_steps):
            prop = x + step * torch.randn_like(x)
            logp_prop = 2.0 * psi_log_fn(prop)
            loga = torch.clamp(logp_prop - logp, max=0.0)
            u = torch.log(torch.rand_like(loga))
            accept = u < loga
            x = torch.where(accept.view(-1, 1, 1), prop, x)
            logp = torch.where(accept, logp_prop, logp)
            accepted += float(accept.double().sum().item())
            total += float(accept.numel())

    accept_rate = accepted / max(total, 1.0)
    target = 0.55
    new_scale = float(mh_step_scale) * (1.0 + 0.15 * (accept_rate - target))
    new_scale = float(min(max(new_scale, 0.05), 1.5))
    return x.detach(), float(accept_rate), new_scale


def importance_resample(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_keep: int,
    n_elec: int,
    dim: int,
    omega: float,
    device: torch.device,
    dtype: torch.dtype,
    n_cand_mult: int,
    sigma_fs: tuple[float, ...],
    min_pair_cutoff: float,
    weight_temp: float,
    logw_clip_q: float,
    langevin_steps: int,
    langevin_step_size: float,
    system: SystemConfig,
    return_weights: bool = False,
) -> tuple[torch.Tensor, float] | tuple[torch.Tensor, float, torch.Tensor]:
    del n_elec, dim, n_cand_mult, sigma_fs, min_pair_cutoff
    del langevin_steps, langevin_step_size

    x = _sample_multiwell_init(n_keep, system=system, device=device, dtype=dtype)
    with torch.no_grad():
        logw = 2.0 * psi_log_fn(x) / max(float(weight_temp), 1e-08)
        clip_q = float(logw_clip_q)
        if 0.0 < clip_q < 0.5:
            lo = torch.quantile(logw, clip_q)
            hi = torch.quantile(logw, 1.0 - clip_q)
            logw = torch.clamp(logw, min=lo, max=hi)
        logw = logw - torch.logsumexp(logw, dim=0)
        w = torch.exp(logw)
        ess = 1.0 / torch.sum(w * w)
        idx = torch.multinomial(w, n_keep, replacement=True)
        x = x[idx]
        w_sel = w[idx]
        w_sel = w_sel / w_sel.sum().clamp_min(1e-12)
    if return_weights:
        return x, float(ess.item()), w_sel
    return x, float(ess.item())
