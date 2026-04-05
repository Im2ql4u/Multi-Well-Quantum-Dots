from __future__ import annotations

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
