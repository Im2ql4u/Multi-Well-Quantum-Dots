from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import torch

from config import SystemConfig
from observables.diagnostics import summarize_training_diagnostics
from training.collocation import colloc_fd_loss, rayleigh_hybrid_loss, weak_form_local_energy
from training.sampling import (
    adapt_sigma_fs,
    importance_resample,
    mcmc_resample,
    multiwell_init_logpdf,
    sample_multiwell_init,
    stratified_logpdf,
    stratified_resample,
)


@dataclass(frozen=True)
class GroundStateTrainingConfig:
    epochs: int = 200
    lr: float = 1e-3
    # Cosine LR schedule with linear warmup.
    # lr_warmup_epochs: number of epochs to linearly ramp from lr*lr_min_factor to lr.
    # lr_min_factor: final lr = lr * lr_min_factor (also the warmup start factor).
    # Set lr_warmup_epochs=0 and lr_min_factor=1.0 to disable (flat lr, original behaviour).
    lr_warmup_epochs: int = 0
    lr_min_factor: float = 1.0
    n_coll: int = 256
    n_cand_mult: int = 8
    loss_type: str = "weak_form"
    laplacian_mode: str = "fd"
    # Robust clipping for local-energy based losses (weak_form/fd_colloc/reinforce_hybrid/residual).
    # 0.0 disables clipping. Positive values apply MAD clipping around the median.
    local_energy_clip_width: float = 0.0
    residual_objective: Literal["residual", "energy", "energy_var"] = "residual"
    residual_target_energy: float | None = None
    alpha_start: float = 0.0
    alpha_end: float = 1.0
    alpha_decay_frac: float = 0.5
    occupancy_penalty_weight: float = 0.0
    occupancy_margin: float = 0.0
    direct_weight: float = 0.1
    fd_h: float = 0.01
    min_pair_cutoff: float = 0.0
    sigma_fs: tuple[float, ...] = (0.8, 1.3, 2.0)
    weight_temp: float = 1.0
    logw_clip_q: float = 0.0
    langevin_steps: int = 0
    langevin_step_size: float = 0.01
    sampler: str = "is"
    non_mcmc_only: bool = False
    sampler_mix_weights: tuple[float, float, float, float, float] = (0.25, 0.2, 0.25, 0.2, 0.1)
    sampler_sigma_center: float = 0.20
    sampler_sigma_tails: float = 1.20
    sampler_sigma_mixed_in: float = 0.25
    sampler_sigma_mixed_out: float = 0.90
    sampler_shell_radius: float = 1.40
    sampler_shell_radius_sigma: float = 0.08
    sampler_dimer_pairs: int = 2
    sampler_dimer_eps_max: float = 0.08
    mh_steps: int = 10
    mh_step_scale: float = 0.25
    mh_decorrelation: int = 1
    grad_clip: float = 1.0
    reinforce_clip_width: float = 5.0
    print_every: int = 10
    seed: int | None = 0
    device: str = "cpu"
    dtype: str = "float64"

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
        }
        if self.dtype not in mapping:
            raise ValueError(f"Unsupported dtype '{self.dtype}'.")
        return mapping[self.dtype]


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_schedule_factor(
    epoch: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    min_factor: float,
) -> float:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if min_factor <= 0.0:
        raise ValueError("min_factor must be positive.")

    warmup = min(warmup_epochs, total_epochs)
    if warmup > 0 and epoch < warmup:
        return min_factor + (1.0 - min_factor) * (epoch / warmup)

    decay_steps = max(1, total_epochs - warmup)
    if decay_steps == 1:
        return min_factor

    t = epoch - warmup
    progress = t / (decay_steps - 1)
    return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))


def compute_eeff(
    *,
    mu: float,
    e_ref: float | None,
    epoch: int,
    total_epochs: int,
    mode: Literal["residual", "energy", "energy_var"],
    alpha_start: float,
    alpha_end: float,
    alpha_decay_frac: float,
) -> float:
    if mode == "residual":
        return float(mu)
    if mode == "energy":
        if e_ref is None:
            raise ValueError("residual_target_energy must be set when residual_objective='energy'.")
        return float(e_ref)
    if mode != "energy_var":
        raise ValueError(f"Unknown residual objective mode '{mode}'.")
    if e_ref is None:
        raise ValueError(
            "residual_target_energy must be set when residual_objective='energy_var'."
        )

    ramp_frac = max(1e-6, float(alpha_decay_frac))
    ramp_epochs = max(1, int(round(ramp_frac * max(total_epochs, 1))))
    t = min(max(int(epoch), 0), ramp_epochs)
    progress = t / ramp_epochs
    alpha = float(alpha_start) + 0.5 * (float(alpha_end) - float(alpha_start)) * (
        1.0 - math.cos(math.pi * progress)
    )
    lo = min(float(alpha_start), float(alpha_end))
    hi = max(float(alpha_start), float(alpha_end))
    alpha = min(max(alpha, lo), hi)
    return float((1.0 - alpha) * mu + alpha * float(e_ref))


def clip_local_energy_by_mad(e_loc: torch.Tensor, clip_width: float) -> torch.Tensor:
    if clip_width <= 0.0:
        return e_loc
    med = e_loc.detach().median()
    mad = (e_loc.detach() - med).abs().median().clamp_min(1e-8)
    lo = med - float(clip_width) * mad
    hi = med + float(clip_width) * mad
    return e_loc.clamp(lo, hi)


def same_dot_occupancy_penalty(x: torch.Tensor, system: SystemConfig, margin: float) -> torch.Tensor:
    if len(system.wells) != 2 or system.n_particles != 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    centers = torch.tensor([well.center for well in system.wells], device=x.device, dtype=x.dtype)
    diff = x.unsqueeze(2) - centers.view(1, 1, 2, system.dim)
    dist_sq = torch.sum(diff * diff, dim=-1)
    nearest = torch.argmin(dist_sq, dim=-1)

    same_dot = (nearest[:, 0] == nearest[:, 1]).to(x.dtype)
    if float(margin) <= 0.0:
        return torch.mean(same_dot)

    center_sep = torch.linalg.norm(centers[1] - centers[0])
    same_well_dist = torch.linalg.norm(x[:, 0, :] - x[:, 1, :], dim=-1)
    slack = torch.clamp(float(margin) * center_sep - same_well_dist, min=0.0)
    return torch.mean(same_dot * slack)


def train_ground_state(
    model: torch.nn.Module,
    system: SystemConfig,
    params: dict,
    train_cfg: GroundStateTrainingConfig,
) -> dict:
    _apply_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    dtype = train_cfg.torch_dtype
    if train_cfg.sampler not in ("is", "mh", "stratified"):
        raise ValueError(f"Unknown sampler '{train_cfg.sampler}', expected 'is', 'mh', or 'stratified'.")
    if train_cfg.non_mcmc_only and train_cfg.sampler == "mh":
        raise ValueError("non_mcmc_only=true forbids sampler='mh'.")
    direct_local_energy_losses = ("weak_form", "fd_colloc", "reinforce_hybrid")
    if train_cfg.loss_type in direct_local_energy_losses and train_cfg.sampler not in ("mh", "is", "stratified"):
        raise ValueError(
            f"loss_type='{train_cfg.loss_type}' requires sampler='mh', sampler='is', or sampler='stratified' in the current trainer. "
            "Use sampler='is' for the simple Gaussian fixed proposal, sampler='stratified' for the structured non-MCMC mixture, "
            "loss_type='residual' for pure residual runs, or sampler='mh' for direct variational minimization."
        )
    if train_cfg.laplacian_mode not in ("fd", "autograd"):
        raise ValueError(
            f"Unknown laplacian_mode '{train_cfg.laplacian_mode}', expected 'fd' or 'autograd'."
        )

    model = model.to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    history: dict[str, list[float]] = {
        "loss": [],
        "energy": [],
        "energy_var": [],
        "ess": [],
    }

    spin_template: torch.Tensor | None = None
    model_spin = getattr(model, "spin_template", None)
    if isinstance(model_spin, torch.Tensor):
        spin_template = model_spin.detach().to(device=device)

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    x_prev: torch.Tensor | None = None
    mh_scale = float(train_cfg.mh_step_scale)

    sigma_fs = adapt_sigma_fs(system.omega, train_cfg.sigma_fs)

    def normalized_is_weights(
        x: torch.Tensor,
        proposal_logpdf: torch.Tensor,
        *,
        detach: bool,
    ) -> torch.Tensor:
        log_psi = model(x)
        if detach:
            log_psi = log_psi.detach()
        logw = 2.0 * log_psi - proposal_logpdf
        clip_q = float(train_cfg.logw_clip_q)
        if 0.0 < clip_q < 0.5:
            lo = torch.quantile(logw.detach(), clip_q)
            hi = torch.quantile(logw.detach(), 1.0 - clip_q)
            logw = torch.clamp(logw, min=lo, max=hi)
        logw = logw - torch.logsumexp(logw, dim=0)
        return torch.exp(logw)

    for epoch in range(train_cfg.epochs):
        current_lr_factor = lr_schedule_factor(
            epoch,
            total_epochs=train_cfg.epochs,
            warmup_epochs=train_cfg.lr_warmup_epochs,
            min_factor=float(train_cfg.lr_min_factor),
        )
        current_lr = train_cfg.lr * current_lr_factor
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        is_weights: torch.Tensor | None = None
        proposal_logpdf: torch.Tensor | None = None
        if train_cfg.sampler == "is":
            if train_cfg.loss_type in direct_local_energy_losses:
                x = sample_multiwell_init(
                    train_cfg.n_coll,
                    system=system,
                    device=device,
                    dtype=dtype,
                )
                proposal_logpdf = multiwell_init_logpdf(x, system=system)
                with torch.no_grad():
                    w_det = normalized_is_weights(x, proposal_logpdf, detach=True)
                    ess = float((1.0 / torch.sum(w_det * w_det)).item())
            else:
                x, ess, is_weights = importance_resample(
                    psi_log_fn,
                    n_keep=train_cfg.n_coll,
                    n_elec=system.n_particles,
                    dim=system.dim,
                    omega=system.omega,
                    device=device,
                    dtype=dtype,
                    n_cand_mult=train_cfg.n_cand_mult,
                    sigma_fs=sigma_fs,
                    min_pair_cutoff=train_cfg.min_pair_cutoff,
                    weight_temp=train_cfg.weight_temp,
                    logw_clip_q=train_cfg.logw_clip_q,
                    langevin_steps=train_cfg.langevin_steps,
                    langevin_step_size=train_cfg.langevin_step_size,
                    system=system,
                    return_weights=True,
                )
        elif train_cfg.sampler == "stratified":
            x, ess = stratified_resample(
                n_keep=train_cfg.n_coll,
                omega=system.omega,
                system=system,
                device=device,
                dtype=dtype,
                component_weights=train_cfg.sampler_mix_weights,
                sigma_center=train_cfg.sampler_sigma_center,
                sigma_tails=train_cfg.sampler_sigma_tails,
                sigma_mixed_in=train_cfg.sampler_sigma_mixed_in,
                sigma_mixed_out=train_cfg.sampler_sigma_mixed_out,
                shell_radius=train_cfg.sampler_shell_radius,
                shell_radius_sigma=train_cfg.sampler_shell_radius_sigma,
                dimer_pairs=train_cfg.sampler_dimer_pairs,
                dimer_eps_max=train_cfg.sampler_dimer_eps_max,
            )
            if train_cfg.loss_type in direct_local_energy_losses:
                proposal_logpdf = stratified_logpdf(
                    x,
                    omega=system.omega,
                    system=system,
                    component_weights=train_cfg.sampler_mix_weights,
                    sigma_center=train_cfg.sampler_sigma_center,
                    sigma_tails=train_cfg.sampler_sigma_tails,
                    sigma_mixed_in=train_cfg.sampler_sigma_mixed_in,
                    sigma_mixed_out=train_cfg.sampler_sigma_mixed_out,
                    shell_radius=train_cfg.sampler_shell_radius,
                    shell_radius_sigma=train_cfg.sampler_shell_radius_sigma,
                    dimer_pairs=train_cfg.sampler_dimer_pairs,
                    dimer_eps_max=train_cfg.sampler_dimer_eps_max,
                )
                with torch.no_grad():
                    w_det = normalized_is_weights(x, proposal_logpdf, detach=True)
                    ess = float((1.0 / torch.sum(w_det * w_det)).item())
        else:
            x, accept_rate, mh_scale = mcmc_resample(
                psi_log_fn,
                x_prev,
                train_cfg.n_coll,
                n_elec=system.n_particles,
                dim=system.dim,
                omega=system.omega,
                device=device,
                dtype=dtype,
                system=system,
                sigma_fs=sigma_fs,
                mh_steps=train_cfg.mh_steps,
                mh_step_scale=mh_scale,
                mh_decorrelation=train_cfg.mh_decorrelation,
            )
            x_prev = x
            ess = float(accept_rate)

        def weighted_mean(values: torch.Tensor) -> torch.Tensor:
            if proposal_logpdf is not None:
                w = normalized_is_weights(x, proposal_logpdf, detach=False)
                return torch.sum(w * values)
            if is_weights is None:
                return values.mean()
            w = is_weights / is_weights.sum().clamp_min(1e-12)
            return torch.sum(w * values)

        def weighted_var(values: torch.Tensor) -> float:
            vals = values.detach()
            if proposal_logpdf is not None:
                w = normalized_is_weights(x, proposal_logpdf, detach=True)
                mean = torch.sum(w * vals)
                return float(torch.sum(w * (vals - mean) * (vals - mean)).item())
            if is_weights is None:
                return float(torch.var(vals, unbiased=False).item())
            w = is_weights / is_weights.sum().clamp_min(1e-12)
            mean = torch.sum(w * vals)
            return float(torch.sum(w * (vals - mean) * (vals - mean)).item())

        optimizer.zero_grad(set_to_none=True)
        if train_cfg.loss_type == "weak_form":
            e_weak = weak_form_local_energy(
                psi_log_fn,
                x,
                omega=system.omega,
                params=params,
                system=system,
                h=train_cfg.fd_h,
                laplacian_mode=train_cfg.laplacian_mode,
                spin=spin_template,
            )
            e_used = clip_local_energy_by_mad(e_weak, train_cfg.local_energy_clip_width)
            loss = weighted_mean(e_used)
            energy = float(loss.detach().item())
            local_energy_samples = e_used.detach()
        elif train_cfg.loss_type == "reinforce_hybrid":
            _, _, e_eff, _ = rayleigh_hybrid_loss(
                psi_log_fn,
                x,
                omega=system.omega,
                params=params,
                system=system,
                direct_weight=train_cfg.direct_weight,
                h=train_cfg.fd_h,
                laplacian_mode=train_cfg.laplacian_mode,
                spin=spin_template,
            )
            e_used = clip_local_energy_by_mad(e_eff, train_cfg.local_energy_clip_width)
            loss = weighted_mean(e_used)
            energy = float(loss.detach().item())
            local_energy_samples = e_used
        elif train_cfg.loss_type == "fd_colloc":
            _, _, e_loc, _ = colloc_fd_loss(
                psi_log_fn,
                x,
                omega=system.omega,
                params=params,
                system=system,
                h=train_cfg.fd_h,
                laplacian_mode=train_cfg.laplacian_mode,
                spin=spin_template,
            )
            e_used = clip_local_energy_by_mad(e_loc, train_cfg.local_energy_clip_width)
            loss = weighted_mean(e_used)
            energy = float(loss.detach().item())
            local_energy_samples = e_used
        elif train_cfg.loss_type == "reinforce":
            # Forward pass to get log_psi (carries gradient).
            log_psi = model(x)
            # Local energy estimate for score-function baseline.
            # For autograd Laplacian we must build derivatives once, then detach.
            if train_cfg.laplacian_mode == "autograd":
                _, _, e_loc, _ = colloc_fd_loss(
                    psi_log_fn,
                    x,
                    omega=system.omega,
                    params=params,
                    system=system,
                    h=train_cfg.fd_h,
                    laplacian_mode=train_cfg.laplacian_mode,
                    spin=spin_template,
                )
                e_loc = e_loc.detach()
            else:
                with torch.no_grad():
                    _, _, e_loc, _ = colloc_fd_loss(
                        psi_log_fn,
                        x,
                        omega=system.omega,
                        params=params,
                        system=system,
                        h=train_cfg.fd_h,
                        laplacian_mode=train_cfg.laplacian_mode,
                        spin=spin_template,
                    )
            # MAD-based outlier clipping (matches old code's clip_el).
            clip_w = float(train_cfg.reinforce_clip_width)
            if clip_w > 0:
                med = e_loc.median()
                mad = (e_loc - med).abs().median().clamp_min(1e-8)
                e_loc = e_loc.clamp(med - clip_w * mad, med + clip_w * mad)
            E_mean = e_loc.mean()
            # REINFORCE: gradient flows only through log_psi.
            loss = 2.0 * ((e_loc - E_mean) * log_psi).mean()
            energy = float(E_mean.item())
            local_energy_samples = e_loc
        elif train_cfg.loss_type == "residual":
            _, _, e_loc, _ = colloc_fd_loss(
                psi_log_fn,
                x,
                omega=system.omega,
                params=params,
                system=system,
                h=train_cfg.fd_h,
                laplacian_mode=train_cfg.laplacian_mode,
                spin=spin_template,
            )
            e_used = clip_local_energy_by_mad(e_loc, train_cfg.local_energy_clip_width)
            mu = float(e_used.detach().mean().item())
            e_eff = compute_eeff(
                mu=mu,
                e_ref=train_cfg.residual_target_energy,
                epoch=epoch,
                total_epochs=train_cfg.epochs,
                mode=train_cfg.residual_objective,
                alpha_start=train_cfg.alpha_start,
                alpha_end=train_cfg.alpha_end,
                alpha_decay_frac=train_cfg.alpha_decay_frac,
            )
            loss = weighted_mean((e_used - e_eff) * (e_used - e_eff))
            if train_cfg.occupancy_penalty_weight > 0.0:
                penalty = same_dot_occupancy_penalty(x, system, train_cfg.occupancy_margin)
                loss = loss + float(train_cfg.occupancy_penalty_weight) * penalty
            energy = mu
            local_energy_samples = e_used.detach()
        else:
            raise ValueError(f"Unknown loss_type '{train_cfg.loss_type}'.")

        energy_var = weighted_var(local_energy_samples)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss is not finite at epoch {epoch}. Check sampling, loss choice, and lr={train_cfg.lr:.2e}."
            )
        loss.backward()

        grad_sq = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                raise RuntimeError(
                    f"Gradient is not finite at epoch {epoch}. Check architecture outputs and sampling weights."
                )
            grad_sq += float(param.grad.pow(2).sum().item())
        grad_norm = math.sqrt(grad_sq) if grad_sq > 0.0 else 0.0
        if train_cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["energy"].append(float(energy))
        history["energy_var"].append(energy_var)
        history["ess"].append(float(ess))

        if epoch % max(1, train_cfg.print_every) == 0 or epoch == train_cfg.epochs - 1:
            gpu_mem_msg = ""
            if device.type == "cuda":
                mem_alloc_gb = torch.cuda.memory_allocated(device) / 1e9
                mem_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                gpu_mem_msg = f" gpu_mem={mem_alloc_gb:.3f}GB peak={mem_peak_gb:.3f}GB"
            print(
                f"epoch={epoch:04d} loss={loss.item():.6f} energy={energy:.6f} e_var={energy_var:.6f} ess={ess:.1f} grad_norm={grad_norm:.3e} lr={current_lr:.2e}{gpu_mem_msg}"
            )

    diagnostics = summarize_training_diagnostics(
        history, n_coll=train_cfg.n_coll, sampler=train_cfg.sampler
    )

    return {
        "history": history,
        "diagnostics": diagnostics,
        "training_config": train_cfg.as_dict(),
        "final_loss": history["loss"][-1],
        "final_energy": history["energy"][-1],
        "final_energy_var": history["energy_var"][-1],
        "final_ess": history["ess"][-1],
    }
