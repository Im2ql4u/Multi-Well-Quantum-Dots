from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from config import SystemConfig
from observables.diagnostics import summarize_training_diagnostics
from training.collocation import colloc_fd_loss, rayleigh_hybrid_loss, weak_form_local_energy
from training.sampling import adapt_sigma_fs, importance_resample


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
    direct_weight: float = 0.1
    fd_h: float = 0.01
    min_pair_cutoff: float = 0.0
    sigma_fs: tuple[float, ...] = (0.8, 1.3, 2.0)
    weight_temp: float = 1.0
    logw_clip_q: float = 0.0
    langevin_steps: int = 0
    langevin_step_size: float = 0.01
    grad_clip: float = 1.0
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


def _make_lr_lambda(train_cfg: GroundStateTrainingConfig):
    warmup_epochs = max(0, train_cfg.lr_warmup_epochs)
    min_factor = float(train_cfg.lr_min_factor)
    decay_epochs = max(1, train_cfg.epochs - warmup_epochs)

    def _lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return min_factor + (1.0 - min_factor) * (epoch / warmup_epochs)
        t = epoch - warmup_epochs
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * t / decay_epochs))

    return _lr_lambda


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: GroundStateTrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_make_lr_lambda(train_cfg))


def train_ground_state(
    model: torch.nn.Module,
    system: SystemConfig,
    params: dict,
    train_cfg: GroundStateTrainingConfig,
) -> dict:
    _apply_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    dtype = train_cfg.torch_dtype
    model = model.to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    scheduler = _build_lr_scheduler(optimizer, train_cfg)

    history: dict[str, list[float]] = {
        "loss": [],
        "energy": [],
        "energy_var": [],
        "ess": [],
    }

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    sigma_fs = adapt_sigma_fs(system.omega, train_cfg.sigma_fs)
    for epoch in range(train_cfg.epochs):
        x, ess = importance_resample(
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
        )

        optimizer.zero_grad(set_to_none=True)
        if train_cfg.loss_type == "weak_form":
            e_weak = weak_form_local_energy(
                psi_log_fn,
                x,
                omega=system.omega,
                params=params,
                system=system,
            )
            loss = e_weak.mean()
            energy = float(loss.item())
            local_energy_samples = e_weak.detach()
        elif train_cfg.loss_type == "reinforce_hybrid":
            loss, energy, e_eff, _ = rayleigh_hybrid_loss(
                psi_log_fn,
                x,
                omega=system.omega,
                params=params,
                system=system,
                direct_weight=train_cfg.direct_weight,
            )
            local_energy_samples = e_eff
        elif train_cfg.loss_type == "fd_colloc":
            loss, energy, e_loc, _ = colloc_fd_loss(
                psi_log_fn,
                x,
                omega=system.omega,
                params=params,
                system=system,
                h=train_cfg.fd_h,
            )
            local_energy_samples = e_loc
        else:
            raise ValueError(f"Unknown loss_type '{train_cfg.loss_type}'.")

        energy_var = float(torch.var(local_energy_samples, unbiased=False).item())

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
        scheduler.step()

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
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch:04d} loss={loss.item():.6f} energy={energy:.6f} e_var={energy_var:.6f} ess={ess:.1f} grad_norm={grad_norm:.3e} lr={current_lr:.2e}{gpu_mem_msg}"
            )

    diagnostics = summarize_training_diagnostics(history, n_coll=train_cfg.n_coll)

    return {
        "history": history,
        "diagnostics": diagnostics,
        "training_config": train_cfg.as_dict(),
        "final_loss": history["loss"][-1],
        "final_energy": history["energy"][-1],
        "final_energy_var": history["energy_var"][-1],
        "final_ess": history["ess"][-1],
    }