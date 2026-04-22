from __future__ import annotations

from typing import Callable

import torch
from config import SystemConfig
from potential import compute_potential


def _potential_energy(
    x: torch.Tensor,
    *,
    omega: float,
    system: SystemConfig,
    spin: torch.Tensor | None = None,
) -> torch.Tensor:
    del omega
    return compute_potential(x, system=system, spin=spin)


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
            # Clamp exponent arguments to avoid overflow/underflow blow-ups.
            dlp = torch.clamp(lp - log_psi0, min=-50.0, max=50.0)
            dlm = torch.clamp(lm - log_psi0, min=-50.0, max=50.0)
            ratio_p = torch.exp(dlp)
            ratio_m = torch.exp(dlm)
            lap = lap + (ratio_p - 2 + ratio_m) / h2
    return lap


def _laplacian_over_psi_autograd(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """Compute (laplacian psi) / psi from autograd on log(psi).

    Uses identity:
        (∇²ψ)/ψ = ∇²logψ + |∇logψ|²
    """
    x_req = x.detach().clone().requires_grad_(True)
    log_psi = psi_log_fn(x_req)
    if log_psi.ndim == 2 and log_psi.shape[1] == 1:
        log_psi = log_psi.squeeze(-1)
    if log_psi.ndim != 1:
        raise ValueError(f"psi_log_fn must return shape (B,) or (B,1), got {tuple(log_psi.shape)}")

    grad_log = torch.autograd.grad(log_psi.sum(), x_req, create_graph=True)[0]
    if grad_log is None:
        raise RuntimeError("Autograd failed to compute first derivative for log(psi).")

    (bsz, n, d) = x_req.shape
    lap_log = torch.zeros(bsz, device=x_req.device, dtype=x_req.dtype)
    for i in range(n):
        for j in range(d):
            g_ij = grad_log[:, i, j]
            grad2 = torch.autograd.grad(g_ij.sum(), x_req, create_graph=True, retain_graph=True)[0]
            if grad2 is None:
                raise RuntimeError("Autograd failed to compute second derivative for log(psi).")
            lap_log = lap_log + grad2[:, i, j]

    grad_sq = (grad_log * grad_log).sum(dim=(1, 2))
    return lap_log + grad_sq


def colloc_fd_loss(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    omega: float,
    params: dict,
    system: SystemConfig,
    h: float,
    laplacian_mode: str = "fd",
    spin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, torch.Tensor, dict]:
    del params
    if laplacian_mode == "fd":
        lap_over_psi = _laplacian_over_psi_fd(psi_log_fn, x, h)
    elif laplacian_mode == "autograd":
        lap_over_psi = _laplacian_over_psi_autograd(psi_log_fn, x)
    else:
        raise ValueError(
            f"Unknown laplacian_mode '{laplacian_mode}'. Expected 'fd' or 'autograd'."
        )
    v = _potential_energy(x, omega=omega, system=system, spin=spin)
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
    h: float = 0.01,
    laplacian_mode: str = "fd",
    spin: torch.Tensor | None = None,
) -> torch.Tensor:
    (_, _, e_loc, _) = colloc_fd_loss(
        psi_log_fn,
        x,
        omega=omega,
        params=params,
        system=system,
        h=h,
        laplacian_mode=laplacian_mode,
        spin=spin,
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
    h: float = 0.01,
    laplacian_mode: str = "fd",
    spin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, torch.Tensor, dict]:
    del direct_weight
    return colloc_fd_loss(
        psi_log_fn,
        x,
        omega=omega,
        params=params,
        system=system,
        h=h,
        laplacian_mode=laplacian_mode,
        spin=spin,
    )
