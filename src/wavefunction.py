from __future__ import annotations

import torch
import torch.nn as nn

from PINN import BackflowNet, CTNNBackflowNet, PINN
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
    """Ground-state log-wavefunction with architecture dispatch.

    This wraps the stable PINN correlator and optionally applies a backflow
    coordinate transform before evaluating log-psi.
    """

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
        del C_occ, params
        self.system = system

        self.arch_type = str(arch_type).lower()
        if self.arch_type not in {"pinn", "ctnn", "unified"}:
            raise ValueError(
                f"Unknown arch_type '{arch_type}'. Expected one of: pinn, ctnn, unified."
            )

        if spin.ndim != 1 or spin.numel() != system.n_particles:
            raise ValueError(
                "GroundStateWF expects a 1D spin template with length system.n_particles."
            )
        self.register_buffer("spin_template", spin.detach().clone().to(torch.long), persistent=False)

        self.pinn = PINN(
            n_particles=system.n_particles,
            d=system.dim,
            omega=system.omega,
            hidden_dim=max(int(pinn_hidden), 16),
            n_layers=max(int(pinn_layers), 1),
            act="gelu",
        )

        self.backflow: nn.Module | None = None
        if use_backflow:
            bf_width = max(int(bf_hidden), 16)
            bf_depth = max(int(bf_layers), 2)
            if self.arch_type == "pinn":
                self.backflow = BackflowNet(
                    d=system.dim,
                    msg_hidden=bf_width,
                    msg_layers=bf_depth,
                    hidden=bf_width,
                    layers=bf_depth,
                    use_spin=True,
                    same_spin_only=False,
                    out_bound="tanh",
                    bf_scale_init=0.05,
                    zero_init_last=True,
                )
            else:
                # Use CTNN message passing for both ctnn and unified modes.
                self.backflow = CTNNBackflowNet(
                    d=system.dim,
                    msg_hidden=bf_width,
                    msg_layers=bf_depth,
                    hidden=bf_width,
                    layers=bf_depth,
                    use_spin=True,
                    same_spin_only=False,
                    out_bound="tanh",
                    bf_scale_init=0.05,
                    zero_init_last=True,
                    omega=system.omega,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,N,D), got {tuple(x.shape)}")
        if x.shape[1] != self.system.n_particles or x.shape[2] != self.system.dim:
            raise ValueError(
                f"Input shape mismatch. Expected N={self.system.n_particles}"
                f", D={self.system.dim}, got {tuple(x.shape)}"
            )

        spin = self.spin_template.to(device=x.device)
        x_eval = x
        if self.backflow is not None:
            dx = self.backflow(x, spin=spin)
            if not torch.isfinite(dx).all():
                raise RuntimeError("Non-finite backflow displacement in GroundStateWF.")
            x_eval = x + dx

        log_psi = self.pinn(x_eval, spin=spin).squeeze(-1)
        if not torch.isfinite(log_psi).all():
            raise RuntimeError("Non-finite log_psi in GroundStateWF forward pass.")
        return log_psi


class SlaterOnlyWF(GroundStateWF):
    """Compatibility alias for legacy scripts expecting this symbol."""

    def __init__(
        self,
        system: SystemConfig,
        C_occ: torch.Tensor,
        spin: torch.Tensor,
        params: dict,
        *,
        pinn_hidden: int = 32,
        pinn_layers: int = 2,
    ) -> None:
        super().__init__(
            system,
            C_occ,
            spin,
            params,
            arch_type="pinn",
            pinn_hidden=pinn_hidden,
            pinn_layers=pinn_layers,
            bf_hidden=32,
            bf_layers=2,
            use_backflow=False,
        )
