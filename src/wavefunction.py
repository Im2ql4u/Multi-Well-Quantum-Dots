from __future__ import annotations

import torch
import torch.nn as nn

from PINN import BackflowNet, CTNNBackflowNet, PINN
from config import SystemConfig, _lookup_dmc_energy
from functions.Slater_Determinant import evaluate_basis_functions_torch


def _min_ho_shell_2d(n_orb: int) -> int:
    """Minimum HO shell S so (S+1)(S+2)/2 >= n_orb in 2D."""
    S = 0
    while (S + 1) * (S + 2) // 2 < n_orb:
        S += 1
    return S


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
    dim = system.dim
    n_wells = len(system.wells)

    # Determine HO basis size for the Slater determinant.
    # For multi-well: each single-center basis produces bonding+anti-bonding
    # LCAO combinations, giving n_wells * n_basis_per_well total orbitals.
    if dim == 2:
        if n_wells > 1:
            import math as _math
            n_per_well = _math.ceil(n_orb / n_wells)
        else:
            n_per_well = n_orb
        S = _min_ho_shell_2d(n_per_well)
        nx = ny = S + 1
        n_basis_per_well = nx * ny
        n_basis = n_wells * n_basis_per_well
    elif dim == 1:
        if n_wells > 1:
            import math as _math
            n_per_well = _math.ceil(n_orb / n_wells)
        else:
            n_per_well = n_orb
        nx = n_per_well
        ny = 1
        n_basis_per_well = nx
        n_basis = n_wells * n_basis_per_well
    else:
        raise ValueError(f"Unsupported dim={dim} for closed-shell setup.")

    C_occ = torch.eye(n_basis, n_orb, device=device, dtype=dtype)
    spin = torch.tensor([0] * n_orb + [1] * n_orb, device=device, dtype=torch.int64)
    well_ids: list[int] = []
    for well_idx, well in enumerate(system.wells):
        well_ids.extend([well_idx] * int(well.n_particles))
    if len(well_ids) != int(system.n_particles):
        raise ValueError(
            "Well occupancy does not match system.n_particles when building well_id mapping."
        )
    well_id = torch.tensor(well_ids, device=device, dtype=torch.long)
    if E_ref == "auto":
        e_ref_val = float(system.n_particles)
    else:
        e_ref_val = float(E_ref)
    params = {
        "E_ref": e_ref_val,
        "n_particles": int(system.n_particles),
        "dim": int(dim),
        "omega": float(system.omega),
        "nx": nx,
        "ny": ny,
        "well_id": well_id,
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
        use_well_features: bool = False,
        use_backflow: bool = True,
    ) -> None:
        super().__init__()
        self.system = system
        self.register_buffer("C_occ", C_occ.detach().clone())
        self.sd_params = dict(params)  # For Slater determinant basis evaluation

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
        default_well_id = torch.zeros(system.n_particles, device=C_occ.device, dtype=torch.long)
        well_id = params.get("well_id", default_well_id)
        if not isinstance(well_id, torch.Tensor):
            well_id = torch.tensor(well_id, device=C_occ.device, dtype=torch.long)
        well_id = well_id.detach().clone().to(device=C_occ.device, dtype=torch.long)
        if well_id.ndim != 1 or well_id.numel() != system.n_particles:
            raise ValueError(
                "GroundStateWF expects well_id as a 1D tensor with length system.n_particles."
            )
        self.register_buffer("well_id", well_id, persistent=False)

        self.pinn = PINN(
            n_particles=system.n_particles,
            d=system.dim,
            omega=system.omega,
            hidden_dim=max(int(pinn_hidden), 16),
            n_layers=max(int(pinn_layers), 1),
            act="gelu",
            use_well_features=bool(use_well_features),
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
                    bf_scale_init=0.01,
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
                    bf_scale_init=0.01,
                    zero_init_last=True,
                    omega=system.omega,
                )

        # Keep module params aligned with the basis tensor dtype/device.
        self.to(device=C_occ.device, dtype=C_occ.dtype)

    def _evaluate_ho_basis_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate 2D HO basis as outer product of 1D bases."""
        nx = self.sd_params["nx"]
        ny = self.sd_params["ny"]
        omega_p = {"omega": float(self.system.omega)}
        phi_x = evaluate_basis_functions_torch(x[..., 0], nx, params=omega_p)
        phi_y = evaluate_basis_functions_torch(x[..., 1], ny, params=omega_p)
        prod = phi_x.unsqueeze(-1) * phi_y.unsqueeze(-2)  # (B,N,nx,ny)
        return prod.reshape(x.shape[0], x.shape[1], nx * ny)

    def _log_slater_det(self, x: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        """Compute log|SD| via LCAO for single- or multi-well systems."""
        B, N, d = x.shape
        wells = self.system.wells

        if len(wells) == 1:
            # Single-well: standard HO Slater determinant.
            center = torch.tensor(
                wells[0].center, device=x.device, dtype=x.dtype
            ).view(1, 1, d)
            x_shifted = x - center
            if d == 2:
                Phi = self._evaluate_ho_basis_2d(x_shifted)
            elif d == 1:
                omega_p = {"omega": float(self.system.omega)}
                Phi = evaluate_basis_functions_torch(
                    x_shifted.squeeze(-1), self.sd_params["nx"], params=omega_p
                )
            else:
                raise ValueError(f"Unsupported dim={d}")
        elif len(wells) == 2:
            # Two-well LCAO: bonding + anti-bonding combinations.
            # Interleave [σ_0, σ*_0, σ_1, σ*_1, ...] so that
            # C_occ=identity selects lowest-energy LCAO orbitals.
            centers = []
            for well in wells:
                centers.append(
                    torch.tensor(well.center, device=x.device, dtype=x.dtype).view(1, 1, d)
                )
            if d == 2:
                Phi_L = self._evaluate_ho_basis_2d(x - centers[0])
                Phi_R = self._evaluate_ho_basis_2d(x - centers[1])
            elif d == 1:
                omega_p = {"omega": float(self.system.omega)}
                Phi_L = evaluate_basis_functions_torch(
                    (x - centers[0]).squeeze(-1), self.sd_params["nx"], params=omega_p
                )
                Phi_R = evaluate_basis_functions_torch(
                    (x - centers[1]).squeeze(-1), self.sd_params["nx"], params=omega_p
                )
            else:
                raise ValueError(f"Unsupported dim={d}")
            bonding = Phi_L + Phi_R            # (B, N, K)
            anti_bonding = Phi_L - Phi_R        # (B, N, K)
            K = bonding.shape[-1]
            Phi = torch.zeros(B, N, 2 * K, device=x.device, dtype=x.dtype)
            Phi[:, :, 0::2] = bonding
            Phi[:, :, 1::2] = anti_bonding
        else:
            raise ValueError(f"Unsupported number of wells: {len(wells)}")

        # Slater matrix: (B, N, n_occ)
        Psi = torch.matmul(Phi, self.C_occ)

        # Split by spin and compute log|det| for each block.
        spin_1d = spin if spin.ndim == 1 else spin[0]
        idx_up = (spin_1d == 0).nonzero(as_tuple=True)[0]
        idx_down = (spin_1d == 1).nonzero(as_tuple=True)[0]

        Psi_up = Psi[:, idx_up, :]    # (B, n_occ, n_occ)
        Psi_down = Psi[:, idx_down, :]  # (B, n_occ, n_occ)

        _, log_up = torch.linalg.slogdet(Psi_up)
        _, log_down = torch.linalg.slogdet(Psi_down)

        return log_up + log_down

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
            dx = self.backflow(x, spin=spin, well_id=self.well_id)
            if not torch.isfinite(dx).all():
                raise RuntimeError("Non-finite backflow displacement in GroundStateWF.")
            x_eval = x + dx

        # Slater determinant envelope (includes Gaussian, excited orbitals,
        # antisymmetry, and multi-center LCAO for double dots).
        log_sd = self._log_slater_det(x, spin)

        correlator = self.pinn(x_eval, spin=spin, well_id=self.well_id).squeeze(-1)
        log_psi = log_sd + correlator
        if torch.isnan(log_psi).any():
            raise RuntimeError("NaN in log_psi in GroundStateWF forward pass.")
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
