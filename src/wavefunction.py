from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

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


def resolve_spin_configuration(
    system: SystemConfig,
    spin_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an explicit spin pattern or sector for a fixed-spin run."""
    n_particles = int(system.n_particles)
    default_n_up = (n_particles + 1) // 2
    default_n_down = n_particles // 2

    if spin_cfg is None:
        pattern = [0] * default_n_up + [1] * default_n_down
        source = "default_closed_shell"
    else:
        if not isinstance(spin_cfg, dict):
            raise ValueError("spin config must be a mapping when provided.")
        pattern_cfg = spin_cfg.get("pattern")
        n_up_cfg = spin_cfg.get("n_up")
        n_down_cfg = spin_cfg.get("n_down")
        if pattern_cfg is not None and (n_up_cfg is not None or n_down_cfg is not None):
            raise ValueError("spin config must specify either pattern or n_up/n_down, not both.")

        if pattern_cfg is not None:
            pattern = [int(value) for value in pattern_cfg]
            source = "explicit_pattern"
        elif n_up_cfg is not None or n_down_cfg is not None:
            if n_up_cfg is None or n_down_cfg is None:
                raise ValueError("spin config must provide both n_up and n_down.")
            n_up = int(n_up_cfg)
            n_down = int(n_down_cfg)
            if n_up < 0 or n_down < 0:
                raise ValueError("spin counts must be non-negative.")
            if n_up + n_down != n_particles:
                raise ValueError(
                    f"spin counts must sum to system.n_particles={n_particles}, got {n_up}+{n_down}."
                )
            pattern = [0] * n_up + [1] * n_down
            source = "sector_counts"
        else:
            pattern = [0] * default_n_up + [1] * default_n_down
            source = "default_closed_shell"

    if len(pattern) != n_particles:
        raise ValueError(
            f"spin pattern length must equal system.n_particles={n_particles}, got {len(pattern)}."
        )
    if any(value not in (0, 1) for value in pattern):
        raise ValueError("spin pattern must contain only 0 (up) and 1 (down).")

    n_up = sum(1 for value in pattern if value == 0)
    n_down = n_particles - n_up
    return {
        "pattern": pattern,
        "n_up": n_up,
        "n_down": n_down,
        "source": source,
        "label": f"{n_up}up_{n_down}down",
    }


def assess_magnetic_response_capability(
    system: SystemConfig,
    spin_template: torch.Tensor,
    *,
    supports_spin_superposition: bool = False,
) -> dict[str, Any]:
    """Summarize whether the current fixed-spin ansatz can represent a magnetic response.

    The current generalized path uses a single fixed spin sector. Under that
    assumption, the implemented Zeeman coupling contributes only a constant
    energy offset for any uniform longitudinal field.
    """
    spin_1d = spin_template.detach().to(device="cpu")
    if spin_1d.ndim != 1 or spin_1d.numel() != system.n_particles:
        raise ValueError(
            "assess_magnetic_response_capability expects a 1D spin template with length system.n_particles."
        )

    bx, by, bz = system.magnetic_field_vector
    transverse_components_present = abs(float(bx)) > 0.0 or abs(float(by)) > 0.0
    longitudinal_component = float(bz)

    if system.zeeman_particle_indices is not None:
        selected_indices = [int(idx) for idx in system.zeeman_particle_indices]
        zeeman_scope = "particle_subset"
    elif system.zeeman_electron1_only:
        selected_indices = [0]
        zeeman_scope = "electron1_only"
    else:
        selected_indices = list(range(system.n_particles))
        zeeman_scope = "all_particles"

    spin_cpu = spin_1d.to(dtype=torch.float64)
    if torch.all((spin_1d == 0) | (spin_1d == 1)):
        spin_z = 1.0 - 2.0 * spin_cpu
    else:
        spin_z = spin_cpu
    selected_spin_projection = float(spin_z[selected_indices].sum().item())
    constant_energy_shift = (
        0.5
        * float(system.g_factor)
        * float(system.mu_B)
        * longitudinal_component
        * selected_spin_projection
    )

    notes: list[str] = []
    if transverse_components_present:
        notes.append(
            "Current potential implementation uses only B_direction[2]; transverse magnetic-field components are ignored."
        )

    zeeman_active = abs(longitudinal_component) > 0.0
    fixed_spin_uniform_zeeman = zeeman_active and not supports_spin_superposition
    if fixed_spin_uniform_zeeman:
        notes.append(
            "Current generalized GroundStateWF uses one fixed spin template, so uniform longitudinal Zeeman coupling is a constant offset rather than a state-changing interaction."
        )
    elif not zeeman_active and abs(float(system.B_magnitude)) > 0.0:
        notes.append(
            "Configured magnetic field has no implemented longitudinal component, so the present Hamiltonian adds no magnetic term."
        )

    state_response_supported = bool(zeeman_active and supports_spin_superposition)
    if state_response_supported:
        classification = "nontrivial_state_response_supported"
    elif fixed_spin_uniform_zeeman:
        classification = "constant_zeeman_shift_only"
    elif abs(float(system.B_magnitude)) > 0.0:
        classification = "no_implemented_longitudinal_coupling"
    else:
        classification = "magnetic_field_disabled"

    return {
        "magnetic_field_configured": abs(float(system.B_magnitude)) > 0.0,
        "magnetic_field_vector": [float(bx), float(by), float(bz)],
        "longitudinal_component": longitudinal_component,
        "transverse_components_present": transverse_components_present,
        "zeeman_scope": zeeman_scope,
        "selected_particle_indices": selected_indices,
        "selected_spin_projection": selected_spin_projection,
        "constant_energy_shift": constant_energy_shift,
        "ansatz_fixed_spin_sector": True,
        "supports_spin_superposition": bool(supports_spin_superposition),
        "state_response_supported": state_response_supported,
        "structurally_trivial_uniform_zeeman": fixed_spin_uniform_zeeman,
        "classification": classification,
        "notes": notes,
    }


def setup_closed_shell_system(
    system: SystemConfig,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    E_ref: str | float,
    allow_missing_dmc: bool = False,
    spin_pattern: Sequence[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if spin_pattern is None:
        spin_meta = resolve_spin_configuration(system, None)
    else:
        if isinstance(spin_pattern, torch.Tensor):
            pattern_values = [int(value) for value in spin_pattern.detach().cpu().tolist()]
        else:
            pattern_values = [int(value) for value in spin_pattern]
        spin_meta = resolve_spin_configuration(system, {"pattern": pattern_values})

    n_up = int(spin_meta["n_up"])
    n_down = int(spin_meta["n_down"])
    is_open_shell = n_up != n_down
    dim = system.dim
    n_wells = len(system.wells)
    # For multi-well systems every particle needs its own spatial orbital so all
    # well-localised HO functions appear in the SD.  Closed-shell doubling
    # (n_orb = n_up) is correct only for a single-centre dot.
    _multi_well = n_wells > 1
    n_orb = (n_up + n_down) if (is_open_shell or _multi_well) else n_up

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

    # Build occupied orbital columns by distributing low local basis functions
    # round-robin across wells: [(well0,orb0), (well1,orb0), ..., (well0,orb1), ...].
    occ_basis_indices: list[int] = []
    local_idx = 0
    while len(occ_basis_indices) < n_orb:
        for w in range(n_wells):
            occ_basis_indices.append(w * n_basis_per_well + local_idx)
            if len(occ_basis_indices) >= n_orb:
                break
        local_idx += 1

    C_occ = torch.zeros(n_basis, n_orb, device=device, dtype=dtype)
    for col, basis_idx in enumerate(occ_basis_indices):
        C_occ[basis_idx, col] = 1.0

    spin = torch.tensor(spin_meta["pattern"], device=device, dtype=torch.int64)
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
        "n_up": int(n_up),
        "n_down": int(n_down),
        "spin_pattern": list(spin_meta["pattern"]),
        "spin_label": str(spin_meta["label"]),
        "up_col_idx": list(range(n_up)),
        "down_col_idx": (list(range(n_up, n_up + n_down)) if (is_open_shell or _multi_well) else list(range(n_up))),
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
        use_well_backflow: bool = False,
        use_backflow: bool = True,
        singlet: bool = False,
        multi_ref: bool = False,
    ) -> None:
        super().__init__()
        self.system = system
        self.register_buffer("C_occ", C_occ.detach().clone())
        self.sd_params = dict(params)  # For Slater determinant basis evaluation

        self.singlet = bool(singlet)
        self.multi_ref = bool(multi_ref)
        self.arch_type = str(arch_type).lower()
        if self.arch_type not in {"pinn", "ctnn", "unified"}:
            raise ValueError(
                f"Unknown arch_type '{arch_type}'. Expected one of: pinn, ctnn, unified."
            )
        if self.singlet and (len(system.wells) != 2 or system.n_particles != 2):
            raise ValueError(
                "Singlet permanent ansatz currently requires N=2 and exactly 2 wells."
            )
        if self.multi_ref and len(system.wells) < 2:
            raise ValueError(
                "Multi-reference ansatz requires at least 2 wells."
            )
        if self.singlet and self.multi_ref:
            raise ValueError(
                "singlet and multi_ref are mutually exclusive."
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
                    use_well_backflow=bool(use_well_backflow),
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
                    use_well_backflow=bool(use_well_backflow),
                    out_bound="tanh",
                    bf_scale_init=0.01,
                    zero_init_last=True,
                    omega=system.omega,
                )

        # Keep module params aligned with the basis tensor dtype/device.
        self.to(device=C_occ.device, dtype=C_occ.dtype)

    def _permanent_sign_logabs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric permanent for the singlet: φ_L(r₁)φ_R(r₂) + φ_R(r₁)φ_L(r₂).

        Uses the lowest HO orbital centered at each well. Only valid for N=2, 2-well.
        Returns (sign, log|permanent|) with shapes (B,).
        """
        B, N, d = x.shape
        wells = self.system.wells
        c0 = torch.tensor(wells[0].center, device=x.device, dtype=x.dtype).view(1, 1, d)
        c1 = torch.tensor(wells[1].center, device=x.device, dtype=x.dtype).view(1, 1, d)

        if d == 2:
            phi_L = self._evaluate_ho_basis_2d(x - c0)[:, :, 0]  # (B, N)
            phi_R = self._evaluate_ho_basis_2d(x - c1)[:, :, 0]  # (B, N)
        elif d == 1:
            omega_p = {"omega": float(self.system.omega)}
            phi_L = evaluate_basis_functions_torch(
                (x - c0).squeeze(-1), 1, params=omega_p
            )[:, :, 0]  # (B, N)
            phi_R = evaluate_basis_functions_torch(
                (x - c1).squeeze(-1), 1, params=omega_p
            )[:, :, 0]  # (B, N)
        else:
            raise ValueError(f"Unsupported dim={d} for permanent ansatz.")

        # permanent = φ_L(r₁)φ_R(r₂) + φ_R(r₁)φ_L(r₂)
        term1 = phi_L[:, 0] * phi_R[:, 1]  # shape (B,)
        term2 = phi_R[:, 0] * phi_L[:, 1]  # shape (B,)
        perm = term1 + term2

        sign_perm = torch.sign(perm)
        log_perm = torch.log(torch.abs(perm).clamp(min=1e-30))
        return sign_perm, log_perm

    def _multi_ref_sign_logabs(
        self,
        x: torch.Tensor,
        spin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum over all C(N_wells, n_up) well-to-spin assignments.

        For one particle per well, each reference assigns n_up wells to the
        spin-up sector and n_down wells to spin-down. The resulting Slater
        determinants are summed with their signs using a numerically stable
        signed logsumexp. For N=2 this exactly recovers the singlet permanent.
        """
        B, N, d = x.shape
        wells = self.system.wells
        n_wells = len(wells)

        # Ground-state HO orbital (K=0) per well at all particle positions.
        # phi_all[b, i, w] = φ_w^0(r_i),  shape (B, N, N_wells)
        Phi_per_well = []
        for well in wells:
            center = torch.tensor(well.center, device=x.device, dtype=x.dtype).view(1, 1, d)
            x_shifted = x - center
            if d == 2:
                phi_w = self._evaluate_ho_basis_2d(x_shifted)[:, :, 0]  # (B, N)
            elif d == 1:
                omega_p = {"omega": float(self.system.omega)}
                phi_w = evaluate_basis_functions_torch(
                    x_shifted.squeeze(-1), 1, params=omega_p
                )[:, :, 0]  # (B, N)
            else:
                raise ValueError(f"Unsupported dim={d} for multi-reference ansatz.")
            Phi_per_well.append(phi_w)
        phi_all = torch.stack(Phi_per_well, dim=-1)  # (B, N, N_wells)

        spin_1d = spin if spin.ndim == 1 else spin[0]
        idx_up = (spin_1d == 0).nonzero(as_tuple=True)[0]
        idx_down = (spin_1d == 1).nonzero(as_tuple=True)[0]
        n_up = int(idx_up.numel())
        n_down = int(idx_down.numel())

        phi_up = phi_all[:, idx_up, :]    # (B, n_up, N_wells)
        phi_down = phi_all[:, idx_down, :]  # (B, n_down, N_wells)

        signs_list: list[torch.Tensor] = []
        logabs_list: list[torch.Tensor] = []

        for up_wells in combinations(range(n_wells), n_up):
            down_wells = [w for w in range(n_wells) if w not in up_wells]
            up_idx = torch.tensor(list(up_wells), device=x.device, dtype=torch.long)
            down_idx = torch.tensor(down_wells, device=x.device, dtype=torch.long)

            if n_up > 0:
                Psi_up_k = phi_up[:, :, up_idx]      # (B, n_up, n_up)
                sign_up_k, log_up_k = torch.linalg.slogdet(Psi_up_k)
            else:
                sign_up_k = x.new_ones(B)
                log_up_k = x.new_zeros(B)

            if n_down > 0:
                Psi_down_k = phi_down[:, :, down_idx]  # (B, n_down, n_down)
                sign_down_k, log_down_k = torch.linalg.slogdet(Psi_down_k)
            else:
                sign_down_k = x.new_ones(B)
                log_down_k = x.new_zeros(B)

            signs_list.append(sign_up_k * sign_down_k)
            logabs_list.append(log_up_k + log_down_k)

        signs = torch.stack(signs_list, dim=0)   # (n_refs, B)
        logabs = torch.stack(logabs_list, dim=0)  # (n_refs, B)

        # Stable signed logsumexp.
        log_max, _ = logabs.max(dim=0)  # (B,)
        norm_sum = (signs * torch.exp(logabs - log_max.unsqueeze(0))).sum(dim=0)  # (B,)
        sign_total = torch.sign(norm_sum)
        log_total = torch.log(norm_sum.abs().clamp(min=1e-30)) + log_max
        return sign_total, log_total

    def _evaluate_ho_basis_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate 2D HO basis as outer product of 1D bases."""
        nx = self.sd_params["nx"]
        ny = self.sd_params["ny"]
        omega_p = {"omega": float(self.system.omega)}
        phi_x = evaluate_basis_functions_torch(x[..., 0], nx, params=omega_p)
        phi_y = evaluate_basis_functions_torch(x[..., 1], ny, params=omega_p)
        prod = phi_x.unsqueeze(-1) * phi_y.unsqueeze(-2)  # (B,N,nx,ny)
        return prod.reshape(x.shape[0], x.shape[1], nx * ny)

    def _slater_det_sign_logabs(
        self,
        x: torch.Tensor,
        spin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute determinant sign and log-absolute value via LCAO."""
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
        elif len(wells) > 2:
            # Multi-well (>2) basis: concatenate per-well HO bases centered at each well.
            # This keeps a simple linear-combination basis and avoids hard-coding a specific
            # bonding pattern beyond the double-dot case.
            Phi_parts = []
            for well in wells:
                center = torch.tensor(well.center, device=x.device, dtype=x.dtype).view(1, 1, d)
                x_shifted = x - center
                if d == 2:
                    Phi_w = self._evaluate_ho_basis_2d(x_shifted)
                elif d == 1:
                    omega_p = {"omega": float(self.system.omega)}
                    Phi_w = evaluate_basis_functions_torch(
                        x_shifted.squeeze(-1), self.sd_params["nx"], params=omega_p
                    )
                else:
                    raise ValueError(f"Unsupported dim={d}")
                Phi_parts.append(Phi_w)
            Phi = torch.cat(Phi_parts, dim=-1)
        else:
            raise ValueError(f"Unsupported number of wells: {len(wells)}")

        # Slater matrix: (B, N, n_occ)
        Psi = torch.matmul(Phi, self.C_occ)

        # Split by spin and compute log|det| for each block.
        spin_1d = spin if spin.ndim == 1 else spin[0]
        idx_up = (spin_1d == 0).nonzero(as_tuple=True)[0]
        idx_down = (spin_1d == 1).nonzero(as_tuple=True)[0]

        n_up = int(idx_up.numel())
        n_down = int(idx_down.numel())
        up_cols = self.sd_params.get("up_col_idx", list(range(n_up)))
        down_cols = self.sd_params.get("down_col_idx", list(range(n_down)))

        Psi_up = Psi[:, idx_up, :][:, :, up_cols]      # (B, n_up, n_up)
        Psi_down = Psi[:, idx_down, :][:, :, down_cols]  # (B, n_down, n_down)

        if n_up > 0:
            sign_up, log_up = torch.linalg.slogdet(Psi_up)
        else:
            sign_up = torch.ones(B, device=x.device, dtype=x.dtype)
            log_up = torch.zeros(B, device=x.device, dtype=x.dtype)

        if n_down > 0:
            sign_down, log_down = torch.linalg.slogdet(Psi_down)
        else:
            sign_down = torch.ones(B, device=x.device, dtype=x.dtype)
            log_down = torch.zeros(B, device=x.device, dtype=x.dtype)

        return sign_up * sign_down, log_up + log_down

    def _log_slater_det(self, x: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        _, logabs = self._slater_det_sign_logabs(x, spin)
        return logabs

    def signed_log_slater(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        return self._slater_det_sign_logabs(x_eval, spin)

    def signed_log_psi(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,N,D), got {tuple(x.shape)}")
        if x.shape[1] != self.system.n_particles or x.shape[2] != self.system.dim:
            raise ValueError(
                f"Input shape mismatch. Expected N={self.system.n_particles}"
                f", D={self.system.dim}, got {tuple(x.shape)}"
            )

        spin = self.spin_template.to(device=x.device)

        if self.singlet:
            # Singlet mode: symmetric permanent × symmetrized PINN correlator.
            # Backflow is intentionally skipped — it would break spatial symmetry.
            sign_perm, log_perm = self._permanent_sign_logabs(x)

            # Symmetrize J: log(J(r₁,r₂) + J(r₂,r₁)) via logaddexp.
            # Swap positions AND well_id so PINN's well-aware features stay consistent.
            idx_swap = torch.tensor([1, 0], device=x.device, dtype=torch.long)
            x_swap = x[:, idx_swap, :]
            well_id_swap = self.well_id[idx_swap]
            c1 = self.pinn(x, spin=spin, well_id=self.well_id, cusp_coords=x).squeeze(-1)
            c2 = self.pinn(x_swap, spin=spin, well_id=well_id_swap, cusp_coords=x_swap).squeeze(-1)
            correlator = torch.logaddexp(c1, c2)

            log_psi = log_perm + correlator
            if torch.isnan(log_psi).any():
                raise RuntimeError("NaN in log_psi (singlet mode) in GroundStateWF forward pass.")
            return sign_perm, log_psi

        if self.multi_ref:
            # Multi-reference: sum over C(N_wells, n_up) well-to-spin assignments.
            # Backflow is skipped to preserve permutation symmetry of the reference sum.
            sign_mr, log_mr = self._multi_ref_sign_logabs(x, spin)
            correlator = self.pinn(x, spin=spin, well_id=self.well_id, cusp_coords=x).squeeze(-1)
            log_psi = log_mr + correlator
            if torch.isnan(log_psi).any():
                raise RuntimeError("NaN in log_psi (multi_ref mode) in GroundStateWF forward pass.")
            return sign_mr, log_psi

        # Default one-per-well path.
        sign_sd, log_sd = self.signed_log_slater(x)
        correlator = self.pinn(x, spin=spin, well_id=self.well_id, cusp_coords=x).squeeze(-1)
        log_psi = log_sd + correlator
        if torch.isnan(log_psi).any():
            raise RuntimeError("NaN in log_psi in GroundStateWF forward pass.")
        return sign_sd, log_psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, log_psi = self.signed_log_psi(x)
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
