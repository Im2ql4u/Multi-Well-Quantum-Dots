#!/usr/bin/env python
"""
Measure spatial entanglement of a VMC N=2 one-per-well wavefunction.

Method:
  ψ(x1, x2) is evaluated on a 2D grid per particle.
  The wavefunction matrix M[i,j] = ψ(x1_i, x2_j) * √w1_i * √w2_j
  is constructed with Gauss-Hermite quadrature weights.

  1. Schmidt decomposition (SVD of M):
       M = U Σ Vᵀ → Schmidt coefficients σ_k
       Von Neumann entropy S = -Σ p_k log(p_k), p_k = σ_k² / Σσ_k²

  2. Partial transpose negativity:
       ρ = |ψ><ψ| reshaped as (d1,d2,d1,d2)
       ρᵀ²(i,j,k,l) = ρ(i,l,k,j)  [transpose subsystem 2]
       Negativity N = Σ_k max(0, -λ_k) where λ_k are eigenvalues of ρᵀ²

  For pure states S=0 ↔ product state, N=0 ↔ product state.

Usage:
  PYTHONPATH=src python scripts/measure_entanglement.py \
      --result-dir results/smoke_p2_n2_1p1w_gs_s42_20260411_074935 \
      --npts 30 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import SystemConfig
from imaginary_time_pinn import SpectralG, TauConditionedG
from observables.entanglement import (
    compute_dot_projected_entanglement,
    compute_particle_entanglement,
)
from run_ground_state import _build_system
from wavefunction import GroundStateWF, setup_closed_shell_system


# ─── Loading ──────────────────────────────────────────────────────────────────

def _load_ground_state_from_state(
    raw_cfg: dict,
    state_dict: dict,
    device: str,
) -> tuple[GroundStateWF, SystemConfig]:
    system = _build_system(raw_cfg["system"])
    if system.n_particles != 2:
        raise ValueError(f"Entanglement measurement requires N=2, got N={system.n_particles}")

    arch_cfg = raw_cfg.get("architecture", {})
    allow_missing_dmc = bool(raw_cfg.get("allow_missing_dmc", True))
    E_ref_raw = raw_cfg.get("E_ref", "auto")

    dev = torch.device(device)
    dtype = torch.float64

    C_occ, spin, params = setup_closed_shell_system(
        system,
        device=str(dev),
        dtype=dtype,
        E_ref=E_ref_raw,
        allow_missing_dmc=allow_missing_dmc,
    )

    model = GroundStateWF(
        system,
        C_occ,
        spin,
        params,
        arch_type=arch_cfg.get("arch_type", "pinn"),
        pinn_hidden=int(arch_cfg.get("pinn_hidden", 64)),
        pinn_layers=int(arch_cfg.get("pinn_layers", 2)),
        bf_hidden=int(arch_cfg.get("bf_hidden", 32)),
        bf_layers=int(arch_cfg.get("bf_layers", 2)),
        use_well_features=bool(arch_cfg.get("use_well_features", False)),
        use_well_backflow=bool(arch_cfg.get("use_well_backflow", False)),
        use_backflow=bool(arch_cfg.get("use_backflow", True)),
        singlet=bool(arch_cfg.get("singlet", False)),
    )

    # Handle C_occ shape mismatch between old (n_orb=n_up) and new (n_orb=n_up+n_down) code.
    ckpt_C_occ = state_dict.get("C_occ", None)
    if ckpt_C_occ is not None and ckpt_C_occ.shape != C_occ.shape:
        print(
            f"  NOTE: checkpoint C_occ shape {tuple(ckpt_C_occ.shape)} != "
            f"model C_occ shape {tuple(C_occ.shape)} — rebuilding model to match checkpoint."
        )
        n_orb_ckpt = ckpt_C_occ.shape[1]
        n_up = int(params["n_up"])
        n_basis = C_occ.shape[0]
        C_occ_compat = torch.zeros(n_basis, n_orb_ckpt, device=dev, dtype=dtype)
        for col in range(n_orb_ckpt):
            C_occ_compat[col, col] = 1.0
        params_compat = dict(params)
        params_compat["up_col_idx"] = list(range(min(n_up, n_orb_ckpt)))
        params_compat["down_col_idx"] = list(range(min(n_up, n_orb_ckpt)))
        model = GroundStateWF(
            system,
            C_occ_compat,
            spin,
            params_compat,
            arch_type=arch_cfg.get("arch_type", "pinn"),
            pinn_hidden=int(arch_cfg.get("pinn_hidden", 64)),
            pinn_layers=int(arch_cfg.get("pinn_layers", 2)),
            bf_hidden=int(arch_cfg.get("bf_hidden", 32)),
            bf_layers=int(arch_cfg.get("bf_layers", 2)),
            use_well_features=bool(arch_cfg.get("use_well_features", False)),
            use_well_backflow=bool(arch_cfg.get("use_well_backflow", False)),
            use_backflow=bool(arch_cfg.get("use_backflow", True)),
            singlet=bool(arch_cfg.get("singlet", False)),
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # If backflow keys are missing but PINN keys are present, the checkpoint uses
        # an older backflow architecture. Rebuild without backflow to load cleanly.
        backflow_missing = [k for k in missing if k.startswith("backflow.")]
        if backflow_missing and len(backflow_missing) == len(missing):
            print(
                f"  NOTE: backflow keys missing ({backflow_missing}) — "
                "rebuilding model with use_backflow=False for clean load."
            )
            model = GroundStateWF(
                system,
                model.C_occ,
                model.spin_template,
                model.sd_params,
                arch_type=arch_cfg.get("arch_type", "pinn"),
                pinn_hidden=int(arch_cfg.get("pinn_hidden", 64)),
                pinn_layers=int(arch_cfg.get("pinn_layers", 2)),
                use_backflow=False,
                singlet=bool(arch_cfg.get("singlet", False)),
            )
            missing2, unexpected2 = model.load_state_dict(state_dict, strict=False)
            if missing2:
                print(f"  WARNING: still missing keys: {missing2}")
        else:
            print(f"  WARNING: missing keys (non-backflow) in checkpoint: {missing}")
    if unexpected:
        print(f"  NOTE: unexpected keys in checkpoint (ignored): {unexpected}")
    model.eval()
    model.to(device=dev, dtype=dtype)
    return model, system


def load_model(result_dir: str | Path, device: str) -> tuple[GroundStateWF, SystemConfig]:
    result_dir = Path(result_dir).expanduser().resolve()
    model_path = result_dir / "model.pt"
    config_path = result_dir / "config.yaml"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found in {result_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {result_dir}")

    with config_path.open("r") as f:
        raw_cfg = yaml.safe_load(f)
    state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    return _load_ground_state_from_state(raw_cfg, state_dict, device)


def _system_max_well_sep(system: SystemConfig) -> float:
    if len(system.wells) < 2:
        return 0.0
    return max(abs(float(w1.center[0]) - float(w2.center[0])) for i, w1 in enumerate(system.wells) for w2 in system.wells[i + 1 :])


def _measurement_system_for_quench(system: SystemConfig, final_sep: float | None) -> SystemConfig:
    if len(system.wells) != 2 or final_sep is None:
        return system
    measure_sep = max(_system_max_well_sep(system), abs(float(final_sep)))
    return SystemConfig.double_dot(
        N_L=int(system.wells[0].n_particles),
        N_R=int(system.wells[1].n_particles),
        sep=measure_sep,
        omega=float(system.wells[0].omega),
        dim=int(system.dim),
    )


class TimeDependentWF(nn.Module):
    def __init__(self, ground_wf: GroundStateWF, g_net: nn.Module, tau: float):
        super().__init__()
        self.ground_wf = ground_wf
        self.g_net = g_net
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, log_psi = self.signed_log_psi(x)
        return log_psi

    def signed_log_psi(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self.ground_wf, "signed_log_psi"):
            raise RuntimeError("Ground-state wavefunction does not support signed evaluation.")
        tau_t = torch.full((x.shape[0],), self.tau, device=x.device, dtype=x.dtype)
        sign, log_psi = self.ground_wf.signed_log_psi(x)
        return sign, log_psi + self.g_net(x, tau_t)


def load_quench_model(
    checkpoint_path: str | Path,
    device: str,
    tau: float,
) -> tuple[nn.Module, SystemConfig]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Quench checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    quench_cfg = payload.get("config", {})
    ground_state_dir = quench_cfg.get("ground_state_dir")
    if not ground_state_dir:
        raise ValueError("Quench checkpoint does not contain ground_state_dir; cannot rebuild ψ0.")

    gs_dir = Path(ground_state_dir).expanduser()
    if not gs_dir.is_absolute():
        gs_dir = (Path(__file__).resolve().parent.parent / gs_dir).resolve()
    gs_config_path = gs_dir / "config.yaml"
    if not gs_config_path.exists():
        raise FileNotFoundError(f"Ground-state config not found: {gs_config_path}")

    with gs_config_path.open("r", encoding="utf-8") as f:
        gs_raw_cfg = yaml.safe_load(f)

    ground_wf, system = _load_ground_state_from_state(gs_raw_cfg, payload["ground_wf_state"], device)

    if bool(quench_cfg.get("use_spectral", True)):
        g_net = SpectralG(
            n_particles=int(quench_cfg["n_particles"]),
            dim=int(quench_cfg["dim"]),
            n_modes=int(quench_cfg.get("g_modes", 3)),
            hidden=int(quench_cfg.get("g_hidden", 32)),
            n_layers=int(quench_cfg.get("g_layers", 2)),
        )
    else:
        g_net = TauConditionedG(
            n_particles=int(quench_cfg["n_particles"]),
            dim=int(quench_cfg["dim"]),
            hidden=int(quench_cfg.get("g_hidden", 64)),
            n_layers=int(quench_cfg.get("g_layers", 3)),
            tau_embed=int(quench_cfg.get("g_tau_embed", 32)),
            n_freq=int(quench_cfg.get("g_n_freq", 6)),
        )
    g_net.load_state_dict(payload["g_net_state"], strict=True)
    g_net.to(device=torch.device(device), dtype=torch.float64)
    g_net.eval()

    model = TimeDependentWF(ground_wf, g_net, tau=tau).to(device=torch.device(device), dtype=torch.float64)
    model.eval()
    measure_system = _measurement_system_for_quench(system, quench_cfg.get("well_sep_final"))
    return model, measure_system


# ─── Grid construction ────────────────────────────────────────────────────────

def _gauss_hermite_grid(center_x: float, omega: float, npts: int) -> tuple[np.ndarray, np.ndarray]:
    """1D Gauss-Hermite nodes and weights shifted to x-space for ω-HO.

    HO ground-state scale: l = 1/√ω. Nodes in probability space → x-space:
       x = center_x + √2 * l * xi,  w_x = w_xi / √π * √2 * l
    """
    xi, w_xi = np.polynomial.hermite.hermgauss(npts)
    ho_len = 1.0 / np.sqrt(max(omega, 1e-8))
    x = center_x + np.sqrt(2.0) * ho_len * xi
    # Weight includes the Gaussian measure factor — for bare quadrature of |ψ|²
    # we need dx integration weights (uniform spacing equivalent):
    w = w_xi * np.exp(xi**2) * np.sqrt(2.0) * ho_len  # removes e^{-x²} weight
    return x, w


def build_grids(
    system: SystemConfig, npts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build 2D grids (x,y) per particle using Gauss-Legendre on a box.

    The box spans from (leftmost well − margin) to (rightmost well + margin),
    where margin = 5 HO lengths. This avoids the double-counting problem that
    arises from naively unioning per-well Gauss-Hermite grids.

    Both particles use the same grid (needed for correct particle-entanglement
    measurement — each particle can be near any well).

    Returns:
        pts1: (nx*ny, 2) float64 — particle-1 (x,y) grid points
        pts2: (nx*ny, 2) float64 — particle-2 (x,y) grid points
        w1:   (nx*ny,)  float64 — quadrature weights for particle 1
        w2:   (nx*ny,)  float64 — quadrature weights for particle 2
    """
    wells = system.wells
    omega = system.omega
    ho_len = 1.0 / np.sqrt(max(omega, 1e-8))
    margin = 5.0 * ho_len

    # x-range: cover all wells plus margin
    cx_all = [float(w.center[0]) for w in wells]
    x_lo = min(cx_all) - margin
    x_hi = max(cx_all) + margin

    # y-range: symmetric around 0
    y_lo = -margin
    y_hi = margin

    # Gauss-Legendre on [x_lo, x_hi] and [y_lo, y_hi]
    xi_x, wi_x = np.polynomial.legendre.leggauss(npts)
    x_1d = 0.5 * (x_hi - x_lo) * xi_x + 0.5 * (x_hi + x_lo)
    wx_1d = 0.5 * (x_hi - x_lo) * wi_x

    xi_y, wi_y = np.polynomial.legendre.leggauss(npts)
    y_1d = 0.5 * (y_hi - y_lo) * xi_y + 0.5 * (y_hi + y_lo)
    wy_1d = 0.5 * (y_hi - y_lo) * wi_y

    # 2D grid: outer product of x_1d × y_1d
    X, Y = np.meshgrid(x_1d, y_1d, indexing="ij")
    WX, WY = np.meshgrid(wx_1d, wy_1d, indexing="ij")

    pts = np.stack([X.ravel(), Y.ravel()], axis=1)   # (npts*npts, 2)
    w = (WX * WY).ravel()

    # Both particles use the same full-coverage grid
    return pts, pts, w, w


# ─── Wavefunction evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate_psi_matrix(
    model: GroundStateWF,
    pts1: np.ndarray,
    pts2: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate ψ(x1_i, x2_j) → matrix of shape (n1, n2).

    Rows = particle-1 grid points, columns = particle-2 grid points.
    Uses batching over particle-1 to stay within GPU memory.
    """
    n1 = pts1.shape[0]
    n2 = pts2.shape[0]
    dev = torch.device(device)

    pts2_t = torch.tensor(pts2, dtype=torch.float64, device=dev)  # (n2, 2)
    psi_matrix = np.zeros((n1, n2), dtype=np.float64)

    for start in range(0, n1, batch_size):
        end = min(start + batch_size, n1)
        chunk = pts1[start:end]          # (B, 2)
        B = len(chunk)

        # Build (B*n2, 2, 2): particle-1 repeated for each particle-2 point
        p1 = torch.tensor(chunk, dtype=torch.float64, device=dev)     # (B, 2)
        p1_exp = p1.unsqueeze(1).expand(B, n2, 2).reshape(B * n2, 2)  # (B*n2, 2)
        p2_exp = pts2_t.unsqueeze(0).expand(B, n2, 2).reshape(B * n2, 2)  # (B*n2, 2)

        # Stack as positions: (batch, n_particles=2, dim=2)
        x = torch.stack([p1_exp, p2_exp], dim=1)  # (B*n2, 2, 2)

        if hasattr(model, "signed_log_psi"):
            sign, log_psi = model.signed_log_psi(x)   # type: ignore[attr-defined]
            psi_vals = (sign * torch.exp(log_psi)).cpu().numpy()
        else:
            log_psi = model(x)
            psi_vals = torch.exp(log_psi).cpu().numpy()
        psi_matrix[start:end, :] = psi_vals.reshape(B, n2)

        if start % (10 * batch_size) == 0:
            frac = end / n1
            print(f"  evaluating ψ matrix: {end}/{n1} ({100*frac:.0f}%)", flush=True)

    return psi_matrix


# ─── Entanglement measures ────────────────────────────────────────────────────

def compute_entanglement(
    psi_matrix: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> dict:
    return compute_particle_entanglement(psi_matrix, w1, w2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Measure entanglement of VMC N=2 wavefunction.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result-dir", help="Path to ground-state result directory (model.pt + config.yaml)")
    source.add_argument("--quench-checkpoint", help="Path to quench checkpoint.pt produced by imaginary_time_pinn.py")
    parser.add_argument("--tau", type=float, default=0.0, help="Imaginary time for --quench-checkpoint evaluation")
    parser.add_argument("--npts", type=int, default=30, help="Number of Gauss-Hermite quadrature points per dimension (default 30 → 900 pts per particle)")
    parser.add_argument("--device", default="cuda:0", help="Torch device for wavefunction evaluation")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for ψ evaluation")
    parser.add_argument(
        "--dot-basis",
        choices=["localized_ho", "region_average"],
        default="localized_ho",
        help="Dot projection basis (default: localized_ho).",
    )
    parser.add_argument(
        "--dot-max-shell",
        type=int,
        default=1,
        help="Maximum 2D HO shell per well for --dot-basis localized_ho.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    source_desc: str
    if args.result_dir:
        print(f"Loading model from {args.result_dir} ...")
        model, system = load_model(args.result_dir, args.device)
        source_desc = str(args.result_dir)
    else:
        print(f"Loading quench checkpoint from {args.quench_checkpoint} at tau={args.tau:.4f} ...")
        model, system = load_quench_model(args.quench_checkpoint, args.device, args.tau)
        source_desc = str(args.quench_checkpoint)
    print(f"  N={system.n_particles}, dim={system.dim}, omega={system.omega}")
    print(f"  Wells: {[(w.center, w.n_particles) for w in system.wells]}")

    print(f"\nBuilding grids ({args.npts} pts/dim = {args.npts**2} per particle) ...")
    pts1, pts2, w1, w2 = build_grids(system, args.npts)
    print(f"  pts1.shape={pts1.shape}, pts2.shape={pts2.shape}")
    print(f"  w1 range: [{w1.min():.3e}, {w1.max():.3e}], sum={w1.sum():.4f}")
    print(f"  w2 range: [{w2.min():.3e}, {w2.max():.3e}], sum={w2.sum():.4f}")

    print(f"\nEvaluating ψ(x1, x2) on {pts1.shape[0]}×{pts2.shape[0]} grid ...")
    psi_matrix = evaluate_psi_matrix(model, pts1, pts2, args.device, batch_size=args.batch_size)
    print(f"  ψ range: [{psi_matrix.min():.4e}, {psi_matrix.max():.4e}]")
    print(f"  Non-finite values: {int(~np.isfinite(psi_matrix).all())}")

    if not np.isfinite(psi_matrix).all():
        n_bad = int(np.sum(~np.isfinite(psi_matrix)))
        print(f"  WARNING: {n_bad} non-finite values in ψ matrix. Replacing with 0.")
        psi_matrix = np.nan_to_num(psi_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    print("\nComputing particle-coordinate entanglement measures ...")
    result = compute_entanglement(psi_matrix, w1, w2)
    print(f"  ‖ψ‖²  (before normalisation)    = {result['norm2_before_normalisation']:.8f}  (should be ≈1 if grid matches training)")
    print(f"  ‖M‖² (should be 1.0 after normalisation) = {result['norm_M_squared']:.8f}")

    print("\n" + "="*60)
    print("  ENTANGLEMENT RESULTS")
    print("="*60)
    print(f"  Von Neumann entropy S       = {result['von_neumann_entropy']:.6f}")
    print(f"  Purity Tr[ρ₁²]             = {result['purity']:.6f}  (1.0 = product state)")
    print(f"  Linear entropy 1-Tr[ρ₁²]   = {result['linear_entropy']:.6f}  (0.0 = product state)")
    print(f"  Negativity N                = {result['negativity']:.6f}  (0.0 = product state)")
    print(f"  Log-negativity E_N          = {result['log_negativity']:.6f}")
    print(f"  Effective Schmidt rank      = {result['effective_schmidt_rank']}")
    print(f"  Schmidt values (top 10):    {[f'{v:.4f}' for v in result['schmidt_values_top10']]}")
    print(f"  Schmidt probs  (top 10):    {[f'{v:.4f}' for v in result['schmidt_probs_top10']]}")
    print("="*60)

    dot_result = None
    if len(system.wells) == 2:
        print("\nComputing dot-projected entanglement measures ...")
        dot_result = compute_dot_projected_entanglement(
            psi_matrix,
            pts1,
            pts2,
            w1,
            w2,
            system,
            projection_basis=args.dot_basis,
            max_ho_shell=args.dot_max_shell,
        )
        print("\n" + "="*60)
        print("  DOT-PROJECTED RESULTS")
        print("="*60)
        print(f"  Projection basis            = {dot_result['projection_basis']}")
        print(f"  Basis dimensions            = {dot_result['basis_dimensions']}")
        print(f"  Projected subspace weight   = {dot_result['projected_subspace_weight']:.6f}")
        print(f"  Von Neumann entropy S       = {dot_result['von_neumann_entropy']:.6f}")
        print(f"  Negativity N                = {dot_result['negativity']:.6f}")
        print(f"  Log-negativity E_N          = {dot_result['log_negativity']:.6f}")
        print(
            f"  Direct PT negativity        = {dot_result['direct_partial_transpose']['negativity_direct']:.6f}"
        )
        print(
            f"  Dot-label PT negativity     = {dot_result['dot_label_partial_transpose']['negativity']:.6f}"
        )
        print(f"  Sector probabilities        = {dot_result['sector_probabilities']}")
        print(f"  Projected sector probs      = {dot_result['projected_sector_probabilities']}")
        print("="*60)

    # Interpretation
    S = result["von_neumann_entropy"]
    if S < 0.01:
        interp = "PRODUCT STATE (no entanglement)"
    elif S < 0.1:
        interp = "WEAKLY ENTANGLED"
    elif S < 0.5:
        interp = "MODERATELY ENTANGLED"
    else:
        interp = "STRONGLY ENTANGLED"
    print(f"\n  Interpretation: {interp}")

    full_result = {
        "source": source_desc,
        "tau": args.tau if args.quench_checkpoint else None,
        "npts": args.npts,
        "n_grid_per_particle": int(pts1.shape[0]),
        "dot_projection_config": {
            "basis": args.dot_basis,
            "max_ho_shell": args.dot_max_shell,
        },
        "system": {
            "n_particles": int(system.n_particles),
            "dim": int(system.dim),
            "omega": float(system.omega),
            "n_wells": len(system.wells),
            "well_centers": [list(w.center) for w in system.wells],
        },
        "entanglement": result,
        "dot_projected_entanglement": dot_result,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(full_result, f, indent=2)
        print(f"\nSaved to {args.out}")

    return full_result


if __name__ == "__main__":
    main()
