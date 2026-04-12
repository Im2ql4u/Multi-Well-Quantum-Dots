from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from run_ground_state import _build_system
from training.collocation import _laplacian_over_psi_fd
from training.sampling import adapt_sigma_fs, mcmc_resample
from wavefunction import GroundStateWF, resolve_reference_energy, setup_closed_shell_system


def potential_split(x: torch.Tensor, system, omega: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, _, dim = x.shape
    dtype = x.dtype
    device = x.device
    well_vals = []
    for well in system.wells:
        center = torch.tensor(well.center, device=device, dtype=dtype).view(1, 1, dim)
        dr = x - center
        v = 0.5 * float(well.omega) ** 2 * torch.sum(dr * dr, dim=-1)
        well_vals.append(v)
    v_stack = torch.stack(well_vals, dim=-1)
    t_soft = float(system.smooth_T)
    v_conf = -t_soft * torch.logsumexp(-v_stack / t_soft, dim=-1)
    v_conf = torch.sum(v_conf, dim=-1)

    if not system.coulomb:
        v_ee = torch.zeros_like(v_conf)
    else:
        eps = 1e-2 / max(float(omega), 1e-8) ** 0.5
        i, j = torch.triu_indices(system.n_particles, system.n_particles, offset=1, device=device)
        rij = x[:, i, :] - x[:, j, :]
        r2 = torch.sum(rij * rij, dim=-1)
        v_ee = torch.sum(1.0 / torch.sqrt(r2 + eps * eps), dim=-1)

    return v_conf + v_ee, v_conf, v_ee


def load_model_from_result_dir(result_dir: Path, device: torch.device, dtype: torch.dtype) -> tuple[GroundStateWF, object, dict]:
    cfg_path = result_dir / "config.yaml"
    model_path = result_dir / "model.pt"
    raw = yaml.safe_load(cfg_path.read_text())

    system = _build_system(raw["system"])
    arch = raw.get("architecture", {})
    allow_missing_dmc = bool(raw.get("allow_missing_dmc", True))
    input_e_ref = raw.get("E_ref", "auto")
    resolved_e_ref = resolve_reference_energy(system, input_e_ref, allow_missing_dmc=allow_missing_dmc)

    c_occ, spin, params = setup_closed_shell_system(
        system,
        device=device,
        dtype=dtype,
        E_ref=resolved_e_ref,
        allow_missing_dmc=allow_missing_dmc,
    )

    model = GroundStateWF(
        system,
        c_occ,
        spin,
        params,
        arch_type=arch.get("arch_type", "pinn"),
        pinn_hidden=int(arch.get("pinn_hidden", 64)),
        pinn_layers=int(arch.get("pinn_layers", 2)),
        bf_hidden=int(arch.get("bf_hidden", 32)),
        bf_layers=int(arch.get("bf_layers", 2)),
        use_well_features=bool(arch.get("use_well_features", False)),
        use_well_backflow=bool(arch.get("use_well_backflow", False)),
        use_backflow=bool(arch.get("use_backflow", True)),
    ).to(device=device, dtype=dtype)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, system, params


def evaluate_result_dir(result_dir: Path, device: torch.device, n_samples: int, mh_steps: int, fd_h: float) -> dict:
    dtype = torch.float64
    model, system, params = load_model_from_result_dir(result_dir, device, dtype)

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    x = None
    sigma_fs = adapt_sigma_fs(system.omega, (0.8, 1.3, 2.0))
    mh_scale = 0.25
    acc = 0.0
    for _ in range(5):
        x, acc, mh_scale = mcmc_resample(
            psi_log_fn,
            x,
            n_samples,
            n_elec=system.n_particles,
            dim=system.dim,
            omega=system.omega,
            device=device,
            dtype=dtype,
            system=system,
            sigma_fs=sigma_fs,
            mh_steps=mh_steps,
            mh_step_scale=mh_scale,
            mh_decorrelation=1,
        )

    with torch.no_grad():
        lap = _laplacian_over_psi_fd(psi_log_fn, x, fd_h)
        v_tot, v_conf, v_ee = potential_split(x, system, system.omega)
        e_loc = -0.5 * lap + v_tot
        kinetic = e_loc - v_tot

        centers = torch.tensor([w.center for w in system.wells], device=x.device, dtype=x.dtype)
        d2 = ((x[:, :, None, :] - centers[None, None, :, :]) ** 2).sum(-1)
        assign = d2.argmin(-1)
        occ = [(assign == w).double().mean().item() for w in range(centers.shape[0])]

    return {
        "result_dir": str(result_dir),
        "n_samples": n_samples,
        "mh_steps_eval": mh_steps,
        "acc_rate": float(acc),
        "E_mean_fd": float(e_loc.mean().item()),
        "T_mean_fd": float(kinetic.mean().item()),
        "V_total_mean": float(v_tot.mean().item()),
        "V_conf_mean": float(v_conf.mean().item()),
        "V_ee_mean": float(v_ee.mean().item()),
        "occupancy_fraction": occ,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate energy components for a saved ground-state result dir.")
    parser.add_argument("--result-dir", required=True, help="Path to results/<run_timestamp> directory.")
    parser.add_argument("--device", default="cuda:1", help="Torch device.")
    parser.add_argument("--n-samples", type=int, default=4096, help="Evaluation sample size.")
    parser.add_argument("--mh-steps", type=int, default=20, help="MH steps per refresh for evaluation sampling.")
    parser.add_argument("--fd-h", type=float, default=0.01, help="FD step for kinetic estimate.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    device = torch.device(args.device)
    out = evaluate_result_dir(result_dir, device, args.n_samples, args.mh_steps, args.fd_h)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
