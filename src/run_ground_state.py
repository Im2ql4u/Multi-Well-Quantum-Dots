from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml

from config import SystemConfig, WellSpec
from training.vmc_colloc import GroundStateTrainingConfig, train_ground_state
from wavefunction import GroundStateWF, resolve_reference_energy, setup_closed_shell_system

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def _build_system(system_cfg: dict) -> SystemConfig:
    import dataclasses

    _replace = dataclasses.replace
    kind = system_cfg.get("type", "single_dot")
    coulomb = system_cfg.get("coulomb", True)
    smooth_t = float(system_cfg.get("smooth_T", 0.2))
    b_magnitude = float(system_cfg.get("B_magnitude", 0.0))
    b_direction = tuple(float(v) for v in system_cfg.get("B_direction", (0.0, 0.0, 1.0)))
    g_factor = float(system_cfg.get("g_factor", 2.0))
    mu_b = float(system_cfg.get("mu_B", 1.0))
    zeeman_electron1_only = bool(system_cfg.get("zeeman_electron1_only", False))
    zeeman_particle_indices_cfg = system_cfg.get("zeeman_particle_indices", None)
    if zeeman_particle_indices_cfg is None:
        zeeman_particle_indices: tuple[int, ...] | None = None
    else:
        zeeman_particle_indices = tuple(int(idx) for idx in zeeman_particle_indices_cfg)
    if kind == "single_dot":
        sys = SystemConfig.single_dot(
            N=int(system_cfg["n_particles"]),
            omega=float(system_cfg["omega"]),
            dim=int(system_cfg.get("dim", 2)),
        )
    elif kind == "double_dot":
        sys = SystemConfig.double_dot(
            N_L=int(system_cfg["n_left"]),
            N_R=int(system_cfg["n_right"]),
            sep=float(system_cfg["separation"]),
            omega=float(system_cfg["omega"]),
            dim=int(system_cfg.get("dim", 2)),
        )
    elif kind == "custom":
        wells = [
            WellSpec(
                center=tuple(float(c) for c in w["center"]),
                omega=float(w["omega"]),
                n_particles=int(w["n_particles"]),
            )
            for w in system_cfg["wells"]
        ]
        sys = SystemConfig.custom(wells=wells, dim=int(system_cfg.get("dim", 2)))
    else:
        raise ValueError(f"Unknown system type '{kind}'.")
    if not coulomb:
        sys = _replace(sys, coulomb=False)
    sys = _replace(
        sys,
        smooth_T=smooth_t,
        B_magnitude=b_magnitude,
        B_direction=b_direction,
        g_factor=g_factor,
        mu_B=mu_b,
        zeeman_electron1_only=zeeman_electron1_only,
        zeeman_particle_indices=zeeman_particle_indices,
    )
    return sys


def _check_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    print(f"Device: {device}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is not available.")
        props = torch.cuda.get_device_properties(device)
        print(f"  {props.name} | {props.total_memory / 1e9:.1f} GB")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train generalized ground-state wavefunctions with collocation only."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)

    run_name = raw_cfg.get("run_name", cfg_path.stem)
    system = _build_system(raw_cfg["system"])
    train_cfg = GroundStateTrainingConfig(**raw_cfg["training"])
    device = _check_device(train_cfg.device)

    arch_cfg = raw_cfg.get("architecture", {})
    allow_missing_dmc = bool(raw_cfg.get("allow_missing_dmc", True))
    input_E_ref = raw_cfg.get("E_ref", "auto")
    resolved_E_ref = resolve_reference_energy(
        system, input_E_ref, allow_missing_dmc=allow_missing_dmc
    )

    (C_occ, spin, params) = setup_closed_shell_system(
        system,
        device=str(device),
        dtype=train_cfg.torch_dtype,
        E_ref=resolved_E_ref,
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

    result = train_ground_state(model, system, params, train_cfg)

    result["reference_energy"] = {
        "input": input_E_ref,
        "resolved": resolved_E_ref,
        "allow_missing_dmc": allow_missing_dmc,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(raw_cfg, f, sort_keys=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
