#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from run_ground_state import run_training_from_config


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diag_sweeps"


@dataclass(frozen=True)
class StagePreset:
    stage_a_loss_type: str = "residual"
    stage_a_sampler: str = "stratified"
    stage_a_epochs: int = 800
    stage_a_print_every: int = 100
    stage_b_loss_type: str = "residual"
    stage_b_residual_objective: str = "residual"
    stage_b_sampler: str = "stratified"
    stage_b_non_mcmc_only: bool = True


@dataclass(frozen=True)
class StageAGate:
    min_final_ess: float = 32.0
    min_final_energy: float = 0.0


def _is_n2_one_per_well(base_cfg: dict[str, Any]) -> bool:
    system = base_cfg.get("system", {})
    return (
        system.get("type") == "double_dot"
        and int(system.get("n_left", 0)) == 1
        and int(system.get("n_right", 0)) == 1
    )


def _apply_legacy_n2_singlet_recipe(cfg: dict[str, Any]) -> None:
    architecture = cfg.setdefault("architecture", {})
    architecture["singlet"] = True
    architecture["use_backflow"] = False

    training = cfg.setdefault("training", {})
    training["sampler"] = "stratified"
    training["non_mcmc_only"] = True
    training["sampler_sigma_center"] = 0.20
    training["sampler_sigma_tails"] = 1.00
    training["sampler_sigma_mixed_in"] = 0.25
    training["sampler_sigma_mixed_out"] = 0.70
    training["sampler_shell_radius"] = 1.20
    training["sampler_shell_radius_sigma"] = 0.06
    training["sampler_dimer_pairs"] = 1
    training["sampler_dimer_eps_max"] = 0.06


def _n_particles_from_cfg(base_cfg: dict[str, Any]) -> int:
    """Return total particle count from a system config."""
    system = base_cfg.get("system", {})
    wells = system.get("wells", [])
    if wells:
        return sum(int(w.get("n_particles", 1)) for w in wells)
    # Fallback for double_dot configs
    return int(system.get("n_left", 1)) + int(system.get("n_right", 1))


def _apply_improved_noref_recipe(cfg: dict[str, Any]) -> None:
    """Wider stratified sampler + no backflow for N≥3 no-ref training.

    Mirrors the N=2 legacy singlet recipe geometry (which proved critical for
    recovering entanglement without E_ref) but without the singlet=True flag
    (which is only valid for N=2). The wider sigma values help the sampler
    explore configurations beyond the single-well localized minima.
    """
    n = _n_particles_from_cfg(cfg)
    architecture = cfg.setdefault("architecture", {})
    architecture["use_backflow"] = False
    architecture["multi_ref"] = True

    training = cfg.setdefault("training", {})
    training["sampler"] = "stratified"
    training["non_mcmc_only"] = True
    training["sampler_sigma_center"] = 0.20
    training["sampler_sigma_tails"] = 1.00
    training["sampler_sigma_mixed_in"] = 0.25
    training["sampler_sigma_mixed_out"] = 0.70
    training["sampler_shell_radius"] = 1.20
    training["sampler_shell_radius_sigma"] = 0.06
    # More dimer pairs for larger N to sample inter-particle correlations.
    training["sampler_dimer_pairs"] = max(1, n - 1)
    training["sampler_dimer_eps_max"] = 0.06


def _infer_stage_a_strategy(base_cfg: dict[str, Any]) -> str:
    training = base_cfg.get("training", {})
    if (
        _is_n2_one_per_well(base_cfg)
        and str(training.get("loss_type", "")) == "residual"
        and str(training.get("residual_objective", "")) == "energy_var"
        and training.get("residual_target_energy") is not None
        and float(training.get("alpha_end", 0.0)) > 0.0
    ):
        # The old successful E_ref lane used the dedicated singlet permanent
        # ansatz plus a broader stratified sampler. Keep those inductive biases
        # when removing E_ref guidance instead of falling back to the generic
        # fixed-spin determinant lane, which has consistently low entanglement.
        return "singlet_self_residual"
    return "guided"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _build_stage_a_cfg(
    base_cfg: dict[str, Any],
    *,
    stage_a_epochs: int,
    suffix: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["run_name"] = f"{cfg.get('run_name', 'ground_state')}{suffix}"
    training = cfg.setdefault("training", {})
    training["epochs"] = int(stage_a_epochs)
    # Stage A should keep the guided warm-start encoded in the source config.
    # For the residual_anneal configs this means residual + annealed E_eff under
    # the proven non-MCMC stratified sampler, which is stable for N=2/3/4.
    training["loss_type"] = str(training.get("loss_type", "residual"))
    training["residual_objective"] = str(training.get("residual_objective", "energy_var"))
    training["sampler"] = str(training.get("sampler", "stratified"))
    training["non_mcmc_only"] = True
    training["print_every"] = int(training.get("print_every", 100))
    training["logw_clip_q"] = float(training.get("logw_clip_q", 0.0))
    return cfg


def _build_stage_a_self_residual_cfg(
    base_cfg: dict[str, Any],
    *,
    stage_a_epochs: int,
    suffix: str,
    use_legacy_n2_singlet: bool = False,
    use_improved_recipe: bool = False,
    seed_override: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["run_name"] = f"{cfg.get('run_name', 'ground_state')}{suffix}"
    training = cfg.setdefault("training", {})
    training["epochs"] = int(stage_a_epochs)
    training["loss_type"] = "residual"
    training["residual_objective"] = "residual"
    training["residual_target_energy"] = None
    training["alpha_start"] = 0.0
    training["alpha_end"] = 0.0
    training["sampler"] = "stratified"
    training["non_mcmc_only"] = True
    training["print_every"] = int(training.get("print_every", 100))
    training["logw_clip_q"] = float(training.get("logw_clip_q", 0.0))
    if seed_override is not None:
        training["seed"] = int(seed_override)
    if use_legacy_n2_singlet:
        _apply_legacy_n2_singlet_recipe(cfg)
    elif use_improved_recipe:
        _apply_improved_noref_recipe(cfg)
    return cfg


def _build_stage_b_cfg(
    base_cfg: dict[str, Any],
    *,
    init_result_dir: Path,
    stage_b_epochs: int | None,
    suffix: str,
    use_legacy_n2_singlet: bool = False,
    use_improved_recipe: bool = False,
    seed_override: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["run_name"] = f"{cfg.get('run_name', 'ground_state')}{suffix}"
    cfg["init_from"] = {"result_dir": str(init_result_dir), "strict": False}
    training = cfg.setdefault("training", {})
    if stage_b_epochs is not None:
        training["epochs"] = int(stage_b_epochs)
    # Preserve reinforce from stage A when set — residual keeps O(2Nd) graphs
    # alive for FD Laplacian and OOMs for large N (N≥16).
    if base_cfg.get("training", {}).get("loss_type") == "reinforce":
        training["loss_type"] = "reinforce"
        training["reinforce_clip_width"] = base_cfg["training"].get("reinforce_clip_width", 5.0)
    else:
        training["loss_type"] = "residual"
        training["residual_objective"] = "residual"
        training["residual_target_energy"] = None
        training["alpha_start"] = 0.0
        training["alpha_end"] = 0.0
    training["sampler"] = "stratified"
    training["non_mcmc_only"] = True
    if seed_override is not None:
        training["seed"] = int(seed_override)
    if use_legacy_n2_singlet:
        _apply_legacy_n2_singlet_recipe(cfg)
    elif use_improved_recipe:
        _apply_improved_noref_recipe(cfg)
    return cfg


def _stage_a_gate_status(stage_a_result: dict[str, Any], gate: StageAGate) -> dict[str, Any]:
    final_energy = float(stage_a_result["final_energy"])
    final_ess = float(stage_a_result["final_ess"])

    failures: list[str] = []
    if not (final_energy > float(gate.min_final_energy)):
        failures.append(
            f"final_energy={final_energy:.6f} <= min_final_energy={float(gate.min_final_energy):.6f}"
        )
    if not (final_ess >= float(gate.min_final_ess)):
        failures.append(f"final_ess={final_ess:.3f} < min_final_ess={float(gate.min_final_ess):.3f}")

    return {
        "passed": len(failures) == 0,
        "min_final_ess": float(gate.min_final_ess),
        "min_final_energy": float(gate.min_final_energy),
        "failures": failures,
    }


def run_two_stage_from_config(
    config_path: Path,
    *,
    stage_a_epochs: int,
    stage_b_epochs: int | None,
    stage_a_gate: StageAGate,
    stage_a_strategy: str = "auto",
    stage_a_suffix: str = "__stageA_energywarm",
    stage_b_suffix: str = "__stageB_noref",
    seed_override: int | None = None,
) -> dict[str, Any]:
    base_cfg = _load_yaml(config_path)
    if seed_override is not None:
        base_cfg.setdefault("training", {})["seed"] = int(seed_override)
    resolved_strategy = (
        _infer_stage_a_strategy(base_cfg) if stage_a_strategy == "auto" else stage_a_strategy
    )
    if resolved_strategy == "guided":
        stage_a_cfg = _build_stage_a_cfg(base_cfg, stage_a_epochs=stage_a_epochs, suffix=stage_a_suffix)
    elif resolved_strategy == "self_residual":
        stage_a_cfg = _build_stage_a_self_residual_cfg(
            base_cfg,
            stage_a_epochs=stage_a_epochs,
            suffix="__stageA_self_residual",
        )
    elif resolved_strategy == "singlet_self_residual":
        stage_a_cfg = _build_stage_a_self_residual_cfg(
            base_cfg,
            stage_a_epochs=stage_a_epochs,
            suffix="__stageA_singlet_self_residual",
            use_legacy_n2_singlet=True,
        )
    elif resolved_strategy == "improved_self_residual":
        stage_a_cfg = _build_stage_a_self_residual_cfg(
            base_cfg,
            stage_a_epochs=stage_a_epochs,
            suffix="__stageA_improved_self_residual",
            use_improved_recipe=True,
        )
    else:
        raise ValueError(
            "Unknown stage_a_strategy "
            f"'{stage_a_strategy}'. Expected one of: "
            "auto, guided, self_residual, singlet_self_residual, improved_self_residual."
        )
    stage_a_dir, stage_a_result = run_training_from_config(stage_a_cfg, config_source=str(config_path))
    gate_status = _stage_a_gate_status(stage_a_result, stage_a_gate)

    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "stage_a_strategy": resolved_strategy,
        "stage_a_gate": gate_status,
        "stage_a": {
            "result_dir": str(stage_a_dir),
            "run_name": stage_a_cfg["run_name"],
            "training": stage_a_cfg["training"],
            "result": {
                "final_energy": stage_a_result["final_energy"],
                "final_loss": stage_a_result["final_loss"],
                "final_energy_var": stage_a_result["final_energy_var"],
                "final_ess": stage_a_result["final_ess"],
            },
        },
    }

    if not gate_status["passed"]:
        summary["stage_b"] = None
        return summary

    stage_b_cfg = _build_stage_b_cfg(
        base_cfg,
        init_result_dir=stage_a_dir,
        stage_b_epochs=stage_b_epochs,
        suffix=stage_b_suffix,
        use_legacy_n2_singlet=(resolved_strategy == "singlet_self_residual"),
        use_improved_recipe=(resolved_strategy == "improved_self_residual"),
    )
    stage_b_dir, stage_b_result = run_training_from_config(stage_b_cfg, config_source=str(config_path))
    summary["stage_b"] = {
        "result_dir": str(stage_b_dir),
        "run_name": stage_b_cfg["run_name"],
        "training": stage_b_cfg["training"],
        "result": {
            "final_energy": stage_b_result["final_energy"],
            "final_loss": stage_b_result["final_loss"],
            "final_energy_var": stage_b_result["final_energy_var"],
            "final_ess": stage_b_result["final_ess"],
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a two-stage ground-state reproduction: guided warm start, then pure no-reference variance refinement."
    )
    parser.add_argument("--config", required=True, help="Base YAML config to reproduce without E_ref guidance.")
    parser.add_argument(
        "--stage-a-epochs",
        type=int,
        default=800,
        help="Epochs for the stage-A guided warm start.",
    )
    parser.add_argument(
        "--stage-b-epochs",
        type=int,
        default=None,
        help="Optional override for stage-B epochs. Defaults to the base config value.",
    )
    parser.add_argument(
        "--stage-a-min-ess",
        type=float,
        default=32.0,
        help="Skip Stage B if Stage A final ESS is below this threshold.",
    )
    parser.add_argument(
        "--stage-a-min-energy",
        type=float,
        default=0.0,
        help="Skip Stage B if Stage A final energy is at or below this threshold.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional explicit JSON path for the run summary.",
    )
    parser.add_argument(
        "--stage-a-strategy",
        default="auto",
        choices=["auto", "guided", "self_residual", "singlet_self_residual", "improved_self_residual"],
        help=(
            "Stage-A warm-start strategy. 'auto' selects singlet_self_residual for N=2 "
            "double_dot configs and guided elsewhere. 'improved_self_residual' applies the "
            "wider stratified sampler geometry (no E_ref) for N>=3 without backflow."
        ),
    )
    parser.add_argument(
        "--seed-override",
        type=int,
        default=None,
        help="Override the seed in the base config for multi-seed sweeps.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    summary = run_two_stage_from_config(
        config_path,
        stage_a_epochs=int(args.stage_a_epochs),
        stage_b_epochs=args.stage_b_epochs,
        stage_a_gate=StageAGate(
            min_final_ess=float(args.stage_a_min_ess),
            min_final_energy=float(args.stage_a_min_energy),
        ),
        stage_a_strategy=str(args.stage_a_strategy),
        seed_override=args.seed_override,
    )

    if args.summary_json:
        out_path = Path(args.summary_json).expanduser().resolve()
    else:
        seed_tag = f"_seed{args.seed_override}" if args.seed_override is not None else ""
        strategy_tag = f"__{args.stage_a_strategy}" if args.stage_a_strategy != "auto" else ""
        out_path = (
            DEFAULT_OUTPUT_DIR
            / f"{config_path.stem}{seed_tag}{strategy_tag}__two_stage_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved two-stage summary to {out_path}")


if __name__ == "__main__":
    main()
