#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from run_ground_state import run_training_from_config

try:
    from scripts.run_two_stage_ground_state import StageAGate, _build_stage_b_cfg, _stage_a_gate_status
except ModuleNotFoundError:
    from run_two_stage_ground_state import StageAGate, _build_stage_b_cfg, _stage_a_gate_status


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diag_sweeps"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _stage_seed(base_cfg: dict[str, Any]) -> int | None:
    training = base_cfg.get("training", {})
    seed = training.get("seed")
    return None if seed is None else int(seed)


def _base_stage_a_cfg(
    base_cfg: dict[str, Any],
    *,
    epochs: int,
    suffix: str,
    init_result_dir: Path | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["run_name"] = f"{cfg.get('run_name', 'ground_state')}{suffix}"
    if init_result_dir is not None:
        cfg["init_from"] = {"result_dir": str(init_result_dir), "strict": False}
    training = cfg.setdefault("training", {})
    training["epochs"] = int(epochs)
    training["loss_type"] = "residual"
    training["sampler"] = "stratified"
    training["non_mcmc_only"] = True
    training["print_every"] = int(training.get("print_every", 100))
    training["logw_clip_q"] = float(training.get("logw_clip_q", 0.0))
    if seed_override is not None:
        training["seed"] = int(seed_override)
    return cfg


def _build_stage_a_self_residual_cfg(
    base_cfg: dict[str, Any],
    *,
    epochs: int,
    suffix: str,
    init_result_dir: Path | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    cfg = _base_stage_a_cfg(
        base_cfg,
        epochs=epochs,
        suffix=suffix,
        init_result_dir=init_result_dir,
        seed_override=seed_override,
    )
    training = cfg["training"]
    training["residual_objective"] = "residual"
    training["residual_target_energy"] = None
    training["alpha_start"] = 0.0
    training["alpha_end"] = 0.0
    return cfg


def _build_stage_a_bootstrap_target_cfg(
    base_cfg: dict[str, Any],
    *,
    epochs: int,
    suffix: str,
    bootstrap_target_energy: float,
    init_result_dir: Path | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    cfg = _base_stage_a_cfg(
        base_cfg,
        epochs=epochs,
        suffix=suffix,
        init_result_dir=init_result_dir,
        seed_override=seed_override,
    )
    training = cfg["training"]
    training["residual_objective"] = "energy_var"
    training["residual_target_energy"] = float(bootstrap_target_energy)
    training["alpha_start"] = 0.0
    training["alpha_end"] = 1.0
    return cfg


def _run_stage(cfg: dict[str, Any], *, config_path: Path) -> tuple[Path, dict[str, Any]]:
    return run_training_from_config(cfg, config_source=str(config_path))


def _serialize_stage(cfg: dict[str, Any], result_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_dir": str(result_dir),
        "run_name": cfg["run_name"],
        "training": cfg["training"],
        "result": {
            "final_energy": result["final_energy"],
            "final_loss": result["final_loss"],
            "final_energy_var": result["final_energy_var"],
            "final_ess": result["final_ess"],
        },
    }


def _candidate_rank_key(candidate: dict[str, Any]) -> tuple[float, float]:
    res = candidate["result"]
    return (float(res["final_energy_var"]), float(res["final_energy"]))


def run_research_two_stage_from_config(
    config_path: Path,
    *,
    strategy: str,
    stage_a_epochs: int,
    stage_b_epochs: int | None,
    stage_a_gate: StageAGate,
    bootstrap_epochs: int,
    multistart_epochs: int,
    multistart_restarts: int,
    multistart_seed_stride: int,
    stage_b_suffix: str = "__stageB_noref",
) -> dict[str, Any]:
    base_cfg = _load_yaml(config_path)
    base_seed = _stage_seed(base_cfg)
    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "strategy": strategy,
    }

    if strategy == "self_residual":
        stage_a_cfg = _build_stage_a_self_residual_cfg(
            base_cfg,
            epochs=stage_a_epochs,
            suffix="__stageA_self_residual",
        )
        stage_a_dir, stage_a_result = _run_stage(stage_a_cfg, config_path=config_path)
        gate_status = _stage_a_gate_status(stage_a_result, stage_a_gate)
        summary["stage_a"] = _serialize_stage(stage_a_cfg, stage_a_dir, stage_a_result)
        summary["stage_a_gate"] = gate_status
        init_result_dir = stage_a_dir

    elif strategy == "bootstrap_target":
        if bootstrap_epochs <= 0 or bootstrap_epochs >= stage_a_epochs:
            raise ValueError("bootstrap_epochs must be in [1, stage_a_epochs-1].")
        pilot_cfg = _build_stage_a_self_residual_cfg(
            base_cfg,
            epochs=bootstrap_epochs,
            suffix="__stageA_bootstrap_pilot",
        )
        pilot_dir, pilot_result = _run_stage(pilot_cfg, config_path=config_path)
        pilot_gate = _stage_a_gate_status(pilot_result, stage_a_gate)
        target_energy = float(pilot_result["final_energy"])
        main_cfg = _build_stage_a_bootstrap_target_cfg(
            base_cfg,
            epochs=stage_a_epochs - bootstrap_epochs,
            suffix="__stageA_bootstrap_target",
            bootstrap_target_energy=target_energy,
            init_result_dir=pilot_dir,
        )
        main_dir, main_result = _run_stage(main_cfg, config_path=config_path)
        gate_status = _stage_a_gate_status(main_result, stage_a_gate)
        summary["bootstrap_pilot"] = _serialize_stage(pilot_cfg, pilot_dir, pilot_result)
        summary["bootstrap_pilot_gate"] = pilot_gate
        summary["bootstrap_target_energy"] = target_energy
        summary["stage_a"] = _serialize_stage(main_cfg, main_dir, main_result)
        summary["stage_a_gate"] = gate_status
        init_result_dir = main_dir

    elif strategy == "multistart_self":
        if multistart_epochs <= 0 or multistart_epochs >= stage_a_epochs:
            raise ValueError("multistart_epochs must be in [1, stage_a_epochs-1].")
        if multistart_restarts < 2:
            raise ValueError("multistart_restarts must be at least 2.")
        candidates: list[dict[str, Any]] = []
        for idx in range(multistart_restarts):
            seed_override = None if base_seed is None else int(base_seed + idx * multistart_seed_stride)
            cand_cfg = _build_stage_a_self_residual_cfg(
                base_cfg,
                epochs=multistart_epochs,
                suffix=f"__stageA_mstart_{idx}",
                seed_override=seed_override,
            )
            cand_dir, cand_result = _run_stage(cand_cfg, config_path=config_path)
            cand_gate = _stage_a_gate_status(cand_result, stage_a_gate)
            candidates.append(
                {
                    "index": idx,
                    "seed": seed_override,
                    "gate": cand_gate,
                    **_serialize_stage(cand_cfg, cand_dir, cand_result),
                }
            )

        passed = [candidate for candidate in candidates if candidate["gate"]["passed"]]
        if not passed:
            summary["multistart_candidates"] = candidates
            summary["stage_a_gate"] = {
                "passed": False,
                "min_final_ess": float(stage_a_gate.min_final_ess),
                "min_final_energy": float(stage_a_gate.min_final_energy),
                "failures": ["no multistart candidate passed the Stage-A gate"],
            }
            summary["stage_a"] = None
            summary["stage_b"] = None
            return summary

        selected = min(passed, key=_candidate_rank_key)
        refine_cfg = _build_stage_a_self_residual_cfg(
            base_cfg,
            epochs=stage_a_epochs - multistart_epochs,
            suffix="__stageA_mstart_refine",
            init_result_dir=Path(selected["result_dir"]),
        )
        refine_dir, refine_result = _run_stage(refine_cfg, config_path=config_path)
        gate_status = _stage_a_gate_status(refine_result, stage_a_gate)
        summary["multistart_candidates"] = candidates
        summary["multistart_selected"] = {
            "index": selected["index"],
            "seed": selected["seed"],
            "result_dir": selected["result_dir"],
            "selection_metric": {
                "final_energy_var": selected["result"]["final_energy_var"],
                "final_energy": selected["result"]["final_energy"],
            },
        }
        summary["stage_a"] = _serialize_stage(refine_cfg, refine_dir, refine_result)
        summary["stage_a_gate"] = gate_status
        init_result_dir = refine_dir

    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    if not summary["stage_a_gate"]["passed"]:
        summary["stage_b"] = None
        return summary

    stage_b_cfg = _build_stage_b_cfg(
        base_cfg,
        init_result_dir=init_result_dir,
        stage_b_epochs=stage_b_epochs,
        suffix=stage_b_suffix,
    )
    stage_b_dir, stage_b_result = _run_stage(stage_b_cfg, config_path=config_path)
    summary["stage_b"] = _serialize_stage(stage_b_cfg, stage_b_dir, stage_b_result)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N=2 research variants for no-reference Stage-A branch selection."
    )
    parser.add_argument("--config", required=True, help="Base YAML config.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["self_residual", "bootstrap_target", "multistart_self"],
        help="Stage-A research strategy to run.",
    )
    parser.add_argument("--stage-a-epochs", type=int, default=800)
    parser.add_argument("--stage-b-epochs", type=int, default=None)
    parser.add_argument("--stage-a-min-ess", type=float, default=32.0)
    parser.add_argument("--stage-a-min-energy", type=float, default=0.0)
    parser.add_argument("--bootstrap-epochs", type=int, default=200)
    parser.add_argument("--multistart-epochs", type=int, default=200)
    parser.add_argument("--multistart-restarts", type=int, default=4)
    parser.add_argument("--multistart-seed-stride", type=int, default=10000)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    summary = run_research_two_stage_from_config(
        config_path,
        strategy=args.strategy,
        stage_a_epochs=int(args.stage_a_epochs),
        stage_b_epochs=args.stage_b_epochs,
        stage_a_gate=StageAGate(
            min_final_ess=float(args.stage_a_min_ess),
            min_final_energy=float(args.stage_a_min_energy),
        ),
        bootstrap_epochs=int(args.bootstrap_epochs),
        multistart_epochs=int(args.multistart_epochs),
        multistart_restarts=int(args.multistart_restarts),
        multistart_seed_stride=int(args.multistart_seed_stride),
    )

    out_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else DEFAULT_OUTPUT_DIR
        / f"{config_path.stem}__{args.strategy}__summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved research summary to {out_path}")


if __name__ == "__main__":
    main()
