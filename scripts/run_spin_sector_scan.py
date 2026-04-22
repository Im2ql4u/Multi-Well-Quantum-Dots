#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from run_ground_state import build_system_from_config, run_training_from_config
from wavefunction import assess_magnetic_response_capability


LOG = logging.getLogger("run_spin_sector_scan")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diag_sweeps"


@dataclass(frozen=True)
class SpinSectorJob:
    n_up: int
    n_down: int
    pattern: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"{self.n_up}up_{self.n_down}down"


def parse_csv_ints(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def build_spin_sector_jobs(
    n_particles: int,
    n_up_values: list[int] | None,
) -> list[SpinSectorJob]:
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")
    candidate_values = list(range(n_particles + 1)) if n_up_values is None else [int(v) for v in n_up_values]
    jobs: list[SpinSectorJob] = []
    for n_up in candidate_values:
        if n_up < 0 or n_up > n_particles:
            raise ValueError(f"n_up must lie in [0, {n_particles}], got {n_up}.")
        n_down = n_particles - n_up
        jobs.append(
            SpinSectorJob(
                n_up=n_up,
                n_down=n_down,
                pattern=tuple([0] * n_up + [1] * n_down),
            )
        )
    return jobs


def build_sector_config(base_cfg: dict[str, Any], job: SpinSectorJob) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    base_run_name = str(cfg.get("run_name", "spin_sector_scan"))
    cfg["run_name"] = f"{base_run_name}__{job.label}"
    cfg["spin"] = {
        "pattern": list(job.pattern),
    }

    system = build_system_from_config(cfg)
    spin_tensor = torch.tensor(job.pattern, dtype=torch.long)
    magnetic_assessment = assess_magnetic_response_capability(system, spin_tensor)
    training_cfg = cfg.get("training", {})
    residual_target_energy = training_cfg.get("residual_target_energy")
    if residual_target_energy is not None and training_cfg.get("loss_type") == "residual":
        residual_objective = str(training_cfg.get("residual_objective", "residual"))
        if residual_objective in {"energy", "energy_var"}:
            adjusted_target = float(residual_target_energy) + float(
                magnetic_assessment["constant_energy_shift"]
            )
            training_cfg["residual_target_energy"] = adjusted_target
            cfg["spin_sector_scan"] = {
                "base_residual_target_energy": float(residual_target_energy),
                "sector_residual_target_energy": adjusted_target,
                "constant_energy_shift": float(magnetic_assessment["constant_energy_shift"]),
            }

    magnetic_section = cfg.setdefault("magnetic_assessment", {})
    if not isinstance(magnetic_section, dict):
        raise ValueError("magnetic_assessment must be a mapping when provided.")
    magnetic_section["mode"] = "off"
    return cfg


def summarise_sector_run(
    job: SpinSectorJob,
    out_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sector": {
            "label": job.label,
            "n_up": job.n_up,
            "n_down": job.n_down,
            "pattern": list(job.pattern),
        },
        "out_dir": str(out_dir),
        "final_energy": float(result["final_energy"]),
        "final_loss": float(result["final_loss"]),
        "final_ess": float(result["final_ess"]),
        "magnetic_assessment": result.get("magnetic_assessment", {}),
        "spin_configuration": result.get("spin_configuration", {}),
        "scan_adjustments": result.get("scan_adjustments", {}),
    }


def run_sector_scan(
    base_cfg: dict[str, Any],
    jobs: list[SpinSectorJob],
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for job in jobs:
        cfg = build_sector_config(base_cfg, job)
        LOG.info("Training spin sector %s", job.label)
        out_dir, result = run_training_from_config(cfg)
        runs.append(summarise_sector_run(job, out_dir, result))

    best_run = min(runs, key=lambda row: float(row["final_energy"]))
    return {
        "sector_competition_required": True,
        "notes": [
            "Each fixed-spin run still sees a constant Zeeman offset within that sector.",
            "Nontrivial uniform-B response in the present generalized lane comes from comparing sectors and selecting the lowest-energy one.",
        ],
        "runs": runs,
        "best_run": best_run,
    }


def default_summary_path(output_dir: Path, config_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{config_path.stem}_spin_sector_scan_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a generalized config across multiple collinear spin sectors and summarize the lowest-energy sector."
    )
    parser.add_argument("--config", required=True, help="Base YAML config for the generalized run.")
    parser.add_argument(
        "--n-up-values",
        default=None,
        help="Optional comma-separated n_up values to scan. Defaults to all sectors 0..N.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional explicit JSON path for the sector-scan summary.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the sector-scan summary JSON.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    cfg_path = Path(args.config).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    system = build_system_from_config(base_cfg)
    if abs(float(system.B_magnitude)) <= 0.0:
        raise ValueError("Spin-sector scan is intended for magnetic runs; system.B_magnitude must be non-zero.")

    n_up_values = parse_csv_ints(args.n_up_values) if args.n_up_values else None
    jobs = build_spin_sector_jobs(system.n_particles, n_up_values)
    summary = run_sector_scan(base_cfg, jobs)
    summary["config"] = {
        "base_config": str(cfg_path),
        "n_particles": int(system.n_particles),
        "n_up_values": [job.n_up for job in jobs],
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else default_summary_path(output_dir, cfg_path)
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOG.info("Saved spin-sector scan summary to %s", summary_path)


if __name__ == "__main__":
    main()