#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from characterize_quench import run_characterization


LOG = logging.getLogger("run_magnetic_reference_sweep")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diag_sweeps"


@dataclass(frozen=True)
class SweepJob:
    sep: float
    b_pre: float
    b_post: float


def parse_csv_floats(text: str) -> list[float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def build_sweep_jobs(separations: list[float], b_pre: float, b_post_values: list[float]) -> list[SweepJob]:
    jobs: list[SweepJob] = []
    for sep in separations:
        for b_post in b_post_values:
            jobs.append(SweepJob(sep=sep, b_pre=b_pre, b_post=b_post))
    return jobs


def summarise_characterization(result: dict[str, Any]) -> dict[str, Any]:
    pre = result["pre"]
    post = result["post"]
    return {
        "sep": result["sep"],
        "B_pre": result["B_pre"],
        "B_post": result["B_post"],
        "pre_energy": pre["E0"],
        "post_energy": post["E0"],
        "pre_gap": pre["gap"],
        "post_gap": post["gap"],
        "pre_spin": pre["dominant_spin"],
        "post_spin": post["dominant_spin"],
        "spin_flip": pre["dominant_spin"] != post["dominant_spin"],
        "pre_entropy": pre["entanglement"]["entropy"],
        "post_entropy": post["entanglement"]["entropy"],
        "pre_negativity": pre["partial_transpose"]["negativity"],
        "post_negativity": post["partial_transpose"]["negativity"],
    }


def default_summary_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f"{prefix}_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a shared-model exact-diag magnetic reference sweep.")
    parser.add_argument("--separations", required=True, help="Comma-separated well separations, for example '2,4,8'.")
    parser.add_argument("--B-pre", type=float, default=0.0, help="Initial magnetic field.")
    parser.add_argument("--B-post-values", required=True, help="Comma-separated post-quench B values.")
    parser.add_argument("--omega", type=float, default=1.0, help="Confinement frequency.")
    parser.add_argument("--kappa", type=float, default=0.7, help="Coulomb strength.")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Soft Coulomb epsilon.")
    parser.add_argument("--nx", type=int, default=20, help="DVR points along x.")
    parser.add_argument("--ny", type=int, default=20, help="DVR points along y.")
    parser.add_argument("--n-sp-states", type=int, default=40, help="Single-particle basis size.")
    parser.add_argument("--n-ci-compute", type=int, default=200, help="CI truncation size.")
    parser.add_argument("--output-prefix", default="magnetic_reference_sweep", help="Output file prefix.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for raw and summary JSON outputs.")
    parser.add_argument("--summary-json", default=None, help="Optional explicit summary JSON path.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    separations = parse_csv_floats(args.separations)
    b_post_values = parse_csv_floats(args.B_post_values)
    jobs = build_sweep_jobs(separations, b_pre=args.B_pre, b_post_values=b_post_values)

    raw_results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for job in jobs:
        LOG.info("Running magnetic reference: sep=%.3f B_pre=%.3f B_post=%.3f", job.sep, job.b_pre, job.b_post)
        result = run_characterization(
            sep=job.sep,
            B_pre=job.b_pre,
            B_post=job.b_post,
            omega=args.omega,
            kappa=args.kappa,
            epsilon=args.epsilon,
            nx=args.nx,
            ny=args.ny,
            n_sp_states=args.n_sp_states,
            n_ci_compute=args.n_ci_compute,
        )
        raw_results.append(result)
        summaries.append(summarise_characterization(result))

    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else default_summary_path(output_dir, args.output_prefix)
    payload = {
        "sweep_config": {
            "separations": separations,
            "B_pre": args.B_pre,
            "B_post_values": b_post_values,
            "omega": args.omega,
            "kappa": args.kappa,
            "epsilon": args.epsilon,
            "nx": args.nx,
            "ny": args.ny,
            "n_sp_states": args.n_sp_states,
            "n_ci_compute": args.n_ci_compute,
        },
        "runs": summaries,
        "raw_results": raw_results,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOG.info("Saved sweep summary to %s", summary_path)


if __name__ == "__main__":
    main()