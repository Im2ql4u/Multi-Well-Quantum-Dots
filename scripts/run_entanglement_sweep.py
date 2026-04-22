#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOG = logging.getLogger("run_entanglement_sweep")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diag_sweeps"


@dataclass(frozen=True)
class SweepJob:
    result_dir: Path
    npts: int
    partition: str
    output_path: Path


def _parse_csv_ints(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_partition_values(text: str) -> list[str]:
    values = [token.strip() for token in text.split(";") if token.strip()]
    if not values:
        raise ValueError("Expected at least one partition specifier.")
    return values


def _partition_label(partition: str) -> str:
    if partition == "auto":
        return "auto"
    return partition.replace(",", "-")


def build_sweep_jobs(
    result_dir: Path,
    npts_values: list[int],
    partitions: list[str],
    output_dir: Path,
    output_prefix: str | None,
) -> list[SweepJob]:
    prefix = output_prefix or result_dir.name
    jobs: list[SweepJob] = []
    for partition in partitions:
        label = _partition_label(partition)
        for npts in npts_values:
            out_name = f"{prefix}__npts{npts}__part_{label}.json"
            jobs.append(
                SweepJob(
                    result_dir=result_dir,
                    npts=npts,
                    partition=partition,
                    output_path=output_dir / out_name,
                )
            )
    return jobs
def _run_measurement(job: SweepJob, device: str, batch_size: int, particle_grid: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/measure_entanglement.py",
        "--result-dir",
        str(job.result_dir),
        "--npts",
        str(job.npts),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--particle-grid",
        particle_grid,
        "--out",
        str(job.output_path),
    ]
    if job.partition != "auto":
        command.extend(["--partition-particles", job.partition])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    LOG.info(
        "Running measurement: result_dir=%s npts=%s partition=%s",
        job.result_dir.name,
        job.npts,
        job.partition,
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    with job.output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_sweep"] = {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "npts": job.npts,
        "partition": job.partition,
    }
    return payload


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    entanglement = payload["entanglement"]
    summary = {
        "npts": payload["npts"],
        "n_grid_per_particle": payload["n_grid_per_particle"],
        "partition": payload["partition"],
        "von_neumann_entropy": entanglement["von_neumann_entropy"],
        "negativity": entanglement["negativity"],
        "log_negativity": entanglement["log_negativity"],
        "purity": entanglement["purity"],
        "effective_schmidt_rank": entanglement["effective_schmidt_rank"],
        "norm2_before_normalisation": entanglement["norm2_before_normalisation"],
        "norm_tensor_squared": entanglement["norm_tensor_squared"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run entanglement measurement sweeps for saved checkpoints.")
    parser.add_argument("--result-dir", required=True, help="Ground-state result directory to measure.")
    parser.add_argument(
        "--npts-values",
        required=True,
        help="Comma-separated npts values, for example '4,5,6'.",
    )
    parser.add_argument(
        "--partitions",
        default="auto",
        help="Semicolon-separated partition specs, for example 'auto;0;1;0,1'.",
    )
    parser.add_argument("--device", required=True, help="Torch device passed through to measure_entanglement.py.")
    parser.add_argument("--batch-size", type=int, default=512, help="Measurement batch size.")
    parser.add_argument(
        "--particle-grid",
        default="hermite_local",
        choices=["hermite_local", "legendre_local"],
        help="Per-particle grid family for N>=3 measurements.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-run JSON outputs and the sweep summary.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional filename prefix. Defaults to the result directory name.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional explicit path for the aggregated summary JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    result_dir = Path(args.result_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    npts_values = _parse_csv_ints(args.npts_values)
    partitions = _parse_partition_values(args.partitions)
    jobs = build_sweep_jobs(result_dir, npts_values, partitions, output_dir, args.output_prefix)

    summaries: list[dict[str, Any]] = []
    for job in jobs:
        payload = _run_measurement(
            job,
            device=args.device,
            batch_size=args.batch_size,
            particle_grid=args.particle_grid,
        )
        entry = {
            "output_json": str(job.output_path.relative_to(REPO_ROOT)),
            **_summarize_payload(payload),
        }
        summaries.append(entry)

    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_dir / (
        f"{args.output_prefix or result_dir.name}__sweep_summary.json"
    )
    summary_payload = {
        "source": str(result_dir),
        "device": args.device,
        "batch_size": args.batch_size,
        "particle_grid": args.particle_grid,
        "npts_values": npts_values,
        "partitions": partitions,
        "runs": summaries,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    LOG.info("Saved sweep summary to %s", summary_path)


if __name__ == "__main__":
    main()