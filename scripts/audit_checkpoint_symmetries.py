"""Retroactively validate quantum-number consistency on existing checkpoints.

Walks a results directory tree, reads each ``result.json`` + ``config.yaml``,
and runs the same structural checks that ``assert_quantum_numbers_consistent``
applies post-training. Emits a CSV summary plus a pass/fail count.

This is the Phase 0.1 anchor's *retrospective* audit: everything trained before
the symmetry assertion was wired in is run through the check here so we know
whether any existing artefact silently violated its own quantum numbers.

Usage
-----
::

    PYTHONPATH=src python3.11 scripts/audit_checkpoint_symmetries.py \
        --results-root results \
        --out-csv reports/audit_symmetries_2026-04-28.csv

The audit refuses to retry checkpoints that have a ``symmetry_check`` block
(written automatically by post-Phase-0.1 runs).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.symmetry_asserts import (  # noqa: E402
    SymmetryViolationError,
    assert_quantum_numbers_consistent,
)

LOGGER = logging.getLogger("audit_checkpoint_symmetries")


@dataclass
class _SystemStub:
    """Minimal duck-type for ``assert_quantum_numbers_consistent``."""

    n_particles: int


def _system_n_particles_from_config(cfg: dict[str, Any]) -> int:
    sys_cfg = cfg.get("system", {})
    if "n_particles" in sys_cfg:
        return int(sys_cfg["n_particles"])
    wells = sys_cfg.get("wells", [])
    return int(sum(int(well.get("n_particles", 0)) for well in wells))


def _params_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    spin_cfg = result.get("spin_configuration")
    if not isinstance(spin_cfg, dict):
        return None
    n_up = spin_cfg.get("n_up")
    n_down = spin_cfg.get("n_down")
    pattern = spin_cfg.get("pattern")
    if n_up is None or n_down is None or pattern is None:
        return None
    return {
        "n_up": int(n_up),
        "n_down": int(n_down),
        "n_particles": int(n_up) + int(n_down),
        "spin_pattern": [int(v) for v in pattern],
    }


def _audit_one(result_path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(result_path.parent),
        "status": "skipped",
        "reason": "",
        "n_up": "",
        "n_down": "",
        "expected_Sz": "",
        "n_particles": "",
    }
    try:
        result = json.loads(result_path.read_text())
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["reason"] = f"result.json parse failed: {exc}"
        return record

    if "symmetry_check" in result and result["symmetry_check"].get("checks"):
        # Already validated at training time.
        record["status"] = "live"
        record["reason"] = "symmetry_check present (run post-Phase-0.1)"
        sc = result["symmetry_check"]
        record["n_up"] = sc.get("n_up", "")
        record["n_down"] = sc.get("n_down", "")
        record["expected_Sz"] = sc.get("expected_Sz", "")
        record["n_particles"] = sc.get("n_particles_system", "")
        return record

    config_path = result_path.parent / "config.yaml"
    if not config_path.exists():
        record["status"] = "skipped"
        record["reason"] = "missing config.yaml"
        return record

    try:
        cfg = yaml.safe_load(config_path.read_text())
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["reason"] = f"config.yaml parse failed: {exc}"
        return record

    n_particles = _system_n_particles_from_config(cfg)
    if n_particles <= 0:
        record["status"] = "skipped"
        record["reason"] = "n_particles unknown from config.yaml"
        return record

    params = _params_from_result(result)
    if params is None:
        record["status"] = "skipped"
        record["reason"] = "spin_configuration absent or incomplete"
        return record

    system_stub = _SystemStub(n_particles=n_particles)
    try:
        diag = assert_quantum_numbers_consistent(
            model=None,
            system=system_stub,
            params=params,
            context=f"audit:{result_path.parent.name}",
        )
    except SymmetryViolationError as exc:
        record["status"] = "fail"
        record["reason"] = str(exc)
        record["n_up"] = params["n_up"]
        record["n_down"] = params["n_down"]
        record["n_particles"] = n_particles
        return record

    record["status"] = "pass"
    record["n_up"] = diag["n_up"]
    record["n_down"] = diag["n_down"]
    record["expected_Sz"] = diag["expected_Sz"]
    record["n_particles"] = diag["n_particles_system"]
    return record


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO / "results",
        help="Root directory to scan for result.json files (default: ./results).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "reports" / "audit_symmetries.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of result.json files to audit.",
    )
    args = parser.parse_args()

    results_root: Path = args.results_root.resolve()
    if not results_root.exists():
        LOGGER.error("results-root %s does not exist", results_root)
        return 2

    paths = sorted(results_root.rglob("result.json"))
    if args.limit is not None:
        paths = paths[: args.limit]
    LOGGER.info("Auditing %d result.json files under %s", len(paths), results_root)

    records: list[dict[str, Any]] = []
    counts = {"pass": 0, "fail": 0, "skipped": 0, "error": 0, "live": 0}
    failures: list[dict[str, Any]] = []
    for path in paths:
        record = _audit_one(path)
        records.append(record)
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        if record["status"] == "fail":
            failures.append(record)
            LOGGER.error("FAIL %s — %s", record["path"], record["reason"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["status", "path", "n_up", "n_down", "expected_Sz", "n_particles", "reason"]
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})

    LOGGER.info(
        "Audit complete. pass=%d  live=%d  skipped=%d  error=%d  fail=%d",
        counts["pass"],
        counts["live"],
        counts["skipped"],
        counts["error"],
        counts["fail"],
    )
    LOGGER.info("Wrote %s", args.out_csv)

    return 1 if counts["fail"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
