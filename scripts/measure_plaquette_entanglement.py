#!/usr/bin/env python3
"""Measure entanglement for all three bipartitions of a 2×2 plaquette model.

For a 2×2 plaquette with wells ordered bottom-left(0), bottom-right(1),
top-right(2), top-left(3), the three non-trivial bipartitions are:
  - LR (left column vs right column): particles {0,3} | {1,2}  (auto)
  - TB (bottom row vs top row):        particles {0,1} | {2,3}
  - Diag (main diagonal):             particles {0,2} | {1,3}

Reads a two-stage summary JSON, extracts the stage-b result directory,
and runs measure_entanglement.py for each bipartition.

Usage:
    python3 scripts/measure_plaquette_entanglement.py SUMMARY_JSON [--device DEVICE] [--npts N]
    python3 scripts/measure_plaquette_entanglement.py \
        results/diag_sweeps/n4_plaq2x2_d4_2up2down_s42_seed42__improved_self_residual__*.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MEASURE_SCRIPT = REPO / "scripts" / "measure_entanglement.py"
PYTHONPATH = str(REPO / "src")


BIPARTITIONS = {
    "LR": "auto",    # left column {0,3} vs right column {1,2}
    "TB": "0,1",     # bottom row {0,1} vs top row {2,3}
    "Diag": "0,2",   # main diagonal {0,2} vs anti-diagonal {1,3}
}


def run_bipartition(
    result_dir: str,
    name: str,
    partition_spec: str,
    device: str,
    npts: int,
    out_path: Path,
) -> dict:
    cmd = [
        sys.executable,
        str(MEASURE_SCRIPT),
        "--result-dir", result_dir,
        "--partition-particles", partition_spec,
        "--device", device,
        "--npts", str(npts),
        "--out", str(out_path),
    ]
    env = {"PYTHONPATH": PYTHONPATH, **__import__("os").environ}
    print(f"\n  [{name}] partition={partition_spec!r}")
    result = subprocess.run(cmd, capture_output=False, env=env)
    if result.returncode != 0:
        print(f"  WARNING: measure_entanglement.py exited with code {result.returncode}")
        return {}
    if out_path.exists():
        return json.loads(out_path.read_text())
    return {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_json",
        help="Two-stage summary JSON (e.g. n4_plaq2x2_d4_…__two_stage_summary_*.json)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--npts", type=int, default=20,
                        help="Gauss-Hermite points per dimension (default 20)")
    args = parser.parse_args(argv)

    summary_path = Path(args.summary_json).resolve()
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        sys.exit(1)

    summary = json.loads(summary_path.read_text())
    result_dir = summary.get("stage_b", {}).get("result_dir")
    if not result_dir:
        print("ERROR: stage_b.result_dir not found in summary JSON")
        sys.exit(1)

    print(f"Model directory: {result_dir}")
    run_name = summary.get("stage_b", {}).get("run_name", "?")
    E = summary.get("stage_b", {}).get("result", {}).get("final_energy", float("nan"))
    print(f"Run: {run_name}, E = {E:.6f}")

    out_dir = summary_path.parent / (summary_path.stem + "_entanglement")
    out_dir.mkdir(exist_ok=True)

    results: dict[str, dict] = {}
    for name, partition_spec in BIPARTITIONS.items():
        out_path = out_dir / f"{name}.json"
        data = run_bipartition(result_dir, name, partition_spec, args.device, args.npts, out_path)
        results[name] = data

    # Print summary table
    print("\n" + "=" * 60)
    print(f"  Plaquette entanglement summary: {summary_path.name}")
    print("=" * 60)
    print(f"  {'Bipartition':<12}  {'S (von Neumann)':>18}  {'Negativity':>12}")
    print(f"  {'-'*50}")
    for name in BIPARTITIONS:
        data = results.get(name, {})
        S = data.get("entanglement", {}).get("entropy", float("nan"))
        neg = data.get("entanglement", {}).get("negativity", float("nan"))
        print(f"  {name:<12}  {S:>18.6f}  {neg:>12.6f}")

    # Save combined result
    combined_out = out_dir / "all_bipartitions.json"
    combined_out.write_text(json.dumps({"source": str(summary_path), "E": E, "bipartitions": results}, indent=2))
    print(f"\nSaved combined results to {combined_out}")


if __name__ == "__main__":
    main()
