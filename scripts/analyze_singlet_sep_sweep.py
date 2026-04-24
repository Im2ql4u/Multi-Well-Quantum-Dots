#!/usr/bin/env python3
"""Analyze no-ref N=2 singlet separation sweep results.

Reads two-stage summary JSONs for the singlet_self_residual separation sweep,
computes E(d) and seed statistics. If entanglement JSON files are present
(from measure_entanglement.py) alongside the summaries, also reports S(d).

Key physics:
  d → ∞: Mott singlet with S = log(2) ≈ 0.693, E → 2 × E_HO(1-electron)
  d → 0: both electrons near origin, correlation energy grows, S → 0

Usage:
    python3 scripts/analyze_singlet_sep_sweep.py [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO / "results" / "diag_sweeps"
LOG2 = math.log(2)   # Mott singlet entanglement limit S = log(2)


def _parse_sep_from_filename(fname: str) -> float | None:
    """Extract well separation from filename: n2_singlet_d4_s42 → 4.0."""
    m = re.search(r"singlet_d(\d+)_", fname)
    if m:
        return float(m.group(1))
    return None


def _final_energy(summary: dict) -> float | None:
    for stage in ("stage_b", "stage_a"):
        block = summary.get(stage, {})
        result = block.get("result", {})
        E = result.get("final_energy")
        if E is not None:
            return float(E)
    return None


def _entanglement_for_summary(summary_path: Path) -> float | None:
    """Look for a sidecar entanglement JSON next to the summary."""
    stem = summary_path.stem
    ent_path = summary_path.parent / f"{stem}_entanglement.json"
    if ent_path.exists():
        try:
            data = json.loads(ent_path.read_text())
            return float(data.get("entanglement", {}).get("entropy", float("nan")))
        except Exception:
            pass
    return None


def load_sep_results(
    results_dir: Path,
    glob_pattern: str = "n2_singlet_d[0-9]*_s42_seed*singlet_self_residual*two_stage_summary_*.json",
    E_min: float = 0.0,
) -> dict[float, list[tuple[float, float | None]]]:
    """Return {d: [(E, S_or_None), ...]} across seeds.

    Files with '_lam' in their name (lambda sweep) are excluded.
    Seeds with E < E_min are flagged as diverged and excluded.
    """
    files = sorted(glob.glob(str(results_dir / glob_pattern)))
    data: dict[float, list[tuple[float, float | None]]] = defaultdict(list)
    diverged: list[tuple[float, int, float]] = []
    for fpath in files:
        fname = os.path.basename(fpath)
        if "_lam" in fname:   # skip lambda sweep files
            continue
        d = _parse_sep_from_filename(fname)
        if d is None:
            continue
        try:
            with open(fpath) as fh:
                summary = json.load(fh)
        except Exception:
            continue
        E = _final_energy(summary)
        if E is None:
            continue
        m = re.search(r"seed(\d+)", fname)
        seed = int(m.group(1)) if m else -1
        if E < E_min:
            diverged.append((d, seed, E))
            continue
        S = _entanglement_for_summary(Path(fpath))
        data[d].append((E, S))

    if diverged:
        print(f"  [warn] {len(diverged)} diverged seed(s) excluded (E < {E_min}):")
        for d_div, seed_div, E_div in sorted(diverged):
            print(f"    d={d_div}, seed={seed_div}: E={E_div:.3f}")
    return dict(data)


def print_sep_summary(data: dict[float, list[tuple[float, float | None]]]) -> None:
    if not data:
        print("No singlet separation sweep results found.")
        print("Run launch_singlet_sep_sweep.sh first.")
        return

    has_entanglement = any(S is not None for runs in data.values() for _, S in runs)

    sorted_ds = sorted(data.keys())
    print(f"\n{'N=2 singlet separation sweep, no CI ref (singlet_self_residual)'}")
    print("=" * 75)
    if has_entanglement:
        header = f"  {'d':>5}  {'E_mean':>12}  {'±std':>10}  {'S_mean':>12}  {'S/log2':>8}  n"
        print(header)
        print(f"  {'-'*70}")
    else:
        header = f"  {'d':>5}  {'E_mean':>12}  {'±std':>10}  n"
        print(header)
        print(f"  {'-'*40}")

    for d in sorted_ds:
        runs = data[d]
        Es = [E for E, _ in runs]
        E_mean = sum(Es) / len(Es)
        E_std = statistics.stdev(Es) if len(Es) > 1 else 0.0

        S_vals = [S for _, S in runs if S is not None]
        if has_entanglement:
            if S_vals:
                S_mean = sum(S_vals) / len(S_vals)
                S_ratio = S_mean / LOG2
                S_str = f"{S_mean:>12.6f}  {S_ratio:>8.4f}"
            else:
                S_str = f"{'—':>12}  {'—':>8}"
            print(f"  {d:>5.0f}  {E_mean:>12.6f}  {E_std:>10.6f}  {S_str}  {len(runs)}")
        else:
            print(f"  {d:>5.0f}  {E_mean:>12.6f}  {E_std:>10.6f}  {len(runs)}")

    print()
    # Non-interacting reference: E_HO = omega * (n_up/2 + n_down/2) per electron in 2D
    # Actually for 2 electrons each in 1D HO with omega=1: E = 2 * 0.5 = 1.0 per electron
    # In 2D: E = 2 * 1.0 = 2.0 total (two HO ground states, each at E=omega=1)
    print(f"  Reference: non-interacting E = 2.0 (two 2D HO ground states, ω=1)")
    print(f"  Mott limit: S → log(2) = {LOG2:.4f} (fully entangled singlet)")
    if has_entanglement:
        print(f"  S/log(2) = 1.0 means perfect singlet entanglement (Mott insulator)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS,
        help="Directory containing two-stage summary JSONs",
    )
    args = parser.parse_args(argv)

    print(f"Scanning {args.results_dir} …")
    data = load_sep_results(args.results_dir)
    if data:
        total = sum(len(v) for v in data.values())
        print(f"Found {len(data)} separations, {total} completed runs.")
    print_sep_summary(data)


if __name__ == "__main__":
    main()
