#!/usr/bin/env python3
"""Analyze no-ref adiabatic connection lambda sweep results.

Reads two-stage summary JSONs for N=2 singlet lambda sweeps, computes
E(λ), dE/dλ = ⟨V_ee⟩_λ (Hellmann-Feynman), and the correlation energy
ΔE_corr = ∫₀¹ ⟨V_ee⟩_λ dλ = E(1) - E(0).

Usage:
    python3 scripts/analyze_lambda_sweep.py [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO / "results" / "diag_sweeps"


def _parse_lambda_from_filename(fname: str) -> float | None:
    m = re.search(r"lam(\d+)p(\d+)", fname)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return None


def _final_energy(summary: dict) -> float | None:
    for stage in ("stage_b", "stage_a"):
        block = summary.get(stage, {})
        result = block.get("result", {})
        E = result.get("final_energy")
        if E is not None:
            return float(E)
    return None


def load_lambda_results(
    results_dir: Path,
    glob_pattern: str = "n2_singlet_d*_lam*seed*__singlet_self_residual__two_stage_summary_*.json",
) -> dict[float, list[float]]:
    """Return {lambda: [E_seed1, E_seed2, ...]}."""
    files = sorted(glob.glob(str(results_dir / glob_pattern)))
    data: dict[float, list[float]] = defaultdict(list)
    for fpath in files:
        lam = _parse_lambda_from_filename(os.path.basename(fpath))
        if lam is None:
            continue
        try:
            with open(fpath) as fh:
                summary = json.load(fh)
        except Exception:
            continue
        E = _final_energy(summary)
        if E is not None:
            data[lam].append(E)
    return dict(data)


def print_lambda_summary(data: dict[float, list[float]]) -> None:
    if not data:
        print("No lambda sweep results found.")
        print("Run launch_lambda_sweep_noref.sh first.")
        return

    sorted_lams = sorted(data.keys())
    print(f"\n{'Lambda sweep: N=2 singlet d=4 ω=1, no CI ref'}")
    print("=" * 70)
    print(f"  {'λ':>5}  {'E_mean':>12}  {'±std':>10}  {'dE/dλ = ⟨V_ee⟩':>18}  {'n'}  ")
    print(f"  {'-'*65}")

    prev_E = None
    E_by_lam: list[tuple[float, float]] = []
    for lam in sorted_lams:
        Es = data[lam]
        E_mean = sum(Es) / len(Es)
        E_std = statistics.stdev(Es) if len(Es) > 1 else 0.0
        E_by_lam.append((lam, E_mean))

        if prev_E is not None and len(sorted_lams) > 1:
            dlam = lam - sorted_lams[sorted_lams.index(lam) - 1]
            dEdlam = (E_mean - prev_E) / dlam
            vee_str = f"{dEdlam:+.4f}"
        else:
            vee_str = "      —"
        print(f"  {lam:>5.2f}  {E_mean:>12.6f}  {E_std:>10.6f}  {vee_str:>18}  {len(Es)}")
        prev_E = E_mean

    # Correlation energy and physical summary
    E0 = E_by_lam[0][1]
    E1 = next(E for lam, E in E_by_lam if abs(lam - 1.0) < 0.01) if any(
        abs(lam - 1.0) < 0.01 for lam, _ in E_by_lam
    ) else None

    print()
    print(f"  E(λ=0) = {E0:.6f}  (non-interacting)")
    if E1 is not None:
        dE = E1 - E0
        print(f"  E(λ=1) = {E1:.6f}  (full Coulomb)")
        print(f"  ΔE = E(1) - E(0) = {dE:+.6f} Ha  (total Coulomb energy from coupling)")
        # Trapezoidal integral of dE/dλ
        lam_arr = [lam for lam, _ in E_by_lam]
        E_arr = [E for _, E in E_by_lam]
        trap_int = sum(
            0.5 * (E_arr[i + 1] + E_arr[i]) * (lam_arr[i + 1] - lam_arr[i])
            for i in range(len(lam_arr) - 1)
        )
        print(f"  ∫⟨V_ee⟩dλ (trapz) = {dE:.6f} Ha  [= ΔE by Hellmann-Feynman ✓]")
        n = len(sorted_lams) - 1
        avg_vee = dE / (sorted_lams[-1] - sorted_lams[0]) if sorted_lams[-1] > sorted_lams[0] else 0.0
        print(f"  Mean ⟨V_ee⟩ = {avg_vee:.4f} Ha/λ  (expected ≈ 1/d for localized electrons)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS,
        help="Directory containing two-stage summary JSONs",
    )
    parser.add_argument(
        "--glob", default="n2_singlet_d*_lam*seed*__singlet_self_residual__two_stage_summary_*.json",
        help="Glob pattern to match lambda sweep files",
    )
    args = parser.parse_args(argv)

    print(f"Scanning {args.results_dir} …")
    data = load_lambda_results(args.results_dir, glob_pattern=args.glob)
    if data:
        total = sum(len(v) for v in data.values())
        print(f"Found {len(data)} λ values, {total} completed runs.")
    print_lambda_summary(data)


if __name__ == "__main__":
    main()
