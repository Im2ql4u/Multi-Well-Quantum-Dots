#!/usr/bin/env python3
"""Analyze N-scaling experiment results.

Reads two-stage summary JSONs for grid configs (n8, n12, n16, n32),
computes E/N, compares to Mott prediction, and reports quantum corrections.

Mott prediction (Mott-insulating limit, large d):
  E/N = ω + z/(2d)  where z = coordination number, d = well spacing
  2D square grid z=4: E/N ≈ 1 + 4/(2d) = 1 + 2/d

Usage:
    python3.11 scripts/analyze_scaling.py
    python3.11 scripts/analyze_scaling.py --results-dir results/diag_sweeps
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO / "results" / "diag_sweeps"


def _final_energy(s: dict) -> float | None:
    best: float | None = None
    for stage in ("stage_b", "stage_a"):
        stage_data = s.get(stage) or {}
        E = stage_data.get("result", {}).get("final_energy")
        if E is not None:
            E = float(E)
            # Reject diverged runs (non-MCMC energy should be positive and finite)
            if not math.isfinite(E) or E < -1e6:
                continue
            best = E if best is None else min(best, E)
    return best


def _parse_N_d(fname: str) -> tuple[int, float] | None:
    import re
    m = re.search(r"n(\d+)_grid_d(\d+)", fname)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None


def load_scaling_results(results_dir: Path) -> dict[tuple[int, float], list[float]]:
    # Three naming conventions: with strategy tag, without strategy tag (seed-only), and no seed tag
    patterns = [
        str(results_dir / "n*_grid_d*_s42_seed*__*__two_stage_summary_*.json"),
        str(results_dir / "n*_grid_d*_s42_seed*__two_stage_summary_*.json"),
        str(results_dir / "n*_grid_d*_s42__two_stage_summary_*.json"),
    ]
    seen: set[str] = set()
    files: list[str] = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            if f not in seen:
                seen.add(f)
                files.append(f)
    data: dict[tuple[int, float], list[float]] = defaultdict(list)
    for f in files:
        key = _parse_N_d(Path(f).name)
        if key is None:
            continue
        try:
            s = json.loads(Path(f).read_text())
        except Exception:
            continue
        E = _final_energy(s)
        if E is not None:
            data[key].append(E)
    return dict(data)


def mott_prediction(N: int, d: float, omega: float = 1.0) -> float:
    """E_Mott = N*omega + sum_over_pairs(1/r_ij) for 1-per-well grid."""
    import math
    n_cols = math.ceil(math.sqrt(N))
    n_rows = math.ceil(N / n_cols)
    centers = []
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            x = (col - (n_cols - 1) / 2.0) * d
            y = (row - (n_rows - 1) / 2.0) * d
            centers.append((x, y))
            idx += 1
    V = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            r = math.sqrt(dx * dx + dy * dy)
            V += 1.0 / r
    return N * omega + V


def print_scaling_table(data: dict[tuple[int, float], list[float]]) -> None:
    if not data:
        print("No scaling results found yet — runs still in progress.")
        return

    print()
    print("=" * 80)
    print("  N-Scaling: Ground State Energy (2D grid, d=6, ω=1, no CI ref)")
    print("=" * 80)
    print(f"  {'N':>4}  {'d':>4}  {'n':>3}  {'E_mean':>12}  {'±std':>8}  "
          f"{'E/N':>10}  {'E_Mott':>10}  {'E_corr':>10}")
    print(f"  {'-'*75}")

    for (N, d), Es in sorted(data.items()):
        m = sum(Es) / len(Es)
        s = statistics.stdev(Es) if len(Es) > 1 else 0.0
        E_mott = mott_prediction(N, d)
        E_corr = m - E_mott
        print(f"  {N:>4}  {d:>4.0f}  {len(Es):>3}  {m:>12.6f}  {s:>8.6f}  "
              f"{m/N:>10.6f}  {E_mott/N:>10.6f}  {E_corr:>+10.4f}")

    print()
    # Thermodynamic limit: E/N as N→∞
    Ns = sorted(set(k[0] for k in data))
    ds = sorted(set(k[1] for k in data))
    for d in ds:
        pts = [(N, sum(data[(N, d)]) / len(data[(N, d)])) for (N, dd), v in data.items()
               if dd == d and len(v) > 0]
        if len(pts) >= 2:
            pts.sort()
            # E/N convergence
            print(f"  E/N convergence at d={d}:")
            for N, E in pts:
                print(f"    N={N:3d}: E/N = {E/N:.6f}")
            if len(pts) >= 2:
                # Extrapolate: E/N ~ a + b/N
                N1, E1 = pts[-2]; N2, E2 = pts[-1]
                if N2 != N1:
                    b = (E2/N2 - E1/N1) / (1/N2 - 1/N1)
                    a = E1/N1 - b/N1
                    print(f"    Thermodynamic limit E/N → {a:.6f}  (1/N extrapolation)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    print(f"Scanning {args.results_dir} for scaling results …")
    data = load_scaling_results(args.results_dir)
    if data:
        print(f"Found {len(data)} (N,d) pairs, "
              f"{sum(len(v) for v in data.values())} total runs.")
    print_scaling_table(data)


if __name__ == "__main__":
    main()
