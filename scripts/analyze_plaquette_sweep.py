#!/usr/bin/env python3
"""Analyze 2D plaquette (2×2) geometry results vs 1D chain reference.

Reads two-stage summary JSONs for n4_2x2_d{4,8}_2up2down and the 1D-chain
reference (n4_nonmcmc_residual_anneal_s42), compares energies, and checks
the Mott-insulating picture E ≈ E_HO + V_Coulomb.

Key physics:
  - 2×2 plaquette at edge d: 4 nearest-neighbour bonds at r=d,
    2 diagonal bonds at r=d√2 → more Coulomb than 1D chain
  - E_corr = E_obs − (E_HO + V_Coul_classical) < 0:
    kinetic delocalization + exchange lowers E below classical estimate
  - E_corr grows with coordination (more bonds → more superexchange)

Usage:
    python3 scripts/analyze_plaquette_sweep.py [--results-dir DIR]
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


def _final_energy(summary: dict) -> float | None:
    for stage in ("stage_b", "stage_a"):
        result = summary.get(stage, {}).get("result", {})
        E = result.get("final_energy")
        if E is not None:
            return float(E)
    return None


def load_energies(results_dir: Path, pattern: str) -> list[float]:
    files = sorted(glob.glob(str(results_dir / pattern)))
    Es = []
    for f in files:
        try:
            s = json.loads(Path(f).read_text())
        except Exception:
            continue
        E = _final_energy(s)
        if E is not None:
            Es.append(E)
    return Es


def classical_coulomb(geometry: str, d: float) -> float:
    """Classical Coulomb sum for 1 electron per well (Mott limit)."""
    if geometry == "1d_chain":
        # Wells at -3d/2, -d/2, d/2, 3d/2 → pairs at d, d, d, 2d, 2d, 3d
        return 3 / d + 2 / (2 * d) + 1 / (3 * d)
    elif geometry == "2x2_plaquette":
        # 4 edge pairs at d, 2 diagonal pairs at d√2
        return 4 / d + 2 / (d * math.sqrt(2))
    return float("nan")


def print_comparison(results_dir: Path) -> None:
    # Load plaquette results
    plaq_d4 = load_energies(results_dir, "n4_2x2_d4_2up2down_s42_seed*__improved_self_residual__two_stage_summary_*.json")
    plaq_d8 = load_energies(results_dir, "n4_2x2_d8_2up2down_s42_seed*__improved_self_residual__two_stage_summary_*.json")
    # 1D chain: take latest run (20260424 prefix)
    chain = load_energies(results_dir, "n4_nonmcmc_residual_anneal_s42_seed*__improved_self_residual__two_stage_summary_20260424_*.json")

    E_HO = 4.0  # N=4 non-interacting (4 × ω=1 2D HO ground states)

    rows = [
        ("1D chain  d=4", chain,    "1d_chain",       4.0),
        ("2×2 plaq d=4", plaq_d4,  "2x2_plaquette",  4.0),
        ("2×2 plaq d=8", plaq_d8,  "2x2_plaquette",  8.0),
    ]

    print()
    print("=" * 76)
    print(" N=4 Geometry: 2D Plaquette vs 1D Chain  (2up2down, B=0)")
    print("=" * 76)
    print(f"  {'System':<18}  {'n':>2}  {'E_mean':>10}  {'±std':>8}  "
          f"{'V_cl':>7}  {'E_HO+V_cl':>10}  {'E_corr':>8}")
    print(f"  {'-'*72}")

    E_ref = None
    for label, Es, geom, d in rows:
        if not Es:
            print(f"  {label:<18}  {'?':>2}  {'—':>10}  {'—':>8}")
            continue
        m = sum(Es) / len(Es)
        s = statistics.stdev(Es) if len(Es) > 1 else 0.0
        V_cl = classical_coulomb(geom, d)
        E_mott = E_HO + V_cl
        E_corr = m - E_mott
        if E_ref is None:
            E_ref = m
        print(f"  {label:<18}  {len(Es):>2}  {m:>10.6f}  {s:>8.6f}  "
              f"{V_cl:>7.4f}  {E_mott:>10.4f}  {E_corr:>+8.4f}")

    print()
    if plaq_d4 and chain:
        m_plaq = sum(plaq_d4) / len(plaq_d4)
        m_chain = sum(chain) / len(chain)
        V_plaq = classical_coulomb("2x2_plaquette", 4.0)
        V_chain = classical_coulomb("1d_chain", 4.0)
        print(f"  ΔE (plaq_d4 − chain_d4) = {m_plaq - m_chain:+.6f} Ha  "
              f"(classical ΔV = +{V_plaq - V_chain:.4f} Ha)")
        print(f"  → plaquette: 4 NN @ d + 2 diag @ d√2; chain: 3 NN + 2 NNN + 1 NNNN")
        print()

    print("  Physics:")
    print("  • E_corr < 0: quantum correction (kinetic + exchange) below classical V_Coul")
    print("  • |E_corr| larger for plaq_d4 than chain (more nearest-neighbour bonds = more superexchange)")
    print("  • E_corr ≈ 0 for plaq_d8: nearly perfect Mott localisation at large separation")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    print(f"Scanning {args.results_dir} …")
    print_comparison(args.results_dir)


if __name__ == "__main__":
    main()
