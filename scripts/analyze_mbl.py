#!/usr/bin/env python3
"""Analyze MBL disorder sweep results.

Reads two-stage summary JSONs for n8_grid_d6_s42_mbl_sig*_r* configs,
groups by disorder strength σ, and reports:
  - E(σ): energy vs disorder (should be non-monotone — Anderson localisation
    lowers E via localised single-particle states, then Coulomb raises it)
  - σ_E(σ): energy variance across disorder realizations (spread grows with σ)
  - Bipartite entanglement S(σ): from entanglement sidecar JSONs if present

Key signatures:
  - Clean (σ=0):   Mott-insulating Wigner-crystal-like state, S ~ area law
  - Weak (σ~0.3):  Correlations compete with disorder, S may grow
  - Strong (σ>d):  Anderson localisation dominates, S decreases again
  - The peak in S(σ) (if present) marks a correlated-metal / MBL crossover

Usage:
    python3.11 scripts/analyze_mbl.py
    python3.11 scripts/analyze_mbl.py --results-dir results/diag_sweeps --plot
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO / "results" / "diag_sweeps"


def _final_energy(s: dict) -> float | None:
    for stage in ("stage_b", "stage_a"):
        E = s.get(stage, {}).get("result", {}).get("final_energy")
        if E is not None:
            return float(E)
    return None


def _parse_sigma_realization(fname: str) -> tuple[float, int] | None:
    m = re.search(r"mbl_sig(\d+p\d+)_r(\d+)", fname)
    if m:
        sig_str = m.group(1).replace("p", ".")
        return float(sig_str), int(m.group(2))
    return None


def _entanglement(summary_path: Path) -> float | None:
    ent_path = summary_path.parent / (summary_path.stem + "_entanglement.json")
    if ent_path.exists():
        try:
            d = json.loads(ent_path.read_text())
            return float(d.get("entanglement", {}).get("entropy", float("nan")))
        except Exception:
            pass
    return None


def load_mbl_results(
    results_dir: Path,
) -> dict[float, list[dict]]:
    """Return {sigma: [{"realization": r, "E": float, "S": float|None}, ...]}"""
    pattern = str(results_dir / "*_mbl_sig*_r*_seed*__*__two_stage_summary_*.json")
    files = sorted(glob.glob(pattern))
    data: dict[float, list[dict]] = defaultdict(list)
    for f in files:
        key = _parse_sigma_realization(Path(f).name)
        if key is None:
            continue
        sigma, real_idx = key
        try:
            s = json.loads(Path(f).read_text())
        except Exception:
            continue
        E = _final_energy(s)
        if E is None:
            continue
        S = _entanglement(Path(f))
        data[sigma].append({"realization": real_idx, "E": E, "S": S, "path": f})
    return dict(data)


def print_mbl_table(data: dict[float, list[dict]]) -> None:
    if not data:
        print("No MBL results found yet — runs still in progress.")
        print(f"Tip: check logs/mbl_sweep/n8_mbl_gpu5.log for progress.")
        return

    has_S = any(r.get("S") is not None for runs in data.values() for r in runs)

    print()
    print("=" * 80)
    print("  MBL Disorder Sweep: N=8, d=6, ω=1 (disorder σ in units of ℓ_HO)")
    print("=" * 80)
    if has_S:
        print(f"  {'σ':>6}  {'n_real':>6}  {'n_seed':>6}  {'E_mean':>12}  "
              f"{'σ_E':>10}  {'S_mean':>10}  {'σ_S':>8}")
    else:
        print(f"  {'σ':>6}  {'n_real':>6}  {'n_seed':>6}  {'E_mean':>12}  {'σ_E':>10}  {'ΔE':>10}")
    print(f"  {'-'*72}")

    clean_E = None
    for sigma in sorted(data.keys()):
        runs = data[sigma]
        Es = [r["E"] for r in runs]
        E_mean = sum(Es) / len(Es)
        E_std = statistics.stdev(Es) if len(Es) > 1 else 0.0
        n_real = len(set(r["realization"] for r in runs))
        n_seed = len(runs)

        if sigma == 0.0:
            clean_E = E_mean

        dE = (E_mean - clean_E) if clean_E is not None else float("nan")

        if has_S:
            S_vals = [r["S"] for r in runs if r.get("S") is not None]
            S_mean = sum(S_vals) / len(S_vals) if S_vals else float("nan")
            S_std = statistics.stdev(S_vals) if len(S_vals) > 1 else 0.0
            print(f"  {sigma:>6.2f}  {n_real:>6}  {n_seed:>6}  {E_mean:>12.6f}  "
                  f"{E_std:>10.6f}  {S_mean:>10.6f}  {S_std:>8.6f}")
        else:
            print(f"  {sigma:>6.2f}  {n_real:>6}  {n_seed:>6}  {E_mean:>12.6f}  "
                  f"{E_std:>10.6f}  {dE:>+10.4f}")

    print()
    print("  Physics interpretation:")
    sigmas = sorted(data.keys())
    if len(sigmas) >= 3:
        Es = [sum(r["E"] for r in data[s]) / len(data[s]) for s in sigmas]
        # Find if there's a non-monotone feature
        if any(Es[i] < Es[i - 1] for i in range(1, len(Es))):
            print("  Energy non-monotone → localisation competing with Coulomb order ✓")
        else:
            print("  Energy monotone with σ — disorder uniformly suppresses correlations")

    if has_S:
        Ss = [(s, sum(r["S"] for r in data[s] if r.get("S") is not None) /
               max(1, sum(1 for r in data[s] if r.get("S") is not None)))
              for s in sigmas if any(r.get("S") is not None for r in data[s])]
        if Ss:
            s_peak = max(Ss, key=lambda x: x[1])
            print(f"  Peak entanglement at σ={s_peak[0]:.2f}: S={s_peak[1]:.4f}")
            print(f"  → Extended-to-MBL crossover near σ_c ~ {s_peak[0]:.2f} ℓ_HO")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    print(f"Scanning {args.results_dir} for MBL results …")
    data = load_mbl_results(args.results_dir)
    if data:
        total = sum(len(v) for v in data.values())
        print(f"Found {len(data)} σ values, {total} completed runs.")
    print_mbl_table(data)


if __name__ == "__main__":
    main()
