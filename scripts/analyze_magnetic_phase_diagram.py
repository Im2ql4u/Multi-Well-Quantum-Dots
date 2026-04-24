#!/usr/bin/env python3
"""Analyze magnetic spin-sector sweep results and map the phase diagram.

Reads two-stage summary JSONs from results/diag_sweeps/ that match
the magnetic sector naming convention (*n3_*up*down*b0p5* and
*n4_*up*down*b0p5*), groups by (N, Sz, B) and computes mean ± std
energies. The sector with the lowest mean energy at each B is the
magnetic ground state.

Usage:
    python3 scripts/analyze_magnetic_phase_diagram.py [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO / "results" / "diag_sweeps"


class SectorKey(NamedTuple):
    N: int
    n_up: int
    n_down: int
    B: float   # in units of ω (from filename tag b0p5 → 0.5)

    @property
    def Sz(self) -> float:
        # Sz = (n_up - n_down) / 2  (each electron has ±1/2)
        return (self.n_up - self.n_down) / 2.0

    def label(self) -> str:
        return f"N={self.N} ↑{self.n_up}↓{self.n_down} Sz={self.Sz:+.1f}"


def _parse_b_from_tag(tag: str) -> float:
    """Convert 'b0p5' or '0p5' → 0.5."""
    m = re.search(r"(\d+)p(\d+)", tag)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return float("nan")


def _parse_sector_from_filename(fname: str) -> SectorKey | None:
    """Extract (N, n_up, n_down, B) from a summary filename or run_name."""
    # Patterns: n3_1up2down_b0p5_s42_seed42__improved_self_residual__two_stage_summary_*.json
    #           n4_2up2down_b0p5_s42_seed314__improved_self_residual__two_stage_summary_*.json
    m = re.search(
        r"n(\d+)_(\d+)up(\d+)down_b(\d+p\d+)",
        fname,
    )
    if m:
        N = int(m.group(1))
        n_up = int(m.group(2))
        n_down = int(m.group(3))
        B = _parse_b_from_tag(m.group(4))
        return SectorKey(N=N, n_up=n_up, n_down=n_down, B=B)
    return None


def _final_energy(summary: dict) -> float | None:
    """Extract best-estimate energy from a two-stage summary dict."""
    for stage in ("stage_b", "stage_a"):
        block = summary.get(stage, {})
        result = block.get("result", {})
        E = result.get("final_energy")
        if E is not None:
            return float(E)
    return None


def load_results(results_dir: Path) -> dict[SectorKey, list[float]]:
    """Return {SectorKey: [E_seed1, E_seed2, ...]} from all matching summaries."""
    pattern = str(results_dir / "*_*up*down_b*__*__two_stage_summary_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        # Broader fallback (handles no strategy tag in older filenames)
        pattern2 = str(results_dir / "*_*up*down_b*two_stage_summary_*.json")
        files = sorted(glob.glob(pattern2))

    data: dict[SectorKey, list[float]] = defaultdict(list)
    skipped = 0
    for fpath in files:
        key = _parse_sector_from_filename(os.path.basename(fpath))
        if key is None:
            skipped += 1
            continue
        try:
            with open(fpath) as fh:
                summary = json.load(fh)
        except Exception:
            skipped += 1
            continue
        E = _final_energy(summary)
        if E is not None:
            data[key].append(E)

    if skipped:
        print(f"[warn] skipped {skipped} files (no matching sector pattern or unreadable)")
    return dict(data)


def print_phase_diagram(data: dict[SectorKey, list[float]]) -> None:
    if not data:
        print("No magnetic sector results found in results/diag_sweeps/.")
        print("Run launch_magnetic_sector_sweep.sh first.")
        return

    # Group by (N, B)
    by_NB: dict[tuple[int, float], dict[SectorKey, list[float]]] = defaultdict(dict)
    for key, Es in data.items():
        by_NB[(key.N, key.B)][key] = Es

    for (N, B), sectors in sorted(by_NB.items()):
        print(f"\n{'='*72}")
        print(f"  N={N}  B={B:.2f}  (d=4, ω=1, no E_ref)")
        print(f"{'='*72}")
        print(f"  {'Sector':<28}  {'n':>3}  {'E_mean':>12}  {'E_std':>10}  {'ΔE vs GS':>10}  {'E_base':>10}")
        print(f"  {'-'*75}")

        # Sort sectors by n_up ascending (Sz ascending)
        sorted_sectors = sorted(sectors.items(), key=lambda kv: kv[0].n_up)

        stats: list[tuple[SectorKey, float, float]] = []
        for key, Es in sorted_sectors:
            import statistics
            E_mean = sum(Es) / len(Es)
            E_std = statistics.stdev(Es) if len(Es) > 1 else 0.0
            stats.append((key, E_mean, E_std))

        # Ground state = lowest mean energy
        E_gs = min(s[1] for s in stats)
        gs_key = next(k for k, E, _ in stats if E == E_gs)

        E_bases = []
        for key, E_mean, E_std in stats:
            dE = E_mean - E_gs
            # E_base = Zeeman-corrected intrinsic energy (should be ~const in Mott limit)
            E_base = E_mean - B * (key.n_up - key.n_down)
            E_bases.append(E_base)
            gs_marker = " ← GS" if key == gs_key else ""
            print(
                f"  {key.label():<28}  {len(sectors[key]):>3}  {E_mean:>12.6f}  "
                f"{E_std:>10.6f}  {dE:>10.6f}  {E_base:>10.6f}{gs_marker}"
            )

        # E_base spread → exchange energy scale and critical field
        if len(E_bases) > 1:
            E_base_spread = max(E_bases) - min(E_bases)
            # The minimum E_base sector is the most correlated (AFM exchange)
            min_E_base = min(E_bases)
            max_E_base = max(E_bases)
            # Critical field: Zeeman energy at which GS switches from ferromagnet to AFM
            # B_c * (2 * g * mu_B / 2) * delta_Sz ≈ delta_E_base per Sz step
            # With g=2, mu_B=1: B_c ≈ E_base_spread / N  (rough estimate for full ladder)
            B_c_est = E_base_spread / N
            print(f"\n  E_base spread = {E_base_spread:.4f} Ha  (AFM exchange scale, B_c ~ {B_c_est:.4f} Ha)")
            if B > B_c_est * 3:
                print(f"  B={B:.2f} >> B_c ~ {B_c_est:.4f}: fully in ferromagnet phase ✓")

        # Physical interpretation
        print()
        Sz_gs = gs_key.Sz
        if Sz_gs == 0 or (N % 2 == 0 and gs_key.n_up == N // 2):
            spin_label = "antiferromagnetic / singlet-like"
        elif abs(Sz_gs) == N / 2.0:
            spin_label = "fully polarized ferromagnet"
        else:
            spin_label = f"partial polarization (Sz={Sz_gs:+.1f})"
        print(f"  Ground state: {gs_key.label()} — {spin_label}")

        # Check for sector degeneracy (within 2σ of GS)
        degenerate = [
            k for k, E, std in stats
            if k != gs_key and (E - E_gs) < max(2 * std, 1e-3)
        ]
        if degenerate:
            print(f"  Near-degenerate sectors (within 2σ): {[k.label() for k in degenerate]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory containing two-stage summary JSONs",
    )
    args = parser.parse_args(argv)

    print(f"Scanning {args.results_dir} for magnetic sector summaries …")
    data = load_results(args.results_dir)

    if data:
        total_runs = sum(len(v) for v in data.values())
        print(f"Found {len(data)} unique sectors, {total_runs} completed runs total.")

    print_phase_diagram(data)


if __name__ == "__main__":
    main()
