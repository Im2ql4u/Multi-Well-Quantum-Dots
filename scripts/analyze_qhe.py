"""Analyze QHE training results.

Reports energy, variance, and angular momentum convergence for all completed
QHE runs in results/diag_sweeps/.  Produces an E/N vs N scaling table and
optionally plots training curves.

Usage:
    python scripts/analyze_qhe.py [--plot]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "diag_sweeps"


def load_qhe_summaries(pattern: str = "qhe_*.json") -> list[dict]:
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    runs = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        if not isinstance(d, dict) or "N" not in d:
            continue
        h = d.get("history", [])
        if not h:
            continue
        # Best epoch (lowest variance, treating as proxy for best state)
        best = min(h, key=lambda x: x["energy_var"])
        final = h[-1]
        runs.append(
            {
                "file": os.path.basename(f),
                "N": d["N"],
                "nu": d.get("nu", "?"),
                "m": d.get("m", "?"),
                "B": round(d.get("B", 0), 3),
                "geometry": "ring" if "ring" in os.path.basename(f) else "dot",
                "seed": d.get("seed", "?"),
                "final_energy": final["energy"],
                "final_var": final["energy_var"],
                "final_imag": final["imag_penalty"],
                "best_energy": best["energy"],
                "best_var": best["energy_var"],
                "best_epoch": best["epoch"],
                "imag_mean_final": final.get("imag_mean", float("nan")),
                "lz_expected": d.get("lz_expected", float("nan")),
                "n_epochs": final["epoch"] + 1,
                "history": h,
            }
        )
    return runs


def lz_actual(run: dict) -> float:
    """Lz of the full wavefunction = Lz of Laughlin phase (config-indep).

    For Ψ = |Ψ_L| × exp(iΦ), the angular momentum imag_mean encodes
      Lz_Phi = -B/2 × (Lz applied to mΣ_{i<j} arg(z_i-z_j))
    which equals -B/2 × (-m × N(N-1)/2) for the Laughlin state.

    The saved imag_mean ≈ -B/2 × m × N(N-1)/2 (exact up to Jastrow correction).
    """
    return run["imag_mean_final"]


def main():
    parser = argparse.ArgumentParser(description="Analyze QHE results")
    parser.add_argument("--plot", action="store_true", help="Plot training curves")
    parser.add_argument("--pattern", default="qhe_*.json", help="Glob pattern")
    args = parser.parse_args()

    runs = load_qhe_summaries(args.pattern)
    if not runs:
        print(f"No QHE summaries found in {RESULTS_DIR}")
        sys.exit(1)

    # Deduplicate: keep best run per (N, geometry, seed) = lowest final variance
    best_per_key: dict[tuple, dict] = {}
    for r in runs:
        key = (r["N"], r["geometry"], r["seed"])
        if key not in best_per_key or r["final_var"] < best_per_key[key]["final_var"]:
            best_per_key[key] = r

    unique_runs = sorted(best_per_key.values(), key=lambda x: (x["N"], x["geometry"]))

    print("=" * 100)
    print(f"{'N':>4} {'geom':>6} {'ν':>5} {'m':>2} {'seed':>5} | "
          f"{'E_final':>12} {'E_best':>12} {'var_best':>10} {'imag_fin':>10} | "
          f"{'E/N':>10} {'best_ep':>8}")
    print("=" * 100)
    for r in unique_runs:
        en_per_particle = r["final_energy"] / r["N"]
        print(
            f"{r['N']:>4} {r['geometry']:>6} {r['nu']:>5} {r['m']:>2} {r['seed']:>5} | "
            f"{r['final_energy']:>12.3f} {r['best_energy']:>12.3f} "
            f"{r['best_var']:>10.3e} {r['final_imag']:>10.3e} | "
            f"{en_per_particle:>10.4f} {r['best_epoch']:>8}"
        )
    print("=" * 100)

    # Lz check
    print("\nAngular momentum check (Lz eigenvalue):")
    print(f"  {'N':>4} {'geom':>6} | {'Lz_expected':>12} {'Lz(imag_mean/B*2)':>20} {'|dev|':>8}")
    for r in unique_runs:
        lz_exp = r["lz_expected"]
        # imag_mean ≈ B/2 × lz_expected for exact Laughlin eigenstate (J=0 Jastrow)
        # Since lz_expected = -m×N(N-1)/2 and imag_mean = -(B/2)×Lz_Phi = (B/2)×lz_expected
        lz_from_imag = r["imag_mean_final"] / (r["B"] / 2.0) if r["B"] != 0 else float("nan")
        try:
            dev = abs(lz_from_imag - lz_exp)
        except TypeError:
            dev = float("nan")
        print(f"  {r['N']:>4} {r['geometry']:>6} | {lz_exp:>12.2f} {lz_from_imag:>20.4f} {dev:>8.4f}")

    # E/N scaling summary (dot geometry only)
    dot_runs = [r for r in unique_runs if r["geometry"] == "dot"]
    if dot_runs:
        print("\nE/N scaling (dot geometry):")
        for r in sorted(dot_runs, key=lambda x: x["N"]):
            print(f"  N={r['N']:2d}: E/N={r['final_energy']/r['N']:.4f} Ha  "
                  f"var_best={r['best_var']:.3e}  B={r['B']}")

    # Training curve summary
    print("\nTraining convergence (final 3 epochs):")
    for r in unique_runs:
        h = r["history"]
        tail = h[-3:]
        print(f"  N={r['N']:2d} {r['geometry']:6s}: "
              + "  ".join(f"ep{x['epoch']} E={x['energy']:.2f} var={x['energy_var']:.3e}"
                          for x in tail))

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle("QHE Training Convergence")
            for r in unique_runs:
                h = r["history"]
                epochs = [x["epoch"] for x in h]
                energies = [x["energy"] for x in h]
                variances = [x["energy_var"] for x in h]
                imags = [x["imag_penalty"] for x in h]
                label = f"N={r['N']} {r['geometry']}"
                axes[0].plot(epochs, energies, marker=".", label=label)
                axes[1].semilogy(epochs, variances, marker=".", label=label)
                axes[2].semilogy(epochs, imags, marker=".", label=label)
            axes[0].set(xlabel="epoch", ylabel="E (Ha)", title="Energy")
            axes[1].set(xlabel="epoch", ylabel="Var(E_L)", title="Variance")
            axes[2].set(xlabel="epoch", ylabel="Var(E_imag)", title="Imag penalty")
            for ax in axes:
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = ROOT / "results" / "diag_sweeps" / "qhe_training_curves.pdf"
            plt.savefig(out, bbox_inches="tight")
            print(f"\nSaved plot to {out}")
        except ImportError:
            print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
