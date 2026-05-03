#!/usr/bin/env python3
"""Fit an effective Heisenberg ``H_eff = sum_{i<j} J_{ij} S_i.S_j`` to a checkpoint.

Phase 2A.6 evaluator. Pairs every PINN spin amplitude vector with a
covariance-method fit of the parent Heisenberg Hamiltonian, reports the
fitted ``J_{ij}`` matrix, the ground-state overlap of the fitted ``H_eff``
with the PINN amplitudes, and a per-pair JSON-serialisable summary.

Examples
--------

    # Default: fit all C(N, 2) pairs, normalise so J_(0,1) = 1.
    PYTHONPATH=src python3.11 scripts/evaluate_effective_heisenberg.py \
        --checkpoint results/p4_n4_nonmcmc_residual_anneal_s42__stageB_noref_20260424_101003

    # N=8: fit only nearest-, next-nearest- and end-to-end couplings.
    PYTHONPATH=src python3.11 scripts/evaluate_effective_heisenberg.py \
        --checkpoint results/n8_run \
        --pairs 0,1 1,2 2,3 3,4 4,5 5,6 6,7 0,2 0,7 \
        --out-json results/n8_run/effective_heisenberg.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observables.effective_heisenberg import evaluate_effective_heisenberg  # noqa: E402


LOGGER = logging.getLogger("evaluate_effective_heisenberg")


def _parse_pair(arg: str) -> tuple[int, int]:
    parts = arg.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Pair must be 'i,j' with two integers, got '{arg}'."
        )
    i, j = int(parts[0]), int(parts[1])
    if i == j:
        raise argparse.ArgumentTypeError(f"Pair must have i != j, got '{arg}'.")
    if i > j:
        i, j = j, i
    return (i, j)


def _print_human(fit: dict) -> None:
    print("=" * 72)
    print(f"  result_dir       = {fit['result_dir']}")
    print(f"  N (particles)    = {fit['n_particles']}")
    print(f"  spin sector      = {fit['spin_sector']}")
    centers = fit["well_centers"]
    print(f"  well centers     = {[[round(x, 4) for x in c] for c in centers]}")
    print()
    print("  --- effective Heisenberg fit ---")
    print(f"  pairs fitted ({len(fit['pairs'])}): {fit['pairs']}")
    print(f"  residual variance        = {fit['residual_variance']:.6e}   (lower is better)")
    print(f"  relative residual        = {fit['relative_residual']:.6e}   (< 1e-3 ~ excellent)")
    print(f"  overlap |<c|psi_0>|      = {fit['overlap_with_ground']:.6f}    (1.0 = perfect)")
    print(f"  energy split E_1 - E_0   = {fit['energy_split']:.6f}    (in J_NN units)")
    print()

    j_matrix = fit["j_matrix"]
    n = len(j_matrix)
    print("  --- J_ij matrix (J_(0,1) = 1 convention) ---")
    header = "        " + "".join(f"{j:>10d}" for j in range(n))
    print(header)
    print("        " + "----------" * n)
    for i in range(n):
        row = "  " + f"{i:3d} | "
        for j in range(n):
            row += f"{j_matrix[i][j]:+10.5f}"
        print(row)
    print()

    print("  --- per-pair fitted couplings (sorted by |J|) ---")
    pairs = fit["pairs"]
    j_vec = fit["j_vector"]
    sorted_pairs = sorted(zip(pairs, j_vec), key=lambda kv: abs(kv[1]), reverse=True)
    for (i, j), val in sorted_pairs:
        sep = abs(j - i)
        print(f"    J_({i},{j})  |i-j|={sep}    {val:+.6f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        type=_parse_pair,
        nargs="+",
        default=None,
        help="Pairs 'i,j' to fit. Default: all C(N, 2) pairs.",
    )
    parser.add_argument(
        "--no-nn-normalise",
        action="store_true",
        help="Skip rescaling J vector so J_(0,1)=1; report raw covariance eigenvector.",
    )
    parser.add_argument(
        "--allow-negative-nn",
        action="store_true",
        help="Skip the global sign flip that enforces J_(0,1) > 0.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    fit = evaluate_effective_heisenberg(
        args.checkpoint,
        pairs=args.pairs,
        nn_normalise=not args.no_nn_normalise,
        enforce_positive_nn=not args.allow_negative_nn,
        device=args.device,
    )

    if not args.quiet:
        _print_human(fit)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as fh:
            json.dump(
                fit,
                fh,
                indent=2,
                default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
            )
        print(f"Wrote payload to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
