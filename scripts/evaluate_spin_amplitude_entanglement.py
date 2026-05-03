#!/usr/bin/env python3
"""Evaluate Mott-projected spin-amplitude entanglement on a trained checkpoint.

This is the Phase 2A.3 evaluator companion to
``src/observables/spin_amplitude_entanglement.py``. It loads a trained
``GroundStateWF`` from a result directory, extracts the Mott spin amplitudes
for all admissible patterns in the trained ``S^z`` sector, computes the
bipartite spin entanglement across a user-supplied (or default left/right)
well partition, and prints / writes a structured payload.

Examples
--------

    # default left/right partition, full payload to stdout
    PYTHONPATH=src python3.11 scripts/evaluate_spin_amplitude_entanglement.py \
        --checkpoint results/p4_n4_nonmcmc_residual_anneal_s42__stageB_noref_20260424_101003

    # custom bipartition, save JSON, choose alternate metric
    PYTHONPATH=src python3.11 scripts/evaluate_spin_amplitude_entanglement.py \
        --checkpoint results/some_n6_run \
        --set-a 0 1 2 \
        --metric negativity \
        --out-json results/some_n6_run/spin_amplitude_entanglement.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observables.spin_amplitude_entanglement import (  # noqa: E402
    evaluate_spin_amplitude_entanglement,
)


LOGGER = logging.getLogger("evaluate_spin_amplitude_entanglement")


def _format_amp_row(pattern: list[int], amp: float) -> str:
    spins = "".join("u" if s == 0 else "d" for s in pattern)
    return f"    sigma = {pattern}  ({spins})    c = {amp:+.6f}"


def _print_human(payload: dict, set_a: list[int]) -> None:
    print("=" * 72)
    print(f"  result_dir = {payload['result_dir']}")
    print(f"  N (particles) = {payload['n_particles']}")
    print(f"  n_wells       = {payload['n_wells']}")
    print(f"  spin sector   = {payload['spin_sector']}")
    centers = payload["well_centers"]
    print(f"  well centers  = {[ [round(x, 4) for x in c] for c in centers ]}")
    print()

    amps = payload["amplitudes"]
    pats = amps["patterns"]
    cvec = amps["amplitudes_normalised"]
    print(f"  --- Mott spin amplitudes ({len(pats)} patterns) ---")
    for pat, amp in zip(pats, cvec):
        print(_format_amp_row(pat, amp))
    print()

    bp = payload["bipartite"]
    print("  --- bipartite spin entanglement ---")
    print(f"    A = {bp['set_a']}, B = {bp['set_b']}")
    print(f"    matrix shape = {bp['n_a_states']} x {bp['n_b_states']}")
    schmidt_str = ", ".join(f"{p:.5f}" for p in bp["schmidt_probs"][:8])
    if len(bp["schmidt_probs"]) > 8:
        schmidt_str += f", ...(+{len(bp['schmidt_probs']) - 8} more)"
    print(f"    Schmidt probs = [{schmidt_str}]")
    n_subsystem = max(len(bp["set_a"]), len(bp["set_b"]))
    s_max = n_subsystem * math.log(2.0)
    print(
        f"    von Neumann entropy = {bp['von_neumann_entropy']:.6f}"
        f"   (max possible = ln(2) * min(|A|,|B|) = {min(len(bp['set_a']), len(bp['set_b'])) * math.log(2.0):.4f})"
    )
    print(f"    purity              = {bp['purity']:.6f}")
    print(f"    linear_entropy      = {bp['linear_entropy']:.6f}")
    print(f"    effective rank      = {bp['effective_schmidt_rank']}")
    print(f"    negativity          = {bp['negativity']:.6f}  (Bell pair = 0.5)")
    print(f"    log_negativity      = {bp['log_negativity']:.6f}  (Bell pair = 1.0)")
    print(f"    PT min eigenvalue   = {bp['min_eigenvalue_pt']:+.6e}")
    print(f"    PT # negative evals = {bp['n_negative_eigenvalues_pt']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a result directory (must contain config.yaml + model.pt).",
    )
    parser.add_argument(
        "--set-a",
        type=int,
        nargs="+",
        default=None,
        help="Indices of wells in subsystem A. Defaults to first n_wells//2 wells.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="von_neumann_entropy",
        help="Metric to highlight in the summary footer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override evaluation device (cpu, cuda:0, ...). Defaults to checkpoint's training device.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write the full JSON payload.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable printout.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    payload = evaluate_spin_amplitude_entanglement(
        args.checkpoint,
        set_a=args.set_a,
        device=args.device,
    )

    if not args.quiet:
        _print_human(payload, payload["set_a"])
        bp = payload["bipartite"]
        if args.metric in bp and isinstance(bp[args.metric], (int, float)):
            print(f"  >>> SCALAR TARGET: {args.metric} = {float(bp[args.metric]):.6f}")
            print()

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as fh:
            json.dump(
                payload,
                fh,
                indent=2,
                default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
            )
        print(f"Wrote payload to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
