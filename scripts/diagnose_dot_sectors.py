#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from observables.entanglement import compute_dot_projected_entanglement
from measure_entanglement import build_grids, evaluate_psi_matrix, load_model


def _summarise_dot_metrics(dot_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "projected_weight": float(dot_metrics["projected_subspace_weight"]),
        "dot_entropy": float(dot_metrics["von_neumann_entropy"]),
        "dot_negativity": float(dot_metrics["negativity"]),
        "dot_label_negativity": float(dot_metrics["dot_label_partial_transpose"]["negativity"]),
        "sector_probabilities": dot_metrics["sector_probabilities"],
        "projected_sector_probabilities": dot_metrics["projected_sector_probabilities"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose LL/LR/RL/RR sectors for Slater-only and full learned states.")
    parser.add_argument("--result-dir", required=True, help="Ground-state result directory.")
    parser.add_argument("--npts", type=int, default=28, help="Gauss-Legendre points per dimension.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for wavefunction evaluation.")
    parser.add_argument("--batch-size", type=int, default=256, help="Wavefunction evaluation batch size.")
    parser.add_argument(
        "--dot-basis",
        choices=["localized_ho", "region_average"],
        default="localized_ho",
        help="Dot projection basis.",
    )
    parser.add_argument("--dot-max-shell", type=int, default=2, help="Localized HO shell cutoff.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    model, system = load_model(args.result_dir, args.device)
    pts1, pts2, w1, w2 = build_grids(system, npts=args.npts)

    full_matrix = evaluate_psi_matrix(model, pts1, pts2, device=args.device, batch_size=args.batch_size)

    class _SlaterView:
        def __init__(self, wf):
            self.wf = wf

        def signed_log_psi(self, x):
            return self.wf.signed_log_slater(x)

    slater_model = _SlaterView(model)
    slater_matrix = evaluate_psi_matrix(slater_model, pts1, pts2, device=args.device, batch_size=args.batch_size)

    full_dot = compute_dot_projected_entanglement(
        full_matrix,
        pts1,
        pts2,
        w1,
        w2,
        system,
        projection_basis=args.dot_basis,
        max_ho_shell=args.dot_max_shell,
    )
    slater_dot = compute_dot_projected_entanglement(
        slater_matrix,
        pts1,
        pts2,
        w1,
        w2,
        system,
        projection_basis=args.dot_basis,
        max_ho_shell=args.dot_max_shell,
    )

    payload = {
        "result_dir": str(Path(args.result_dir).resolve()),
        "npts": args.npts,
        "dot_basis": args.dot_basis,
        "dot_max_shell": args.dot_max_shell,
        "slater_only": _summarise_dot_metrics(slater_dot),
        "full_state": _summarise_dot_metrics(full_dot),
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.result_dir) / f"dot_sector_diagnostic_n{args.npts}_{args.dot_basis}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())