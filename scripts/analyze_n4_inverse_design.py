#!/usr/bin/env python3
"""Analyse a Phase 2A.5 N=4 inverse-design run: per-step amplitudes, entanglement, Heisenberg overlap.

For every centre checkpoint produced by the inverse-design loop we extract:

  1. The Mott spin amplitudes ``c_sigma`` and the bipartite entanglement
     metrics for the same well-set partition the optimiser used.
  2. The OBC Heisenberg ground state computed in the same basis (uniform
     bond couplings by default; the user can override).
  3. The overlap ``|<c_pinn|c_Heis>|`` between the two.

We then write a CSV summary and a 6-panel matplotlib figure showing the
trajectory of (a) target metric vs step, (b) energy vs step, (c) Heisenberg
overlap vs step, (d) Schmidt probabilities per step, (e) amplitude evolution
of each pattern across steps, (f) gradient norm vs step.

Usage
-----

    PYTHONPATH=src python3.11 scripts/analyze_n4_inverse_design.py \
        --run-dir results/inverse_design/n4_flagship_p2a \
        --out-png results/inverse_design/n4_flagship_p2a/trajectory.png \
        --out-csv results/inverse_design/n4_flagship_p2a/trajectory.csv

If ``--out-png`` is omitted we still print a tabular summary to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observables.checkpoint_loader import load_wavefunction_from_dir  # noqa: E402
from observables.heisenberg_reference import (  # noqa: E402
    align_amplitude_signs,
    heisenberg_obc_ground_state,
)
from observables.spin_amplitude_entanglement import (  # noqa: E402
    extract_spin_amplitudes,
    well_set_bipartite_entropy,
)


LOGGER = logging.getLogger("analyze_n4_inverse_design")


def _spin_string(pattern) -> str:
    return "".join("u" if int(s) == 0 else "d" for s in pattern)


def _load_history(run_dir: Path) -> list[dict]:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        opt_path = run_dir / "optimal_geometry.json"
        if not opt_path.exists():
            raise FileNotFoundError(f"No history.json or optimal_geometry.json in {run_dir}.")
        with opt_path.open() as fh:
            payload = json.load(fh)
        return list(payload["history"])
    with history_path.open() as fh:
        return list(json.load(fh))


def _resolve_centre_dir(run_dir: Path, rec: dict) -> Path:
    centre = rec.get("centre_result_dir") or rec.get("center_result_dir")
    if centre is None:
        return run_dir / f"step{rec['step']:03d}_centre"
    return Path(centre)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--set-a", type=int, nargs="+", default=None,
                        help="Override set_a (defaults to first half).")
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    history = _load_history(args.run_dir)
    LOGGER.info("Loaded %d step records from %s", len(history), args.run_dir)

    rows: list[dict] = []
    pattern_labels: list[str] | None = None
    schmidt_per_step: list[list[float]] = []
    amplitudes_per_step: list[list[float]] = []
    heis_amplitudes_per_step: list[list[float]] = []

    for rec in history:
        step = int(rec["step"])
        theta = list(rec["theta"])
        e_centre = float(rec["energy"])
        target_centre = float(rec["target"])
        wells = rec.get("wells", None)
        ckpt_dir = _resolve_centre_dir(args.run_dir, rec)
        if not ckpt_dir.exists():
            LOGGER.warning("step %d: missing centre_result_dir %s; skipping", step, ckpt_dir)
            continue
        try:
            loaded = load_wavefunction_from_dir(ckpt_dir, device=args.device)
        except Exception as exc:
            LOGGER.warning("step %d: failed to load %s: %s", step, ckpt_dir, exc)
            continue
        try:
            payload = extract_spin_amplitudes(loaded)
        except Exception as exc:
            LOGGER.warning("step %d: extraction failed: %s", step, exc)
            continue
        amps = payload.normalised()

        if pattern_labels is None:
            pattern_labels = [_spin_string(p) for p in payload.pattern]

        N = payload.n_wells
        if args.set_a is None:
            set_a = list(range(N // 2))
        else:
            set_a = list(args.set_a)
        bp = well_set_bipartite_entropy(payload, set_a=set_a)

        heis = heisenberg_obc_ground_state(N, payload.n_down)
        aligned, overlap = align_amplitude_signs(heis.ground_amplitudes, amps)

        amplitudes_per_step.append(aligned.tolist())
        heis_amplitudes_per_step.append(heis.ground_amplitudes.tolist())
        schmidt_per_step.append(list(bp["schmidt_probs"]))

        grad_norm = float(np.linalg.norm(rec.get("grad_theta", [0.0]))) if rec.get("grad_theta") else float("nan")
        rows.append({
            "step": step,
            "theta": theta,
            "wells": wells,
            "energy": e_centre,
            "target": target_centre,
            "S_vN": float(bp["von_neumann_entropy"]),
            "negativity": float(bp["negativity"]),
            "log_negativity": float(bp["log_negativity"]),
            "schmidt_rank": int(bp["effective_schmidt_rank"]),
            "purity": float(bp["purity"]),
            "schmidt_probs": list(bp["schmidt_probs"]),
            "heisenberg_overlap": overlap,
            "grad_norm": grad_norm,
        })

    if not rows:
        print("No step records could be analysed.")
        return 1

    print()
    print(f"=== N=4 inverse-design trajectory (run={args.run_dir.name}) ===")
    header = (
        f"  {'step':>4} {'theta':<22} {'E':>10} {'S':>8} {'neg':>8} {'logneg':>8}"
        f" {'rank':>4} {'<H|psi>':>9} {'|grad|':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        theta_str = ",".join(f"{x:+.3f}" for x in r["theta"])
        print(
            f"  {r['step']:>4} [{theta_str:<20}] {r['energy']:>10.4f} "
            f"{r['S_vN']:>8.4f} {r['negativity']:>8.4f} {r['log_negativity']:>8.4f}"
            f" {r['schmidt_rank']:>4d} {r['heisenberg_overlap']:>9.5f}"
            f" {r['grad_norm']:>8.4f}"
        )
    print()

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "step", "theta", "energy", "S_vN", "negativity", "log_negativity",
                "schmidt_rank", "purity", "schmidt_probs", "heisenberg_overlap",
                "grad_norm",
            ])
            for r in rows:
                w.writerow([
                    r["step"],
                    json.dumps(r["theta"]),
                    r["energy"],
                    r["S_vN"],
                    r["negativity"],
                    r["log_negativity"],
                    r["schmidt_rank"],
                    r["purity"],
                    json.dumps(r["schmidt_probs"]),
                    r["heisenberg_overlap"],
                    r["grad_norm"],
                ])
        print(f"Wrote CSV summary -> {args.out_csv}")

    if args.out_png is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            LOGGER.warning("matplotlib not available; skipping figure generation.")
            return 0

        steps = [r["step"] for r in rows]
        s_vn = [r["S_vN"] for r in rows]
        neg = [r["negativity"] for r in rows]
        ene = [r["energy"] for r in rows]
        overlap = [r["heisenberg_overlap"] for r in rows]
        grad = [r["grad_norm"] for r in rows]
        amps_arr = np.asarray(amplitudes_per_step, dtype=np.float64)
        heis_amps_arr = np.asarray(heis_amplitudes_per_step, dtype=np.float64)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        ax = axes[0, 0]
        ax.plot(steps, s_vn, "o-", color="steelblue", label="S_vN(PINN)")
        ax.plot(steps, neg, "s--", color="cornflowerblue", alpha=0.7, label="negativity")
        ax.axhline(np.log(2.0), color="red", linestyle=":", alpha=0.5, label="ln 2")
        ax.set_xlabel("step")
        ax.set_ylabel("entanglement metric")
        ax.set_title("Bipartite entanglement vs step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        ax = axes[0, 1]
        ax.plot(steps, ene, "o-", color="darkorange")
        ax.set_xlabel("step")
        ax.set_ylabel("Energy (Ha)")
        ax.set_title("Centre energy vs step")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(steps, overlap, "o-", color="darkgreen")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="pure Heisenberg")
        ax.set_xlabel("step")
        ax.set_ylabel("|<c_pinn | c_Heis>|")
        ax.set_title("Heisenberg overlap")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        ax = axes[1, 0]
        sp = np.asarray(schmidt_per_step, dtype=np.float64)
        if sp.ndim == 2:
            for k in range(sp.shape[1]):
                ax.plot(steps, sp[:, k], "o-", label=f"$\\lambda^2_{k+1}$")
        ax.set_xlabel("step")
        ax.set_ylabel("Schmidt probability")
        ax.set_title("Schmidt distribution evolution")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        ax = axes[1, 1]
        if pattern_labels is not None:
            for k in range(amps_arr.shape[1]):
                ax.plot(steps, amps_arr[:, k], "o-", label=pattern_labels[k])
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("c_sigma (sign-aligned to Heis)")
        ax.set_title("Mott spin-amplitude trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

        ax = axes[1, 2]
        ax.plot(steps, grad, "o-", color="purple")
        ax.set_xlabel("step")
        ax.set_ylabel(r"$\|\nabla_\theta T\|$")
        ax.set_title("Gradient norm vs step")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"N=4 inverse-design trajectory: {args.run_dir.name}", fontsize=12)
        fig.tight_layout()
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=120, bbox_inches="tight")
        print(f"Wrote figure -> {args.out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
