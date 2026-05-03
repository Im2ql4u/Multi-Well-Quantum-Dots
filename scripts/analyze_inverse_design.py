#!/usr/bin/env python3
"""Plot inverse-design optimisation trajectory.

Reads ``results/inverse_design/<run>/history.json`` (or
``optimal_geometry.json`` when present) and produces:

  * A 3-panel figure with target, energy, and |grad_theta| vs outer step.
  * If the parameter is 1D, an overlay plot of T(theta) including the
    perturbed FD evaluations.

Usage:
    PYTHONPATH=src python3.11 scripts/analyze_inverse_design.py \
        --run-dir results/inverse_design/n2_smoke_p1e

If ``--run-dir`` is omitted, the most recent run under
``results/inverse_design/`` is selected automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent


def _load_history(run_dir: Path) -> tuple[list[dict], dict | None]:
    optimal = run_dir / "optimal_geometry.json"
    history_only = run_dir / "history.json"
    if optimal.exists():
        with optimal.open() as fh:
            payload = json.load(fh)
        return payload["history"], payload
    if history_only.exists():
        with history_only.open() as fh:
            return json.load(fh), None
    raise FileNotFoundError(f"No history found in {run_dir}")


def _gather_perturbed(history: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rec in history:
        for pt in rec.get("perturbed_targets", []) or []:
            rows.append({
                "step": rec["step"],
                "theta": pt["theta_plus"],
                "T": pt["T_plus"],
            })
            rows.append({
                "step": rec["step"],
                "theta": pt["theta_minus"],
                "T": pt["T_minus"],
            })
    return rows


def _format_run_summary(history: list[dict], payload: dict | None) -> str:
    lines: list[str] = []
    if payload is not None:
        lines.append(f"target = {payload.get('target', '?')}, sense = {payload.get('sense', '?')}")
        lines.append(f"theta* = {payload.get('optimal_theta', '?')}")
    lines.append("")
    lines.append(f"  {'step':>4} {'theta':<22} {'E':>10} {'T':>10} {'|grad|':>8}")
    for rec in history:
        theta = ", ".join(f"{v:+.3f}" for v in rec["theta"])
        gnorm = float(np.linalg.norm(rec["grad_theta"])) if rec["grad_theta"] else float("nan")
        lines.append(
            f"  {rec['step']:>4} [{theta:<20}] {rec['energy']:>10.4f} {rec['target']:>10.4f} {gnorm:>8.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ideal-target", type=float, default=None,
                        help="Optional reference value to overlay on the target panel (e.g. 0.5 for max Bell entanglement).")
    args = parser.parse_args(argv)

    if args.run_dir is None:
        candidates = sorted(
            (REPO / "results" / "inverse_design").glob("*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates = [c for c in candidates if c.is_dir()]
        if not candidates:
            raise SystemExit("No runs under results/inverse_design/")
        run_dir = candidates[0]
        print(f"Auto-selected run: {run_dir}")
    else:
        run_dir = args.run_dir

    history, payload = _load_history(run_dir)
    if not history:
        raise SystemExit("History is empty.")

    print(_format_run_summary(history, payload))
    print()

    steps = np.array([rec["step"] for rec in history], dtype=int)
    targets = np.array([rec["target"] for rec in history], dtype=float)
    energies = np.array([rec["energy"] for rec in history], dtype=float)
    grad_norms = np.array(
        [float(np.linalg.norm(rec["grad_theta"])) if rec["grad_theta"] else np.nan for rec in history],
        dtype=float,
    )
    thetas = np.array([rec["theta"] for rec in history], dtype=float)

    perturbed = _gather_perturbed(history)

    is_1d = thetas.shape[1] == 1
    n_panels = 4 if is_1d else 3
    fig, axs = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.7))

    axs[0].plot(steps, targets, "o-", label="Target (centre)")
    if args.ideal_target is not None:
        axs[0].axhline(args.ideal_target, color="red", linestyle="--", lw=1.0, label=f"ideal={args.ideal_target}")
    axs[0].set_xlabel("Outer step")
    axs[0].set_ylabel("Target")
    axs[0].set_title("Target vs step")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best", fontsize=9)

    axs[1].plot(steps, energies, "s-", color="tab:orange")
    axs[1].set_xlabel("Outer step")
    axs[1].set_ylabel("Energy")
    axs[1].set_title("Energy vs step")
    axs[1].grid(True, alpha=0.3)

    axs[2].semilogy(steps, np.maximum(grad_norms, 1e-6), "^-", color="tab:green")
    axs[2].set_xlabel("Outer step")
    axs[2].set_ylabel("|grad theta|")
    axs[2].set_title("Gradient norm")
    axs[2].grid(True, alpha=0.3, which="both")

    if is_1d:
        ax = axs[3]
        ax.plot(thetas[:, 0], targets, "o-", label="centre")
        if perturbed:
            t_pert = np.array([row["theta"][0] for row in perturbed])
            T_pert = np.array([row["T"] for row in perturbed])
            ax.plot(t_pert, T_pert, "x", color="grey", alpha=0.55, label="FD eval")
        if args.ideal_target is not None:
            ax.axhline(args.ideal_target, color="red", linestyle="--", lw=1.0, label=f"ideal={args.ideal_target}")
        ax.set_xlabel(r"$\theta_0$ (well separation)")
        ax.set_ylabel("Target")
        ax.set_title("T(theta)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"Inverse design: {run_dir.name}", fontsize=11)
    fig.tight_layout()

    out_path = args.out or (run_dir / "trajectory.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
