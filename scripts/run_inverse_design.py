#!/usr/bin/env python3
"""Run inverse design: optimise quantum dot geometry for a target property.

Uses Hellmann-Feynman gradients to update well positions after each training run.
The target property can be: energy, entanglement, spin_gap, or a custom function.

Examples:

  # Minimise energy for N=8 (find optimal lattice spacing)
  CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \\
      --config configs/scaling/n8_grid_d6_s42.yaml --target energy --n-steps 10

  # Maximise bipartite entanglement between left/right halves
  CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \\
      --config configs/scaling/n8_grid_d6_s42.yaml --target entanglement --n-steps 15

  # Find geometry with specific spin gap
  CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \\
      --config configs/scaling/n8_grid_d6_s42.yaml --target spin_gap --n-steps 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from geometry_optimizer import GeometryOptimizer


def _print_history(history: list[dict]) -> None:
    if not history:
        return
    print("\n=== Inverse Design History ===")
    print(f"  {'Step':>4}  {'Energy':>12}  {'Target':>10}  {'|∇|':>8}  {'dt(s)':>8}")
    print(f"  {'-'*50}")
    for rec in history:
        print(f"  {rec['step']:>4}  {rec['energy']:>12.6f}  {rec.get('target', 0):>10.6f}"
              f"  {rec.get('grad_norm', 0):>8.4f}  {rec.get('dt_sec', 0):>8.0f}")
    print()
    best = min(history, key=lambda r: r["energy"])
    print(f"  Best energy at step {best['step']}: {best['energy']:.6f}")
    print(f"  Optimal geometry:")
    for k, center in enumerate(best["wells"]):
        print(f"    well {k}: [{center[0]:+.3f}, {center[1]:+.3f}]")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", type=str, default="energy",
                        choices=["energy", "entanglement", "spin_gap", "pair_corr_r0"],
                        help="Optimisation target")
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.3, help="Geometry learning rate (ℓ_HO)")
    parser.add_argument("--stage-a-epochs", type=int, default=2000)
    parser.add_argument("--stage-b-epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hf-samples", type=int, default=2048,
                        help="MC samples for Hellmann-Feynman gradient")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    gpu_idx = os.environ.get("CUDA_MANUAL_DEVICE", "0")
    device = f"cuda:{gpu_idx}"

    print(f"Inverse design: target={args.target}, device={device}")
    print(f"  config={args.config.name}, n_steps={args.n_steps}, lr={args.lr}")

    optimizer = GeometryOptimizer(
        base_config_path=args.config,
        target=args.target,
        n_outer_steps=args.n_steps,
        lr_geometry=args.lr,
        n_hf_samples=args.n_hf_samples,
        stage_a_epochs=args.stage_a_epochs,
        stage_b_epochs=args.stage_b_epochs,
        device=device,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    optimal_geo, history = optimizer.run()

    _print_history(history)

    # Save final geometry
    out_path = optimizer.out_dir / "optimal_geometry.json"
    out_path.write_text(json.dumps({
        "target": args.target,
        "optimal_centers": optimal_geo,
        "history": history,
    }, indent=2))
    print(f"\nOptimal geometry saved → {out_path}")


if __name__ == "__main__":
    main()
