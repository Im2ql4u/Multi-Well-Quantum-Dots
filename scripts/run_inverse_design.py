#!/usr/bin/env python3
"""Run inverse design: optimise quantum-dot geometry for a target property.

Bilevel loop
------------
* Outer:  update parameter θ (e.g. dot separation, lattice spacing) using a
          finite-difference gradient on the *real* target (entanglement, gap,
          pair correlation) or a Hellmann-Feynman gradient on energy.
* Inner:  ``scripts/run_two_stage_ground_state.py`` retrains Ψ at each
          parameter value, warm-starting from the previous outer step.

Examples
--------

# Phase 1E smoke test: maximise N=2 entanglement starting at d=2.
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \\
    --config configs/one_per_well/n2_singlet_d2_s42.yaml \\
    --target entanglement_n2 \\
    --param-init 2.0 --param-step 0.4 --param-lower 1.0 --param-upper 8.0 \\
    --n-steps 6 --lr 0.6 \\
    --stage-a-epochs 1500 --stage-b-epochs 1000 \\
    --stage-a-strategy singlet_self_residual

# Find energy-minimising spacing for an N=8 chain.
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \\
    --config configs/scaling/n8_grid_d6_s42.yaml --target energy --n-steps 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from geometry_optimizer import (  # noqa: E402
    GeometryOptimizer,
    default_param_to_wells_n2,
    make_dimer_chain_n4_param_to_wells,
    make_dimer_chain_n8_param_to_wells,
    make_dimer_pair_n8_param_to_wells,
    make_displacement_2d_param_to_wells,
    make_per_bond_chain_param_to_wells,
    make_uniform_chain_param_to_wells,
)


LOGGER = logging.getLogger("run_inverse_design")


_BUILTIN_TARGETS = {
    "energy",
    "entanglement_n2",
    "well_set_entanglement",
    "exchange_gap",
    "spin_correlator",
    "effective_J",
    "pair_corr",
}
_PARAMETRISATIONS = {
    "n2",
    "uniform_chain",
    "per_bond_chain",
    "dimer_chain_n4",
    "dimer_chain_n8",
    "dimer_pair_n8",
    "displacement_2d",
}
_CORRELATOR_MODES = {"value", "neg_value", "neg_squared_error", "squared_error"}
_WELL_SET_METRICS = {
    "von_neumann_entropy",
    "negativity",
    "log_negativity",
    "linear_entropy",
    "effective_schmidt_rank",
}


def _print_history(history: list, target: str, sense: str) -> None:
    if not history:
        return
    print()
    print(f"=== Inverse-design history (target={target}, sense={sense}) ===")
    header = f"  {'Step':>4}  {'theta':<22}  {'Energy':>10}  {'Target':>10}  {'|grad|':>8}  {'dt(s)':>8}"
    print(header)
    print(f"  {'-'*len(header)}")
    for rec in history:
        theta = ", ".join(f"{v:+.3f}" for v in rec.theta)
        gnorm = float(np.linalg.norm(rec.grad_theta)) if rec.grad_theta else float("nan")
        print(
            f"  {rec.step:>4}  [{theta:<20}]  {rec.energy:>10.4f}  {rec.target:>10.4f}"
            f"  {gnorm:>8.4f}  {rec.dt_sec:>8.0f}"
        )
    print()
    if sense == "max":
        best = max(history, key=lambda r: r.target)
    else:
        best = min(history, key=lambda r: r.target)
    print(f"  Best target value at step {best.step}: T={best.target:.5f}, E={best.energy:.5f}")
    print(f"  theta* = {np.round(best.theta, 5).tolist()}")
    print(f"  wells:")
    for k, c in enumerate(best.wells):
        print(f"    well {k}: [{c[0]:+.3f}, {c[1]:+.3f}]")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run bilevel inverse-design optimisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--target", type=str, default="entanglement_n2",
        choices=sorted(_BUILTIN_TARGETS),
        help="Built-in optimisation target.",
    )
    parser.add_argument(
        "--sense", type=str, default=None, choices=["max", "min"],
        help="Override the optimisation direction. Defaults to 'min' for energy and 'max' for everything else.",
    )

    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.5, help="Outer-loop learning rate (per parameter).")
    parser.add_argument(
        "--gradient-method",
        type=str,
        default="auto",
        choices=["auto", "hf", "fd_central", "fd_forward", "fd_backward"],
        help=(
            "Outer-loop gradient method. 'fd_central' (default for non-energy "
            "targets) trains theta+eps and theta-eps per parameter. "
            "'fd_forward' / 'fd_backward' train only one side, halving the "
            "gradient cost — useful for expensive (large N) systems."
        ),
    )

    parser.add_argument(
        "--param-init", type=float, nargs="+", default=None,
        help="Initial parameter vector. If omitted, inferred from the base config (for double_dot configs only).",
    )
    parser.add_argument(
        "--param-step", type=float, nargs="+", default=None,
        help="Per-component finite-difference step. Defaults to max(0.05, 0.1 * |theta_0|).",
    )
    parser.add_argument(
        "--param-lower", type=float, nargs="+", default=None,
        help="Optional per-component lower bound (clip after each update).",
    )
    parser.add_argument(
        "--param-upper", type=float, nargs="+", default=None,
        help="Optional per-component upper bound.",
    )

    parser.add_argument("--stage-a-epochs", type=int, default=2000)
    parser.add_argument("--stage-b-epochs", type=int, default=1500)
    parser.add_argument(
        "--stage-a-strategy", type=str, default="auto",
        choices=["auto", "guided", "self_residual", "singlet_self_residual", "improved_self_residual"],
        help="Forwarded to scripts/run_two_stage_ground_state.py.",
    )
    parser.add_argument("--stage-a-min-energy", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hf-samples", type=int, default=2048)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)

    parser.add_argument(
        "--parametrisation", type=str, default="n2",
        choices=sorted(_PARAMETRISATIONS),
        help=(
            "How theta maps to wells. "
            "'n2' = double dot with theta=[d]; "
            "'uniform_chain' = N-well chain with theta=[d] (uniform spacing); "
            "'per_bond_chain' = N-well chain with theta=[d_01, ..., d_{N-2,N-1}]; "
            "'dimer_chain_n4' = symmetric N=4 chain with theta=[d_outer, d_middle]; "
            "'dimer_chain_n8' = symmetric N=8 chain with theta=[d1, d2, d3, d4] "
            "(bonds: d1|d2|d3|d4|d3|d2|d1); "
            "'dimer_pair_n8' = SSH-like alternating-bond N=8 chain with "
            "theta=[d_short, d_long] (bonds: d_s|d_l|d_s|d_l|d_s|d_l|d_s); "
            "'displacement_2d' = free 2D displacements per well from a fixed "
            "base layout (theta=[dx_0, dy_0, dx_1, dy_1, ...]; base layout is "
            "read from the config's system.wells, the optimiser searches over "
            "the full 2N-dimensional disorder landscape)."
        ),
    )
    parser.add_argument(
        "--n-wells", type=int, default=None,
        help="Number of wells for chain parametrisations. Required for 'uniform_chain' and 'per_bond_chain'.",
    )
    parser.add_argument(
        "--omega", type=float, default=1.0,
        help="Per-well harmonic confinement omega used by the parametrisation factory.",
    )

    parser.add_argument(
        "--metric", type=str, default=None,
        help=(
            "Metric for the chosen target. "
            "For entanglement_n2: dot_label_negativity (default), dot_label_log_negativity, "
            "dot_label_von_neumann_entropy, particle_von_neumann_entropy. "
            "For well_set_entanglement: von_neumann_entropy (default), negativity, "
            "log_negativity, linear_entropy, effective_schmidt_rank."
        ),
    )
    parser.add_argument(
        "--set-a", type=int, nargs="+", default=None,
        help=(
            "Indices of wells in subsystem A for the well_set_entanglement target. "
            "Defaults to the first half of wells."
        ),
    )

    parser.add_argument(
        "--singlet-spin", type=int, nargs=2, metavar=("N_UP", "N_DOWN"),
        default=None,
        help=(
            "[exchange_gap] Override the singlet sector spin counts. "
            "Default: (N//2, N - N//2). For N=2: (1, 1)."
        ),
    )
    parser.add_argument(
        "--triplet-spin", type=int, nargs=2, metavar=("N_UP", "N_DOWN"),
        default=None,
        help=(
            "[exchange_gap] Override the triplet sector spin counts. "
            "Default: singlet_n_up + 1 / singlet_n_down - 1 (S^z=1 sector). "
            "For N=2: (2, 0) (fully polarised m=+1 triplet, single Slater det)."
        ),
    )
    parser.add_argument(
        "--triplet-stage-a-strategy", type=str, default="improved_self_residual",
        choices=["auto", "guided", "self_residual", "singlet_self_residual", "improved_self_residual"],
        help=(
            "[exchange_gap] Stage A warm-start strategy for the triplet sector. "
            "The N=2 'singlet permanent' ansatz only supports n_up=n_down=1, so the "
            "triplet sector defaults to the wider improved_self_residual recipe."
        ),
    )
    parser.add_argument(
        "--target-J", type=float, default=None,
        help=(
            "[exchange_gap] If set, optimise -(E_T - E_S - target_J)^2 (with sense=max) "
            "to drive the gap to a SPECIFIC value. Without this flag the optimiser "
            "maximises (or minimises) the signed gap directly."
        ),
    )
    parser.add_argument(
        "--unsigned-gap", action="store_true",
        help=(
            "[exchange_gap] Optimise |E_T - E_S| instead of the signed gap. "
            "Useful when you want to push the system *away* from spin-degeneracy."
        ),
    )

    parser.add_argument(
        "--pair", type=int, nargs=2, metavar=("I", "J"), default=None,
        help=(
            "[spin_correlator | effective_J] Pair (I, J) of well indices "
            "specifying the bond whose <S_I.S_J> correlator (or fitted J_(I,J)) "
            "is the scalar target."
        ),
    )
    parser.add_argument(
        "--mode", type=str, default=None, choices=sorted(_CORRELATOR_MODES),
        help=(
            "[spin_correlator | effective_J] Reduction mode. "
            "'value' (default for spin_correlator/effective_J) returns the raw "
            "correlator/J. 'neg_value' returns -<S.S> for AFM-strengthening with "
            "--sense=max. 'neg_squared_error' returns -(corr - target_value)^2 "
            "for engineer-to-spec optimisation with --sense=max. "
            "'squared_error' is the explicit minimisation form."
        ),
    )
    parser.add_argument(
        "--target-value", type=float, default=None,
        help=(
            "[spin_correlator | effective_J] Target correlator/J value, "
            "required for --mode neg_squared_error and squared_error. "
            "For AFM singlet bond limit: -0.75. For FM aligned bond: +0.25."
        ),
    )
    parser.add_argument(
        "--corr-spin-sector", type=int, nargs=2, metavar=("N_UP", "N_DOWN"),
        default=None,
        help=(
            "[spin_correlator | effective_J] Optional explicit (n_up, n_down) "
            "sector to extract from the trained checkpoint. Defaults to the "
            "spin block of the base config."
        ),
    )
    parser.add_argument(
        "--effJ-pairs", type=str, nargs="+", default=None,
        metavar="I,J",
        help=(
            "[effective_J] Optional list of pairs to include in the H_eff fit "
            "basis. Format: 'I,J' tokens. Default: all C(N, 2) pairs. "
            "Example: --effJ-pairs 0,1 1,2 2,3 for an N=4 NN-only fit."
        ),
    )
    parser.add_argument(
        "--r0", type=float, default=None,
        help=(
            "[pair_corr] Pair separation (Bohr) at which to evaluate "
            "g_sigma(r0). Required for --target pair_corr."
        ),
    )
    parser.add_argument(
        "--sigma", type=float, default=None,
        help=(
            "[pair_corr] Gaussian broadening width for the pair-correlation "
            "kernel. Defaults to 0.15 * r0."
        ),
    )
    parser.add_argument(
        "--n-corr-samples", type=int, default=4096,
        help=(
            "[pair_corr] Number of MCMC samples for g(r0) estimation. "
            "Use 4-16k for development, 64k+ for production-grade gradients."
        ),
    )
    parser.add_argument(
        "--corr-mh-warmup", type=int, default=400,
        help="[pair_corr] MH warmup steps before sampling (default 400).",
    )
    parser.add_argument(
        "--corr-mh-decorrelation", type=int, default=4,
        help="[pair_corr] MH decorrelation factor for production sweep.",
    )
    parser.add_argument(
        "--corr-seed", type=int, default=42,
        help=(
            "[pair_corr] MCMC seed for reproducibility. Same seed at theta+eps "
            "and theta-eps cuts FD noise considerably. Use --corr-seed -1 for "
            "stochastic sampling (no seeding)."
        ),
    )

    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    gpu_idx = os.environ.get("CUDA_MANUAL_DEVICE", "0")
    device = f"cuda:{gpu_idx}"

    LOGGER.info(
        "Inverse design: target=%s, device=%s, config=%s, parametrisation=%s",
        args.target, device, args.config.name, args.parametrisation,
    )

    if args.parametrisation == "n2":
        param_to_wells = default_param_to_wells_n2
    elif args.parametrisation == "uniform_chain":
        if args.n_wells is None:
            parser.error("--n-wells is required for parametrisation=uniform_chain")
        param_to_wells = make_uniform_chain_param_to_wells(args.n_wells, omega=args.omega)
    elif args.parametrisation == "per_bond_chain":
        if args.n_wells is None:
            parser.error("--n-wells is required for parametrisation=per_bond_chain")
        param_to_wells = make_per_bond_chain_param_to_wells(args.n_wells, omega=args.omega)
    elif args.parametrisation == "dimer_chain_n4":
        param_to_wells = make_dimer_chain_n4_param_to_wells(omega=args.omega)
    elif args.parametrisation == "dimer_chain_n8":
        param_to_wells = make_dimer_chain_n8_param_to_wells(omega=args.omega)
    elif args.parametrisation == "dimer_pair_n8":
        param_to_wells = make_dimer_pair_n8_param_to_wells(omega=args.omega)
    elif args.parametrisation == "displacement_2d":
        import yaml
        base_cfg_for_layout = yaml.safe_load(args.config.read_text())
        base_wells_cfg = (base_cfg_for_layout.get("system") or {}).get("wells") or []
        if not base_wells_cfg:
            parser.error(
                "parametrisation=displacement_2d requires the base config to "
                "specify system.wells (it provides the base layout that "
                "displacements are added to)."
            )
        dim = int((base_cfg_for_layout.get("system") or {}).get("dim", 2))
        base_centers = np.asarray(
            [w["center"] for w in base_wells_cfg], dtype=np.float64
        )
        if base_centers.shape[1] != dim:
            parser.error(
                f"system.dim={dim} does not match shape of system.wells centres ({base_centers.shape[1]}D)."
            )
        param_to_wells = make_displacement_2d_param_to_wells(
            base_centers, omega=args.omega, dim=dim,
        )
        if args.param_init is None:
            args.param_init = [0.0] * (dim * base_centers.shape[0])
            LOGGER.info(
                "displacement_2d: --param-init not supplied; defaulting to all zeros (= base layout) of length %d.",
                len(args.param_init),
            )
    else:
        parser.error(f"Unknown parametrisation '{args.parametrisation}'.")

    target_kwargs: dict = {}
    if args.metric is not None:
        target_kwargs["metric"] = args.metric
    if args.set_a is not None:
        target_kwargs["set_a"] = list(args.set_a)

    if args.target in ("spin_correlator", "effective_J"):
        if args.pair is None:
            parser.error(
                f"--target {args.target} requires --pair I J (the bond whose "
                "correlator/J is the target)."
            )
        target_kwargs["pair"] = [int(args.pair[0]), int(args.pair[1])]
        if args.mode is not None:
            target_kwargs["mode"] = args.mode
        if args.target_value is not None:
            target_kwargs["target_value"] = float(args.target_value)
        if args.corr_spin_sector is not None:
            target_kwargs["spin_sector"] = [
                int(args.corr_spin_sector[0]),
                int(args.corr_spin_sector[1]),
            ]
        # Validate target_value requirement for "engineer-to-spec" modes.
        mode = target_kwargs.get("mode", "value")
        if mode in ("neg_squared_error", "squared_error") and "target_value" not in target_kwargs:
            parser.error(
                f"--mode {mode} requires --target-value (the correlator/J value to engineer)."
            )
        if args.target == "effective_J" and args.effJ_pairs is not None:
            pairs_parsed: list[tuple[int, int]] = []
            for tok in args.effJ_pairs:
                parts = str(tok).replace(" ", "").split(",")
                if len(parts) != 2:
                    parser.error(f"--effJ-pairs entry '{tok}' is not 'I,J'.")
                a, b = int(parts[0]), int(parts[1])
                if a == b:
                    parser.error(f"--effJ-pairs entry '{tok}' has i == j.")
                if a > b:
                    a, b = b, a
                pairs_parsed.append((a, b))
            target_kwargs["pairs"] = pairs_parsed

    if args.target == "pair_corr":
        if args.r0 is None:
            parser.error("--target pair_corr requires --r0 (Bohr).")
        if args.r0 <= 0:
            parser.error(f"--r0 must be positive, got {args.r0!r}.")
        target_kwargs["r0"] = float(args.r0)
        if args.sigma is not None:
            target_kwargs["sigma"] = float(args.sigma)
        if args.mode is not None:
            target_kwargs["mode"] = args.mode
        if args.target_value is not None:
            target_kwargs["target_value"] = float(args.target_value)
        target_kwargs["n_samples"] = int(args.n_corr_samples)
        target_kwargs["mh_warmup"] = int(args.corr_mh_warmup)
        target_kwargs["mh_decorrelation"] = int(args.corr_mh_decorrelation)
        target_kwargs["seed"] = (
            None if int(args.corr_seed) < 0 else int(args.corr_seed)
        )
        mode = target_kwargs.get("mode", "value")
        if mode in ("neg_squared_error", "squared_error") and "target_value" not in target_kwargs:
            parser.error(
                f"--mode {mode} requires --target-value (the g(r0) value to engineer)."
            )

    # Build per-target spin overrides (multi-sector inverse design).
    spin_overrides: dict[str, dict] | None = None
    if args.target == "exchange_gap":
        if args.unsigned_gap:
            target_kwargs["signed"] = False
        if args.target_J is not None:
            target_kwargs["target_J"] = float(args.target_J)
        if args.singlet_spin is not None or args.triplet_spin is not None:
            if args.singlet_spin is None or args.triplet_spin is None:
                parser.error(
                    "When overriding exchange_gap spin sectors, supply BOTH "
                    "--singlet-spin and --triplet-spin."
                )
            n_total_singlet = int(args.singlet_spin[0]) + int(args.singlet_spin[1])
            n_total_triplet = int(args.triplet_spin[0]) + int(args.triplet_spin[1])
            if n_total_singlet != n_total_triplet:
                parser.error(
                    "exchange_gap singlet/triplet sectors must have the same particle count."
                )
            spin_overrides = {
                "singlet": {
                    "n_up": int(args.singlet_spin[0]),
                    "n_down": int(args.singlet_spin[1]),
                    "force_no_singlet_arch": False,
                },
                "triplet": {
                    "n_up": int(args.triplet_spin[0]),
                    "n_down": int(args.triplet_spin[1]),
                    "force_no_singlet_arch": True,
                    "stage_a_strategy": args.triplet_stage_a_strategy,
                },
            }
        else:
            # Defaults are filled in by GeometryOptimizer; we only need to inject
            # the per-sector triplet stage-A strategy here.
            spin_overrides = None  # let optimizer build defaults
            # We pass the strategy through target_kwargs so the optimizer can
            # patch the default override after construction.
            target_kwargs["_default_triplet_stage_a_strategy"] = (
                args.triplet_stage_a_strategy
            )

    optimizer = GeometryOptimizer(
        base_config_path=args.config,
        target=args.target,
        param_to_wells=param_to_wells,
        param_init=args.param_init,
        param_step=args.param_step,
        param_lower=args.param_lower,
        param_upper=args.param_upper,
        n_outer_steps=args.n_steps,
        lr_param=args.lr,
        sense=args.sense,
        gradient_method=args.gradient_method,
        n_hf_samples=args.n_hf_samples,
        stage_a_epochs=args.stage_a_epochs,
        stage_b_epochs=args.stage_b_epochs,
        stage_a_min_energy=args.stage_a_min_energy,
        stage_a_strategy=args.stage_a_strategy,
        device=device,
        seed=args.seed,
        out_dir=args.out_dir,
        warm_start=not args.no_warm_start,
        target_kwargs=target_kwargs,
        spin_overrides=spin_overrides,
    )

    # Patch in per-sector stage-A strategy when using built-in defaults.
    if (
        args.target == "exchange_gap"
        and spin_overrides is None
        and optimizer.spin_overrides is not None
    ):
        triplet_spec = optimizer.spin_overrides.get("triplet")
        if triplet_spec is not None and "stage_a_strategy" not in triplet_spec:
            triplet_spec["stage_a_strategy"] = args.triplet_stage_a_strategy
        # Don't pollute target_kwargs with the per-sector hint at evaluation time.
        optimizer.target_kwargs.pop("_default_triplet_stage_a_strategy", None)

    optimal_theta, history = optimizer.run()

    _print_history(history, target=args.target, sense=optimizer.sense)

    out_path = optimizer.out_dir / "optimal_geometry.json"
    out_path.write_text(json.dumps({
        "target": args.target,
        "sense": optimizer.sense,
        "optimal_theta": optimal_theta.tolist(),
        "history": [vars(rec) for rec in history],
    }, indent=2, default=str))
    print(f"Optimal geometry saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
