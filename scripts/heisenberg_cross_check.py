#!/usr/bin/env python3
"""Compare PINN-extracted Mott amplitudes against an OBC Heisenberg reference.

Phase 2A.4 cross-check. For a one-electron-per-well chain trained at fixed
spin sector ``S^z = (n_up - n_down) / 2`` we

  1. extract Mott amplitudes ``c_sigma`` from the trained NQS;
  2. diagonalise the OBC Heisenberg Hamiltonian in the same basis (with
     either uniform bond couplings or user-supplied ``J_i``);
  3. report the overlap ``|<c_pinn | c_Heis>|`` and component-wise residuals;
  4. report the bipartite spin entanglement of both states for the same
     well-set partition.

A high overlap is evidence that the PINN has discovered the correct spin
physics in the deep-Mott limit; a low overlap *at large d* signals an ansatz
or training problem.

Examples
--------

    PYTHONPATH=src python3.11 scripts/heisenberg_cross_check.py \\
        --checkpoint results/p4_n4_nonmcmc_residual_anneal_s42__stageB_noref_20260424_101003

    # Use non-uniform bond couplings derived from super-exchange estimates:
    PYTHONPATH=src python3.11 scripts/heisenberg_cross_check.py \\
        --checkpoint results/some_n4_run --bond-couplings 1.0 0.5 1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observables.heisenberg_reference import (  # noqa: E402
    align_amplitude_signs,
    heisenberg_obc_ground_state,
)
from observables.spin_amplitude_entanglement import (  # noqa: E402
    evaluate_spin_amplitude_entanglement,
    well_set_bipartite_entropy,
    SpinAmplitudePayload,
)
from observables.checkpoint_loader import load_wavefunction_from_dir  # noqa: E402
from observables.spin_amplitude_entanglement import extract_spin_amplitudes  # noqa: E402


LOGGER = logging.getLogger("heisenberg_cross_check")


def _spin_string(pattern: tuple[int, ...]) -> str:
    return "".join("u" if s == 0 else "d" for s in pattern)


def _print_amplitudes(
    patterns: list[tuple[int, ...]],
    pinn_amps: np.ndarray,
    heis_amps: np.ndarray,
) -> None:
    print(f"  {'sigma':<22} {'spins':<10} {'c_pinn':>10} {'c_Heis':>10} {'diff':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for sigma, c_pinn, c_heis in zip(patterns, pinn_amps, heis_amps):
        diff = c_pinn - c_heis
        print(
            f"  {str(list(sigma)):<22} {_spin_string(sigma):<10} "
            f"{c_pinn:+10.6f} {c_heis:+10.6f} {diff:+10.6f}"
        )


def _heis_payload_for_bipartite(
    pinn_payload: SpinAmplitudePayload,
    heis_amps: np.ndarray,
) -> SpinAmplitudePayload:
    """Build a SpinAmplitudePayload with the Heisenberg amplitudes (perm signs already absorbed)."""
    log_abs = np.log(np.maximum(np.abs(heis_amps), 1e-300))
    sign = np.sign(heis_amps)
    sign[sign == 0] = 1.0
    return SpinAmplitudePayload(
        pattern=list(pinn_payload.pattern),
        log_abs_psi=log_abs,
        sign_psi=sign,
        perm_sign=np.ones_like(heis_amps),
        sigma_z_total=pinn_payload.sigma_z_total,
        n_up=pinn_payload.n_up,
        n_down=pinn_payload.n_down,
        n_wells=pinn_payload.n_wells,
        well_centers=pinn_payload.well_centers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Result directory (must contain config.yaml + model.pt).",
    )
    parser.add_argument(
        "--bond-couplings",
        type=float,
        nargs="+",
        default=None,
        help="Length-(N-1) bond couplings. Defaults to uniform J_i = 1.",
    )
    parser.add_argument(
        "--set-a",
        type=int,
        nargs="+",
        default=None,
        help="Indices of wells in subsystem A for the bipartite report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override evaluation device (cpu, cuda:0, ...).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write the full JSON payload.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    loaded = load_wavefunction_from_dir(args.checkpoint, device=args.device)
    pinn_payload = extract_spin_amplitudes(loaded)
    pinn_amps = pinn_payload.normalised()

    N = pinn_payload.n_wells
    n_down = pinn_payload.n_down
    bonds = args.bond_couplings
    heis = heisenberg_obc_ground_state(N, n_down, bond_couplings=bonds)

    if heis.patterns != list(pinn_payload.pattern):
        raise RuntimeError(
            "Heisenberg pattern enumeration disagrees with PINN extraction; "
            "this should never happen."
        )

    aligned, overlap = align_amplitude_signs(heis.ground_amplitudes, pinn_amps)
    diff = aligned - heis.ground_amplitudes
    l2_diff = float(np.linalg.norm(diff))
    linf_diff = float(np.max(np.abs(diff)))

    print("=" * 72)
    print(f"  result_dir   = {args.checkpoint}")
    print(f"  N (sites)    = {N}, n_down = {n_down}")
    print(f"  Heisenberg J = {heis.bond_couplings.tolist()}")
    print(f"  H eigenvalues (lowest {len(heis.eigenvalues)}): "
          + ", ".join(f"{e:+.6f}" for e in heis.eigenvalues))
    print(f"  GS multiplicity (within tol 1e-9): {heis.multiplicity}")
    print()

    print("  --- amplitude comparison ---")
    _print_amplitudes(pinn_payload.pattern, aligned, heis.ground_amplitudes)
    print()
    print(f"  overlap |<c_pinn|c_Heis>| = {overlap:.8f}")
    print(f"  L2 residual               = {l2_diff:.6e}")
    print(f"  Linf residual             = {linf_diff:.6e}")
    print()

    if args.set_a is None:
        set_a = list(range(N // 2))
    else:
        set_a = list(args.set_a)
    pinn_bp = well_set_bipartite_entropy(pinn_payload, set_a=set_a)
    heis_payload_for_bp = _heis_payload_for_bipartite(pinn_payload, heis.ground_amplitudes)
    heis_bp = well_set_bipartite_entropy(heis_payload_for_bp, set_a=set_a)

    print("  --- bipartite entanglement (A | B) ---")
    print(f"    A = {pinn_bp['set_a']}, B = {pinn_bp['set_b']}")
    print(
        f"    PINN: S = {pinn_bp['von_neumann_entropy']:.6f}, "
        f"neg = {pinn_bp['negativity']:.6f}, "
        f"log_neg = {pinn_bp['log_negativity']:.6f}, "
        f"effrank = {pinn_bp['effective_schmidt_rank']}"
    )
    print(
        f"    Heis: S = {heis_bp['von_neumann_entropy']:.6f}, "
        f"neg = {heis_bp['negativity']:.6f}, "
        f"log_neg = {heis_bp['log_negativity']:.6f}, "
        f"effrank = {heis_bp['effective_schmidt_rank']}"
    )
    print()

    if args.out_json is not None:
        payload = {
            "checkpoint": str(args.checkpoint),
            "N": N,
            "n_down": n_down,
            "bond_couplings": heis.bond_couplings.tolist(),
            "heisenberg_eigenvalues": heis.eigenvalues.tolist(),
            "heisenberg_multiplicity": heis.multiplicity,
            "patterns": [list(p) for p in pinn_payload.pattern],
            "pinn_amplitudes": aligned.tolist(),
            "heisenberg_amplitudes": heis.ground_amplitudes.tolist(),
            "overlap": overlap,
            "l2_residual": l2_diff,
            "linf_residual": linf_diff,
            "set_a": set_a,
            "pinn_bipartite": pinn_bp,
            "heisenberg_bipartite": heis_bp,
        }
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
