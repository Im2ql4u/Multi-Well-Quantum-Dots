#!/usr/bin/env python3
"""Phase 0A: Calibrate Löwdin dot-label entanglement against the CI reference.

Background
----------
The N=2 singlet PINN runs at d ≥ 6 give ``dot_label_negativity ≈ 0.26-0.30``,
substantially below the pure-singlet expectation of ``0.50``.  Two competing
explanations:

  (i) Measurement bias: the Löwdin basis is truncated at ``max_ho_shell=1``
      (just the ground HO orbital per well), so a few percent of the true
      wavefunction sits outside the projected subspace.  As ``max_ho_shell``
      grows, the projected weight should approach 1 and the negativity should
      approach 0.5 *for any state that is structurally a singlet*.

  (ii) Ansatz limitation: the PINN itself adds intra-well correlations that
       genuinely dilute the 2x2 dot-label density matrix below the pure
       singlet form, so even an exact measurement would give neg < 0.5.

This script settles (i) by running the measurement on the **CI ground state**
(which we know is structurally a singlet at large d to numerical precision)
across a ladder of ``max_ho_shell`` values.  If CI gives neg → 0.5 as the
shell grows, the metric is sound and the PINN deficit is real (case ii above).
If CI saturates below 0.5, the metric itself is biased and we either need a
larger shell budget or a different basis (e.g. natural orbitals from the
1-RDM).

Usage
-----
    PYTHONPATH=src python3.11 scripts/calibrate_lowdin_entanglement.py \
        --separations 2 4 8 \
        --max-ho-shells 1 2 3 4 6 \
        --n-sp-states 30 \
        --n-ci-compute 200

Output is written to
``results/diag_sweeps/lowdin_calibration_<timestamp>.json``
with one entry per (separation, max_ho_shell) pair, plus a concise text
summary printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from config import SystemConfig  # noqa: E402
from observables.exact_diag_reference import (  # noqa: E402
    build_dvr_points_and_weights,
    build_shared_orbital_coefficient_matrix,
    compute_shared_ci_grid_entanglement,
)

from exact_diag_double_dot import (  # noqa: E402
    DiagConfig,
    build_2d_dvr,
    build_ci_hamiltonian,
    build_potential_matrix,
    build_slater_basis_sorted,
    infer_box_half_widths,
    precompute_coulomb_kernel,
    single_particle_eigenstates,
)


LOGGER = logging.getLogger("calibrate_lowdin_entanglement")


def solve_shared_ci(diag_cfg: DiagConfig):
    """Diagonalize the shared-model two-electron Hamiltonian and return all
    pieces needed to compute real-space entanglement."""
    x_grid, y_grid, w_x, w_y, t2d = build_2d_dvr(
        nx=diag_cfg.nx,
        ny=diag_cfg.ny,
        x_half_width=diag_cfg.x_half_width,
        y_half_width=diag_cfg.y_half_width,
    )
    t2d = diag_cfg.kinetic_prefactor * t2d
    v2d = build_potential_matrix(
        x_grid=x_grid,
        y_grid=y_grid,
        sep=diag_cfg.sep,
        omega=diag_cfg.omega,
        smooth_t=diag_cfg.smooth_t,
    )
    sp_energies, sp_vecs = single_particle_eigenstates(
        t2d=t2d,
        v2d=v2d,
        n_sp_states=diag_cfg.n_sp_states,
    )
    slater_basis = build_slater_basis_sorted(diag_cfg.n_sp_states, sp_energies)
    kernel = precompute_coulomb_kernel(
        x_grid=x_grid,
        y_grid=y_grid,
        w_x=w_x,
        w_y=w_y,
        kappa=diag_cfg.kappa,
        epsilon=diag_cfg.epsilon,
        include_quadrature_weights=True,
    )
    h_ci = build_ci_hamiltonian(
        slater_basis=slater_basis,
        single_energies=sp_energies,
        single_vecs=sp_vecs,
        kernel=kernel,
        n_ci_compute=diag_cfg.n_ci_compute,
        b_field=diag_cfg.b_field,
        g_factor=diag_cfg.g_factor,
        mu_b=diag_cfg.mu_b,
    )
    eigvals, eigvecs = eigh(h_ci)
    return eigvals, eigvecs, slater_basis, sp_vecs, x_grid, y_grid, w_x, w_y


def calibrate_one_separation(
    sep: float,
    *,
    nx: int,
    ny: int,
    omega: float,
    smooth_t: float,
    kappa: float,
    epsilon: float,
    n_sp_states: int,
    n_ci_compute: int,
    max_ho_shells: list[int],
) -> dict[str, Any]:
    """Run CI once and re-evaluate Löwdin entanglement at each shell budget."""
    x_half, y_half = infer_box_half_widths(sep=sep, omega=omega, n_wells=2)
    diag_cfg = DiagConfig(
        nx=nx,
        ny=ny,
        sep=sep,
        omega=omega,
        smooth_t=smooth_t,
        kappa=kappa,
        epsilon=epsilon,
        n_sp_states=n_sp_states,
        n_ci_compute=n_ci_compute,
        b_field=0.0,
        g_factor=2.0,
        mu_b=1.0,
        x_half_width=x_half,
        y_half_width=y_half,
        model_mode="shared",
        kinetic_prefactor=0.5,
    )

    eigvals, eigvecs, slater_basis, sp_vecs, x_grid, y_grid, w_x, w_y = solve_shared_ci(diag_cfg)
    points, weights = build_dvr_points_and_weights(x_grid, y_grid, w_x, w_y)

    orbital_coefficients = build_shared_orbital_coefficient_matrix(
        eigvec=eigvecs[:, 0],
        slater_basis=slater_basis,
        n_orbitals=n_sp_states,
        n_ci=n_ci_compute,
    )

    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=sep, omega=omega)

    shell_results: list[dict[str, Any]] = []
    for shell in max_ho_shells:
        ent = compute_shared_ci_grid_entanglement(
            single_particle_vectors=sp_vecs,
            orbital_coefficients=orbital_coefficients,
            points=points,
            weights=weights,
            system=system,
            projection_basis="localized_ho",
            max_ho_shell=shell,
        )
        dot = ent["dot_projected_entanglement"]
        particle = ent["particle_entanglement"]
        record = {
            "max_ho_shell": int(shell),
            "n_basis_total": int(dot.get("basis_dimensions", {}).get("particle_1", -1)),
            "projected_subspace_weight": float(dot["projected_subspace_weight"]),
            "dot_label_negativity": float(
                dot["dot_label_partial_transpose"]["negativity"]
            ),
            "dot_label_log_negativity": float(
                dot["dot_label_partial_transpose"]["log_negativity"]
            ),
            "projected_sector_probabilities": dot["projected_sector_probabilities"],
            "von_neumann_entropy": float(dot["von_neumann_entropy"]),
            "particle_negativity": float(particle["negativity"]),
            "particle_von_neumann_entropy": float(particle["von_neumann_entropy"]),
        }
        shell_results.append(record)
        LOGGER.info(
            "d=%.1f  shell=%d  proj_w=%.4f  dot_neg=%.5f  S_vN=%.4f",
            sep,
            shell,
            record["projected_subspace_weight"],
            record["dot_label_negativity"],
            record["von_neumann_entropy"],
        )

    return {
        "separation": float(sep),
        "ci": {
            "ground_energy": float(eigvals[0]),
            "first_excited_energy": float(eigvals[1]),
            "gap": float(eigvals[1] - eigvals[0]),
            "n_sp_states": int(n_sp_states),
            "n_ci_compute": int(n_ci_compute),
            "nx": int(nx),
            "ny": int(ny),
            "x_half_width": float(x_half),
            "y_half_width": float(y_half),
        },
        "shell_sweep": shell_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--separations", type=float, nargs="+", default=[2.0, 4.0, 8.0]
    )
    parser.add_argument(
        "--max-ho-shells", type=int, nargs="+", default=[1, 2, 3, 4, 6]
    )
    parser.add_argument("--n-sp-states", type=int, default=30)
    parser.add_argument("--n-ci-compute", type=int, default=200)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--smooth-t", type=float, default=0.2)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def adapt_grid_for_separation(args: argparse.Namespace, sep: float) -> tuple[int, int]:
    """Ensure dx <= ~0.8 bohr along x for adequate resolution at large d.

    For sep d, we need x_half ≈ d/2 + 4*ℓ_HO so dx = 2*x_half / nx;
    pick nx so dx ≤ 0.8.
    """
    x_half, _y_half = infer_box_half_widths(sep=sep, omega=args.omega, n_wells=2)
    nx_needed = max(args.nx, int(np.ceil(2.0 * x_half / 0.8)))
    return nx_needed, args.ny


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    out_path = args.out_json or (
        REPO_ROOT
        / "results"
        / "diag_sweeps"
        / f"lowdin_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sep_results: list[dict[str, Any]] = []
    for sep in args.separations:
        nx, ny = adapt_grid_for_separation(args, sep)
        LOGGER.info(
            "Calibrating d=%.1f with nx=%d, ny=%d, n_sp=%d, n_ci=%d",
            sep, nx, ny, args.n_sp_states, args.n_ci_compute,
        )
        sep_results.append(
            calibrate_one_separation(
                sep=sep,
                nx=nx,
                ny=ny,
                omega=args.omega,
                smooth_t=args.smooth_t,
                kappa=args.kappa,
                epsilon=args.epsilon,
                n_sp_states=args.n_sp_states,
                n_ci_compute=args.n_ci_compute,
                max_ho_shells=sorted(set(args.max_ho_shells)),
            )
        )

    payload = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": (
            "Phase 0A calibration of the Löwdin dot-label entanglement metric "
            "against the shared-model CI ground state. Each (separation, "
            "max_ho_shell) pair reports the projected subspace weight and the "
            "dot-label negativity. For a structurally pure singlet, neg = 0.5 "
            "is expected; deviations diagnose whether the metric is biased."
        ),
        "settings": {
            "separations": args.separations,
            "max_ho_shells": sorted(set(args.max_ho_shells)),
            "n_sp_states": args.n_sp_states,
            "n_ci_compute": args.n_ci_compute,
            "omega": args.omega,
            "smooth_t": args.smooth_t,
            "kappa": args.kappa,
            "epsilon": args.epsilon,
            "kinetic_prefactor": 0.5,
        },
        "separations": sep_results,
    }

    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    LOGGER.info("Saved calibration to %s", out_path)

    print()
    print("=" * 80)
    print("  Löwdin dot-label entanglement calibration on CI ground state")
    print("=" * 80)
    print(f"  {'d':>5} {'shell':>6} {'n_basis':>8} {'proj_w':>8} {'dot_neg':>9} "
          f"{'log_neg':>8} {'S_vN':>8} {'p_LR':>7}")
    print(f"  {'-'*72}")
    for sep_entry in sep_results:
        d = sep_entry["separation"]
        E0 = sep_entry["ci"]["ground_energy"]
        for s in sep_entry["shell_sweep"]:
            sec = s["projected_sector_probabilities"]
            p_lr = float(sec.get("LR", 0.0)) + float(sec.get("RL", 0.0))
            print(
                f"  {d:>5.1f} {s['max_ho_shell']:>6} {s['n_basis_total']:>8} "
                f"{s['projected_subspace_weight']:>8.4f} "
                f"{s['dot_label_negativity']:>9.5f} "
                f"{s['dot_label_log_negativity']:>8.4f} "
                f"{s['von_neumann_entropy']:>8.4f} {p_lr:>7.4f}"
            )
        print(f"  ({'CI E0=':>5}{E0:>9.5f})")
    print()

    print("  Verdict (interpretation):")
    print("    - If dot_neg → 0.50 as shell grows → metric is sound, basis-limited")
    print("    - If dot_neg saturates < 0.40            → metric biased; switch to NOs")
    print("    - p_LR ≈ 1.0 confirms the LR singlet sector is dominant")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
