from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh

from config import SystemConfig
from observables.exact_diag_reference import (
    build_one_per_well_orbital_coefficient_matrix,
    build_dvr_points_and_weights,
    build_shared_orbital_coefficient_matrix,
    compute_one_per_well_ci_grid_entanglement,
    compute_shared_ci_grid_entanglement,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from exact_diag_double_dot import (
    DiagConfig,
    build_2d_dvr,
    build_ci_hamiltonian,
    build_centered_harmonic_potential_matrix,
    build_potential_matrix,
    build_slater_basis_sorted,
    infer_box_half_widths,
    precompute_coulomb_kernel,
    single_particle_eigenstates,
)


LOGGER = logging.getLogger("compare_ci_vmc_dot_entanglement")


@dataclass(frozen=True)
class SweepConfig:
    nx: int
    ny: int
    omega: float
    smooth_t: float
    kappa: float
    epsilon: float
    kinetic_prefactor: float
    b_field: float
    g_factor: float
    mu_b: float
    projection_basis: str
    max_ho_shell: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare VMC dot-label entanglement against a shared-model exact-diag grid reference."
    )
    parser.add_argument(
        "--vmc-summary",
        type=Path,
        default=Path("results/diag_sweeps/nonmcmc_entanglement_summary_signed_localized_highres_20260414.json"),
        help="Path to the high-resolution VMC entanglement summary JSON.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["d2", "d4", "d8"],
        help="Ground-state labels from the VMC summary to validate.",
    )
    parser.add_argument(
        "--n-sp-states",
        nargs="+",
        type=int,
        default=[20, 30, 40],
        help="Single-particle basis sizes to sweep.",
    )
    parser.add_argument(
        "--n-ci-compute",
        nargs="+",
        type=int,
        default=[100, 200, 300],
        help="CI truncation sizes to sweep.",
    )
    parser.add_argument("--nx", type=int, default=20, help="DVR points along x.")
    parser.add_argument("--ny", type=int, default=20, help="DVR points along y.")
    parser.add_argument("--omega", type=float, default=1.0, help="Harmonic confinement frequency.")
    parser.add_argument("--smooth-t", type=float, default=0.2, help="Soft-min temperature.")
    parser.add_argument("--kappa", type=float, default=1.0, help="Coulomb interaction strength.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Soft Coulomb epsilon.")
    parser.add_argument("--kinetic-prefactor", type=float, default=0.5, help="Kinetic prefactor in front of the DVR Laplacian.")
    parser.add_argument("--b-field", type=float, default=0.0, help="Magnetic field strength.")
    parser.add_argument("--g-factor", type=float, default=2.0, help="g-factor for Zeeman term.")
    parser.add_argument("--mu-b", type=float, default=1.0, help="Bohr magneton factor.")
    parser.add_argument(
        "--projection-basis",
        type=str,
        default="localized_ho",
        choices=["region_average", "localized_ho"],
        help="Dot projection basis.",
    )
    parser.add_argument(
        "--max-ho-shell",
        type=int,
        default=2,
        help="Maximum localized harmonic-oscillator shell for localized projection.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional explicit output JSON path. Defaults to a timestamped file in results/diag_sweeps/.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def label_to_sep(label: str) -> float:
    if not label.startswith("d"):
        raise ValueError(f"Expected label of the form d<sep>, got '{label}'.")
    return float(label[1:])


def solve_shared_ci_reference(
    diag_cfg: DiagConfig,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, str, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    single_energies, single_vecs = single_particle_eigenstates(
        t2d=t2d,
        v2d=v2d,
        n_sp_states=diag_cfg.n_sp_states,
    )
    slater_basis = build_slater_basis_sorted(diag_cfg.n_sp_states, single_energies)
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
        single_energies=single_energies,
        single_vecs=single_vecs,
        kernel=kernel,
        n_ci_compute=diag_cfg.n_ci_compute,
        b_field=diag_cfg.b_field,
        g_factor=diag_cfg.g_factor,
        mu_b=diag_cfg.mu_b,
    )
    eigvals, eigvecs = eigh(h_ci)
    return eigvals, eigvecs, slater_basis, single_vecs, x_grid, y_grid, w_x, w_y


def solve_one_per_well_ci_reference(
    diag_cfg: DiagConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[float, int, int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    x_grid, y_grid, w_x, w_y, t2d = build_2d_dvr(
        nx=diag_cfg.nx,
        ny=diag_cfg.ny,
        x_half_width=diag_cfg.x_half_width,
        y_half_width=diag_cfg.y_half_width,
    )
    t2d = diag_cfg.kinetic_prefactor * t2d
    v_left = build_centered_harmonic_potential_matrix(
        x_grid=x_grid,
        y_grid=y_grid,
        omega=diag_cfg.omega,
        center_x=-0.5 * diag_cfg.sep,
    )
    v_right = build_centered_harmonic_potential_matrix(
        x_grid=x_grid,
        y_grid=y_grid,
        omega=diag_cfg.omega,
        center_x=0.5 * diag_cfg.sep,
    )
    left_energies, left_vecs = single_particle_eigenstates(
        t2d=t2d,
        v2d=v_left,
        n_sp_states=diag_cfg.n_sp_states,
    )
    right_energies, right_vecs = single_particle_eigenstates(
        t2d=t2d,
        v2d=v_right,
        n_sp_states=diag_cfg.n_sp_states,
    )
    kernel = precompute_coulomb_kernel(
        x_grid=x_grid,
        y_grid=y_grid,
        w_x=w_x,
        w_y=w_y,
        kappa=diag_cfg.kappa,
        epsilon=diag_cfg.epsilon,
        include_quadrature_weights=False,
    )

    product_basis: list[tuple[float, int, int]] = []
    for left_idx in range(diag_cfg.n_sp_states):
        for right_idx in range(diag_cfg.n_sp_states):
            product_basis.append((float(left_energies[left_idx] + right_energies[right_idx]), left_idx, right_idx))
    product_basis.sort(key=lambda item: (item[0], item[1], item[2]))
    product_basis = product_basis[: diag_cfg.n_ci_compute]

    n_cfg = len(product_basis)
    h_ci = np.zeros((n_cfg, n_cfg), dtype=np.float64)
    two_e_cache: dict[tuple[int, int, int, int], float] = {}

    def two_e(left_bra: int, right_bra: int, left_ket: int, right_ket: int) -> float:
        key = (left_bra, right_bra, left_ket, right_ket)
        if key not in two_e_cache:
            two_e_cache[key] = float(
                (left_vecs[:, left_bra] * left_vecs[:, left_ket])
                @ kernel
                @ (right_vecs[:, right_bra] * right_vecs[:, right_ket])
            )
        return two_e_cache[key]

    zeeman_single = 0.5 * diag_cfg.g_factor * diag_cfg.mu_b * diag_cfg.b_field
    for idx_i, (_energy_i, left_i, right_i) in enumerate(product_basis):
        h_ci[idx_i, idx_i] = float(left_energies[left_i] + right_energies[right_i] + zeeman_single)
        for idx_j in range(idx_i, n_cfg):
            _energy_j, left_j, right_j = product_basis[idx_j]
            h_ci[idx_i, idx_j] += two_e(left_i, right_i, left_j, right_j)
            if idx_i != idx_j:
                h_ci[idx_j, idx_i] = h_ci[idx_i, idx_j]

    eigvals, eigvecs = eigh(h_ci)
    return eigvals, eigvecs, product_basis, left_vecs, right_vecs, x_grid, y_grid, w_x, w_y


def summarise_entanglement_run(
    n_sp_states: int,
    n_ci_compute: int,
    eigvals: np.ndarray,
    entanglement: dict[str, Any],
) -> dict[str, Any]:
    return {
        "n_sp_states": n_sp_states,
        "n_ci_compute": n_ci_compute,
        "ground_energy": float(eigvals[0]),
        "particle_entropy": float(entanglement["particle_entanglement"]["von_neumann_entropy"]),
        "particle_negativity": float(entanglement["particle_entanglement"]["negativity"]),
        "projected_weight": float(entanglement["dot_projected_entanglement"]["projected_subspace_weight"]),
        "dot_entropy": float(entanglement["dot_projected_entanglement"]["von_neumann_entropy"]),
        "dot_negativity": float(entanglement["dot_projected_entanglement"]["negativity"]),
        "dot_label_negativity": float(
            entanglement["dot_projected_entanglement"]["dot_label_partial_transpose"]["negativity"]
        ),
        "sector_probabilities": entanglement["dot_projected_entanglement"]["sector_probabilities"],
        "projected_sector_probabilities": entanglement["dot_projected_entanglement"]["projected_sector_probabilities"],
    }


def compare_reference_model(
    label: str,
    system: SystemConfig,
    sweep_cfg: SweepConfig,
    n_sp_choices: list[int],
    n_ci_choices: list[int],
    *,
    reference_model: str,
) -> dict[str, Any]:
    sep = label_to_sep(label)
    x_half_width, y_half_width = infer_box_half_widths(sep=sep, omega=sweep_cfg.omega, n_wells=2)
    runs: list[dict[str, Any]] = []
    preferred_pair = (max(n_sp_choices), max(n_ci_choices))
    preferred_run: dict[str, Any] | None = None

    for n_sp_states in sorted(n_sp_choices):
        for n_ci_compute in sorted(n_ci_choices):
            diag_cfg = DiagConfig(
                nx=sweep_cfg.nx,
                ny=sweep_cfg.ny,
                sep=sep,
                omega=sweep_cfg.omega,
                smooth_t=sweep_cfg.smooth_t,
                kappa=sweep_cfg.kappa,
                epsilon=sweep_cfg.epsilon,
                n_sp_states=n_sp_states,
                n_ci_compute=n_ci_compute,
                b_field=sweep_cfg.b_field,
                g_factor=sweep_cfg.g_factor,
                mu_b=sweep_cfg.mu_b,
                kinetic_prefactor=sweep_cfg.kinetic_prefactor,
                x_half_width=x_half_width,
                y_half_width=y_half_width,
                model_mode=reference_model,
            )
            if reference_model == "shared":
                eigvals, eigvecs, slater_basis, single_vecs, x_grid, y_grid, w_x, w_y = solve_shared_ci_reference(diag_cfg)
                points, weights = build_dvr_points_and_weights(x_grid, y_grid, w_x, w_y)
                orbital_coefficients = build_shared_orbital_coefficient_matrix(
                    eigvec=eigvecs[:, 0],
                    slater_basis=slater_basis,
                    n_orbitals=n_sp_states,
                    n_ci=n_ci_compute,
                )
                entanglement = compute_shared_ci_grid_entanglement(
                    single_particle_vectors=single_vecs,
                    orbital_coefficients=orbital_coefficients,
                    points=points,
                    weights=weights,
                    system=system,
                    projection_basis=sweep_cfg.projection_basis,
                    max_ho_shell=sweep_cfg.max_ho_shell,
                )
            elif reference_model == "one_per_well":
                eigvals, eigvecs, product_basis, left_vecs, right_vecs, x_grid, y_grid, w_x, w_y = solve_one_per_well_ci_reference(diag_cfg)
                points, weights = build_dvr_points_and_weights(x_grid, y_grid, w_x, w_y)
                orbital_coefficients = build_one_per_well_orbital_coefficient_matrix(
                    eigvec=eigvecs[:, 0],
                    product_basis=product_basis,
                    n_left_orbitals=n_sp_states,
                    n_right_orbitals=n_sp_states,
                    n_ci=n_ci_compute,
                )
                entanglement = compute_one_per_well_ci_grid_entanglement(
                    left_vectors=left_vecs,
                    right_vectors=right_vecs,
                    orbital_coefficients=orbital_coefficients,
                    points=points,
                    weights=weights,
                    system=system,
                    projection_basis=sweep_cfg.projection_basis,
                    max_ho_shell=sweep_cfg.max_ho_shell,
                )
            else:
                raise ValueError(f"Unsupported reference_model '{reference_model}'.")

            run_summary = summarise_entanglement_run(
                n_sp_states=n_sp_states,
                n_ci_compute=n_ci_compute,
                eigvals=eigvals,
                entanglement=entanglement,
            )
            runs.append(run_summary)
            LOGGER.info(
                "%s | %s | n_sp=%d n_ci=%d | E=%.8f | dot_label_neg=%.8f | proj_weight=%.6f",
                label,
                reference_model,
                n_sp_states,
                n_ci_compute,
                run_summary["ground_energy"],
                run_summary["dot_label_negativity"],
                run_summary["projected_weight"],
            )
            if (n_sp_states, n_ci_compute) == preferred_pair:
                preferred_run = run_summary

    if preferred_run is None:
        raise RuntimeError(f"Preferred run not found for {label} [{reference_model}].")

    for run in runs:
        run["delta_to_preferred_energy"] = run["ground_energy"] - preferred_run["ground_energy"]
        run["delta_to_preferred_dot_label_negativity"] = (
            run["dot_label_negativity"] - preferred_run["dot_label_negativity"]
        )
        run["delta_to_preferred_projected_weight"] = run["projected_weight"] - preferred_run["projected_weight"]

    return {
        "reference_model": reference_model,
        "preferred": preferred_run,
        "sweep": runs,
    }


def compare_label(
    label: str,
    vmc_entry: dict[str, Any],
    sweep_cfg: SweepConfig,
    n_sp_choices: list[int],
    n_ci_choices: list[int],
) -> dict[str, Any]:
    sep = label_to_sep(label)
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=sep, omega=sweep_cfg.omega)
    vmc_preferred = vmc_entry["preferred"]

    shared_reference = compare_reference_model(
        label=label,
        system=system,
        sweep_cfg=sweep_cfg,
        n_sp_choices=n_sp_choices,
        n_ci_choices=n_ci_choices,
        reference_model="shared",
    )
    one_per_well_reference = compare_reference_model(
        label=label,
        system=system,
        sweep_cfg=sweep_cfg,
        n_sp_choices=n_sp_choices,
        n_ci_choices=n_ci_choices,
        reference_model="one_per_well",
    )

    references = {
        "shared": shared_reference,
        "one_per_well": one_per_well_reference,
    }
    vmc_vs_reference: dict[str, dict[str, float]] = {}
    for reference_name, reference_payload in references.items():
        preferred_run = reference_payload["preferred"]
        vmc_vs_reference[reference_name] = {
            "vmc_dot_label_negativity": float(vmc_preferred["dot_label_negativity"]),
            "reference_dot_label_negativity": float(preferred_run["dot_label_negativity"]),
            "absolute_difference": float(
                preferred_run["dot_label_negativity"] - float(vmc_preferred["dot_label_negativity"])
            ),
            "vmc_projected_weight": float(vmc_preferred["projected_weight"]),
            "reference_projected_weight": float(preferred_run["projected_weight"]),
            "reference_ground_energy": float(preferred_run["ground_energy"]),
        }

    return {
        "label": label,
        "separation": sep,
        "vmc_reference": vmc_preferred,
        "references": references,
        "vmc_vs_reference": vmc_vs_reference,
    }


def choose_output_path(path_arg: Path | None) -> Path:
    if path_arg is not None:
        return path_arg
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"results/diag_sweeps/ci_vmc_dot_validation_{timestamp}.json")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    with args.vmc_summary.open("r", encoding="utf-8") as handle:
        vmc_summary = json.load(handle)

    sweep_cfg = SweepConfig(
        nx=args.nx,
        ny=args.ny,
        omega=args.omega,
        smooth_t=args.smooth_t,
        kappa=args.kappa,
        epsilon=args.epsilon,
        kinetic_prefactor=args.kinetic_prefactor,
        b_field=args.b_field,
        g_factor=args.g_factor,
        mu_b=args.mu_b,
        projection_basis=args.projection_basis,
        max_ho_shell=args.max_ho_shell,
    )
    label_results = []
    for label in args.labels:
        vmc_entry = vmc_summary["ground_states"][label]
        label_results.append(
            compare_label(
                label=label,
                vmc_entry=vmc_entry,
                sweep_cfg=sweep_cfg,
                n_sp_choices=args.n_sp_states,
                n_ci_choices=args.n_ci_compute,
            )
        )

    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "purpose": "Grid-based CI validation for VMC dot-label entanglement using shared and one-per-well references.",
        "caveats": {
            "shared": (
                "Shared exact-diag uses the soft-min double-well Hamiltonian and the same real-space dot projection "
                "observable as the VMC pipeline, but it does not enforce the one-per-well occupancy structure of the VMC ansatz."
            ),
            "one_per_well": (
                "One-per-well exact-diag enforces left/right channel occupancy, but its one-body basis is built from separate localized harmonic wells rather than the shared soft-min potential used in training."
            ),
        },
        "sweep_config": asdict(sweep_cfg),
        "labels": args.labels,
        "n_sp_states": args.n_sp_states,
        "n_ci_compute": args.n_ci_compute,
        "results": label_results,
    }
    out_path = choose_output_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Wrote CI/VMC comparison summary to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())