#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from exact_diag_double_dot import (
    DiagConfig,
    build_2d_dvr,
    build_centered_harmonic_potential_matrix,
    compute_entanglement_one_per_well,
    infer_box_half_widths,
    run_exact_diagonalization_one_per_well_multi,
    select_low_energy_product_basis,
    single_particle_eigenstates,
    well_centers_linear,
)
from observables.entanglement import compute_block_partition_entanglement


LOG = logging.getLogger("run_one_per_well_multi_magnetic_reference")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diag_sweeps"


@dataclass(frozen=True)
class SweepJob:
    n_wells: int
    sep: float
    b_pre: float
    b_post: float


def parse_csv_ints(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_csv_floats(text: str) -> list[float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def build_jobs(n_wells_values: list[int], sep: float, b_pre: float, b_post_values: list[float]) -> list[SweepJob]:
    return [SweepJob(n_wells=n_wells, sep=sep, b_pre=b_pre, b_post=b_post) for n_wells in n_wells_values for b_post in b_post_values]


def build_product_basis(cfg: DiagConfig) -> tuple[list[np.ndarray], list[tuple[float, tuple[int, ...]]]]:
    x_grid, y_grid, _w_x, _w_y, t2d = build_2d_dvr(
        nx=cfg.nx,
        ny=cfg.ny,
        x_half_width=cfg.x_half_width,
        y_half_width=cfg.y_half_width,
    )
    t2d = cfg.kinetic_prefactor * t2d
    centers = well_centers_linear(cfg.n_wells, cfg.sep)
    per_well_energies: list[np.ndarray] = []
    for center_x in centers:
        v_well = build_centered_harmonic_potential_matrix(
            x_grid=x_grid,
            y_grid=y_grid,
            omega=cfg.omega,
            center_x=center_x,
        )
        energies_w, _vecs_w = single_particle_eigenstates(t2d=t2d, v2d=v_well, n_sp_states=cfg.n_sp_states)
        per_well_energies.append(energies_w)
    basis = select_low_energy_product_basis(per_well_energies, cfg.n_ci_compute)
    return per_well_energies, basis


def eigenvector_to_tensor(eigvec: np.ndarray, basis: list[tuple[float, tuple[int, ...]]], n_sp_states: int, n_wells: int) -> np.ndarray:
    coeff_tensor = np.zeros((n_sp_states,) * n_wells, dtype=np.float64)
    for idx, (_energy, orb_tuple) in enumerate(basis):
        coeff_tensor[orb_tuple] = eigvec[idx]
    return coeff_tensor


def default_partition_axes(n_wells: int) -> list[int]:
    left_count = max(1, n_wells // 2)
    return list(range(left_count))


def compute_reference_state(cfg: DiagConfig) -> dict[str, Any]:
    eigvals, eigvecs, _merged_sp = run_exact_diagonalization_one_per_well_multi(cfg)
    _per_well_energies, basis = build_product_basis(cfg)
    gs_vec = eigvecs[:, 0]
    partition_axes = default_partition_axes(cfg.n_wells)
    coeff_tensor = eigenvector_to_tensor(gs_vec, basis, cfg.n_sp_states, cfg.n_wells)
    entanglement = compute_block_partition_entanglement(
        coeff_tensor,
        particle_weights=[np.ones(cfg.n_sp_states, dtype=np.float64) for _ in range(cfg.n_wells)],
        subsystem_axes=partition_axes,
    )
    return {
        "E0": float(eigvals[0]),
        "gap": float(eigvals[1] - eigvals[0]) if eigvals.size > 1 else None,
        "eigenvalues": eigvals[:8].tolist(),
        "partition_axes": partition_axes,
        "entanglement": {
            "entropy": entanglement["von_neumann_entropy"],
            "purity": entanglement["purity"],
            "negativity": entanglement["negativity"],
            "log_negativity": entanglement["log_negativity"],
            "effective_rank": entanglement["effective_schmidt_rank"],
        },
        "gs_vector": gs_vec,
    }


def summarise_transition(pre: dict[str, Any], post: dict[str, Any], cfg: DiagConfig, b_pre: float, b_post: float) -> dict[str, Any]:
    overlap = float(abs(np.vdot(pre["gs_vector"], post["gs_vector"])))
    expected_shift = 0.5 * cfg.g_factor * cfg.mu_b * (b_post - b_pre)
    actual_shift = float(post["E0"] - pre["E0"])
    entropy_delta = float(post["entanglement"]["entropy"] - pre["entanglement"]["entropy"])
    negativity_delta = float(post["entanglement"]["negativity"] - pre["entanglement"]["negativity"])
    trivial_uniform_b = (
        abs(actual_shift - expected_shift) < 1e-10
        and overlap > 1.0 - 1e-10
        and abs(entropy_delta) < 1e-12
        and abs(negativity_delta) < 1e-12
    )
    return {
        "pre_energy": pre["E0"],
        "post_energy": post["E0"],
        "pre_gap": pre["gap"],
        "post_gap": post["gap"],
        "expected_energy_shift": expected_shift,
        "actual_energy_shift": actual_shift,
        "energy_shift_error": actual_shift - expected_shift,
        "ground_state_overlap_abs": overlap,
        "pre_entropy": pre["entanglement"]["entropy"],
        "post_entropy": post["entanglement"]["entropy"],
        "entropy_delta": entropy_delta,
        "pre_negativity": pre["entanglement"]["negativity"],
        "post_negativity": post["entanglement"]["negativity"],
        "negativity_delta": negativity_delta,
        "partition_axes": pre["partition_axes"],
        "trivial_uniform_b": trivial_uniform_b,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-per-well multiwell magnetic exact-diag reference sweeps.")
    parser.add_argument("--n-wells-values", required=True, help="Comma-separated n_wells values, for example '3,4'.")
    parser.add_argument("--sep", type=float, default=4.0, help="Linear spacing between neighboring wells.")
    parser.add_argument("--B-pre", type=float, default=0.0, help="Initial uniform magnetic field.")
    parser.add_argument("--B-post-values", required=True, help="Comma-separated post-field values.")
    parser.add_argument("--omega", type=float, default=1.0, help="Confinement frequency.")
    parser.add_argument("--smooth-t", type=float, default=0.2, help="Soft-min temperature.")
    parser.add_argument("--kappa", type=float, default=1.0, help="Coulomb strength.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Coulomb softening.")
    parser.add_argument("--nx", type=int, default=20, help="DVR x points.")
    parser.add_argument("--ny", type=int, default=20, help="DVR y points.")
    parser.add_argument("--n-sp-states", type=int, default=16, help="Single-particle states per well.")
    parser.add_argument("--n-ci-compute", type=int, default=200, help="CI basis truncation.")
    parser.add_argument("--g-factor", type=float, default=2.0, help="g-factor.")
    parser.add_argument("--mu-b", type=float, default=1.0, help="Bohr magneton scaling.")
    parser.add_argument("--kinetic-prefactor", type=float, default=0.5, help="Kinetic prefactor.")
    parser.add_argument("--output-json", required=True, help="Output summary JSON.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    n_wells_values = parse_csv_ints(args.n_wells_values)
    b_post_values = parse_csv_floats(args.B_post_values)
    jobs = build_jobs(n_wells_values, sep=args.sep, b_pre=args.B_pre, b_post_values=b_post_values)

    runs: list[dict[str, Any]] = []
    for job in jobs:
        x_half, y_half = infer_box_half_widths(job.sep, args.omega, n_wells=job.n_wells)
        pre_cfg = DiagConfig(
            nx=args.nx,
            ny=args.ny,
            sep=job.sep,
            omega=args.omega,
            smooth_t=args.smooth_t,
            kappa=args.kappa,
            epsilon=args.epsilon,
            n_sp_states=args.n_sp_states,
            n_ci_compute=args.n_ci_compute,
            b_field=job.b_pre,
            g_factor=args.g_factor,
            mu_b=args.mu_b,
            x_half_width=x_half,
            y_half_width=y_half,
            n_wells=job.n_wells,
            model_mode="one_per_well",
            confinement_mode="localized",
            kinetic_prefactor=args.kinetic_prefactor,
        )
        post_cfg = DiagConfig(**{**pre_cfg.__dict__, "b_field": job.b_post})

        LOG.info("Running one-per-well multiwell magnetic reference: n_wells=%d sep=%.3f B_pre=%.3f B_post=%.3f", job.n_wells, job.sep, job.b_pre, job.b_post)
        pre = compute_reference_state(pre_cfg)
        post = compute_reference_state(post_cfg)
        summary = summarise_transition(pre, post, pre_cfg, job.b_pre, job.b_post)
        runs.append({
            "n_wells": job.n_wells,
            "sep": job.sep,
            "B_pre": job.b_pre,
            "B_post": job.b_post,
            **summary,
        })

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sweep_config": {
            "n_wells_values": n_wells_values,
            "sep": args.sep,
            "B_pre": args.B_pre,
            "B_post_values": b_post_values,
            "omega": args.omega,
            "smooth_t": args.smooth_t,
            "kappa": args.kappa,
            "epsilon": args.epsilon,
            "nx": args.nx,
            "ny": args.ny,
            "n_sp_states": args.n_sp_states,
            "n_ci_compute": args.n_ci_compute,
            "kinetic_prefactor": args.kinetic_prefactor,
        },
        "runs": runs,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOG.info("Saved one-per-well multiwell magnetic reference to %s", output_path)


if __name__ == "__main__":
    main()
