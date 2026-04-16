from __future__ import annotations

from typing import Any

import numpy as np

from config import SystemConfig
from observables.entanglement import compute_dot_projected_entanglement, compute_particle_entanglement


def build_shared_orbital_coefficient_matrix(
    eigvec: np.ndarray,
    slater_basis: list[tuple[int, int, str, str]],
    n_orbitals: int,
    n_ci: int | None = None,
) -> np.ndarray:
    """Build the shared-model spatial coefficient matrix A[a,b].

    The returned matrix satisfies

        Psi(r1, r2) = sum_{a,b} A[a,b] phi_a(r1) phi_b(r2)

    for the spatial part of the CI eigenstate. Singlet sectors map to a symmetric
    coefficient matrix and triplet sectors to an antisymmetric one.
    """
    if eigvec.ndim != 1:
        raise ValueError(f"Expected 1D eigvec, got shape {eigvec.shape}.")

    n_terms = len(slater_basis) if n_ci is None else min(int(n_ci), len(slater_basis))
    if eigvec.shape[0] < n_terms:
        raise ValueError(
            f"eigvec length {eigvec.shape[0]} is smaller than requested n_terms={n_terms}."
        )

    amplitude_matrix = np.zeros((n_orbitals, n_orbitals), dtype=np.float64)
    for idx in range(n_terms):
        a, b, _spin_cfg, spin_type = slater_basis[idx]
        coeff = float(eigvec[idx])
        if a == b:
            amplitude_matrix[a, a] += coeff
        elif spin_type == "singlet":
            amplitude_matrix[a, b] += coeff / np.sqrt(2.0)
            amplitude_matrix[b, a] += coeff / np.sqrt(2.0)
        elif spin_type.startswith("triplet"):
            amplitude_matrix[a, b] += coeff / np.sqrt(2.0)
            amplitude_matrix[b, a] -= coeff / np.sqrt(2.0)
        else:
            raise ValueError(f"Unsupported spin_type '{spin_type}'.")

    return amplitude_matrix


def build_real_space_wavefunction_matrix(
    single_particle_vectors: np.ndarray,
    orbital_coefficients: np.ndarray,
) -> np.ndarray:
    """Project orbital-basis CI coefficients onto a real-space grid basis."""
    if single_particle_vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D single_particle_vectors, got shape {single_particle_vectors.shape}."
        )
    if orbital_coefficients.ndim != 2:
        raise ValueError(
            f"Expected 2D orbital_coefficients, got shape {orbital_coefficients.shape}."
        )
    if single_particle_vectors.shape[1] != orbital_coefficients.shape[0]:
        raise ValueError(
            "single_particle_vectors and orbital_coefficients have incompatible shapes: "
            f"{single_particle_vectors.shape} vs {orbital_coefficients.shape}."
        )
    if orbital_coefficients.shape[0] != orbital_coefficients.shape[1]:
        raise ValueError("orbital_coefficients must be square.")

    return single_particle_vectors @ orbital_coefficients @ single_particle_vectors.T


def build_one_per_well_orbital_coefficient_matrix(
    eigvec: np.ndarray,
    product_basis: list[tuple[float, int, int]],
    n_left_orbitals: int,
    n_right_orbitals: int,
    n_ci: int | None = None,
) -> np.ndarray:
    """Build the one-per-well coefficient matrix C[a_left, b_right]."""
    if eigvec.ndim != 1:
        raise ValueError(f"Expected 1D eigvec, got shape {eigvec.shape}.")

    n_terms = len(product_basis) if n_ci is None else min(int(n_ci), len(product_basis))
    if eigvec.shape[0] < n_terms:
        raise ValueError(
            f"eigvec length {eigvec.shape[0]} is smaller than requested n_terms={n_terms}."
        )

    amplitude_matrix = np.zeros((n_left_orbitals, n_right_orbitals), dtype=np.float64)
    for idx in range(n_terms):
        _energy, left_orbital, right_orbital = product_basis[idx]
        amplitude_matrix[left_orbital, right_orbital] = float(eigvec[idx])
    return amplitude_matrix


def build_bipartite_real_space_wavefunction_matrix(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    orbital_coefficients: np.ndarray,
) -> np.ndarray:
    """Project asymmetric left/right orbital coefficients onto a real-space grid basis."""
    if left_vectors.ndim != 2 or right_vectors.ndim != 2:
        raise ValueError(
            "left_vectors and right_vectors must both be 2D arrays: "
            f"{left_vectors.shape}, {right_vectors.shape}."
        )
    if orbital_coefficients.ndim != 2:
        raise ValueError(
            f"Expected 2D orbital_coefficients, got shape {orbital_coefficients.shape}."
        )
    if left_vectors.shape[1] != orbital_coefficients.shape[0]:
        raise ValueError(
            "left_vectors and orbital_coefficients have incompatible shapes: "
            f"{left_vectors.shape} vs {orbital_coefficients.shape}."
        )
    if right_vectors.shape[1] != orbital_coefficients.shape[1]:
        raise ValueError(
            "right_vectors and orbital_coefficients have incompatible shapes: "
            f"{right_vectors.shape} vs {orbital_coefficients.shape}."
        )

    return left_vectors @ orbital_coefficients @ right_vectors.T


def build_dvr_points_and_weights(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    w_x: np.ndarray,
    w_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a 2D DVR grid into point coordinates and quadrature weights."""
    x2d, y2d = np.meshgrid(x_grid, y_grid, indexing="ij")
    wx2d, wy2d = np.meshgrid(w_x, w_y, indexing="ij")
    points = np.stack([x2d.ravel(), y2d.ravel()], axis=1)
    weights = (wx2d * wy2d).ravel()
    return points, weights


def compute_shared_ci_grid_entanglement(
    single_particle_vectors: np.ndarray,
    orbital_coefficients: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    system: SystemConfig,
    *,
    projection_basis: str = "localized_ho",
    max_ho_shell: int = 2,
) -> dict[str, Any]:
    """Evaluate particle and dot entanglement for a shared-model CI wavefunction on a grid."""
    psi_matrix = build_real_space_wavefunction_matrix(single_particle_vectors, orbital_coefficients)
    particle = compute_particle_entanglement(psi_matrix, weights, weights)
    dot = compute_dot_projected_entanglement(
        psi_matrix,
        points,
        points,
        weights,
        weights,
        system,
        projection_basis=projection_basis,
        max_ho_shell=max_ho_shell,
    )
    return {
        "particle_entanglement": particle,
        "dot_projected_entanglement": dot,
    }


def compute_one_per_well_ci_grid_entanglement(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    orbital_coefficients: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    system: SystemConfig,
    *,
    projection_basis: str = "localized_ho",
    max_ho_shell: int = 2,
) -> dict[str, Any]:
    """Evaluate particle and dot entanglement for a one-per-well CI wavefunction on a grid."""
    psi_matrix = build_bipartite_real_space_wavefunction_matrix(
        left_vectors=left_vectors,
        right_vectors=right_vectors,
        orbital_coefficients=orbital_coefficients,
    )
    particle = compute_particle_entanglement(psi_matrix, weights, weights)
    dot = compute_dot_projected_entanglement(
        psi_matrix,
        points,
        points,
        weights,
        weights,
        system,
        projection_basis=projection_basis,
        max_ho_shell=max_ho_shell,
    )
    return {
        "particle_entanglement": particle,
        "dot_projected_entanglement": dot,
    }