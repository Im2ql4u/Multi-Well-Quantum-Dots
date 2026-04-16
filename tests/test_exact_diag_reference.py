from __future__ import annotations

import numpy as np

from observables.exact_diag_reference import (
    build_one_per_well_orbital_coefficient_matrix,
    build_shared_orbital_coefficient_matrix,
)


def test_build_shared_orbital_coefficient_matrix_maps_singlet_symmetrically() -> None:
    eigvec = np.array([1.0], dtype=np.float64)
    slater_basis = [(0, 1, "triplet_0", "singlet")]

    amplitude_matrix = build_shared_orbital_coefficient_matrix(
        eigvec=eigvec,
        slater_basis=slater_basis,
        n_orbitals=2,
    )

    expected = np.array(
        [
            [0.0, 1.0 / np.sqrt(2.0)],
            [1.0 / np.sqrt(2.0), 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(amplitude_matrix, expected)


def test_build_shared_orbital_coefficient_matrix_maps_triplet_antisymmetrically() -> None:
    eigvec = np.array([1.0], dtype=np.float64)
    slater_basis = [(0, 1, "triplet_0", "triplet_0")]

    amplitude_matrix = build_shared_orbital_coefficient_matrix(
        eigvec=eigvec,
        slater_basis=slater_basis,
        n_orbitals=2,
    )

    expected = np.array(
        [
            [0.0, 1.0 / np.sqrt(2.0)],
            [-1.0 / np.sqrt(2.0), 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(amplitude_matrix, expected)


def test_build_shared_orbital_coefficient_matrix_keeps_double_occupancy_diagonal() -> None:
    eigvec = np.array([2.0], dtype=np.float64)
    slater_basis = [(1, 1, "ud", "singlet")]

    amplitude_matrix = build_shared_orbital_coefficient_matrix(
        eigvec=eigvec,
        slater_basis=slater_basis,
        n_orbitals=3,
    )

    expected = np.zeros((3, 3), dtype=np.float64)
    expected[1, 1] = 2.0
    np.testing.assert_allclose(amplitude_matrix, expected)


def test_build_one_per_well_orbital_coefficient_matrix_preserves_product_layout() -> None:
    eigvec = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    product_basis = [
        (0.0, 0, 1),
        (0.1, 1, 0),
        (0.2, 1, 2),
    ]

    amplitude_matrix = build_one_per_well_orbital_coefficient_matrix(
        eigvec=eigvec,
        product_basis=product_basis,
        n_left_orbitals=2,
        n_right_orbitals=3,
    )

    expected = np.array(
        [
            [0.0, 0.25, 0.0],
            [-0.5, 0.0, 0.75],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(amplitude_matrix, expected)