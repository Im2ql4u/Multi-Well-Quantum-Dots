from __future__ import annotations

import math

import numpy as np

from config import SystemConfig
from observables.entanglement import compute_dot_projected_entanglement


def _toy_points() -> np.ndarray:
    return np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)


def _box_grid(
    nx: int = 101,
    ny: int = 81,
    x_half_width: float = 6.0,
    y_half_width: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-x_half_width, x_half_width, nx)
    y = np.linspace(-y_half_width, y_half_width, ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    weights = np.full(points.shape[0], dx * dy, dtype=np.float64)
    return points, weights


def _hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    coeffs = [0.0] * n + [1.0]
    return np.polynomial.hermite.hermval(x, coeffs)


def _ho_mode_2d(
    points: np.ndarray,
    center: tuple[float, float],
    omega: float,
    n_x: int,
    n_y: int,
) -> np.ndarray:
    x_shift = points[:, 0] - float(center[0])
    y_shift = points[:, 1] - float(center[1])
    xi_x = math.sqrt(omega) * x_shift
    xi_y = math.sqrt(omega) * y_shift
    norm_x = (omega / np.pi) ** 0.25 / math.sqrt((2.0**n_x) * math.factorial(n_x))
    norm_y = (omega / np.pi) ** 0.25 / math.sqrt((2.0**n_y) * math.factorial(n_y))
    phi_x = norm_x * np.exp(-0.5 * omega * x_shift**2) * _hermite_polynomial(n_x, xi_x)
    phi_y = norm_y * np.exp(-0.5 * omega * y_shift**2) * _hermite_polynomial(n_y, xi_y)
    return phi_x * phi_y


def test_dot_projected_entanglement_product_state_is_separable() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=2.0, omega=1.0, dim=2)
    points = _toy_points()
    weights = np.ones(2, dtype=np.float64)
    psi_matrix = np.asarray(
        [
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = compute_dot_projected_entanglement(
        psi_matrix,
        points,
        points,
        weights,
        weights,
        system,
        projection_basis="region_average",
    )

    assert abs(result["projected_subspace_weight"] - 1.0) < 1e-12
    assert abs(result["von_neumann_entropy"] - 0.0) < 1e-12
    assert abs(result["negativity"] - 0.0) < 1e-12
    assert abs(result["direct_partial_transpose"]["negativity_direct"] - 0.0) < 1e-12
    assert abs(result["sector_probabilities"]["LR"] - 1.0) < 1e-12


def test_dot_projected_entanglement_bell_state_matches_known_values() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=2.0, omega=1.0, dim=2)
    points = _toy_points()
    weights = np.ones(2, dtype=np.float64)
    psi_matrix = np.asarray(
        [
            [1.0 / np.sqrt(2.0), 0.0],
            [0.0, 1.0 / np.sqrt(2.0)],
        ],
        dtype=np.float64,
    )

    result = compute_dot_projected_entanglement(
        psi_matrix,
        points,
        points,
        weights,
        weights,
        system,
        projection_basis="region_average",
    )

    assert abs(result["projected_subspace_weight"] - 1.0) < 1e-12
    assert abs(result["von_neumann_entropy"] - np.log(2.0)) < 1e-12
    assert abs(result["negativity"] - 0.5) < 1e-12
    assert abs(result["log_negativity"] - 1.0) < 1e-12
    assert abs(result["direct_partial_transpose"]["negativity_direct"] - 0.5) < 1e-12
    assert result["direct_partial_transpose"]["n_negative_eigenvalues"] == 1


def test_localized_ho_projection_bell_state_matches_known_values() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=8.0, omega=1.0, dim=2)
    points, weights = _box_grid()
    left_center = tuple(system.wells[0].center)
    right_center = tuple(system.wells[1].center)
    phi_left = _ho_mode_2d(points, left_center, omega=1.0, n_x=0, n_y=0)
    phi_right = _ho_mode_2d(points, right_center, omega=1.0, n_x=0, n_y=0)
    psi_matrix = (
        np.outer(phi_left, phi_right) + np.outer(phi_right, phi_left)
    ) / np.sqrt(2.0)

    result = compute_dot_projected_entanglement(
        psi_matrix,
        points,
        points,
        weights,
        weights,
        system,
        projection_basis="localized_ho",
        max_ho_shell=0,
    )

    assert result["projected_subspace_weight"] > 0.99
    assert abs(result["von_neumann_entropy"] - np.log(2.0)) < 5e-2
    assert abs(result["negativity"] - 0.5) < 5e-2
    assert abs(result["direct_partial_transpose"]["negativity_direct"] - 0.5) < 5e-2


def test_localized_ho_projection_captures_excited_product_state() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=8.0, omega=1.0, dim=2)
    points, weights = _box_grid()
    left_center = tuple(system.wells[0].center)
    right_center = tuple(system.wells[1].center)
    phi_left_excited = _ho_mode_2d(points, left_center, omega=1.0, n_x=1, n_y=0)
    phi_right_ground = _ho_mode_2d(points, right_center, omega=1.0, n_x=0, n_y=0)
    psi_matrix = np.outer(phi_left_excited, phi_right_ground)

    localized_result = compute_dot_projected_entanglement(
        psi_matrix,
        points,
        points,
        weights,
        weights,
        system,
        projection_basis="localized_ho",
        max_ho_shell=1,
    )
    region_result = compute_dot_projected_entanglement(
        psi_matrix,
        points,
        points,
        weights,
        weights,
        system,
        projection_basis="region_average",
    )

    assert localized_result["projected_subspace_weight"] > 0.95
    assert abs(localized_result["negativity"]) < 1e-6
    assert localized_result["projected_subspace_weight"] > region_result["projected_subspace_weight"] + 0.25