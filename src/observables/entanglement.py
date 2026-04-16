from __future__ import annotations

import math
from typing import Any

import numpy as np

from config import SystemConfig


_ENTANGLEMENT_EPS = 1e-15


def _schmidt_metrics_from_sigma(sigma: np.ndarray) -> dict[str, Any]:
    norm_sq = float(np.sum(np.abs(sigma) ** 2))
    if norm_sq < _ENTANGLEMENT_EPS:
        return {
            "von_neumann_entropy": 0.0,
            "purity": 1.0,
            "linear_entropy": 0.0,
            "n_schmidt_terms": int(sigma.size),
            "effective_schmidt_rank": 0,
            "negativity": 0.0,
            "log_negativity": 0.0,
            "schmidt_values_top10": [],
            "schmidt_probs_top10": [],
        }

    sigma_norm = np.abs(sigma) / np.sqrt(norm_sq)
    probs = sigma_norm**2
    safe_probs = probs[probs > _ENTANGLEMENT_EPS]
    entropy = float(-np.sum(safe_probs * np.log(safe_probs)))
    purity = float(np.sum(probs**2))
    sigma_sum = float(np.sum(sigma_norm))
    negativity = float((sigma_sum**2 - 1.0) / 2.0)
    log_negativity = float(np.log2(2.0 * negativity + 1.0)) if negativity > _ENTANGLEMENT_EPS else 0.0
    effective_rank = int(np.sum(probs > 0.01 * probs[0])) if probs.size else 0
    return {
        "von_neumann_entropy": entropy,
        "purity": purity,
        "linear_entropy": 1.0 - purity,
        "n_schmidt_terms": int(sigma.size),
        "effective_schmidt_rank": effective_rank,
        "negativity": negativity,
        "log_negativity": log_negativity,
        "schmidt_values_top10": sigma_norm[:10].tolist(),
        "schmidt_probs_top10": probs[:10].tolist(),
    }


def build_weighted_wavefunction_matrix(
    psi_matrix: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> dict[str, Any]:
    """Return the normalized quadrature-weighted coefficient matrix."""
    norm2 = float(np.einsum("i,ij,j->", w1, np.abs(psi_matrix) ** 2, w2))
    if norm2 < 1e-30:
        raise RuntimeError("Wavefunction norm is essentially zero on the grid. Check grid coverage.")

    psi_norm = psi_matrix / np.sqrt(norm2)
    sqrt_w1 = np.sqrt(np.maximum(w1, 0.0))
    sqrt_w2 = np.sqrt(np.maximum(w2, 0.0))
    weighted_matrix = sqrt_w1[:, None] * psi_norm * sqrt_w2[None, :]
    norm_M = float(np.sum(np.abs(weighted_matrix) ** 2))
    return {
        "norm2_before_normalisation": norm2,
        "norm_M_squared": norm_M,
        "weighted_matrix": weighted_matrix,
    }


def compute_particle_entanglement(
    psi_matrix: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> dict[str, Any]:
    prepared = build_weighted_wavefunction_matrix(psi_matrix, w1, w2)
    sigma = np.linalg.svd(prepared["weighted_matrix"], compute_uv=False, full_matrices=False)
    return {
        "norm2_before_normalisation": prepared["norm2_before_normalisation"],
        "norm_M_squared": prepared["norm_M_squared"],
        **_schmidt_metrics_from_sigma(sigma),
    }


def _nearest_well_assignments(points: np.ndarray, system: SystemConfig) -> np.ndarray:
    if len(system.wells) != 2:
        raise ValueError("Dot-projected entanglement currently requires exactly two wells.")
    centers = np.asarray([well.center for well in system.wells], dtype=np.float64)
    dist_sq = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(dist_sq, axis=1)


def _hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    coeffs = [0.0] * n + [1.0]
    return np.polynomial.hermite.hermval(x, coeffs)


def _harmonic_oscillator_mode_2d(
    points: np.ndarray,
    center: np.ndarray,
    omega: float,
    n_x: int,
    n_y: int,
) -> np.ndarray:
    if points.shape[1] != 2:
        raise ValueError("Localized HO dot projection currently requires 2D points.")

    x_shift = points[:, 0] - float(center[0])
    y_shift = points[:, 1] - float(center[1])
    sqrt_omega = math.sqrt(float(omega))
    xi_x = sqrt_omega * x_shift
    xi_y = sqrt_omega * y_shift

    h_x = _hermite_polynomial(n_x, xi_x)
    h_y = _hermite_polynomial(n_y, xi_y)
    norm_x = (float(omega) / math.pi) ** 0.25 / math.sqrt((2.0**n_x) * math.factorial(n_x))
    norm_y = (float(omega) / math.pi) ** 0.25 / math.sqrt((2.0**n_y) * math.factorial(n_y))
    phi_x = norm_x * np.exp(-0.5 * float(omega) * x_shift**2) * h_x
    phi_y = norm_y * np.exp(-0.5 * float(omega) * y_shift**2) * h_y
    return phi_x * phi_y


def _enumerate_2d_ho_modes(max_shell: int) -> list[tuple[int, int]]:
    if max_shell < 0:
        raise ValueError(f"max_shell must be non-negative, got {max_shell}.")

    modes: list[tuple[int, int]] = []
    for shell in range(max_shell + 1):
        for n_x in range(shell + 1):
            modes.append((n_x, shell - n_x))
    return modes


def _orthonormalize_columns(columns: np.ndarray) -> tuple[np.ndarray, int]:
    if columns.ndim != 2:
        raise ValueError(f"Expected 2D columns array, got shape {columns.shape}.")
    if columns.shape[1] == 0:
        raise ValueError("Cannot orthonormalize an empty column set.")

    q, r = np.linalg.qr(columns, mode="reduced")
    diag = np.abs(np.diag(r)) if r.ndim == 2 else np.asarray([], dtype=np.float64)
    if diag.size == 0:
        raise RuntimeError("QR orthonormalization failed to produce any basis vectors.")

    threshold = max(_ENTANGLEMENT_EPS, 1e-10 * float(np.max(diag)))
    rank = int(np.sum(diag > threshold))
    if rank == 0:
        raise RuntimeError("Projection basis is rank-deficient on the supplied quadrature grid.")
    return q[:, :rank], rank


def _build_region_average_basis(
    points: np.ndarray,
    weights: np.ndarray,
    system: SystemConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    assignments = _nearest_well_assignments(points, system)
    basis = np.zeros((points.shape[0], 2), dtype=np.float64)
    region_measures: list[float] = []
    point_counts: list[int] = []

    for well_idx in range(2):
        mask = assignments == well_idx
        measure = float(np.sum(weights[mask]))
        if measure <= _ENTANGLEMENT_EPS:
            raise RuntimeError(f"Region {well_idx} has zero quadrature weight; cannot build dot basis.")
        basis[mask, well_idx] = np.sqrt(weights[mask] / measure)
        region_measures.append(measure)
        point_counts.append(int(np.sum(mask)))

    return basis, np.asarray([0, 1], dtype=np.int64), assignments, {
        "basis_family": "nearest_well_region_average",
        "n_basis_total": 2,
        "region_measures": region_measures,
        "region_point_counts": point_counts,
        "well_centers": [list(w.center) for w in system.wells],
    }


def _build_localized_ho_basis(
    points: np.ndarray,
    weights: np.ndarray,
    system: SystemConfig,
    max_ho_shell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a Löwdin-orthogonalized HO basis spanning both wells.

    Each HO function is evaluated globally (no Voronoi masking) so that
    cross-well overlap is captured correctly.  The full overlap matrix
    S_{ab} = integral phi_a phi_b is then symmetrically orthogonalized via
    S^{-1/2}, preserving the one-to-one correspondence between raw and
    orthonormalized functions.  Each Löwdin orbital inherits the well
    label of the raw HO function it was derived from, giving an orthonormal
    left/right partition that is accurate even when the orbital overlap
    exp(-d^2/4) is non-negligible (small well separations d <= 4).
    """
    assignments = _nearest_well_assignments(points, system)
    sqrt_weights = np.sqrt(np.maximum(weights, 0.0))
    modes = _enumerate_2d_ho_modes(max_ho_shell)

    # --- Step 1: evaluate all HO functions at ALL grid points (unmasked) ---
    raw_columns: list[np.ndarray] = []
    raw_well_indices: list[int] = []
    well_metadata: list[dict[str, Any]] = []

    for well_idx, well in enumerate(system.wells):
        per_well_raw: list[np.ndarray] = []
        for n_x, n_y in modes:
            phi = _harmonic_oscillator_mode_2d(
                points,
                center=np.asarray(well.center, dtype=np.float64),
                omega=float(well.omega),
                n_x=n_x,
                n_y=n_y,
            )
            col = sqrt_weights * phi  # sqrt-weight-scaled column for quadrature
            raw_columns.append(col)
            raw_well_indices.append(well_idx)
            per_well_raw.append(col)

        well_metadata.append(
            {
                "well_index": well_idx,
                "well_center": list(well.center),
                "raw_mode_labels": [[n_x, n_y] for n_x, n_y in modes],
                "n_raw_modes": len(modes),
            }
        )

    if not raw_columns:
        raise RuntimeError("Localized HO projection produced no basis functions.")

    raw_matrix = np.column_stack(raw_columns)  # (N_pts, N_raw)

    # --- Step 2: Löwdin orthogonalization via S^{-1/2} ---
    # S_{ab} = sum_i raw_a(i) raw_b(i)  [already includes sqrt-weight factors]
    S = raw_matrix.T @ raw_matrix

    eigvals_S, eigvecs_S = np.linalg.eigh(S)
    # Threshold: discard modes with eigenvalue < 1e-6 * max (numerically rank-deficient)
    eig_threshold = max(_ENTANGLEMENT_EPS, 1e-6 * float(np.max(eigvals_S)))
    valid_eig = eigvals_S > eig_threshold
    inv_sqrt_vals = np.where(valid_eig, 1.0 / np.sqrt(np.maximum(eigvals_S, eig_threshold)), 0.0)
    S_inv_half = (eigvecs_S * inv_sqrt_vals) @ eigvecs_S.T  # (N_raw, N_raw)

    lowdin_matrix = raw_matrix @ S_inv_half  # (N_pts, N_raw)

    # --- Step 3: keep only columns with non-negligible norm ---
    col_norms = np.linalg.norm(lowdin_matrix, axis=0)
    col_threshold = max(_ENTANGLEMENT_EPS, 1e-6)

    basis_columns: list[np.ndarray] = []
    basis_well_indices: list[int] = []
    n_kept_per_well = [0, 0]

    for col_idx, (norm, well_idx) in enumerate(zip(col_norms, raw_well_indices)):
        if norm > col_threshold:
            basis_columns.append(lowdin_matrix[:, col_idx])
            basis_well_indices.append(well_idx)
            n_kept_per_well[well_idx] += 1

    if not basis_columns:
        raise RuntimeError("Löwdin orthogonalization produced no valid basis functions.")

    for well_idx in range(len(system.wells)):
        well_metadata[well_idx]["n_orthonormal_modes"] = n_kept_per_well[well_idx]
        mask = assignments == well_idx
        well_metadata[well_idx]["region_measure"] = float(np.sum(weights[mask]))
        well_metadata[well_idx]["region_point_count"] = int(np.sum(mask))

    basis = np.column_stack(basis_columns)
    return basis, np.asarray(basis_well_indices, dtype=np.int64), assignments, {
        "basis_family": "lowdin_localized_ho",
        "max_ho_shell": max_ho_shell,
        "n_basis_total": int(basis.shape[1]),
        "overlap_matrix_rank": int(np.sum(valid_eig)),
        "well_metadata": well_metadata,
    }


def _build_projection_basis(
    points: np.ndarray,
    weights: np.ndarray,
    system: SystemConfig,
    projection_basis: str,
    max_ho_shell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if projection_basis == "region_average":
        return _build_region_average_basis(points, weights, system)
    if projection_basis == "localized_ho":
        return _build_localized_ho_basis(points, weights, system, max_ho_shell=max_ho_shell)
    raise ValueError(
        f"Unknown projection_basis '{projection_basis}'. Expected 'region_average' or 'localized_ho'."
    )


def _sector_probabilities(weighted_matrix: np.ndarray, assign1: np.ndarray, assign2: np.ndarray) -> dict[str, float]:
    labels = {0: "L", 1: "R"}
    probs: dict[str, float] = {}
    abs_sq = np.abs(weighted_matrix) ** 2
    for left_idx in range(2):
        mask1 = assign1 == left_idx
        for right_idx in range(2):
            mask2 = assign2 == right_idx
            probs[f"{labels[left_idx]}{labels[right_idx]}"] = float(np.sum(abs_sq[np.ix_(mask1, mask2)]))
    return probs


def _partial_transpose_metrics(amplitude_matrix: np.ndarray) -> dict[str, Any]:
    dims = amplitude_matrix.shape
    rho = np.einsum("ab,cd->abcd", amplitude_matrix, amplitude_matrix.conj())
    rho_pt = np.transpose(rho, (0, 3, 2, 1)).reshape(dims[0] * dims[1], dims[0] * dims[1])
    rho_pt = rho_pt / max(float(np.sum(np.abs(amplitude_matrix) ** 2)), _ENTANGLEMENT_EPS)
    rho_pt = 0.5 * (rho_pt + rho_pt.conj().T)
    eigvals = np.linalg.eigvalsh(rho_pt)
    negativity = float(np.sum((np.abs(eigvals) - eigvals) / 2.0))
    log_negativity = float(np.log2(2.0 * negativity + 1.0)) if negativity > _ENTANGLEMENT_EPS else 0.0
    neg_eigs = eigvals[eigvals < -1e-12]
    return {
        "negativity_direct": negativity,
        "log_negativity_direct": log_negativity,
        "min_eigenvalue": float(np.min(eigvals)),
        "n_negative_eigenvalues": int(neg_eigs.size),
        "negative_eigenvalues": neg_eigs.tolist(),
    }


def _density_matrix_partial_transpose_metrics(
    density_matrix: np.ndarray,
    dims: tuple[int, int],
) -> dict[str, Any]:
    rho = density_matrix.reshape(dims[0], dims[1], dims[0], dims[1])
    rho_pt = np.transpose(rho, (0, 3, 2, 1)).reshape(dims[0] * dims[1], dims[0] * dims[1])
    rho_pt = 0.5 * (rho_pt + rho_pt.conj().T)
    eigvals = np.linalg.eigvalsh(rho_pt)
    negativity = float(np.sum((np.abs(eigvals) - eigvals) / 2.0))
    log_negativity = float(np.log2(2.0 * negativity + 1.0)) if negativity > _ENTANGLEMENT_EPS else 0.0
    neg_eigs = eigvals[eigvals < -1e-12]
    return {
        "negativity": negativity,
        "log_negativity": log_negativity,
        "min_eigenvalue": float(np.min(eigvals)),
        "n_negative_eigenvalues": int(neg_eigs.size),
        "negative_eigenvalues": neg_eigs.tolist(),
    }


def _dot_label_density_matrix(
    amplitude_matrix: np.ndarray,
    basis1_well_indices: np.ndarray,
    basis2_well_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    row_groups = [np.flatnonzero(basis1_well_indices == well_idx) for well_idx in range(2)]
    col_groups = [np.flatnonzero(basis2_well_indices == well_idx) for well_idx in range(2)]
    rho_dot = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    labels = {0: "L", 1: "R"}
    projected_sector_probs: dict[str, float] = {}

    for left_idx in range(2):
        rows_left = row_groups[left_idx]
        for right_idx in range(2):
            cols_right = col_groups[right_idx]
            block_left = amplitude_matrix[np.ix_(rows_left, cols_right)]
            projected_sector_probs[f"{labels[left_idx]}{labels[right_idx]}"] = float(
                np.sum(np.abs(block_left) ** 2)
            )
            for left_idx_2 in range(2):
                rows_left_2 = row_groups[left_idx_2]
                for right_idx_2 in range(2):
                    cols_right_2 = col_groups[right_idx_2]
                    block_right = amplitude_matrix[np.ix_(rows_left_2, cols_right_2)]
                    rho_dot[left_idx, right_idx, left_idx_2, right_idx_2] = np.vdot(
                        block_right,
                        block_left,
                    )

    return rho_dot.reshape(4, 4), projected_sector_probs


def _serialise_matrix(matrix: np.ndarray) -> list[list[float]] | dict[str, list[list[float]]]:
    if np.max(np.abs(matrix.imag)) < 1e-12:
        return matrix.real.tolist()
    return {
        "real": matrix.real.tolist(),
        "imag": matrix.imag.tolist(),
    }


def compute_dot_projected_entanglement(
    psi_matrix: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    system: SystemConfig,
    projection_basis: str = "localized_ho",
    max_ho_shell: int = 1,
) -> dict[str, Any]:
    """Project the N=2 wavefunction onto a left/right localized basis.

    ``projection_basis='region_average'`` reproduces the original 2x2 coarse basis.
    ``projection_basis='localized_ho'`` uses a richer masked harmonic-oscillator
    basis inside each nearest-well region, which captures intradot structure while
    preserving the left/right dot partition.
    """
    prepared = build_weighted_wavefunction_matrix(psi_matrix, w1, w2)
    weighted_matrix = prepared["weighted_matrix"]

    basis1, basis1_wells, assign1, meta1 = _build_projection_basis(
        pts1,
        w1,
        system,
        projection_basis=projection_basis,
        max_ho_shell=max_ho_shell,
    )
    basis2, basis2_wells, assign2, meta2 = _build_projection_basis(
        pts2,
        w2,
        system,
        projection_basis=projection_basis,
        max_ho_shell=max_ho_shell,
    )
    amplitude_matrix = basis1.T @ weighted_matrix @ basis2
    projected_norm_sq = float(np.sum(np.abs(amplitude_matrix) ** 2))
    if projected_norm_sq < _ENTANGLEMENT_EPS:
        return {
            "projection_basis": meta1["basis_family"],
            "projection_basis_config": {
                "projection_basis": projection_basis,
                "max_ho_shell": max_ho_shell,
            },
            "projected_subspace_weight": projected_norm_sq,
            "sector_probabilities": _sector_probabilities(weighted_matrix, assign1, assign2),
            "projected_sector_probabilities": {"LL": 0.0, "LR": 0.0, "RL": 0.0, "RR": 0.0},
            "region_metadata": {
                "particle_1": meta1,
                "particle_2": meta2,
            },
            "basis_dimensions": {
                "particle_1": int(basis1.shape[1]),
                "particle_2": int(basis2.shape[1]),
            },
            "amplitude_matrix": np.zeros((basis1.shape[1], basis2.shape[1]), dtype=np.float64).tolist(),
            "direct_partial_transpose": {
                "negativity_direct": 0.0,
                "log_negativity_direct": 0.0,
                "min_eigenvalue": 0.0,
                "n_negative_eigenvalues": 0,
                "negative_eigenvalues": [],
            },
            "dot_label_partial_transpose": {
                "negativity": 0.0,
                "log_negativity": 0.0,
                "min_eigenvalue": 0.0,
                "n_negative_eigenvalues": 0,
                "negative_eigenvalues": [],
            },
            **_schmidt_metrics_from_sigma(np.asarray([], dtype=np.float64)),
        }

    amplitude_matrix = amplitude_matrix / np.sqrt(projected_norm_sq)
    sigma = np.linalg.svd(amplitude_matrix, compute_uv=False, full_matrices=False)
    schmidt_metrics = _schmidt_metrics_from_sigma(sigma)
    pt_metrics = _partial_transpose_metrics(amplitude_matrix)
    rho_dot, projected_sector_probs = _dot_label_density_matrix(amplitude_matrix, basis1_wells, basis2_wells)
    dot_pt_metrics = _density_matrix_partial_transpose_metrics(rho_dot, dims=(2, 2))
    return {
        "projection_basis": meta1["basis_family"],
        "projection_basis_config": {
            "projection_basis": projection_basis,
            "max_ho_shell": max_ho_shell,
        },
        "projected_subspace_weight": projected_norm_sq,
        "sector_probabilities": _sector_probabilities(weighted_matrix, assign1, assign2),
        "projected_sector_probabilities": projected_sector_probs,
        "region_metadata": {
            "particle_1": meta1,
            "particle_2": meta2,
        },
        "basis_dimensions": {
            "particle_1": int(basis1.shape[1]),
            "particle_2": int(basis2.shape[1]),
        },
        "amplitude_matrix": _serialise_matrix(amplitude_matrix),
        "direct_partial_transpose": pt_metrics,
        "dot_label_density_matrix": _serialise_matrix(rho_dot),
        "dot_label_partial_transpose": dot_pt_metrics,
        **schmidt_metrics,
    }