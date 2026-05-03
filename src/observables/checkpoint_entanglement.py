"""Compute bipartite entanglement metrics from a trained ``GroundStateWF`` checkpoint.

This module provides the bridge from a saved ``model.pt``/``config.yaml`` pair
to a ``dict`` of entanglement observables suitable for inverse-design targets.

For now only the ``N=2`` quadrature path is implemented; this is the case
needed to launch Phase 1A (inverse-design entanglement target on the N=2
double dot) and to validate the metric against the calibrated CI value of
``dot_label_negativity = 0.5`` in the deep Mott regime.

Design notes
------------
* Quadrature grid for the 2D-DVR is reused from
  ``scripts.exact_diag_double_dot.build_2d_dvr`` so that the projection basis
  exactly matches the CI calibration done in Phase 0A.
* Adaptive box sizing is borrowed from ``infer_box_half_widths``.
* All evaluations of ``psi`` are done in batches with ``torch.no_grad``;
  the model is automatically moved to the requested device.
* Default ``max_ho_shell=2`` matches the convention locked in Phase 0A.
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from observables.checkpoint_loader import LoadedWavefunction, load_wavefunction_from_dir
from observables.entanglement import (
    compute_dot_projected_entanglement,
    compute_particle_entanglement,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from exact_diag_double_dot import build_2d_dvr, infer_box_half_widths  # noqa: E402


LOGGER = logging.getLogger("checkpoint_entanglement")


@dataclass
class GridSpec:
    """Quadrature grid specification for a 2D wavefunction evaluation."""

    nx: int
    ny: int
    x_half_width: float
    y_half_width: float

    @classmethod
    def from_n2_system(
        cls,
        sep: float,
        omega: float,
        *,
        nx_per_unit: float = 1.4,
        ny_min: int = 16,
    ) -> "GridSpec":
        """Choose ``nx, ny`` so that ``dx ~ 1/(nx_per_unit)`` along the chain.

        For the standard ``omega=1`` / soft-min potential we want a per-bin
        spacing of ~0.7 Bohr, which gives ~1.4 points per unit length.
        """
        x_half, y_half = infer_box_half_widths(sep=sep, omega=omega, n_wells=2)
        nx = int(np.ceil(2.0 * x_half * nx_per_unit))
        ny = max(ny_min, int(np.ceil(2.0 * y_half * nx_per_unit)))
        return cls(nx=nx, ny=ny, x_half_width=x_half, y_half_width=y_half)


def _build_quadrature_arrays(grid: GridSpec):
    x_grid, y_grid, w_x, w_y, _t2d = build_2d_dvr(
        nx=grid.nx,
        ny=grid.ny,
        x_half_width=grid.x_half_width,
        y_half_width=grid.y_half_width,
    )
    x2d, y2d = np.meshgrid(x_grid, y_grid, indexing="ij")
    wx2d, wy2d = np.meshgrid(w_x, w_y, indexing="ij")
    points = np.stack([x2d.ravel(), y2d.ravel()], axis=1)
    weights = (wx2d * wy2d).ravel()
    return x_grid, y_grid, w_x, w_y, points, weights


def _evaluate_n2_psi_matrix(
    loaded: LoadedWavefunction,
    points: np.ndarray,
    *,
    batch_size: int = 16384,
) -> np.ndarray:
    """Evaluate ``psi(r_1, r_2)`` for all ``(r_1, r_2)`` on the product grid.

    Returns a complex128 array of shape ``(P, P)`` where ``P = points.shape[0]``.
    """
    P = points.shape[0]
    pts_t = torch.from_numpy(points).to(device=loaded.device, dtype=loaded.dtype)

    out = np.zeros((P, P), dtype=np.float64)

    total_pairs = P * P
    n_batches = (total_pairs + batch_size - 1) // batch_size

    flat_idx = 0
    for b in range(n_batches):
        start = b * batch_size
        stop = min(total_pairs, start + batch_size)

        i_idx = np.arange(start, stop) // P
        j_idx = np.arange(start, stop) % P

        x = torch.empty((stop - start, 2, 2), device=loaded.device, dtype=loaded.dtype)
        x[:, 0, :] = pts_t[i_idx]
        x[:, 1, :] = pts_t[j_idx]

        sign, logp = loaded.signed_log_psi(x)
        psi_vals = (sign * torch.exp(logp)).detach().cpu().numpy()

        out.flat[start:stop] = psi_vals
        flat_idx = stop

    if flat_idx != total_pairs:
        raise RuntimeError("Mismatch between filled entries and total pairs.")
    return out


def evaluate_n2_entanglement(
    result_dir: Path | str,
    *,
    grid: GridSpec | None = None,
    max_ho_shell: int = 2,
    projection_basis: str = "localized_ho",
    device: str | torch.device | None = None,
    batch_size: int = 16384,
) -> dict[str, Any]:
    """Compute particle and dot-label entanglement for an N=2 checkpoint.

    Parameters
    ----------
    result_dir
        Directory containing ``config.yaml`` + ``model.pt``.
    grid
        Optional explicit grid. If omitted, a sensible grid is built from the
        well separation found in the config.
    max_ho_shell
        HO-shell budget for the Löwdin projection. Default ``2`` matches the
        Phase 0A calibration convention.
    projection_basis
        Forwarded to ``compute_dot_projected_entanglement``.

    Returns
    -------
    dict
        A dictionary with the same shape as the existing
        ``compute_shared_ci_grid_entanglement`` payload, plus a
        ``settings`` block describing the evaluation.
    """
    loaded = load_wavefunction_from_dir(result_dir, device=device)

    if loaded.system.n_particles != 2:
        raise NotImplementedError(
            "evaluate_n2_entanglement only supports N=2 systems; got "
            f"N={loaded.system.n_particles}."
        )
    if len(loaded.system.wells) != 2:
        raise ValueError(
            "Dot-label entanglement requires exactly two wells; got "
            f"{len(loaded.system.wells)}."
        )

    centers = [w.center for w in loaded.system.wells]
    sep = float(math.hypot(centers[0][0] - centers[1][0], centers[0][1] - centers[1][1]))
    omega = float(loaded.system.wells[0].omega)

    if grid is None:
        grid = GridSpec.from_n2_system(sep=sep, omega=omega)

    x_grid, y_grid, w_x, w_y, points, weights = _build_quadrature_arrays(grid)

    LOGGER.info(
        "Evaluating psi on %d x %d grid (%d total points) for d=%.3f, omega=%.3f",
        grid.nx, grid.ny, points.shape[0], sep, omega,
    )

    psi_matrix = _evaluate_n2_psi_matrix(loaded, points, batch_size=batch_size)

    particle = compute_particle_entanglement(psi_matrix, weights, weights)
    dot = compute_dot_projected_entanglement(
        psi_matrix,
        points,
        points,
        weights,
        weights,
        loaded.system,
        projection_basis=projection_basis,
        max_ho_shell=max_ho_shell,
    )

    return {
        "result_dir": str(loaded.result_dir),
        "settings": {
            "grid_nx": grid.nx,
            "grid_ny": grid.ny,
            "grid_x_half_width": grid.x_half_width,
            "grid_y_half_width": grid.y_half_width,
            "n_grid_points": int(points.shape[0]),
            "max_ho_shell": int(max_ho_shell),
            "projection_basis": projection_basis,
            "device": str(loaded.device),
            "dtype": str(loaded.dtype),
            "well_separation": sep,
            "omega": omega,
        },
        "particle_entanglement": particle,
        "dot_projected_entanglement": dot,
    }


def entanglement_target_n2(
    result_dir: Path | str,
    *,
    metric: str = "dot_label_negativity",
    max_ho_shell: int = 2,
) -> float:
    """Reduce the N=2 entanglement payload to a single scalar target value.

    Available ``metric`` choices:
      - ``"dot_label_negativity"`` (default): textbook Bell-state metric.
      - ``"dot_label_log_negativity"``: log_2(2*neg + 1).
      - ``"dot_label_von_neumann_entropy"``: ``S_vN`` of the projected
        amplitude matrix.
      - ``"particle_von_neumann_entropy"``: ``S_vN`` from the full SVD.
    """
    payload = evaluate_n2_entanglement(result_dir, max_ho_shell=max_ho_shell)
    dot = payload["dot_projected_entanglement"]
    particle = payload["particle_entanglement"]

    if metric == "dot_label_negativity":
        return float(dot["dot_label_partial_transpose"]["negativity"])
    if metric == "dot_label_log_negativity":
        return float(dot["dot_label_partial_transpose"]["log_negativity"])
    if metric == "dot_label_von_neumann_entropy":
        return float(dot["von_neumann_entropy"])
    if metric == "particle_von_neumann_entropy":
        return float(particle["von_neumann_entropy"])
    raise ValueError(f"Unknown metric '{metric}'.")
