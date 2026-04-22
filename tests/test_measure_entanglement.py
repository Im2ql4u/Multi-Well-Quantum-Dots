from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import torch

from observables.entanglement import build_weighted_wavefunction_tensor
from config import SystemConfig


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "measure_entanglement.py"
_SPEC = importlib.util.spec_from_file_location("measure_entanglement", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {_MODULE_PATH}.")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
evaluate_psi_matrix = _MODULE.evaluate_psi_matrix
evaluate_psi_tensor = _MODULE.evaluate_psi_tensor
build_particle_local_grids = _MODULE.build_particle_local_grids


class _SignedToyModel(torch.nn.Module):
    def signed_log_psi(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sign = torch.sign(x[:, 0, 0] * x[:, 1, 0])
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        logabs = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        return sign, logabs


class _SlaterToyModel(torch.nn.Module):
    def signed_log_slater(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sign = torch.sign(x[:, 0, 0] + x[:, 1, 0])
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        logabs = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        return sign, logabs


class _SlaterView(torch.nn.Module):
    def __init__(self, wf: torch.nn.Module) -> None:
        super().__init__()
        self.wf = wf

    def signed_log_psi(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.wf.signed_log_slater(x)


class _SignedToyModel3(torch.nn.Module):
    def signed_log_psi(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sign = torch.sign(x[:, 0, 0] * x[:, 1, 0] * x[:, 2, 0])
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        logabs = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        return sign, logabs


def test_evaluate_psi_matrix_preserves_signed_amplitudes() -> None:
    model = _SignedToyModel()
    points = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    psi_matrix = evaluate_psi_matrix(
        model,
        points,
        points,
        device="cpu",
        batch_size=2,
    )

    expected = np.asarray(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(psi_matrix, expected)


def test_evaluate_psi_matrix_supports_slater_only_adapter() -> None:
    model = _SlaterView(_SlaterToyModel())
    points = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    psi_matrix = evaluate_psi_matrix(
        model,
        points,
        points,
        device="cpu",
        batch_size=2,
    )

    expected = np.asarray(
        [
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(psi_matrix, expected)


def test_evaluate_psi_tensor_preserves_signed_amplitudes_for_three_particles() -> None:
    model = _SignedToyModel3()
    points = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    psi_tensor = evaluate_psi_tensor(
        model,
        points,
        n_particles=3,
        device="cpu",
        batch_size=4,
    )

    expected = np.zeros((2, 2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected[i, j, k] = points[i, 0] * points[j, 0] * points[k, 0]
    expected = np.sign(expected)
    expected[expected == 0.0] = 1.0

    assert np.allclose(psi_tensor, expected)


def test_build_particle_local_grids_hermite_normalizes_exact_weight_matched_gaussian() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=8.0, omega=1.0, dim=2)

    for npts in (3, 4, 5):
        particle_points, particle_weights = build_particle_local_grids(
            system,
            npts=npts,
            grid_family="hermite_local",
        )
        points = particle_points[0]
        weights = particle_weights[0]
        center = np.asarray(system.wells[0].center, dtype=np.float64)
        displacement = points - center[None, :]
        radius_sq = np.sum(displacement**2, axis=1)
        psi = np.sqrt(system.omega / (2.0 * np.pi)) * np.exp(-0.25 * system.omega * radius_sq)

        prepared = build_weighted_wavefunction_tensor(psi, [weights])

        assert abs(prepared["norm2_before_normalisation"] - 1.0) < 1e-12
        assert abs(prepared["norm_tensor_squared"] - 1.0) < 1e-12