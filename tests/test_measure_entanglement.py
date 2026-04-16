from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "measure_entanglement.py"
_SPEC = importlib.util.spec_from_file_location("measure_entanglement", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {_MODULE_PATH}.")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
evaluate_psi_matrix = _MODULE.evaluate_psi_matrix


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