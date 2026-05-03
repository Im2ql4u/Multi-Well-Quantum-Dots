"""Unit tests for the structural quantum-number assertions (Phase 0.1)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from training.symmetry_asserts import (
    SymmetryViolationError,
    assert_quantum_numbers_consistent,
)


class _DummyWell:
    def __init__(self, center: tuple[float, ...], n_particles: int = 1) -> None:
        self.center = center
        self.n_particles = n_particles


class _DummySystem:
    def __init__(self, n_particles: int, wells: int = 2) -> None:
        self.n_particles = n_particles
        self.wells = [_DummyWell((float(i), 0.0)) for i in range(wells)]
        self.dim = 2
        self.omega = 1.0


class _DummyModel(torch.nn.Module):
    def __init__(self, spin_pattern: list[int], multi_ref: bool = False) -> None:
        super().__init__()
        spin = torch.tensor(spin_pattern, dtype=torch.long)
        self.register_buffer("spin_template", spin, persistent=False)
        self.multi_ref = multi_ref


def _good_params(n_up: int, n_down: int) -> dict:
    pattern = [0] * n_up + [1] * n_down
    return {
        "n_up": n_up,
        "n_down": n_down,
        "n_particles": n_up + n_down,
        "spin_pattern": pattern,
    }


def test_passes_on_consistent_n2_singlet() -> None:
    system = _DummySystem(n_particles=2)
    params = _good_params(1, 1)
    model = _DummyModel(params["spin_pattern"])
    diag = assert_quantum_numbers_consistent(
        model=model, system=system, params=params, context="unit"
    )
    assert diag["n_up"] == 1
    assert diag["n_down"] == 1
    assert diag["expected_Sz"] == pytest.approx(0.0)
    assert diag["checks"]["count_consistency"] == "ok"


def test_passes_on_n4_4up0down_polarised() -> None:
    system = _DummySystem(n_particles=4, wells=4)
    params = _good_params(4, 0)
    model = _DummyModel(params["spin_pattern"])
    diag = assert_quantum_numbers_consistent(
        model=model, system=system, params=params, context="unit"
    )
    assert diag["expected_Sz"] == pytest.approx(2.0)


def test_raises_when_counts_dont_sum_to_particles() -> None:
    system = _DummySystem(n_particles=4)
    params = _good_params(1, 1)  # n_up+n_down = 2, system has 4
    model = _DummyModel(params["spin_pattern"])
    with pytest.raises(SymmetryViolationError, match="system.n_particles"):
        assert_quantum_numbers_consistent(
            model=model, system=system, params=params, context="unit"
        )


def test_raises_when_spin_pattern_wrong_length() -> None:
    system = _DummySystem(n_particles=4, wells=4)
    params = _good_params(2, 2)
    params["spin_pattern"] = [0, 1, 0]  # length mismatch
    model = _DummyModel(params["spin_pattern"])
    with pytest.raises(SymmetryViolationError, match="len\\(spin_pattern\\)"):
        assert_quantum_numbers_consistent(
            model=model, system=system, params=params, context="unit"
        )


def test_raises_when_pattern_counts_disagree_with_params() -> None:
    system = _DummySystem(n_particles=4, wells=4)
    params = _good_params(2, 2)
    params["spin_pattern"] = [0, 0, 0, 1]  # 3 up, 1 down — disagrees with (2,2)
    model = _DummyModel(params["spin_pattern"])
    with pytest.raises(SymmetryViolationError, match="counts"):
        assert_quantum_numbers_consistent(
            model=model, system=system, params=params, context="unit"
        )


def test_raises_when_pattern_has_invalid_value() -> None:
    system = _DummySystem(n_particles=2)
    params = _good_params(1, 1)
    params["spin_pattern"] = [0, 2]  # 2 not in {0,1}
    model = _DummyModel([0, 1])  # model still valid
    with pytest.raises(SymmetryViolationError, match="outside"):
        assert_quantum_numbers_consistent(
            model=model, system=system, params=params, context="unit"
        )


def test_raises_when_model_template_drifts_from_params() -> None:
    system = _DummySystem(n_particles=2)
    params = _good_params(1, 1)
    model = _DummyModel([1, 0])  # different ordering than params [0, 1]
    with pytest.raises(SymmetryViolationError, match="differs from"):
        assert_quantum_numbers_consistent(
            model=model, system=system, params=params, context="unit"
        )


def test_passes_without_model_for_audit_use_case() -> None:
    system = _DummySystem(n_particles=4, wells=4)
    params = _good_params(3, 1)
    diag = assert_quantum_numbers_consistent(
        model=None, system=system, params=params, context="audit"
    )
    assert diag["expected_Sz"] == pytest.approx(1.0)


def test_multi_ref_consistent_template_passes() -> None:
    system = _DummySystem(n_particles=8, wells=8)
    params = _good_params(4, 4)
    model = _DummyModel(params["spin_pattern"], multi_ref=True)
    diag = assert_quantum_numbers_consistent(
        model=model, system=system, params=params, context="unit"
    )
    assert diag["multi_ref"] is True
    assert diag["checks"]["multi_ref_consistency"] == "ok"


def test_sz_residual_within_tolerance() -> None:
    system = _DummySystem(n_particles=8, wells=8)
    params = _good_params(5, 3)  # Sz = +1
    model = _DummyModel(params["spin_pattern"])
    diag = assert_quantum_numbers_consistent(
        model=model, system=system, params=params, context="unit"
    )
    assert diag["expected_Sz"] == pytest.approx(1.0)
    assert abs(diag["checks"]["sz_residual"]) < 1e-12
