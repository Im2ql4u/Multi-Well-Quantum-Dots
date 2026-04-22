from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_one_per_well_multi_magnetic_reference.py"
_SPEC = importlib.util.spec_from_file_location("run_one_per_well_multi_magnetic_reference", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {_MODULE_PATH}.")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_jobs = _MODULE.build_jobs
eigenvector_to_tensor = _MODULE.eigenvector_to_tensor
summarise_transition = _MODULE.summarise_transition


def test_build_jobs_crosses_wells_and_fields() -> None:
    jobs = build_jobs([3, 4], sep=4.0, b_pre=0.0, b_post_values=[0.0, 0.5])

    assert len(jobs) == 4
    assert jobs[0].n_wells == 3
    assert jobs[-1].n_wells == 4
    assert jobs[-1].b_post == 0.5


def test_eigenvector_to_tensor_places_coefficients_in_selected_basis() -> None:
    basis = [(0.0, (0, 1, 0)), (0.1, (1, 0, 1))]
    vec = np.array([0.6, -0.8])

    tensor = eigenvector_to_tensor(vec, basis, n_sp_states=2, n_wells=3)

    assert abs(tensor[0, 1, 0] - 0.6) < 1e-12
    assert abs(tensor[1, 0, 1] + 0.8) < 1e-12
    assert abs(np.sum(np.abs(tensor)) - 1.4) < 1e-12


def test_summarise_transition_detects_trivial_uniform_field() -> None:
    pre = {
        "E0": 3.2,
        "gap": 0.4,
        "partition_axes": [0],
        "entanglement": {"entropy": 0.6, "negativity": 0.5},
        "gs_vector": np.array([1.0, 0.0]),
    }
    post = {
        "E0": 3.7,
        "gap": 0.4,
        "partition_axes": [0],
        "entanglement": {"entropy": 0.6, "negativity": 0.5},
        "gs_vector": np.array([-1.0, 0.0]),
    }

    class Cfg:
        g_factor = 2.0
        mu_b = 1.0

    summary = summarise_transition(pre, post, Cfg(), b_pre=0.0, b_post=0.5)

    assert summary["trivial_uniform_b"] is True
    assert abs(summary["ground_state_overlap_abs"] - 1.0) < 1e-12
    assert abs(summary["actual_energy_shift"] - 0.5) < 1e-12