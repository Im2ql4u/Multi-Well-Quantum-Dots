from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_magnetic_reference_sweep.py"
_SPEC = importlib.util.spec_from_file_location("run_magnetic_reference_sweep", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {_MODULE_PATH}.")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
build_sweep_jobs = _MODULE.build_sweep_jobs
summarise_characterization = _MODULE.summarise_characterization


def test_build_sweep_jobs_crosses_separations_and_fields() -> None:
    jobs = build_sweep_jobs([2.0, 4.0], b_pre=0.0, b_post_values=[0.0, 0.5])

    assert len(jobs) == 4
    assert jobs[0].sep == 2.0
    assert jobs[0].b_pre == 0.0
    assert jobs[0].b_post == 0.0
    assert jobs[-1].sep == 4.0
    assert jobs[-1].b_post == 0.5


def test_summarise_characterization_extracts_key_magnetic_metrics() -> None:
    result = {
        "sep": 4.0,
        "B_pre": 0.0,
        "B_post": 0.5,
        "pre": {
            "E0": 2.6,
            "gap": 0.1,
            "dominant_spin": "singlet",
            "entanglement": {"entropy": 0.2},
            "partial_transpose": {"negativity": 0.25},
        },
        "post": {
            "E0": 2.3,
            "gap": 0.3,
            "dominant_spin": "triplet_m",
            "entanglement": {"entropy": 0.69},
            "partial_transpose": {"negativity": 0.5},
        },
    }

    summary = summarise_characterization(result)

    assert summary["sep"] == 4.0
    assert summary["spin_flip"] is True
    assert summary["pre_spin"] == "singlet"
    assert summary["post_spin"] == "triplet_m"
    assert abs(summary["post_negativity"] - 0.5) < 1e-12