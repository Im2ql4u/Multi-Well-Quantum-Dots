from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_entanglement_sweep.py"
_SPEC = importlib.util.spec_from_file_location("run_entanglement_sweep", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {_MODULE_PATH}.")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
build_sweep_jobs = _MODULE.build_sweep_jobs


def test_build_sweep_jobs_generates_expected_output_names(tmp_path: Path) -> None:
    result_dir = tmp_path / "results" / "demo_run"
    output_dir = tmp_path / "diag_sweeps"

    jobs = build_sweep_jobs(
        result_dir=result_dir,
        npts_values=[4, 6],
        partitions=["auto", "0,1"],
        output_dir=output_dir,
        output_prefix="demo",
    )

    assert len(jobs) == 4
    assert jobs[0].npts == 4
    assert jobs[0].partition == "auto"
    assert jobs[0].output_path == output_dir / "demo__npts4__part_auto.json"
    assert jobs[-1].npts == 6
    assert jobs[-1].partition == "0,1"
    assert jobs[-1].output_path == output_dir / "demo__npts6__part_0-1.json"