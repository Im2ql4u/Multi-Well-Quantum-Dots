from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_spin_sector_scan.py"
_SPEC = importlib.util.spec_from_file_location("run_spin_sector_scan", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {_MODULE_PATH}.")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
SpinSectorJob = _MODULE.SpinSectorJob
build_sector_config = _MODULE.build_sector_config
build_spin_sector_jobs = _MODULE.build_spin_sector_jobs
run_sector_scan = _MODULE.run_sector_scan


def test_build_spin_sector_jobs_defaults_to_all_collinear_sectors() -> None:
    jobs = build_spin_sector_jobs(3, None)

    assert [job.n_up for job in jobs] == [0, 1, 2, 3]
    assert jobs[2].n_down == 1
    assert jobs[2].pattern == (0, 0, 1)


def test_build_sector_config_injects_spin_and_disables_guard() -> None:
    base_cfg = {
        "run_name": "trial",
        "magnetic_assessment": {"mode": "error"},
        "system": {
            "type": "single_dot",
            "n_particles": 3,
            "omega": 1.0,
            "dim": 2,
            "coulomb": False,
            "B_magnitude": 0.5,
            "B_direction": [0.0, 0.0, 1.0],
            "g_factor": 2.0,
            "mu_B": 1.0,
        },
        "training": {
            "loss_type": "residual",
            "residual_objective": "energy_var",
            "residual_target_energy": 3.0,
        },
    }
    job = SpinSectorJob(n_up=2, n_down=1, pattern=(0, 0, 1))

    cfg = build_sector_config(base_cfg, job)

    assert cfg["run_name"] == "trial__2up_1down"
    assert cfg["spin"]["pattern"] == [0, 0, 1]
    assert cfg["magnetic_assessment"]["mode"] == "off"
    assert abs(cfg["training"]["residual_target_energy"] - 3.5) < 1e-12
    assert abs(cfg["spin_sector_scan"]["constant_energy_shift"] - 0.5) < 1e-12


def test_run_sector_scan_selects_lowest_energy_sector(monkeypatch, tmp_path) -> None:
    energies = {
        "0up_3down": 5.0,
        "1up_2down": 4.0,
        "2up_1down": 3.5,
        "3up_0down": 4.5,
    }

    def _fake_train(cfg):
        n_up = sum(1 for value in cfg["spin"]["pattern"] if value == 0)
        n_down = sum(1 for value in cfg["spin"]["pattern"] if value == 1)
        label = f"{n_up}up_{n_down}down"
        out_dir = tmp_path / label
        result = {
            "final_energy": energies[label],
            "final_loss": 0.1,
            "final_ess": 16.0,
            "magnetic_assessment": {"classification": "constant_zeeman_shift_only"},
            "spin_configuration": {
                "label": label,
                "n_up": n_up,
                "n_down": n_down,
            },
            "scan_adjustments": {},
        }
        return out_dir, result

    monkeypatch.setattr(_MODULE, "run_training_from_config", _fake_train)
    jobs = build_spin_sector_jobs(3, None)
    summary = run_sector_scan(
        {
            "run_name": "trial",
            "system": {
                "type": "single_dot",
                "n_particles": 3,
                "omega": 1.0,
                "dim": 2,
                "coulomb": False,
                "B_magnitude": 0.5,
                "B_direction": [0.0, 0.0, 1.0],
                "g_factor": 2.0,
                "mu_B": 1.0,
            },
            "training": {
                "loss_type": "residual",
                "residual_objective": "energy_var",
                "residual_target_energy": 3.0,
            },
        },
        jobs,
    )

    assert summary["sector_competition_required"] is True
    assert summary["best_run"]["sector"]["label"] == "2up_1down"
    assert abs(summary["best_run"]["final_energy"] - 3.5) < 1e-12