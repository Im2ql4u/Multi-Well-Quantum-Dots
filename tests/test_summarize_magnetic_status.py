from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_magnetic_status.py"
_SPEC = importlib.util.spec_from_file_location("summarize_magnetic_status", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {_MODULE_PATH}.")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_summary_payload = _MODULE.build_summary_payload
parse_log_text_artifact = _MODULE.parse_log_text_artifact
summarise_current_json_artifact = _MODULE.summarise_current_json_artifact
summarise_entanglement_measurements = _MODULE.summarise_entanglement_measurements


def test_summarise_current_json_artifact_extracts_uniform_field_summary(tmp_path: Path) -> None:
    artifact_path = tmp_path / "n2_current.json"
    artifact_path.write_text(
        json.dumps(
            {
                "d": 8.0,
                "magnetic_B_initial": 0.0,
                "magnetic_B": 0.5,
                "zeeman_electron1_only": False,
                "zeeman_particle_indices": None,
                "E_vmc": 3.6,
                "trajectory": [{"E": 4.1}],
                "fit_restricted": {"success": True, "E0": 4.05},
                "fit_best": {"success": True, "gap": 0.8},
            }
        ),
        encoding="utf-8",
    )

    summary = summarise_current_json_artifact(artifact_path, n_particles=2)

    assert summary["protocol_family"] == "uniform_field_all_electrons"
    assert summary["post_energy_estimate"] == 4.05
    assert abs(summary["energy_shift"] - 0.45) < 1e-12


def test_build_summary_payload_treats_particle_mismatch_as_missing_current() -> None:
    payload = build_summary_payload(
        reference_summary={
            "runs": [
                {
                    "sep": 8.0,
                    "B_pre": 0.0,
                    "B_post": 0.5,
                    "post_energy": 2.3,
                    "post_spin": "triplet_m",
                    "spin_flip": True,
                    "post_negativity": 0.5,
                }
            ]
        },
        n2_current={
            "n_particles": 2,
            "artifact_matches_expected_n_particles": False,
        },
        n3_current=None,
        energy_alignment_tol=0.05,
        entanglement_spread_tol=0.05,
        reference_summary_path=Path("reference.json"),
    )

    systems = {entry["n_particles"]: entry for entry in payload["systems"]}

    assert systems[2]["status"] == "reference_only"
    assert "artifact_n_particles_mismatch" in systems[2]["evidence_gaps"]


def test_parse_log_text_artifact_marks_missing_saved_json(tmp_path: Path) -> None:
    log_path = tmp_path / "n3.log"
    log_path.write_text(
        """
=================================================================
  interacting: d=0.0, ω=1.0, E_ref=3.00000
  magnetic: B_initial=0.0000 (VMC), B_evolution=0.5000 (PINN)
            g=2.000, mu_B=1.000, zeeman=all-electrons
=================================================================
  Loaded GS from results/example_locked_state
  Using E_ref = 3.63433
       tau           E       E_err     g_rms     n_eff
  --------  ----------  ----------  --------  --------
    2.0000     4.13715     0.00174    0.7123      3912
  [Exp fit]  E_0=4.13270±0.00003, gap=0.9110±0.0007
  [BEST]     gap=0.8450±0.2263 (ensemble of 4)
    Checkpoint saved: /tmp/does_not_exist.pt
  SAVED_JSON=/tmp/does_not_exist.json
""",
        encoding="utf-8",
    )

    summary = parse_log_text_artifact(log_path, n_particles=3)

    assert summary["post_energy_source"] == "exp_fit"
    assert abs(summary["post_energy_estimate"] - 4.13270) < 1e-12
    assert summary["saved_json_exists"] is False
    assert summary["checkpoint_exists"] is False
    assert "reported_saved_json_missing" in summary["notes"]


def test_summarise_entanglement_measurements_reports_spread(tmp_path: Path) -> None:
    path_a = tmp_path / "m4.json"
    path_b = tmp_path / "m5.json"
    for path, npts, entropy, negativity in (
        (path_a, 4, 0.02, 0.08),
        (path_b, 5, 0.22, 0.27),
    ):
        path.write_text(
            json.dumps(
                {
                    "tau": 2.0,
                    "npts": npts,
                    "partition": {"mode": "auto_left_right_well_blocks"},
                    "entanglement": {
                        "von_neumann_entropy": entropy,
                        "negativity": negativity,
                        "effective_schmidt_rank": 2,
                    },
                }
            ),
            encoding="utf-8",
        )

    summary = summarise_entanglement_measurements([path_a, path_b])

    assert summary is not None
    assert summary["n_measurements"] == 2
    assert abs(summary["max_entropy_spread"] - 0.2) < 1e-12
    assert abs(summary["max_negativity_spread"] - 0.19) < 1e-12


def test_build_summary_payload_classifies_statuses() -> None:
    reference_summary = {
        "runs": [
            {
                "sep": 8.0,
                "B_pre": 0.0,
                "B_post": 0.5,
                "post_energy": 2.328607153289828,
                "post_spin": "triplet_m",
                "spin_flip": True,
                "post_negativity": 0.5,
            }
        ]
    }
    n2_current = {
        "n_particles": 2,
        "protocol_family": "uniform_field_all_electrons",
        "post_energy_estimate": 4.1327,
        "initial_energy": 3.63433,
    }
    n3_current = {
        "n_particles": 3,
        "protocol_family": "uniform_field_all_electrons",
        "post_energy_estimate": 4.1327,
        "initial_energy": 3.63433,
        "saved_json_exists": False,
    }

    payload = build_summary_payload(
        reference_summary=reference_summary,
        n2_current=n2_current,
        n3_current=n3_current,
        energy_alignment_tol=0.05,
        entanglement_spread_tol=0.05,
        reference_summary_path=Path("reference.json"),
    )

    systems = {entry["n_particles"]: entry for entry in payload["systems"]}

    assert systems[2]["status"] == "misaligned_with_reference"
    assert systems[2]["comparison"]["energy_aligned"] is False
    assert systems[3]["status"] == "current_only_unvalidated"
    assert "reported_saved_json_missing" in systems[3]["evidence_gaps"]
    assert systems[4]["status"] == "missing_evidence"


def test_build_summary_payload_marks_n3_entanglement_not_converged() -> None:
    payload = build_summary_payload(
        reference_summary={"runs": []},
        n2_current=None,
        n3_current={
            "n_particles": 3,
            "protocol_family": "uniform_field_all_electrons",
            "post_energy_estimate": 4.1327,
            "initial_energy": 3.63433,
            "has_post_entanglement_measurement": True,
            "post_quench_entanglement": {
                "entropy_range": [0.02, 0.22],
                "negativity_range": [0.08, 0.27],
                "max_entropy_spread": 0.2,
                "max_negativity_spread": 0.19,
            },
        },
        energy_alignment_tol=0.05,
        entanglement_spread_tol=0.05,
        reference_summary_path=Path("reference.json"),
    )

    systems = {entry["n_particles"]: entry for entry in payload["systems"]}

    assert systems[3]["status"] == "current_only_unvalidated"
    assert "post_quench_entanglement_not_converged" in systems[3]["evidence_gaps"]
    assert "entropy spans 0.020000 to 0.220000" in systems[3]["assessment"]