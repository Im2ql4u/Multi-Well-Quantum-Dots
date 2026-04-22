#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch


LOG = logging.getLogger("summarize_magnetic_status")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diag_sweeps"
DEFAULT_REFERENCE_SUMMARY = DEFAULT_OUTPUT_DIR / "magnetic_reference_n2_20260416_summary.json"
DEFAULT_N2_CURRENT_JSON = REPO_ROOT / "results" / "imag_time_pinn" / "pinn_quench_single_fast_B0p50.json"
DEFAULT_N3_CURRENT_LOG = REPO_ROOT / "results" / "phase0_logs" / "n3_quench_fast.log"
DEFAULT_ENTANGLEMENT_SPREAD_TOL = 0.05


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def parse_csv_paths(text: str | None) -> list[Path]:
    if text is None:
        return []
    return [Path(token.strip()).expanduser().resolve() for token in text.split(",") if token.strip()]


def infer_zeeman_mode(payload: dict[str, Any]) -> str:
    particle_indices = payload.get("zeeman_particle_indices")
    if particle_indices:
        return f"particles:{particle_indices}"
    if bool(payload.get("zeeman_electron1_only")):
        return "electron1"
    return "all-electrons"


def infer_protocol_family(zeeman_mode: str) -> str:
    if zeeman_mode == "all-electrons":
        return "uniform_field_all_electrons"
    return "subset_zeeman_quench"


def choose_post_energy_estimate(payload: dict[str, Any]) -> tuple[float | None, str]:
    for key in ("fit_restricted", "fit_single", "fit_optimal_E0"):
        fit = payload.get(key)
        if isinstance(fit, dict) and fit.get("success") and "E0" in fit:
            return float(fit["E0"]), key
    trajectory = payload.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        last_row = trajectory[-1]
        if isinstance(last_row, dict) and "E" in last_row:
            return float(last_row["E"]), "trajectory_last"
    return None, "unavailable"


def load_checkpoint_metadata_from_json(payload: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint = payload.get("checkpoint")
    if not checkpoint:
        return None

    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    if not checkpoint_path.exists():
        return {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_exists": False,
        }

    ckpt_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt_payload.get("config", {})
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": True,
        "checkpoint_n_particles": config.get("n_particles"),
        "checkpoint_ground_state_dir": config.get("ground_state_dir"),
    }


def summarise_current_json_artifact(path: Path, *, n_particles: int) -> dict[str, Any]:
    payload = load_json(path)
    post_energy, post_energy_source = choose_post_energy_estimate(payload)
    initial_energy = float(payload["E_vmc"])
    late_tau_energy = float(payload["trajectory"][-1]["E"]) if payload.get("trajectory") else None
    zeeman_mode = infer_zeeman_mode(payload)
    checkpoint_metadata = load_checkpoint_metadata_from_json(payload)
    checkpoint_n_particles = None if checkpoint_metadata is None else checkpoint_metadata.get("checkpoint_n_particles")
    artifact_matches_expected = checkpoint_n_particles in (None, n_particles)
    notes: list[str] = []
    if checkpoint_metadata is not None and checkpoint_metadata.get("checkpoint_exists") is False:
        notes.append("reported_checkpoint_missing")
    if checkpoint_n_particles is not None and checkpoint_n_particles != n_particles:
        notes.append(f"artifact_n_particles_mismatch:{checkpoint_n_particles}")

    return {
        "n_particles": n_particles,
        "source_path": str(path),
        "artifact_kind": "json",
        "protocol_family": infer_protocol_family(zeeman_mode),
        "zeeman_mode": zeeman_mode,
        "legacy_sep": float(payload.get("d", 0.0)),
        "b_pre": float(payload.get("magnetic_B_initial", 0.0)),
        "b_post": float(payload.get("magnetic_B", 0.0)),
        "initial_energy": initial_energy,
        "post_energy_estimate": post_energy,
        "post_energy_source": post_energy_source,
        "late_tau_energy": late_tau_energy,
        "energy_shift": (post_energy - initial_energy) if post_energy is not None else None,
        "fit_gap": float(payload["fit_best"]["gap"]) if isinstance(payload.get("fit_best"), dict) and payload["fit_best"].get("success") else None,
        "checkpoint_path": None if checkpoint_metadata is None else checkpoint_metadata.get("checkpoint_path"),
        "checkpoint_exists": None if checkpoint_metadata is None else checkpoint_metadata.get("checkpoint_exists"),
        "checkpoint_n_particles": checkpoint_n_particles,
        "checkpoint_ground_state_dir": None if checkpoint_metadata is None else checkpoint_metadata.get("checkpoint_ground_state_dir"),
        "artifact_matches_expected_n_particles": artifact_matches_expected,
        "has_post_entanglement_measurement": False,
        "notes": notes,
    }


def summarise_entanglement_measurements(paths: list[Path]) -> dict[str, Any] | None:
    if not paths:
        return None

    runs: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        entanglement = payload["entanglement"]
        partition = payload.get("partition")
        runs.append(
            {
                "path": str(path),
                "tau": payload.get("tau"),
                "npts": int(payload["npts"]),
                "partition_mode": None if partition is None else partition.get("mode"),
                "von_neumann_entropy": float(entanglement["von_neumann_entropy"]),
                "negativity": float(entanglement["negativity"]),
                "effective_schmidt_rank": int(entanglement["effective_schmidt_rank"]),
            }
        )

    entropy_values = [run["von_neumann_entropy"] for run in runs]
    negativity_values = [run["negativity"] for run in runs]
    return {
        "runs": runs,
        "n_measurements": len(runs),
        "entropy_range": [min(entropy_values), max(entropy_values)],
        "negativity_range": [min(negativity_values), max(negativity_values)],
        "max_entropy_spread": max(entropy_values) - min(entropy_values),
        "max_negativity_spread": max(negativity_values) - min(negativity_values),
    }


def parse_log_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        return None
    return float(match.group(1))


def parse_log_text_artifact(path: Path, *, n_particles: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    header_match = re.search(
        r"interacting: d=([0-9.]+), .*?\n\s+magnetic: B_initial=([0-9.]+) \(VMC\), B_evolution=([0-9.]+) \(PINN\)\n\s+g=.*?, mu_B=.*?, zeeman=([^\n]+)",
        text,
        re.MULTILINE,
    )
    if header_match is None:
        raise ValueError(f"Could not parse magnetic header from {path}.")

    ground_state_match = re.search(r"Loaded GS from ([^\s]+)", text)
    saved_json_match = re.search(r"(?:Saved single-B quench result|SAVED_JSON)=([^\s]+)", text)
    checkpoint_match = re.search(r"Checkpoint saved:\s+([^\s]+)", text)
    exp_fit_e0 = parse_log_float(r"\[Exp fit\]\s+E_0=([0-9.]+)", text)
    fit_gap = parse_log_float(r"\[BEST\]\s+gap=([0-9.]+)", text)
    initial_energy = parse_log_float(r"Using E_ref = ([0-9.]+)", text)

    table_rows = re.findall(
        r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        text,
        re.MULTILINE,
    )
    late_tau_energy = float(table_rows[-1][1]) if table_rows else None

    saved_json_path = Path(saved_json_match.group(1)).resolve() if saved_json_match else None
    checkpoint_path = Path(checkpoint_match.group(1)).resolve() if checkpoint_match else None
    zeeman_mode = header_match.group(4).strip()
    notes: list[str] = []
    if ground_state_match is not None:
        notes.append(f"locked_ground_state={ground_state_match.group(1)}")
    if saved_json_path is not None and not saved_json_path.exists():
        notes.append("reported_saved_json_missing")
    if checkpoint_path is not None and not checkpoint_path.exists():
        notes.append("reported_checkpoint_missing")

    return {
        "n_particles": n_particles,
        "source_path": str(path),
        "artifact_kind": "log",
        "protocol_family": infer_protocol_family(zeeman_mode),
        "zeeman_mode": zeeman_mode,
        "legacy_sep": float(header_match.group(1)),
        "b_pre": float(header_match.group(2)),
        "b_post": float(header_match.group(3)),
        "initial_energy": initial_energy,
        "post_energy_estimate": exp_fit_e0 if exp_fit_e0 is not None else late_tau_energy,
        "post_energy_source": "exp_fit" if exp_fit_e0 is not None else "trajectory_last",
        "late_tau_energy": late_tau_energy,
        "energy_shift": ((exp_fit_e0 if exp_fit_e0 is not None else late_tau_energy) - initial_energy) if initial_energy is not None and (exp_fit_e0 is not None or late_tau_energy is not None) else None,
        "fit_gap": fit_gap,
        "ground_state_dir": ground_state_match.group(1) if ground_state_match is not None else None,
        "saved_json_path": str(saved_json_path) if saved_json_path is not None else None,
        "saved_json_exists": saved_json_path.exists() if saved_json_path is not None else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_exists": checkpoint_path.exists() if checkpoint_path is not None else None,
        "has_post_entanglement_measurement": False,
        "notes": notes,
    }


def find_reference_run(payload: dict[str, Any], *, sep: float, b_pre: float, b_post: float) -> dict[str, Any] | None:
    runs = payload.get("runs", [])
    for run in runs:
        if not isinstance(run, dict):
            continue
        if abs(float(run.get("sep", 1e9)) - sep) > 1e-9:
            continue
        if abs(float(run.get("B_pre", 1e9)) - b_pre) > 1e-9:
            continue
        if abs(float(run.get("B_post", 1e9)) - b_post) > 1e-9:
            continue
        return run
    return None


def build_system_status(
    *,
    n_particles: int,
    current_artifact: dict[str, Any] | None,
    reference_run: dict[str, Any] | None,
    energy_alignment_tol: float,
    entanglement_spread_tol: float,
) -> dict[str, Any]:
    evidence_gaps: list[str] = []
    comparison: dict[str, Any] = {
        "comparable": False,
        "energy_alignment_tol": energy_alignment_tol,
    }
    usable_current_artifact = current_artifact
    if current_artifact is not None and current_artifact.get("artifact_matches_expected_n_particles") is False:
        evidence_gaps.append("artifact_n_particles_mismatch")
        usable_current_artifact = None

    if usable_current_artifact is None and reference_run is None:
        status = "missing_evidence"
        evidence_gaps.append("no_current_model_magnetic_artifact")
        evidence_gaps.append("no_reference_magnetic_sweep")
    elif usable_current_artifact is None:
        status = "reference_only"
        evidence_gaps.append("no_current_model_magnetic_artifact")
    elif reference_run is None:
        status = "current_only_unvalidated"
        evidence_gaps.append("no_reference_magnetic_sweep")
        if not bool(usable_current_artifact.get("has_post_entanglement_measurement")):
            evidence_gaps.append("no_current_post_entanglement_measurement")
        if usable_current_artifact.get("saved_json_exists") is False:
            evidence_gaps.append("reported_saved_json_missing")
        if usable_current_artifact.get("checkpoint_exists") is False:
            evidence_gaps.append("reported_checkpoint_missing")
        entanglement_summary = usable_current_artifact.get("post_quench_entanglement")
        if isinstance(entanglement_summary, dict):
            entropy_spread = float(entanglement_summary["max_entropy_spread"])
            negativity_spread = float(entanglement_summary["max_negativity_spread"])
            if entropy_spread > entanglement_spread_tol or negativity_spread > entanglement_spread_tol:
                evidence_gaps.append("post_quench_entanglement_not_converged")
    else:
        comparison["comparable"] = usable_current_artifact["protocol_family"] == "uniform_field_all_electrons"
        comparison["reference_spin_flip"] = bool(reference_run["spin_flip"])
        comparison["reference_post_spin"] = reference_run["post_spin"]
        comparison["reference_post_negativity"] = reference_run["post_negativity"]
        current_post_energy = usable_current_artifact.get("post_energy_estimate")
        reference_post_energy = float(reference_run["post_energy"])
        comparison["reference_post_energy"] = reference_post_energy
        comparison["current_post_energy"] = current_post_energy
        if current_post_energy is not None:
            energy_abs_diff = abs(float(current_post_energy) - reference_post_energy)
            comparison["post_energy_abs_diff"] = energy_abs_diff
            comparison["energy_aligned"] = energy_abs_diff <= energy_alignment_tol
            initial_energy = usable_current_artifact.get("initial_energy")
            if initial_energy is not None:
                comparison["moves_toward_reference"] = abs(float(current_post_energy) - reference_post_energy) < abs(float(initial_energy) - reference_post_energy)
        else:
            comparison["post_energy_abs_diff"] = None
            comparison["energy_aligned"] = False
        if not bool(usable_current_artifact.get("has_post_entanglement_measurement")):
            evidence_gaps.append("no_current_post_entanglement_measurement")
        if not comparison["comparable"]:
            status = "not_comparable"
            evidence_gaps.append("protocol_family_mismatch")
        elif comparison.get("energy_aligned"):
            status = "aligned_with_reference"
        else:
            status = "misaligned_with_reference"

    assessment = build_assessment(status=status, n_particles=n_particles, current_artifact=usable_current_artifact, reference_run=reference_run, comparison=comparison)
    return {
        "n_particles": n_particles,
        "status": status,
        "has_reference": reference_run is not None,
        "has_current_artifact": usable_current_artifact is not None,
        "current_model": current_artifact,
        "reference": reference_run,
        "comparison": comparison,
        "evidence_gaps": evidence_gaps,
        "assessment": assessment,
    }


def build_assessment(
    *,
    status: str,
    n_particles: int,
    current_artifact: dict[str, Any] | None,
    reference_run: dict[str, Any] | None,
    comparison: dict[str, Any],
) -> str:
    if status == "missing_evidence":
        return f"N={n_particles}: no magnetic artifact and no magnetic reference are available."
    if status == "reference_only":
        return f"N={n_particles}: reference magnetic physics exists, but there is no current-model artifact to compare against."
    if status == "current_only_unvalidated":
        protocol = current_artifact["protocol_family"] if current_artifact is not None else "unknown"
        if current_artifact is not None and isinstance(current_artifact.get("post_quench_entanglement"), dict):
            ent_summary = current_artifact["post_quench_entanglement"]
            return (
                f"N={n_particles}: current-model magnetic artifact exists ({protocol}), but there is no reference lane yet; "
                f"post-quench entropy spans {ent_summary['entropy_range'][0]:.6f} to {ent_summary['entropy_range'][1]:.6f} "
                f"across the measured grids."
            )
        return f"N={n_particles}: current-model magnetic artifact exists ({protocol}), but there is no reference lane yet."
    if status == "not_comparable":
        return f"N={n_particles}: current artifact and reference use different magnetic protocols, so direct alignment is not defensible."
    if status == "aligned_with_reference":
        return f"N={n_particles}: current magnetic artifact matches the available reference within the configured energy tolerance."
    if reference_run is not None and current_artifact is not None:
        return (
            f"N={n_particles}: current magnetic artifact does not reproduce the reference post-field state; "
            f"reference ends at E={float(reference_run['post_energy']):.6f} with spin {reference_run['post_spin']}, "
            f"while the current artifact estimates E={float(current_artifact['post_energy_estimate']):.6f}."
        )
    return f"N={n_particles}: status unresolved."


def build_summary_payload(
    *,
    reference_summary: dict[str, Any],
    n2_current: dict[str, Any] | None,
    n3_current: dict[str, Any] | None,
    energy_alignment_tol: float,
    entanglement_spread_tol: float,
    reference_summary_path: Path,
) -> dict[str, Any]:
    systems = [
        build_system_status(
            n_particles=2,
            current_artifact=n2_current,
            reference_run=find_reference_run(reference_summary, sep=8.0, b_pre=0.0, b_post=0.5),
            energy_alignment_tol=energy_alignment_tol,
            entanglement_spread_tol=entanglement_spread_tol,
        ),
        build_system_status(
            n_particles=3,
            current_artifact=n3_current,
            reference_run=None,
            energy_alignment_tol=energy_alignment_tol,
            entanglement_spread_tol=entanglement_spread_tol,
        ),
        build_system_status(
            n_particles=4,
            current_artifact=None,
            reference_run=None,
            energy_alignment_tol=energy_alignment_tol,
            entanglement_spread_tol=entanglement_spread_tol,
        ),
    ]
    return {
        "summary_scope": "magnetic_reference_vs_current_model",
        "reference_summary_path": str(reference_summary_path),
        "energy_alignment_tol": energy_alignment_tol,
        "entanglement_spread_tol": entanglement_spread_tol,
        "systems": systems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize magnetic-field status across reference and current-model artifacts.")
    parser.add_argument("--reference-summary", default=str(DEFAULT_REFERENCE_SUMMARY), help="Path to the N=2 magnetic reference sweep summary JSON.")
    parser.add_argument("--n2-current-json", default=str(DEFAULT_N2_CURRENT_JSON), help="Path to the current N=2 magnetic artifact JSON.")
    parser.add_argument("--n3-current-log", default=str(DEFAULT_N3_CURRENT_LOG), help="Path to the current N=3 magnetic artifact log.")
    parser.add_argument("--n3-entanglement-jsons", default="", help="Comma-separated post-quench entanglement JSONs for the N=3 current artifact.")
    parser.add_argument("--energy-alignment-tol", type=float, default=0.05, help="Absolute energy tolerance for calling a current artifact aligned with reference.")
    parser.add_argument("--entanglement-spread-tol", type=float, default=DEFAULT_ENTANGLEMENT_SPREAD_TOL, help="Tolerance for calling multiple entanglement measurements numerically consistent.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_DIR / "magnetic_status_summary_20260416.json"), help="Output JSON path.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    reference_summary_path = Path(args.reference_summary).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    reference_summary = load_json(reference_summary_path)

    n2_current_path = Path(args.n2_current_json).expanduser().resolve()
    n3_current_path = Path(args.n3_current_log).expanduser().resolve()
    n3_entanglement_paths = parse_csv_paths(args.n3_entanglement_jsons)

    n2_current = summarise_current_json_artifact(n2_current_path, n_particles=2) if n2_current_path.exists() else None
    n3_current = parse_log_text_artifact(n3_current_path, n_particles=3) if n3_current_path.exists() else None
    if n3_current is not None:
        entanglement_summary = summarise_entanglement_measurements([path for path in n3_entanglement_paths if path.exists()])
        n3_current["post_quench_entanglement"] = entanglement_summary
        n3_current["has_post_entanglement_measurement"] = entanglement_summary is not None

    payload = build_summary_payload(
        reference_summary=reference_summary,
        n2_current=n2_current,
        n3_current=n3_current,
        energy_alignment_tol=args.energy_alignment_tol,
        entanglement_spread_tol=args.entanglement_spread_tol,
        reference_summary_path=reference_summary_path,
    )

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    LOG.info("Saved magnetic status summary to %s", output_json)


if __name__ == "__main__":
    main()