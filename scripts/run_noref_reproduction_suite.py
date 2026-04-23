#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diag_sweeps"


@dataclass(frozen=True)
class ReproductionJob:
    suite: str
    label: str
    legacy_config: str
    legacy_result_glob: str
    validation_bundle: str
    notes: tuple[str, ...] = ()


def _gs_seed_sweep_jobs() -> list[ReproductionJob]:
    jobs: list[ReproductionJob] = []
    for n in (2, 3, 4):
        for seed in (42, 314, 901):
            if seed == 42:
                cfg = f"configs/one_per_well/n{n}_nonmcmc_residual_anneal_s42.yaml"
            else:
                cfg = f"configs/one_per_well/seed_sweep/n{n}_nonmcmc_residual_anneal_s{seed}.yaml"
            jobs.append(
                ReproductionJob(
                    suite="ground_states",
                    label=f"n{n}_seed{seed}",
                    legacy_config=cfg,
                    legacy_result_glob=f"results/p4_n{n}_nonmcmc_residual_anneal_s{seed}_*",
                    validation_bundle="training+fd_components+generalized_virial",
                    notes=(
                        "Primary benchmark lane from the 2026-04-13 non-MCMC seed sweep report.",
                        "Compare to finite-basis CI using one-sided exceedance and to legacy runs using FD/virial.",
                    ),
                )
            )
    return jobs


def _singlet_jobs() -> list[ReproductionJob]:
    jobs: list[ReproductionJob] = []
    for d in (2, 4, 6, 8, 12, 20):
        jobs.append(
            ReproductionJob(
                suite="singlet_separation_sweep",
                label=f"n2_singlet_d{d}",
                legacy_config=f"configs/one_per_well/n2_singlet_d{d}_s42.yaml",
                legacy_result_glob=f"results/p4_n2_singlet_d{d}_s42_*",
                validation_bundle="training+fd_components+particle_and_dot_entanglement",
                notes=(
                    "Match the 2026-04-16 singlet permanent sweep.",
                    "Use the same Lowdin entanglement measurements and sector-probability checks.",
                ),
            )
        )
    return jobs


def _magnetic_jobs() -> list[ReproductionJob]:
    return [
        ReproductionJob(
            suite="magnetic_sectors",
            label="n3_0up3down_b0p5",
            legacy_config="configs/magnetic/n3_0up3down_b0p5_s42.yaml",
            legacy_result_glob="results/p5_n3_mag_0up_b0p5_s42_*",
            validation_bundle="training+fd_components+sector_ranking",
            notes=("Fixed-spin sector energy ladder under uniform B.",),
        ),
        ReproductionJob(
            suite="magnetic_sectors",
            label="n3_1up2down_b0p5",
            legacy_config="configs/magnetic/n3_1up2down_b0p5_s42.yaml",
            legacy_result_glob="results/p5_n3_mag_1up_b0p5_s42_*",
            validation_bundle="training+fd_components+sector_ranking",
        ),
        ReproductionJob(
            suite="magnetic_sectors",
            label="n3_2up1down_b0p5",
            legacy_config="configs/magnetic/n3_2up1down_b0p5_s42.yaml",
            legacy_result_glob="results/p5_n3_mag_2up_b0p5_s42_*",
            validation_bundle="training+fd_components+sector_ranking",
        ),
        ReproductionJob(
            suite="magnetic_sectors",
            label="n3_3up0down_b0p5",
            legacy_config="configs/magnetic/n3_3up0down_b0p5_s42.yaml",
            legacy_result_glob="results/p5_n3_mag_3up_b0p5_s42_*",
            validation_bundle="training+fd_components+sector_ranking",
        ),
        ReproductionJob(
            suite="magnetic_sectors",
            label="n4_0up4down_b0p5",
            legacy_config="configs/magnetic/n4_0up4down_b0p5_s42.yaml",
            legacy_result_glob="results/p5_n4_mag_0up4down_b0p5_s42_*",
            validation_bundle="training+fd_components+sector_ranking",
        ),
        ReproductionJob(
            suite="magnetic_sectors",
            label="n4_2up2down_b0p5",
            legacy_config="configs/magnetic/n4_2up2down_b0p5_s42.yaml",
            legacy_result_glob="results/p5_n4_mag_2up2down_b0p5_s42_*",
            validation_bundle="training+fd_components+sector_ranking",
        ),
        ReproductionJob(
            suite="magnetic_sectors",
            label="n4_4up0down_b0p5",
            legacy_config="configs/magnetic/n4_4up0down_b0p5_s42.yaml",
            legacy_result_glob="results/p5_n4_mag_4up_b0p5_s42_*",
            validation_bundle="training+fd_components+sector_ranking",
        ),
    ]


def _already_reference_free_jobs() -> list[ReproductionJob]:
    return [
        ReproductionJob(
            suite="already_reference_free",
            label="n2_singlet_d4_lambda_sweep",
            legacy_config="configs/magnetic/n2_singlet_d4_lam0p00_s42.yaml .. lam0p75",
            legacy_result_glob="results/p5_n2_singlet_d4_lam*_s42_*",
            validation_bundle="training+energy_monotonicity+entanglement_optional",
            notes=(
                "These lambda runs already used alpha_end=0 and are effectively CI-free.",
                "They should be kept as the post-reproduction extension rather than the initial benchmark gate.",
            ),
        )
    ]


def build_jobs(suite: str) -> list[ReproductionJob]:
    all_jobs = (
        _gs_seed_sweep_jobs()
        + _singlet_jobs()
        + _magnetic_jobs()
        + _already_reference_free_jobs()
    )
    if suite == "all":
        return all_jobs
    return [job for job in all_jobs if job.suite == suite]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit the canonical no-reference reproduction matrix for the previous E_ref success lanes."
    )
    parser.add_argument(
        "--suite",
        default="all",
        choices=[
            "all",
            "ground_states",
            "singlet_separation_sweep",
            "magnetic_sectors",
            "already_reference_free",
        ],
        help="Subset of the reproduction matrix to emit.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional explicit JSON path for the emitted manifest.",
    )
    args = parser.parse_args()

    jobs = build_jobs(args.suite)
    payload: dict[str, Any] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "scope": args.suite,
        "jobs": [asdict(job) for job in jobs],
        "notes": [
            "ground_states must be the first gate because they feed all later physics products.",
            "singlet_separation_sweep is the second gate because it checks the dedicated singlet ansatz and entanglement measurement path.",
            "magnetic_sectors are fixed-spin energy ladders only; they do not validate nontrivial uniform-B response by themselves.",
            "lambda_sweep is already effectively CI-free in the legacy lane and should be resumed after the reproduction gates pass.",
        ],
    }

    out_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else DEFAULT_OUTPUT_DIR / f"noref_reproduction_manifest_{args.suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved reproduction manifest to {out_path}")


if __name__ == "__main__":
    main()
