#!/usr/bin/env python3
"""Unified imaginary-time TDSE runner with strategy comparison and plotting.

This script orchestrates two existing pipelines:
- Tau-conditioned VMC (src/imaginary_time_vmc.py)
- Spectral PINN (src/imaginary_time_pinn.py)

It adds:
- Reproducible run manifests
- Optional cleanup/archiving of legacy result files
- Strategy comparison plots and summary tables
- A smoke mode for quick end-to-end verification
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

import imaginary_time_pinn as it_pinn
import imaginary_time_vmc as it_vmc

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "imag_time_runs"
ARCHIVE_ROOT = RESULTS_ROOT / "imag_time_archive"


@dataclass
class SuiteConfig:
    profile: str
    distances: list[float]
    coulomb: bool
    tau_max: float
    omega: float
    strategies: list[str]
    tag: str


def parse_distances(raw: str) -> list[float]:
    vals: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("At least one distance is required.")
    return vals


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)


def safe_name(x: float) -> str:
    return str(x).replace(".", "p")


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _legacy_targets() -> list[Path]:
    return [
        RESULTS_ROOT / "imag_time_vmc",
        RESULTS_ROOT / "imag_time_pinn",
        RESULTS_ROOT / "imag_time_report.html",
    ]


def archive_legacy_results(dry_run: bool = False) -> dict:
    """Move old imaginary-time outputs into a timestamped archive directory."""
    ensure_dirs()
    archive_dir = ARCHIVE_ROOT / f"legacy_{utc_stamp()}"
    moved: list[str] = []
    skipped: list[str] = []

    for path in _legacy_targets():
        if not path.exists():
            skipped.append(str(path.relative_to(ROOT)))
            continue
        if dry_run:
            moved.append(str(path.relative_to(ROOT)))
            continue

        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / path.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(path), str(dest))
        moved.append(str(path.relative_to(ROOT)))

    return {
        "archive_dir": str(archive_dir.relative_to(ROOT)),
        "dry_run": dry_run,
        "moved": moved,
        "skipped_missing": skipped,
    }


def vmc_cfg(profile: str, d: float, coulomb: bool, omega: float, tau_max: float) -> it_vmc.VMCConfig:
    base = it_vmc.VMCConfig(
        omega=omega,
        well_sep=d,
        E_ref=(3.0 if coulomb else 2.0),
        coulomb=coulomb,
        tau_max=tau_max,
    )

    if profile == "smoke":
        base.n_epochs_vmc = 8
        base.n_samples_vmc = 128
        base.n_precompute = 256
        base.n_epochs_pde = 20
        base.batch_pde = 64
        base.n_tau_eval = 16
        base.n_samples_eval = 512
        base.mcmc_warmup_eval = 80
        base.lambda_ic = 20.0
        base.lambda_vmc = 0.2
        base.pde_vmc_freq = 4
    elif profile == "tiny":
        base.n_epochs_vmc = 180
        base.n_samples_vmc = 512
        base.n_precompute = 2048
        base.n_epochs_pde = 1200
        base.batch_pde = 128
        base.n_tau_eval = 30
        base.n_samples_eval = 1500
        base.mcmc_warmup_eval = 220
        base.lambda_ic = 40.0
        base.lambda_vmc = 0.5
    elif profile == "baseline":
        base.n_epochs_vmc = 800
        base.n_samples_vmc = 1024
        base.n_precompute = 8192
        base.n_epochs_pde = 10000
        base.batch_pde = 256
        base.n_tau_eval = 60
        base.n_samples_eval = 5000
        base.mcmc_warmup_eval = 500
        base.lambda_ic = 80.0
        base.lambda_vmc = 1.0
    elif profile == "production":
        base.n_epochs_vmc = 1600
        base.n_samples_vmc = 1024
        base.n_precompute = 16384
        base.n_epochs_pde = 18000
        base.batch_pde = 256
        base.n_tau_eval = 70
        base.n_samples_eval = 8000
        base.mcmc_warmup_eval = 700
        base.lambda_ic = 80.0
        base.lambda_vmc = 1.0
    else:
        raise ValueError(f"Unknown profile: {profile}")

    return base


def pinn_cfg(profile: str, d: float, coulomb: bool, omega: float, tau_max: float) -> it_pinn.PINNConfig:
    base = it_pinn.PINNConfig(
        omega=omega,
        well_sep=d,
        E_ref=(3.0 if coulomb else 2.0),
        coulomb=coulomb,
        tau_max=tau_max,
    )

    if profile == "smoke":
        base.n_epochs_vmc = 8
        base.n_samples_vmc = 128
        base.n_precompute = 256
        base.n_epochs_pde = 24
        base.batch_pde = 64
        base.n_tau_eval = 16
        base.n_samples_eval = 512
        base.g_modes = 2
        base.g_hidden = 24
        base.g_layers = 2
    elif profile == "tiny":
        base.n_epochs_vmc = 140
        base.n_samples_vmc = 512
        base.n_precompute = 2048
        base.n_epochs_pde = 1000
        base.batch_pde = 128
        base.n_tau_eval = 30
        base.n_samples_eval = 1500
        base.g_modes = 3
    elif profile == "baseline":
        base.n_epochs_vmc = 700
        base.n_samples_vmc = 512
        base.n_precompute = 8192
        base.n_epochs_pde = 9000
        base.batch_pde = 256
        base.n_tau_eval = 50
        base.n_samples_eval = 6000
        base.g_modes = 3
    elif profile == "production":
        base.n_epochs_vmc = 1200
        base.n_samples_vmc = 1024
        base.n_precompute = 16384
        base.n_epochs_pde = 18000
        base.batch_pde = 512
        base.n_tau_eval = 70
        base.n_samples_eval = 9000
        base.g_modes = 5
        base.g_hidden = 64
        base.g_layers = 3
    else:
        raise ValueError(f"Unknown profile: {profile}")

    return base


def _fit_gap(result: dict) -> float:
    fit_best = result.get("fit_best", {})
    if fit_best.get("success") and np.isfinite(fit_best.get("gap", np.nan)):
        return float(fit_best["gap"])
    fit_single = result.get("fit_single", {})
    if fit_single.get("success") and np.isfinite(fit_single.get("gap", np.nan)):
        return float(fit_single["gap"])
    return float("nan")


def _time_seconds(result: dict) -> float:
    parts = [
        float(result.get("t_vmc", 0.0)),
        float(result.get("t_pde", result.get("t_pinn", 0.0))),
        float(result.get("t_eval", 0.0)),
    ]
    return float(sum(parts))


def run_suite(cfg: SuiteConfig, out_dir: Path) -> dict:
    results: list[dict] = []
    runs_meta: list[dict] = []

    for strategy in cfg.strategies:
        for d in cfg.distances:
            print("\n" + "=" * 72)
            print(f"Running strategy={strategy} d={d:.3f} profile={cfg.profile} coulomb={cfg.coulomb}")
            print("=" * 72)

            if strategy == "vmc":
                c = vmc_cfg(cfg.profile, d, cfg.coulomb, cfg.omega, cfg.tau_max)
                result = it_vmc.run_single(c, tag=f"{cfg.tag}{strategy}_d{safe_name(d)}_")
                conf = asdict(c)
            elif strategy == "pinn":
                c = pinn_cfg(cfg.profile, d, cfg.coulomb, cfg.omega, cfg.tau_max)
                result = it_pinn.run_single(c, tag=f"{cfg.tag}{strategy}_d{safe_name(d)}_")
                conf = asdict(c)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            item = {
                "strategy": strategy,
                "distance": d,
                "profile": cfg.profile,
                "config": conf,
                "result": result,
                "summary": {
                    "E_vmc": float(result.get("E_vmc", np.nan)),
                    "gap_best": _fit_gap(result),
                    "seconds_total": _time_seconds(result),
                },
            }
            results.append(item)

            runs_meta.append(
                {
                    "strategy": strategy,
                    "distance": d,
                    "json": f"{strategy}_d{safe_name(d)}.json",
                }
            )
            save_json(item, out_dir / f"{strategy}_d{safe_name(d)}.json")

    suite = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite_config": asdict(cfg),
        "runs": runs_meta,
        "results": results,
    }
    save_json(suite, out_dir / "suite_results.json")
    return suite


def _group_by_strategy(suite: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for item in suite.get("results", []):
        grouped.setdefault(item["strategy"], []).append(item)
    for strategy in grouped:
        grouped[strategy].sort(key=lambda x: x["distance"])
    return grouped


def make_plots(suite: dict, out_dir: Path) -> list[str]:
    grouped = _group_by_strategy(suite)
    made: list[str] = []

    # 1) Energy vs distance
    plt.figure(figsize=(8, 5))
    for strategy, items in grouped.items():
        xs = [x["distance"] for x in items]
        ys = [x["summary"]["E_vmc"] for x in items]
        plt.plot(xs, ys, marker="o", linewidth=2, label=strategy.upper())

    if xs:
        x_ref = np.linspace(min(xs), max(xs), 200)
        inv_x = np.zeros_like(x_ref)
        np.divide(1.0, x_ref, out=inv_x, where=x_ref > 1e-10)
        y_ref = 2.0 + np.where(x_ref > 1e-10, inv_x, 1.0)
        plt.plot(x_ref, y_ref, linestyle="--", color="black", alpha=0.6, label="2 + 1/d (guide)")

    plt.title("Ground-State Energy Across Well Separation")
    plt.xlabel("d")
    plt.ylabel("E_vmc")
    plt.grid(alpha=0.3)
    plt.legend()
    p = out_dir / "figure_energy_vs_d.png"
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    made.append(p.name)

    # 2) Gap vs distance
    plt.figure(figsize=(8, 5))
    for strategy, items in grouped.items():
        xs = [x["distance"] for x in items]
        ys = [x["summary"]["gap_best"] for x in items]
        plt.plot(xs, ys, marker="s", linewidth=2, label=strategy.upper())
    plt.axhline(1.0, linestyle="--", color="black", alpha=0.6, label="Kohn gap = 1")
    plt.title("Imaginary-Time Gap Estimate")
    plt.xlabel("d")
    plt.ylabel("gap")
    plt.grid(alpha=0.3)
    plt.legend()
    p = out_dir / "figure_gap_vs_d.png"
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    made.append(p.name)

    # 3) E(tau) overlays per distance
    distances = sorted({float(x["distance"]) for x in suite.get("results", [])})
    for d in distances:
        plt.figure(figsize=(8, 5))
        for strategy, items in grouped.items():
            for item in items:
                if abs(float(item["distance"]) - d) > 1e-9:
                    continue
                traj = item["result"].get("trajectory", [])
                if not traj:
                    continue
                tau = np.array([pt["tau"] for pt in traj], dtype=float)
                en = np.array([pt["E"] for pt in traj], dtype=float)
                err = np.array([pt.get("E_err", 0.0) for pt in traj], dtype=float)
                plt.plot(tau, en, linewidth=2, label=strategy.upper())
                plt.fill_between(tau, en - err, en + err, alpha=0.15)

        plt.title(f"Imaginary-Time Trajectory E(tau), d={d:g}")
        plt.xlabel("tau")
        plt.ylabel("E(tau)")
        plt.grid(alpha=0.3)
        plt.legend()
        p = out_dir / f"figure_traj_d{safe_name(d)}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        made.append(p.name)

    # 4) Runtime comparison
    labels: list[str] = []
    values: list[float] = []
    for strategy, items in grouped.items():
        sec = [float(x["summary"]["seconds_total"]) for x in items]
        if not sec:
            continue
        labels.append(strategy.upper())
        values.append(float(np.mean(sec)))

    if labels:
        plt.figure(figsize=(7, 4.5))
        bars = plt.bar(labels, values)
        for b, v in zip(bars, values):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}s", ha="center", va="bottom")
        plt.title("Average Runtime Per Strategy")
        plt.ylabel("seconds")
        plt.grid(axis="y", alpha=0.3)
        p = out_dir / "figure_runtime.png"
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        made.append(p.name)

    return made


def write_test_plan(path: Path) -> None:
    lines = [
        "# Imaginary-Time TDSE Test Plan",
        "",
        "## Physics checks",
        "1. Non-interacting d=0, coulomb=off: verify E0~2 and gap~1.",
        "2. Interacting d=0, coulomb=on: verify Kohn mode gap~1.",
        "3. Intermediate d=4: compare VMC vs PINN consistency.",
        "4. Large separation d=8: check slower decay and stable E(tau).",
        "",
        "## Numerical checks",
        "1. No NaN/Inf in trajectory energies and fit outputs.",
        "2. Acceptance ratios remain in a reasonable range for eval MCMC.",
        "3. Fit methods agree within uncertainty where successful.",
        "4. Runtime and memory are tracked for each strategy.",
        "",
        "## Suggested progression",
        "1. Run smoke profile on d=0 for both strategies.",
        "2. Run tiny profile on d=0,4 for both strategies.",
        "3. Run baseline profile on d=0,4,8.",
        "4. Run production profile after baseline diagnostics are clean.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_summary_table(suite: dict) -> list[dict]:
    rows: list[dict] = []
    for item in suite.get("results", []):
        result = item.get("result", {})
        rows.append(
            {
                "strategy": item["strategy"],
                "d": float(item["distance"]),
                "E_vmc": float(result.get("E_vmc", np.nan)),
                "gap_best": float(item["summary"].get("gap_best", np.nan)),
                "runtime_s": float(item["summary"].get("seconds_total", np.nan)),
            }
        )
    rows.sort(key=lambda r: (r["d"], r["strategy"]))
    return rows


def print_summary(rows: Iterable[dict]) -> None:
    print("\n" + "=" * 72)
    print("Strategy summary")
    print("=" * 72)
    print(f"{'strategy':>10s}  {'d':>6s}  {'E_vmc':>12s}  {'gap_best':>10s}  {'runtime_s':>10s}")
    print(f"{'-'*10}  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}")
    for r in rows:
        print(
            f"{r['strategy']:>10s}  {r['d']:6.2f}  {r['E_vmc']:12.6f}  "
            f"{r['gap_best']:10.5f}  {r['runtime_s']:10.2f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified imaginary-time TDSE strategy runner")
    p.add_argument("--mode", choices=["plan", "run", "smoke", "clean"], default="plan")
    p.add_argument("--profile", choices=["smoke", "tiny", "baseline", "production"], default="tiny")
    p.add_argument("--strategies", default="vmc,pinn", help="Comma-separated: vmc,pinn")
    p.add_argument("--distances", default="0,4,8", help="Comma-separated distances")
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--tau-max", type=float, default=5.0)
    p.add_argument("--coulomb", action="store_true", help="Enable interaction")
    p.add_argument("--no-coulomb", action="store_true", help="Disable interaction")
    p.add_argument("--tag", default="")
    p.add_argument("--archive-dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    coulomb = True
    if args.no_coulomb:
        coulomb = False
    elif args.coulomb:
        coulomb = True

    distances = parse_distances(args.distances)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in {"vmc", "pinn"}:
            raise ValueError(f"Unsupported strategy: {s}")

    if args.mode == "plan":
        plan_path = RUNS_ROOT / "imag_time_test_plan.md"
        write_test_plan(plan_path)
        print(f"Wrote test plan: {plan_path}")
        return

    if args.mode == "clean":
        report = archive_legacy_results(dry_run=args.archive_dry_run)
        out = RUNS_ROOT / f"cleanup_{utc_stamp()}.json"
        save_json(report, out)
        print(f"Cleanup report: {out}")
        print(json.dumps(report, indent=2))
        return

    profile = "smoke" if args.mode == "smoke" else args.profile
    suite_cfg = SuiteConfig(
        profile=profile,
        distances=distances,
        coulomb=coulomb,
        tau_max=float(args.tau_max),
        omega=float(args.omega),
        strategies=strategies,
        tag=args.tag,
    )

    run_dir = RUNS_ROOT / f"{utc_stamp()}_{profile}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(asdict(suite_cfg), run_dir / "suite_config.json")
    suite = run_suite(suite_cfg, run_dir)

    figures = make_plots(suite, run_dir)
    summary_rows = make_summary_table(suite)
    save_json(summary_rows, run_dir / "summary_table.json")
    print_summary(summary_rows)

    run_manifest = {
        "run_dir": str(run_dir),
        "figures": figures,
        "summary_table": "summary_table.json",
        "suite_results": "suite_results.json",
    }
    save_json(run_manifest, run_dir / "run_manifest.json")

    print("\nGenerated files:")
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
