#!/usr/bin/env python3
"""Phase 4 wrap-up: plot the deterministic non-MCMC NQS scaling curve.

Aggregates the validated ground-state energies from the various per-N
configurations into a single E vs N (and wall-time vs N) plot.

Usage
-----
::

    PYTHONPATH=src python3.11 scripts/plot_scaling_curve.py \\
        --out-png results/scaling/scaling_curve.png \\
        --out-csv results/scaling/scaling_curve.csv

If a particular N's summary JSON is missing, the script logs a warning and
continues; the resulting plot only contains the N values that resolved.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger("plot_scaling_curve")

REPO = Path(__file__).resolve().parent.parent

def _newest(matches: list[Path]) -> Path | None:
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _glob_newest(pattern: str) -> list[Path]:
    """Return the newest single match (or empty) for a glob relative to REPO."""
    matches = list(REPO.glob(pattern))
    newest = _newest(matches)
    return [newest] if newest is not None else []


DEFAULT_SOURCES: dict[int, list[Path]] = {
    2: _glob_newest("results/p4_n2_singlet_d4_s42__stageB_noref_*/result.json"),
    3: _glob_newest("results/p4_n3_nonmcmc_residual_anneal_s42__stageB_noref_*/result.json"),
    4: _glob_newest("results/p4_n4_nonmcmc_residual_anneal_s42__stageB_noref_*/result.json"),
    8: [REPO / "results/d_sweep/n8_uniform_s42/summary_d4.json"],
    12: [REPO / "results/scaling/n12_grid_d6_s42_seed314_self_resid_summary.json"],
    16: [REPO / "results/scaling/n16_grid_d6_s42_seed314_self_resid_summary.json"],
}


def _resolve_summary(N: int, candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    matches = sorted((REPO / "results/scaling").glob(f"n{N}_*summary.json"))
    if matches:
        return matches[0]
    matches = sorted((REPO / "results").glob(f"*n{N}*summary*.json"))
    if matches:
        return matches[0]
    return None


def _extract_energy_var_walltime(payload: dict) -> tuple[float | None, float | None, float | None]:
    """Try several known schemas to find (E_final, var_final, wall_time_seconds)."""
    sb = payload.get("stage_b")
    sa = payload.get("stage_a")

    if sb and isinstance(sb, dict) and "result" in sb and sb["result"]:
        res = sb["result"]
        return (
            float(res.get("final_energy", float("nan"))),
            float(res.get("final_energy_var", float("nan"))),
            float(res.get("wall_time_seconds", float("nan"))) if "wall_time_seconds" in res else None,
        )
    if sa and isinstance(sa, dict) and "result" in sa and sa["result"]:
        res = sa["result"]
        return (
            float(res.get("final_energy", float("nan"))),
            float(res.get("final_energy_var", float("nan"))),
            float(res.get("wall_time_seconds", float("nan"))) if "wall_time_seconds" in res else None,
        )
    if "final_energy" in payload:
        return (
            float(payload.get("final_energy", float("nan"))),
            float(payload.get("final_energy_var", float("nan"))),
            float(payload.get("wall_time_seconds", float("nan"))) if "wall_time_seconds" in payload else None,
        )
    return None, None, None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-png",
        type=Path,
        default=REPO / "results/scaling/scaling_curve.png",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "results/scaling/scaling_curve.csv",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for N in sorted(DEFAULT_SOURCES.keys()):
        path = _resolve_summary(N, DEFAULT_SOURCES[N])
        if path is None:
            LOGGER.warning("[N=%d] no summary JSON found; skipping", N)
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[N=%d] could not parse %s: %s", N, path, exc)
            continue
        E, var, wt = _extract_energy_var_walltime(payload)
        if E is None:
            LOGGER.warning("[N=%d] could not extract energy from %s", N, path)
            continue
        rows.append({
            "N": N,
            "summary_path": str(path),
            "final_energy_Ha": E,
            "final_energy_var": var if var is not None else float("nan"),
            "wall_time_seconds": wt if wt is not None else float("nan"),
            "energy_per_particle_Ha": E / N if E is not None else float("nan"),
        })
        LOGGER.info(
            "[N=%d] E=%.4f Ha (var=%.2e) E/N=%.4f Ha  src=%s",
            N,
            E,
            var if var is not None else float("nan"),
            E / N if E is not None else float("nan"),
            path.relative_to(REPO),
        )

    if not rows:
        LOGGER.error("No valid rows; aborting")
        return 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "N",
                "final_energy_Ha",
                "final_energy_var",
                "wall_time_seconds",
                "energy_per_particle_Ha",
                "summary_path",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    LOGGER.info("Wrote %s", args.out_csv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib unavailable, skipping PNG: %s", exc)
        return 0

    Ns = np.array([r["N"] for r in rows])
    Es = np.array([r["final_energy_Ha"] for r in rows])
    EsN = np.array([r["energy_per_particle_Ha"] for r in rows])
    vars_ = np.array([r["final_energy_var"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    axes[0].plot(Ns, Es, "o-", lw=2)
    axes[0].set_xlabel("N (electrons / wells)")
    axes[0].set_ylabel("Final E [Ha]")
    axes[0].set_title("(a) Total E vs N")
    axes[0].grid(alpha=0.3)
    for n, e in zip(Ns, Es):
        axes[0].annotate(f"{e:.2f}", (n, e), textcoords="offset points", xytext=(5, 5), fontsize=9)

    axes[1].plot(Ns, EsN, "o-", lw=2, color="C1")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("E / N [Ha]")
    axes[1].set_title("(b) Energy per particle")
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(Ns, np.maximum(vars_, 1e-12), "o-", lw=2, color="C2")
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("Final variance")
    axes[2].set_title("(c) Variational variance vs N")
    axes[2].grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Deterministic non-MCMC NQS scaling — N \u2208 {{{', '.join(str(n) for n in Ns)}}}",
        fontsize=12,
    )
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    plt.close(fig)
    LOGGER.info("Wrote %s", args.out_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
