#!/usr/bin/env python3
"""Phase 2A.8 trajectory analyser: spin correlators + effective ``J_ij`` per outer step.

For every centre checkpoint of an inverse-design run we extract:

  * The Mott spin amplitudes ``c_sigma`` per outer step.
  * The full ``<S_i.S_j>`` matrix (all ``C(N, 2)`` pairs).
  * The effective Heisenberg fit ``J_{ij}`` (covariance method, NN-only basis
    by default for N>=4 to keep the search well-conditioned; all pairs for N=2,3).
  * The fit overlap ``|<c|psi_0(H_eff)>|`` and the relative residual variance.
  * The total energy and target value at the same step.

Outputs:

  * CSV  — one row per step with the scalar metrics (target, energy, NN bonds,
    end-to-end correlator, Heisenberg overlap, residual variance, ...).
  * NPZ  — full per-step arrays (amplitude vectors, full correlator matrices,
    full ``J_{ij}`` matrices) for downstream plotting / publication-grade
    figures.
  * PNG  — 6-panel matplotlib summary (target, energy, end-to-end correlator,
    Heisenberg overlap, ``J`` matrix at the final step, correlator heatmap at
    the final step).

Works for any N supported by the Mott spin-amplitude extractor. Particularly
useful for Phase 2A flagship trajectories (N=4) and Phase 2B engineering runs
(N=8 spin-correlator engineering).

Usage
-----

    PYTHONPATH=src python3.11 scripts/amplitude_evolution.py \\
        --run-dir results/inverse_design/n4_flagship_p2a_aggressive \\
        --out-csv results/inverse_design/n4_flagship_p2a_aggressive/amplitude_evolution.csv \\
        --out-npz results/inverse_design/n4_flagship_p2a_aggressive/amplitude_evolution.npz \\
        --out-png results/inverse_design/n4_flagship_p2a_aggressive/amplitude_evolution.png

If ``--effJ-pairs`` is omitted: NN-only basis for chains of length >= 4, all
pairs for N=2 and N=3.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observables.checkpoint_loader import load_wavefunction_from_dir  # noqa: E402
from observables.effective_heisenberg import (  # noqa: E402
    fit_effective_heisenberg,
    spin_pair_correlator,
)
from observables.spin_amplitude_entanglement import (  # noqa: E402
    extract_spin_amplitudes,
)


LOGGER = logging.getLogger("amplitude_evolution")


def _load_history(run_dir: Path) -> list[dict]:
    """Read either ``history.json`` (in-progress run) or ``optimal_geometry.json``."""
    history_path = run_dir / "history.json"
    if history_path.exists():
        with history_path.open() as fh:
            return list(json.load(fh))
    opt_path = run_dir / "optimal_geometry.json"
    if not opt_path.exists():
        raise FileNotFoundError(f"No history.json or optimal_geometry.json in {run_dir}.")
    with opt_path.open() as fh:
        payload = json.load(fh)
    return list(payload["history"])


def _resolve_centre_dir(run_dir: Path, rec: dict) -> Path:
    """Pick the centre training dir (handle multi-sector runs by preferring 'singlet')."""
    centre = rec.get("centre_result_dir") or rec.get("center_result_dir")
    if centre is not None:
        return Path(centre)
    sec_dirs = rec.get("sector_result_dirs") or {}
    if "singlet" in sec_dirs:
        return Path(sec_dirs["singlet"])
    if sec_dirs:
        return Path(next(iter(sec_dirs.values())))
    return run_dir / f"step{rec['step']:03d}_centre"


def _default_effJ_pairs(n_wells: int) -> list[tuple[int, int]]:
    if n_wells <= 3:
        return [(i, j) for i in range(n_wells) for j in range(i + 1, n_wells)]
    # NN-only basis: gives a 2D null space for chains, well-conditioned.
    return [(i, i + 1) for i in range(n_wells - 1)]


def _parse_pair(arg: str) -> tuple[int, int]:
    parts = arg.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Pair must be 'i,j', got '{arg}'.")
    a, b = int(parts[0]), int(parts[1])
    if a == b:
        raise argparse.ArgumentTypeError(f"Pair must have i != j, got '{arg}'.")
    if a > b:
        a, b = b, a
    return (a, b)


def analyse_run(
    run_dir: Path,
    *,
    effJ_pairs: list[tuple[int, int]] | None = None,
    spin_sector: tuple[int, int] | None = None,
    device: str | None = None,
) -> dict:
    history = _load_history(run_dir)
    if not history:
        raise RuntimeError(f"Empty history in {run_dir}.")

    rows: list[dict] = []
    per_step_data: list[dict] = []

    n_wells_global = None

    for rec in history:
        step = int(rec["step"])
        centre_dir = _resolve_centre_dir(run_dir, rec)
        if not centre_dir.exists():
            LOGGER.warning("step %d: centre dir %s missing; skipping.", step, centre_dir)
            continue

        try:
            loaded = load_wavefunction_from_dir(centre_dir, device=device)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("step %d: failed to load checkpoint: %s", step, exc)
            continue

        try:
            payload = extract_spin_amplitudes(loaded, spin_sector=spin_sector)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("step %d: amplitude extraction failed: %s", step, exc)
            continue

        n_wells = payload.n_wells
        if n_wells_global is None:
            n_wells_global = n_wells
        elif n_wells_global != n_wells:
            LOGGER.warning(
                "step %d: n_wells changed (%d != %d), aborting trajectory.",
                step, n_wells, n_wells_global,
            )
            break

        effJ_pairs_step = effJ_pairs if effJ_pairs is not None else _default_effJ_pairs(n_wells)

        # Direct spin-spin correlators: full C(N, 2) matrix.
        all_pairs = [(i, j) for i in range(n_wells) for j in range(i + 1, n_wells)]
        corr_payload = spin_pair_correlator(payload, pairs=all_pairs)
        c_matrix = np.asarray(corr_payload["c_matrix"], dtype=np.float64)

        # Effective Heisenberg fit on the chosen pair basis.
        try:
            fit = fit_effective_heisenberg(payload, pairs=effJ_pairs_step)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("step %d: effective_J fit failed: %s — recording correlators only.", step, exc)
            fit = None

        amps = payload.normalised().astype(np.float64)

        nn_pairs = [(i, i + 1) for i in range(n_wells - 1)]
        nn_correlators = [float(c_matrix[i, j]) for (i, j) in nn_pairs]

        end_pair = (0, n_wells - 1)
        c_end = float(c_matrix[end_pair[0], end_pair[1]])
        c_nn_min = float(min(nn_correlators)) if nn_correlators else float("nan")
        c_nn_max = float(max(nn_correlators)) if nn_correlators else float("nan")
        c_nn_mean = float(np.mean(nn_correlators)) if nn_correlators else float("nan")

        row: dict = {
            "step": step,
            "theta": rec.get("theta"),
            "target": rec.get("target"),
            "energy": rec.get("energy"),
            "wells": rec.get("wells"),
            "C_end": c_end,
            "C_NN_min": c_nn_min,
            "C_NN_max": c_nn_max,
            "C_NN_mean": c_nn_mean,
        }
        for k, (i, j) in enumerate(nn_pairs):
            row[f"C_{i}{j}"] = nn_correlators[k]
        if n_wells >= 3:
            row["C_NNN_0_2"] = float(c_matrix[0, 2])
        row["C_end_to_end"] = c_end

        if fit is not None:
            j_matrix = fit.j_matrix
            j_vec = fit.j_vector.tolist()
            row["effJ_residual_variance"] = float(fit.residual_variance)
            row["effJ_relative_residual"] = float(fit.relative_residual)
            row["effJ_overlap"] = float(fit.overlap_with_ground)
            row["effJ_energy_split"] = float(fit.energy_split)
            for k, (i, j) in enumerate(effJ_pairs_step):
                row[f"J_{i}{j}"] = float(j_vec[k])

        rows.append(row)

        per_step_data.append({
            "step": step,
            "theta": np.asarray(rec.get("theta", []), dtype=np.float64),
            "target": float(rec.get("target", float("nan"))),
            "energy": float(rec.get("energy", float("nan"))),
            "amps": amps,
            "patterns": [list(p) for p in payload.pattern],
            "c_matrix": c_matrix,
            "j_matrix": (fit.j_matrix if fit is not None else np.zeros((n_wells, n_wells))),
            "j_vector": (np.asarray(fit.j_vector, dtype=np.float64) if fit is not None else np.zeros(0)),
            "effJ_pairs": [list(p) for p in effJ_pairs_step],
            "wells": np.asarray(rec.get("wells", []), dtype=np.float64),
            "effJ_overlap": float(fit.overlap_with_ground) if fit is not None else float("nan"),
            "effJ_relative_residual": float(fit.relative_residual) if fit is not None else float("nan"),
        })

    return {
        "n_wells": n_wells_global,
        "rows": rows,
        "per_step": per_step_data,
        "run_dir": str(run_dir),
    }


def write_csv(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        LOGGER.warning("No rows to write to %s.", out_csv)
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row_out = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, (list, tuple, np.ndarray)):
                    row_out[k] = json.dumps(np.asarray(v).tolist())
                else:
                    row_out[k] = v
            writer.writerow(row_out)


def write_npz(per_step: list[dict], out_npz: Path) -> None:
    if not per_step:
        LOGGER.warning("No per-step data to write to %s.", out_npz)
        return
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    pack: dict = {}

    n_steps = len(per_step)
    n_wells = per_step[0]["c_matrix"].shape[0]

    pack["steps"] = np.array([d["step"] for d in per_step], dtype=np.int32)
    pack["targets"] = np.array([d["target"] for d in per_step], dtype=np.float64)
    pack["energies"] = np.array([d["energy"] for d in per_step], dtype=np.float64)
    pack["effJ_overlaps"] = np.array([d["effJ_overlap"] for d in per_step], dtype=np.float64)
    pack["effJ_relative_residuals"] = np.array(
        [d["effJ_relative_residual"] for d in per_step], dtype=np.float64
    )

    pack["c_matrices"] = np.stack([d["c_matrix"] for d in per_step], axis=0)
    pack["j_matrices"] = np.stack([d["j_matrix"] for d in per_step], axis=0)

    # Pad amplitudes / patterns to max P, since spin sector is fixed across steps.
    max_p = max(d["amps"].size for d in per_step)
    amps_arr = np.full((n_steps, max_p), np.nan, dtype=np.float64)
    for k, d in enumerate(per_step):
        amps_arr[k, : d["amps"].size] = d["amps"]
    pack["amplitudes"] = amps_arr
    pack["patterns"] = np.array(per_step[0]["patterns"], dtype=np.int8)

    # theta/wells may have different shapes across runs; we serialise as object arrays.
    theta_dim = max(d["theta"].size for d in per_step)
    theta_arr = np.full((n_steps, theta_dim), np.nan, dtype=np.float64)
    for k, d in enumerate(per_step):
        theta_arr[k, : d["theta"].size] = d["theta"]
    pack["thetas"] = theta_arr

    wells_arr = np.full((n_steps, n_wells, 2), np.nan, dtype=np.float64)
    for k, d in enumerate(per_step):
        if d["wells"].ndim == 2 and d["wells"].shape == (n_wells, 2):
            wells_arr[k] = d["wells"]
    pack["wells"] = wells_arr

    np.savez_compressed(out_npz, **pack)


def render_png(per_step: list[dict], out_png: Path, *, run_label: str | None = None) -> None:
    if not per_step:
        LOGGER.warning("No per-step data to render in %s.", out_png)
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available; skipping PNG.")
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    n_wells = per_step[0]["c_matrix"].shape[0]
    steps = [d["step"] for d in per_step]
    targets = [d["target"] for d in per_step]
    energies = [d["energy"] for d in per_step]

    nn_pairs = [(i, i + 1) for i in range(n_wells - 1)]
    end_pair = (0, n_wells - 1)
    c_end_traj = [float(d["c_matrix"][end_pair[0], end_pair[1]]) for d in per_step]
    nn_traj = [
        [float(d["c_matrix"][i, j]) for (i, j) in nn_pairs] for d in per_step
    ]
    nn_traj_arr = np.array(nn_traj)  # (n_steps, n_NN)
    overlaps = [d["effJ_overlap"] for d in per_step]
    residuals = [d["effJ_relative_residual"] for d in per_step]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Inverse-design trajectory analysis"
        + (f" — {run_label}" if run_label else ""),
        fontsize=13,
    )

    # 1. Target evolution.
    ax = axes[0, 0]
    ax.plot(steps, targets, "o-", color="#1f77b4")
    ax.set_xlabel("outer step")
    ax.set_ylabel("target T")
    ax.set_title("(a) Target value")
    ax.grid(alpha=0.3)

    # 2. Energy.
    ax = axes[0, 1]
    ax.plot(steps, energies, "s-", color="#d62728")
    ax.set_xlabel("outer step")
    ax.set_ylabel("E (Ha)")
    ax.set_title("(b) Total energy")
    ax.grid(alpha=0.3)

    # 3. NN correlators evolution + end-to-end overlay.
    ax = axes[0, 2]
    cmap = plt.get_cmap("viridis")
    for k, (i, j) in enumerate(nn_pairs):
        col = cmap(k / max(1, len(nn_pairs) - 1))
        ax.plot(steps, nn_traj_arr[:, k], "o-", color=col, label=f"<S_{i}.S_{j}>", markersize=4)
    ax.plot(steps, c_end_traj, "D-", color="black", label=f"<S_{end_pair[0]}.S_{end_pair[1]}>", markersize=5)
    ax.set_xlabel("outer step")
    ax.set_ylabel(r"$\langle S_i \cdot S_j \rangle$")
    ax.set_title("(c) Spin correlators (NN + end-to-end)")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axhline(-0.75, color="red", linewidth=0.4, linestyle=":", label="singlet limit")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # 4. Heisenberg fit overlap + relative residual.
    ax = axes[1, 0]
    ax.plot(steps, overlaps, "o-", color="#2ca02c", label="|<c|psi_0>|")
    ax.set_xlabel("outer step")
    ax.set_ylabel("Heisenberg overlap")
    ax.set_title("(d) Heisenberg overlap")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.semilogy(steps, np.maximum(np.abs(residuals), 1e-12), "x:", color="#9467bd", label="|rel. residual|")
    ax2.set_ylabel("relative residual (log)", color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")

    # 5. Final-step C matrix heatmap.
    ax = axes[1, 1]
    final_C = per_step[-1]["c_matrix"]
    vmax = float(np.max(np.abs(final_C)))
    im = ax.imshow(final_C, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_xticks(range(n_wells))
    ax.set_yticks(range(n_wells))
    ax.set_title(f"(e) Final $\\langle S_i.S_j\\rangle$ (step {steps[-1]})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n_wells):
        for j in range(n_wells):
            v = final_C[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=("white" if abs(v) > 0.4 * vmax else "black"),
                    fontsize=7)

    # 6. Final-step J matrix heatmap (NN-only fit).
    ax = axes[1, 2]
    final_J = per_step[-1]["j_matrix"]
    vmax_j = float(np.max(np.abs(final_J))) or 1.0
    im = ax.imshow(final_J, vmin=-vmax_j, vmax=vmax_j, cmap="RdBu_r")
    ax.set_xticks(range(n_wells))
    ax.set_yticks(range(n_wells))
    ax.set_title(f"(f) Final $J_{{ij}}$ fit (step {steps[-1]})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n_wells):
        for j in range(n_wells):
            v = final_J[i, j]
            if abs(v) > 1e-3:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color=("white" if abs(v) > 0.4 * vmax_j else "black"),
                        fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        print("No rows to summarise.")
        return
    print()
    print("=" * 78)
    print(f"  Trajectory summary ({len(rows)} steps)")
    print("=" * 78)
    fields = ["step", "target", "energy", "C_end_to_end", "C_NN_mean", "effJ_overlap", "effJ_relative_residual"]
    header = "  " + "".join(f"{f:>22s}" for f in fields)
    print(header)
    print("  " + "-" * (22 * len(fields)))
    for r in rows:
        line = "  "
        for f in fields:
            v = r.get(f, "")
            if isinstance(v, float):
                if abs(v) < 1e-3 and v != 0:
                    line += f"{v:>22.3e}"
                else:
                    line += f"{v:>22.5f}"
            elif v is None:
                line += f"{'N/A':>22s}"
            else:
                line += f"{str(v):>22s}"
        print(line)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--out-csv", type=Path, default=None,
        help="CSV output path. Default: <run-dir>/amplitude_evolution.csv unless --no-default-outputs.",
    )
    parser.add_argument(
        "--out-npz", type=Path, default=None,
        help="NPZ output path. Default: <run-dir>/amplitude_evolution.npz unless --no-default-outputs.",
    )
    parser.add_argument(
        "--out-png", type=Path, default=None,
        help="PNG output path. Default: <run-dir>/amplitude_evolution.png unless --no-default-outputs.",
    )
    parser.add_argument(
        "--no-default-outputs", action="store_true",
        help="Disable auto-defaulting of out-csv/npz/png into the run directory.",
    )
    parser.add_argument(
        "--effJ-pairs", type=_parse_pair, nargs="+", default=None,
        help="Pairs to use for the H_eff fit basis. Default: NN-only for chains.",
    )
    parser.add_argument(
        "--spin-sector", type=int, nargs=2, metavar=("N_UP", "N_DOWN"),
        default=None,
        help="Override spin sector for amplitude extraction.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    if not args.no_default_outputs:
        if args.out_csv is None:
            args.out_csv = args.run_dir / "amplitude_evolution.csv"
        if args.out_npz is None:
            args.out_npz = args.run_dir / "amplitude_evolution.npz"
        if args.out_png is None:
            args.out_png = args.run_dir / "amplitude_evolution.png"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_dir = args.run_dir
    if not run_dir.exists():
        parser.error(f"--run-dir {run_dir} does not exist.")

    spin_sector = (
        (int(args.spin_sector[0]), int(args.spin_sector[1]))
        if args.spin_sector is not None
        else None
    )
    out = analyse_run(
        run_dir,
        effJ_pairs=args.effJ_pairs,
        spin_sector=spin_sector,
        device=args.device,
    )

    rows = out["rows"]
    _print_summary(rows)

    if args.out_csv is not None:
        write_csv(rows, args.out_csv)
        print(f"Wrote CSV → {args.out_csv}")
    if args.out_npz is not None:
        write_npz(out["per_step"], args.out_npz)
        print(f"Wrote NPZ → {args.out_npz}")
    if args.out_png is not None:
        render_png(
            out["per_step"], args.out_png,
            run_label=Path(out["run_dir"]).name,
        )
        print(f"Wrote PNG → {args.out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
