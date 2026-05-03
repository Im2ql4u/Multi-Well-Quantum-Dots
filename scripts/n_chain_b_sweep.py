#!/usr/bin/env python3
"""Sweep an N-well chain over the magnetic-field magnitude ``B`` at fixed
geometry and compare PINN spin observables against the OBC Heisenberg
reference.

Geometry is taken **as-is** from the base YAML config (so the user can supply
either a uniform chain, a `dimer_pair_n8` snapshot from an inverse-design
run, etc.). For each requested ``B`` value we:

  1. Write a per-B config with ``system.B_magnitude = B``.
  2. Train via ``scripts/run_two_stage_ground_state.py`` (Stage A only by
     default, mirroring the inverse-design lane).
  3. Reload the trained checkpoint and extract Mott spin amplitudes,
     bipartite well-set entanglement, and the full ``<S_i.S_j>`` matrix.
  4. Diagonalise the OBC Heisenberg Hamiltonian for the same ``N`` and
     ``n_down`` and report the PINN/Heisenberg overlap, residual, and
     reference entanglement (uniform J = 1).

Outputs (under ``--out-dir``):

  * ``B_sweep.csv`` — one row per ``B`` with the scalar metrics.
  * ``B_sweep.json`` — full per-B payload (correlator matrices, eigenvalues,
    amplitudes).
  * ``B_sweep.png`` — multi-panel summary figure.

This is the Phase 3B background task: a magnetic characterisation of the
N=8 chain at fixed d=4 (or, optionally, at a post-inverse-design SSH
geometry) to map out how the magnetic field reshapes the ground-state
spin observables. The full B-sweep (5 values × ~30 min on a free 2080 Ti)
fits in a single overnight slot.

Usage
-----

::

    PYTHONPATH=src CUDA_MANUAL_DEVICE=3 \\
        python3.11 scripts/n_chain_b_sweep.py \\
            --config configs/one_per_well/n8_invdes_fast_s42.yaml \\
            --b-values 0.0 0.05 0.1 0.2 0.5 \\
            --stage-a-epochs 1500 \\
            --stage-a-strategy improved_self_residual \\
            --stage-a-min-energy 999.0 \\
            --out-dir results/b_sweep/n8_uniform_d4_s42

The trainer is invoked through ``run_two_stage_ground_state.py`` so all the
existing safety nets (Stage A gating, summary JSON emission, unbuffered
trainer logs) apply.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the diagnostic core from the d-sweep so we get exactly the same
# correlator / Heisenberg-overlap / entanglement metrics for free.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from n_chain_d_sweep import analyse_one  # noqa: E402

LOGGER = logging.getLogger("n_chain_b_sweep")


def _b_tag(b: float) -> str:
    s = f"{b:g}"
    return s.replace(".", "p").replace("-", "m")


def _write_per_b_config(base_cfg: dict[str, Any], b: float, out_path: Path) -> None:
    cfg = copy.deepcopy(base_cfg)
    system = cfg.setdefault("system", {})
    system["B_magnitude"] = float(b)
    cfg["run_name"] = f"{cfg.get('run_name', 'b_sweep')}_b{_b_tag(b)}"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _resolve_result_dir_from_summary(summary_json: Path) -> Path:
    payload = json.loads(summary_json.read_text())
    stage_b = payload.get("stage_b")
    if stage_b is not None:
        return Path(stage_b["result_dir"])
    return Path(payload["stage_a"]["result_dir"])


def _train_one(
    *,
    cfg_path: Path,
    out_dir: Path,
    tag: str,
    stage_a_epochs: int,
    stage_b_epochs: int,
    stage_a_strategy: str,
    stage_a_min_energy: float,
    seed_override: int | None,
    device: str,
) -> Path:
    summary_json = out_dir / f"summary_{tag}.json"
    log_path = out_dir / f"train_{tag}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    if device.startswith("cuda:"):
        env["CUDA_MANUAL_DEVICE"] = device.replace("cuda:", "")
    cmd = [
        sys.executable, "-u", str(REPO / "scripts" / "run_two_stage_ground_state.py"),
        "--config", str(cfg_path),
        "--stage-a-strategy", stage_a_strategy,
        "--stage-a-epochs", str(int(stage_a_epochs)),
        "--stage-b-epochs", str(int(stage_b_epochs)),
        "--stage-a-min-energy", str(float(stage_a_min_energy)),
        "--summary-json", str(summary_json),
    ]
    if seed_override is not None:
        cmd += ["--seed-override", str(int(seed_override))]
    LOGGER.info("[%s] launching trainer (log: %s)", tag, log_path.name)
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Trainer for tag={tag!r} exited with code {proc.returncode}; see {log_path}"
        )
    return _resolve_result_dir_from_summary(summary_json)


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        LOGGER.warning("No rows to write to %s.", out_path)
        return
    fields = [
        "B", "n_wells", "n_down", "energy",
        "S_pinn", "S_heis", "S_excess",
        "overlap", "residual_l2",
        "C_NN_min_pinn", "C_NN_max_pinn", "C_end_to_end_pinn",
        "C_NN_min_heis", "C_NN_max_heis", "C_end_to_end_heis",
    ]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, float("nan")) for k in fields})


def render_png(rows: list[dict[str, Any]], out_path: Path, *, run_label: str) -> None:
    if not rows:
        LOGGER.warning("No rows to plot in %s.", out_path)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_sorted = sorted(rows, key=lambda r: r["B"])
    B = np.array([r["B"] for r in rows_sorted], dtype=float)
    S_pinn = np.array([r["S_pinn"] for r in rows_sorted], dtype=float)
    S_heis = np.array([r["S_heis"] for r in rows_sorted], dtype=float)
    overlap = np.array([r["overlap"] for r in rows_sorted], dtype=float)
    res = np.array([r["residual_l2"] for r in rows_sorted], dtype=float)
    energy = np.array([r["energy"] for r in rows_sorted], dtype=float)
    C_end_pinn = np.array([r["C_end_to_end_pinn"] for r in rows_sorted], dtype=float)
    C_end_heis = np.array([r["C_end_to_end_heis"] for r in rows_sorted], dtype=float)
    C_nn_min_pinn = np.array([r["C_NN_min_pinn"] for r in rows_sorted], dtype=float)
    C_nn_min_heis = np.array([r["C_NN_min_heis"] for r in rows_sorted], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"B-sweep: {run_label}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(B, S_pinn, "o-", color="#1f77b4", label="S(PINN)")
    ax.plot(B, S_heis, "s--", color="#7f7f7f", label="S(Heisenberg, uniform J)")
    ax.set_xlabel("B (a.u.)")
    ax.set_ylabel("bipartite von Neumann entropy")
    ax.set_title("(a) Spin entanglement vs B")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(B, overlap, "o-", color="#2ca02c")
    ax.set_xlabel("B (a.u.)")
    ax.set_ylabel(r"$|\langle c_{\rm PINN}|c_{\rm Heis}\rangle|$")
    ax.set_title("(b) Heisenberg overlap")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.semilogy(B, np.maximum(res, 1e-12), "x:", color="#9467bd", label="L2 residual")
    ax2.set_ylabel("L2 residual (log)", color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")

    ax = axes[1, 0]
    ax.plot(B, C_end_pinn, "o-", color="#d62728", label="<S_0.S_{N-1}> PINN")
    ax.plot(B, C_end_heis, "s--", color="#7f7f7f", label="<S_0.S_{N-1}> Heisenberg")
    ax.plot(B, C_nn_min_pinn, "^-", color="#ff7f0e", label="min NN PINN")
    ax.plot(B, C_nn_min_heis, "v--", color="#bcbd22", label="min NN Heisenberg")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("B (a.u.)")
    ax.set_ylabel(r"$\langle S_i\cdot S_j\rangle$")
    ax.set_title("(c) Spin correlators")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(B, energy, "o-", color="#17becf")
    ax.set_xlabel("B (a.u.)")
    ax.set_ylabel("E (Ha)")
    ax.set_title("(d) Total energy")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Base YAML config (architecture + training + geometry).")
    parser.add_argument("--b-values", type=float, nargs="+", required=True, metavar="B",
                        help="Magnetic field magnitudes (in atomic units).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None,
                        help="Override CUDA device for training (e.g. cuda:3). "
                             "Default: read from CUDA_MANUAL_DEVICE env or fall back to cuda:0.")
    parser.add_argument("--stage-a-epochs", type=int, default=1500)
    parser.add_argument("--stage-b-epochs", type=int, default=1)
    parser.add_argument("--stage-a-strategy", type=str, default="improved_self_residual",
                        choices=["auto", "guided", "self_residual", "singlet_self_residual", "improved_self_residual"])
    parser.add_argument("--stage-a-min-energy", type=float, default=999.0)
    parser.add_argument("--seed-override", type=int, default=None)
    parser.add_argument("--set-a", type=int, nargs="+", default=None,
                        help="Indices defining the bipartite cut for entanglement (default: left half).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip training a given B if its summary JSON already exists.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = yaml.safe_load(args.config.read_text())

    device = args.device
    if device is None:
        env_dev = os.environ.get("CUDA_MANUAL_DEVICE")
        device = f"cuda:{env_dev}" if env_dev is not None else "cuda:0"

    rows: list[dict[str, Any]] = []
    full_payloads: list[dict[str, Any]] = []
    for b in args.b_values:
        tag = f"b{_b_tag(b)}"
        cfg_path = args.out_dir / f"cfg_{tag}.yaml"
        summary_path = args.out_dir / f"summary_{tag}.json"
        _write_per_b_config(base_cfg, b, cfg_path)

        if args.skip_existing and summary_path.exists():
            LOGGER.info("[%s] skipping training (summary exists)", tag)
            ckpt = _resolve_result_dir_from_summary(summary_path)
        else:
            t0 = time.time()
            ckpt = _train_one(
                cfg_path=cfg_path,
                out_dir=args.out_dir,
                tag=tag,
                stage_a_epochs=args.stage_a_epochs,
                stage_b_epochs=args.stage_b_epochs,
                stage_a_strategy=args.stage_a_strategy,
                stage_a_min_energy=args.stage_a_min_energy,
                seed_override=args.seed_override,
                device=device,
            )
            LOGGER.info("[%s] training done in %.0f s -> %s", tag, time.time() - t0, ckpt)

        summary_payload = json.loads(summary_path.read_text())
        if summary_payload.get("stage_b") is not None:
            energy = float(summary_payload["stage_b"]["result"]["final_energy"])
        else:
            energy = float(summary_payload["stage_a"]["result"]["final_energy"])

        diag = analyse_one(ckpt, set_a=args.set_a, device=device)
        diag["B"] = float(b)
        diag["energy"] = energy
        full_payloads.append(diag)

        n_wells = int(diag["n_wells"])
        C_pinn = np.array(diag["pinn"]["C_matrix"])
        C_heis = np.array(diag["heisenberg"]["C_matrix"])
        nn_pinn = np.array([C_pinn[k, k + 1] for k in range(n_wells - 1)])
        nn_heis = np.array([C_heis[k, k + 1] for k in range(n_wells - 1)])
        rows.append({
            "B": float(b),
            "n_wells": n_wells,
            "n_down": int(diag["n_down"]),
            "energy": energy,
            "S_pinn": diag["pinn"]["von_neumann_entropy"],
            "S_heis": diag["heisenberg"]["von_neumann_entropy"],
            "S_excess": diag["pinn"]["von_neumann_entropy"] - diag["heisenberg"]["von_neumann_entropy"],
            "overlap": diag["heisenberg"]["overlap"],
            "residual_l2": diag["heisenberg"]["residual_l2"],
            "C_NN_min_pinn": float(np.min(nn_pinn)),
            "C_NN_max_pinn": float(np.max(nn_pinn)),
            "C_end_to_end_pinn": float(C_pinn[0, -1]),
            "C_NN_min_heis": float(np.min(nn_heis)),
            "C_NN_max_heis": float(np.max(nn_heis)),
            "C_end_to_end_heis": float(C_heis[0, -1]),
        })

    print()
    print("=" * 92)
    print(f"  B-sweep summary  ({len(rows)} B values, device={device})")
    print("=" * 92)
    print(
        f"  {'B':>6} {'E':>10} {'S_pinn':>9} {'S_heis':>9} "
        f"{'overlap':>9} {'res_L2':>10} {'C_end(pinn)':>12} {'C_end(heis)':>12}"
    )
    for r in sorted(rows, key=lambda x: x["B"]):
        print(
            f"  {r['B']:>6.3f} {r['energy']:>10.4f} {r['S_pinn']:>9.4f} {r['S_heis']:>9.4f} "
            f"{r['overlap']:>9.5f} {r['residual_l2']:>10.3e} {r['C_end_to_end_pinn']:>12.5f} {r['C_end_to_end_heis']:>12.5f}"
        )
    print()

    write_csv(rows, args.out_dir / "B_sweep.csv")
    (args.out_dir / "B_sweep.json").write_text(json.dumps({"rows": rows, "full": full_payloads}, indent=2))
    render_png(rows, args.out_dir / "B_sweep.png", run_label=args.out_dir.name)
    LOGGER.info("Wrote: %s, %s, %s",
                args.out_dir / "B_sweep.csv", args.out_dir / "B_sweep.png", args.out_dir / "B_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
