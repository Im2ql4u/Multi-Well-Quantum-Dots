#!/usr/bin/env python3
"""Sweep an N-well uniform chain over inter-well distance ``d`` and compare
PINN observables against the OBC Heisenberg reference.

For each requested ``d``:

  1. Build a ``custom`` system with ``N`` wells centred on the chain axis at
     positions ``[-(N-1)/2, ..., +(N-1)/2] * d`` (other system parameters
     copied from the base config).
  2. Train via :mod:`scripts.run_two_stage_ground_state` (Stage A only by
     default — Stage B is skipped via ``--stage-a-min-energy`` ≥ all
     reasonable energies, mirroring the inverse-design lane).
  3. Reload the trained checkpoint, extract Mott spin amplitudes, compute
     the bipartite well-set entanglement and the full ``<S_i.S_j>`` matrix.
  4. Diagonalise the OBC Heisenberg Hamiltonian for the same ``N`` and
     ``n_down`` and report the overlap, residual, and reference
     entanglement (uniform J = 1 by default; configurable with
     ``--bond-couplings-uniform 1.0``).

The output is

  * ``<out-dir>/d_sweep.csv`` — one row per ``d`` with the scalar metrics.
  * ``<out-dir>/d_sweep.png`` — multi-panel summary figure.
  * ``<out-dir>/d_sweep.json`` — full per-d payload (correlator matrices,
    Heisenberg eigenvalues, etc.).

Use this for the *Phase 2A bonus* N=4 d-sweep + Heisenberg cross-check, and
for the planned N=8 d-sweep on background GPUs (``--n-wells 8``).

Usage
-----

::

    PYTHONPATH=src CUDA_MANUAL_DEVICE=3 \\
        python3.11 scripts/n_chain_d_sweep.py \\
            --config configs/one_per_well/n4_invdes_baseline_s42.yaml \\
            --n-wells 4 \\
            --d-values 3 4 5 6 8 \\
            --stage-a-epochs 2500 \\
            --stage-a-strategy improved_self_residual \\
            --stage-a-min-energy 999.0 \\
            --out-dir results/d_sweep/n4_uniform_s42

The trainer is invoked via ``run_two_stage_ground_state.py`` so all the
existing safety nets (Stage A gating, lockless training, summary JSON
emission) apply.
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

from observables.checkpoint_loader import load_wavefunction_from_dir  # noqa: E402
from observables.spin_amplitude_entanglement import (  # noqa: E402
    extract_spin_amplitudes,
    well_set_bipartite_entropy,
)
from observables.heisenberg_reference import (  # noqa: E402
    align_amplitude_signs,
    heisenberg_obc_ground_state,
)
from observables.effective_heisenberg import spin_pair_correlator  # noqa: E402


LOGGER = logging.getLogger("n_chain_d_sweep")


def _build_uniform_chain_wells(n_wells: int, d: float, *, omega: float, dim: int) -> list[dict[str, Any]]:
    centre_offsets = (np.arange(n_wells) - (n_wells - 1) / 2.0) * float(d)
    wells: list[dict[str, Any]] = []
    for c in centre_offsets:
        position = [float(c)] + [0.0] * (dim - 1)
        wells.append({"center": position, "omega": float(omega), "n_particles": 1})
    return wells


def _write_per_d_config(base_cfg: dict[str, Any], n_wells: int, d: float, dim: int, out_path: Path) -> None:
    cfg = copy.deepcopy(base_cfg)
    system = cfg.setdefault("system", {})
    omega = float(system.get("wells", [{}])[0].get("omega", 1.0))
    system["type"] = "custom"
    system["dim"] = int(dim)
    system["coulomb"] = bool(system.get("coulomb", True))
    system["wells"] = _build_uniform_chain_wells(n_wells, d, omega=omega, dim=dim)
    cfg.setdefault("training", {})["seed"] = int(cfg["training"].get("seed", 42))
    cfg["run_name"] = f"{cfg.get('run_name', 'd_sweep')}_d{d:g}"
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
    # Force unbuffered stdout/stderr in the trainer subprocess so progress lines
    # appear in train_{tag}.log promptly (Python defaults to block buffering when
    # stdout is redirected to a file, which can hide an hours-long run behind a
    # zero-byte log).
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


def _full_pair_correlator_matrix(payload) -> np.ndarray:
    """Return the (N, N) symmetric ``<S_i.S_j>`` matrix from a Mott payload."""
    out = spin_pair_correlator(payload, pairs=None)
    return np.asarray(out["c_matrix"], dtype=np.float64)


def analyse_one(checkpoint: Path, *, set_a: list[int] | None = None, device: str | None = None) -> dict[str, Any]:
    """Reload one checkpoint and return the full diagnostic payload."""
    loaded = load_wavefunction_from_dir(checkpoint, device=device)
    payload = extract_spin_amplitudes(loaded)
    pinn_amps = payload.normalised()

    n_wells = int(payload.n_wells)
    n_down = int(payload.n_down)
    set_a_eff = set_a if set_a is not None else list(range(n_wells // 2))

    # Bipartite spin entanglement.
    bp = well_set_bipartite_entropy(payload, set_a=set_a_eff)

    # Heisenberg reference.
    heis = heisenberg_obc_ground_state(n_wells, n_down, bond_couplings=None)
    aligned, overlap = align_amplitude_signs(heis.ground_amplitudes, pinn_amps)
    residual_l2 = float(np.linalg.norm(aligned - heis.ground_amplitudes))

    # Heisenberg-reference entanglement (use the same payload spec).
    from observables.spin_amplitude_entanglement import SpinAmplitudePayload
    log_abs = np.log(np.maximum(np.abs(heis.ground_amplitudes), 1e-300))
    sign = np.sign(heis.ground_amplitudes)
    sign[sign == 0] = 1.0
    heis_payload = SpinAmplitudePayload(
        pattern=list(payload.pattern),
        log_abs_psi=log_abs,
        sign_psi=sign,
        perm_sign=np.ones_like(heis.ground_amplitudes),
        sigma_z_total=payload.sigma_z_total,
        n_up=payload.n_up,
        n_down=payload.n_down,
        n_wells=payload.n_wells,
        well_centers=payload.well_centers,
    )
    heis_bp = well_set_bipartite_entropy(heis_payload, set_a=set_a_eff)

    # Full <S_i.S_j> matrix.
    C_pinn = _full_pair_correlator_matrix(payload)
    C_heis = _full_pair_correlator_matrix(heis_payload)

    # Off-Mott norm leak: extract_spin_amplitudes returns a normalised vector
    # restricted to the singly-occupied basis. We can quantify how much the
    # raw |psi|^2 mass sits OUTSIDE that subspace by comparing
    # sum exp(2 log_abs) before and after normalisation. The current API
    # doesn't expose this directly, so we leave a hook and report 1 - 1 = 0
    # for now; this can be wired up once the extractor returns the raw
    # masses.
    out = {
        "result_dir": str(checkpoint),
        "n_wells": int(n_wells),
        "n_down": int(n_down),
        "set_a": list(set_a_eff),
        "pinn": {
            "von_neumann_entropy": float(bp["von_neumann_entropy"]),
            "negativity": float(bp.get("negativity", float("nan"))),
            "log_negativity": float(bp.get("log_negativity", float("nan"))),
            "linear_entropy": float(bp["linear_entropy"]),
            "C_matrix": C_pinn.tolist(),
        },
        "heisenberg": {
            "uniform_J": True,
            "ground_eigenvalue": float(heis.eigenvalues[0]),
            "multiplicity": int(heis.multiplicity),
            "overlap": float(overlap),
            "residual_l2": residual_l2,
            "von_neumann_entropy": float(heis_bp["von_neumann_entropy"]),
            "negativity": float(heis_bp.get("negativity", float("nan"))),
            "C_matrix": C_heis.tolist(),
        },
        "pinn_amplitudes": aligned.tolist(),
        "heis_amplitudes": heis.ground_amplitudes.tolist(),
        "patterns": [list(p) for p in payload.pattern],
    }
    return out


def _safe_get(d: dict[str, Any], path: list[str], default: Any = float("nan")) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        LOGGER.warning("No rows to write to %s.", out_path)
        return
    fields = [
        "d", "n_wells", "n_down", "energy",
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

    rows_sorted = sorted(rows, key=lambda r: r["d"])
    d = np.array([r["d"] for r in rows_sorted], dtype=float)
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
    fig.suptitle(f"d-sweep: {run_label}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(d, S_pinn, "o-", color="#1f77b4", label="S(PINN)")
    ax.plot(d, S_heis, "s--", color="#7f7f7f", label="S(Heisenberg, uniform J)")
    ax.set_xlabel("inter-well distance d (Bohr)")
    ax.set_ylabel("bipartite von Neumann entropy")
    ax.set_title("(a) Spin entanglement vs d")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(d, overlap, "o-", color="#2ca02c")
    ax.set_xlabel("d (Bohr)")
    ax.set_ylabel(r"$|\langle c_{\rm PINN}|c_{\rm Heis}\rangle|$")
    ax.set_title("(b) Heisenberg overlap")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.semilogy(d, np.maximum(res, 1e-12), "x:", color="#9467bd", label="L2 residual")
    ax2.set_ylabel("L2 residual (log)", color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")

    ax = axes[1, 0]
    ax.plot(d, C_end_pinn, "o-", color="#d62728", label="<S_0.S_{N-1}> PINN")
    ax.plot(d, C_end_heis, "s--", color="#7f7f7f", label="<S_0.S_{N-1}> Heisenberg")
    ax.plot(d, C_nn_min_pinn, "^-", color="#ff7f0e", label="min NN PINN")
    ax.plot(d, C_nn_min_heis, "v--", color="#bcbd22", label="min NN Heisenberg")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("d (Bohr)")
    ax.set_ylabel(r"$\langle S_i\cdot S_j\rangle$")
    ax.set_title("(c) Spin correlators")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(d, energy, "o-", color="#17becf")
    ax.set_xlabel("d (Bohr)")
    ax.set_ylabel("E (Ha)")
    ax.set_title("(d) Total energy")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Base YAML config (architecture + training).")
    parser.add_argument("--n-wells", type=int, required=True)
    parser.add_argument("--d-values", type=float, nargs="+", required=True, metavar="D")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None,
                        help="Override CUDA device for training (e.g. cuda:3). "
                             "Default: read from CUDA_MANUAL_DEVICE env or fall back to cuda:0.")
    parser.add_argument("--stage-a-epochs", type=int, default=2500)
    parser.add_argument("--stage-b-epochs", type=int, default=1)
    parser.add_argument("--stage-a-strategy", type=str, default="improved_self_residual",
                        choices=["auto", "guided", "self_residual", "singlet_self_residual", "improved_self_residual"])
    parser.add_argument("--stage-a-min-energy", type=float, default=999.0)
    parser.add_argument("--seed-override", type=int, default=None)
    parser.add_argument("--set-a", type=int, nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip training a given d if its summary JSON already exists.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = yaml.safe_load(args.config.read_text())
    dim = int(base_cfg.get("system", {}).get("dim", 2))

    device = args.device
    if device is None:
        env_dev = os.environ.get("CUDA_MANUAL_DEVICE")
        device = f"cuda:{env_dev}" if env_dev is not None else "cuda:0"

    rows: list[dict[str, Any]] = []
    full_payloads: list[dict[str, Any]] = []
    for d in args.d_values:
        tag = f"d{d:g}"
        cfg_path = args.out_dir / f"cfg_{tag}.yaml"
        summary_path = args.out_dir / f"summary_{tag}.json"
        _write_per_d_config(base_cfg, args.n_wells, d, dim, cfg_path)

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

        # Pull the final energy from the summary.
        summary_payload = json.loads(summary_path.read_text())
        if summary_payload.get("stage_b") is not None:
            energy = float(summary_payload["stage_b"]["result"]["final_energy"])
        else:
            energy = float(summary_payload["stage_a"]["result"]["final_energy"])

        # Analyse.
        diag = analyse_one(ckpt, set_a=args.set_a, device=device)
        diag["d"] = float(d)
        diag["energy"] = energy
        full_payloads.append(diag)

        C_pinn = np.array(diag["pinn"]["C_matrix"])
        C_heis = np.array(diag["heisenberg"]["C_matrix"])
        nn_pinn = np.array([C_pinn[k, k + 1] for k in range(args.n_wells - 1)])
        nn_heis = np.array([C_heis[k, k + 1] for k in range(args.n_wells - 1)])
        rows.append({
            "d": float(d),
            "n_wells": int(diag["n_wells"]),
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

    # Print summary.
    print()
    print("=" * 92)
    print(f"  d-sweep summary  ({args.n_wells}-well chain, {len(rows)} d values, device={device})")
    print("=" * 92)
    print(
        f"  {'d':>5} {'E':>10} {'S_pinn':>9} {'S_heis':>9} "
        f"{'overlap':>9} {'res_L2':>10} {'C_end(pinn)':>12} {'C_end(heis)':>12}"
    )
    for r in sorted(rows, key=lambda x: x["d"]):
        print(
            f"  {r['d']:>5.2f} {r['energy']:>10.4f} {r['S_pinn']:>9.4f} {r['S_heis']:>9.4f} "
            f"{r['overlap']:>9.5f} {r['residual_l2']:>10.3e} {r['C_end_to_end_pinn']:>12.5f} {r['C_end_to_end_heis']:>12.5f}"
        )
    print()

    write_csv(rows, args.out_dir / "d_sweep.csv")
    (args.out_dir / "d_sweep.json").write_text(json.dumps({"rows": rows, "full": full_payloads}, indent=2))
    render_png(rows, args.out_dir / "d_sweep.png", run_label=args.out_dir.name)
    LOGGER.info("Wrote: %s, %s, %s",
                args.out_dir / "d_sweep.csv", args.out_dir / "d_sweep.png", args.out_dir / "d_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
