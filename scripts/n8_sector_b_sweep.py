#!/usr/bin/env python3
"""Sector-aware magnetic phase-diagram sweep for an N-well chain.

Background
----------
The earlier ``n_chain_b_sweep.py`` exposed a known limitation of the
current Hamiltonian + fixed-Sz ansatz combination: under uniform
longitudinal Zeeman coupling the field only adds a *constant* shift per
fixed Sz sector, so a single-sector B-sweep produces no state-changing
response. The 2026-04-28 5-point sweep returned bit-identical spin
observables across ``B ∈ {0, 0.05, 0.2, 0.5, 1.0}`` (see
``results/b_sweep/n8_uniform_d4_s42/``).

The right way to get a non-trivial magnetic phase diagram with this
Hamiltonian is the **sector-aware** sweep used for the existing N=3, 4
magnetic configs (``configs/magnetic/n*_*up*down_*.yaml``):

  1.  Train one PINN per fixed Sz sector at a *single* reference field
      (we use ``B=0``).  In each sector the wavefunction is
      B-independent — only the orbital (kinetic + Coulomb + soft-min
      confinement) energy ``E_orbital(sector)`` is computed by the
      trainer.  This is what makes the sweep cheap: ``n_sectors``
      trainings instead of ``n_sectors × n_B``.

  2.  For each magnetic field ``B`` of interest, assemble the total
      energy analytically:

          E(B, sector) = E_orbital(sector) + 0.5·g·μ_B·B·(n_up - n_down)

      In atomic units with ``g=2, μ_B=1`` this simplifies to
      ``E(B, sector) = E_orbital(sector) + B·(n_up - n_down)``.

  3.  The ground state at field ``B`` is the sector that minimises
      ``E(B, sector)``.  The level crossings give the critical fields
      ``B_c(k)`` at which the GS Sz changes by one unit — that is the
      magnetic phase diagram of the chain.

This script implements that workflow and emits a CSV + JSON + PNG
phase diagram.  It also reuses the ``analyse_one`` helper from the
old per-B sweep to capture the *spin observables* in each sector at
B=0 (same spin-correlator and Heisenberg-overlap metrics as before),
so the deliverables are directly comparable to the d-sweep / B-sweep
JSONs.

Usage
-----
::

    PYTHONPATH=src CUDA_MANUAL_DEVICE=3 \\
        python3.11 scripts/n8_sector_b_sweep.py \\
            --config configs/magnetic/n8_chain_d4_4up4down_b0_s42.yaml \\
            --sectors 4_4 5_3 6_2 7_1 8_0 \\
            --b-values 0.0 0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 \\
            --stage-a-epochs 1500 \\
            --stage-a-strategy improved_self_residual \\
            --out-dir results/b_sweep/n8_chain_d4_sector_aware_s42

By Sz → −Sz symmetry of the spin-isotropic Hamiltonian, only positive
Sz sectors need to be trained; the script mirrors the energies for the
negative-Sz sectors automatically.
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

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from n_chain_d_sweep import analyse_one  # noqa: E402

LOGGER = logging.getLogger("n8_sector_b_sweep")


def _sector_tag(n_up: int, n_down: int) -> str:
    return f"{n_up}up{n_down}down"


def _write_per_sector_config(
    base_cfg: dict[str, Any],
    n_up: int,
    n_down: int,
    out_path: Path,
) -> None:
    cfg = copy.deepcopy(base_cfg)
    spin = cfg.setdefault("spin", {})
    spin["n_up"] = int(n_up)
    spin["n_down"] = int(n_down)
    system = cfg.setdefault("system", {})
    system["B_magnitude"] = 0.0
    base_run = cfg.get("run_name", "sector_sweep")
    cfg["run_name"] = f"{base_run}_{_sector_tag(n_up, n_down)}"
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


def _zeeman_shift(
    n_up: int,
    n_down: int,
    B: float,
    *,
    g_factor: float = 2.0,
    mu_B: float = 1.0,
) -> float:
    """V_zeeman = 0.5·g·μ_B·B·Σ s_iz with s_iz = ±1, so net = B·(n_up - n_down)
    when g=2, μ_B=1.
    """
    return 0.5 * float(g_factor) * float(mu_B) * float(B) * float(n_up - n_down)


def _parse_sector(spec: str) -> tuple[int, int]:
    if "_" not in spec:
        raise argparse.ArgumentTypeError(
            f"Sector spec {spec!r} must look like 'n_up_n_down', e.g. '4_4'."
        )
    parts = spec.split("_")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Sector spec {spec!r} must have exactly two underscore-separated ints."
        )
    return int(parts[0]), int(parts[1])


def write_phase_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        LOGGER.warning("No rows to write to %s.", out_path)
        return
    fields = list(rows[0].keys())
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render_phase_png(
    sectors: list[tuple[int, int]],
    E_orbital: dict[tuple[int, int], float],
    b_values: list[float],
    out_path: Path,
    *,
    g_factor: float,
    mu_B: float,
    run_label: str,
    sector_obs: dict[tuple[int, int], dict[str, float]] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sectors_sorted = sorted(sectors, key=lambda kv: -(kv[0] - kv[1]))  # high Sz first
    cmap = plt.get_cmap("viridis")
    colours = [cmap(i / max(len(sectors_sorted) - 1, 1)) for i in range(len(sectors_sorted))]

    B_arr = np.array(sorted(b_values), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Sector-aware magnetic phase diagram — {run_label}", fontsize=13)

    ax = axes[0, 0]
    E_grid = np.zeros((len(sectors_sorted), len(B_arr)))
    for i, (n_up, n_down) in enumerate(sectors_sorted):
        E0 = E_orbital[(n_up, n_down)]
        Sz = (n_up - n_down) / 2.0
        Es = E0 + np.array(
            [_zeeman_shift(n_up, n_down, b, g_factor=g_factor, mu_B=mu_B) for b in B_arr]
        )
        E_grid[i] = Es
        ax.plot(B_arr, Es, "-", color=colours[i], label=f"Sz={Sz:+.1f}")
    ax.set_xlabel("B (a.u.)")
    ax.set_ylabel("E(B, sector) [Ha]")
    ax.set_title("(a) Total energy per sector vs B")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    gs_idx = np.argmin(E_grid, axis=0)
    gs_E = np.min(E_grid, axis=0)
    gs_Sz = np.array([(sectors_sorted[i][0] - sectors_sorted[i][1]) / 2.0 for i in gs_idx])
    ax.plot(B_arr, gs_Sz, "o-", color="#d62728")
    ax.set_xlabel("B (a.u.)")
    ax.set_ylabel(r"GS $S_z$")
    ax.set_title("(b) Ground-state Sz vs B")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    Sz_vals = np.array([(n_up - n_down) / 2.0 for n_up, n_down in sectors_sorted])
    E_base = np.array([E_orbital[s] for s in sectors_sorted])
    E_base_centered = E_base - np.min(E_base)
    ax.plot(Sz_vals, E_base_centered, "o-", color="#1f77b4")
    ax.set_xlabel(r"$S_z$")
    ax.set_ylabel(r"$E_{\rm orbital} - \min_{\rm sectors}\;E_{\rm orbital}$ [Ha]")
    ax.set_title(r"(c) Orbital energy spread (AFM exchange scale)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(B_arr, gs_E, "-", color="#2ca02c")
    ax.set_xlabel("B (a.u.)")
    ax.set_ylabel(r"GS energy $E(B)$ [Ha]")
    ax.set_title("(d) Ground-state energy envelope")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Base YAML config; spin block is overridden per sector.")
    parser.add_argument("--sectors", type=str, nargs="+", required=True,
                        metavar="n_up_n_down",
                        help="List of spin sectors, e.g. '4_4 5_3 6_2 7_1 8_0'.")
    parser.add_argument("--b-values", type=float, nargs="+", required=True,
                        help="Magnetic field values (a.u.) for the post-hoc phase diagram.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stage-a-epochs", type=int, default=1500)
    parser.add_argument("--stage-b-epochs", type=int, default=1)
    parser.add_argument("--stage-a-strategy", type=str, default="improved_self_residual",
                        choices=["auto", "guided", "self_residual",
                                 "singlet_self_residual", "improved_self_residual"])
    parser.add_argument("--stage-a-min-energy", type=float, default=999.0)
    parser.add_argument("--seed-override", type=int, default=None)
    parser.add_argument("--mirror-sz", action="store_true", default=True,
                        help="Add the Sz→-Sz mirror sectors to the phase diagram via "
                             "spin-isotropic symmetry (default: on).")
    parser.add_argument("--no-mirror-sz", action="store_false", dest="mirror_sz")
    parser.add_argument("--analyse-sector", type=str, default="4_4",
                        help="Optional sector tag (e.g. '4_4') to run the d-sweep "
                             "spin-observable analysis on; default: 4_4 (Sz=0). "
                             "Pass '' to skip.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip retraining sectors that already have a summary JSON.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = yaml.safe_load(args.config.read_text())
    g_factor = float(base_cfg.get("system", {}).get("g_factor", 2.0))
    mu_B = float(base_cfg.get("system", {}).get("mu_B", 1.0))

    device = args.device
    if device is None:
        env_dev = os.environ.get("CUDA_MANUAL_DEVICE")
        device = f"cuda:{env_dev}" if env_dev is not None else "cuda:0"

    requested_sectors: list[tuple[int, int]] = [_parse_sector(s) for s in args.sectors]

    E_orbital: dict[tuple[int, int], float] = {}
    sector_summary: list[dict[str, Any]] = []
    sector_obs: dict[tuple[int, int], dict[str, Any]] = {}

    for n_up, n_down in requested_sectors:
        tag = _sector_tag(n_up, n_down)
        cfg_path = args.out_dir / f"cfg_{tag}.yaml"
        summary_path = args.out_dir / f"summary_{tag}.json"
        _write_per_sector_config(base_cfg, n_up, n_down, cfg_path)

        if args.skip_existing and summary_path.exists():
            LOGGER.info("[%s] reusing existing summary", tag)
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

        payload = json.loads(summary_path.read_text())
        if payload.get("stage_b") is not None:
            E0 = float(payload["stage_b"]["result"]["final_energy"])
        else:
            E0 = float(payload["stage_a"]["result"]["final_energy"])
        E_orbital[(n_up, n_down)] = E0

        Sz = (n_up - n_down) / 2.0
        sector_summary.append({
            "n_up": n_up,
            "n_down": n_down,
            "Sz": Sz,
            "E_orbital_Ha": E0,
            "result_dir": str(ckpt),
        })
        LOGGER.info("[%s] E_orbital = %.6f Ha (Sz=%+.1f)", tag, E0, Sz)

        if args.analyse_sector and tag == args.analyse_sector:
            try:
                obs = analyse_one(ckpt, set_a=None, device=device)
                sector_obs[(n_up, n_down)] = obs
                LOGGER.info("[%s] analyse_one OK (S_pinn=%.4f, overlap=%.4f)",
                            tag, obs["pinn"]["von_neumann_entropy"],
                            obs["heisenberg"]["overlap"])
            except Exception as exc:
                LOGGER.warning("[%s] analyse_one failed: %s", tag, exc)

    if args.mirror_sz:
        for (n_up, n_down), E0 in list(E_orbital.items()):
            mirror = (n_down, n_up)
            if n_up != n_down and mirror not in E_orbital:
                E_orbital[mirror] = E0
                sector_summary.append({
                    "n_up": mirror[0],
                    "n_down": mirror[1],
                    "Sz": (mirror[0] - mirror[1]) / 2.0,
                    "E_orbital_Ha": E0,
                    "result_dir": "(mirror of (%d,%d) by Sz->-Sz symmetry)" % (n_up, n_down),
                })

    sectors_full = list(E_orbital.keys())

    rows: list[dict[str, Any]] = []
    for B in sorted(args.b_values):
        per_sector_E = {
            (n_up, n_down): E0 + _zeeman_shift(n_up, n_down, B,
                                                g_factor=g_factor, mu_B=mu_B)
            for (n_up, n_down), E0 in E_orbital.items()
        }
        gs_sector, gs_E = min(per_sector_E.items(), key=lambda kv: kv[1])
        row = {
            "B": float(B),
            "gs_n_up": gs_sector[0],
            "gs_n_down": gs_sector[1],
            "gs_Sz": (gs_sector[0] - gs_sector[1]) / 2.0,
            "gs_energy_Ha": gs_E,
        }
        for (n_up, n_down), E in per_sector_E.items():
            row[f"E_{n_up}up{n_down}down"] = E
        rows.append(row)

    print()
    print("=" * 92)
    print(f"  Sector-aware B-sweep summary  ({len(sector_summary)} sectors, "
          f"{len(rows)} B values, device={device})")
    print("=" * 92)
    print(f"  {'B':>7}  {'GS sector':>14}  {'GS Sz':>7}  {'GS E (Ha)':>11}")
    print("  " + "-" * 50)
    for r in sorted(rows, key=lambda x: x["B"]):
        gs_label = f"{r['gs_n_up']}↑{r['gs_n_down']}↓"
        print(f"  {r['B']:>7.4f}  {gs_label:>14}  {r['gs_Sz']:>+7.1f}  {r['gs_energy_Ha']:>11.6f}")
    print()
    Sz_seq = [r["gs_Sz"] for r in sorted(rows, key=lambda x: x["B"])]
    crossings = []
    sorted_rows = sorted(rows, key=lambda x: x["B"])
    for i in range(1, len(sorted_rows)):
        if Sz_seq[i] != Sz_seq[i - 1]:
            crossings.append((sorted_rows[i - 1]["B"], sorted_rows[i]["B"],
                              Sz_seq[i - 1], Sz_seq[i]))
    if crossings:
        print("  GS-sector level crossings (B_lo, B_hi, Sz_before -> Sz_after):")
        for c in crossings:
            print(f"    {c[0]:.4f} → {c[1]:.4f}  :  Sz {c[2]:+.1f} → {c[3]:+.1f}")
    else:
        print("  No GS-sector level crossings within the requested B range.")
    print()

    write_phase_csv(rows, args.out_dir / "B_sweep.csv")

    sector_obs_serializable: dict[str, Any] = {}
    for (n_up, n_down), obs in sector_obs.items():
        sector_obs_serializable[_sector_tag(n_up, n_down)] = obs
    payload = {
        "sectors": sector_summary,
        "phase_diagram": rows,
        "g_factor": g_factor,
        "mu_B": mu_B,
        "sector_observables": sector_obs_serializable,
        "config": str(args.config),
        "level_crossings": [
            {"B_lo": c[0], "B_hi": c[1], "Sz_before": c[2], "Sz_after": c[3]}
            for c in crossings
        ],
    }
    (args.out_dir / "B_sweep.json").write_text(json.dumps(payload, indent=2))

    render_phase_png(
        sectors=sectors_full,
        E_orbital=E_orbital,
        b_values=list(args.b_values),
        out_path=args.out_dir / "B_sweep.png",
        g_factor=g_factor,
        mu_B=mu_B,
        run_label=args.out_dir.name,
        sector_obs=sector_obs,
    )

    LOGGER.info(
        "Wrote: %s, %s, %s",
        args.out_dir / "B_sweep.csv",
        args.out_dir / "B_sweep.json",
        args.out_dir / "B_sweep.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
