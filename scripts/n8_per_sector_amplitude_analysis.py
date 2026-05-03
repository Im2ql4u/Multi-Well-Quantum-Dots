#!/usr/bin/env python3
"""Per-sector spin-amplitude / entanglement / Heisenberg-overlap analysis.

For each Sz sector trained by ``n8_sector_b_sweep.py``, reload its checkpoint
and run ``analyse_one`` to obtain:

  * Mott spin amplitudes.
  * Bipartite (set_a vs complement) von-Neumann entropy and negativity.
  * Heisenberg-OBC reference overlap and residual L2 (only meaningful for
    the lower-Sz sectors; the fully polarised 8↑0↓ sector has trivial
    Heisenberg overlap).
  * Full N×N ⟨S_i·S_j⟩ matrix.

This was the missing piece in the original ``n8_sector_b_sweep`` run: the
``--analyse-sector`` flag was matched against the wrong tag format
(``4_4`` vs ``4up4down``), so ``analyse_one`` was never called. This script
backfills that analysis post-hoc by reading the per-sector ``summary_*.json``
files in the sweep output directory.

Usage
-----
::

    PYTHONPATH=src python3.11 scripts/n8_per_sector_amplitude_analysis.py \\
        --sweep-dir results/b_sweep/n8_chain_d4_sector_aware_s42 \\
        --out-json results/b_sweep/n8_chain_d4_sector_aware_s42/per_sector_observables.json \\
        --out-png  results/b_sweep/n8_chain_d4_sector_aware_s42/per_sector_observables.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from n_chain_d_sweep import analyse_one  # noqa: E402

LOGGER = logging.getLogger("n8_per_sector_amplitude")


def _per_sector_summary_files(sweep_dir: Path) -> list[Path]:
    files = sorted(sweep_dir.glob("summary_*up*down.json"))
    if not files:
        raise FileNotFoundError(
            f"No 'summary_*up*down.json' files found in {sweep_dir}"
        )
    return files


def _resolve_ckpt_from_summary(summary_json: Path) -> Path:
    payload = json.loads(summary_json.read_text())
    stage_b = payload.get("stage_b")
    if stage_b is not None:
        return Path(stage_b["result_dir"])
    return Path(payload["stage_a"]["result_dir"])


def _parse_tag(summary_path: Path) -> tuple[int, int, str]:
    """summary_4up4down.json -> (4, 4, '4up4down')."""
    name = summary_path.stem.replace("summary_", "")
    nu_str, nd_str = name.split("up")
    nd_str = nd_str.replace("down", "")
    return int(nu_str), int(nd_str), name


def _flatten_observable(name: str, sec: dict) -> float:
    pinn = sec.get("pinn", {})
    heis = sec.get("heisenberg", {})
    if name == "S_pinn":
        return float(pinn.get("von_neumann_entropy", float("nan")))
    if name == "S_heis":
        return float(heis.get("von_neumann_entropy", float("nan")))
    if name == "overlap":
        return float(heis.get("overlap", float("nan")))
    if name == "residual_l2":
        return float(heis.get("residual_l2", float("nan")))
    if name == "C_NN_pinn":
        C = np.asarray(pinn.get("C_matrix", [[]]))
        if C.size == 0 or C.ndim != 2:
            return float("nan")
        return float(np.mean([C[i, i + 1] for i in range(C.shape[0] - 1)]))
    if name == "C_end_to_end_pinn":
        C = np.asarray(pinn.get("C_matrix", [[]]))
        if C.size == 0 or C.ndim != 2:
            return float("nan")
        return float(C[0, -1])
    return float("nan")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Output dir of n8_sector_b_sweep.py (contains summary_*up*down.json).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Combined per-sector observables JSON output. "
        "Defaults to <sweep-dir>/per_sector_observables.json.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="6-panel PNG summary. Defaults to <sweep-dir>/per_sector_observables.png.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load checkpoints onto (default: cpu — analysis is small).",
    )
    parser.add_argument(
        "--set-a",
        type=int,
        nargs="+",
        default=None,
        help="Subset of well indices for bipartite entropy (defaults to first N//2).",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    out_json = args.out_json or (sweep_dir / "per_sector_observables.json")
    out_png = args.out_png or (sweep_dir / "per_sector_observables.png")

    summary_files = _per_sector_summary_files(sweep_dir)
    LOGGER.info("Found %d sector summaries", len(summary_files))

    per_sector: list[dict] = []
    for sf in summary_files:
        n_up, n_down, tag = _parse_tag(sf)
        ckpt = _resolve_ckpt_from_summary(sf)
        if not ckpt.exists():
            LOGGER.warning("[%s] checkpoint dir does not exist: %s", tag, ckpt)
            continue
        Sz = (n_up - n_down) / 2.0
        try:
            obs = analyse_one(ckpt, set_a=args.set_a, device=args.device)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[%s] analyse_one failed: %s", tag, exc)
            obs = {"error": str(exc)}
        record = {
            "tag": tag,
            "n_up": n_up,
            "n_down": n_down,
            "Sz": Sz,
            "checkpoint": str(ckpt),
            "observables": obs,
        }
        per_sector.append(record)
        if "error" not in obs:
            LOGGER.info(
                "[%s] Sz=%+.1f S_pinn=%.4f overlap=%.4f residual_l2=%.4f C_NN=%.4f",
                tag,
                Sz,
                _flatten_observable("S_pinn", obs),
                _flatten_observable("overlap", obs),
                _flatten_observable("residual_l2", obs),
                _flatten_observable("C_NN_pinn", obs),
            )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"sectors": per_sector}, indent=2))
    LOGGER.info("Wrote %s", out_json)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib unavailable, skipping PNG: %s", exc)
        return 0

    rows = [r for r in per_sector if "error" not in r["observables"]]
    if not rows:
        LOGGER.warning("No usable sector rows for plotting")
        return 0
    rows.sort(key=lambda r: r["Sz"])
    Sz = np.array([r["Sz"] for r in rows])
    S_pinn = np.array([_flatten_observable("S_pinn", r["observables"]) for r in rows])
    S_heis = np.array([_flatten_observable("S_heis", r["observables"]) for r in rows])
    overlap = np.array([_flatten_observable("overlap", r["observables"]) for r in rows])
    res_l2 = np.array([_flatten_observable("residual_l2", r["observables"]) for r in rows])
    C_NN = np.array([_flatten_observable("C_NN_pinn", r["observables"]) for r in rows])
    C_end = np.array([_flatten_observable("C_end_to_end_pinn", r["observables"]) for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes = axes.flatten()

    axes[0].plot(Sz, S_pinn, "o-", label="PINN")
    axes[0].plot(Sz, S_heis, "s--", label="Heisenberg ref")
    axes[0].set_xlabel("Sz")
    axes[0].set_ylabel("Bipartite entropy")
    axes[0].set_title("(a) Set-A bipartite vN entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(Sz, overlap, "o-")
    axes[1].set_xlabel("Sz")
    axes[1].set_ylabel("|<PINN|Heis>|")
    axes[1].set_title("(b) Heisenberg-overlap")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].plot(Sz, res_l2, "o-")
    axes[2].set_xlabel("Sz")
    axes[2].set_ylabel("|PINN - Heis|_2")
    axes[2].set_title("(c) Amplitude residual L2")
    axes[2].grid(alpha=0.3)

    axes[3].plot(Sz, C_NN, "o-", label="<S0.S1> avg NN")
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_xlabel("Sz")
    axes[3].set_ylabel("<S_i.S_{i+1}> avg")
    axes[3].set_title("(d) NN spin correlator")
    axes[3].grid(alpha=0.3)

    axes[4].plot(Sz, C_end, "o-")
    axes[4].axhline(0, color="k", lw=0.5)
    axes[4].set_xlabel("Sz")
    axes[4].set_ylabel("<S_0.S_{N-1}>")
    axes[4].set_title("(e) End-to-end spin correlator")
    axes[4].grid(alpha=0.3)

    pinn_for_heatmap = next(
        (r for r in rows if r["Sz"] == 0.0),
        rows[0],
    )
    C_pinn = np.asarray(pinn_for_heatmap["observables"]["pinn"]["C_matrix"])
    im = axes[5].imshow(C_pinn, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[5].set_title(f"(f) <S_i.S_j> heatmap, Sz={pinn_for_heatmap['Sz']:+.1f}")
    axes[5].set_xlabel("j")
    axes[5].set_ylabel("i")
    fig.colorbar(im, ax=axes[5], shrink=0.7)

    fig.suptitle(
        f"Per-sector spin observables — {sweep_dir.name}",
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    LOGGER.info("Wrote %s", out_png)

    return 0


if __name__ == "__main__":
    sys.exit(main())
