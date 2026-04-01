#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class CaseData:
    combo: str
    run_dir: Path
    distance: float
    coulomb: bool
    loss: str
    tau_sampling: str
    x_sampling: str
    tau: np.ndarray
    energy: np.ndarray
    energy_err: np.ndarray
    e_gt: float
    gt_label: str


def latest_strategy_dir(root: Path) -> Path:
    runs_dir = root / "results" / "imag_time_runs"
    dirs = sorted(runs_dir.glob("strategy_search_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError("No strategy_search_* directory found")
    return dirs[0]


def _gt_key(distance: float, coulomb: bool) -> Tuple[float, bool]:
    return (round(float(distance), 8), bool(coulomb))


def load_ed_ground_truth(path: Path) -> Dict[Tuple[float, bool], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[Tuple[float, bool], dict] = {}
    if "results" in data:
        for row in data.get("results", []):
            k = _gt_key(row.get("distance", 0.0), row.get("coulomb", False))
            out[k] = row
    elif "summary" in data:
        for row in data.get("summary", []):
            k = _gt_key(row.get("distance", 0.0), row.get("coulomb", False))
            out[k] = row
    return out


def ground_truth_energy(
    coulomb: bool,
    distance: float,
    ed_gt: Dict[Tuple[float, bool], dict] | None = None,
) -> tuple[float, str]:
    if ed_gt is not None:
        rec = ed_gt.get(_gt_key(distance, coulomb))
        if rec is not None:
            # Converged file may contain converged=true and uncertainty E0_unc.
            if rec.get("E0") is not None:
                if rec.get("converged") is False:
                    return float(rec["E0"]), "FD diagonalization (not fully converged)"
                unc = rec.get("E0_unc")
                if unc is not None:
                    return float(rec["E0"]), f"FD diagonalization E0 (unc ±{float(unc):.3e})"
                return float(rec["E0"]), "Finite-difference diagonalization E0"

    if not coulomb:
        return 2.0, "Exact non-interacting E0 = 2"
    if distance <= 1e-12:
        return 3.0, "Approx interacting reference E0 = 3 (d=0)"
    return 2.0 + 1.0 / distance, "Separated-dot asymptotic E0 ≈ 2 + 1/d"


def read_case_rows(score_file: Path) -> list[dict]:
    with score_file.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_case_jsons(row: dict, ed_gt: Dict[Tuple[float, bool], dict] | None = None) -> list[CaseData]:
    run_dir = Path(row["run_dir"])
    combo = row["combo"]
    loss = row["loss"]
    tau_sampling = row["tau_sampling"]
    x_sampling = row["x_sampling"]
    coulomb = row["coulomb"].strip().lower() == "true"

    cases: list[CaseData] = []
    for jf in sorted(run_dir.glob("pinn_d*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        result = data["result"]
        traj = result.get("trajectory", [])
        if not traj:
            continue
        tau = np.array([float(t["tau"]) for t in traj], dtype=float)
        energy = np.array([float(t["E"]) for t in traj], dtype=float)
        energy_err = np.array([float(t.get("E_err", 0.0)) for t in traj], dtype=float)
        d = float(result["d"])
        e_gt, gt_label = ground_truth_energy(coulomb=coulomb, distance=d, ed_gt=ed_gt)

        cases.append(
            CaseData(
                combo=combo,
                run_dir=run_dir,
                distance=d,
                coulomb=coulomb,
                loss=loss,
                tau_sampling=tau_sampling,
                x_sampling=x_sampling,
                tau=tau,
                energy=energy,
                energy_err=energy_err,
                e_gt=e_gt,
                gt_label=gt_label,
            )
        )
    return cases


def case_slug(case: CaseData) -> str:
    c = "on" if case.coulomb else "off"
    return f"{case.combo}_d{case.distance:.1f}_{c}".replace(".", "p")


def fit_decay_rate(tau: np.ndarray, abs_err: np.ndarray) -> float | None:
    mask = abs_err > 1e-5
    if mask.sum() < 5:
        return None
    x = tau[mask]
    y = np.log(abs_err[mask])
    a, b = np.polyfit(x, y, 1)
    if a >= 0:
        return None
    return float(-a)


def render_case_artifacts(case: CaseData, out_dir: Path) -> dict:
    slug = case_slug(case)
    abs_err = np.abs(case.energy - case.e_gt)

    # Main trajectory plot.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(case.tau, case.energy, color="#1f77b4", lw=2.2, label="PINN E(tau)")
    ax.fill_between(
        case.tau,
        case.energy - case.energy_err,
        case.energy + case.energy_err,
        color="#1f77b4",
        alpha=0.2,
        label="MC error band",
    )
    ax.axhline(case.e_gt, color="#d62728", ls="--", lw=2.0, label=f"Ground truth ({case.e_gt:.6f})")
    ax.set_title(f"Energy Evolution: combo={case.combo}, d={case.distance:.1f}, coulomb={'on' if case.coulomb else 'off'}")
    ax.set_xlabel("tau")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    energy_plot = out_dir / f"{slug}_energy.png"
    fig.savefig(energy_plot, dpi=140)
    plt.close(fig)

    # Error plot.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(case.tau, abs_err, color="#ff7f0e", lw=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("tau")
    ax.set_ylabel("|E(tau) - E_gt|")
    ax.set_title(f"Absolute Error vs Ground Truth: combo={case.combo}, d={case.distance:.1f}")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    err_plot = out_dir / f"{slug}_error.png"
    fig.savefig(err_plot, dpi=140)
    plt.close(fig)

    # Animation: trajectory revelation over tau.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(float(case.tau.min()), float(case.tau.max()))
    y_lo = min(float(case.energy.min()), case.e_gt) - 0.05
    y_hi = max(float(case.energy.max()), case.e_gt) + 0.05
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("tau")
    ax.set_ylabel("Energy")
    ax.set_title(f"Animated Evolution: combo={case.combo}, d={case.distance:.1f}")
    ax.grid(alpha=0.25)
    ax.axhline(case.e_gt, color="#d62728", ls="--", lw=2.0, label="Ground truth")
    line, = ax.plot([], [], color="#1f77b4", lw=2.5, label="PINN E(tau)")
    point, = ax.plot([], [], "o", color="#1f77b4")
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", ha="left", fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"))
    ax.legend(loc="best")

    n = len(case.tau)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        text.set_text("")
        return line, point, text

    def update(i: int):
        i = min(i, n - 1)
        line.set_data(case.tau[: i + 1], case.energy[: i + 1])
        point.set_data([case.tau[i]], [case.energy[i]])
        text.set_text(
            f"tau={case.tau[i]:.3f}\\nE={case.energy[i]:.6f}\\n|E-Egt|={abs_err[i]:.3e}"
        )
        return line, point, text

    ani = FuncAnimation(fig, update, frames=n, init_func=init, interval=130, blit=True)
    anim_path = out_dir / f"{slug}_evolution.gif"
    ani.save(anim_path, writer=PillowWriter(fps=7))
    plt.close(fig)

    decay = fit_decay_rate(case.tau, abs_err)
    final_err = float(abs_err[-1])
    rmse = float(np.sqrt(np.mean((case.energy - case.e_gt) ** 2)))

    return {
        "slug": slug,
        "distance": case.distance,
        "coulomb": case.coulomb,
        "combo": case.combo,
        "loss": case.loss,
        "tau_sampling": case.tau_sampling,
        "x_sampling": case.x_sampling,
        "gt_label": case.gt_label,
        "e_gt": case.e_gt,
        "e_final": float(case.energy[-1]),
        "final_abs_error": final_err,
        "rmse": rmse,
        "decay_rate": decay,
        "energy_plot": energy_plot.name,
        "error_plot": err_plot.name,
        "animation": anim_path.name,
    }


def render_summary_plot(rows: list[dict], out_dir: Path) -> str:
    labels = [f"{r['combo']} d={r['distance']:.1f} {'on' if r['coulomb'] else 'off'}" for r in rows]
    vals = [r["final_abs_error"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(np.arange(len(rows)), vals, color="#2ca02c")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Final |E(T)-E_gt|")
    ax.set_title("Final Ground-Truth Error by Case")
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2e}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = out_dir / "summary_final_error.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def write_html(
    search_dir: Path,
    rows: list[dict],
    summary_plot: str,
    out_html: Path,
    gt_mode: str,
    gt_source: str,
) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def fmt_decay(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "n/a"

    table_rows = "\n".join(
        "<tr>"
        f"<td>{r['combo']}</td>"
        f"<td>{r['distance']:.1f}</td>"
        f"<td>{'on' if r['coulomb'] else 'off'}</td>"
        f"<td>{r['loss']}</td>"
        f"<td>{r['tau_sampling']}</td>"
        f"<td>{r['x_sampling']}</td>"
        f"<td>{r['e_gt']:.6f}</td>"
        f"<td>{r['e_final']:.6f}</td>"
        f"<td>{r['final_abs_error']:.3e}</td>"
        f"<td>{r['rmse']:.3e}</td>"
        f"<td>{fmt_decay(r['decay_rate'])}</td>"
        "</tr>"
        for r in rows
    )

    blocks = []
    for r in rows:
        # Check for density GIF artifacts
        density_section = ""
        density_dir = out_html.parent if out_html else Path(".")
        onebody_gif = r.get("onebody_gif", "")
        pair_gif = r.get("pair_gif", "")
        comparison_png = r.get("comparison_png", "")

        if onebody_gif or pair_gif or comparison_png:
            parts = []
            if comparison_png:
                parts.append(f"""
    <figure>
      <img src=\"{comparison_png}\" alt=\"Density comparison {r['slug']}\" />
      <figcaption>One-body density: perturbed state (tau=0) vs ground state (tau=tau_max).</figcaption>
    </figure>""")
            gif_parts = []
            if onebody_gif:
                gif_parts.append(f"""
      <figure>
        <img src=\"{onebody_gif}\" alt=\"One-body density {r['slug']}\" />
        <figcaption>One-body density rho_1(x,y) evolving over imaginary time.</figcaption>
      </figure>""")
            if pair_gif:
                gif_parts.append(f"""
      <figure>
        <img src=\"{pair_gif}\" alt=\"Pair correlation {r['slug']}\" />
        <figcaption>Pair correlation g(r_12) evolving over imaginary time.</figcaption>
      </figure>""")
            if gif_parts:
                parts.append(f"""
    <div class=\"grid\">{''.join(gif_parts)}
    </div>""")
            density_section = f"""
  <h4>Physical Densities</h4>{''.join(parts)}"""

        block = f"""
<section class=\"case\">
  <h3>Case: combo={r['combo']}, d={r['distance']:.1f}, coulomb={'on' if r['coulomb'] else 'off'}</h3>
  <p><strong>Ground truth reference:</strong> {r['gt_label']} (E_gt = {r['e_gt']:.6f})</p>
  <div class=\"grid\">
    <figure>
      <img src=\"{r['energy_plot']}\" alt=\"Energy plot {r['slug']}\" />
      <figcaption>Energy trajectory with MC uncertainty and ground-truth line.</figcaption>
    </figure>
    <figure>
      <img src=\"{r['error_plot']}\" alt=\"Error plot {r['slug']}\" />
      <figcaption>Absolute error decay on log scale.</figcaption>
    </figure>
  </div>
  <figure>
    <img src=\"{r['animation']}\" alt=\"Animation {r['slug']}\" />
    <figcaption>Animation of E(tau) approaching ground truth over imaginary time.</figcaption>
  </figure>{density_section}
</section>
"""
        blocks.append(block)

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Imaginary-Time Evolution Report</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 28px auto; padding: 0 16px; max-width: 1100px; color: #1b1b1b; }}
    h1, h2 {{ color: #1c4e80; }}
    .meta {{ color: #555; margin-bottom: 14px; }}
    .panel {{ background: #f8fbff; border: 1px solid #c8ddf0; border-radius: 10px; padding: 14px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border: 1px solid #d7d7d7; padding: 7px 8px; text-align: center; font-size: 0.94rem; }}
    th {{ background: #eef5fc; }}
    figure {{ margin: 10px 0; }}
    img {{ width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
    figcaption {{ font-size: 0.92rem; color: #555; margin-top: 4px; }}
    .case {{ margin: 28px 0 36px; border-top: 2px solid #e8e8e8; padding-top: 16px; }}
  </style>
</head>
<body>
  <h1>Imaginary-Time Evolution: Plots, Animations, and Ground-Truth Comparison</h1>
  <p class=\"meta\">Generated: {generated}</p>
  <p class=\"meta\">Source run: {search_dir}</p>

  <div class=\"panel\">
    <h2>Summary</h2>
    <p>This report compares measured PINN imaginary-time trajectories against ground-truth references.</p>
    <p><strong>Ground-truth mode:</strong> {gt_mode} ({gt_source})</p>
    <img src=\"{summary_plot}\" alt=\"Summary final error\" />
    <table>
      <thead>
        <tr>
          <th>combo</th><th>d</th><th>coulomb</th><th>loss</th><th>tau_sampling</th><th>x_sampling</th>
          <th>E_gt</th><th>E(T)</th><th>final |err|</th><th>RMSE</th><th>decay rate</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>

  <h2>Per-Case Time Evolution</h2>
  {''.join(blocks)}
</body>
</html>
"""

    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate time-evolution plots/animations and HTML report.")
    parser.add_argument("--search-dir", type=str, default="", help="Path to strategy_search_* directory")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory (default: <search-dir>/time_evolution_report)")
    parser.add_argument(
        "--ed-ground-truth",
        type=str,
        default="",
        help="Path to ed_ground_truth.json. If provided, E0 uses diagonalization results.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    search_dir = Path(args.search_dir) if args.search_dir else latest_strategy_dir(root)
    if not search_dir.exists():
        raise FileNotFoundError(f"search dir does not exist: {search_dir}")

    score_file = search_dir / "strategy_scores.tsv"
    if not score_file.exists():
        raise FileNotFoundError(f"Missing score file: {score_file}")

    out_dir = Path(args.out_dir) if args.out_dir else (search_dir / "time_evolution_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    ed_gt: Dict[Tuple[float, bool], dict] | None = None
    gt_mode = "asymptotic"
    gt_source = "non-interacting exact + interacting 2+1/d approximation"
    if args.ed_ground_truth:
        ed_path = Path(args.ed_ground_truth)
        if not ed_path.exists():
            raise FileNotFoundError(f"Missing ED ground-truth JSON: {ed_path}")
        ed_gt = load_ed_ground_truth(ed_path)
        gt_mode = "finite-difference diagonalization"
        gt_source = str(ed_path)

    rows = read_case_rows(score_file)
    all_cases: List[CaseData] = []
    for row in rows:
        all_cases.extend(read_case_jsons(row, ed_gt=ed_gt))

    if not all_cases:
        raise RuntimeError("No case data found from strategy score rows")

    report_rows = [render_case_artifacts(case, out_dir) for case in all_cases]
    report_rows.sort(key=lambda r: (r["combo"], r["coulomb"], r["distance"]))

    summary_plot = render_summary_plot(report_rows, out_dir)
    out_html = out_dir / "time_evolution_report.html"
    write_html(
        search_dir=search_dir,
        rows=report_rows,
        summary_plot=summary_plot,
        out_html=out_html,
        gt_mode=gt_mode,
        gt_source=gt_source,
    )

    print(f"Report generated: {out_html}")
    print(f"Artifacts directory: {out_dir}")


if __name__ == "__main__":
    main()
