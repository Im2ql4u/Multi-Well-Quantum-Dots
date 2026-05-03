#!/usr/bin/env python3
"""Real-time PINN: 2D harmonic-oscillator ω-quench at N=2, against closed-form.

This is the **first non-trivial** test of the real-time NQS PINN
(``src/realtime_pinn.py``). The setup uses an analytical ground state
(Gaussian product) of an isotropic 2D harmonic oscillator at frequency
``ω₀`` and applies a sudden frequency change to ``ω₁`` at ``t = 0``. The
analytical breathing-mode evolution then provides a closed-form benchmark
for ``⟨Σ_i |x_i|²⟩(t)``.

Outputs (under ``results/realtime_pinn/omega_quench/<run_name>/``):

* ``omega_quench.json`` — config, training history, PINN observable on a
  dense time grid, analytical reference at the same times, residuals.
* ``omega_quench.png`` — 2-panel figure: (i) ``⟨|x|²⟩(t)`` PINN vs analytic;
  (ii) the norm-ratio diagnostic ``Z(t) := ⟨e^{2 g_R}⟩``.

Usage
-----
::

    PYTHONPATH=src python3.11 scripts/run_realtime_n2_omega_quench.py \\
        --device cuda:6 \\
        --omega0 1.0 --omega1 2.0 \\
        --epochs 1200 \\
        --t-max 1.5  # ≈ one breathing period at ω₁=2 is π/2 ≈ 1.57
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch.set_default_dtype(torch.float64)

from realtime_pinn import (  # noqa: E402
    PolynomialQuenchNet,
    RealTimeNet,
    RealTimeTrainConfig,
    train_realtime_pinn,
)
from realtime_quench_ho import (  # noqa: E402
    HOQuenchConfig,
    analytical_breathing_period,
    analytical_x2_aggregate,
    build_ho_quench_pool,
    pinn_x2_aggregate,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="cuda:6", help="torch device (cuda:N or cpu)")
    p.add_argument("--n-particles", type=int, default=2)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--omega1", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n-pool", type=int, default=2048)
    p.add_argument("--n-eval", type=int, default=512)
    p.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="If unset, defaults to the analytical breathing period π/ω₁.",
    )
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument(
        "--norm-weight",
        type=float,
        default=0.0,
        help="Weight for the unitarity regularizer (log Z(t))². 0 disables.",
    )
    p.add_argument(
        "--anchor-weight",
        type=float,
        default=0.0,
        help="Weight for the small-t analytic anchor loss. 0 disables.",
    )
    p.add_argument(
        "--anchor-t-frac",
        type=float,
        default=0.02,
        help="Fraction of t_max used as the small-t anchor point.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--t-embed", type=int, default=24)
    p.add_argument("--n-freq", type=int, default=4)
    p.add_argument("--output-scale", type=float, default=1.0)
    p.add_argument(
        "--quad-feats",
        action="store_true",
        help="Enable per-coordinate x², per-particle |x|², and total Σ|x|² "
             "as extra spatial features (makes Gaussian-quench geometry "
             "easier to represent at the cost of a noisier optimisation).",
    )
    p.add_argument(
        "--ansatz",
        choices=["mlp", "polynomial"],
        default="mlp",
        help="Network architecture: 'mlp' uses RealTimeNet (general); "
             "'polynomial' uses PolynomialQuenchNet which has the exact "
             "analytical structure for separable Gaussian quenches.",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="If unset, derived from physics parameters (omega0/omega1).",
    )
    p.add_argument(
        "--out-root",
        default=str(REPO / "results" / "realtime_pinn" / "omega_quench"),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu.")
        args.device = "cpu"
    device = torch.device(args.device)

    cfg = HOQuenchConfig(
        n_particles=args.n_particles,
        dim=2,
        omega_0=args.omega0,
        omega_1=args.omega1,
    )
    breathing_T = analytical_breathing_period(cfg)
    t_max = float(args.t_max) if args.t_max is not None else breathing_T

    nw_tag = f"_nw{args.norm_weight:g}" if args.norm_weight > 0 else ""
    run_name = args.run_name or (
        f"n{cfg.n_particles}_w0{cfg.omega_0:g}_w1{cfg.omega_1:g}"
        f"_e{args.epochs}{nw_tag}_s{args.seed}"
    )
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Quench: N={cfg.n_particles}, dim={cfg.dim}, ω₀={cfg.omega_0:g}, ω₁={cfg.omega_1:g}"
    )
    print(f"Breathing period T = π/ω₁ = {breathing_T:.4f}; t_max = {t_max:.4f}")
    print(f"Output: {out_dir}")

    pool = build_ho_quench_pool(
        cfg, n_samples=args.n_pool, seed=args.seed, device=device, dtype=torch.float64
    )
    print(
        f"Pool: n={args.n_pool}, σ={pool['sigma']:.4f}, "
        f"E₀={pool['E_0']:.4f}, ΔV mean={pool['deltaV'].mean().item():.4f}, "
        f"⟨Σ|x_i|²⟩₀ (sample)={(pool['x'] ** 2).sum(dim=(1, 2)).mean().item():.4f}"
    )

    if args.ansatz == "polynomial":
        net = PolynomialQuenchNet(
            n_particles=cfg.n_particles,
            dim=cfg.dim,
            hidden=args.hidden,
            t_embed=args.t_embed,
            n_freq=args.n_freq,
            t_scale=1.0 / max(t_max, 1e-8),
        ).to(device)
    else:
        net = RealTimeNet(
            n_particles=cfg.n_particles,
            dim=cfg.dim,
            hidden=args.hidden,
            n_layers=args.n_layers,
            t_embed=args.t_embed,
            n_freq=args.n_freq,
            t_scale=1.0 / max(t_max, 1e-8),
            output_scale=args.output_scale,
            use_quadratic_features=args.quad_feats,
        ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Net: {n_params} params")

    train_cfg = RealTimeTrainConfig(
        n_epochs=args.epochs,
        batch_pde=args.batch,
        t_max=t_max,
        lr=args.lr,
        norm_weight=args.norm_weight,
        anchor_weight=args.anchor_weight,
        anchor_t_frac=args.anchor_t_frac,
        print_every=max(1, args.epochs // 12),
        history_every=max(1, args.epochs // 60),
        seed=args.seed,
    )
    t0 = time.time()
    out = train_realtime_pinn(
        net,
        x_pool=pool["x"],
        E_L0_pool=pool["E_L0"],
        grad_log_psi0_pool=pool["grad_log_psi0"],
        deltaV_pool=pool["deltaV"],
        train_cfg=train_cfg,
    )
    train_wall = time.time() - t0
    print(f"Train wall = {train_wall:.1f}s")

    # Eval pool: independent samples from |ψ₀|² (different seed) so we don't
    # measure on training points only.
    eval_pool = build_ho_quench_pool(
        cfg, n_samples=args.n_eval, seed=args.seed + 1000, device=device, dtype=torch.float64
    )

    n_t = 60
    t_grid = np.linspace(0.0, t_max, n_t)

    pinn = pinn_x2_aggregate(net, eval_pool, t_grid)
    analytic = analytical_x2_aggregate(cfg, t_grid)
    abs_err = np.abs(pinn["mean"] - analytic)
    rel_err = abs_err / np.maximum(np.abs(analytic), 1e-12)

    rms_abs = float(np.sqrt(np.mean(abs_err**2)))
    rms_rel = float(np.sqrt(np.mean(rel_err**2)))
    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())
    mean_Z = float(pinn["Z"].mean())
    max_Z_dev = float(np.abs(pinn["Z"] - 1.0).max())

    print("\nResult summary:")
    print(f"  RMS abs err on ⟨Σ|x_i|²⟩(t) : {rms_abs:.4e}")
    print(f"  RMS rel err on ⟨Σ|x_i|²⟩(t) : {rms_rel:.4e}  ({rms_rel * 100:.2f}%)")
    print(f"  Max abs / rel err          : {max_abs:.4e} / {max_rel:.4e}")
    print(f"  ⟨Z(t)⟩ ≈ {mean_Z:.4f}; max |Z(t)-1| = {max_Z_dev:.4e}")

    # Print a sparse table at canonical times (0, T/4, T/2, 3T/4, T).
    print("\n  t        analytic        PINN       abs err   rel err     Z(t)")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t_val = frac * breathing_T
        if t_val > t_max + 1e-12:
            continue
        i = int(np.argmin(np.abs(t_grid - t_val)))
        print(
            f"  {t_grid[i]:.4f}  {analytic[i]:.6f}  {pinn['mean'][i]:.6f}  "
            f"{abs_err[i]:.3e}  {rel_err[i]:.3e}  {pinn['Z'][i]:.4f}"
        )

    summary = {
        "run_name": run_name,
        "config": {
            "device": str(device),
            "n_particles": cfg.n_particles,
            "dim": cfg.dim,
            "omega_0": cfg.omega_0,
            "omega_1": cfg.omega_1,
            "epochs": args.epochs,
            "batch": args.batch,
            "n_pool": args.n_pool,
            "n_eval": args.n_eval,
            "t_max": t_max,
            "breathing_T": breathing_T,
            "lr": args.lr,
            "seed": args.seed,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "t_embed": args.t_embed,
            "n_freq": args.n_freq,
            "output_scale": args.output_scale,
        },
        "n_params": n_params,
        "train_wall_seconds": train_wall,
        "final_loss": out["final_loss"],
        "history": out["history"],
        "evaluation": {
            "t": t_grid.tolist(),
            "pinn_x2_aggregate": pinn["mean"].tolist(),
            "analytic_x2_aggregate": analytic.tolist(),
            "Z": pinn["Z"].tolist(),
            "abs_err": abs_err.tolist(),
            "rel_err": rel_err.tolist(),
        },
        "metrics": {
            "rms_abs_err": rms_abs,
            "rms_rel_err": rms_rel,
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "mean_Z": mean_Z,
            "max_Z_dev": max_Z_dev,
        },
    }
    out_json = out_dir / "omega_quench.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax_obs, ax_norm = axes

        ax_obs.plot(
            t_grid, analytic, "k--", lw=1.6,
            label=rf"analytic: $\frac{{N}}{{\omega_0}}\cos^2(\omega_1 t)+\frac{{N\omega_0}}{{\omega_1^2}}\sin^2(\omega_1 t)$",
        )
        ax_obs.plot(
            t_grid, pinn["mean"], "o-", color="C3", lw=1.5, ms=3,
            label=r"PINN reweighted",
        )
        ax_obs.set_xlabel("t")
        ax_obs.set_ylabel(r"$\langle\sum_i |x_i|^2\rangle(t)$")
        ax_obs.set_title(
            rf"$\omega_0\!=\!{cfg.omega_0:g}\to\omega_1\!=\!{cfg.omega_1:g}$ quench, N={cfg.n_particles}"
        )
        ax_obs.axvline(breathing_T, color="0.5", lw=0.8, ls=":")
        ax_obs.text(
            breathing_T, ax_obs.get_ylim()[1] * 0.9, r"  $T=\pi/\omega_1$",
            ha="left", va="top", color="0.5",
        )
        ax_obs.legend(fontsize=8, loc="best")
        ax_obs.grid(True, alpha=0.3)

        ax_norm.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.6,
                        label=r"unitary: $Z(t)\equiv 1$")
        ax_norm.plot(t_grid, pinn["Z"], "o-", color="C2", lw=1.4, ms=3)
        ax_norm.set_xlabel("t")
        ax_norm.set_ylabel(r"$Z(t)=\langle e^{2g_R}\rangle_{|\psi_0|^2}$")
        ax_norm.set_title("Norm-conservation diagnostic")
        ax_norm.legend(fontsize=8, loc="best")
        ax_norm.grid(True, alpha=0.3)

        out_png = out_dir / "omega_quench.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"Wrote {out_png}")
    except ImportError:
        print("matplotlib not available; skipped plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
