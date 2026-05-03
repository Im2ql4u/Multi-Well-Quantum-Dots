#!/usr/bin/env python3
"""Real-time PINN: 2D HO + softcore-Coulomb quench at N=2.

This is the **non-separable** stress test of the real-time PINN
(``src/realtime_pinn.py``) and the polynomial-backbone hybrid network
(``HybridPolyMLPNet``). Setup details live in
``src/realtime_quench_coulomb.py``: a non-interacting Gaussian ground state
at frequency ``ω₀`` is suddenly evolved under

    H = H_0 + ½(ω₁² - ω₀²) Σ|x_i|² + λ Σ_{i<j} 1/√(|x_i - x_j|² + ε²).

The polynomial ansatz exactly absorbs the ω-quench piece; the MLP residual
is in charge of representing the genuinely non-separable Coulomb
correction. We measure two observables vs time: the breathing observable
``⟨Σ|x_i|²⟩(t)`` and the pair-interaction expectation ``⟨V_int⟩(t)``.

Outputs (under ``results/realtime_pinn/coulomb_quench/<run_name>/``):

* ``coulomb_quench.json`` — config, training history, observables on a
  dense time grid, residuals.
* ``coulomb_quench.png`` — three panels: (i) ``⟨Σ|x_i|²⟩(t)`` PINN vs
  ``λ=0`` analytical baseline; (ii) ``⟨V_int⟩(t)`` PINN; (iii) Z(t).

Usage
-----
::

    PYTHONPATH=src python3.11 scripts/run_realtime_n2_coulomb_quench.py \\
        --device cuda:6 \\
        --omega0 1.0 --omega1 2.0 \\
        --lambda-coul 0.5 --epsilon-coul 0.05 \\
        --ansatz hybrid \\
        --epochs 2000

Setting ``--lambda-coul 0`` reduces the experiment to the closed-form
ω-quench (a sanity check against the polynomial ansatz). Setting
``--ansatz polynomial`` with non-zero λ shows the breakdown of the
polynomial-only representation.
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
    HybridPolyMLPNet,
    PolynomialQuenchNet,
    RealTimeNet,
    RealTimeTrainConfig,
    train_realtime_pinn,
)
from realtime_quench_coulomb import (  # noqa: E402
    CoulombQuenchConfig,
    analytical_breathing_period_coulomb,
    analytical_x2_aggregate_ohne_coulomb,
    build_coulomb_quench_pool,
    initial_pair_interaction_value,
    pinn_pair_interaction_aggregate,
)
from realtime_quench_ho import pinn_x2_aggregate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="cuda:6", help="torch device (cuda:N or cpu)")
    p.add_argument("--n-particles", type=int, default=2)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--omega1", type=float, default=2.0)
    p.add_argument(
        "--lambda-coul", type=float, default=0.5,
        help="Strength λ of the softcore-Coulomb pair interaction.",
    )
    p.add_argument(
        "--epsilon-coul", type=float, default=0.05,
        help="Softcore regularisation length ε in 1/√(r²+ε²).",
    )
    p.add_argument("--epochs", type=int, default=2000)
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
        "--norm-weight", type=float, default=0.0,
        help="Weight for the unitarity regularizer (log Z(t))². 0 disables.",
    )
    p.add_argument("--seed", type=int, default=0)
    # Polynomial backbone hyperparameters.
    p.add_argument("--poly-hidden", type=int, default=32)
    p.add_argument("--poly-t-embed", type=int, default=24)
    p.add_argument("--poly-n-freq", type=int, default=4)
    # MLP / hybrid residual hyperparameters.
    p.add_argument("--mlp-hidden", type=int, default=48)
    p.add_argument("--mlp-n-layers", type=int, default=3)
    p.add_argument("--mlp-t-embed", type=int, default=24)
    p.add_argument("--mlp-n-freq", type=int, default=4)
    p.add_argument(
        "--residual-scale", type=float, default=0.1,
        help="Hybrid: scalar multiplier on the MLP-residual contribution.",
    )
    p.add_argument(
        "--ansatz",
        choices=["mlp", "polynomial", "hybrid"],
        default="hybrid",
        help="Network architecture.",
    )
    p.add_argument(
        "--run-name", default=None,
        help="If unset, derived from physics parameters and ansatz.",
    )
    p.add_argument(
        "--out-root",
        default=str(REPO / "results" / "realtime_pinn" / "coulomb_quench"),
    )
    return p.parse_args()


def _build_net(args: argparse.Namespace, cfg: CoulombQuenchConfig, t_max: float, device):
    t_scale = 1.0 / max(t_max, 1e-8)
    if args.ansatz == "polynomial":
        return PolynomialQuenchNet(
            n_particles=cfg.n_particles,
            dim=cfg.dim,
            hidden=args.poly_hidden,
            t_embed=args.poly_t_embed,
            n_freq=args.poly_n_freq,
            t_scale=t_scale,
        ).to(device)
    if args.ansatz == "mlp":
        return RealTimeNet(
            n_particles=cfg.n_particles,
            dim=cfg.dim,
            hidden=args.mlp_hidden,
            n_layers=args.mlp_n_layers,
            t_embed=args.mlp_t_embed,
            n_freq=args.mlp_n_freq,
            t_scale=t_scale,
            output_scale=1.0,
        ).to(device)
    if args.ansatz == "hybrid":
        return HybridPolyMLPNet(
            n_particles=cfg.n_particles,
            dim=cfg.dim,
            poly_hidden=args.poly_hidden,
            poly_t_embed=args.poly_t_embed,
            poly_n_freq=args.poly_n_freq,
            mlp_hidden=args.mlp_hidden,
            mlp_n_layers=args.mlp_n_layers,
            mlp_t_embed=args.mlp_t_embed,
            mlp_n_freq=args.mlp_n_freq,
            t_scale=t_scale,
            residual_scale=args.residual_scale,
        ).to(device)
    raise ValueError(f"Unknown ansatz: {args.ansatz}.")


def main() -> int:
    args = _parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu.")
        args.device = "cpu"
    device = torch.device(args.device)

    cfg = CoulombQuenchConfig(
        n_particles=args.n_particles,
        dim=2,
        omega_0=args.omega0,
        omega_1=args.omega1,
        lambda_coul=args.lambda_coul,
        epsilon_coul=args.epsilon_coul,
    )
    breathing_T = analytical_breathing_period_coulomb(cfg)
    t_max = float(args.t_max) if args.t_max is not None else breathing_T

    lam_tag = f"_l{cfg.lambda_coul:g}_eps{cfg.epsilon_coul:g}"
    nw_tag = f"_nw{args.norm_weight:g}" if args.norm_weight > 0 else ""
    run_name = args.run_name or (
        f"{args.ansatz}_n{cfg.n_particles}_w0{cfg.omega_0:g}_w1{cfg.omega_1:g}"
        f"{lam_tag}_e{args.epochs}{nw_tag}_s{args.seed}"
    )
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Quench: N={cfg.n_particles}, dim={cfg.dim}, ω₀={cfg.omega_0:g}, "
        f"ω₁={cfg.omega_1:g}, λ={cfg.lambda_coul:g}, ε={cfg.epsilon_coul:g}"
    )
    print(f"Breathing period (ω-piece) T = π/ω₁ = {breathing_T:.4f}; t_max = {t_max:.4f}")
    print(f"Output: {out_dir}")

    pool = build_coulomb_quench_pool(
        cfg, n_samples=args.n_pool, seed=args.seed, device=device, dtype=torch.float64
    )
    coul0 = initial_pair_interaction_value(pool)
    omega_mean = pool["deltaV_omega_piece"].mean().item()
    print(
        f"Pool: n={args.n_pool}, σ={pool['sigma']:.4f}, E₀={pool['E_0']:.4f}, "
        f"⟨ΔV_ω⟩₀={omega_mean:.4f}, ⟨V_int⟩₀={coul0:.4f}, "
        f"⟨Σ|x_i|²⟩₀ (sample)={(pool['x'] ** 2).sum(dim=(1, 2)).mean().item():.4f}"
    )

    net = _build_net(args, cfg, t_max, device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Net ({args.ansatz}): {n_params} params")

    train_cfg = RealTimeTrainConfig(
        n_epochs=args.epochs,
        batch_pde=args.batch,
        t_max=t_max,
        lr=args.lr,
        norm_weight=args.norm_weight,
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

    eval_pool = build_coulomb_quench_pool(
        cfg, n_samples=args.n_eval, seed=args.seed + 1000, device=device, dtype=torch.float64
    )

    n_t = 60
    t_grid = np.linspace(0.0, t_max, n_t)

    pinn_x2 = pinn_x2_aggregate(net, eval_pool, t_grid)
    pinn_v_int = pinn_pair_interaction_aggregate(
        net,
        eval_pool,
        t_grid,
        epsilon=cfg.epsilon_coul,
        lambda_coul=cfg.lambda_coul,
    )
    analytic_x2_ohne = analytical_x2_aggregate_ohne_coulomb(cfg, t_grid)

    coulomb_shift_x2 = pinn_x2["mean"] - analytic_x2_ohne
    abs_x2_shift = float(np.max(np.abs(coulomb_shift_x2)))
    rel_x2_shift = abs_x2_shift / max(np.abs(analytic_x2_ohne).max(), 1e-12)
    mean_Z = float(pinn_x2["Z"].mean())
    max_Z_dev = float(np.abs(pinn_x2["Z"] - 1.0).max())

    print("\nResult summary:")
    print(
        f"  Max |⟨Σ|x|²⟩_PINN(t) - ⟨Σ|x|²⟩_λ=0(t)| (Coulomb shift): "
        f"{abs_x2_shift:.4e}  ({rel_x2_shift * 100:.2f}%)"
    )
    print(f"  ⟨V_int⟩_PINN(0)  ≈ {pinn_v_int['mean'][0]:.4e} (pool: {coul0:.4e})")
    print(f"  ⟨V_int⟩_PINN(T/2)≈ {pinn_v_int['mean'][n_t // 2]:.4e}")
    print(f"  ⟨V_int⟩_PINN(T)  ≈ {pinn_v_int['mean'][-1]:.4e}")
    print(f"  ⟨Z(t)⟩ ≈ {mean_Z:.4f}; max |Z(t)-1| = {max_Z_dev:.4e}")

    print("\n  t        x2_PINN     x2_λ=0      Δx2          V_int(t)     Z(t)")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t_val = frac * breathing_T
        if t_val > t_max + 1e-12:
            continue
        i = int(np.argmin(np.abs(t_grid - t_val)))
        print(
            f"  {t_grid[i]:.4f}  {pinn_x2['mean'][i]:.6f}  "
            f"{analytic_x2_ohne[i]:.6f}  {coulomb_shift_x2[i]:+.3e}   "
            f"{pinn_v_int['mean'][i]:.4e}   {pinn_x2['Z'][i]:.4f}"
        )

    summary = {
        "run_name": run_name,
        "config": {
            "device": str(device),
            "ansatz": args.ansatz,
            "n_particles": cfg.n_particles,
            "dim": cfg.dim,
            "omega_0": cfg.omega_0,
            "omega_1": cfg.omega_1,
            "lambda_coul": cfg.lambda_coul,
            "epsilon_coul": cfg.epsilon_coul,
            "epochs": args.epochs,
            "batch": args.batch,
            "n_pool": args.n_pool,
            "n_eval": args.n_eval,
            "t_max": t_max,
            "breathing_T": breathing_T,
            "lr": args.lr,
            "seed": args.seed,
            "poly_hidden": args.poly_hidden,
            "poly_t_embed": args.poly_t_embed,
            "poly_n_freq": args.poly_n_freq,
            "mlp_hidden": args.mlp_hidden,
            "mlp_n_layers": args.mlp_n_layers,
            "mlp_t_embed": args.mlp_t_embed,
            "mlp_n_freq": args.mlp_n_freq,
            "residual_scale": args.residual_scale,
            "norm_weight": args.norm_weight,
        },
        "n_params": n_params,
        "train_wall_seconds": train_wall,
        "final_loss": out["final_loss"],
        "history": out["history"],
        "evaluation": {
            "t": t_grid.tolist(),
            "pinn_x2_aggregate": pinn_x2["mean"].tolist(),
            "analytic_x2_ohne_coulomb": analytic_x2_ohne.tolist(),
            "coulomb_shift_x2": coulomb_shift_x2.tolist(),
            "pinn_pair_interaction": pinn_v_int["mean"].tolist(),
            "Z": pinn_x2["Z"].tolist(),
        },
        "metrics": {
            "max_coulomb_shift_x2_abs": abs_x2_shift,
            "max_coulomb_shift_x2_rel": float(rel_x2_shift),
            "mean_Z": mean_Z,
            "max_Z_dev": max_Z_dev,
            "v_int_t0_pool": coul0,
            "v_int_t0_pinn": float(pinn_v_int["mean"][0]),
        },
    }
    out_json = out_dir / "coulomb_quench.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        ax_x2, ax_v, ax_z = axes

        ax_x2.plot(
            t_grid, analytic_x2_ohne, "k--", lw=1.5,
            label=r"$\lambda=0$ analytic ($\omega$-quench only)",
        )
        ax_x2.plot(
            t_grid, pinn_x2["mean"], "o-", color="C3", lw=1.4, ms=3,
            label=rf"PINN ({args.ansatz}, $\lambda={cfg.lambda_coul:g}$)",
        )
        ax_x2.set_xlabel("t")
        ax_x2.set_ylabel(r"$\langle\sum_i |x_i|^2\rangle(t)$")
        ax_x2.set_title("Breathing observable")
        ax_x2.axvline(breathing_T, color="0.5", lw=0.8, ls=":")
        ax_x2.legend(fontsize=8, loc="best")
        ax_x2.grid(True, alpha=0.3)

        ax_v.plot(
            t_grid, pinn_v_int["mean"], "o-", color="C0", lw=1.4, ms=3,
            label=rf"PINN $\langle V_{{\rm int}}\rangle(t)$",
        )
        ax_v.axhline(coul0, color="k", lw=1.0, ls="--", alpha=0.7,
                     label=rf"$\langle V_{{\rm int}}\rangle(0)$ from $|\psi_0|^2$")
        ax_v.set_xlabel("t")
        ax_v.set_ylabel(r"$\langle V_{\rm int}\rangle(t)$")
        ax_v.set_title("Pair-interaction observable")
        ax_v.legend(fontsize=8, loc="best")
        ax_v.grid(True, alpha=0.3)

        ax_z.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.7,
                     label=r"unitary: $Z(t)\equiv 1$")
        ax_z.plot(t_grid, pinn_x2["Z"], "o-", color="C2", lw=1.4, ms=3)
        ax_z.set_xlabel("t")
        ax_z.set_ylabel(r"$Z(t)$")
        ax_z.set_title("Norm-conservation diagnostic")
        ax_z.legend(fontsize=8, loc="best")
        ax_z.grid(True, alpha=0.3)

        suptitle = (
            rf"$\omega_0\!=\!{cfg.omega_0:g}\to\omega_1\!=\!{cfg.omega_1:g}$, "
            rf"$\lambda\!=\!{cfg.lambda_coul:g}$, $\varepsilon\!=\!{cfg.epsilon_coul:g}$, "
            rf"N={cfg.n_particles} ({args.ansatz})"
        )
        fig.suptitle(suptitle, y=1.02, fontsize=11)
        out_png = out_dir / "coulomb_quench.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_png}")
    except ImportError:
        print("matplotlib not available; skipped plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
