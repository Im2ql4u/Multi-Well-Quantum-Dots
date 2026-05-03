#!/usr/bin/env python3
"""Trivial-evolution smoke test for the real-time NQS PINN (T1 prototype).

Run with::

    PYTHONPATH=src python3.11 scripts/smoke_realtime_pinn_trivial.py \\
        --device cuda:6 --epochs 600

Verifies the **defining sanity check** of ``src/realtime_pinn.py``:
when the system is evolved under the same ``H_0`` whose ground state was
used as ``ψ_0``, the analytical solution is ``ψ(x, t) = e^{-i E_0 t} ψ_0(x)``,
i.e. ``g_R(x, t) ≡ 0`` and ``g_I(x, t) = -E_0 t``.

Outputs (under ``results/realtime_pinn/trivial/``):

* ``trivial_smoke.json`` — final ``g_R``, ``g_I`` statistics on a held-out
  grid, plus the residual / loss history.
* ``trivial_smoke.png`` — a 2-panel diagnostic figure: training loss vs
  epoch, and ``⟨g_I⟩(t)`` vs the analytical ``-E_0 t``.

This script is **part of the T1 paper trail**; the equivalent pytest
(``tests/test_realtime_pinn.py::test_trivial_evolution_sign_and_monotonicity_cpu_budget``)
covers only the CPU-friendly qualitative behaviour.
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
    RealTimeNet,
    RealTimeTrainConfig,
    evaluate_realtime_state,
    train_realtime_pinn,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="cuda:0", help="torch device (cuda:N or cpu)")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n_pool", type=int, default=1024)
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--t_max", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_particles", type=int, default=2)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument(
        "--out_dir",
        default=str(REPO / "results" / "realtime_pinn" / "trivial"),
        help="output directory for JSON + PNG",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu.")
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    n, dim = args.n_particles, args.dim
    sigma = args.sigma
    alpha = 1.0 / sigma**2
    # 2D harmonic oscillator at ω=α; closed-shell ground state energy is N·ω.
    E_0 = float(n * alpha)

    x_np = rng.normal(0.0, sigma, size=(args.n_pool, n, dim))
    x_pool = torch.tensor(x_np, device=device)
    grad_log_psi0 = (-alpha * x_pool).clone()
    E_L0_pool = torch.full((args.n_pool,), E_0, device=device)

    print(f"Device: {device}")
    print(
        f"Pool: N={n}, dim={dim}, sigma={sigma}, alpha={alpha}, E_0={E_0:.3f}, "
        f"n_pool={args.n_pool}"
    )

    net = RealTimeNet(
        n_particles=n,
        dim=dim,
        hidden=64,
        n_layers=3,
        t_embed=24,
        n_freq=4,
        t_scale=1.0,
        output_scale=1.0,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Net: {n_params} params")

    cfg = RealTimeTrainConfig(
        n_epochs=args.epochs,
        batch_pde=args.batch,
        t_max=args.t_max,
        lr=args.lr,
        print_every=max(1, args.epochs // 12),
        history_every=max(1, args.epochs // 60),
        seed=args.seed,
    )
    t0 = time.time()
    out = train_realtime_pinn(
        net,
        x_pool=x_pool,
        E_L0_pool=E_L0_pool,
        grad_log_psi0_pool=grad_log_psi0,
        train_cfg=cfg,
    )
    train_wall = time.time() - t0
    print(f"Train wall = {train_wall:.1f}s")

    n_eval = args.n_eval
    x_eval_np = rng.normal(0.0, sigma, size=(n_eval, n, dim))
    x_eval = torch.tensor(x_eval_np, device=device)

    t_grid = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0]) * args.t_max
    rows = []
    print(
        f"\n{'t':>6} {'<g_R>':>10} {'|g_R|_rms':>10} {'<g_I>':>10} {'target':>10} "
        f"{'|g_I-target|_rms':>16} {'rel_err':>10}"
    )
    for t_val in t_grid:
        t_eval = torch.full((n_eval,), float(t_val), device=device)
        diag = evaluate_realtime_state(net, x_eval, t_eval)
        g_R, g_I = diag["g_R"], diag["g_I"]
        target = -E_0 * float(t_val)
        g_R_mean = g_R.mean().item()
        g_R_rms = g_R.pow(2).mean().sqrt().item()
        g_I_mean = g_I.mean().item()
        err_rms = (g_I - target).pow(2).mean().sqrt().item()
        rel = err_rms / max(abs(target), 1e-12)
        rows.append(
            {
                "t": float(t_val),
                "g_R_mean": g_R_mean,
                "g_R_rms": g_R_rms,
                "g_I_mean": g_I_mean,
                "target": target,
                "g_I_err_rms": err_rms,
                "rel_err": rel,
            }
        )
        print(
            f"{t_val:>6.2f} {g_R_mean:>10.4e} {g_R_rms:>10.4e} {g_I_mean:>10.4e} "
            f"{target:>10.4f} {err_rms:>16.4e} {rel:>10.2%}"
        )

    summary = {
        "config": {
            "device": str(device),
            "epochs": args.epochs,
            "batch": args.batch,
            "n_pool": args.n_pool,
            "n_eval": args.n_eval,
            "t_max": args.t_max,
            "lr": args.lr,
            "n_particles": n,
            "dim": dim,
            "sigma": sigma,
            "alpha": alpha,
            "E_0": E_0,
            "seed": args.seed,
        },
        "n_params": n_params,
        "train_wall_seconds": train_wall,
        "final_loss": out["final_loss"],
        "history": out["history"],
        "evaluation": rows,
    }
    out_json = out_dir / "trivial_smoke.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax_loss, ax_phase = axes

        epochs = out["history"]["epoch"]
        loss = out["history"]["loss"]
        ax_loss.semilogy(epochs, loss, "-", color="C0", lw=1.6)
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("PDE residual loss")
        ax_loss.set_title("Real-time PINN training loss")
        ax_loss.grid(True, alpha=0.3)

        ts = np.array([r["t"] for r in rows])
        g_I_means = np.array([r["g_I_mean"] for r in rows])
        target = -E_0 * ts
        ax_phase.plot(
            ts, target, "k--", lw=1.5, label=r"analytic: $-E_0 t$"
        )
        ax_phase.plot(
            ts, g_I_means, "o-", color="C3", lw=1.5, label=r"$\langle g_I\rangle(t)$",
        )
        ax_phase.set_xlabel("t")
        ax_phase.set_ylabel(r"$\langle g_I\rangle(t)$")
        ax_phase.set_title(rf"Trivial evolution recovers $-E_0 t$ phase ($E_0={E_0:g}$)")
        ax_phase.legend()
        ax_phase.grid(True, alpha=0.3)

        out_png = out_dir / "trivial_smoke.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"Wrote {out_png}")
    except ImportError:
        print("matplotlib not available; skipped plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
