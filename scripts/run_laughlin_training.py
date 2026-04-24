#!/usr/bin/env python3
"""Train a Laughlin × PINN wavefunction for fractional quantum Hall states.

The wavefunction is Ψ(x) = Ψ_Laughlin(x) × exp(J_PINN(x)) where:
  - Ψ_Laughlin = ∏_{i<j}(z_i - z_j)^m × exp(-B/4 Σ|r_i|²)  [analytic base]
  - J_PINN = PINN Jastrow factor (learns deviations from pure Laughlin)

Training minimises Var(Re[E_L]) + imag_penalty × ⟨Im[E_L]²⟩
where E_L is the complex QHE local energy.

Physics targets (per run):
  - Angular momentum Lz → -m N(N-1)/2  (Laughlin quantum number)
  - Pair correlation g(r) ~ r^{2m}  at short distance (Laughlin gap)
  - Energy comparison: E_PINN vs E_Laughlin_exact (for N=4 benchmarkable via ED)

Usage:
    CUDA_MANUAL_DEVICE=0 PYTHONPATH=src python3.11 scripts/run_laughlin_training.py \\
        --config configs/qhe/n6_dot_nu0p333_s42.yaml --seed 42

    CUDA_MANUAL_DEVICE=1 PYTHONPATH=src python3.11 scripts/run_laughlin_training.py \\
        --config configs/qhe/n6_ring_nu0p333_s42.yaml --seed 42 --no-laughlin-base
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from laughlin import LaughlinJastrowWF, laughlin_log_amplitude, laughlin_phase
from training.qhe_collocation import qhe_loss
from training.stratified_sampler import stratified_sample
from PINN import PINN


def _build_pinn_jastrow(cfg: dict, n_particles: int, device: str, dtype: torch.dtype) -> PINN:
    arch = cfg.get("architecture", {})
    net = PINN(
        n_particles=n_particles,
        d=2,
        omega=float(cfg["system"]["wells"][0].get("omega", 1.0)),
        hidden_dim=arch.get("pinn_hidden", 128),
        n_layers=arch.get("pinn_layers", 3),
    ).to(device=device, dtype=dtype)
    return net


def _make_sampler_cfg(cfg: dict) -> dict:
    tr = cfg.get("training", {})
    return {
        "sigma_center": tr.get("sampler_sigma_center", 0.5),
        "sigma_tails": tr.get("sampler_sigma_tails", 1.0),
        "sigma_mixed_in": tr.get("sampler_sigma_mixed_in", 0.3),
        "sigma_mixed_out": tr.get("sampler_sigma_mixed_out", 0.8),
        "shell_radius": tr.get("sampler_shell_radius", 1.5),
        "shell_radius_sigma": tr.get("sampler_shell_radius_sigma", 0.2),
        "dimer_pairs": tr.get("sampler_dimer_pairs", 2),
        "dimer_eps_max": tr.get("sampler_dimer_eps_max", 0.05),
        "mix_weights": tr.get("sampler_mix_weights", [0.7, 0.05, 0.15, 0.05, 0.05]),
    }


def train(
    config_path: Path,
    *,
    seed: int = 42,
    use_laughlin_base: bool = True,
    device: str = "cuda:0",
    epochs: int | None = None,
    imag_penalty: float = 0.5,
) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    torch.manual_seed(seed)

    tr = cfg["training"]
    device_str = device
    dtype = torch.float64 if tr.get("dtype", "float64") == "float64" else torch.float32

    qhe_cfg = cfg.get("qhe", {})
    m = int(qhe_cfg.get("laughlin_m", 3))
    B = float(qhe_cfg.get("B_orbital", cfg["system"].get("B_magnitude", 6.0)))
    nu = float(qhe_cfg.get("filling_nu", 1 / m))
    omega = float(cfg["system"]["wells"][0].get("omega", 1.0))

    n_up = int(cfg["spin"]["n_up"])
    n_down = int(cfg["spin"].get("n_down", 0))
    N = n_up + n_down

    n_epochs = epochs or int(tr.get("epochs", 6000))
    n_coll = int(tr.get("n_coll", 256))
    lr = float(tr.get("lr", 5e-5))
    clip_w = float(tr.get("local_energy_clip_width", 5.0))
    print_every = int(tr.get("print_every", 200))

    # Build wavefunction
    jastrow_net = _build_pinn_jastrow(cfg, N, device_str, dtype) if use_laughlin_base else None
    wf = LaughlinJastrowWF(N, m=m, B=B, jastrow=jastrow_net).to(device_str)

    # If not using Laughlin base, train plain PINN with QHE local energy
    if not use_laughlin_base:
        plain_pinn = _build_pinn_jastrow(cfg, N, device_str, dtype)
        log_amp_fn = lambda x: plain_pinn(x).squeeze(-1)
        phase_fn = lambda x: laughlin_phase(x, m=m)
        params = list(plain_pinn.parameters())
    else:
        log_amp_fn = wf.log_amplitude
        phase_fn = wf.phase
        params = list(jastrow_net.parameters()) if jastrow_net is not None else []

    if not params:
        raise ValueError("No trainable parameters — set use_laughlin_base=True or provide PINN")

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.05)

    # Sampler config — build well centers tensor
    wells = cfg["system"]["wells"]
    from config import SystemConfig, WellConfig
    system = SystemConfig(
        type="custom", dim=2, coulomb=True,
        B_magnitude=B, B_direction=[0., 0., 1.], g_factor=2.0, mu_B=1.0,
        wells=[WellConfig(center=w["center"], omega=w.get("omega", 1.0),
                          n_particles=w.get("n_particles", 1)) for w in wells],
    )

    sampler_kw = _make_sampler_cfg(cfg)

    run_name = cfg.get("run_name", config_path.stem) + f"_seed{seed}"
    out_dir = REPO / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"QHE Training: N={N}, ν={nu:.3f} (m={m}), B={B:.3f}, l_B={1/B**0.5:.3f}")
    print(f"  epochs={n_epochs}, n_coll={n_coll}, lr={lr}, imag_penalty={imag_penalty}")
    print(f"  use_laughlin_base={use_laughlin_base}, params={sum(p.numel() for p in params):,}")

    history = []
    best_energy = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        # Sample from |Ψ|² via stratified sampler (ignores phase for sampling)
        with torch.no_grad():
            x = stratified_sample(
                n_samples=n_coll, n_particles=N, d=2,
                system=system, dtype=dtype, device=device_str,
                **sampler_kw,
            )

        # Compute QHE loss
        loss, E_mean, e_loc, diag = qhe_loss(
            log_amp_fn, phase_fn, x,
            B=B, omega=omega,
            imag_penalty=imag_penalty,
            clip_width=clip_w,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(tr.get("grad_clip", 0.1)))
        optimizer.step()
        scheduler.step()

        if epoch % print_every == 0 or epoch == n_epochs - 1:
            lz_expected = -m * N * (N - 1) / 2.0
            print(
                f"epoch={epoch:5d}  E={E_mean:.6f}  var={diag['energy_var']:.2e}"
                f"  imag={diag['imag_penalty']:.2e}  lr={scheduler.get_last_lr()[0]:.2e}"
            )
            history.append({"epoch": epoch, **diag})
            if E_mean < best_energy:
                best_energy = E_mean
                best_state = {k: v.clone() for k, v in (
                    plain_pinn if not use_laughlin_base else jastrow_net
                ).state_dict().items()} if params else None

    # Save results
    summary = {
        "config": str(config_path),
        "seed": seed,
        "use_laughlin_base": use_laughlin_base,
        "m": m, "B": B, "nu": nu, "N": N,
        "final_energy": best_energy,
        "lz_expected": -m * N * (N - 1) / 2.0,
        "history": history,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = (
        REPO / "results" / "diag_sweeps" /
        f"{run_name}__laughlin_m{m}__summary_{ts}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary → {summary_path.name}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-laughlin-base", action="store_true",
                        help="Train plain PINN with QHE local energy (no Laughlin factor)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imag-penalty", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    import os
    gpu_idx = os.environ.get("CUDA_MANUAL_DEVICE", "0")
    device = args.device or f"cuda:{gpu_idx}"

    train(
        args.config,
        seed=args.seed,
        use_laughlin_base=not args.no_laughlin_base,
        device=device,
        epochs=args.epochs,
        imag_penalty=args.imag_penalty,
    )


if __name__ == "__main__":
    main()
