#!/usr/bin/env python3
"""Generate Quantum Hall / Laughlin configs.

Places N electrons in a single harmonic dot (or shallow multi-well trap),
applies a strong perpendicular magnetic field B, and targets filling ν = 1/m.

For Laughlin ν=1/3: use m=3, so B = N/ν × 2π/Area.
In our units (ℏ=e=m=1, ω=1): the magnetic length l_B = 1/√B.
Filling ν = ω_c / (2ω) where ω_c = B (cyclotron frequency).
At ν=1/3: B = 6ω = 6.

We put all N electrons in a SINGLE quantum dot (no separate wells)
to study Laughlin physics in a clean setting, then also run
a ring-of-wells geometry to study fragmentation.

Usage:
    python3 scripts/gen_qhe_configs.py
    python3 scripts/gen_qhe_configs.py --N-list 4 6 8 --nu 0.333 --omega 1.0
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "configs" / "qhe"

# Laughlin exponent m for filling ν = 1/m
_NU_TO_M = {
    0.5: 2,    # bosonic Laughlin (ν=1/2), not physical for electrons but useful test
    0.333: 3,  # ν=1/3 Laughlin (most studied)
    0.2: 5,    # ν=1/5 Laughlin
    1.0: 1,    # integer QHE (ν=1), filled Landau level
}


def _b_for_filling(nu: float, omega: float = 1.0) -> float:
    """Magnetic field B for filling ν in a harmonic dot.

    In a 2D harmonic trap with frequency ω, filling ν = 1/2 × ω_c/ω
    where ω_c = eB/m = B in our units.
    So B = 2ω/ν.
    """
    return 2.0 * omega / nu


def gen_single_dot_config(N: int, nu: float, omega: float = 1.0) -> dict:
    """Single large dot with N electrons — clean Laughlin setting."""
    B = _b_for_filling(nu, omega)
    m = _NU_TO_M.get(round(nu, 3), round(1 / nu))
    l_B = 1.0 / math.sqrt(B)

    # Effective confinement radius for N electrons at filling ν
    # In LLL: ⟨r²⟩ ≈ 2 l_B² N  →  R_dot ≈ l_B √(2N)
    # Use this to set the effective dot radius but keep ω from config

    cfg: dict = {
        "allow_missing_dmc": True,
        "run_name": f"qhe_n{N}_nu{str(nu).replace('.', 'p')}_s42",
        "_comment": f"QHE: N={N}, ν={nu} (m={m}), B={B:.3f}, l_B={l_B:.3f}",
        "architecture": {
            "arch_type": "pinn",
            "pinn_hidden": 128,
            "pinn_layers": 3,
            "use_backflow": False,
        },
        "system": {
            "type": "custom",
            "dim": 2,
            "coulomb": True,
            "B_magnitude": B,
            "B_direction": [0.0, 0.0, 1.0],
            "g_factor": 2.0,
            "mu_B": 1.0,
            "wells": [
                {
                    "center": [0.0, 0.0],
                    "omega": omega,
                    "n_particles": N,
                }
            ],
        },
        "spin": {
            # Fully polarised (all spins aligned): correct for Laughlin state
            "n_up": N,
            "n_down": 0,
        },
        "qhe": {
            "laughlin_m": m,
            "filling_nu": nu,
            "B_orbital": B,
            "l_B": round(l_B, 4),
            "use_laughlin_base": True,  # flag for run script
        },
        "training": {
            "epochs": 6000,
            "lr": 0.00005,
            "lr_warmup_epochs": 500,
            "lr_min_factor": 0.05,
            "n_coll": 256,
            "loss_type": "residual",
            "residual_objective": "energy_var",
            "residual_target_energy": None,
            "alpha_start": 0.0,
            "alpha_end": 0.0,
            "alpha_decay_frac": 0.6,
            "local_energy_clip_width": 5.0,
            "laplacian_mode": "autograd",
            "fd_h": 0.01,
            "sampler": "stratified",
            "non_mcmc_only": True,
            "sampler_mix_weights": [0.70, 0.05, 0.15, 0.05, 0.05],
            "sampler_sigma_center": float(round(l_B, 4)),
            "sampler_sigma_tails": float(round(2 * l_B, 4)),
            "sampler_sigma_mixed_in": float(round(l_B * 0.5, 4)),
            "sampler_sigma_mixed_out": float(round(l_B * 1.5, 4)),
            "sampler_shell_radius": float(round(l_B * math.sqrt(2 * N), 4)),
            "sampler_shell_radius_sigma": float(round(l_B * 0.3, 4)),
            "sampler_dimer_pairs": max(1, N - 1),
            "sampler_dimer_eps_max": float(round(l_B * 0.1, 4)),
            "grad_clip": 0.1,
            "print_every": 200,
            "seed": 42,
            "device": "cuda:0",
            "dtype": "float64",
        },
    }
    return cfg


def gen_ring_config(N: int, nu: float, omega: float = 1.0) -> dict:
    """N electrons in N wells arranged in a ring — fragmented Laughlin test.

    This probes whether the Laughlin state survives spatial fragmentation.
    """
    B = _b_for_filling(nu, omega)
    m = _NU_TO_M.get(round(nu, 3), round(1 / nu))
    l_B = 1.0 / math.sqrt(B)
    R_ring = l_B * math.sqrt(2 * N) * 1.5  # ring radius slightly outside cloud

    wells = []
    for k in range(N):
        angle = 2 * math.pi * k / N
        wells.append({
            "center": [round(R_ring * math.cos(angle), 4), round(R_ring * math.sin(angle), 4)],
            "omega": omega,
            "n_particles": 1,
        })

    cfg = gen_single_dot_config(N, nu, omega)
    cfg["run_name"] = f"qhe_ring_n{N}_nu{str(nu).replace('.', 'p')}_s42"
    cfg["system"]["wells"] = wells
    cfg["spin"]["n_up"] = N // 2
    cfg["spin"]["n_down"] = N - N // 2
    cfg["qhe"]["geometry"] = "ring"
    return cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N-list", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--nu", type=float, default=0.333, help="Filling factor ν")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--ring", action="store_true", help="Also generate ring geometry")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for N in args.N_list:
        for gen_fn, label in ([(gen_single_dot_config, "dot")] +
                               ([(gen_ring_config, "ring")] if args.ring else [])):
            cfg = gen_fn(N, args.nu, args.omega)
            nu_tag = str(args.nu).replace(".", "p")
            fname = f"n{N}_{label}_nu{nu_tag}_s42.yaml"
            out = OUT_DIR / fname
            comment = (
                f"# QHE: N={N} electrons, ν={args.nu}, B={_b_for_filling(args.nu, args.omega):.2f}\n"
                f"# Laughlin m={_NU_TO_M.get(round(args.nu, 3), '?')}, geometry={label}\n"
                f"# Generated by gen_qhe_configs.py\n"
            )
            yaml_str = comment + yaml.dump(cfg, default_flow_style=False, sort_keys=False)
            if args.dry_run:
                print(f"[DRY] {fname}")
            else:
                out.write_text(yaml_str)
                print(f"Written: {fname}")

    if not args.dry_run:
        print(f"\nConfigs in {OUT_DIR}")
        print("NOTE: QHE training requires src/training/qhe_collocation.py local energy")
        print("      Set use_laughlin_base: true to enable Laughlin × PINN wavefunction")


if __name__ == "__main__":
    main()
