#!/usr/bin/env python3
"""Post-training validation: virial theorem check + energy decomposition.

Loads trained models from result directories, evaluates energy components
(T, V_trap, V_int) with high-statistics importance-resampled collocation,
and checks the virial theorem:
  2<T> = 2<V_trap> - <V_int>   (for harmonic confinement + Coulomb)

Usage:
    python scripts/run_virial_check.py --result-dirs dir1 dir2 ... --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import SystemConfig, WellSpec
from observables.diagnostics import compute_virial_metrics
from potential import compute_potential
from training.sampling import importance_resample, sample_mixture, sample_multiwell
from wavefunction import GroundStateWF, setup_closed_shell_system, resolve_reference_energy


def _build_system(system_cfg: dict) -> SystemConfig:
    from dataclasses import replace as _replace

    kind = system_cfg.get("type", "single_dot")
    coulomb = system_cfg.get("coulomb", True)
    if kind == "single_dot":
        sys = SystemConfig.single_dot(
            N=int(system_cfg["n_particles"]),
            omega=float(system_cfg["omega"]),
            dim=int(system_cfg.get("dim", 2)),
        )
    elif kind == "double_dot":
        sys = SystemConfig.double_dot(
            N_L=int(system_cfg["n_left"]),
            N_R=int(system_cfg["n_right"]),
            sep=float(system_cfg["separation"]),
            omega=float(system_cfg["omega"]),
            dim=int(system_cfg.get("dim", 2)),
        )
    else:
        raise ValueError(f"Unsupported system type '{kind}'.")
    if not coulomb:
        sys = _replace(sys, coulomb=False)
    return sys


def _compute_local_energy_components(
    psi_log_fn,
    x: torch.Tensor,
    system: SystemConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute T, V_trap, V_int, E_L per sample using autograd.

    Uses the same soft-min multi-well potential as training (compute_potential).
    Returns (E_L, T, V_trap, V_int), each shape (B,).
    """
    from dataclasses import replace as _replace

    B, N, d = x.shape
    n_flat = N * d
    device, dtype = x.device, x.dtype

    # Kinetic energy via autograd: T = -0.5 * (lap_logpsi + |grad_logpsi|^2)
    x_req = x.detach().clone().requires_grad_(True)
    logpsi = psi_log_fn(x_req)  # (B,)
    grad_logpsi = torch.autograd.grad(
        logpsi.sum(), x_req, create_graph=True
    )[0]  # (B, N, d)
    grad_sq = (grad_logpsi ** 2).sum(dim=(1, 2))  # (B,)

    # Exact diagonal Laplacian: sum_i d^2(logpsi)/dx_i^2
    grad_flat = grad_logpsi.reshape(B, n_flat)  # (B, n_flat)
    lap_logpsi = torch.zeros(B, device=device, dtype=dtype)
    for i in range(n_flat):
        g_i = grad_flat[:, i]  # (B,)
        d2 = torch.autograd.grad(
            g_i.sum(), x_req, retain_graph=True
        )[0]  # (B, N, d)
        lap_logpsi = lap_logpsi + d2.reshape(B, n_flat)[:, i]

    T = -0.5 * (lap_logpsi + grad_sq)  # (B,)
    T = T.detach()

    with torch.no_grad():
        # Use compute_potential to get V_ext (same soft-min as training)
        # Compute total V with coulomb, then V_ext without coulomb
        sys_no_coul = _replace(system, coulomb=False)
        V_trap = compute_potential(x, sys_no_coul)  # (B,)

        # V_int (Coulomb) separately
        V_int = torch.zeros(B, device=device, dtype=dtype)
        if system.coulomb:
            V_with_coul = compute_potential(x, system)  # (B,)
            V_int = V_with_coul - V_trap

        E_L = T + V_trap + V_int

    return E_L, T, V_trap, V_int


@torch.no_grad()
def _sample_mh_batch(
    psi_log_fn,
    *,
    n_keep: int,
    n_elec: int,
    dim: int,
    omega: float,
    device: torch.device,
    dtype: torch.dtype,
    system: SystemConfig,
    sigma_fs: list[float],
    burn_in: int,
    step_scale: float,
) -> tuple[torch.Tensor, float]:
    use_multiwell = any(any(c != 0.0 for c in w.center) for w in system.wells)
    if use_multiwell:
        x, _ = sample_multiwell(n_keep, system, device=device, dtype=dtype, sigma_fs=sigma_fs)
    else:
        x, _ = sample_mixture(
            n_keep,
            n_elec,
            dim,
            omega,
            device=device,
            dtype=dtype,
            sigma_fs=sigma_fs,
        )

    logp = 2.0 * psi_log_fn(x)
    step_sigma = step_scale / math.sqrt(max(float(omega), 1e-8))
    accepted_total = 0.0
    tried_total = 0

    for _ in range(max(1, burn_in)):
        proposal = x + step_sigma * torch.randn_like(x)
        logp_prop = 2.0 * psi_log_fn(proposal)
        log_alpha = logp_prop - logp
        u = torch.rand(n_keep, device=device, dtype=dtype)
        accept = torch.log(u) < torch.minimum(log_alpha, torch.zeros_like(log_alpha))
        accept_mask = accept.view(-1, 1, 1)
        x = torch.where(accept_mask, proposal, x)
        logp = torch.where(accept, logp_prop, logp)
        accepted_total += float(accept.sum().item())
        tried_total += int(accept.numel())

    accept_rate = accepted_total / max(tried_total, 1)
    return x, accept_rate


def run_virial_check(
    result_dir: Path,
    device: str,
    n_samples: int,
    *,
    n_cand_mult: int,
    sigma_fs: list[float],
    langevin_steps: int,
    langevin_step_size: float,
    report_is_stats: bool,
    sampler: str,
    mh_burn_in: int,
    mh_step_scale: float,
) -> dict:
    """Load model from result_dir, evaluate energy decomposition, check virial."""
    config_path = result_dir / "config.yaml"
    model_path = result_dir / "model.pt"
    if not config_path.exists() or not model_path.exists():
        print(f"  SKIP {result_dir.name}: missing config.yaml or model.pt")
        return {}

    with config_path.open() as f:
        raw_cfg = yaml.safe_load(f)

    system = _build_system(raw_cfg["system"])
    arch_cfg = raw_cfg.get("architecture", {})
    train_cfg = raw_cfg.get("training", {})
    dtype_str = train_cfg.get("dtype", "float64")
    torch_dtype = torch.float64 if dtype_str == "float64" else torch.float32

    allow_missing_dmc = bool(raw_cfg.get("allow_missing_dmc", True))
    resolved_E_ref = resolve_reference_energy(
        system, raw_cfg.get("E_ref", "auto"), allow_missing_dmc=allow_missing_dmc,
    )
    C_occ, spin, params = setup_closed_shell_system(
        system,
        device=device,
        dtype=torch_dtype,
        E_ref=resolved_E_ref,
        allow_missing_dmc=allow_missing_dmc,
    )

    model = GroundStateWF(
        system, C_occ, spin, params,
        arch_type=arch_cfg.get("arch_type", "pinn"),
        pinn_hidden=int(arch_cfg.get("pinn_hidden", 64)),
        pinn_layers=int(arch_cfg.get("pinn_layers", 2)),
        bf_hidden=int(arch_cfg.get("bf_hidden", 32)),
        bf_layers=int(arch_cfg.get("bf_layers", 2)),
        use_backflow=bool(arch_cfg.get("use_backflow", True)),
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    dev = torch.device(device)
    model.to(dev).to(torch_dtype).eval()

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    N = sum(w.n_particles for w in system.wells)
    if sigma_fs:
        sigma_fs_eval = [float(s) for s in sigma_fs]
    else:
        sigma_fs_eval = [float(s) for s in train_cfg.get("sigma_fs", [0.8, 1.3, 2.0])]

    # Draw importance-resampled collocation points in batches
    batch_size = 2048
    n_batches = (n_samples + batch_size - 1) // batch_size

    sum_E, sum_E2 = 0.0, 0.0
    sum_T, sum_T2 = 0.0, 0.0
    sum_Vt, sum_Vt2 = 0.0, 0.0
    sum_Vi, sum_Vi2 = 0.0, 0.0
    total = 0
    ess_vals: list[float] = []
    top1_vals: list[float] = []
    top10_vals: list[float] = []
    mh_accept_rates: list[float] = []

    print(f"  Running {n_batches} batches of ~{batch_size} samples each...")

    for batch_i in range(n_batches):
        bsz = min(batch_size, n_samples - total)

        with torch.no_grad():
            if sampler == "is":
                resampled = importance_resample(
                    psi_log_fn, bsz, N, system.dim, system.omega,
                    device=dev, dtype=torch_dtype,
                    n_cand_mult=n_cand_mult,
                    sigma_fs=sigma_fs_eval,
                    langevin_steps=langevin_steps,
                    langevin_step_size=langevin_step_size,
                    return_stats=report_is_stats,
                    system=system,
                )
                if report_is_stats:
                    x_resampled, ess, is_stats = resampled
                    top1_vals.append(float(is_stats["top1_mass"]))
                    top10_vals.append(float(is_stats["top10_mass"]))
                else:
                    x_resampled, ess = resampled
                ess_vals.append(float(ess))
            else:
                x_resampled, accept_rate = _sample_mh_batch(
                    psi_log_fn,
                    n_keep=bsz,
                    n_elec=N,
                    dim=system.dim,
                    omega=system.omega,
                    device=dev,
                    dtype=torch_dtype,
                    system=system,
                    sigma_fs=sigma_fs_eval,
                    burn_in=mh_burn_in,
                    step_scale=mh_step_scale,
                )
                mh_accept_rates.append(float(accept_rate))

        # Compute energy components
        E_L, T, V_trap, V_int = _compute_local_energy_components(
            psi_log_fn, x_resampled, system,
        )

        sum_E += float(E_L.sum().item())
        sum_E2 += float((E_L ** 2).sum().item())
        sum_T += float(T.sum().item())
        sum_T2 += float((T ** 2).sum().item())
        sum_Vt += float(V_trap.sum().item())
        sum_Vt2 += float((V_trap ** 2).sum().item())
        sum_Vi += float(V_int.sum().item())
        sum_Vi2 += float((V_int ** 2).sum().item())
        total += bsz

        if (batch_i + 1) % 5 == 0 or batch_i == n_batches - 1:
            E_so_far = sum_E / total
            if sampler == "is":
                print(f"    batch {batch_i+1}/{n_batches}: E≈{E_so_far:.6f}, ESS={ess:.0f} (n={total})")
            else:
                print(
                    f"    batch {batch_i+1}/{n_batches}: E≈{E_so_far:.6f}, "
                    f"accept={mh_accept_rates[-1]:.3f} (n={total})"
                )

        del x_resampled, E_L, T, V_trap, V_int
        torch.cuda.empty_cache()

    def _stats(s1, s2, n):
        mean = s1 / n
        var = max(s2 / n - mean ** 2, 0.0)
        std = math.sqrt(var)
        stderr = std / math.sqrt(n)
        return mean, std, stderr

    E_mean, E_std, E_stderr = _stats(sum_E, sum_E2, total)
    T_mean, T_std, T_stderr = _stats(sum_T, sum_T2, total)
    Vt_mean, Vt_std, Vt_stderr = _stats(sum_Vt, sum_Vt2, total)
    Vi_mean, Vi_std, Vi_stderr = _stats(sum_Vi, sum_Vi2, total)

    # Virial theorem for harmonic confinement + Coulomb:
    # V_trap is homogeneous degree 2 in x → x·∇V_trap = 2V_trap
    # V_ee  is homogeneous degree -1 in (x_i - x_j) → Σ x·∇V_ee = -V_ee
    # Therefore: 2<T> = 2<V_trap> - <V_ee>
    virial_metrics = compute_virial_metrics(
        T_mean=T_mean,
        V_trap_mean=Vt_mean,
        V_int_mean=Vi_mean,
        E_mean=E_mean,
    )
    virial_lhs = virial_metrics["virial_lhs"]
    virial_rhs = virial_metrics["virial_rhs"]
    virial_residual = virial_metrics["virial_residual"]
    virial_relative = virial_metrics["virial_relative"]

    out = {
        "dir": result_dir.name,
        "E_mean": E_mean,
        "E_std": E_std,
        "E_stderr": E_stderr,
        "T_mean": T_mean,
        "T_stderr": T_stderr,
        "V_trap_mean": Vt_mean,
        "V_trap_stderr": Vt_stderr,
        "V_int_mean": Vi_mean,
        "V_int_stderr": Vi_stderr,
        "virial_residual": virial_residual,
        "virial_relative": virial_relative,
        "sampler": sampler,
        "ess_min": float(min(ess_vals)) if ess_vals else None,
        "ess_median": float(sorted(ess_vals)[len(ess_vals) // 2]) if ess_vals else None,
        "ess_max": float(max(ess_vals)) if ess_vals else None,
        "mh_accept_rate_mean": (float(sum(mh_accept_rates) / len(mh_accept_rates)) if mh_accept_rates else None),
        "n_cand_mult": int(n_cand_mult),
        "sigma_fs_eval": sigma_fs_eval,
        "langevin_steps": int(langevin_steps),
        "langevin_step_size": float(langevin_step_size),
        "mh_burn_in": int(mh_burn_in),
        "mh_step_scale": float(mh_step_scale),
        "n_samples": total,
    }
    if report_is_stats and top1_vals:
        out["top1_mass_mean"] = float(sum(top1_vals) / len(top1_vals))
        out["top10_mass_mean"] = float(sum(top10_vals) / len(top10_vals))

    print(f"\n  === {result_dir.name} ===")
    print(f"  E     = {E_mean:.6f} ± {E_stderr:.6f}")
    print(f"  T     = {T_mean:.6f} ± {T_stderr:.6f}")
    print(f"  V_trap= {Vt_mean:.6f} ± {Vt_stderr:.6f}")
    print(f"  V_int = {Vi_mean:.6f} ± {Vi_stderr:.6f}")
    print(f"  Virial: 2T = {virial_lhs:.6f},  2V_trap - V_int = {virial_rhs:.6f}")
    print(f"  Virial residual = {virial_residual:.6f}  (relative = {virial_relative:.4f})")
    if sampler == "is":
        print(f"  ESS (min/median/max) = {out['ess_min']:.1f} / {out['ess_median']:.1f} / {out['ess_max']:.1f}")
    else:
        print(f"  MH accept rate (mean) = {out['mh_accept_rate_mean']:.3f}")
    if sampler == "is" and report_is_stats and top1_vals:
        print(f"  IS mass concentration: top1≈{out['top1_mass_mean']:.4f}, top10≈{out['top10_mass_mean']:.4f}")
    if virial_relative < 0.02:
        print(f"  Virial: PASS (< 2%)")
    elif virial_relative < 0.05:
        print(f"  Virial: MARGINAL (2-5%)")
    else:
        print(f"  Virial: FAIL (> 5%)")

    return out


def main():
    parser = argparse.ArgumentParser(description="Post-training virial theorem check.")
    parser.add_argument("--result-dirs", nargs="+", required=True, help="Result directories with model.pt + config.yaml")
    parser.add_argument("--device", default="cuda:0", help="Device for evaluation")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of MCMC samples for energy evaluation")
    parser.add_argument("--n-cand-mult", type=int, default=8, help="Number of proposal candidates per kept sample")
    parser.add_argument("--sigma-fs", nargs="*", type=float, default=[], help="Override proposal sigma tiers (if omitted, read from config)")
    parser.add_argument("--langevin-steps", type=int, default=0, help="Langevin refinement steps before importance weighting")
    parser.add_argument("--langevin-step-size", type=float, default=0.01, help="Langevin step size when --langevin-steps > 0")
    parser.add_argument("--report-is-stats", action="store_true", help="Report importance-weight concentration diagnostics")
    parser.add_argument("--sampler", choices=["mh", "is"], default="mh", help="Sampling mode: mh for final evaluation, is for proposal diagnostics")
    parser.add_argument("--mh-burn-in", type=int, default=40, help="MH burn-in steps per batch")
    parser.add_argument("--mh-step-scale", type=float, default=0.25, help="MH random-walk step scale")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    results = []
    for d in args.result_dirs:
        p = Path(d)
        if not p.exists() and not p.is_absolute():
            p = Path("results") / d
        try:
            r = run_virial_check(
                p,
                args.device,
                args.n_samples,
                n_cand_mult=args.n_cand_mult,
                sigma_fs=args.sigma_fs,
                langevin_steps=args.langevin_steps,
                langevin_step_size=args.langevin_step_size,
                report_is_stats=args.report_is_stats,
                sampler=args.sampler,
                mh_burn_in=args.mh_burn_in,
                mh_step_scale=args.mh_step_scale,
            )
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  ERROR on {p.name}: {exc}")
            import traceback
            traceback.print_exc()

    if results:
        print("\n\n========== SUMMARY ==========")
        for r in results:
            status = "PASS" if r["virial_relative"] < 0.02 else ("MARGINAL" if r["virial_relative"] < 0.05 else "FAIL")
            if r["sampler"] == "is":
                sampler_metric = f"ESSmed={r['ess_median']:.1f}"
            else:
                sampler_metric = f"MHacc={r['mh_accept_rate_mean']:.3f}"
            print(
                f"  {r['dir']}: E={r['E_mean']:.6f}  T={r['T_mean']:.6f}  V_trap={r['V_trap_mean']:.6f}  "
                f"V_int={r['V_int_mean']:.6f}  virial={r['virial_relative']:.4f}  "
                f"{sampler_metric} [{status}]"
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
