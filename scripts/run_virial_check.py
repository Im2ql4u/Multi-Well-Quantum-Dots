#!/usr/bin/env python3
from __future__ import annotations

"""Post-training virial diagnostics for completed ground-state runs.

This script loads result directories containing model + config, resamples points with
MH or IS using the current training APIs, computes local-energy components, and reports
virial residuals for harmonic confinement + Coulomb interactions.

Virial condition used:
    2<T> = 2<V_trap> - <V_int>
"""

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import SystemConfig
from observables.validation import compute_virial_metrics
from training.collocation import _laplacian_over_psi_fd, _potential_energy
from training.sampling import importance_resample, mcmc_resample
from wavefunction import GroundStateWF, resolve_reference_energy, setup_closed_shell_system


def _build_system(system_cfg: dict[str, Any]) -> SystemConfig:
    kind = system_cfg.get("type", "single_dot")
    coulomb = bool(system_cfg.get("coulomb", True))

    if kind == "single_dot":
        sys_cfg = SystemConfig.single_dot(
            N=int(system_cfg["n_particles"]),
            omega=float(system_cfg["omega"]),
            dim=int(system_cfg.get("dim", 2)),
        )
    elif kind == "double_dot":
        sys_cfg = SystemConfig.double_dot(
            N_L=int(system_cfg["n_left"]),
            N_R=int(system_cfg["n_right"]),
            sep=float(system_cfg["separation"]),
            omega=float(system_cfg["omega"]),
            dim=int(system_cfg.get("dim", 2)),
        )
    else:
        raise ValueError(f"Unsupported system type '{kind}'.")

    if not coulomb:
        sys_cfg = replace(sys_cfg, coulomb=False)
    return sys_cfg


def _compute_local_energy_components(
    psi_log_fn,
    x: torch.Tensor,
    system: SystemConfig,
    *,
    fd_h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute E, T, V_trap, V_int for each sample.

    Kinetic term is matched to training's finite-difference local energy path to avoid
    decomposition mismatch:
        E = -0.5 * (∇²ψ / ψ) + V_total
    with T := E - V_total.
    """
    lap_over_psi = _laplacian_over_psi_fd(psi_log_fn, x, float(fd_h))

    with torch.no_grad():
        # Match training Hamiltonian exactly for evaluator parity.
        v_trap = _potential_energy(x, omega=system.omega, system=replace(system, coulomb=False))
        v_tot = _potential_energy(x, omega=system.omega, system=system)
        v_int = v_tot - v_trap
        e_loc = -0.5 * lap_over_psi + v_tot
        t_loc = e_loc - v_tot

    if not torch.isfinite(e_loc).all():
        raise RuntimeError("Non-finite E_L encountered during virial check.")
    if not torch.isfinite(t_loc).all():
        raise RuntimeError("Non-finite kinetic term encountered during virial check.")

    return (e_loc, t_loc, v_trap, v_int)


def _stats(sum_1: float, sum_2: float, n: int) -> tuple[float, float, float]:
    mean = sum_1 / n
    var = max(sum_2 / n - mean * mean, 0.0)
    std = math.sqrt(var)
    stderr = std / math.sqrt(n)
    return (mean, std, stderr)


def run_virial_check(
    result_dir: Path,
    *,
    device: str,
    n_samples: int,
    sampler: str,
    n_cand_mult: int,
    sigma_fs: tuple[float, ...],
    mh_steps: int,
    mh_step_scale: float,
    mh_decorrelation: int,
    mh_warmup_batches: int,
    fd_h: float | None,
) -> dict[str, Any]:
    cfg_path = result_dir / "config.yaml"
    model_path = result_dir / "model.pt"
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing model/config in {result_dir}. Need config.yaml and model.pt."
        )

    raw_cfg = yaml.safe_load(cfg_path.read_text())
    system = _build_system(raw_cfg["system"])
    arch_cfg = raw_cfg.get("architecture", {})
    train_cfg = raw_cfg.get("training", {})

    dtype_str = str(train_cfg.get("dtype", "float64"))
    torch_dtype = torch.float64 if dtype_str == "float64" else torch.float32
    dev = torch.device(device)

    allow_missing_dmc = bool(raw_cfg.get("allow_missing_dmc", True))
    resolved_e_ref = resolve_reference_energy(
        system,
        raw_cfg.get("E_ref", "auto"),
        allow_missing_dmc=allow_missing_dmc,
    )

    c_occ, spin, params = setup_closed_shell_system(
        system,
        device=device,
        dtype=torch_dtype,
        E_ref=resolved_e_ref,
        allow_missing_dmc=allow_missing_dmc,
    )

    model = GroundStateWF(
        system,
        c_occ,
        spin,
        params,
        arch_type=str(arch_cfg.get("arch_type", "pinn")),
        pinn_hidden=int(arch_cfg.get("pinn_hidden", 64)),
        pinn_layers=int(arch_cfg.get("pinn_layers", 2)),
        bf_hidden=int(arch_cfg.get("bf_hidden", 32)),
        bf_layers=int(arch_cfg.get("bf_layers", 2)),
        use_backflow=bool(arch_cfg.get("use_backflow", True)),
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(dev).to(torch_dtype).eval()

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    if fd_h is None:
        fd_h_eval = float(train_cfg.get("fd_h", 0.01))
    else:
        fd_h_eval = float(fd_h)

    sigma_use = sigma_fs or tuple(float(s) for s in train_cfg.get("sigma_fs", (0.8, 1.3, 2.0)))

    batch_size = min(2048, max(256, n_samples))
    n_batches = (n_samples + batch_size - 1) // batch_size

    sum_e = sum_e2 = 0.0
    sum_t = sum_t2 = 0.0
    sum_vt = sum_vt2 = 0.0
    sum_vi = sum_vi2 = 0.0
    total = 0

    x_prev: torch.Tensor | None = None
    mh_scale = float(mh_step_scale)
    mh_accept_rates: list[float] = []
    ess_vals: list[float] = []

    print(f"  Running {n_batches} batches (~{batch_size} each), sampler={sampler} ...")

    if sampler == "mh" and mh_warmup_batches > 0:
        print(f"  MH warmup: {mh_warmup_batches} batches (not scored) ...")
        warm_bsz = batch_size
        for wi in range(mh_warmup_batches):
            x_prev, accept_rate, mh_scale = mcmc_resample(
                psi_log_fn,
                x_prev,
                warm_bsz,
                n_elec=system.n_particles,
                dim=system.dim,
                omega=system.omega,
                device=dev,
                dtype=torch_dtype,
                system=system,
                sigma_fs=sigma_use,
                mh_steps=mh_steps,
                mh_step_scale=mh_scale,
                mh_decorrelation=mh_decorrelation,
            )
            x_prev = x_prev.detach()
            if (wi + 1) % 5 == 0 or wi == mh_warmup_batches - 1:
                print(
                    f"    warmup {wi+1}/{mh_warmup_batches}: "
                    f"acc={accept_rate:.3f} mh_scale={mh_scale:.3f}"
                )

    for bi in range(n_batches):
        bsz_req = min(batch_size, n_samples - total)

        if sampler == "mh":
            x_batch, accept_rate, mh_scale = mcmc_resample(
                psi_log_fn,
                x_prev,
                bsz_req,
                n_elec=system.n_particles,
                dim=system.dim,
                omega=system.omega,
                device=dev,
                dtype=torch_dtype,
                system=system,
                sigma_fs=sigma_use,
                mh_steps=mh_steps,
                mh_step_scale=mh_scale,
                mh_decorrelation=mh_decorrelation,
            )
            x_prev = x_batch.detach()
            mh_accept_rates.append(float(accept_rate))
        else:
            x_batch, ess = importance_resample(
                psi_log_fn,
                n_keep=bsz_req,
                n_elec=system.n_particles,
                dim=system.dim,
                omega=system.omega,
                device=dev,
                dtype=torch_dtype,
                n_cand_mult=n_cand_mult,
                sigma_fs=sigma_use,
                min_pair_cutoff=0.0,
                weight_temp=1.0,
                logw_clip_q=0.0,
                langevin_steps=0,
                langevin_step_size=0.01,
                system=system,
                return_weights=False,
            )
            ess_vals.append(float(ess))

        e_loc, t_loc, v_trap, v_int = _compute_local_energy_components(
            psi_log_fn,
            x_batch,
            system,
            fd_h=fd_h_eval,
        )

        # mcmc_resample reuses chain shape from x_prev. Use actual sample count.
        bsz = int(x_batch.shape[0])

        sum_e += float(e_loc.sum().item())
        sum_e2 += float((e_loc * e_loc).sum().item())
        sum_t += float(t_loc.sum().item())
        sum_t2 += float((t_loc * t_loc).sum().item())
        sum_vt += float(v_trap.sum().item())
        sum_vt2 += float((v_trap * v_trap).sum().item())
        sum_vi += float(v_int.sum().item())
        sum_vi2 += float((v_int * v_int).sum().item())
        total += bsz

        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            e_so_far = sum_e / total
            if sampler == "mh":
                print(
                    f"    batch {bi+1}/{n_batches}: E≈{e_so_far:.6f}, "
                    f"acc={mh_accept_rates[-1]:.3f} (n={total})"
                )
            else:
                print(
                    f"    batch {bi+1}/{n_batches}: E≈{e_so_far:.6f}, "
                    f"ESS={ess_vals[-1]:.1f} (n={total})"
                )

    e_mean, e_std, e_stderr = _stats(sum_e, sum_e2, total)
    t_mean, _, t_stderr = _stats(sum_t, sum_t2, total)
    vt_mean, _, vt_stderr = _stats(sum_vt, sum_vt2, total)
    vi_mean, _, vi_stderr = _stats(sum_vi, sum_vi2, total)

    vir_lhs, vir_rhs, vir_res, vir_rel = compute_virial_metrics(
        T_mean=t_mean,
        V_trap_mean=vt_mean,
        V_int_mean=vi_mean,
        E_mean=e_mean,
    )

    out: dict[str, Any] = {
        "dir": result_dir.name,
        "sampler": sampler,
        "n_samples": int(total),
        "fd_h_eval": fd_h_eval,
        "E_mean": e_mean,
        "E_std": e_std,
        "E_stderr": e_stderr,
        "T_mean": t_mean,
        "T_stderr": t_stderr,
        "V_trap_mean": vt_mean,
        "V_trap_stderr": vt_stderr,
        "V_int_mean": vi_mean,
        "V_int_stderr": vi_stderr,
        "virial_lhs": vir_lhs,
        "virial_rhs": vir_rhs,
        "virial_residual": vir_res,
        "virial_relative": vir_rel,
        "mh_accept_rate_mean": float(sum(mh_accept_rates) / len(mh_accept_rates)) if mh_accept_rates else None,
        "mh_warmup_batches": int(mh_warmup_batches),
        "ess_median": float(torch.tensor(sorted(ess_vals)).median().item()) if ess_vals else None,
    }

    print(f"\n  === {result_dir.name} ===")
    print(f"  E      = {e_mean:.6f} +- {e_stderr:.6f}")
    print(f"  T      = {t_mean:.6f} +- {t_stderr:.6f}")
    print(f"  V_trap = {vt_mean:.6f} +- {vt_stderr:.6f}")
    print(f"  V_int  = {vi_mean:.6f} +- {vi_stderr:.6f}")
    print(f"  Virial: 2T={vir_lhs:.6f}, 2Vt-Vi={vir_rhs:.6f}")
    print(f"  Virial residual={vir_res:.6f}, relative={100.0*vir_rel:.2f}%")
    if sampler == "mh":
        print(f"  MH accept mean={out['mh_accept_rate_mean']:.3f}")
    else:
        print(f"  IS ESS median={out['ess_median']:.1f}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Virial diagnostics for trained runs.")
    parser.add_argument("--result-dirs", nargs="+", required=True, help="Result directories containing config.yaml/model.pt")
    parser.add_argument("--device", default="cuda:0", help="Evaluation device")
    parser.add_argument("--n-samples", type=int, default=20000, help="Total samples per result")
    parser.add_argument("--sampler", choices=("mh", "is"), default="mh", help="Sampler for evaluation")
    parser.add_argument("--n-cand-mult", type=int, default=8, help="IS candidate multiplier")
    parser.add_argument("--sigma-fs", nargs="*", type=float, default=[], help="Override sigma tiers")
    parser.add_argument("--mh-steps", type=int, default=10, help="MH steps per epoch")
    parser.add_argument("--mh-step-scale", type=float, default=0.25, help="Initial MH step scale")
    parser.add_argument("--mh-decorrelation", type=int, default=1, help="MH decorrelation multiplier")
    parser.add_argument("--mh-warmup-batches", type=int, default=20, help="Number of MH warmup batches before scoring")
    parser.add_argument("--fd-h", type=float, default=None, help="Override FD step for local energy")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    sigma_fs = tuple(float(s) for s in args.sigma_fs)

    results: list[dict[str, Any]] = []
    for rd in args.result_dirs:
        path = Path(rd)
        if not path.exists() and not path.is_absolute():
            path = Path("results") / rd
        try:
            res = run_virial_check(
                path,
                device=args.device,
                n_samples=int(args.n_samples),
                sampler=str(args.sampler),
                n_cand_mult=int(args.n_cand_mult),
                sigma_fs=sigma_fs,
                mh_steps=int(args.mh_steps),
                mh_step_scale=float(args.mh_step_scale),
                mh_decorrelation=int(args.mh_decorrelation),
                mh_warmup_batches=int(args.mh_warmup_batches),
                fd_h=args.fd_h,
            )
            results.append(res)
        except Exception as exc:
            print(f"  ERROR on {path.name}: {exc}")

    if results:
        print("\n========== SUMMARY ==========")
        for r in results:
            status = "PASS" if r["virial_relative"] < 0.02 else ("MARGINAL" if r["virial_relative"] < 0.05 else "FAIL")
            aux = f"acc={r['mh_accept_rate_mean']:.3f}" if r["sampler"] == "mh" else f"ESSmed={r['ess_median']:.1f}"
            print(
                f"  {r['dir']}: E={r['E_mean']:.6f}, T={r['T_mean']:.6f}, "
                f"Vt={r['V_trap_mean']:.6f}, Vi={r['V_int_mean']:.6f}, "
                f"virial={100.0*r['virial_relative']:.2f}% [{status}] {aux}"
            )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
