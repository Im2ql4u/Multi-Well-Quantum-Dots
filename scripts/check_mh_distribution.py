#!/usr/bin/env python3
from __future__ import annotations

"""Check MH sampling fairness for a trained ground-state model.

Loads a completed run (config + model), draws MH samples, reports per-well
occupancy and coordinate statistics, and saves a 2D density plot.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import SystemConfig
from training.sampling import mcmc_resample
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


def _load_model(result_dir: Path, device: str) -> tuple[GroundStateWF, SystemConfig, dict[str, Any], torch.dtype]:
    cfg_path = result_dir / "config.yaml"
    model_path = result_dir / "model.pt"
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing config.yaml/model.pt in {result_dir}."
        )

    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    system = _build_system(raw_cfg["system"])
    arch_cfg = raw_cfg.get("architecture", {})
    train_cfg = raw_cfg.get("training", {})

    dtype_str = str(train_cfg.get("dtype", "float64"))
    torch_dtype = torch.float64 if dtype_str == "float64" else torch.float32

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
        use_well_features=bool(arch_cfg.get("use_well_features", False)),
        use_well_backflow=bool(arch_cfg.get("use_well_backflow", False)),
        use_backflow=bool(arch_cfg.get("use_backflow", True)),
    )

    state = torch.load(model_path, map_location=device, weights_only=True)
    load_info = model.load_state_dict(state, strict=False)
    allowed_missing = {"backflow.w_intra", "backflow.w_inter"}
    unexpected = list(load_info.unexpected_keys)
    missing = [k for k in load_info.missing_keys if k not in allowed_missing]
    if unexpected or missing:
        raise RuntimeError(
            "Checkpoint/model mismatch while loading state_dict. "
            f"Unexpected keys: {unexpected}; Missing keys: {missing}"
        )
    model.to(torch.device(device)).to(torch_dtype).eval()

    return model, system, train_cfg, torch_dtype


def _analyze_assignments(
    x_batch: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each particle to nearest well center.

    Returns:
      assignments: (B, N) well index per particle
      per_sample_counts: (B, W) occupancy counts
      dist2: (B, N, W) squared distances
    """
    dist2 = torch.sum((x_batch.unsqueeze(2) - centers.view(1, 1, *centers.shape)) ** 2, dim=-1)
    assignments = torch.argmin(dist2, dim=-1)
    n_wells = int(centers.shape[0])
    per_sample_counts = torch.stack(
        [(assignments == w).sum(dim=1) for w in range(n_wells)],
        dim=1,
    )
    return assignments, per_sample_counts, dist2


def _plot_density(points: torch.Tensor, centers: torch.Tensor, out_path: Path) -> None:
    arr = points.detach().cpu()
    c = centers.detach().cpu()

    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(arr[:, 0].numpy(), arr[:, 1].numpy(), bins=160, cmap="magma")
    for i in range(c.shape[0]):
        ax.scatter(c[i, 0].item(), c[i, 1].item(), marker="x", s=100, c="cyan")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MH particle density and well centers")
    fig.colorbar(h[3], ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MH distribution fairness for a trained run.")
    parser.add_argument("--result-dir", required=True, help="Result directory containing config.yaml/model.pt")
    parser.add_argument("--device", default="cuda:0", help="Sampling device")
    parser.add_argument("--n-samples", type=int, default=50000, help="Total MH samples")
    parser.add_argument("--mh-steps", type=int, default=100, help="MH steps per batch")
    parser.add_argument("--mh-step-scale", type=float, default=0.12, help="Initial MH proposal scale")
    parser.add_argument("--mh-decorrelation", type=int, default=2, help="MH decorrelation multiplier")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for MH collection")
    parser.add_argument("--warmup-batches", type=int, default=20, help="Warmup batches before scoring")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists() and not result_dir.is_absolute():
        result_dir = Path("results") / result_dir

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA is unavailable.")

    model, system, train_cfg, torch_dtype = _load_model(result_dir, str(device))

    centers = torch.tensor(
        [well.center for well in system.wells],
        device=device,
        dtype=torch_dtype,
    )
    if centers.shape[1] != system.dim:
        raise RuntimeError("Well center dimensionality mismatch.")

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    total_samples = int(args.n_samples)
    batch_size = max(64, int(args.batch_size))
    n_batches = (total_samples + batch_size - 1) // batch_size

    x_prev: torch.Tensor | None = None
    mh_scale = float(args.mh_step_scale)
    accept_rates: list[float] = []

    n_wells = len(system.wells)
    per_well_total = torch.zeros(n_wells, dtype=torch.float64)
    per_well_pos_sum = torch.zeros((n_wells, system.dim), dtype=torch.float64)
    per_well_pos_count = torch.zeros(n_wells, dtype=torch.float64)

    count_exact_expected = 0
    expected_counts = [int(w.n_particles) for w in system.wells]

    sampled_points: list[torch.Tensor] = []
    n_collected = 0

    print(f"result_dir={result_dir}")
    print(f"device={device} | n_samples={total_samples} | batch_size={batch_size}")
    print(
        "config_sampler="
        f"{train_cfg.get('sampler', 'unknown')} "
        f"config_mh_steps={train_cfg.get('mh_steps', 'n/a')} "
        f"config_mh_step_scale={train_cfg.get('mh_step_scale', 'n/a')}"
    )

    with torch.no_grad():
        for wi in range(int(args.warmup_batches)):
            x_prev, acc, mh_scale = mcmc_resample(
                psi_log_fn,
                x_prev,
                batch_size,
                n_elec=system.n_particles,
                dim=system.dim,
                omega=system.omega,
                device=device,
                dtype=torch_dtype,
                system=system,
                sigma_fs=(0.8, 1.3, 2.0),
                mh_steps=int(args.mh_steps),
                mh_step_scale=mh_scale,
                mh_decorrelation=int(args.mh_decorrelation),
            )
            if (wi + 1) % 5 == 0 or wi == int(args.warmup_batches) - 1:
                print(f"warmup {wi+1}/{args.warmup_batches}: acc={acc:.3f} mh_scale={mh_scale:.3f}")

        for bi in range(n_batches):
            need = min(batch_size, total_samples - n_collected)
            x_batch, acc, mh_scale = mcmc_resample(
                psi_log_fn,
                x_prev,
                need,
                n_elec=system.n_particles,
                dim=system.dim,
                omega=system.omega,
                device=device,
                dtype=torch_dtype,
                system=system,
                sigma_fs=(0.8, 1.3, 2.0),
                mh_steps=int(args.mh_steps),
                mh_step_scale=mh_scale,
                mh_decorrelation=int(args.mh_decorrelation),
            )
            x_prev = x_batch.detach()
            accept_rates.append(float(acc))

            # MH chain may keep a fixed batch shape from x_prev; score only what we still need.
            x_use = x_batch[:need]

            if not torch.isfinite(x_use).all():
                raise RuntimeError("Non-finite coordinates detected in MH samples.")

            assignments, per_sample_counts, _ = _analyze_assignments(x_use, centers)

            for w in range(n_wells):
                mask = assignments == w
                count_w = int(mask.sum().item())
                per_well_total[w] += float(count_w)
                if count_w > 0:
                    per_well_pos_sum[w] += x_use[mask].double().sum(dim=0).cpu()
                    per_well_pos_count[w] += float(count_w)

            expected_tensor = torch.tensor(expected_counts, device=per_sample_counts.device)
            count_exact_expected += int((per_sample_counts == expected_tensor).all(dim=1).sum().item())

            sampled_points.append(x_use.reshape(-1, system.dim).detach().cpu())
            n_collected += int(x_use.shape[0])

            if (bi + 1) % 5 == 0 or bi == n_batches - 1:
                mean_counts = (per_well_total / max(n_collected, 1)).tolist()
                print(
                    f"batch {bi+1}/{n_batches}: collected={n_collected} "
                    f"acc={acc:.3f} mean_counts={mean_counts}"
                )

    if n_collected != total_samples:
        raise RuntimeError(f"Collected {n_collected} samples, expected {total_samples}.")

    mean_counts_per_sample = per_well_total / float(total_samples)
    mean_positions = torch.zeros((n_wells, system.dim), dtype=torch.float64)
    for w in range(n_wells):
        if per_well_pos_count[w] > 0:
            mean_positions[w] = per_well_pos_sum[w] / per_well_pos_count[w]

    exact_expected_rate = count_exact_expected / float(total_samples)
    acc_mean = sum(accept_rates) / max(len(accept_rates), 1)

    all_points = torch.cat(sampled_points, dim=0)
    out_plot = result_dir / "mh_distribution_check.png"
    _plot_density(all_points, centers, out_plot)

    print("\n=== MH Distribution Summary ===")
    print(f"total_samples={total_samples}")
    print(f"mh_accept_rate_mean={acc_mean:.4f}")
    print(f"mh_step_scale_final={mh_scale:.4f}")
    print(f"expected_counts_per_sample={expected_counts}")
    print(f"mean_counts_per_sample={mean_counts_per_sample.tolist()}")
    print(f"exact_expected_occupancy_rate={exact_expected_rate:.4f}")
    for w in range(n_wells):
        print(f"well_{w}_mean_position={mean_positions[w].tolist()}")
    print(f"saved_density_plot={out_plot}")

    summary = {
        "total_samples": total_samples,
        "mh_accept_rate_mean": float(acc_mean),
        "mh_step_scale_final": float(mh_scale),
        "expected_counts_per_sample": expected_counts,
        "mean_counts_per_sample": [float(v) for v in mean_counts_per_sample.tolist()],
        "exact_expected_occupancy_rate": float(exact_expected_rate),
        "well_mean_positions": [[float(x) for x in row] for row in mean_positions.tolist()],
        "saved_density_plot": str(out_plot),
    }
    out_json = result_dir / "mh_distribution_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved_summary_json={out_json}")


if __name__ == "__main__":
    main()
