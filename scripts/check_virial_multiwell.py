#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import SystemConfig
from observables.validation import compute_virial_metrics
from potential import compute_potential
from training.collocation import _laplacian_over_psi_fd, _potential_energy
from training.sampling import mcmc_resample
from wavefunction import GroundStateWF, resolve_reference_energy, setup_closed_shell_system
from functions.Slater_Determinant import slater_determinant_closed_shell

LOG = logging.getLogger("check_virial_multiwell")


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class LegacyCTNNJastrow(nn.Module):
    def __init__(
        self,
        n_particles: int,
        dim: int,
        node_hidden: int,
        edge_hidden: int,
        n_mp_steps: int,
        f_in_dim: int,
    ) -> None:
        super().__init__()
        self.n_particles = int(n_particles)
        self.dim = int(dim)
        self.node_embed = nn.Linear(dim + 1, node_hidden)
        self.edge_embed = _make_mlp(dim + 2, edge_hidden, edge_hidden)
        self.rho_v_to_e = nn.ModuleList(
            [nn.Linear(node_hidden, edge_hidden, bias=False) for _ in range(n_mp_steps)]
        )
        self.edge_updates = nn.ModuleList(
            [_make_mlp(3 * edge_hidden, edge_hidden, edge_hidden) for _ in range(n_mp_steps)]
        )
        self.rho_e_to_v = nn.ModuleList(
            [nn.Linear(edge_hidden, node_hidden, bias=False) for _ in range(n_mp_steps)]
        )
        self.node_updates = nn.ModuleList(
            [_make_mlp(2 * node_hidden, node_hidden, node_hidden) for _ in range(n_mp_steps)]
        )
        self.f_head = nn.Sequential(
            nn.Linear(f_in_dim, node_hidden),
            nn.GELU(),
            nn.Linear(node_hidden, node_hidden),
            nn.GELU(),
            nn.Linear(node_hidden, 1),
        )

    def forward(self, x: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        bsz, n_particles, dim = x.shape
        spin_feat = spin.view(bsz, n_particles, 1).to(dtype=x.dtype)
        node_in = torch.cat([x, spin_feat], dim=-1)
        h_v = self.node_embed(node_in)

        src = []
        dst = []
        edge_feats = []
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j:
                    continue
                dr = x[:, j, :] - x[:, i, :]
                r2 = torch.sum(dr * dr, dim=-1, keepdim=True)
                rr = torch.sqrt(r2 + 1e-12)
                edge_feats.append(torch.cat([dr, rr, r2], dim=-1))
                src.append(i)
                dst.append(j)
        e_count = len(src)
        h_e = self.edge_embed(torch.stack(edge_feats, dim=1))

        for step in range(len(self.rho_v_to_e)):
            v_to_e = self.rho_v_to_e[step](h_v)
            upd = []
            for eidx in range(e_count):
                i = src[eidx]
                j = dst[eidx]
                upd.append(torch.cat([h_e[:, eidx, :], v_to_e[:, i, :], v_to_e[:, j, :]], dim=-1))
            h_e = self.edge_updates[step](torch.stack(upd, dim=1))

            msg = self.rho_e_to_v[step](h_e)
            agg = torch.zeros_like(h_v)
            cnt = torch.zeros(bsz, n_particles, 1, device=x.device, dtype=x.dtype)
            for eidx in range(e_count):
                j = dst[eidx]
                agg[:, j, :] = agg[:, j, :] + msg[:, eidx, :]
                cnt[:, j, :] = cnt[:, j, :] + 1.0
            agg = agg / cnt.clamp_min(1.0)
            h_v = self.node_updates[step](torch.cat([h_v, agg], dim=-1))

        hv_flat = h_v.reshape(bsz, -1)
        he_flat = h_e.reshape(bsz, -1)
        r2_all = torch.sum(x * x, dim=(1, 2), keepdim=False).unsqueeze(-1)
        r_pair = torch.zeros(bsz, 1, device=x.device, dtype=x.dtype)
        if n_particles > 1:
            diff = x[:, 0, :] - x[:, 1, :]
            r_pair = torch.sqrt(torch.sum(diff * diff, dim=-1, keepdim=True) + 1e-12)
        f_in = torch.cat([hv_flat, he_flat, r_pair, r2_all], dim=-1)
        in_dim = int(self.f_head[0].weight.shape[1])
        if f_in.shape[1] > in_dim:
            f_in = f_in[:, :in_dim]
        elif f_in.shape[1] < in_dim:
            pad = torch.zeros(bsz, in_dim - f_in.shape[1], device=x.device, dtype=x.dtype)
            f_in = torch.cat([f_in, pad], dim=-1)
        return self.f_head(f_in)


class LegacyCTNNBackflow(nn.Module):
    def __init__(self, dim: int, node_hidden: int, edge_hidden: int) -> None:
        super().__init__()
        self.node_embed = nn.Linear(dim + 1, node_hidden)
        self.edge_embed = _make_mlp(dim + 2, edge_hidden, edge_hidden)
        self.rho_v_to_e = nn.Linear(node_hidden, edge_hidden, bias=False)
        self.rho_e_to_v = nn.Linear(edge_hidden, node_hidden, bias=False)
        self.edge_update = _make_mlp(3 * edge_hidden, edge_hidden, edge_hidden)
        self.node_update = _make_mlp(2 * node_hidden, node_hidden, node_hidden)
        self.dx_head = nn.Linear(node_hidden, dim)
        self.bf_scale_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        bsz, n_particles, _ = x.shape
        spin_feat = spin.view(bsz, n_particles, 1).to(dtype=x.dtype)
        node_in = torch.cat([x, spin_feat], dim=-1)
        h_v = self.node_embed(node_in)

        src = []
        dst = []
        edge_feats = []
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j:
                    continue
                dr = x[:, j, :] - x[:, i, :]
                r2 = torch.sum(dr * dr, dim=-1, keepdim=True)
                rr = torch.sqrt(r2 + 1e-12)
                edge_feats.append(torch.cat([dr, rr, r2], dim=-1))
                src.append(i)
                dst.append(j)
        e_count = len(src)
        h_e = self.edge_embed(torch.stack(edge_feats, dim=1))

        v_to_e = self.rho_v_to_e(h_v)
        upd = []
        for eidx in range(e_count):
            i = src[eidx]
            j = dst[eidx]
            upd.append(torch.cat([h_e[:, eidx, :], v_to_e[:, i, :], v_to_e[:, j, :]], dim=-1))
        h_e = self.edge_update(torch.stack(upd, dim=1))

        msg = self.rho_e_to_v(h_e)
        agg = torch.zeros_like(h_v)
        cnt = torch.zeros(bsz, n_particles, 1, device=x.device, dtype=x.dtype)
        for eidx in range(e_count):
            j = dst[eidx]
            agg[:, j, :] = agg[:, j, :] + msg[:, eidx, :]
            cnt[:, j, :] = cnt[:, j, :] + 1.0
        agg = agg / cnt.clamp_min(1.0)
        h_v = self.node_update(torch.cat([h_v, agg], dim=-1))

        dx = self.dx_head(h_v)
        scale = torch.nn.functional.softplus(self.bf_scale_raw)
        return torch.tanh(dx) * scale


def _infer_nx_ny(n_basis: int) -> tuple[int, int]:
    best = (n_basis, 1)
    best_gap = abs(n_basis - 1)
    for ny in range(1, int(math.sqrt(n_basis)) + 1):
        if n_basis % ny == 0:
            nx = n_basis // ny
            gap = abs(nx - ny)
            if gap < best_gap:
                best = (nx, ny)
                best_gap = gap
    return best


def _build_legacy_psi_from_state(
    state: dict[str, torch.Tensor],
    *,
    system: SystemConfig,
    dev: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, torch.Tensor]:
    f_state = {k[len("f_net.") :]: v for k, v in state.items() if k.startswith("f_net.")}
    bf_state = {k[len("bf_net.") :]: v for k, v in state.items() if k.startswith("bf_net.")}
    if not f_state:
        raise RuntimeError("Legacy checkpoint missing f_net.* keys.")

    spin_ckpt = state.get("spin")
    if isinstance(spin_ckpt, torch.Tensor) and spin_ckpt.numel() == system.n_particles:
        spin_vec = spin_ckpt.to(device=dev, dtype=torch.long)
    else:
        n_up = system.n_particles // 2
        spin_vec = torch.cat(
            [
                torch.zeros(n_up, device=dev, dtype=torch.long),
                torch.ones(system.n_particles - n_up, device=dev, dtype=torch.long),
            ]
        )

    node_hidden = int(f_state["node_embed.weight"].shape[0])
    edge_hidden = int(f_state["edge_embed.0.weight"].shape[0])
    n_mp_steps = len([k for k in f_state.keys() if k.startswith("rho_v_to_e.") and k.endswith(".weight")])
    f_in_dim = int(f_state["f_head.0.weight"].shape[1])
    f_net = LegacyCTNNJastrow(
        system.n_particles,
        system.dim,
        node_hidden,
        edge_hidden,
        n_mp_steps,
        f_in_dim,
    )
    f_net.load_state_dict(f_state, strict=True)
    f_net.to(device=dev, dtype=dtype).eval()

    backflow: LegacyCTNNBackflow | None = None
    if bf_state:
        bf_node_hidden = int(bf_state["node_embed.weight"].shape[0])
        bf_edge_hidden = int(bf_state["edge_embed.0.weight"].shape[0])
        backflow = LegacyCTNNBackflow(system.dim, bf_node_hidden, bf_edge_hidden)
        backflow.load_state_dict(bf_state, strict=True)
        backflow.to(device=dev, dtype=dtype).eval()

    c_occ = state.get("C_occ")
    if not isinstance(c_occ, torch.Tensor):
        raise RuntimeError("Legacy checkpoint missing C_occ tensor.")
    c_occ = c_occ.to(device=dev, dtype=dtype)
    nx, ny = _infer_nx_ny(int(c_occ.shape[0]))
    sd_params = {"omega": float(system.omega), "nx": int(nx), "ny": int(ny), "basis": "cart"}

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        spin_bn = spin_vec.view(1, -1).expand(bsz, -1)
        x_eff = x
        if backflow is not None:
            x_eff = x + backflow(x, spin=spin_bn)
        _, logabs = slater_determinant_closed_shell(
            x_config=x_eff,
            C_occ=c_occ,
            params=sd_params,
            spin=spin_bn,
            normalize=True,
        )
        jastrow = f_net(x, spin=spin_bn).squeeze(-1)
        # Legacy checkpoints can hit singular SD regions under long MH chains; keep values finite.
        logabs = torch.nan_to_num(logabs, nan=-60.0, posinf=60.0, neginf=-60.0)
        jastrow = torch.nan_to_num(jastrow, nan=0.0, posinf=60.0, neginf=-60.0)
        out = torch.clamp(logabs + jastrow, min=-120.0, max=120.0)
        return out

    return psi_log_fn, spin_vec


def _build_legacy_mapped_groundstate_psi(
    state: dict[str, torch.Tensor],
    *,
    system: SystemConfig,
    arch_cfg: dict[str, Any],
    raw_cfg: dict[str, Any],
    dev: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, torch.Tensor]:
    allow_missing_dmc = bool(raw_cfg.get("allow_missing_dmc", True))
    resolved_e_ref = resolve_reference_energy(
        system,
        raw_cfg.get("E_ref", "auto"),
        allow_missing_dmc=allow_missing_dmc,
    )
    c_occ, spin, params = setup_closed_shell_system(
        system,
        device=str(dev),
        dtype=dtype,
        E_ref=resolved_e_ref,
        allow_missing_dmc=allow_missing_dmc,
    )

    ckpt_c_occ = state.get("C_occ")
    if isinstance(ckpt_c_occ, torch.Tensor):
        ckpt_c_occ = ckpt_c_occ.to(device=dev, dtype=dtype)
        nx, ny = _infer_nx_ny(int(ckpt_c_occ.shape[0]))
        params = dict(params)
        params["nx"] = int(nx)
        params["ny"] = int(ny)
        c_occ = ckpt_c_occ

    spin_ckpt = state.get("spin")
    if isinstance(spin_ckpt, torch.Tensor) and spin_ckpt.numel() == system.n_particles:
        spin = spin_ckpt.to(device=dev, dtype=torch.long)

    legacy_bf_is_ctnn = any(k.startswith("bf_net.node_embed") for k in state.keys())
    arch_type_load = str(arch_cfg.get("arch_type", "pinn"))
    if legacy_bf_is_ctnn and arch_type_load == "pinn":
        arch_type_load = "ctnn"

    model = GroundStateWF(
        system,
        c_occ,
        spin,
        params,
        arch_type=arch_type_load,
        pinn_hidden=int(arch_cfg.get("pinn_hidden", 64)),
        pinn_layers=int(arch_cfg.get("pinn_layers", 2)),
        bf_hidden=int(arch_cfg.get("bf_hidden", 32)),
        bf_layers=int(arch_cfg.get("bf_layers", 2)),
        use_well_features=bool(arch_cfg.get("use_well_features", False)),
        use_well_backflow=bool(arch_cfg.get("use_well_backflow", False)),
        use_backflow=bool(arch_cfg.get("use_backflow", True)),
    )

    remap: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("f_net."):
            remap[f"pinn.{k[len('f_net.'): ]}"] = v
        elif k.startswith("bf_net."):
            remap[f"backflow.{k[len('bf_net.'): ]}"] = v

    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in remap.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v

    load_info = model.load_state_dict(filtered, strict=False)
    allowed_missing = {"backflow.w_intra", "backflow.w_inter"}
    missing = [k for k in load_info.missing_keys if k not in allowed_missing]
    if load_info.unexpected_keys:
        raise RuntimeError(f"Legacy mapped load has unexpected keys: {load_info.unexpected_keys}")
    # Only fail if core PINN/backflow keys are missing after remap.
    core_missing = [k for k in missing if k.startswith("pinn.") or k.startswith("backflow.")]
    if core_missing:
        raise RuntimeError(f"Legacy mapped load missing core keys: {core_missing}")

    model.to(dev).to(dtype).eval()

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if not torch.isfinite(out).all():
            raise RuntimeError("Non-finite logpsi in legacy mapped GroundStateWF path.")
        return out

    return psi_log_fn, model.spin_template.to(dev)


def _build_system(system_cfg: dict[str, Any]) -> SystemConfig:
    kind = system_cfg.get("type", "single_dot")
    coulomb = bool(system_cfg.get("coulomb", True))

    if kind == "single_dot":
        system = SystemConfig.single_dot(
            N=int(system_cfg["n_particles"]),
            omega=float(system_cfg["omega"]),
            dim=int(system_cfg.get("dim", 2)),
        )
    elif kind == "double_dot":
        system = SystemConfig.double_dot(
            N_L=int(system_cfg["n_left"]),
            N_R=int(system_cfg["n_right"]),
            sep=float(system_cfg["separation"]),
            omega=float(system_cfg["omega"]),
            dim=int(system_cfg.get("dim", 2)),
        )
    elif kind == "custom":
        wells = []
        for w in system_cfg["wells"]:
            wells.append(
                {
                    "center": tuple(float(c) for c in w["center"]),
                    "omega": float(w["omega"]),
                    "n_particles": int(w["n_particles"]),
                }
            )
        system = SystemConfig.custom(wells=wells, dim=int(system_cfg.get("dim", 2)))
    else:
        raise ValueError(f"Unsupported system type '{kind}'.")

    if not coulomb:
        system = replace(system, coulomb=False)
    return system


def _local_energy_components_fd(
    psi_log_fn,
    x: torch.Tensor,
    *,
    system: SystemConfig,
    fd_h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lap_over_psi = _laplacian_over_psi_fd(psi_log_fn, x, float(fd_h))
    with torch.no_grad():
        v_trap = _potential_energy(x, omega=system.omega, system=replace(system, coulomb=False))
        v_tot = _potential_energy(x, omega=system.omega, system=system)
        v_int = v_tot - v_trap
        e_loc = -0.5 * lap_over_psi + v_tot
        t_loc = e_loc - v_tot
    return (e_loc, t_loc, v_trap, v_int)


def _stats(sum_1: float, sum_2: float, n: int) -> tuple[float, float, float]:
    mean = sum_1 / n
    var = max(sum_2 / n - mean * mean, 0.0)
    std = math.sqrt(var)
    stderr = std / math.sqrt(n)
    return (mean, std, stderr)


def _r_dot_grad_v_total(x: torch.Tensor, *, system: SystemConfig, spin: torch.Tensor) -> torch.Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    v = compute_potential(x_req, system=system, spin=spin)
    grad_v = torch.autograd.grad(v.sum(), x_req, create_graph=False, retain_graph=False)[0]
    if not torch.isfinite(grad_v).all():
        raise RuntimeError("Non-finite potential gradient in generalized virial term.")
    return torch.sum(x_req * grad_v, dim=(1, 2)).detach()


def run_check(
    result_dir: Path,
    *,
    device: str,
    n_samples: int,
    mh_steps: int,
    mh_step_scale: float,
    mh_decorrelation: int,
    mh_warmup_batches: int,
    fd_h: float | None,
) -> dict[str, Any]:
    cfg_path = result_dir / "config.yaml"
    model_path = result_dir / "model.pt"
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing config/model in {result_dir}.")

    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    system = _build_system(raw_cfg["system"])
    arch_cfg = raw_cfg.get("architecture", {})
    train_cfg = raw_cfg.get("training", {})

    dtype_str = str(train_cfg.get("dtype", "float64"))
    torch_dtype = torch.float64 if dtype_str == "float64" else torch.float32
    dev = torch.device(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    if any(k.startswith("f_net.") for k in state.keys()):
        if any(k.startswith("f_net.node_embed.") for k in state.keys()):
            LOG.info("Detected legacy CTNN-jastrow checkpoint format. Using explicit legacy loader.")
            psi_log_fn, spin_dev = _build_legacy_psi_from_state(state, system=system, dev=dev, dtype=torch_dtype)
        else:
            LOG.info("Detected legacy PINN checkpoint format. Using remapped GroundStateWF loader.")
            psi_log_fn, spin_dev = _build_legacy_mapped_groundstate_psi(
                state,
                system=system,
                arch_cfg=arch_cfg,
                raw_cfg=raw_cfg,
                dev=dev,
                dtype=torch_dtype,
            )
    else:
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
        model_state = model.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        dropped_shape: list[str] = []
        for key, value in state.items():
            if key not in model_state:
                filtered_state[key] = value
                continue
            if model_state[key].shape != value.shape:
                dropped_shape.append(key)
                continue
            filtered_state[key] = value

        if dropped_shape:
            LOG.warning("Dropping %d incompatible checkpoint keys: %s", len(dropped_shape), dropped_shape)

        load_info = model.load_state_dict(filtered_state, strict=False)
        allowed_missing = {"backflow.w_intra", "backflow.w_inter"}
        unexpected = list(load_info.unexpected_keys)
        missing = [k for k in load_info.missing_keys if k not in allowed_missing]
        if unexpected or missing:
            raise RuntimeError(
                "Checkpoint/model mismatch while loading state_dict. "
                f"Unexpected keys: {unexpected}; Missing keys: {missing}"
            )

        model.to(dev).to(torch_dtype).eval()

        def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
            return model(x)

        spin_dev = spin.to(dev)

    fd_h_eval = float(fd_h if fd_h is not None else train_cfg.get("fd_h", 0.01))
    sigma_use = tuple(float(s) for s in train_cfg.get("sigma_fs", (0.8, 1.3, 2.0)))

    batch_size = min(2048, max(256, n_samples))
    n_batches = (n_samples + batch_size - 1) // batch_size

    sum_e = sum_e2 = 0.0
    sum_t = sum_t2 = 0.0
    sum_vt = sum_vt2 = 0.0
    sum_vi = sum_vi2 = 0.0
    sum_rdv = sum_rdv2 = 0.0
    total = 0

    x_prev: torch.Tensor | None = None
    mh_scale = float(mh_step_scale)
    

    LOG.info("Running %d batches, sampler=mh", n_batches)

    if mh_warmup_batches > 0:
        LOG.info("MH warmup: %d batches", mh_warmup_batches)
        for wi in range(mh_warmup_batches):
            x_prev, accept_rate, mh_scale = mcmc_resample(
                psi_log_fn,
                x_prev,
                batch_size,
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
                LOG.info("  warmup %d/%d: acc=%.3f mh_scale=%.3f", wi + 1, mh_warmup_batches, accept_rate, mh_scale)

    for bi in range(n_batches):
        bsz_req = min(batch_size, n_samples - total)
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

        e_loc, t_loc, v_trap, v_int = _local_energy_components_fd(
            psi_log_fn,
            x_batch,
            system=system,
            fd_h=fd_h_eval,
        )
        r_dot_grad = _r_dot_grad_v_total(x_batch, system=system, spin=spin_dev)

        if not torch.isfinite(e_loc).all() or not torch.isfinite(t_loc).all() or not torch.isfinite(r_dot_grad).all():
            raise RuntimeError("Non-finite values encountered while computing virial metrics.")

        bsz = int(x_batch.shape[0])
        sum_e += float(e_loc.sum().item())
        sum_e2 += float((e_loc * e_loc).sum().item())
        sum_t += float(t_loc.sum().item())
        sum_t2 += float((t_loc * t_loc).sum().item())
        sum_vt += float(v_trap.sum().item())
        sum_vt2 += float((v_trap * v_trap).sum().item())
        sum_vi += float(v_int.sum().item())
        sum_vi2 += float((v_int * v_int).sum().item())
        sum_rdv += float(r_dot_grad.sum().item())
        sum_rdv2 += float((r_dot_grad * r_dot_grad).sum().item())
        total += bsz

        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            LOG.info(
                "  batch %d/%d: E≈%.6f acc=%.3f n=%d",
                bi + 1,
                n_batches,
                sum_e / total,
                accept_rate,
                total,
            )

    e_mean, e_std, e_stderr = _stats(sum_e, sum_e2, total)
    t_mean, _, t_stderr = _stats(sum_t, sum_t2, total)
    vt_mean, _, vt_stderr = _stats(sum_vt, sum_vt2, total)
    vi_mean, _, vi_stderr = _stats(sum_vi, sum_vi2, total)
    rdv_mean, _, rdv_stderr = _stats(sum_rdv, sum_rdv2, total)

    old_lhs, old_rhs, old_res, old_rel = compute_virial_metrics(
        T_mean=t_mean,
        V_trap_mean=vt_mean,
        V_int_mean=vi_mean,
        E_mean=e_mean,
    )

    new_lhs = 2.0 * t_mean
    new_rhs = rdv_mean
    new_res = new_lhs - new_rhs
    new_rel = abs(new_res) / max(abs(e_mean), 1e-10)

    out: dict[str, Any] = {
        "dir": result_dir.name,
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
        "r_dot_gradV_mean": rdv_mean,
        "r_dot_gradV_stderr": rdv_stderr,
        "old_virial_lhs": old_lhs,
        "old_virial_rhs": old_rhs,
        "old_virial_residual": old_res,
        "old_virial_relative": old_rel,
        "new_virial_lhs": new_lhs,
        "new_virial_rhs": new_rhs,
        "new_virial_residual": new_res,
        "new_virial_relative": new_rel,
    }

    LOG.info("=== %s ===", result_dir.name)
    LOG.info("E      = %.6f +- %.6f", e_mean, e_stderr)
    LOG.info("T      = %.6f +- %.6f", t_mean, t_stderr)
    LOG.info("V_trap = %.6f +- %.6f", vt_mean, vt_stderr)
    LOG.info("V_int  = %.6f +- %.6f", vi_mean, vi_stderr)
    LOG.info("r·∇V   = %.6f +- %.6f", rdv_mean, rdv_stderr)
    LOG.info("Old virial : 2T=%.6f, 2Vt-Vi=%.6f, residual=%.6f, relative=%.2f%%", old_lhs, old_rhs, old_res, 100.0 * old_rel)
    LOG.info("New virial : 2T=%.6f, <r·∇V>=%.6f, residual=%.6f, relative=%.2f%%", new_lhs, new_rhs, new_res, 100.0 * new_rel)
    LOG.info("Delta residual(old-new)=%.6f", old_res - new_res)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old vs generalized virial diagnostics on saved runs.")
    parser.add_argument("--result-dir", required=True, help="Directory containing config.yaml and model.pt")
    parser.add_argument("--device", default="cuda:0", help="Evaluation device")
    parser.add_argument("--n-samples", type=int, default=8192, help="Total MH samples")
    parser.add_argument("--mh-steps", type=int, default=40, help="MH steps per batch")
    parser.add_argument("--mh-step-scale", type=float, default=0.25, help="Initial MH proposal scale")
    parser.add_argument("--mh-decorrelation", type=int, default=1, help="MH decorrelation multiplier")
    parser.add_argument("--mh-warmup-batches", type=int, default=20, help="Warmup batches before scoring")
    parser.add_argument("--fd-h", type=float, default=None, help="Optional FD step override")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path = Path(args.result_dir)
    if not path.exists() and not path.is_absolute():
        path = Path("results") / args.result_dir

    res = run_check(
        path,
        device=str(args.device),
        n_samples=int(args.n_samples),
        mh_steps=int(args.mh_steps),
        mh_step_scale=float(args.mh_step_scale),
        mh_decorrelation=int(args.mh_decorrelation),
        mh_warmup_batches=int(args.mh_warmup_batches),
        fd_h=args.fd_h,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        LOG.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
