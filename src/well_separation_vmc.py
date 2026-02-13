#!/usr/bin/env python
"""
Double-Well Ground-State VMC
=============================

Two modes:
  1) Per-d training:    Independent model per well separation d
  2) d-Conditioned:     One model with d as a native input, trained on all d

Both use the same SD × Backflow × Jastrow architecture from imaginary_time_vmc.py.

Usage:
  python well_separation_vmc.py --per_d           # Train independent models
  python well_separation_vmc.py --d_conditioned   # Train single d-conditioned model
  python well_separation_vmc.py --all             # Both
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["CUDA_MANUAL_DEVICE"] = "0"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from functions.Slater_Determinant import slater_determinant_closed_shell


def setup_sd(n_particles, dim, omega, E_ref):
    """Set up Slater-determinant orbitals (copied from imaginary_time_pinn)."""
    n_occ = n_particles // 2
    nx = max(2, int(math.ceil(math.sqrt(2 * n_occ))))
    ny = nx
    config.update(
        device=DEVICE, omega=omega, n_particles=n_particles,
        d=dim, dimensions=dim, basis="cart", nx=nx, ny=ny, E=E_ref,
    )
    params = config.get().as_dict()
    n_basis = nx * ny
    pairs = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    pairs.sort(key=lambda t: (t[0] + t[1], t[0]))
    sel = pairs[:n_occ]
    cols = [ix * ny + iy for (ix, iy) in sel]
    C_occ = torch.zeros(n_basis, n_occ, dtype=DTYPE, device=DEVICE)
    for j, c in enumerate(cols):
        C_occ[c, j] = 1.0
    spin = torch.cat([
        torch.zeros(n_particles // 2, dtype=torch.long, device=DEVICE),
        torch.ones(n_particles - n_particles // 2, dtype=torch.long, device=DEVICE),
    ])
    return C_occ, spin, params

DEVICE = torch.device("cpu")
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
torch.set_num_threads(4)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "well_separation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================
@dataclass
class WSConfig:
    n_particles: int = 2
    dim: int = 2
    omega: float = 1.0
    smooth_T: float = 0.2
    coulomb: bool = True
    # Training
    n_epochs: int = 800
    n_samples: int = 1024
    lr: float = 3e-3
    # Architecture
    pinn_hidden: int = 64
    pinn_layers: int = 2
    pinn_dL: int = 5
    bf_hidden: int = 32
    bf_layers: int = 2
    use_backflow: bool = True
    # Evaluation
    n_eval_samples: int = 4000
    eval_warmup: int = 500
    # d-conditioned mode
    d_embed_dim: int = 16
    d_values: tuple = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)


# ============================================================
# Soft-min double-well potential
# ============================================================
def compute_potential(x, omega, well_sep, smooth_T=0.2, coulomb=True):
    """V_ext + V_Coulomb with soft-core."""
    B, N, d = x.shape
    if well_sep <= 1e-10:
        V_ext = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    else:
        R_L = torch.zeros(d, dtype=x.dtype, device=x.device)
        R_R = torch.zeros(d, dtype=x.dtype, device=x.device)
        R_L[0] = -well_sep / 2
        R_R[0] = +well_sep / 2
        V_L = 0.5 * omega**2 * ((x - R_L)**2).sum(dim=-1)
        V_R = 0.5 * omega**2 * ((x - R_R)**2).sum(dim=-1)
        T = smooth_T
        V_per_particle = -T * torch.logaddexp(-V_L / T, -V_R / T)
        V_ext = V_per_particle.sum(dim=1)

    V_coul = torch.zeros(B, device=x.device, dtype=x.dtype)
    if coulomb:
        eps_sc = 1e-6 / max(omega, 1e-6)**0.5
        for i in range(N):
            for j in range(i + 1, N):
                r2 = ((x[:, i] - x[:, j])**2).sum(dim=-1)
                rij = torch.sqrt(r2 + eps_sc**2)
                V_coul = V_coul + 1.0 / rij
    return V_ext + V_coul


def compute_potential_decomposed(x, omega, well_sep, smooth_T=0.2, coulomb=True):
    """Return (V_ext, V_coul, r12) individually."""
    B, N, d = x.shape
    if well_sep <= 1e-10:
        V_ext = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    else:
        R_L = torch.zeros(d, dtype=x.dtype, device=x.device)
        R_R = torch.zeros(d, dtype=x.dtype, device=x.device)
        R_L[0] = -well_sep / 2
        R_R[0] = +well_sep / 2
        V_L = 0.5 * omega**2 * ((x - R_L)**2).sum(dim=-1)
        V_R = 0.5 * omega**2 * ((x - R_R)**2).sum(dim=-1)
        T = smooth_T
        V_per_particle = -T * torch.logaddexp(-V_L / T, -V_R / T)
        V_ext = V_per_particle.sum(dim=1)

    V_coul = torch.zeros(B, device=x.device, dtype=x.dtype)
    r12 = torch.zeros(B, device=x.device, dtype=x.dtype)
    if coulomb and N >= 2:
        eps_sc = 1e-6 / max(omega, 1e-6)**0.5
        r2 = ((x[:, 0] - x[:, 1])**2).sum(dim=-1)
        rij = torch.sqrt(r2 + eps_sc**2)
        V_coul = 1.0 / rij
        r12 = rij
    return V_ext, V_coul, r12


# ============================================================
# MCMC sampling
# ============================================================
def mcmc_sample(log_psi_fn, n_samples, n_particles, dim, omega, well_sep=0.0,
                n_warmup=400, step_size=0.5, target_acc=0.45):
    """Adaptive Metropolis from |Ψ|²."""
    sigma = 1.0 / math.sqrt(omega)
    x = torch.randn(n_samples, n_particles, dim, device=DEVICE, dtype=DTYPE) * sigma
    if well_sep > 1e-10:
        x[:, 0, 0] -= well_sep / 2
        if n_particles > 1:
            x[:, 1, 0] += well_sep / 2

    log_prob = 2.0 * log_psi_fn(x)
    step = step_size * sigma
    n_acc = 0

    for s in range(n_warmup):
        x_new = x + step * torch.randn_like(x)
        lp_new = 2.0 * log_psi_fn(x_new)
        accept = torch.log(torch.rand(n_samples, device=DEVICE, dtype=DTYPE)) < (lp_new - log_prob)
        x = torch.where(accept.view(-1, 1, 1), x_new, x)
        log_prob = torch.where(accept, lp_new, log_prob)
        rate = accept.float().mean().item()
        if s < n_warmup - 50:
            if rate > target_acc + 0.05: step *= 1.05
            elif rate < target_acc - 0.05: step *= 0.95
        if s >= n_warmup - 50:
            n_acc += rate
    return x.detach(), n_acc / 50


# ============================================================
# Local energy  (exact Laplacian)
# ============================================================
def compute_local_energy(wf, x, omega, well_sep, smooth_T=0.2, coulomb=True, **wf_kwargs):
    """E_L = -½(∇²logΨ + |∇logΨ|²) + V(x). Returns (E_L, T, V_ext, V_coul, r12)."""
    B, N, d = x.shape
    x = x.detach().requires_grad_(True)
    log_psi = wf(x, **wf_kwargs)

    grad_log = torch.autograd.grad(log_psi.sum(), x, create_graph=True)[0]
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            d2 = torch.autograd.grad(grad_log[:, i, j].sum(), x, create_graph=True, retain_graph=True)[0]
            lap = lap + d2[:, i, j]

    grad_sq = (grad_log**2).sum(dim=(1, 2))
    T = -0.5 * (lap + grad_sq)

    with torch.no_grad():
        V_ext, V_coul, r12 = compute_potential_decomposed(x, omega, well_sep, smooth_T, coulomb)

    E_L = T + V_ext + V_coul
    return E_L, T.detach(), V_ext, V_coul, r12


# ============================================================
# Jastrow  (standard 3-branch, no τ)
# ============================================================
class Jastrow(nn.Module):
    """3-branch PINN Jastrow: φ(r_i) + ψ(r_ij) → ρ readout."""
    def __init__(self, n_particles, dim, omega, hidden=64, layers=2, dL=5):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.omega = omega

        # φ branch (single-particle)
        phi_layers = [nn.Linear(dim + 1, hidden), nn.GELU()]
        for _ in range(layers - 1):
            phi_layers += [nn.Linear(hidden, hidden), nn.GELU()]
        phi_layers.append(nn.Linear(hidden, dL))
        self.phi = nn.Sequential(*phi_layers)

        # ψ branch (pair)
        psi_layers = [nn.Linear(6, hidden), nn.GELU()]
        for _ in range(layers - 1):
            psi_layers += [nn.Linear(hidden, hidden), nn.GELU()]
        psi_layers.append(nn.Linear(hidden, dL))
        self.psi = nn.Sequential(*psi_layers)

        # ρ readout
        self.rho = nn.Sequential(
            nn.Linear(2 * dL, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.rho[-1].bias)
        nn.init.normal_(self.rho[-1].weight, std=0.01)

        # Kato cusp
        if dim == 2:
            self.cusp_gamma_anti = 1.0 / (dim - 1) if dim > 1 else 1.0
        else:
            self.cusp_gamma_anti = 1.0 / (dim - 1)

    def forward(self, x, spin=None):
        B, N, d = x.shape
        xs = x * math.sqrt(self.omega)

        # φ: per-particle + spin
        r2 = (xs**2).sum(dim=-1, keepdim=True)
        phi_in = torch.cat([xs, r2], dim=-1).view(B * N, -1)
        phi_out = self.phi(phi_in).view(B, N, -1).mean(dim=1)

        # ψ: per-pair
        psi_sum = torch.zeros(B, self.psi[-1].out_features, device=x.device, dtype=x.dtype)
        cusp_total = torch.zeros(B, device=x.device, dtype=x.dtype)
        n_pairs = 0
        for i in range(N):
            for j in range(i+1, N):
                dr = x[:, i] - x[:, j]
                r2_ij = (dr**2).sum(dim=-1)
                rij = torch.sqrt(r2_ij + 1e-12)
                feat = torch.stack([
                    torch.log1p(rij),
                    rij / (1 + rij),
                    torch.exp(-rij**2),
                    torch.exp(-0.5 * rij),
                    torch.exp(-rij),
                    torch.exp(-2 * rij),
                ], dim=-1)
                psi_sum = psi_sum + self.psi(feat)
                cusp_total = cusp_total + self.cusp_gamma_anti * rij * torch.exp(-rij)
                n_pairs += 1

        if n_pairs > 0:
            psi_sum = psi_sum / n_pairs

        rho_in = torch.cat([phi_out, psi_sum], dim=-1)
        f = self.rho(rho_in)
        return f.squeeze(-1) + cusp_total


# ============================================================
# Backflow (CTNN, no τ)
# ============================================================
class BackflowNet(nn.Module):
    """CTNN message-passing backflow."""
    def __init__(self, d=2, omega=1.0, hidden=32, layers=2, msg_hidden=32, msg_layers=2,
                 bf_scale_init=0.3):
        super().__init__()
        self.d = d
        self.omega = omega

        # Node embedding: (x*√ω, spin) → hidden
        self.node_embed = nn.Sequential(
            nn.Linear(d + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # Edge embedding: (Δr, |r|, |r|²) → hidden
        edge_in = d + 2
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in, msg_hidden),
            nn.GELU(),
            nn.Linear(msg_hidden, msg_hidden),
        )

        # Transport maps
        self.v2e = nn.ModuleList()
        self.e2v = nn.ModuleList()
        for _ in range(layers):
            self.v2e.append(nn.Sequential(
                nn.Linear(hidden + msg_hidden, msg_hidden),
                nn.GELU(),
                nn.Linear(msg_hidden, msg_hidden),
            ))
            self.e2v.append(nn.Sequential(
                nn.Linear(msg_hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.scale = nn.Parameter(torch.tensor(math.log(math.exp(bf_scale_init) - 1)))

    def forward(self, x, spin=None):
        B, N, d = x.shape
        xs = x * math.sqrt(self.omega)
        s = spin if spin is not None else torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)
        if s.dim() == 2:
            s = s.unsqueeze(-1)

        h_v = self.node_embed(torch.cat([xs, s], dim=-1))

        edges, h_e = [], []
        for i in range(N):
            for j in range(N):
                if i == j: continue
                dr = xs[:, j] - xs[:, i]
                r2 = (dr**2).sum(dim=-1, keepdim=True)
                rr = torch.sqrt(r2 + 1e-12)
                e_feat = torch.cat([dr, rr, r2], dim=-1)
                edges.append((i, j))
                h_e.append(self.edge_embed(e_feat))
        h_e = torch.stack(h_e, dim=1)

        for layer_idx in range(len(self.v2e)):
            new_h_e = []
            for eidx, (i, j) in enumerate(edges):
                cat = torch.cat([h_v[:, i], h_e[:, eidx]], dim=-1)
                new_h_e.append(self.v2e[layer_idx](cat))
            h_e = torch.stack(new_h_e, dim=1)

            new_h_v = torch.zeros_like(h_v)
            for eidx, (i, j) in enumerate(edges):
                new_h_v[:, j] = new_h_v[:, j] + self.e2v[layer_idx](h_e[:, eidx])
            h_v = h_v + new_h_v / max(N - 1, 1)

        dx = self.head(h_v) * F.softplus(self.scale)
        dx = dx - dx.mean(dim=1, keepdim=True)  # zero COM
        return dx


# ============================================================
# Full wavefunction: SD × BF × Jastrow (no τ)
# ============================================================
class Wavefunction(nn.Module):
    def __init__(self, n_particles, dim, omega, C_occ, spin, params, cfg):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.omega = omega
        self.register_buffer("C_occ", C_occ)
        self.register_buffer("spin", spin)
        self.params = params

        self.jastrow = Jastrow(n_particles, dim, omega,
                               hidden=cfg.pinn_hidden, layers=cfg.pinn_layers, dL=cfg.pinn_dL)
        self.bf_net = None
        if cfg.use_backflow:
            self.bf_net = BackflowNet(d=dim, omega=omega, hidden=cfg.bf_hidden,
                                      layers=cfg.bf_layers, bf_scale_init=0.3)

    def forward(self, x):
        B = x.shape[0]
        spin = self.spin.expand(B, -1)
        x_eff = x
        if self.bf_net is not None:
            dx = self.bf_net(x, spin=spin)
            x_eff = x + dx
        _, log_sd = slater_determinant_closed_shell(x_eff, self.C_occ, params=self.params, spin=spin)
        f_jastrow = self.jastrow(x, spin=spin)
        return log_sd + f_jastrow


# ============================================================
# d-Conditioned Wavefunction
# ============================================================
class DistanceEmbedding(nn.Module):
    """Map scalar d into a learned embedding vector."""
    def __init__(self, embed_dim=16, n_freq=8, max_freq=5.0):
        super().__init__()
        freqs = torch.linspace(0.1, max_freq, n_freq, dtype=DTYPE)
        self.register_buffer("freqs", freqs)
        raw_dim = 2 * n_freq + 1
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.normal_(self.proj[-1].weight, std=0.01)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """d: (B,) → (B, embed_dim)."""
        t = d.unsqueeze(-1)
        features = torch.cat([t, torch.sin(self.freqs * t), torch.cos(self.freqs * t)], dim=-1)
        return self.proj(features)


class DCondJastrow(nn.Module):
    """Jastrow with d-embedding injected into readout."""
    def __init__(self, n_particles, dim, omega, hidden=64, layers=2, dL=5, d_embed_dim=16):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.omega = omega

        phi_layers = [nn.Linear(dim + 1, hidden), nn.GELU()]
        for _ in range(layers - 1):
            phi_layers += [nn.Linear(hidden, hidden), nn.GELU()]
        phi_layers.append(nn.Linear(hidden, dL))
        self.phi = nn.Sequential(*phi_layers)

        psi_layers = [nn.Linear(6, hidden), nn.GELU()]
        for _ in range(layers - 1):
            psi_layers += [nn.Linear(hidden, hidden), nn.GELU()]
        psi_layers.append(nn.Linear(hidden, dL))
        self.psi = nn.Sequential(*psi_layers)

        # Readout takes φ_mean, ψ_mean, AND d-embedding
        self.rho = nn.Sequential(
            nn.Linear(2 * dL + d_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.rho[-1].bias)
        nn.init.normal_(self.rho[-1].weight, std=0.01)

        if dim == 2:
            self.cusp_gamma_anti = 1.0 / (dim - 1)
        else:
            self.cusp_gamma_anti = 1.0 / (dim - 1)

    def forward(self, x, d_emb, spin=None):
        B, N, d = x.shape
        xs = x * math.sqrt(self.omega)

        r2 = (xs**2).sum(dim=-1, keepdim=True)
        phi_in = torch.cat([xs, r2], dim=-1).view(B * N, -1)
        phi_out = self.phi(phi_in).view(B, N, -1).mean(dim=1)

        psi_sum = torch.zeros(B, self.psi[-1].out_features, device=x.device, dtype=x.dtype)
        cusp_total = torch.zeros(B, device=x.device, dtype=x.dtype)
        n_pairs = 0
        for i in range(N):
            for j in range(i+1, N):
                dr = x[:, i] - x[:, j]
                r2_ij = (dr**2).sum(dim=-1)
                rij = torch.sqrt(r2_ij + 1e-12)
                feat = torch.stack([
                    torch.log1p(rij), rij / (1 + rij), torch.exp(-rij**2),
                    torch.exp(-0.5 * rij), torch.exp(-rij), torch.exp(-2 * rij),
                ], dim=-1)
                psi_sum = psi_sum + self.psi(feat)
                cusp_total = cusp_total + self.cusp_gamma_anti * rij * torch.exp(-rij)
                n_pairs += 1
        if n_pairs > 0:
            psi_sum = psi_sum / n_pairs

        rho_in = torch.cat([phi_out, psi_sum, d_emb], dim=-1)
        f = self.rho(rho_in)
        return f.squeeze(-1) + cusp_total


class DCondBackflow(nn.Module):
    """CTNN backflow with d-embedding in node features."""
    def __init__(self, d=2, omega=1.0, hidden=32, layers=2, msg_hidden=32, d_embed_dim=16,
                 bf_scale_init=0.3):
        super().__init__()
        self.d_dim = d
        self.omega = omega

        self.node_embed = nn.Sequential(
            nn.Linear(d + 1 + d_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        edge_in = d + 2
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in, msg_hidden),
            nn.GELU(),
            nn.Linear(msg_hidden, msg_hidden),
        )

        self.v2e = nn.ModuleList()
        self.e2v = nn.ModuleList()
        for _ in range(layers):
            self.v2e.append(nn.Sequential(
                nn.Linear(hidden + msg_hidden, msg_hidden),
                nn.GELU(),
                nn.Linear(msg_hidden, msg_hidden),
            ))
            self.e2v.append(nn.Sequential(
                nn.Linear(msg_hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.scale = nn.Parameter(torch.tensor(math.log(math.exp(bf_scale_init) - 1)))

    def forward(self, x, d_emb, spin=None):
        B, N, d = x.shape
        xs = x * math.sqrt(self.omega)
        s = spin if spin is not None else torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)
        if s.dim() == 2: s = s.unsqueeze(-1)

        # Broadcast d-embedding to all particles
        d_emb_exp = d_emb.unsqueeze(1).expand(B, N, -1)
        h_v = self.node_embed(torch.cat([xs, s, d_emb_exp], dim=-1))

        edges, h_e = [], []
        for i in range(N):
            for j in range(N):
                if i == j: continue
                dr = xs[:, j] - xs[:, i]
                r2 = (dr**2).sum(dim=-1, keepdim=True)
                rr = torch.sqrt(r2 + 1e-12)
                e_feat = torch.cat([dr, rr, r2], dim=-1)
                edges.append((i, j))
                h_e.append(self.edge_embed(e_feat))
        h_e = torch.stack(h_e, dim=1)

        for layer_idx in range(len(self.v2e)):
            new_h_e = []
            for eidx, (i, j) in enumerate(edges):
                cat = torch.cat([h_v[:, i], h_e[:, eidx]], dim=-1)
                new_h_e.append(self.v2e[layer_idx](cat))
            h_e = torch.stack(new_h_e, dim=1)
            new_h_v = torch.zeros_like(h_v)
            for eidx, (i, j) in enumerate(edges):
                new_h_v[:, j] = new_h_v[:, j] + self.e2v[layer_idx](h_e[:, eidx])
            h_v = h_v + new_h_v / max(N - 1, 1)

        dx = self.head(h_v) * F.softplus(self.scale)
        dx = dx - dx.mean(dim=1, keepdim=True)
        return dx


class DCondWavefunction(nn.Module):
    """Full wavefunction with d as a native input."""
    def __init__(self, n_particles, dim, omega, C_occ, spin, params, cfg):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.omega = omega
        self.register_buffer("C_occ", C_occ)
        self.register_buffer("spin", spin)
        self.params = params

        self.d_embed = DistanceEmbedding(embed_dim=cfg.d_embed_dim)
        self.jastrow = DCondJastrow(n_particles, dim, omega,
                                     hidden=cfg.pinn_hidden, layers=cfg.pinn_layers,
                                     dL=cfg.pinn_dL, d_embed_dim=cfg.d_embed_dim)
        self.bf_net = None
        if cfg.use_backflow:
            self.bf_net = DCondBackflow(d=dim, omega=omega, hidden=cfg.bf_hidden,
                                        layers=cfg.bf_layers, d_embed_dim=cfg.d_embed_dim)

    def forward(self, x, d_val=None):
        """x: (B, N, d), d_val: (B,) well separation."""
        B = x.shape[0]
        spin = self.spin.expand(B, -1)
        d_emb = self.d_embed(d_val)  # (B, embed_dim)

        x_eff = x
        if self.bf_net is not None:
            dx = self.bf_net(x, d_emb, spin=spin)
            x_eff = x + dx

        _, log_sd = slater_determinant_closed_shell(x_eff, self.C_occ, params=self.params, spin=spin)
        f_jastrow = self.jastrow(x, d_emb, spin=spin)
        return log_sd + f_jastrow


# ============================================================
# Training: Per-d mode
# ============================================================
def train_per_d(cfg: WSConfig, well_sep: float, warm_start_state=None) -> dict:
    """Train an independent model for a single well separation d.
    If warm_start_state is given, initialise from it (curriculum transfer).
    """
    print(f"\n{'='*60}")
    print(f"  Per-d training: d = {well_sep:.1f}")
    print(f"{'='*60}")
    t0 = time.time()

    C_occ, spin, params = setup_sd(cfg.n_particles, cfg.dim, cfg.omega, 3.0)
    wf = Wavefunction(cfg.n_particles, cfg.dim, cfg.omega, C_occ, spin, params, cfg).to(DEVICE)
    if warm_start_state is not None:
        wf.load_state_dict(warm_start_state, strict=False)
        print(f"  Warm-started from previous d")
    n_params = sum(p.numel() for p in wf.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = optim.Adam(wf.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_epochs, eta_min=cfg.lr / 10)

    print_every = max(cfg.n_epochs // 8, 1)
    best_E, best_state = float("inf"), None
    energies_train = []

    for epoch in range(cfg.n_epochs):
        with torch.no_grad():
            log_fn = lambda x_: wf(x_)
            x, acc = mcmc_sample(log_fn, cfg.n_samples, cfg.n_particles, cfg.dim,
                                 cfg.omega, well_sep, n_warmup=300, step_size=0.5)

        E_L, T, V_ext, V_coul, r12 = compute_local_energy(
            wf, x, cfg.omega, well_sep, cfg.smooth_T, cfg.coulomb)

        # ── NaN/Inf guard + outlier clip ──
        valid = torch.isfinite(E_L)
        if valid.sum() < 32:
            # Too few valid samples — skip this epoch
            energies_train.append(float("nan"))
            if epoch % print_every == 0:
                print(f"    [VMC] Ep {epoch:4d}/{cfg.n_epochs} | SKIPPED (insufficient valid samples)")
            continue
        E_L_clean = E_L[valid]
        E_med = E_L_clean.median()
        E_mad = (E_L_clean - E_med).abs().median() * 1.4826  # robust σ
        clip_lo = E_med - 5 * max(E_mad, 0.1)
        clip_hi = E_med + 5 * max(E_mad, 0.1)
        E_L = torch.where(valid, E_L, E_med)  # replace NaN with median
        E_L = torch.clamp(E_L, clip_lo.item(), clip_hi.item())

        E_mean = E_L.mean()
        loss = ((E_L - E_mean.detach())**2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wf.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        E_val = E_mean.item()
        energies_train.append(E_val)
        if not math.isnan(E_val) and E_val < best_E:
            best_E = E_val
            best_state = {k: v.clone() for k, v in wf.state_dict().items()}

        if epoch % print_every == 0:
            E_err = E_L.detach().std().item() / math.sqrt(cfg.n_samples)
            print(f"    [VMC] Ep {epoch:4d}/{cfg.n_epochs} | E={E_val:.5f}±{E_err:.5f} | "
                  f"var={loss.item():.4f} | acc={acc:.2f}")

    if best_state is not None:
        wf.load_state_dict(best_state)

    # Final evaluation with more samples
    result = evaluate_model(wf, cfg, well_sep, log_fn_factory=lambda wf_: lambda x_: wf_(x_))
    result["train_energies"] = energies_train
    result["n_params"] = n_params
    result["train_time"] = time.time() - t0

    print(f"  Final: E = {result['E']:.5f} ± {result['E_err']:.5f} | "
          f"T={result['T']:.4f} | V_coul={result['V_coul']:.4f} | r12={result['r12']:.3f}")
    # Return final state_dict for curriculum transfer
    result["_state_dict"] = {k: v.clone() for k, v in wf.state_dict().items()}
    return result


def evaluate_model(wf, cfg, well_sep, log_fn_factory, d_val_tensor=None):
    """Evaluate a trained model at a specific well separation."""
    with torch.no_grad():
        if d_val_tensor is not None:
            log_fn = lambda x_: wf(x_, d_val=d_val_tensor[:x_.shape[0]])
        else:
            log_fn = log_fn_factory(wf)
        x_eval, acc = mcmc_sample(log_fn, cfg.n_eval_samples, cfg.n_particles, cfg.dim,
                                   cfg.omega, well_sep, n_warmup=cfg.eval_warmup, step_size=0.5)

    if d_val_tensor is not None:
        d_batch = d_val_tensor[:x_eval.shape[0]]
        E_L, T, V_ext, V_coul, r12 = compute_local_energy(
            wf, x_eval, cfg.omega, well_sep, cfg.smooth_T, cfg.coulomb, d_val=d_batch)
    else:
        E_L, T, V_ext, V_coul, r12 = compute_local_energy(
            wf, x_eval, cfg.omega, well_sep, cfg.smooth_T, cfg.coulomb)

    E_vals = E_L.detach()
    ns = cfg.n_eval_samples
    return {
        "d": well_sep,
        "E": E_vals.mean().item(),
        "E_err": E_vals.std().item() / math.sqrt(ns),
        "E_var": E_vals.var().item(),
        "T": T.mean().item(),
        "V_ext": V_ext.mean().item(),
        "V_coul": V_coul.mean().item(),
        "r12": r12.mean().item(),
        "acc": acc,
    }


# ============================================================
# Training: d-Conditioned mode
# ============================================================
def train_d_conditioned(cfg: WSConfig) -> list:
    """Train a single d-conditioned model on all d values simultaneously.
    
    Uses round-robin mini-epochs: each step trains on ONE d value (cycled),
    which is ~6x faster per epoch than sampling all d simultaneously.
    """
    print(f"\n{'='*60}")
    print(f"  d-Conditioned training: d = {list(cfg.d_values)}")
    print(f"{'='*60}")
    t0 = time.time()

    C_occ, spin, params = setup_sd(cfg.n_particles, cfg.dim, cfg.omega, 3.0)
    wf = DCondWavefunction(cfg.n_particles, cfg.dim, cfg.omega, C_occ, spin, params, cfg).to(DEVICE)
    n_params = sum(p.numel() for p in wf.parameters() if p.requires_grad)
    print(f"  d-Conditioned WF: {n_params:,} parameters")

    optimizer = optim.Adam(wf.parameters(), lr=cfg.lr)
    n_total_epochs = cfg.n_epochs * 2  # 1600 total steps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_total_epochs, eta_min=cfg.lr / 10)

    print_every = max(n_total_epochs // 16, 1)
    best_loss_avg, best_state = float("inf"), None
    d_vals = list(cfg.d_values)
    n_d = len(d_vals)
    n_samp = 512  # Samples per mini-epoch (single d)
    loss_tracker = {d: [] for d in d_vals}

    for epoch in range(n_total_epochs):
        # Round-robin: pick one d per step
        d_val = d_vals[epoch % n_d]
        d_tensor = torch.full((n_samp,), d_val, dtype=DTYPE, device=DEVICE)

        with torch.no_grad():
            log_fn = lambda x_, dt=d_tensor: wf(x_, d_val=dt[:x_.shape[0]])
            x, acc = mcmc_sample(log_fn, n_samp, cfg.n_particles, cfg.dim,
                                 cfg.omega, d_val, n_warmup=200, step_size=0.5)

        # Compute local energy
        E_L, T_k, V_ext, V_coul, r12 = compute_local_energy(
            wf, x, cfg.omega, d_val, cfg.smooth_T, cfg.coulomb, d_val=d_tensor)

        # NaN/Inf guard
        valid = torch.isfinite(E_L)
        if valid.sum() < 32:
            scheduler.step()
            continue
        E_L_clean = E_L[valid]
        E_med = E_L_clean.median()
        E_mad = (E_L_clean - E_med).abs().median() * 1.4826
        clip_r = 5 * max(E_mad.item(), 0.1)
        E_L = torch.where(valid, E_L, E_med)
        E_L = torch.clamp(E_L, E_med.item() - clip_r, E_med.item() + clip_r)

        E_mean = E_L.mean()
        loss = ((E_L - E_mean.detach())**2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wf.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_tracker[d_val].append(E_mean.item())
        # Track rolling average for checkpointing (last full cycle)
        if epoch >= n_d and epoch % n_d == 0:
            avg_E = sum(loss_tracker[dv][-1] for dv in d_vals if loss_tracker[dv]) / n_d
            if avg_E < best_loss_avg:
                best_loss_avg = avg_E
                best_state = {k: v.clone() for k, v in wf.state_dict().items()}

        if epoch % print_every == 0:
            per_d_str = ""
            for dv in d_vals:
                if loss_tracker[dv]:
                    per_d_str += f" d={dv:.0f}:{loss_tracker[dv][-1]:.3f}"
            print(f"    [d-VMC] Ep {epoch:4d}/{n_total_epochs} | d={d_val:.0f} E={E_mean.item():.5f} | "
                  f"var={loss.item():.4f} |{per_d_str}")

    if best_state is not None:
        wf.load_state_dict(best_state)

    # Evaluate at each d
    results = []
    for d_val in d_vals:
        d_tensor = torch.full((cfg.n_eval_samples,), d_val, dtype=DTYPE, device=DEVICE)
        result = evaluate_model(wf, cfg, d_val, log_fn_factory=None, d_val_tensor=d_tensor)
        result["mode"] = "d-conditioned"
        result["n_params"] = n_params
        results.append(result)
        print(f"  d={d_val:.1f}: E={result['E']:.5f}±{result['E_err']:.5f} | "
              f"T={result['T']:.4f} | V_coul={result['V_coul']:.4f} | r12={result['r12']:.3f}")

    train_time = time.time() - t0
    for r in results:
        r["train_time"] = train_time / n_d  # Amortised

    return results


# ============================================================
# Plotting
# ============================================================
def plot_results(per_d_results, dcond_results=None, ref_data=None):
    """Generate comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ds_pd = [r["d"] for r in per_d_results]
    Es_pd = [r["E"] for r in per_d_results]
    Eerr_pd = [r["E_err"] for r in per_d_results]

    # Energy vs d
    ax = axes[0, 0]
    ax.errorbar(ds_pd, Es_pd, yerr=Eerr_pd, fmt='o-', label="Per-d VMC", capsize=3)
    if dcond_results:
        ds_dc = [r["d"] for r in dcond_results]
        Es_dc = [r["E"] for r in dcond_results]
        Eerr_dc = [r["E_err"] for r in dcond_results]
        ax.errorbar(ds_dc, Es_dc, yerr=Eerr_dc, fmt='s--', label="d-Conditioned VMC", capsize=3)
    if ref_data:
        ds_ref = [r["d"] for r in ref_data]
        Es_ref = [r["E_ref"] for r in ref_data]
        ax.plot(ds_ref, Es_ref, 'k--', alpha=0.5, label="Reference")
    ax.set_xlabel("Well separation d")
    ax.set_ylabel("E")
    ax.set_title("Ground-State Energy")
    ax.legend()

    # V_coul vs d
    ax = axes[0, 1]
    ax.plot(ds_pd, [r["V_coul"] for r in per_d_results], 'o-', label="Per-d")
    if dcond_results:
        ax.plot([r["d"] for r in dcond_results], [r["V_coul"] for r in dcond_results], 's--', label="d-Cond")
    ax.set_xlabel("d"); ax.set_ylabel("V_coul"); ax.set_title("Coulomb Energy")
    ax.legend()

    # r12 vs d
    ax = axes[1, 0]
    ax.plot(ds_pd, [r["r12"] for r in per_d_results], 'o-', label="Per-d")
    if dcond_results:
        ax.plot([r["d"] for r in dcond_results], [r["r12"] for r in dcond_results], 's--', label="d-Cond")
    ax.plot(ds_pd, ds_pd, 'k:', alpha=0.3, label="r12 = d")
    ax.set_xlabel("d"); ax.set_ylabel("<r12>"); ax.set_title("Pair Distance")
    ax.legend()

    # T vs d
    ax = axes[1, 1]
    ax.plot(ds_pd, [r["T"] for r in per_d_results], 'o-', label="Per-d")
    if dcond_results:
        ax.plot([r["d"] for r in dcond_results], [r["T"] for r in dcond_results], 's--', label="d-Cond")
    ax.axhline(1.0, color='k', ls=':', alpha=0.3, label="T=1.0")
    ax.set_xlabel("d"); ax.set_ylabel("T"); ax.set_title("Kinetic Energy")
    ax.legend()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "well_separation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {RESULTS_DIR / 'well_separation_comparison.png'}")


# ============================================================
# Reference energies (from existing data)
# ============================================================
def get_reference_energies():
    """Load reference data from existing results if available."""
    ref_path = Path(__file__).resolve().parent.parent / "results" / "double_well" / "imaginary_time_final.json"
    if ref_path.exists():
        with open(ref_path) as f:
            data = json.load(f)
        refs = []
        for sep in data["separations"]:
            k = str(float(sep))
            v = data["vmc"][k]
            refs.append({"d": sep, "E_ref": v["E_ref"]})
        return refs
    return None


# ============================================================
# Main entry points
# ============================================================
def run_per_d(cfg: WSConfig):
    """Run per-d sweep with incremental saving and curriculum warm-start."""
    out_path = RESULTS_DIR / "per_d_results.json"
    # Load existing results to skip already-completed d values
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            for r in json.load(f):
                if not math.isnan(r.get("E", float("nan"))):
                    existing[r["d"]] = r

    results = []
    prev_state = None  # For curriculum warm-start
    for d_val in cfg.d_values:
        if d_val in existing:
            print(f"\n  Skipping d={d_val:.1f} (already computed: E={existing[d_val]['E']:.5f})")
            results.append(existing[d_val])
            prev_state = None  # Can't warm-start from cached (no state_dict)
            continue
        r = train_per_d(cfg, d_val, warm_start_state=prev_state)
        prev_state = r.pop("_state_dict", None)
        r["mode"] = "per-d"
        if "train_energies" in r:
            r["train_energies"] = r["train_energies"][-10:]
        results.append(r)
        # Save incrementally
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n  Results saved: {out_path}")
    return results


def run_d_conditioned(cfg: WSConfig):
    """Run d-conditioned training."""
    results = train_d_conditioned(cfg)
    with open(RESULTS_DIR / "d_conditioned_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {RESULTS_DIR / 'd_conditioned_results.json'}")
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Double-Well Ground-State VMC")
    p.add_argument("--per_d", action="store_true", help="Train independent models per d")
    p.add_argument("--d_conditioned", action="store_true", help="Train d-conditioned model")
    p.add_argument("--all", action="store_true", help="Run both modes")
    args = p.parse_args()

    cfg = WSConfig(
        d_values=(0.0, 1.0, 2.0, 4.0, 6.0, 8.0),
        n_epochs=800,
        n_samples=1024,
        lr=3e-3,
        n_eval_samples=4000,
        eval_warmup=500,
    )

    per_d_results, dcond_results = None, None

    if args.per_d or args.all:
        per_d_results = run_per_d(cfg)
    if args.d_conditioned or args.all:
        dcond_results = run_d_conditioned(cfg)

    if per_d_results is None and not (args.per_d or args.d_conditioned or args.all):
        # Default: run both
        per_d_results = run_per_d(cfg)
        dcond_results = run_d_conditioned(cfg)

    # Plot
    ref = get_reference_energies()
    if per_d_results:
        plot_results(per_d_results, dcond_results, ref)

    print("\nDone!")
