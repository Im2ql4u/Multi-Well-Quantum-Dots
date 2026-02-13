#!/usr/bin/env python
"""
Imaginary-Time VMC with τ-Conditioned Wavefunction
===================================================

The FULL wavefunction takes imaginary time τ as a native input:

    log Ψ(x, τ) = log|SD(x + BF(x,τ))| + J(x, τ) + cusp(x)

where SD is the Slater determinant, BF is a τ-conditioned CTNN backflow,
J is a τ-conditioned PINN Jastrow, and cusp is the analytic e-e cusp.

Key advantage over the SpectralG approach:
  - Full ~28K parameter network evolves with τ (not a tiny 3K add-on)
  - Backflow redistributes particles as function of τ
  - Jastrow correlations adapt to the quantum state at each τ
  - No importance-weight collapse: can MCMC sample |Ψ(x,τ)|² directly

Training:
  Phase 1: VMC at τ = τ_max → ground state (standard variance minimisation)
  Phase 2: PDE training across all τ:
           ∂_τ log Ψ + E_L(τ) - E₀ = 0
           with IC at τ=0 (perturbed excited state)
  Phase 3: Evaluate E(τ) via direct MCMC at each τ, fit exponential gaps

Usage:
  python imaginary_time_vmc.py --test_free   # Non-interacting validation
  python imaginary_time_vmc.py --tiny        # Interacting N=2, quick test
  python imaginary_time_vmc.py --full        # Converged interacting
  python imaginary_time_vmc.py --sweep       # Distance sweep d=[0,1,2,4]

NOTE: This file does NOT modify PINN.py or any other main code.
"""

from __future__ import annotations

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

from functions.Slater_Determinant import slater_determinant_closed_shell

# Reuse fitting utilities from the PDE-only approach
from imaginary_time_pinn import (
    _save,
    fit_best_estimate,
    fit_double_exponential,
    fit_log_linear,
    fit_optimal_E0,
    fit_restricted_exponential,
    fit_single_exponential,
    setup_sd,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
torch.set_num_threads(4)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "imag_time_vmc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Improved fitting: scan E₀ over wider range with focused τ window
# ============================================================
def fit_scan_E0(traj, E_ref, E_vmc=None, tau_range=(0.1, 2.0)):
    """Scan E₀ from well below E_ref to the tail average.

    Key insight: the effective E₀ for log-linear fitting is often
    slightly *below* both the exact E_ref and the VMC energy because
    the wavefunction at large τ hasn't fully relaxed (PDE residual).
    Scanning downward from E_tail finds this sweet spot.
    """
    tau = np.array([r["tau"] for r in traj])
    E = np.array([r["E"] for r in traj])
    E_err = np.clip(np.array([r["E_err"] for r in traj]), 1e-6, None)
    tmin, tmax = tau_range

    E_tail = np.mean([r["E"] for r in traj[-5:]])
    E_lo = E_ref - 0.05  # scan well below E_ref
    E_hi = E_tail - 1e-5  # must be below E(τ→∞)

    if E_lo >= E_hi:
        E_lo = E_hi - 0.05

    best = {"success": False}
    best_residual = float("inf")

    for E0_trial in np.linspace(E_lo, E_hi, 500):
        dE = E - E0_trial
        mask = (dE > 1e-4) & (tau > tmin) & (tau < tmax)
        n = mask.sum()
        if n < 6:
            continue
        t_m, lndE = tau[mask], np.log(dE[mask])
        w = dE[mask] / E_err[mask]
        w = np.clip(w, 0.1, 100)
        W = np.diag(w**2)
        A = np.column_stack([np.ones(n), t_m])
        try:
            AtwA = A.T @ W @ A
            Atwy = A.T @ W @ lndE
            params = np.linalg.solve(AtwA, Atwy)
            pred = A @ params
            # Residual = weighted MSE
            res = np.sum(w**2 * (lndE - pred) ** 2) / n
            if res < best_residual:
                best_residual = res
                cov = np.linalg.inv(AtwA)
                gamma = -params[1]
                gamma_err = np.sqrt(cov[1, 1])
                best = {
                    "E0": E0_trial,
                    "gamma": gamma,
                    "gamma_err": gamma_err,
                    "gap": gamma / 2,
                    "gap_err": gamma_err / 2,
                    "residual": float(res),
                    "n_points": int(n),
                    "tau_range": list(tau_range),
                    "method": "scan_E0",
                    "success": True,
                }
        except np.linalg.LinAlgError:
            continue
    return best


# ============================================================
# Configuration
# ============================================================
@dataclass
class VMCConfig:
    n_particles: int = 2
    dim: int = 2
    omega: float = 1.0
    well_sep: float = 0.0
    smooth_T: float = 0.2
    E_ref: float = 3.0
    coulomb: bool = True
    tau_max: float = 5.0
    # Phase 1: VMC ground state
    n_epochs_vmc: int = 1200
    n_samples_vmc: int = 1024
    lr_vmc: float = 3e-3
    # Phase 2: PDE training
    n_precompute: int = 8192
    n_epochs_pde: int = 8000
    batch_pde: int = 256
    lr_pde: float = 5e-4
    lambda_ic: float = 50.0
    lambda_reg: float = 0.005
    ic_amplitude: float = 0.5
    ic_type: str = "dipole"
    pde_vmc_freq: int = 5  # compute VMC anchor every N steps
    lambda_vmc: float = 0.5  # VMC anchor loss weight
    # Phase 3: evaluation
    n_tau_eval: int = 50
    n_samples_eval: int = 4000
    mcmc_warmup_eval: int = 400
    # Architecture
    pinn_hidden: int = 64
    pinn_layers: int = 2
    pinn_dL: int = 5
    bf_hidden: int = 32
    bf_layers: int = 2
    use_backflow: bool = True
    tau_embed_dim: int = 16
    tau_n_freq: int = 8


# ============================================================
# τ Embedding
# ============================================================
class SinusoidalEmbedding(nn.Module):
    """Fourier feature embedding for imaginary time τ."""

    def __init__(self, embed_dim: int = 16, n_freq: int = 8, max_freq: float = 10.0):
        super().__init__()
        freqs = torch.linspace(0.1, max_freq, n_freq, dtype=DTYPE)
        self.register_buffer("freqs", freqs)
        raw_dim = 2 * n_freq + 1
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Small init so τ-conditioning starts near zero
        nn.init.normal_(self.proj[-1].weight, std=0.01)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (B,) → (B, embed_dim)."""
        t = tau.unsqueeze(-1)  # (B, 1)
        features = torch.cat([t, torch.sin(self.freqs * t), torch.cos(self.freqs * t)], dim=-1)
        return self.proj(features)


# ============================================================
# τ-Conditioned Jastrow  (3-branch PINN architecture + τ)
# ============================================================
class TauJastrow(nn.Module):
    """
    f(x, τ) = ρ([φ_mean, ψ_mean, extras, τ_emb]) + cusp(x)

    Same 3-branch architecture as PINN:
      φ: per-particle features (one-body)
      ψ: pair features on safe radial channels (two-body)
      ρ: readout MLP, now with τ embedding appended

    The cusp correction is τ-independent (exact Kato cusp).
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        dL: int = 5,
        hidden_dim: int = 64,
        n_layers: int = 2,
        tau_embed_dim: int = 16,
        tau_n_freq: int = 8,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.d = d
        self.dL = dL
        self.omega = omega

        # Pair indices
        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

        act = nn.GELU()

        # φ branch: per-particle MLP
        self.phi = self._build_mlp(d, hidden_dim, dL, n_layers, act)

        # ψ branch: pair MLP on 6 safe radial features
        self.psi = self._build_mlp(6, hidden_dim, dL, n_layers, act)

        # ρ readout: [φ_mean, ψ_mean, extras, τ_emb] → 1
        rho_in_dim = 2 * dL + 2 + tau_embed_dim
        self.rho = self._build_mlp(rho_in_dim, hidden_dim, 1, n_layers, act)

        # τ embedding
        self.tau_emb = SinusoidalEmbedding(tau_embed_dim, tau_n_freq)

        # Gate and feature params (same as PINN)
        self.gate_radius_aho = 0.30
        self.eps_feat_aho = 0.20

        # Analytic cusp
        if d == 1:
            self.gamma_apara = 0.0
            self.gamma_para = 0.0
        else:
            self.gamma_apara = 1.0 / (d - 1)
            self.gamma_para = 1.0 / (d + 1)
        self.cusp_len = 1.0 / (omega**0.5)

        self._init_weights()

    def _build_mlp(self, in_dim, hidden, out_dim, n_layers, act):
        layers = [nn.Linear(in_dim, hidden), act]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act]
        layers += [nn.Linear(hidden, out_dim)]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _safe_pair_features(self, r):
        """r: (B,P,1) → features (B,P,6), s1_mean (B,1)."""
        a_ho = 1.0 / (self.omega**0.5)
        eps = self.eps_feat_aho * a_ho
        r2 = r * r
        rt = torch.sqrt(r2 + eps * eps)
        s1 = torch.log1p((rt / eps) ** 2)
        s2 = r2 / (r2 + eps * eps)
        s3 = (rt / eps) ** 2 * torch.exp(-((rt / eps) ** 2))
        g_rbf = torch.as_tensor([0.25, 1.0, 4.0], device=r.device, dtype=r.dtype).view(1, 1, -1)
        rbf = torch.exp(-g_rbf * s1)
        feat = torch.cat([s1, s2, s3, rbf], dim=-1)
        s1_mean = s1.mean(dim=1, keepdim=False)
        return feat, s1_mean

    def _short_range_gate(self, r):
        a_ho = 1.0 / (self.omega**0.5)
        rg = self.gate_radius_aho * a_ho
        r2 = r * r
        return r2 / (r2 + rg * rg)

    def forward(
        self, x: torch.Tensor, tau: torch.Tensor, spin: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (B, N, d), tau: (B,), spin: (B, N) or None
        Returns: (B, 1)  log-Jastrow + cusp
        """
        B, N, d = x.shape

        # ---- Feature extraction (same as PINN) ----
        x_scaled = x * (self.omega**0.5)
        diff = x_scaled.unsqueeze(2) - x_scaled.unsqueeze(1)
        diff_pairs = diff[:, self.idx_i, self.idx_j, :]
        r2 = (diff_pairs * diff_pairs).sum(dim=-1, keepdim=True)
        r = torch.sqrt(r2 + torch.finfo(x.dtype).eps)

        # φ branch
        phi_out = self.phi(x_scaled.reshape(B * N, d)).reshape(B, N, self.dL)
        phi_mean = phi_out.mean(dim=1)

        # ψ branch
        psi_in, s1_mean = self._safe_pair_features(r)
        psi_out = self.psi(psi_in.reshape(-1, 6)).reshape(B, -1, self.dL)
        gate = self._short_range_gate(r)
        psi_out = psi_out * gate
        psi_mean = psi_out.mean(dim=1)

        # Extras
        r2_mean = (x_scaled**2).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        extras = torch.cat([r2_mean, s1_mean], dim=1)

        # ---- τ conditioning ----
        tau_features = self.tau_emb(tau)  # (B, tau_embed_dim)

        # ---- Readout ----
        rho_in = torch.cat([phi_mean, psi_mean, extras, tau_features], dim=1)
        out = self.rho(rho_in)  # (B, 1)

        # ---- Analytic cusp (τ-independent, on physical coords) ----
        diff_phys = x.unsqueeze(2) - x.unsqueeze(1)
        diff_phys_pairs = diff_phys[:, self.idx_i, self.idx_j, :]
        r2_c = (diff_phys_pairs**2).sum(dim=-1, keepdim=True)
        r_c = torch.sqrt(r2_c + torch.finfo(x.dtype).eps)

        if spin is None:
            up = N // 2
            spin = (
                torch.cat(
                    [
                        torch.zeros(up, dtype=torch.long, device=x.device),
                        torch.ones(N - up, dtype=torch.long, device=x.device),
                    ]
                )
                .unsqueeze(0)
                .expand(B, -1)
            )
        else:
            spin = spin.to(x.device).long()
            if spin.dim() == 1:
                spin = spin.unsqueeze(0).expand(B, -1)

        si, sj = spin[:, self.idx_i], spin[:, self.idx_j]
        same_spin = (si == sj).to(x.dtype).unsqueeze(-1)
        gamma_para = torch.as_tensor(self.gamma_para, dtype=x.dtype, device=x.device)
        gamma_apara = torch.as_tensor(self.gamma_apara, dtype=x.dtype, device=x.device)
        gamma = same_spin * gamma_para + (1.0 - same_spin) * gamma_apara
        pair_u = gamma * r_c * torch.exp(-r_c)
        cusp_term = pair_u.sum(dim=1)

        return out + cusp_term


# ============================================================
# τ-Conditioned CTNN Backflow
# ============================================================
class TauBackflow(nn.Module):
    """
    CTNN-style backflow with τ as input to node features.

    Node input: [x_i * √ω, spin_i, τ_emb]  (broadcast τ_emb to all particles)
    Edge input: [Δr, |r|, |r|²]
    Transport maps: node→edge (ρ_v2e), edge→node (ρ_e2v)
    Edge update: MLP on [h_e, ρ_v2e(h_i), ρ_v2e(h_j)]
    Node update: residual MLP on [h_v, m_v]
    Output: Δx = tanh(dx_head(h_v)) * softplus(scale), zero-COM
    """

    def __init__(
        self,
        d: int,
        omega: float = 1.0,
        *,
        hidden: int = 32,
        layers: int = 2,
        msg_hidden: int = 32,
        msg_layers: int = 2,
        tau_embed_dim: int = 16,
        tau_n_freq: int = 8,
        bf_scale_init: float = 0.3,
    ):
        super().__init__()
        self.d = d
        self.omega = omega
        act = nn.GELU()

        # τ embedding
        self.tau_emb = SinusoidalEmbedding(tau_embed_dim, tau_n_freq)

        # Node embedding: (x_scaled, spin, tau_emb) → hidden
        node_in_dim = d + 1 + tau_embed_dim
        self.node_embed = nn.Linear(node_in_dim, hidden)

        # Edge embedding: (Δr, |r|, |r|²) → msg_hidden
        edge_in_dim = d + 2
        self.edge_embed = self._mlp(edge_in_dim, msg_hidden, msg_hidden, msg_layers, act)

        # Transport maps
        self.rho_v2e = nn.Linear(hidden, msg_hidden, bias=False)
        self.rho_e2v = nn.Linear(msg_hidden, hidden, bias=False)

        # Edge update: [h_e, ρ_v2e(h_i), ρ_v2e(h_j)] → msg_hidden
        self.edge_update = self._mlp(3 * msg_hidden, msg_hidden, msg_hidden, msg_layers, act)

        # Node update: [h_v, m_v] → hidden (residual)
        self.node_update = self._mlp(2 * hidden, hidden, hidden, layers, act)

        # Output head
        self.dx_head = nn.Linear(hidden, d)
        nn.init.zeros_(self.dx_head.weight)
        nn.init.zeros_(self.dx_head.bias)

        # Learnable scale
        self.bf_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(bf_scale_init) - 1.0), dtype=DTYPE)
        )

    def _mlp(self, in_dim, hid, out_dim, n_layers, act):
        if n_layers == 1:
            return nn.Sequential(nn.Linear(in_dim, out_dim))
        layers = [nn.Linear(in_dim, hid), act]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hid, hid), act]
        layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, tau: torch.Tensor, spin: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (B, N, d), tau: (B,), spin: (B, N) or (N,) or None
        Returns: Δx (B, N, d)
        """
        B, N, d = x.shape
        x_sc = x * (self.omega**0.5)

        # τ features broadcast to all particles
        tau_feat = self.tau_emb(tau)  # (B, tau_embed_dim)
        tau_feat_exp = tau_feat.unsqueeze(1).expand(B, N, -1)  # (B, N, tau_embed_dim)

        # Spin features
        if spin is None:
            up = N // 2
            spin = torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=x.device),
                    torch.ones(N - up, dtype=torch.long, device=x.device),
                ]
            )
        if spin.dim() == 1:
            spin_feat = spin.view(1, N, 1).to(x.dtype).expand(B, N, 1)
        else:
            spin_feat = spin.view(B, N, 1).to(x.dtype)

        # Node input
        node_in = torch.cat([x_sc, spin_feat, tau_feat_exp], dim=-1)  # (B, N, d+1+τ)
        h_v = self.node_embed(node_in)  # (B, N, hidden)

        # Edge geometry
        r_vec = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)  # (B, N, N, d)
        r2 = (r_vec**2).sum(dim=-1, keepdim=True)
        r1 = torch.sqrt(r2 + 1e-12)
        edge_in = torch.cat([r_vec, r1, r2], dim=-1)  # (B, N, N, d+2)
        h_e = self.edge_embed(edge_in)  # (B, N, N, msg_hidden)

        # Mask (no self-messages)
        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        mask = 1.0 - eye

        # Node → edge transport
        h_v_i = h_v.unsqueeze(2).expand(B, N, N, -1)
        h_v_j = h_v.unsqueeze(1).expand(B, N, N, -1)
        v_i2e = self.rho_v2e(h_v_i)
        v_j2e = self.rho_v2e(h_v_j)

        # Edge update
        h_e_new = self.edge_update(torch.cat([h_e, v_i2e, v_j2e], dim=-1))

        # Edge → node transport + aggregate
        msgs = self.rho_e2v(h_e_new) * mask  # (B, N, N, hidden)
        m_v = msgs.mean(dim=2)  # (B, N, hidden)

        # Node update (residual)
        delta_h = self.node_update(torch.cat([h_v, m_v], dim=-1))
        h_v = h_v + delta_h

        # Output
        dx = torch.tanh(self.dx_head(h_v))  # (B, N, d)
        dx = dx - dx.mean(dim=1, keepdim=True)  # zero COM shift
        scale = F.softplus(self.bf_scale_raw)
        return dx * scale


# ============================================================
# Full τ-Conditioned Wavefunction
# ============================================================
class TauWavefunction(nn.Module):
    """
    Ψ(x, τ) = SD(x + BF(x,τ)) × exp(J(x,τ))

    The full many-body wavefunction with imaginary time τ as native input.
    """

    def __init__(self, n_particles, dim, omega, C_occ, spin, params, cfg: VMCConfig):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.omega = omega
        self.register_buffer("C_occ", C_occ)
        self.register_buffer("spin", spin)
        self.params = params

        self.jastrow = TauJastrow(
            n_particles,
            dim,
            omega,
            dL=cfg.pinn_dL,
            hidden_dim=cfg.pinn_hidden,
            n_layers=cfg.pinn_layers,
            tau_embed_dim=cfg.tau_embed_dim,
            tau_n_freq=cfg.tau_n_freq,
        )

        self.bf_net = None
        if cfg.use_backflow:
            self.bf_net = TauBackflow(
                d=dim,
                omega=omega,
                hidden=cfg.bf_hidden,
                layers=cfg.bf_layers,
                msg_hidden=cfg.bf_hidden,
                msg_layers=2,
                tau_embed_dim=cfg.tau_embed_dim,
                tau_n_freq=cfg.tau_n_freq,
                bf_scale_init=0.3,
            )

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d), tau: (B,)
        Returns: log|Ψ(x, τ)| as (B,)
        """
        B = x.shape[0]
        spin = self.spin.expand(B, -1)

        x_eff = x
        if self.bf_net is not None:
            dx = self.bf_net(x, tau, spin=spin)
            x_eff = x + dx

        _, log_sd = slater_determinant_closed_shell(
            x_eff,
            self.C_occ,
            params=self.params,
            spin=spin,
        )

        f_jastrow = self.jastrow(x, tau, spin=spin)  # (B, 1)

        return log_sd + f_jastrow.squeeze(-1)


# ============================================================
# Proper potential with soft-core Coulomb
# ============================================================
def compute_potential(
    x: torch.Tensor, omega: float, well_sep: float, smooth_T: float = 0.2, coulomb: bool = True
) -> torch.Tensor:
    """V_ext + V_Coulomb. Soft-core Coulomb for numerical stability."""
    B, N, d = x.shape

    # External potential
    if well_sep <= 1e-10:
        V_ext = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    else:
        R_L = torch.zeros(d, dtype=x.dtype, device=x.device)
        R_R = torch.zeros(d, dtype=x.dtype, device=x.device)
        R_L[0] = -well_sep / 2
        R_R[0] = +well_sep / 2
        V_L = 0.5 * omega**2 * ((x - R_L) ** 2).sum(dim=-1)
        V_R = 0.5 * omega**2 * ((x - R_R) ** 2).sum(dim=-1)
        T = smooth_T
        V_per_particle = -T * torch.logaddexp(-V_L / T, -V_R / T)
        V_ext = V_per_particle.sum(dim=1)

    # Coulomb with soft-core
    V_coul = torch.zeros(B, device=x.device, dtype=x.dtype)
    if coulomb:
        eps_sc = 1e-6 / max(omega, 1e-6) ** 0.5  # small soft-core
        for i in range(N):
            for j in range(i + 1, N):
                r2 = ((x[:, i] - x[:, j]) ** 2).sum(dim=-1)
                rij = torch.sqrt(r2 + eps_sc**2)
                V_coul = V_coul + 1.0 / rij

    return V_ext + V_coul


# ============================================================
# MCMC sampling from |Ψ(x, τ)|²
# ============================================================
def mcmc_sample(
    log_psi_fn,
    n_samples,
    n_particles,
    dim,
    omega,
    well_sep=0.0,
    n_warmup=400,
    step_size=0.5,
    target_acc=0.45,
):
    """Adaptive Metropolis sampling from |Ψ|²."""
    sigma = 1.0 / math.sqrt(omega)
    x = torch.randn(n_samples, n_particles, dim, device=DEVICE, dtype=DTYPE) * sigma

    # Initialise particles near wells for double-well
    if well_sep > 1e-10:
        x[:, 0, 0] = x[:, 0, 0] - well_sep / 2
        if n_particles > 1:
            x[:, 1, 0] = x[:, 1, 0] + well_sep / 2

    log_prob = 2.0 * log_psi_fn(x)
    step = step_size * sigma
    n_accept_total = 0

    for s in range(n_warmup):
        x_new = x + step * torch.randn_like(x)
        log_prob_new = 2.0 * log_psi_fn(x_new)
        accept = torch.log(torch.rand(n_samples, device=DEVICE, dtype=DTYPE)) < (
            log_prob_new - log_prob
        )
        x = torch.where(accept.view(-1, 1, 1), x_new, x)
        log_prob = torch.where(accept, log_prob_new, log_prob)

        # Adaptive step size
        acc_rate = accept.float().mean().item()
        if s < n_warmup - 50:
            if acc_rate > target_acc + 0.05:
                step *= 1.05
            elif acc_rate < target_acc - 0.05:
                step *= 0.95
        if s >= n_warmup - 50:
            n_accept_total += acc_rate

    avg_acc = n_accept_total / 50
    return x.detach(), avg_acc


# ============================================================
# Local energy computation
# ============================================================
def compute_local_energy(
    wf: TauWavefunction,
    x: torch.Tensor,
    tau: torch.Tensor,
    omega: float,
    well_sep: float,
    smooth_T: float = 0.2,
    coulomb: bool = True,
):
    """
    Compute E_L(x, τ) = -½(∇²logΨ + |∇logΨ|²) + V(x)

    Returns: E_L (B,), grad_log_psi (B, N, d)
    All with create_graph=True for PDE backprop.
    """
    B, N, d = x.shape
    x = x.detach().requires_grad_(True)

    log_psi = wf(x, tau)

    # ∇_x log Ψ
    grad_log = torch.autograd.grad(log_psi.sum(), x, create_graph=True, retain_graph=True)[
        0
    ]  # (B, N, d)

    # ∇² log Ψ (exact Laplacian)
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            d2 = torch.autograd.grad(
                grad_log[:, i, j].sum(), x, create_graph=True, retain_graph=True
            )[0]
            lap = lap + d2[:, i, j]

    # Kinetic energy
    grad_sq = (grad_log**2).sum(dim=(1, 2))
    T = -0.5 * (lap + grad_sq)

    # Potential energy (no grad needed — V doesn't depend on network weights)
    with torch.no_grad():
        V = compute_potential(x, omega, well_sep, smooth_T, coulomb)

    E_L = T + V
    return E_L, grad_log, log_psi


# ============================================================
# Phase 1: VMC ground state at τ = τ_max
# ============================================================
def train_vmc(cfg: VMCConfig):
    """Standard VMC training with τ fixed at τ_max."""
    C_occ, spin, params = setup_sd(cfg.n_particles, cfg.dim, cfg.omega, cfg.E_ref)

    wf = TauWavefunction(cfg.n_particles, cfg.dim, cfg.omega, C_occ, spin, params, cfg).to(DEVICE)

    n_params = sum(p.numel() for p in wf.parameters() if p.requires_grad)
    print(f"  τ-Conditioned WF: {n_params:,} parameters")

    optimizer = optim.Adam(wf.parameters(), lr=cfg.lr_vmc)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.n_epochs_vmc, eta_min=cfg.lr_vmc / 10
    )

    print_every = max(cfg.n_epochs_vmc // 10, 1)
    best_E, best_state = float("inf"), None
    tau_fixed = torch.full((cfg.n_samples_vmc,), cfg.tau_max, dtype=DTYPE, device=DEVICE)

    for epoch in range(cfg.n_epochs_vmc):
        # MCMC from |Ψ(x, τ_max)|²
        with torch.no_grad():
            log_fn = lambda x_: wf(x_, tau_fixed[: x_.shape[0]])
            x, acc = mcmc_sample(
                log_fn,
                cfg.n_samples_vmc,
                cfg.n_particles,
                cfg.dim,
                cfg.omega,
                cfg.well_sep,
                n_warmup=300,
                step_size=0.5,
            )

        tau_batch = torch.full((x.shape[0],), cfg.tau_max, dtype=DTYPE, device=DEVICE)
        E_L, grad_log, log_psi = compute_local_energy(
            wf, x, tau_batch, cfg.omega, cfg.well_sep, cfg.smooth_T, cfg.coulomb
        )

        E_mean = E_L.mean()
        loss = ((E_L - E_mean.detach()) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wf.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        E_val = E_mean.item()
        if E_val < best_E:
            best_E = E_val
            best_state = {k: v.clone() for k, v in wf.state_dict().items()}

        if epoch % print_every == 0:
            E_err = E_L.detach().std().item() / math.sqrt(cfg.n_samples_vmc)
            print(
                f"    [VMC] Ep {epoch:4d}/{cfg.n_epochs_vmc} | "
                f"E={E_val:.5f}±{E_err:.5f} | var={loss.item():.4f} | acc={acc:.2f}"
            )

    if best_state is not None:
        wf.load_state_dict(best_state)

    # Final energy estimate with more samples
    with torch.no_grad():
        tau_eval = torch.full((2000,), cfg.tau_max, dtype=DTYPE, device=DEVICE)
        log_fn = lambda x_: wf(x_, tau_eval[: x_.shape[0]])
        x_final, _ = mcmc_sample(
            log_fn,
            2000,
            cfg.n_particles,
            cfg.dim,
            cfg.omega,
            cfg.well_sep,
            n_warmup=500,
            step_size=0.5,
        )

    tau_final = torch.full((2000,), cfg.tau_max, dtype=DTYPE, device=DEVICE)
    E_L_final, _, _ = compute_local_energy(
        wf, x_final, tau_final, cfg.omega, cfg.well_sep, cfg.smooth_T, cfg.coulomb
    )
    E_final = E_L_final.detach().mean().item()
    E_err = E_L_final.detach().std().item() / math.sqrt(2000)
    print(f"  VMC converged: E₀ = {E_final:.5f} ± {E_err:.5f}  (ref={cfg.E_ref:.5f})")

    return wf, E_final, E_err


# ============================================================
# Pre-compute ground-state samples for Phase 2
# ============================================================
def precompute_pool(wf: TauWavefunction, cfg: VMCConfig) -> torch.Tensor:
    """Sample large pool of x from |Ψ(x, τ_max)|² for PDE training."""
    n = cfg.n_precompute
    print(f"  Pre-computing pool of {n} walkers...")
    with torch.no_grad():
        tau_pool = torch.full((n,), cfg.tau_max, dtype=DTYPE, device=DEVICE)
        log_fn = lambda x_: wf(x_, tau_pool[: x_.shape[0]])
        x_pool, acc = mcmc_sample(
            log_fn,
            n,
            cfg.n_particles,
            cfg.dim,
            cfg.omega,
            cfg.well_sep,
            n_warmup=600,
            step_size=0.5,
        )
    print(f"  Pool: {n} samples, acc={acc:.2f}")
    return x_pool


# ============================================================
# Initial condition perturbation
# ============================================================
def make_perturbation_fn(cfg: VMCConfig):
    """Return p(x) target for logΨ(x,0) - logΨ(x,τ_max)."""
    A = cfg.ic_amplitude
    if cfg.ic_type == "dipole":

        def p(x):
            return A * x[:, :, 0].mean(dim=1)  # X_cm

        return p, "dipole (X_cm)"
    elif cfg.ic_type == "quadrupole":

        def p(x):
            return A * (x**2).sum(dim=(1, 2)) / cfg.n_particles

        return p, "quadrupole (R²/N)"
    else:
        raise ValueError(f"Unknown IC type: {cfg.ic_type}")


# ============================================================
# Phase 2: PDE training
# ============================================================
def train_pde(cfg: VMCConfig, wf: TauWavefunction, E0: float, x_pool: torch.Tensor):
    """
    Train the τ-conditioned wavefunction on the imaginary-time PDE:
        ∂_τ log Ψ + E_L(τ) - E₀ = 0

    with IC: logΨ(x,0) - logΨ(x,τ_max) ≈ p(x)

    Strategy: Differential LR — τ-specific params (embedding, readout with τ,
    node_embed with τ) train at full LR; base spatial weights at 10× lower LR.
    This keeps the ground state stable while learning τ-dependence.
    """
    perturbation_fn, ic_desc = make_perturbation_fn(cfg)
    print(f"  IC: {ic_desc}, amplitude={cfg.ic_amplitude}")

    # Separate param groups: τ-specific vs base spatial
    tau_params, base_params = [], []
    tau_keywords = {"tau_emb", "rho"}  # readout has τ input
    if wf.bf_net is not None:
        tau_keywords.add("node_embed")  # node_embed has τ input

    for name, param in wf.named_parameters():
        is_tau = any(kw in name for kw in tau_keywords)
        if is_tau:
            tau_params.append(param)
        else:
            base_params.append(param)

    n_tau = sum(p.numel() for p in tau_params)
    n_base = sum(p.numel() for p in base_params)
    print(f"  Params: {n_tau:,} τ-specific + {n_base:,} base spatial")

    optimizer = optim.Adam(
        [
            {"params": tau_params, "lr": cfg.lr_pde},
            {"params": base_params, "lr": cfg.lr_pde * 0.1},  # 10× lower LR for base
        ]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.n_epochs_pde, eta_min=cfg.lr_pde / 30
    )

    pool_size = x_pool.shape[0]
    print_every = max(cfg.n_epochs_pde // 20, 1)
    history = {"pde": [], "ic": [], "vmc": [], "total": []}

    # Save ground-state checkpoint for fallback
    gs_state = {k: v.clone() for k, v in wf.state_dict().items()}

    for epoch in range(cfg.n_epochs_pde):
        # --- Sample batch from pool ---
        idx = torch.randint(pool_size, (cfg.batch_pde,))
        x_batch = x_pool[idx]

        # --- Random τ (biased: mixture of uniform + near-zero + near-τ_max) ---
        B = cfg.batch_pde
        n_uniform = B // 2
        n_early = B // 4
        n_late = B - n_uniform - n_early
        tau_uniform = torch.rand(n_uniform, dtype=DTYPE, device=DEVICE) * cfg.tau_max
        tau_early = torch.rand(n_early, dtype=DTYPE, device=DEVICE) ** 2 * cfg.tau_max * 0.3
        tau_late = (
            cfg.tau_max - torch.rand(n_late, dtype=DTYPE, device=DEVICE) ** 2 * cfg.tau_max * 0.2
        )
        tau_pde = torch.cat([tau_uniform, tau_early, tau_late])

        # --- PDE loss: ∂_τ logΨ + E_L - E₀ = 0 ---
        x_pde = x_batch.detach().requires_grad_(True)
        tau_pde = tau_pde.detach().requires_grad_(True)

        log_psi = wf(x_pde, tau_pde)

        # ∂_τ logΨ  (keep graph → loss.backward flows through here)
        dlog_dtau = torch.autograd.grad(
            log_psi.sum(), tau_pde, create_graph=True, retain_graph=True
        )[
            0
        ]  # (B,)

        # E_L computation (detached — avoids 3rd-order)
        dlog_dx = torch.autograd.grad(log_psi.sum(), x_pde, create_graph=True, retain_graph=True)[
            0
        ]  # (B, N, d)

        N, d = cfg.n_particles, cfg.dim
        lap = torch.zeros(B, device=DEVICE, dtype=DTYPE)
        for i in range(N):
            for j in range(d):
                d2 = torch.autograd.grad(dlog_dx[:, i, j].sum(), x_pde, retain_graph=True)[0]
                lap = lap + d2[:, i, j]

        grad_sq = (dlog_dx.detach() ** 2).sum(dim=(1, 2))
        T = -0.5 * (lap.detach() + grad_sq)
        with torch.no_grad():
            V = compute_potential(x_pde, cfg.omega, cfg.well_sep, cfg.smooth_T, cfg.coulomb)
        E_L = T + V

        residual = dlog_dtau + E_L - E0

        # Outlier clipping: clip extreme residuals to prevent divergence
        res_clip = 5.0 * max(abs(E0), 1.0)
        residual = torch.clamp(residual, -res_clip, res_clip)
        L_pde = (residual**2).mean()

        # --- IC loss: logΨ(x,0) - logΨ(x,τ_max) ≈ p(x) ---
        tau_zero = torch.zeros(cfg.batch_pde, dtype=DTYPE, device=DEVICE)
        tau_end = torch.full((cfg.batch_pde,), cfg.tau_max, dtype=DTYPE, device=DEVICE)
        x_ic = x_pool[torch.randint(pool_size, (cfg.batch_pde,))]

        log_psi_0 = wf(x_ic, tau_zero)
        log_psi_T = wf(x_ic, tau_end)
        p_target = perturbation_fn(x_ic)
        delta_log = log_psi_0 - log_psi_T
        L_ic = ((delta_log - p_target) ** 2).mean()

        # --- VMC anchor at τ_max (every step to keep ground state) ---
        L_vmc = torch.zeros(1, device=DEVICE, dtype=DTYPE)
        if epoch % cfg.pde_vmc_freq == 0:
            idx_vmc = torch.randint(pool_size, (min(256, cfg.batch_pde),))
            x_vmc = x_pool[idx_vmc]
            tau_vmc = torch.full((x_vmc.shape[0],), cfg.tau_max, dtype=DTYPE, device=DEVICE)
            E_L_vmc, _, _ = compute_local_energy(
                wf, x_vmc, tau_vmc, cfg.omega, cfg.well_sep, cfg.smooth_T, cfg.coulomb
            )
            E_mean_vmc = E_L_vmc.mean()
            L_vmc = ((E_L_vmc - E_mean_vmc.detach()) ** 2).mean()

        # --- Weight regularization: penalise drift from ground state ---
        L_wreg = torch.zeros(1, device=DEVICE, dtype=DTYPE)
        if cfg.lambda_reg > 0:
            for name, param in wf.named_parameters():
                if name in gs_state:
                    L_wreg = L_wreg + ((param - gs_state[name].detach()) ** 2).sum()
            L_wreg = L_wreg * cfg.lambda_reg

        # --- Curriculum ---
        progress = epoch / max(cfg.n_epochs_pde - 1, 1)
        w_ic = cfg.lambda_ic * max(1.0 - 0.5 * progress, 0.2)
        w_pde = 1.0 + 4.0 * min(progress * 2, 1.0)  # gentler ramp than before

        loss = w_pde * L_pde + w_ic * L_ic + cfg.lambda_vmc * L_vmc + L_wreg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wf.parameters(), 0.5)  # tighter clipping
        optimizer.step()
        scheduler.step()

        if epoch % print_every == 0:
            history["pde"].append(L_pde.item())
            history["ic"].append(L_ic.item())
            history["vmc"].append(L_vmc.item())
            history["total"].append(loss.item())
            print(
                f"    [PDE] Ep {epoch:5d}/{cfg.n_epochs_pde} | "
                f"PDE={L_pde.item():.6f} IC={L_ic.item():.6f} "
                f"VMC={L_vmc.item():.4f} | "
                f"E_L_mean={E_L.detach().mean().item():.4f}"
            )

    print(f"  Final PDE loss: {history['pde'][-1]:.6f}")
    return history


# ============================================================
# Phase 3: Evaluate E(τ) via direct MCMC
# ============================================================
def evaluate_trajectory(wf: TauWavefunction, cfg: VMCConfig, E0: float):
    """
    At each τ, run MCMC from |Ψ(x,τ)|² and compute E(τ) = ⟨E_L⟩.
    No importance weights — direct sampling.
    """
    wf.eval()
    tau_values = np.concatenate([[0.0], np.geomspace(0.01, cfg.tau_max, cfg.n_tau_eval - 1)])
    results = []
    n = cfg.n_samples_eval

    for tau_val in tau_values:
        tau_t = torch.full((n,), tau_val, dtype=DTYPE, device=DEVICE)

        # MCMC from |Ψ(x, τ)|²
        with torch.no_grad():
            log_fn = lambda x_: wf(x_, tau_t[: x_.shape[0]])
            x_samp, acc = mcmc_sample(
                log_fn,
                n,
                cfg.n_particles,
                cfg.dim,
                cfg.omega,
                cfg.well_sep,
                n_warmup=cfg.mcmc_warmup_eval,
                step_size=0.5,
            )

        # Compute E_L at these samples
        E_L, _, log_psi = compute_local_energy(
            wf, x_samp, tau_t, cfg.omega, cfg.well_sep, cfg.smooth_T, cfg.coulomb
        )
        E_L_np = E_L.detach().numpy()
        E_mean = float(E_L_np.mean())
        E_std = float(E_L_np.std())
        E_err = E_std / math.sqrt(n)

        results.append(
            {
                "tau": float(tau_val),
                "E": E_mean,
                "E_std": E_std,
                "E_err": E_err,
                "acc": float(acc),
            }
        )

    wf.train()
    return results


# ============================================================
# Plotting
# ============================================================
def plot_results(traj, fits, cfg, E_ref, save_dir, tag=""):
    tau = np.array([r["tau"] for r in traj])
    E = np.array([r["E"] for r in traj])
    E_err = np.array([r["E_err"] for r in traj])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: E(τ)
    ax = axes[0]
    ax.errorbar(tau, E, yerr=E_err, fmt="o", markersize=3, color="C0", label="E(τ)")
    fit_s = fits.get("exp", {})
    if fit_s.get("success"):
        t_fit = np.linspace(0, tau.max(), 200)
        E_fit = fit_s["E0"] + fit_s["dE"] * np.exp(-fit_s["gamma"] * t_fit)
        ax.plot(
            t_fit,
            E_fit,
            "--",
            color="C3",
            lw=2,
            label=f"Fit: gap={fit_s['gap']:.4f}±{fit_s['gap_err']:.4f}",
        )
    ax.axhline(E_ref, color="gray", ls=":", alpha=0.5, label=f"E₀={E_ref:.4f}")
    ax.set_xlabel("τ")
    ax.set_ylabel("E(τ)")
    ax.set_title(f"d={cfg.well_sep:.1f}, ω={cfg.omega:.1f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: log(E - E₀) for gap extraction
    ax = axes[1]
    if fit_s.get("success"):
        dE = E - fit_s["E0"]
        mask = dE > 0
        if mask.sum() > 2:
            ax.semilogy(tau[mask], dE[mask], "o", ms=3, color="C0")
            t_fit = np.linspace(0, tau.max(), 200)
            dE_fit = fit_s["dE"] * np.exp(-fit_s["gamma"] * t_fit)
            ax.semilogy(t_fit, dE_fit, "--", color="C3", lw=2)
    ax.set_xlabel("τ")
    ax.set_ylabel("E(τ) - E₀")
    ax.set_title("Exponential decay")
    ax.grid(True, alpha=0.3)

    # Panel 3: MCMC acceptance rate
    ax = axes[2]
    acc_vals = [r.get("acc", 0) for r in traj]
    ax.plot(tau, acc_vals, "o-", ms=3, color="C2")
    ax.set_xlabel("τ")
    ax.set_ylabel("MCMC acceptance")
    ax.set_title("Sampling quality")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    name = f"vmc_{tag}d{cfg.well_sep:.1f}_w{cfg.omega:.1f}.png"
    path = save_dir / name
    plt.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"  Figure: {path}")


# ============================================================
# Run single configuration
# ============================================================
def run_single(cfg: VMCConfig, tag="") -> dict:
    coul_str = "interacting" if cfg.coulomb else "NON-interacting"
    print(f"\n{'=' * 65}")
    print(f"  {coul_str}: d={cfg.well_sep:.1f}, ω={cfg.omega:.1f}, E_ref={cfg.E_ref:.5f}")
    print("  τ-conditioned SD × BF(τ) × PINN(τ) wavefunction")
    print(f"{'=' * 65}")

    # Phase 1: VMC ground state
    t0 = time.time()
    print("\n  Phase 1: VMC ground state at τ=τ_max...")
    wf, E_vmc, E_vmc_err = train_vmc(cfg)
    t_vmc = time.time() - t0
    print(f"  Phase 1 time: {t_vmc:.0f}s")

    E_ref = E_vmc if cfg.well_sep > 0.01 else cfg.E_ref
    print(f"  Using E₀ = {E_ref:.5f}")

    # Pre-compute sample pool
    t0 = time.time()
    print("\n  Pre-computing sample pool...")
    x_pool = precompute_pool(wf, cfg)
    t_pre = time.time() - t0
    print(f"  Pre-compute time: {t_pre:.0f}s")

    # Phase 2: PDE training
    t0 = time.time()
    print(f"\n  Phase 2: PDE training ({cfg.n_epochs_pde} epochs)...")
    history = train_pde(cfg, wf, E_ref, x_pool)
    t_pde = time.time() - t0
    print(f"  Phase 2 time: {t_pde:.0f}s")

    # Phase 3: Evaluate
    t0 = time.time()
    print(f"\n  Phase 3: Evaluating E(τ) via direct MCMC ({cfg.n_tau_eval} τ points)...")
    traj = evaluate_trajectory(wf, cfg, E_ref)
    t_eval = time.time() - t0
    print(f"  Phase 3 time: {t_eval:.0f}s")

    # Print trajectory
    print(f"\n  {'tau':>8s}  {'E':>10s}  {'E_err':>10s}  {'acc':>6s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 6}")
    for pt in traj:
        print(f"  {pt['tau']:8.4f}  {pt['E']:10.5f}  {pt['E_err']:10.5f}  {pt['acc']:6.2f}")

    # Fitting
    print("\n  Fitting...")
    fit_s = fit_single_exponential(traj, E_ref)
    fit_d = fit_double_exponential(traj, E_ref)
    fit_r = fit_restricted_exponential(traj, E_ref, tau_min=0.15, tau_max=3.0)
    E0_for_log = np.mean([r["E"] for r in traj[-5:]])
    fit_ll = fit_log_linear(traj, E0_for_log)
    fit_ll_vmc = fit_log_linear(traj, E_vmc, tau_min=0.1, tau_max=2.5)
    fit_ll_win = fit_log_linear(traj, E0_for_log, tau_min=0.15, tau_max=2.5)
    fit_opt = fit_optimal_E0(traj, E_ref)
    fit_scan = fit_scan_E0(traj, E_ref, E_vmc)
    fit_best = fit_best_estimate(
        {
            "exp": fit_s,
            "restricted": fit_r,
            "loglin": fit_ll,
            "loglin_vmc": fit_ll_vmc,
            "scan_E0": fit_scan,
        }
    )

    expected = cfg.omega if cfg.well_sep < 0.01 and cfg.coulomb else None
    if not cfg.coulomb and cfg.well_sep < 0.01:
        expected = cfg.omega  # exact for non-interacting too

    # Report results
    print("\n  =============================================")
    print(
        f"  RESULT: d={cfg.well_sep:.1f}, ω={cfg.omega:.1f}, "
        f"coulomb={'on' if cfg.coulomb else 'off'}"
    )
    print(f"  E_vmc = {E_vmc:.5f} ± {E_vmc_err:.5f}")

    fits = {
        "exp": fit_s,
        "restricted": fit_r,
        "loglin": fit_ll,
        "loglin_vmc": fit_ll_vmc,
        "windowed": fit_ll_win,
        "double": fit_d,
        "optimal_E0": fit_opt,
        "scan_E0": fit_scan,
        "best": fit_best,
    }

    for name, fit in fits.items():
        if not fit.get("success"):
            continue
        g = fit.get("gap", fit.get("gap1", float("nan")))
        ge = fit.get("gap_err", 0)
        err_str = f", err={abs(g - expected) / expected * 100:.2f}%" if expected else ""
        info = ""
        if "n_points" in fit:
            info = f" ({fit['n_points']} pts)"
        if "E0" in fit and "E0_err" in fit:
            print(
                f"  [{name:12s}] E₀={fit['E0']:.5f}±{fit['E0_err']:.5f}, "
                f"gap={g:.4f}±{ge:.4f}{err_str}{info}"
            )
        elif "E0" in fit:
            print(f"  [{name:12s}] E₀={fit['E0']:.5f}, " f"gap={g:.4f}±{ge:.4f}{err_str}{info}")
        else:
            print(f"  [{name:12s}] gap={g:.4f}±{ge:.4f}{err_str}{info}")

    if expected:
        print(f"  Expected = {expected:.4f}")
    else:
        print(f"  (No exact reference for d={cfg.well_sep:.1f})")
    print("  =============================================")

    plot_results(traj, fits, cfg, E_ref, RESULTS_DIR, tag)

    result = {
        "d": cfg.well_sep,
        "omega": cfg.omega,
        "E_ref": E_ref,
        "E_vmc": E_vmc,
        "E_vmc_err": E_vmc_err,
        "coulomb": cfg.coulomb,
        "trajectory": traj,
        "fit_single": fit_s,
        "fit_double": fit_d,
        "fit_restricted": fit_r,
        "fit_log_linear": fit_ll,
        "fit_loglin_vmc": fit_ll_vmc,
        "fit_windowed": fit_ll_win,
        "fit_optimal_E0": fit_opt,
        "fit_scan_E0": fit_scan,
        "fit_best": fit_best,
        "t_vmc": t_vmc,
        "t_pde": t_pde,
        "t_eval": t_eval,
        "ic_type": cfg.ic_type,
        "n_params": sum(p.numel() for p in wf.parameters()),
    }
    return result


# ============================================================
# CLI modes
# ============================================================
def test_free():
    """Non-interacting N=2, ω=1: E_0=2.0, gap=1.0 (exact)."""
    print("\n" + "=" * 70)
    print("VALIDATION: Non-interacting N=2, ω=1.0")
    print("Expected: E₀=2.0, gap=ω=1.0 (exact)")
    print("=" * 70)
    cfg = VMCConfig(
        omega=1.0,
        well_sep=0.0,
        E_ref=2.0,
        coulomb=False,
        tau_max=5.0,
        n_epochs_vmc=800,
        n_samples_vmc=1024,
        lr_vmc=3e-3,
        n_precompute=8192,
        n_epochs_pde=12000,
        batch_pde=256,
        lr_pde=2e-4,
        ic_amplitude=2.0,
        ic_type="dipole",
        lambda_ic=80.0,
        lambda_reg=1e-4,
        lambda_vmc=1.0,
        pde_vmc_freq=3,
        n_tau_eval=60,
        n_samples_eval=6000,
        mcmc_warmup_eval=500,
        use_backflow=False,
        pinn_hidden=64,
        pinn_layers=2,
        pinn_dL=5,
        tau_embed_dim=16,
        tau_n_freq=8,
    )
    result = run_single(cfg, tag="free_")
    _save(result, RESULTS_DIR / "vmc_free.json")
    print(f"\nSaved: {RESULTS_DIR / 'vmc_free.json'}")


def tiny_test():
    """Interacting N=2, ω=1, d=0: E_0=3.0, gap=1.0 (Kohn)."""
    print("\n" + "=" * 70)
    print("TINY: Interacting N=2, ω=1.0, d=0")
    print("Expected: E₀=3.0, gap=ω=1.0 (Kohn theorem)")
    print("=" * 70)
    cfg = VMCConfig(
        omega=1.0,
        well_sep=0.0,
        E_ref=3.0,
        coulomb=True,
        tau_max=5.0,
        n_epochs_vmc=1200,
        n_samples_vmc=1024,
        lr_vmc=3e-3,
        n_precompute=8192,
        n_epochs_pde=12000,
        batch_pde=256,
        lr_pde=2e-4,
        ic_amplitude=2.0,
        ic_type="dipole",
        lambda_ic=80.0,
        lambda_reg=1e-4,
        lambda_vmc=1.0,
        pde_vmc_freq=3,
        n_tau_eval=60,
        n_samples_eval=6000,
        mcmc_warmup_eval=500,
        use_backflow=True,
        pinn_hidden=64,
        pinn_layers=2,
        pinn_dL=5,
        bf_hidden=32,
        bf_layers=2,
        tau_embed_dim=16,
        tau_n_freq=8,
    )
    result = run_single(cfg, tag="tiny_")
    _save(result, RESULTS_DIR / "vmc_tiny.json")
    print(f"\nSaved: {RESULTS_DIR / 'vmc_tiny.json'}")


def full_test():
    """Converged interacting test."""
    print("\n" + "=" * 70)
    print("FULL: Interacting N=2, ω=1.0, d=0")
    print("=" * 70)
    cfg = VMCConfig(
        omega=1.0,
        well_sep=0.0,
        E_ref=3.0,
        coulomb=True,
        tau_max=5.0,
        n_epochs_vmc=2000,
        n_samples_vmc=1024,
        lr_vmc=3e-3,
        n_precompute=16384,
        n_epochs_pde=15000,
        batch_pde=256,
        lr_pde=3e-4,
        ic_amplitude=2.0,
        ic_type="dipole",
        lambda_ic=80.0,
        lambda_reg=5e-5,
        lambda_vmc=1.0,
        pde_vmc_freq=3,
        n_tau_eval=60,
        n_samples_eval=6000,
        mcmc_warmup_eval=500,
        use_backflow=True,
        pinn_hidden=64,
        pinn_layers=2,
        pinn_dL=5,
        bf_hidden=32,
        bf_layers=2,
        tau_embed_dim=16,
        tau_n_freq=8,
    )
    result = run_single(cfg, tag="full_")
    _save(result, RESULTS_DIR / "vmc_full.json")
    print(f"\nSaved: {RESULTS_DIR / 'vmc_full.json'}")


def sweep_distances():
    """Distance sweep at ω=1.0."""
    print("\n" + "=" * 70)
    print("SWEEP: ω=1.0, d=[0, 1, 2, 4]")
    print("=" * 70)
    all_results = []

    for d_val in [0.0, 1.0, 2.0, 4.0]:
        # Scale VMC effort with distance
        extra_vmc = int(400 * min(d_val, 2.0))
        cfg = VMCConfig(
            omega=1.0,
            well_sep=d_val,
            E_ref=3.0,
            coulomb=True,
            tau_max=5.0,
            n_epochs_vmc=1200 + extra_vmc,
            n_samples_vmc=1024,
            lr_vmc=3e-3,
            n_precompute=8192,
            n_epochs_pde=12000,
            batch_pde=256,
            lr_pde=2e-4,
            ic_amplitude=2.0,
            ic_type="dipole",
            lambda_ic=80.0,
            lambda_reg=1e-4,
            lambda_vmc=1.0,
            pde_vmc_freq=3,
            n_tau_eval=60,
            n_samples_eval=6000,
            mcmc_warmup_eval=500,
            use_backflow=True,
            pinn_hidden=64,
            pinn_layers=2,
            pinn_dL=5,
            bf_hidden=32,
            bf_layers=2,
            tau_embed_dim=16,
            tau_n_freq=8,
        )
        result = run_single(cfg, tag=f"sweep_d{d_val:.0f}_")
        all_results.append(result)

    _save(all_results, RESULTS_DIR / "vmc_sweep.json")

    # Summary table
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY (τ-Conditioned VMC)")
    print("=" * 70)
    print(
        f"  {'d':>4s}  {'E_vmc':>10s}  {'gap_exp':>9s}  {'gap_scan':>9s}  {'gap_ll':>9s}  {'gap_best':>10s}"
    )
    print(f"  {'-' * 4}  {'-' * 10}  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 10}")
    for r in all_results:
        ge = r.get("fit_single", {}).get("gap", float("nan"))
        gs = r.get("fit_scan_E0", {}).get("gap", float("nan"))
        gl = r.get("fit_log_linear", {}).get("gap", float("nan"))
        gb = r.get("fit_best", {}).get("gap", float("nan"))
        print(f"  {r['d']:4.1f}  {r['E_vmc']:10.5f}  {ge:9.4f}  {gs:9.4f}  {gl:9.4f}  {gb:10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Imaginary-Time VMC with τ-Conditioned Wavefunction")
    p.add_argument("--test_free", action="store_true", help="Non-interacting validation")
    p.add_argument("--tiny", action="store_true", help="Quick interacting test")
    p.add_argument("--full", action="store_true", help="Converged interacting test")
    p.add_argument("--sweep", action="store_true", help="Distance sweep d=[0,1,2,4]")
    args = p.parse_args()

    if args.test_free:
        test_free()
    elif args.tiny:
        tiny_test()
    elif args.full:
        full_test()
    elif args.sweep:
        sweep_distances()
    else:
        test_free()
