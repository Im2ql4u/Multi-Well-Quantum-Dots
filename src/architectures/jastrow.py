from __future__ import annotations

import torch
import torch.nn as nn

from PINN import PINN  # noqa: F401


class CuspMixin:
    """Shared analytic electron-electron cusp utilities for Jastrow models."""

    def _init_cusps(self, n_particles: int, d: int, omega: float) -> None:
        self._n_particles_cusp = int(n_particles)
        self._d_cusp = int(d)
        self._omega_cusp = float(omega)
        self.gamma_apara = 1 / (d - 1)
        self.gamma_para = 1 / (d + 1)
        self.cusp_len = 1 / omega**0.5
        (ii, jj) = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("_cusp_idx_i", ii, persistent=False)
        self.register_buffer("_cusp_idx_j", jj, persistent=False)

    def _compute_cusps(self, x: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        (batch_size, n_particles, _) = x.shape
        diff = x[:, self._cusp_idx_i, :] - x[:, self._cusp_idx_j, :]
        r = torch.sqrt((diff**2).sum(-1, keepdim=True) + 1e-30)
        # Determine same-spin vs opposite-spin pairs
        si = spin[self._cusp_idx_i] if spin.ndim == 1 else spin[:, self._cusp_idx_i]
        sj = spin[self._cusp_idx_j] if spin.ndim == 1 else spin[:, self._cusp_idx_j]
        same_sp = (si == sj).float()
        if same_sp.ndim == 1:
            same_sp = same_sp.unsqueeze(0).expand(batch_size, -1)
        gamma = same_sp * self.gamma_para + (1 - same_sp) * self.gamma_apara
        if gamma.ndim == 2:
            gamma = gamma.unsqueeze(-1)
        cusp = gamma * r / (1.0 + r / self.cusp_len)
        return cusp.sum(dim=(-2, -1))


class CTNNJastrow(CuspMixin, nn.Module):
    """Continuous-filter message-passing Jastrow factor.

    Reconstructed from bytecode; simplified but interface-compatible.
    """

    def __init__(
        self,
        n_particles: int = 2,
        d: int = 2,
        omega: float = 1.0,
        node_hidden: int = 32,
        edge_hidden: int = 32,
        n_mp_steps: int = 2,
        msg_layers: int = 2,
        node_layers: int = 2,
        readout_hidden: int = 16,
        readout_layers: int = 2,
        act: str = "tanh",
        aggregation: str = "sum",
        use_spin: bool = False,
    ) -> None:
        super().__init__()
        self._init_cusps(n_particles, d, omega)
        self.n_particles = n_particles
        self.d = d
        self.omega = omega
        self.n_mp_steps = n_mp_steps
        self.use_spin = use_spin
        self.aggregation = aggregation

        _act: nn.Module = nn.Tanh()
        node_in_dim = d + 1 if use_spin else d
        edge_in_dim = 1 + 2 * node_hidden

        self.node_embed = nn.Linear(node_in_dim, node_hidden, bias=False)
        self.edge_embed = nn.Linear(edge_in_dim, edge_hidden, bias=False)

        self.rho_v_to_e = nn.ModuleList()
        self.edge_updates = nn.ModuleList()
        self.rho_e_to_v = nn.ModuleList()
        self.node_updates = nn.ModuleList()

        for _ in range(n_mp_steps):
            self.rho_v_to_e.append(nn.Linear(node_hidden, edge_hidden, bias=False))
            self.edge_updates.append(
                self._mlp(edge_hidden * 3, edge_hidden, edge_hidden, msg_layers, _act)
            )
            self.rho_e_to_v.append(nn.Linear(edge_hidden, node_hidden, bias=False))
            self.node_updates.append(
                self._mlp(node_hidden * 2, node_hidden, node_hidden, node_layers, _act)
            )

        # Initialize with small weights
        for mod in [self.node_embed, self.edge_embed]:
            nn.init.normal_(mod.weight, std=0.001)

        # Readout head
        f_layers: list[nn.Module] = [nn.Linear(node_hidden, readout_hidden), _act]
        for _ in range(readout_layers - 2):
            f_layers.extend([nn.Linear(readout_hidden, readout_hidden), _act])
        f_layers.append(nn.Linear(readout_hidden, 1))
        self.f_head = nn.Sequential(*f_layers)

        (ii, jj) = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

    def _mlp(
        self, in_dim: int, hid: int, out_dim: int, n_layers: int, act_fn: nn.Module
    ) -> nn.Module:
        mods: list[nn.Module] = [nn.Linear(in_dim, hid), act_fn]
        for _ in range(n_layers - 2):
            mods.extend([nn.Linear(hid, hid), act_fn])
        mods.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*mods)

    def _aggregate(self, msgs: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        elif self.aggregation == "mean":
            return msgs.mean(dim=2)
        return msgs.max(dim=2).values

    def _resolve_spin(
        self, spin: torch.Tensor, batch_size: int, n_particles: int, device: torch.device
    ) -> torch.Tensor:
        if spin.dim() == 2:
            return spin.unsqueeze(-1).expand(batch_size, n_particles, 1)
        return spin.unsqueeze(0).unsqueeze(-1).expand(batch_size, n_particles, 1).to(device)

    def forward(self, x: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        batch_size, n_particles, _ = x.shape
        x_sc = x * self.omega**0.5

        if self.use_spin:
            spin_feat = self._resolve_spin(spin, batch_size, n_particles, x.device)
            node_in = torch.cat([x_sc, spin_feat.to(dtype=x.dtype)], dim=-1)
        else:
            node_in = x_sc

        h_v = self.node_embed(node_in)

        # Dense edge features
        r = x_sc.unsqueeze(1) - x_sc.unsqueeze(2)
        r2 = (r * r).sum(dim=-1, keepdim=True)
        r1 = torch.sqrt(r2 + 1e-12)

        h_v_i = h_v.unsqueeze(2).expand(-1, -1, n_particles, -1)
        h_v_j = h_v.unsqueeze(1).expand(-1, n_particles, -1, -1)
        edge_in = torch.cat([r1, h_v_i, h_v_j], dim=-1)
        h_e = self.edge_embed(edge_in)

        # Message passing steps
        eye = torch.eye(n_particles, device=x.device, dtype=x.dtype)
        for step in range(self.n_mp_steps):
            v_to_e = self.rho_v_to_e[step](h_v)
            v_i_to_e = v_to_e.unsqueeze(2).expand_as(h_e)
            v_j_to_e = v_to_e.unsqueeze(1).expand_as(h_e)
            edge_upd_in = torch.cat([h_e, v_i_to_e, v_j_to_e], dim=-1)
            h_e = self.edge_updates[step](edge_upd_in)

            e_to_v = self.rho_e_to_v[step](h_e)
            m_v = self._aggregate(e_to_v)
            node_upd_in = torch.cat([h_v, m_v], dim=-1)
            h_v = self.node_updates[step](node_upd_in)

        # Readout: per-particle contribution + pair distance features
        h_v_mean = h_v.mean(dim=1)
        r_pairs = x[:, self.idx_i, :] - x[:, self.idx_j, :]
        r_phys = torch.sqrt((r_pairs**2).sum(dim=-1) + 1e-12)
        pair_feat = torch.log1p(r_phys).mean(dim=-1, keepdim=True)
        f_in = torch.cat([h_v_mean, pair_feat], dim=-1) if pair_feat.shape[-1] > 0 else h_v_mean
        f_nn = self.f_head(h_v_mean)  # (B, 1)

        # Cusp contribution
        cusp = 0.2 * self._compute_cusps(x, spin)
        return f_nn + cusp.unsqueeze(-1)
