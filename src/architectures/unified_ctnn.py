from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedCTNN(nn.Module):
    """Unified continuous-filter message-passing network emitting both
    backflow displacements *dx* and a Jastrow scalar *f*.

    Reconstructed from bytecode; simplified but interface-compatible.
    """

    def __init__(
        self,
        d: int = 2,
        n_particles: int = 2,
        omega: float = 1.0,
        node_hidden: int = 32,
        edge_hidden: int = 32,
        msg_layers: int = 2,
        node_layers: int = 2,
        n_mp_steps: int = 2,
        act: str = "tanh",
        aggregation: str = "sum",
        use_spin: bool = False,
        same_spin_only: bool = False,
        out_bound: str = "tanh",
        bf_scale_init: float = 1.0,
        zero_init_last: bool = False,
        jastrow_hidden: int = 16,
        jastrow_layers: int = 2,
        envelope_width_aho: float = 1.0,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_particles = n_particles
        self.omega = float(omega)
        self.envelope_width_aho = envelope_width_aho
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only
        self.aggregation = aggregation
        self.out_bound = out_bound
        self.n_mp_steps = n_mp_steps

        _act: nn.Module = nn.Tanh()

        (ii, jj) = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

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

        # Backflow head
        self.dx_head = nn.Linear(node_hidden, d, bias=False)
        self.bf_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(bf_scale_init) - 1.0))
        )

        if zero_init_last:
            nn.init.zeros_(self.dx_head.weight)
            if self.dx_head.bias is not None:
                nn.init.zeros_(self.dx_head.bias)

        # Jastrow head
        f_in_dim = node_hidden + edge_hidden + 1  # h_v_mean + h_e_mean + r2_mean
        layers_f: list[nn.Module] = [nn.Linear(f_in_dim, jastrow_hidden), _act]
        dim_f = jastrow_hidden
        for _ in range(jastrow_layers - 2):
            layers_f.extend([nn.Linear(dim_f, dim_f), _act])
        layers_f.append(nn.Linear(dim_f, 1))
        self.f_head = nn.Sequential(*layers_f)

        # Cusp parameters
        self.gamma_apara = 1 / max(d - 1, 1)
        self.gamma_para = 1 / (d + 1)
        self.cusp_len = 1 / omega**0.5

    def _mlp(
        self, in_dim: int, hid: int, out_dim: int, n_layers: int, act: nn.Module
    ) -> nn.Module:
        layers: list[nn.Module] = [nn.Linear(in_dim, hid), act]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hid, hid), act])
        layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    def _aggregate_dense(self, msgs: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        elif self.aggregation == "mean":
            return msgs.mean(dim=2)
        elif self.aggregation == "max":
            return msgs.max(dim=2).values
        raise ValueError(f"Unknown aggregation '{self.aggregation}'")

    def forward(
        self, x: torch.Tensor, spin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_particles, dim = x.shape
        if dim != self.d or n_particles != self.n_particles:
            raise ValueError(
                f"Expected x with shape (B, {self.n_particles}, {self.d}), got {tuple(x.shape)}."
            )

        x_sc = x * self.omega**0.5

        if self.use_spin:
            spin_feat = spin.view(1, n_particles, 1).to(dtype=x.dtype).expand(batch_size, -1, -1)
            node_in = torch.cat([x_sc, spin_feat], dim=-1)
        else:
            node_in = x_sc

        h_v = self.node_embed(node_in)

        # Dense edge features
        r_vec = x_sc.unsqueeze(1) - x_sc.unsqueeze(2)
        r2 = (r_vec * r_vec).sum(dim=-1, keepdim=True)
        r1 = torch.sqrt(r2 + 1e-12)

        h_v_i = h_v.unsqueeze(2).expand(-1, -1, n_particles, -1)
        h_v_j = h_v.unsqueeze(1).expand(-1, n_particles, -1, -1)
        edge_in = torch.cat([r1, h_v_i, h_v_j], dim=-1)
        h_e = self.edge_embed(edge_in)

        # Message passing
        eye = torch.eye(n_particles, device=x.device, dtype=x.dtype)
        for step in range(self.n_mp_steps):
            v_to_e = self.rho_v_to_e[step](h_v)
            v_i_to_e = v_to_e.unsqueeze(2).expand_as(h_e)
            v_j_to_e = v_to_e.unsqueeze(1).expand_as(h_e)
            edge_upd_in = torch.cat([h_e, v_i_to_e, v_j_to_e], dim=-1)
            h_e = self.edge_updates[step](edge_upd_in)

            msgs = self.rho_e_to_v[step](h_e)
            m_v = self._aggregate_dense(msgs)
            node_upd_in = torch.cat([h_v, m_v], dim=-1)
            h_v = self.node_updates[step](node_upd_in)

        # Backflow output
        dx = self.dx_head(h_v)
        if self.out_bound == "tanh":
            dx = torch.tanh(dx)
        scale = F.softplus(self.bf_scale_raw)
        dx = scale * dx

        # Jastrow readout
        h_v_mean = h_v.mean(dim=1)
        h_e_pairs = h_e[:, self.idx_i, self.idx_j, :]
        h_e_mean = h_e_pairs.mean(dim=1) if h_e_pairs.shape[1] > 0 else torch.zeros(
            batch_size, h_e.shape[-1], device=x.device, dtype=x.dtype
        )
        r2_mean = torch.sqrt(r2.squeeze(-1) + 1e-12)
        # Exclude self-distances (diagonal zeros) to avoid NaN gradients
        mask = 1.0 - eye
        r2_mean = (r2_mean * mask).sum(dim=(1, 2)) / max(mask.sum().item(), 1.0)
        r2_mean = r2_mean.unsqueeze(-1)

        f_in = torch.cat([h_v_mean, h_e_mean, r2_mean], dim=-1)
        f_nn = self.f_head(f_in)

        # Envelope
        ell2 = self.envelope_width_aho / max(self.omega, 1e-12)
        r2_phys_total = (x * x).sum(dim=(1, 2))
        envelope = torch.exp(-0.5 * r2_phys_total / ell2).unsqueeze(-1)

        # Cusps
        diff_phys = x[:, self.idx_i, :] - x[:, self.idx_j, :]
        r_phys = torch.sqrt((diff_phys**2).sum(dim=-1) + 1e-30)

        sp = spin.long() if spin.ndim == 1 else spin
        if sp.ndim == 1:
            si = sp[self.idx_i]
            sj = sp[self.idx_j]
        else:
            si = sp[:, self.idx_i]
            sj = sp[:, self.idx_j]
        same_sp = (si == sj).float()
        gamma = same_sp * self.gamma_para + (1.0 - same_sp) * self.gamma_apara
        cusp = (gamma * r_phys / (1.0 + r_phys / self.cusp_len)).sum(dim=-1, keepdim=True)

        f = f_nn * envelope + 0.2 * cusp

        return dx, f
