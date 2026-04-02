from __future__ import annotations

import torch
import torch.nn as nn

from PINN import BackflowNet, CTNNBackflowNet  # noqa: F401


class OrbitalBackflowNet(nn.Module):
    """Message-passing backflow network that emits per-orbital corrections.

    Reconstructed from bytecode; simplified but interface-compatible.
    """

    def __init__(
        self,
        d: int = 2,
        n_occ: int = 2,
        msg_hidden: int = 32,
        msg_layers: int = 2,
        hidden: int = 32,
        layers: int = 2,
        act: str = "tanh",
        aggregation: str = "sum",
        use_spin: bool = False,
        same_spin_only: bool = False,
        out_bound: str = "tanh",
        bf_scale_init: float = 1.0,
        zero_init_last: bool = False,
        omega: float = 1.0,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_occ = n_occ
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only
        self.aggregation = aggregation
        self.out_bound = out_bound
        self.omega = omega
        self._scale_override = False

        import math

        self.bf_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(bf_scale_init) - 1.0))
        )

        _act: nn.Module = nn.Tanh()
        node_in_dim = d + 1 if use_spin else d
        node_hidden = hidden
        edge_hidden = msg_hidden

        self.node_embed = nn.Linear(node_in_dim, node_hidden, bias=False)
        edge_in_dim = 1 + 2 * node_hidden
        self.edge_embed = nn.Linear(edge_in_dim, edge_hidden, bias=False)

        self.rho_v_to_e = nn.Linear(node_hidden, edge_hidden, bias=False)
        self.rho_e_to_v = nn.Linear(edge_hidden, node_hidden, bias=False)
        self.edge_update = self._mlp(edge_hidden * 3, edge_hidden, edge_hidden, msg_layers, _act)
        self.node_update = self._mlp(node_hidden * 2, node_hidden, node_hidden, layers, _act)
        self.orb_head = nn.Linear(node_hidden, d, bias=False)

        if zero_init_last:
            nn.init.zeros_(self.orb_head.weight)
            if self.orb_head.bias is not None:
                nn.init.zeros_(self.orb_head.bias)

    def _mlp(self, in_dim: int, hid: int, out_dim: int, num_layers: int, act: nn.Module) -> nn.Module:
        mods: list[nn.Module] = []
        mods.append(nn.Linear(in_dim, hid))
        mods.append(act)
        for _ in range(num_layers - 2):
            mods.append(nn.Linear(hid, hid))
            mods.append(act)
        mods.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*mods)

    @property
    def bf_scale(self) -> torch.Tensor:
        return nn.functional.softplus(self.bf_scale_raw)

    def _aggregate(self, msgs: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        elif self.aggregation == "mean":
            return msgs.mean(dim=2)
        elif self.aggregation == "max":
            return msgs.max(dim=2).values
        raise ValueError(f"Unknown aggregation '{self.aggregation}'")

    def forward(
        self, x: torch.Tensor, spin: torch.Tensor, scale_override: float | None = None
    ) -> torch.Tensor:
        batch_size, n_particles, dim = x.shape
        if dim != self.d:
            raise ValueError(f"Expected dim={self.d}, got {dim}.")

        x_sc = x * self.omega**0.5

        if self.use_spin:
            spin_feat = spin.view(1, n_particles, 1).to(dtype=x.dtype).expand(batch_size, -1, -1)
            node_in = torch.cat([x_sc, spin_feat], dim=-1)
        else:
            node_in = x_sc

        h_v = self.node_embed(node_in)

        # Build edge features
        r = x_sc.unsqueeze(1) - x_sc.unsqueeze(2)  # (B, N, N, D)
        r2 = (r * r).sum(dim=-1, keepdim=True)  # (B, N, N, 1)
        r1 = torch.sqrt(r2 + 1e-12)

        h_v_i = h_v.unsqueeze(2).expand(-1, -1, n_particles, -1)
        h_v_j = h_v.unsqueeze(1).expand(-1, n_particles, -1, -1)
        edge_in = torch.cat([r1, h_v_i, h_v_j], dim=-1)
        h_e = self.edge_embed(edge_in)

        # Message passing
        v_i_to_e = self.rho_v_to_e(h_v).unsqueeze(2).expand_as(h_e)
        v_j_to_e = self.rho_v_to_e(h_v).unsqueeze(1).expand_as(h_e)
        edge_update_in = torch.cat([h_e, v_i_to_e, v_j_to_e], dim=-1)
        h_e = self.edge_update(edge_update_in)

        msgs = self.rho_e_to_v(h_e)
        m_v = self._aggregate(msgs)
        node_update_in = torch.cat([h_v, m_v], dim=-1)
        h_v = self.node_update(node_update_in)

        # Output
        dpsi = self.orb_head(h_v)  # (B, N, D)

        if self.out_bound == "tanh":
            dpsi = torch.tanh(dpsi)

        scale = self.bf_scale if scale_override is None else scale_override
        dpsi = scale * dpsi
        return dpsi
