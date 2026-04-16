from __future__ import annotations

import pytest
import torch

from config import SystemConfig
from training.vmc_colloc import GroundStateTrainingConfig, same_dot_occupancy_penalty, train_ground_state


class _ConstantModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class _QuadraticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.scale * (x * x).sum(dim=(1, 2))


def test_same_dot_occupancy_penalty_detects_same_well_configurations() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=4.0, omega=1.0)
    x = torch.tensor(
        [
            [[-2.1, 0.0], [-1.9, 0.0]],
            [[-2.0, 0.0], [2.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    penalty = same_dot_occupancy_penalty(x, system, margin=0.0)

    assert torch.isclose(penalty, torch.tensor(0.5, dtype=torch.float64))


def test_same_dot_occupancy_penalty_uses_margin_as_soft_distance_weight() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=4.0, omega=1.0)
    x = torch.tensor(
        [
            [[-2.0, 0.0], [-1.8, 0.0]],
            [[-2.0, 0.0], [2.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    penalty = same_dot_occupancy_penalty(x, system, margin=0.5)

    assert penalty.item() > 0.0
    assert penalty.item() < 1.0


def test_direct_variational_losses_require_mh_sampler() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=4.0, omega=1.0)
    train_cfg = GroundStateTrainingConfig(
        epochs=1,
        n_coll=8,
        loss_type="weak_form",
        sampler="stratified",
        non_mcmc_only=True,
        device="cpu",
        dtype="float64",
    )

    with pytest.raises(ValueError, match="requires sampler='mh' or sampler='is'"):
        train_ground_state(_ConstantModel(), system, params={}, train_cfg=train_cfg)


def test_direct_variational_losses_allow_fixed_proposal_is_sampler() -> None:
    system = SystemConfig.double_dot(N_L=1, N_R=1, sep=4.0, omega=1.0)
    train_cfg = GroundStateTrainingConfig(
        epochs=1,
        n_coll=16,
        loss_type="weak_form",
        sampler="is",
        non_mcmc_only=True,
        device="cpu",
        dtype="float64",
    )

    result = train_ground_state(_QuadraticModel(), system, params={}, train_cfg=train_cfg)

    assert torch.isfinite(torch.tensor(result["final_energy"], dtype=torch.float64))
    assert torch.isfinite(torch.tensor(result["final_loss"], dtype=torch.float64))
    assert result["final_ess"] > 0.0