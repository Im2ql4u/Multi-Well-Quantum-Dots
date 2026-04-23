from __future__ import annotations

from pathlib import Path

from scripts.run_two_stage_ground_state import (
    StageAGate,
    _build_stage_a_cfg,
    _build_stage_a_self_residual_cfg,
    _build_stage_b_cfg,
    _infer_stage_a_strategy,
    _stage_a_gate_status,
)


def test_build_stage_a_cfg_preserves_guided_warm_start_settings() -> None:
    base_cfg = {
        "run_name": "demo",
        "training": {
            "epochs": 3000,
            "loss_type": "residual",
            "residual_objective": "energy_var",
            "residual_target_energy": 2.25,
            "alpha_start": 0.0,
            "alpha_end": 1.0,
            "sampler": "stratified",
            "non_mcmc_only": False,
            "print_every": 50,
        },
    }

    cfg = _build_stage_a_cfg(base_cfg, stage_a_epochs=800, suffix="__stageA")

    assert cfg["run_name"] == "demo__stageA"
    assert cfg["training"]["epochs"] == 800
    assert cfg["training"]["loss_type"] == "residual"
    assert cfg["training"]["residual_objective"] == "energy_var"
    assert cfg["training"]["residual_target_energy"] == 2.25
    assert cfg["training"]["alpha_start"] == 0.0
    assert cfg["training"]["alpha_end"] == 1.0
    assert cfg["training"]["sampler"] == "stratified"
    assert cfg["training"]["non_mcmc_only"] is True
    assert cfg["training"]["print_every"] == 50


def test_stage_a_gate_requires_positive_energy_and_minimum_ess() -> None:
    gate = StageAGate(min_final_ess=32.0, min_final_energy=0.0)

    failed = _stage_a_gate_status(
        {"final_energy": -0.1, "final_ess": 4.0},
        gate,
    )
    passed = _stage_a_gate_status(
        {"final_energy": 2.2, "final_ess": 128.0},
        gate,
    )

    assert failed["passed"] is False
    assert len(failed["failures"]) == 2
    assert passed["passed"] is True
    assert passed["failures"] == []


def test_build_stage_a_self_residual_cfg_removes_target_guidance() -> None:
    base_cfg = {
        "run_name": "demo",
        "training": {
            "epochs": 3000,
            "loss_type": "residual",
            "residual_objective": "energy_var",
            "residual_target_energy": 2.25,
            "alpha_start": 0.0,
            "alpha_end": 1.0,
            "sampler": "is",
            "non_mcmc_only": False,
            "print_every": 50,
        },
    }

    cfg = _build_stage_a_self_residual_cfg(base_cfg, stage_a_epochs=800, suffix="__stageA")

    assert cfg["run_name"] == "demo__stageA"
    assert cfg["training"]["epochs"] == 800
    assert cfg["training"]["loss_type"] == "residual"
    assert cfg["training"]["residual_objective"] == "residual"
    assert cfg["training"]["residual_target_energy"] is None
    assert cfg["training"]["alpha_start"] == 0.0
    assert cfg["training"]["alpha_end"] == 0.0
    assert cfg["training"]["sampler"] == "stratified"
    assert cfg["training"]["non_mcmc_only"] is True
    assert cfg["training"]["print_every"] == 50


def test_build_stage_a_self_residual_cfg_can_apply_legacy_n2_singlet_recipe() -> None:
    base_cfg = {
        "run_name": "demo",
        "architecture": {"arch_type": "pinn", "use_backflow": True},
        "training": {
            "epochs": 3000,
            "loss_type": "residual",
            "residual_objective": "energy_var",
            "residual_target_energy": 2.25,
            "alpha_start": 0.0,
            "alpha_end": 1.0,
            "sampler_sigma_center": 0.15,
            "sampler_sigma_tails": 0.8,
        },
    }

    cfg = _build_stage_a_self_residual_cfg(
        base_cfg,
        stage_a_epochs=800,
        suffix="__stageA",
        use_legacy_n2_singlet=True,
    )

    assert cfg["architecture"]["singlet"] is True
    assert cfg["architecture"]["use_backflow"] is False
    assert cfg["training"]["sampler_sigma_center"] == 0.20
    assert cfg["training"]["sampler_sigma_tails"] == 1.00
    assert cfg["training"]["sampler_sigma_mixed_out"] == 0.70
    assert cfg["training"]["sampler_shell_radius"] == 1.20


def test_build_stage_b_cfg_can_preserve_legacy_n2_singlet_recipe() -> None:
    base_cfg = {
        "run_name": "demo",
        "architecture": {"arch_type": "pinn"},
        "training": {"epochs": 3000},
    }

    cfg = _build_stage_b_cfg(
        base_cfg,
        init_result_dir=Path("/tmp/demo"),
        stage_b_epochs=None,
        suffix="__stageB",
        use_legacy_n2_singlet=True,
    )

    assert cfg["architecture"]["singlet"] is True
    assert cfg["architecture"]["use_backflow"] is False
    assert cfg["training"]["sampler_sigma_center"] == 0.20
    assert cfg["training"]["sampler_sigma_tails"] == 1.00


def test_infer_stage_a_strategy_uses_singlet_self_residual_for_n2_annealed_target() -> None:
    base_cfg = {
        "system": {"type": "double_dot", "n_left": 1, "n_right": 1},
        "training": {
            "loss_type": "residual",
            "residual_objective": "energy_var",
            "residual_target_energy": 2.25,
            "alpha_end": 1.0,
        },
    }

    assert _infer_stage_a_strategy(base_cfg) == "singlet_self_residual"


def test_infer_stage_a_strategy_keeps_guided_for_larger_systems() -> None:
    base_cfg = {
        "system": {"type": "double_dot", "n_left": 2, "n_right": 1},
        "training": {
            "loss_type": "residual",
            "residual_objective": "energy_var",
            "residual_target_energy": 3.6,
            "alpha_end": 1.0,
        },
    }

    assert _infer_stage_a_strategy(base_cfg) == "guided"
