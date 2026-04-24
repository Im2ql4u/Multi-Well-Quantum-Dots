from __future__ import annotations

from scripts.run_two_stage_research import (
    _build_stage_a_bootstrap_target_cfg,
    _build_stage_a_self_residual_cfg,
    _candidate_rank_key,
)


def test_build_stage_a_self_residual_cfg_clears_external_target_guidance() -> None:
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
        },
    }

    cfg = _build_stage_a_self_residual_cfg(base_cfg, epochs=400, suffix="__self")

    assert cfg["run_name"] == "demo__self"
    assert cfg["training"]["epochs"] == 400
    assert cfg["training"]["loss_type"] == "residual"
    assert cfg["training"]["residual_objective"] == "residual"
    assert cfg["training"]["residual_target_energy"] is None
    assert cfg["training"]["alpha_start"] == 0.0
    assert cfg["training"]["alpha_end"] == 0.0
    assert cfg["training"]["sampler"] == "stratified"
    assert cfg["training"]["non_mcmc_only"] is True


def test_build_stage_a_bootstrap_target_cfg_uses_internal_target() -> None:
    base_cfg = {
        "run_name": "demo",
        "training": {
            "epochs": 3000,
            "seed": 42,
        },
    }

    cfg = _build_stage_a_bootstrap_target_cfg(
        base_cfg,
        epochs=600,
        suffix="__boot",
        bootstrap_target_energy=2.19,
        seed_override=314,
    )

    assert cfg["run_name"] == "demo__boot"
    assert cfg["training"]["epochs"] == 600
    assert cfg["training"]["residual_objective"] == "energy_var"
    assert cfg["training"]["residual_target_energy"] == 2.19
    assert cfg["training"]["alpha_start"] == 0.0
    assert cfg["training"]["alpha_end"] == 1.0
    assert cfg["training"]["seed"] == 314


def test_candidate_rank_key_prefers_lower_variance_then_lower_energy() -> None:
    candidate_a = {"result": {"final_energy_var": 2.0e-6, "final_energy": 2.24}}
    candidate_b = {"result": {"final_energy_var": 2.0e-6, "final_energy": 2.23}}
    candidate_c = {"result": {"final_energy_var": 3.0e-6, "final_energy": 2.20}}

    ranked = sorted([candidate_a, candidate_b, candidate_c], key=_candidate_rank_key)

    assert ranked[0] is candidate_b
    assert ranked[1] is candidate_a
    assert ranked[2] is candidate_c
