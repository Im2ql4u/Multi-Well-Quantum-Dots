from __future__ import annotations

from pathlib import Path

import torch

from config import SystemConfig
from run_ground_state import _maybe_load_initial_state
from wavefunction import GroundStateWF, setup_closed_shell_system


def _make_model() -> GroundStateWF:
    system = SystemConfig.single_dot(N=2, omega=1.0, dim=2)
    c_occ, spin, params = setup_closed_shell_system(
        system,
        device="cpu",
        dtype=torch.float64,
        E_ref=3.0,
        allow_missing_dmc=True,
    )
    return GroundStateWF(system, c_occ, spin, params).double()


def test_init_from_result_dir_loads_checkpoint(tmp_path: Path) -> None:
    source_model = _make_model()
    for param in source_model.parameters():
        with torch.no_grad():
            param.add_(0.123)

    result_dir = tmp_path / "legacy_run"
    result_dir.mkdir()
    torch.save(source_model.state_dict(), result_dir / "model.pt")

    target_model = _make_model()
    first_name, first_param = next(iter(target_model.named_parameters()))
    before = first_param.detach().clone()

    info = _maybe_load_initial_state(
        target_model,
        {"init_from": {"result_dir": str(result_dir)}},
        device=torch.device("cpu"),
        config_source=None,
    )

    assert info is not None
    assert Path(info["path"]) == (result_dir / "model.pt")
    assert info["missing_keys"] == []
    assert info["unexpected_keys"] == []

    after = dict(target_model.named_parameters())[first_name].detach().clone()
    assert not torch.allclose(before, after)
    assert torch.allclose(after, dict(source_model.named_parameters())[first_name].detach())
