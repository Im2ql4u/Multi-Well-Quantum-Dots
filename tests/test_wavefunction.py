from __future__ import annotations

import torch
from config import SystemConfig
from wavefunction import GroundStateWF, SlaterOnlyWF, setup_closed_shell_system


def _assert_finite_grads(model: torch.nn.Module) -> None:
    saw_grad = False
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            saw_grad = True
    assert saw_grad


def _make_reference_state():
    system = SystemConfig.single_dot(N=2, omega=1)
    (C_occ, spin, params) = setup_closed_shell_system(
        system, device="cpu", dtype=torch.float64, E_ref=3
    )
    return (system, C_occ, spin, params)


def test_slater_only_wavefunction_forward_is_finite():
    (system, C_occ, spin, params) = _make_reference_state()
    model = SlaterOnlyWF(system, C_occ, spin, params).double()
    x = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_ground_state_wavefunction_supports_all_architectures():
    (system, C_occ, spin, params) = _make_reference_state()
    x = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    model = GroundStateWF(system, C_occ, spin, params).double()
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    _assert_finite_grads(model)
