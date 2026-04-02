from __future__ import annotations

import torch
from architectures import CTNNBackflowNet, CTNNJastrow, OrbitalBackflowNet, PINN, UnifiedCTNN


def _assert_finite_parameter_grads(model: torch.nn.Module) -> None:
    saw_grad = False
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            saw_grad = True
    assert saw_grad


def test_pinn_architecture_package_reexports_existing_model():
    model = PINN(n_particles=2, d=2, omega=1).double()
    x = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    out = model(x, spin=torch.tensor([0, 1]))
    assert out.shape == (4, 1)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    _assert_finite_parameter_grads(model)


def test_ctnn_jastrow_forward_and_backward_are_finite():
    model = CTNNJastrow(n_particles=2, d=2, omega=1).double()
    x = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    out = model(x, spin=torch.tensor([0, 1]))
    assert out.shape == (4, 1)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    _assert_finite_parameter_grads(model)


def test_unified_ctnn_emits_backflow_and_jastrow():
    model = UnifiedCTNN(d=2, n_particles=2, omega=1).double()
    x = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    (dx, f) = model(x, spin=torch.tensor([0, 1]))
    assert dx.shape == (4, 2, 2)
    assert f.shape == (4, 1)
    loss = dx.sum() + f.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    _assert_finite_parameter_grads(model)


def test_orbital_backflow_emits_orbital_corrections():
    model = OrbitalBackflowNet(d=2, n_occ=2, omega=1).double()
    x = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    out = model(x, spin=torch.tensor([0, 1]))
    assert out.shape == (4, 2, 2)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    _assert_finite_parameter_grads(model)


def test_ctnn_backflow_reexport_runs():
    model = CTNNBackflowNet(d=2, omega=1).double()
    x = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    out = model(x, spin=torch.tensor([0, 1]))
    assert out.shape == (4, 2, 2)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    _assert_finite_parameter_grads(model)
