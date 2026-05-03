"""Reload a trained ``GroundStateWF`` from a result directory.

This is a small wrapper that reproduces the construction sequence from
``run_ground_state.run_training_from_config`` but stops short of training,
so that downstream code (entanglement evaluation, observable measurement,
inverse-design targets) can grab a callable wavefunction object without
duplicating boilerplate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from config import SystemConfig
from run_ground_state import build_system_from_config
from wavefunction import (
    GroundStateWF,
    resolve_spin_configuration,
    setup_closed_shell_system,
)


@dataclass
class LoadedWavefunction:
    """Bundle of objects returned by :func:`load_wavefunction_from_dir`."""

    model: GroundStateWF
    system: SystemConfig
    config: dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    result_dir: Path

    @torch.no_grad()
    def signed_log_psi(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate ``(sign, log|psi|)`` for input ``x: (B, N, D)``.

        Inputs are coerced to the model's device/dtype automatically.
        """
        x_t = x.to(device=self.device, dtype=self.dtype)
        return self.model.signed_log_psi(x_t)

    @torch.no_grad()
    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate ``psi(x) = sign · exp(log|psi|)`` for ``x: (B, N, D)``."""
        sign, logp = self.signed_log_psi(x)
        return sign * torch.exp(logp)


def _select_device(requested: str | torch.device | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _resolve_dtype(name: str | None) -> torch.dtype:
    name = (name or "float32").lower()
    return {"float32": torch.float32, "float64": torch.float64}[name]


def load_wavefunction_from_dir(
    result_dir: Path | str,
    *,
    device: str | torch.device | None = None,
    weights_only: bool = False,
) -> LoadedWavefunction:
    """Reconstruct a ``GroundStateWF`` from a saved training run.

    Parameters
    ----------
    result_dir
        Directory containing ``config.yaml`` and ``model.pt`` from a
        ``run_ground_state`` run (or any equivalent ``two_stage`` stage).
    device
        Optional override for the evaluation device. Defaults to the
        device recorded in the config, or CUDA-0 if available.
    weights_only
        Forwarded to :func:`torch.load`. Set to ``True`` for stricter
        deserialisation when the checkpoint is fully tensor-only.
    """
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")

    config_path = result_dir / "config.yaml"
    model_path = result_dir / "model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {result_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {result_dir}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)

    system = build_system_from_config(raw_cfg)
    arch_cfg = raw_cfg.get("architecture", {})
    train_cfg = raw_cfg.get("training", {})

    requested_device = device if device is not None else train_cfg.get("device", None)
    if isinstance(requested_device, str) and requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    selected_device = _select_device(requested_device)
    dtype = _resolve_dtype(train_cfg.get("dtype", "float32"))

    spin_cfg = raw_cfg.get("spin")
    spin_meta = resolve_spin_configuration(system, spin_cfg)
    allow_missing_dmc = bool(raw_cfg.get("allow_missing_dmc", True))

    # ``setup_closed_shell_system`` resolves an internal ``E_ref``; for
    # post-training evaluation any value works (it is not used by ``forward``).
    (C_occ, spin, params) = setup_closed_shell_system(
        system,
        device=str(selected_device),
        dtype=dtype,
        E_ref="auto",
        allow_missing_dmc=allow_missing_dmc,
        spin_pattern=spin_meta["pattern"],
    )

    model = GroundStateWF(
        system,
        C_occ,
        spin,
        params,
        arch_type=arch_cfg.get("arch_type", "pinn"),
        pinn_hidden=int(arch_cfg.get("pinn_hidden", 64)),
        pinn_layers=int(arch_cfg.get("pinn_layers", 2)),
        bf_hidden=int(arch_cfg.get("bf_hidden", 32)),
        bf_layers=int(arch_cfg.get("bf_layers", 2)),
        use_well_features=bool(arch_cfg.get("use_well_features", False)),
        use_well_backflow=bool(arch_cfg.get("use_well_backflow", False)),
        use_backflow=bool(arch_cfg.get("use_backflow", True)),
        singlet=bool(arch_cfg.get("singlet", False)),
        multi_ref=bool(arch_cfg.get("multi_ref", False)),
    )

    state = torch.load(model_path, map_location=selected_device, weights_only=weights_only)
    model.load_state_dict(state, strict=True)
    model.to(device=selected_device, dtype=dtype)
    model.eval()

    return LoadedWavefunction(
        model=model,
        system=system,
        config=raw_cfg,
        device=selected_device,
        dtype=dtype,
        result_dir=result_dir,
    )
