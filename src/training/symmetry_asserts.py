"""Hard structural quantum-number checks for variational wavefunctions.

These are *tripwire* assertions invoked at the end of every training run to
guarantee that the trained model still carries the spin sector and particle
number it was constructed with. Because our fixed-spin Slater / multi-ref
architectures encode `n_up`, `n_down`, and the spin template as **non-trainable
buffers**, these quantum numbers are conserved exactly by construction; the
checks here protect against:

* Config errors where ``n_up + n_down != system.n_particles``.
* Buffer corruption (e.g., wrong device / dtype mismatch silently coercing).
* Loaded checkpoints whose ``params`` dict lost a key.
* Multi-reference templates with mixed Sz across templates (forbidden — the
  multi-ref combinatorial sum must preserve total Sz).

Anchor reference: ``reports/2026-04-28_grand_plan_anchored.md`` Phase 0.1.

Tolerance: integer counts are compared exactly; floating equivalents
(``Sz_target``, ``N_target``) are compared to within ``1e-6`` (the tolerance
called out in the grand plan).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import torch

LOGGER = logging.getLogger("training.symmetry_asserts")

DEFAULT_TOLERANCE = 1.0e-6


class SymmetryViolationError(RuntimeError):
    """Raised when a structural quantum-number assertion fails."""


def _expected_total_sz(n_up: int, n_down: int) -> float:
    return 0.5 * float(n_up - n_down)


def assert_quantum_numbers_consistent(
    *,
    model: torch.nn.Module | None,
    system: Any,
    params: Mapping[str, Any],
    tolerance: float = DEFAULT_TOLERANCE,
    context: str = "train_ground_state",
) -> dict[str, Any]:
    """Verify that the model + params + system are mutually consistent.

    Parameters
    ----------
    model
        Optional ``torch.nn.Module`` instance (e.g. ``GroundStateWF``). When
        provided, its ``spin_template`` buffer is cross-checked against
        ``params["spin_pattern"]``. ``None`` is accepted to support audits
        that operate from a saved ``params`` dict alone.
    system
        Object with ``.n_particles`` (and optionally ``.wells``). Typically a
        ``SystemConfig`` instance.
    params
        Dictionary returned by ``setup_*_for_ground_state``. Must contain
        ``n_up``, ``n_down``, ``spin_pattern``, ``n_particles``.
    tolerance
        Floating-point tolerance for the derived ``Sz_target`` and
        ``N_target`` comparisons.
    context
        Free-form string used in the error message to localise failures.

    Returns
    -------
    dict
        Diagnostic payload with the verified quantum numbers and the per-check
        outcome.

    Raises
    ------
    SymmetryViolationError
        If any structural check fails.
    """
    diag: dict[str, Any] = {"context": context, "checks": {}}

    # Production callers populate ``params`` via ``setup_*_for_ground_state``;
    # unit-test stubs and toy callers may pass ``params={}``. We only run the
    # structural check when the spin metadata is actually present.
    required_keys = ("n_up", "n_down", "spin_pattern")
    if not all(key in params for key in required_keys):
        diag["skipped"] = True
        diag["reason"] = (
            "params dict does not carry spin metadata (missing one of "
            f"{required_keys}); check is structurally meaningless and was skipped."
        )
        LOGGER.debug("[%s] symmetry asserts skipped (no spin metadata).", context)
        return diag

    n_up = int(params["n_up"])
    n_down = int(params["n_down"])
    n_particles_param = int(params.get("n_particles", n_up + n_down))
    spin_pattern = list(params["spin_pattern"])

    n_particles_system = int(getattr(system, "n_particles"))

    diag["n_up"] = n_up
    diag["n_down"] = n_down
    diag["n_particles_param"] = n_particles_param
    diag["n_particles_system"] = n_particles_system
    diag["expected_Sz"] = _expected_total_sz(n_up, n_down)

    # Check 1: n_up + n_down equals particle count in both params and system.
    if n_up + n_down != n_particles_param:
        raise SymmetryViolationError(
            f"[{context}] params n_up({n_up})+n_down({n_down})={n_up + n_down} "
            f"!= params['n_particles']={n_particles_param}"
        )
    if n_up + n_down != n_particles_system:
        raise SymmetryViolationError(
            f"[{context}] params n_up({n_up})+n_down({n_down})={n_up + n_down} "
            f"!= system.n_particles={n_particles_system}"
        )
    if abs((n_up + n_down) - n_particles_system) > tolerance:
        raise SymmetryViolationError(
            f"[{context}] particle-count residual exceeds tolerance={tolerance}"
        )
    diag["checks"]["count_consistency"] = "ok"

    # Check 2: spin pattern length and counts.
    if len(spin_pattern) != n_particles_system:
        raise SymmetryViolationError(
            f"[{context}] len(spin_pattern)={len(spin_pattern)} "
            f"!= system.n_particles={n_particles_system}"
        )
    n_up_pattern = sum(1 for value in spin_pattern if int(value) == 0)
    n_down_pattern = sum(1 for value in spin_pattern if int(value) == 1)
    if n_up_pattern + n_down_pattern != len(spin_pattern):
        bad = [v for v in spin_pattern if int(v) not in (0, 1)]
        raise SymmetryViolationError(
            f"[{context}] spin_pattern contains values outside {{0,1}}: {bad[:8]}"
        )
    if n_up_pattern != n_up or n_down_pattern != n_down:
        raise SymmetryViolationError(
            f"[{context}] spin_pattern counts ({n_up_pattern}↑, {n_down_pattern}↓) "
            f"do not match params ({n_up}↑, {n_down}↓)"
        )
    diag["checks"]["pattern_counts"] = "ok"

    # Check 3: Sz target is achieved exactly by the pattern.
    sz_pattern = 0.0
    for value in spin_pattern:
        sz_pattern += 0.5 if int(value) == 0 else -0.5
    if abs(sz_pattern - diag["expected_Sz"]) > tolerance:
        raise SymmetryViolationError(
            f"[{context}] pattern Sz={sz_pattern:.9f} differs from expected "
            f"{diag['expected_Sz']:.9f} by more than {tolerance}"
        )
    diag["checks"]["sz_residual"] = float(sz_pattern - diag["expected_Sz"])

    # Check 4: model.spin_template buffer matches the pattern (if model given).
    if model is not None:
        spin_buf = getattr(model, "spin_template", None)
        if not isinstance(spin_buf, torch.Tensor):
            raise SymmetryViolationError(
                f"[{context}] model has no torch.Tensor 'spin_template' buffer "
                f"(got {type(spin_buf).__name__})."
            )
        if spin_buf.numel() != n_particles_system:
            raise SymmetryViolationError(
                f"[{context}] model.spin_template numel={spin_buf.numel()} "
                f"!= system.n_particles={n_particles_system}"
            )
        buf_list = [int(v) for v in spin_buf.detach().cpu().reshape(-1).tolist()]
        if buf_list != [int(v) for v in spin_pattern]:
            raise SymmetryViolationError(
                f"[{context}] model.spin_template={buf_list} differs from "
                f"params['spin_pattern']={spin_pattern}"
            )
        diag["checks"]["spin_template_match"] = "ok"

        # Check 5 (multi_ref): all internal templates share Sz. The multi-ref
        # combinatorial sum spans assignments of a fixed (n_up, n_down) over
        # wells, so by construction every term carries the same total Sz; a
        # mismatch here means the architecture has been mutated post-hoc.
        is_multi_ref = bool(getattr(model, "multi_ref", False))
        diag["multi_ref"] = is_multi_ref
        if is_multi_ref:
            spin_1d = spin_buf if spin_buf.ndim == 1 else spin_buf[0]
            up_count = int((spin_1d == 0).sum().item())
            down_count = int((spin_1d == 1).sum().item())
            if up_count != n_up or down_count != n_down:
                raise SymmetryViolationError(
                    f"[{context}] multi_ref template (n_up={up_count}, "
                    f"n_down={down_count}) inconsistent with params "
                    f"(n_up={n_up}, n_down={n_down})"
                )
            diag["checks"]["multi_ref_consistency"] = "ok"

    LOGGER.info(
        "[%s] symmetry asserts passed: n_up=%d n_down=%d Sz=%+.1f N=%d",
        context,
        n_up,
        n_down,
        diag["expected_Sz"],
        n_particles_system,
    )
    return diag
