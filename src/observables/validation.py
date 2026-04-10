from __future__ import annotations


def compute_virial_metrics(
    T_mean: float,
    V_trap_mean: float,
    V_int_mean: float,
    E_mean: float,
    r_dot_grad_v_mean: float | None = None,
) -> tuple[float, float, float, float]:
    """Return virial diagnostics with optional generalized multi-well RHS.

    If ``r_dot_grad_v_mean`` is provided, use the generalized virial identity
    ``2<T> = <r · ∇V>``. Otherwise, fall back to the legacy harmonic+Coulomb
    expression ``2<T> = 2<V_trap> - <V_ee>`` for backward compatibility.
    """
    virial_lhs = 2 * T_mean
    if r_dot_grad_v_mean is None:
        virial_rhs = 2 * V_trap_mean - V_int_mean
    else:
        virial_rhs = float(r_dot_grad_v_mean)
    virial_residual = virial_lhs - virial_rhs
    virial_relative = abs(virial_residual) / max(abs(E_mean), 1e-10)
    return (virial_lhs, virial_rhs, virial_residual, virial_relative)
