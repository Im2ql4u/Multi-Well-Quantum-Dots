from __future__ import annotations


def compute_virial_metrics(
    T_mean: float,
    V_trap_mean: float,
    V_int_mean: float,
    E_mean: float,
) -> tuple[float, float, float, float]:
    """Return virial theorem diagnostics for harmonic confinement plus Coulomb.

    For a harmonic trap and Coulomb interaction,
    2<T> = 2<V_trap> - <V_ee>.
    """
    virial_lhs = 2.0 * T_mean
    virial_rhs = 2.0 * V_trap_mean - V_int_mean
    virial_residual = virial_lhs - virial_rhs
    virial_relative = abs(virial_residual) / max(abs(E_mean), 1e-10)
    return virial_lhs, virial_rhs, virial_residual, virial_relative