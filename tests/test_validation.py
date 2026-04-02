from __future__ import annotations

from observables.validation import compute_virial_metrics


def test_compute_virial_metrics_uses_negative_coulomb_sign():
    (virial_lhs, virial_rhs, virial_residual, virial_relative) = compute_virial_metrics(
        T_mean=0.89855, V_trap_mean=1.1151, V_int_mean=0.1677, E_mean=2.1814
    )
    assert abs(virial_lhs - 1.7971) < 0.0001
    assert abs(virial_rhs - (2 * 1.1151 - 0.1677)) < 1e-12
    assert virial_residual < 0
    assert 0.1 < virial_relative < 0.13
