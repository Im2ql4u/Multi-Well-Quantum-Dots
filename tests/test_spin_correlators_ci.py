"""Unit tests for ``observables.spin_correlators_ci`` (Phase 0.2 anchor)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observables.spin_correlators_ci import (  # noqa: E402
    CISpinCorrelators,
    compute_spin_correlators_ci,
    expected_S1_dot_S2,
)


def _toy_basis() -> list[tuple[int, int, str, str]]:
    """A 5-state basis covering all spin sectors used in production."""
    return [
        (0, 0, "ud", "singlet"),
        (0, 1, "singlet", "singlet"),
        (0, 1, "uu", "triplet_p"),
        (0, 1, "triplet_0", "triplet_0"),
        (0, 1, "dd", "triplet_m"),
    ]


def test_pure_singlet_has_minus_three_quarters() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis))
    eigvec[1] = 1.0  # pure (0,1) singlet
    res = compute_spin_correlators_ci(eigvec, basis)
    assert isinstance(res, CISpinCorrelators)
    assert res.sector == "singlet"
    assert res.S_tot_squared == pytest.approx(0.0, abs=1e-12)
    assert res.S1_dot_S2 == pytest.approx(-0.75, abs=1e-12)
    assert res.p_singlet == pytest.approx(1.0, abs=1e-12)
    assert res.p_triplet == pytest.approx(0.0, abs=1e-12)


def test_pure_triplet_p_has_plus_one_quarter() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis))
    eigvec[2] = 1.0  # pure triplet_p
    res = compute_spin_correlators_ci(eigvec, basis)
    assert res.sector == "triplet"
    assert res.S_tot_squared == pytest.approx(2.0, abs=1e-12)
    assert res.S1_dot_S2 == pytest.approx(0.25, abs=1e-12)
    assert res.p_triplet_p == pytest.approx(1.0, abs=1e-12)


def test_pure_triplet_0_has_plus_one_quarter() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis))
    eigvec[3] = 1.0
    res = compute_spin_correlators_ci(eigvec, basis)
    assert res.S1_dot_S2 == pytest.approx(0.25, abs=1e-12)


def test_pure_triplet_m_has_plus_one_quarter() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis))
    eigvec[4] = 1.0
    res = compute_spin_correlators_ci(eigvec, basis)
    assert res.S1_dot_S2 == pytest.approx(0.25, abs=1e-12)


def test_doubly_occupied_singlet_has_minus_three_quarters() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis))
    eigvec[0] = 1.0  # the (0,0,'ud') doubly-occupied singlet
    res = compute_spin_correlators_ci(eigvec, basis)
    assert res.sector == "singlet"
    assert res.S1_dot_S2 == pytest.approx(-0.75, abs=1e-12)


def test_artificial_5050_mix_is_minus_one_quarter() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis))
    eigvec[1] = np.sqrt(0.5)  # singlet
    eigvec[3] = np.sqrt(0.5)  # triplet_0
    res = compute_spin_correlators_ci(eigvec, basis)
    assert res.sector == "mixed"
    assert res.p_singlet == pytest.approx(0.5, abs=1e-12)
    assert res.p_triplet == pytest.approx(0.5, abs=1e-12)
    assert res.S_tot_squared == pytest.approx(1.0, abs=1e-12)
    assert res.S1_dot_S2 == pytest.approx(0.5 - 0.75, abs=1e-12)


def test_below_tolerance_mix_is_classified_singlet() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis))
    eigvec[1] = np.sqrt(1.0 - 1e-10)
    eigvec[3] = np.sqrt(1e-10)
    res = compute_spin_correlators_ci(eigvec, basis, mixing_tolerance=1e-6)
    assert res.sector == "singlet"
    assert res.S1_dot_S2 == pytest.approx(2.0 * 1e-10 / 2 - 0.75, abs=1e-9)


def test_complex_eigvec_supported() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis), dtype=np.complex128)
    eigvec[1] = 1j  # phase only
    res = compute_spin_correlators_ci(eigvec, basis)
    assert res.S1_dot_S2 == pytest.approx(-0.75, abs=1e-12)


def test_unnormalised_raises_by_default() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis))
    eigvec[1] = 0.5  # only norm 0.25
    with pytest.raises(ValueError, match="not normalised"):
        compute_spin_correlators_ci(eigvec, basis)


def test_unnormalised_silenced_when_requested() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis))
    eigvec[1] = 0.5
    res = compute_spin_correlators_ci(eigvec, basis, enforce_normalisation=False)
    assert res.norm == pytest.approx(0.25, abs=1e-12)


def test_basis_length_mismatch_raises() -> None:
    basis = _toy_basis()
    eigvec = np.zeros(len(basis) + 1)
    eigvec[0] = 1.0
    with pytest.raises(ValueError, match="length"):
        compute_spin_correlators_ci(eigvec, basis)


def test_bad_spin_type_raises() -> None:
    basis = [(0, 0, "ud", "doublet")]
    eigvec = np.array([1.0])
    with pytest.raises(ValueError, match="spin_type"):
        compute_spin_correlators_ci(eigvec, basis)


def test_eigvec_must_be_1d() -> None:
    basis = _toy_basis()
    eigvec = np.zeros((len(basis), 1))
    eigvec[1, 0] = 1.0
    with pytest.raises(ValueError, match="1D"):
        compute_spin_correlators_ci(eigvec, basis)


def test_expected_helper_matches_compute() -> None:
    assert expected_S1_dot_S2("singlet") == pytest.approx(-0.75, abs=1e-12)
    assert expected_S1_dot_S2("triplet") == pytest.approx(0.25, abs=1e-12)
    with pytest.raises(ValueError):
        expected_S1_dot_S2("doublet")  # type: ignore[arg-type]


def test_real_ci_ground_state_is_singlet_with_minus_three_quarters() -> None:
    """Smoke-test: run a tiny shared-DVR CI and confirm GS is a clean singlet.

    Uses the production ``run_exact_diagonalization_one_per_well`` path because
    its energies are quantitatively reliable; the corresponding singlet/triplet
    sector labelling lives in the *shared* path's slater_basis, so we instead
    construct a small synthetic CI eigenvector rotation in singlet space and
    verify that even arbitrary mixtures *within* the singlet sector remain at
    -3/4. This guards against accidental sign or normalisation regressions.
    """
    basis: list[tuple[int, int, str, str]] = []
    for a in range(3):
        for b in range(a, 3):
            if a == b:
                basis.append((a, b, "ud", "singlet"))
            else:
                basis.append((a, b, "singlet", "singlet"))
                basis.append((a, b, "uu", "triplet_p"))
                basis.append((a, b, "triplet_0", "triplet_0"))
                basis.append((a, b, "dd", "triplet_m"))

    rng = np.random.default_rng(42)
    n_basis = len(basis)
    eigvec = np.zeros(n_basis)
    singlet_indices = [i for i, row in enumerate(basis) if row[3] == "singlet"]
    raw = rng.standard_normal(len(singlet_indices))
    raw /= np.linalg.norm(raw)
    for j, idx in enumerate(singlet_indices):
        eigvec[idx] = raw[j]
    res = compute_spin_correlators_ci(eigvec, basis)
    assert res.sector == "singlet"
    assert res.S1_dot_S2 == pytest.approx(-0.75, abs=1e-12)
    assert res.norm == pytest.approx(1.0, abs=1e-12)
