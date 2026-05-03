"""Spin correlators from a 2-electron CI eigenvector on a shared-DVR Slater basis.

This is the second Phase-0 anchor (``A0.2``) called for in the grand plan
(``reports/2026-04-28_grand_plan_anchored.md``): given a CI eigenvector and the
energy-sorted Slater basis built by
``scripts.exact_diag_double_dot.build_slater_basis_sorted``, compute the
total-spin observables that pin down the spin sector of the state.

For two electrons,

.. math::

    \\mathbf{S}_1 \\cdot \\mathbf{S}_2 =
    \\tfrac{1}{2}\\big(\\mathbf{S}_{\\text{tot}}^2 - \\mathbf{S}_1^2 - \\mathbf{S}_2^2\\big)
    = \\tfrac{1}{2}\\mathbf{S}_{\\text{tot}}^2 - \\tfrac{3}{4},

so a singlet (:math:`S_{\\text{tot}} = 0`) gives :math:`-3/4` and any triplet
(:math:`S_{\\text{tot}} = 1`, :math:`M_S \\in \\{-1, 0, +1\\}`) gives :math:`+1/4`.
The energy-sorted Slater basis used elsewhere in the codebase already labels
each determinant with a definite (S, M_S) sector, and the spin-conserving
two-electron CI Hamiltonian is block-diagonal in those sectors, so any CI
eigenvector is **pure singlet or pure triplet** up to numerical noise. Mixed
populations larger than ``DEFAULT_MIXING_TOL`` therefore indicate a code/basis
bug and are reported for the caller to assert against.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

# ``spin_type`` strings agreed with ``scripts/exact_diag_double_dot.py``.
SINGLET = "singlet"
TRIPLET_P = "triplet_p"
TRIPLET_0 = "triplet_0"
TRIPLET_M = "triplet_m"

TRIPLET_TYPES: tuple[str, ...] = (TRIPLET_P, TRIPLET_0, TRIPLET_M)
ALL_TYPES: tuple[str, ...] = (SINGLET, *TRIPLET_TYPES)

DEFAULT_MIXING_TOL = 1.0e-6


@dataclass(frozen=True)
class CISpinCorrelators:
    """Bundle of total-spin observables for a single CI eigenvector."""

    norm: float
    p_singlet: float
    p_triplet_p: float
    p_triplet_0: float
    p_triplet_m: float
    S_tot_squared: float
    S1_dot_S2: float
    sector: Literal["singlet", "triplet", "mixed"]

    @property
    def p_triplet(self) -> float:
        return self.p_triplet_p + self.p_triplet_0 + self.p_triplet_m

    def to_dict(self) -> dict[str, float | str]:
        return {
            "norm": self.norm,
            "p_singlet": self.p_singlet,
            "p_triplet_p": self.p_triplet_p,
            "p_triplet_0": self.p_triplet_0,
            "p_triplet_m": self.p_triplet_m,
            "p_triplet": self.p_triplet,
            "S_tot_squared": self.S_tot_squared,
            "S1_dot_S2": self.S1_dot_S2,
            "sector": self.sector,
        }


def _validate_slater_basis(slater_basis: Iterable[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    rows = list(slater_basis)
    if not rows:
        raise ValueError("slater_basis is empty.")
    for idx, row in enumerate(rows):
        if not isinstance(row, tuple) or len(row) != 4:
            raise ValueError(
                f"slater_basis[{idx}] = {row!r} is not a 4-tuple (a, b, spin_cfg, spin_type)."
            )
        _, _, _, spin_type = row
        if spin_type not in ALL_TYPES:
            raise ValueError(
                f"slater_basis[{idx}] has spin_type='{spin_type}', expected one of {ALL_TYPES}."
            )
    return rows


def compute_spin_correlators_ci(
    eigvec: np.ndarray,
    slater_basis: Iterable[tuple[int, int, str, str]],
    *,
    mixing_tolerance: float = DEFAULT_MIXING_TOL,
    enforce_normalisation: bool = True,
    norm_tolerance: float = 1.0e-6,
) -> CISpinCorrelators:
    """Return ``<S_tot^2>``, ``<S_1 . S_2>``, and per-sector populations.

    Parameters
    ----------
    eigvec
        1D numpy array of CI coefficients of length ``len(slater_basis)``. May
        be real or complex; only the squared magnitudes contribute.
    slater_basis
        Energy-sorted Slater basis from
        ``scripts.exact_diag_double_dot.build_slater_basis_sorted``: a list of
        ``(a, b, spin_cfg, spin_type)`` tuples with
        ``spin_type in {'singlet','triplet_p','triplet_0','triplet_m'}``.
    mixing_tolerance
        Cross-sector population tolerance used to label the result as
        ``'singlet'``, ``'triplet'``, or ``'mixed'``. Computed values that
        violate ``min(p_singlet, p_triplet) > mixing_tolerance`` are flagged
        ``'mixed'``.
    enforce_normalisation
        If True (default), raise ``ValueError`` when ``|<Psi|Psi> - 1| >
        norm_tolerance``. Set False for diagnostic use on un-normalised states.
    norm_tolerance
        Numerical tolerance for the normalisation check.

    Returns
    -------
    CISpinCorrelators
        Frozen dataclass; see field docstring.
    """
    eigvec = np.asarray(eigvec)
    if eigvec.ndim != 1:
        raise ValueError(f"eigvec must be 1D, got shape {eigvec.shape}.")

    rows = _validate_slater_basis(slater_basis)
    if eigvec.shape[0] != len(rows):
        raise ValueError(
            f"eigvec length {eigvec.shape[0]} does not match slater_basis length {len(rows)}."
        )

    populations = {key: 0.0 for key in ALL_TYPES}
    weights_sq = np.abs(eigvec) ** 2
    for amp_sq, (_a, _b, _spin_cfg, spin_type) in zip(weights_sq, rows, strict=True):
        populations[spin_type] += float(amp_sq)

    norm = float(weights_sq.sum())
    if enforce_normalisation and abs(norm - 1.0) > norm_tolerance:
        raise ValueError(
            f"eigvec is not normalised: <Psi|Psi>={norm:.6e} (tol={norm_tolerance:.0e}). "
            f"Pass enforce_normalisation=False to silence this check."
        )

    p_singlet = populations[SINGLET]
    p_triplet_p = populations[TRIPLET_P]
    p_triplet_0 = populations[TRIPLET_0]
    p_triplet_m = populations[TRIPLET_M]
    p_triplet = p_triplet_p + p_triplet_0 + p_triplet_m

    s2 = 0.0 * p_singlet + 2.0 * p_triplet
    s1_dot_s2 = 0.5 * s2 - 0.75

    if min(p_singlet, p_triplet) <= mixing_tolerance:
        sector: Literal["singlet", "triplet", "mixed"] = (
            "singlet" if p_singlet > p_triplet else "triplet"
        )
    else:
        sector = "mixed"

    return CISpinCorrelators(
        norm=norm,
        p_singlet=p_singlet,
        p_triplet_p=p_triplet_p,
        p_triplet_0=p_triplet_0,
        p_triplet_m=p_triplet_m,
        S_tot_squared=s2,
        S1_dot_S2=s1_dot_s2,
        sector=sector,
    )


def expected_S1_dot_S2(sector: Literal["singlet", "triplet"]) -> float:
    """Return the analytical :math:`\\langle\\mathbf{S}_1 \\cdot \\mathbf{S}_2\\rangle`.

    Mirrors :func:`observables.heitler_london.spin_correlator_S1S2` so consumers
    can pull either reference without worrying about provenance.
    """
    if sector == "singlet":
        return -0.75
    if sector == "triplet":
        return 0.25
    raise ValueError(f"unknown sector '{sector}', expected 'singlet' or 'triplet'.")
