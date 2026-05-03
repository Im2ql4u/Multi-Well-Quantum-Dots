#!/usr/bin/env python3
"""Open-boundary Heisenberg chain ground state in the same Mott basis as the PINN.

This module builds the spin-1/2 OBC Heisenberg Hamiltonian

    H = sum_i J_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z)

in the fixed-magnetisation sector ``S^z = (n_up - n_down) / 2`` and
diagonalises it. The basis is enumerated in **the same order** as
:func:`observables.spin_amplitude_entanglement.enumerate_patterns`, so the
ground-state coefficients can be compared component-by-component against the
Mott amplitudes extracted from a trained ``GroundStateWF``.

This is the Phase 2A.4 cross-check: at large enough well separation ``d`` the
deep-Mott projection of the continuum Coulomb problem reduces to a Heisenberg
chain (with super-exchange-derived bond couplings ``J_i = 4 t_i^2 / U_i``).
A high overlap between the PINN-extracted ``c_sigma`` and the Heisenberg
ground state ``c_sigma^Heis`` is an unsupervised, parameter-free demonstration
that the trained network has discovered the correct spin physics.

Public API
==========
- :func:`enumerate_patterns_obc` — same convention as
  :func:`spin_amplitude_entanglement.enumerate_patterns`.
- :func:`build_heisenberg_obc_hamiltonian` — sparse-friendly dense matrix in
  the (n_up, n_down) sector.
- :func:`heisenberg_obc_ground_state` — diagonalises and returns
  ``(eigvals[:k], coeffs[:, 0])`` plus a useful payload.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def enumerate_patterns_obc(N: int, n_down: int) -> list[tuple[int, ...]]:
    """Same ordering convention as :func:`spin_amplitude_entanglement.enumerate_patterns`.

    Returns spin patterns ``sigma in {0, 1}^N`` (0=up, 1=down) with
    ``sum(sigma) == n_down`` in lexicographic order over the sorted index
    set of down sites.
    """
    if not 0 <= n_down <= N:
        raise ValueError(f"n_down must be in [0, N], got n_down={n_down}, N={N}.")
    out: list[tuple[int, ...]] = []
    for down_set in itertools.combinations(range(N), n_down):
        sigma = tuple(1 if i in down_set else 0 for i in range(N))
        out.append(sigma)
    return out


def _flip_pattern_bits(pattern: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    arr = list(pattern)
    arr[i], arr[j] = arr[j], arr[i]
    return tuple(arr)


def build_heisenberg_obc_hamiltonian(
    N: int,
    n_down: int,
    bond_couplings: Sequence[float] | None = None,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Build the dense OBC Heisenberg Hamiltonian in the ``(N - n_down, n_down)`` sector.

    Parameters
    ----------
    N
        Number of sites / particles.
    n_down
        Number of down spins (``S^z_total = (N - 2 n_down) / 2``).
    bond_couplings
        Length ``N - 1`` array of bond couplings ``J_i`` for the OBC chain
        ``S_0-S_1-S_2-...-S_{N-1}``. Defaults to uniform ``J_i = 1``.

    Returns
    -------
    (H, patterns)
        H is shape ``(P, P)`` with ``P = C(N, n_down)``; ``patterns`` is the
        list of basis states in the same order as the rows / columns of H.
    """
    if N < 2:
        raise ValueError("Need at least N=2 for an OBC bond.")
    if bond_couplings is None:
        bonds = np.ones(N - 1, dtype=np.float64)
    else:
        bonds = np.asarray(bond_couplings, dtype=np.float64)
        if bonds.size != N - 1:
            raise ValueError(
                f"bond_couplings must have length N-1 = {N - 1}, got {bonds.size}."
            )

    patterns = enumerate_patterns_obc(N, n_down)
    P = len(patterns)
    pattern_index = {sigma: i for i, sigma in enumerate(patterns)}
    H = np.zeros((P, P), dtype=np.float64)
    for i, sigma in enumerate(patterns):
        for b, J in enumerate(bonds):
            s_a = sigma[b]
            s_b = sigma[b + 1]
            if s_a == s_b:
                H[i, i] += 0.25 * J
            else:
                H[i, i] += -0.25 * J
                flipped = _flip_pattern_bits(sigma, b, b + 1)
                j = pattern_index.get(flipped)
                if j is not None:
                    H[i, j] += 0.5 * J
    return H, patterns


@dataclass
class HeisenbergReferenceResult:
    """Container for :func:`heisenberg_obc_ground_state`."""

    N: int
    n_down: int
    bond_couplings: np.ndarray
    patterns: list[tuple[int, ...]]
    eigenvalues: np.ndarray
    ground_amplitudes: np.ndarray
    multiplicity: int

    def amplitude_dict(self) -> dict[tuple[int, ...], float]:
        return {sigma: float(self.ground_amplitudes[i]) for i, sigma in enumerate(self.patterns)}


def heisenberg_obc_ground_state(
    N: int,
    n_down: int,
    bond_couplings: Sequence[float] | None = None,
    n_eigvals: int = 4,
    degenerate_tol: float = 1e-9,
) -> HeisenbergReferenceResult:
    """Diagonalise OBC Heisenberg in the requested sector and return GS amplitudes.

    The amplitudes are real (Heisenberg with real bond couplings is real) and
    normalised to unit L2 norm. Their sign convention is *only* fixed up to
    a global phase; the caller may want to align signs against the PINN
    extraction with :func:`align_amplitude_signs`.
    """
    H, patterns = build_heisenberg_obc_hamiltonian(N, n_down, bond_couplings)
    eigvals, eigvecs = np.linalg.eigh(H)
    n_eigvals = max(1, min(n_eigvals, eigvals.size))
    e0 = float(eigvals[0])
    multiplicity = int(np.sum(np.abs(eigvals - e0) < degenerate_tol))
    ground = eigvecs[:, 0].astype(np.float64)
    return HeisenbergReferenceResult(
        N=N,
        n_down=n_down,
        bond_couplings=np.asarray(bond_couplings if bond_couplings is not None else np.ones(N - 1, dtype=np.float64)),
        patterns=patterns,
        eigenvalues=eigvals[:n_eigvals],
        ground_amplitudes=ground,
        multiplicity=multiplicity,
    )


def align_amplitude_signs(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Flip ``candidate`` sign so it has positive overlap with ``reference``.

    Returns the (possibly sign-flipped) candidate and the resulting overlap
    ``<reference|candidate>``.
    """
    overlap = float(np.dot(reference, candidate))
    if overlap < 0:
        candidate = -candidate
        overlap = -overlap
    return candidate, overlap


__all__ = [
    "HeisenbergReferenceResult",
    "align_amplitude_signs",
    "build_heisenberg_obc_hamiltonian",
    "enumerate_patterns_obc",
    "heisenberg_obc_ground_state",
]
