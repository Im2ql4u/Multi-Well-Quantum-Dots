"""Mott-projected spin-amplitude entanglement for one-electron-per-well systems.

This module implements the **Phase 2A flagship** capability of the inverse-design
stack: extracting bipartite spin entanglement of a trained ``GroundStateWF`` for
``N >= 2`` particles when the wavefunction lives in the Mott (one-per-well) regime.

Physical setup
==============
For ``N`` electrons on ``N`` wells with one electron per well, we work in the
deep-Mott approximation: each electron is localized at its well, and the only
remaining freedom is the spin pattern ``sigma in {up, down}^N``. A trained
NQS model with fixed spin template ``sigma_T`` (e.g. ``[up, up, ..., down, down]``
with ``n_up`` ups followed by ``n_down`` downs) represents

    |Psi> = integral_R Psi(R; sigma_T) a^dag_{sigma_T_1}(r_1) ... a^dag_{sigma_T_N}(r_N) |0>

In second quantization, the amplitude for finding the system in a basis state
|sigma> = a^dag_{sigma_1}(w_1) ... a^dag_{sigma_N}(w_N) |0> with one electron
of spin ``sigma_i`` at well ``w_i`` is

    c_sigma = <sigma|Psi> = sgn(pi_sigma) * Psi(R^*(sigma); sigma_T) / N!

where ``R^*(sigma)`` places the up-labeled particles (positions 1..n_up) at the
"up wells" of ``sigma`` (in increasing index order) and the down-labeled
particles (positions n_up+1..N) at the "down wells" (in increasing index order).
``sgn(pi_sigma)`` is the sign of the permutation that maps ``(1, 2, ..., N)``
to that interleaved sequence ``(up_wells_of_sigma..., down_wells_of_sigma...)``.

Only patterns with the trained sector ``(n_up, n_down)`` get nonzero amplitude;
other ``S^z`` sectors require independently-trained models.

Mott validity
=============
The procedure is exact for the Mott projector. The off-Mott contamination
(double occupancy, hopping fluctuations) is suppressed as ``exp(-d^2/2)`` for
isotropic harmonic-oscillator wells of unit ``omega``. At ``d >= 4`` this is
``< 0.04%``; at ``d >= 6`` it is ``< 1e-7``.

Public API
==========
- ``enumerate_patterns(N, n_down)`` -- list of all sigma in {0,1}^N with sum=n_down.
- ``localized_config_for_pattern(pattern, well_centers)`` -- builds R^*(sigma).
- ``permutation_sign(pattern)`` -- sgn(pi_sigma) computed from inversions.
- ``extract_spin_amplitudes(loaded, ...)`` -- evaluate Psi at all R^*(sigma).
- ``well_set_bipartite_entropy(payload, set_a)`` -- entanglement entropy and
  negativity for the bipartition ``A | B`` with ``A = set_a``.
- ``evaluate_spin_amplitude_entanglement(result_dir, ...)`` -- end-to-end load +
  extract + analyze.
- ``spin_entanglement_target(result_dir, ...)`` -- single scalar target.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from observables.checkpoint_loader import LoadedWavefunction, load_wavefunction_from_dir


LOGGER = logging.getLogger("spin_amplitude_entanglement")

_AMPLITUDE_EPS = 1e-30
_PROB_EPS = 1e-15


# ---------------------------------------------------------------------------
# Core combinatorial helpers
# ---------------------------------------------------------------------------


def enumerate_patterns(N: int, n_down: int) -> list[tuple[int, ...]]:
    """All spin patterns ``sigma in {0, 1}^N`` with ``sum(sigma) == n_down``.

    The convention is ``0 = up``, ``1 = down``. There are ``C(N, n_down)``
    such patterns; for ``N=4, n_down=2`` we get 6 patterns; for ``N=8,
    n_down=4`` we get 70 patterns.
    """
    if not 0 <= n_down <= N:
        raise ValueError(f"n_down must be in [0, N], got n_down={n_down}, N={N}.")
    patterns: list[tuple[int, ...]] = []
    for down_set in itertools.combinations(range(N), n_down):
        sigma = tuple(1 if i in down_set else 0 for i in range(N))
        patterns.append(sigma)
    return patterns


def localized_config_for_pattern(
    pattern: Sequence[int],
    well_centers: np.ndarray,
) -> np.ndarray:
    """Build ``R^*(sigma)`` of shape ``(N, dim)``.

    The layout is:

      * particles ``1..n_up`` placed at the up-wells of sigma in increasing index;
      * particles ``n_up+1..N`` placed at the down-wells of sigma in increasing index.
    """
    pattern_arr = np.asarray(pattern, dtype=np.int64)
    if pattern_arr.ndim != 1:
        raise ValueError(f"pattern must be 1D, got shape {pattern_arr.shape}.")
    centers = np.asarray(well_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] != pattern_arr.shape[0]:
        raise ValueError(
            f"well_centers must have shape (N, dim) with N={pattern_arr.shape[0]}, "
            f"got {centers.shape}."
        )
    up_wells = np.flatnonzero(pattern_arr == 0).tolist()
    down_wells = np.flatnonzero(pattern_arr == 1).tolist()
    ordered = up_wells + down_wells
    return centers[ordered].copy()


def permutation_sign(pattern: Sequence[int]) -> int:
    """Sign of the permutation that interleaves up-then-down well indices.

    Given a pattern ``sigma``, the permutation in question is the one that
    maps the natural order ``(0, 1, ..., N-1)`` to the sequence
    ``(up_wells_of_sigma..., down_wells_of_sigma...)``. We compute its parity
    by counting inversions in that sequence.
    """
    pattern_arr = np.asarray(pattern, dtype=np.int64)
    up_wells = np.flatnonzero(pattern_arr == 0).tolist()
    down_wells = np.flatnonzero(pattern_arr == 1).tolist()
    sequence = up_wells + down_wells
    inversions = 0
    n = len(sequence)
    for i in range(n):
        for j in range(i + 1, n):
            if sequence[i] > sequence[j]:
                inversions += 1
    return -1 if (inversions & 1) else 1


# ---------------------------------------------------------------------------
# Spin-amplitude extraction
# ---------------------------------------------------------------------------


@dataclass
class SpinAmplitudePayload:
    """Container for the raw output of :func:`extract_spin_amplitudes`.

    Fields
    ------
    pattern
        Ordered list of all spin patterns evaluated.
    log_abs_psi
        ``log|Psi(R^*(sigma); sigma_T)|`` for each pattern, shape ``(P,)``.
    sign_psi
        Sign returned by ``model.signed_log_psi`` for each pattern.
    perm_sign
        Permutation sign ``sgn(pi_sigma)`` per pattern, shape ``(P,)``.
    """

    pattern: list[tuple[int, ...]]
    log_abs_psi: np.ndarray
    sign_psi: np.ndarray
    perm_sign: np.ndarray
    sigma_z_total: int
    n_up: int
    n_down: int
    n_wells: int
    well_centers: np.ndarray

    def signed_log_amplitudes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(sign_c_sigma, log|c_sigma|_unnormalised)``.

        The combined sign ``sgn(pi_sigma) * sign(Psi)`` factors out the
        permutation sign explicitly so downstream code can sum amplitudes
        without recomputing it. The log-amplitude is the same for all
        patterns (no normalisation applied yet).
        """
        return self.sign_psi * self.perm_sign, self.log_abs_psi

    def normalised(self) -> np.ndarray:
        """Return ``c_sigma`` as a unit-L2 vector indexed by ``self.pattern``.

        Numerically stable: factors out ``max(log|Psi|)`` before exponentiating.
        """
        signs, logs = self.signed_log_amplitudes()
        log_max = float(np.max(logs))
        rel = signs * np.exp(logs - log_max)
        norm = float(np.linalg.norm(rel))
        if norm < _AMPLITUDE_EPS:
            raise RuntimeError(
                "Spin amplitude vector has essentially zero norm; cannot normalise."
            )
        return rel / norm


def extract_spin_amplitudes(
    loaded: LoadedWavefunction,
    *,
    spin_sector: tuple[int, int] | None = None,
    well_centers: np.ndarray | None = None,
) -> SpinAmplitudePayload:
    """Evaluate ``Psi`` at every localised configuration to extract Mott amplitudes.

    Parameters
    ----------
    loaded
        Loaded wavefunction (model + system).
    spin_sector
        Optional ``(n_up, n_down)`` to evaluate. By default uses the model's
        training sector. Cross-sector evaluation is not yet supported (the
        model is structurally restricted to one ``S^z`` sector).
    well_centers
        Optional override; defaults to the centers stored in
        ``loaded.system.wells``.
    """
    system = loaded.system
    N = int(system.n_particles)
    n_wells = len(system.wells)
    if N != n_wells:
        raise ValueError(
            f"Mott projection requires one particle per well; got N={N}, "
            f"n_wells={n_wells}."
        )

    if well_centers is None:
        well_centers = np.asarray([w.center for w in system.wells], dtype=np.float64)
    else:
        well_centers = np.asarray(well_centers, dtype=np.float64)
    if well_centers.shape != (N, system.dim):
        raise ValueError(
            f"well_centers must have shape (N, dim) = ({N}, {system.dim}), "
            f"got {well_centers.shape}."
        )

    n_up_train = int(loaded.model.sd_params["n_up"])
    n_down_train = int(loaded.model.sd_params["n_down"])
    if spin_sector is None:
        n_up, n_down = n_up_train, n_down_train
    else:
        n_up, n_down = spin_sector
        if (n_up, n_down) != (n_up_train, n_down_train):
            raise NotImplementedError(
                f"Cross-sector evaluation not supported (model trained at "
                f"({n_up_train}, {n_down_train}), requested ({n_up}, {n_down})). "
                f"Train a separate model in the requested sector."
            )
    if n_up + n_down != N:
        raise ValueError(
            f"n_up + n_down = {n_up + n_down}, expected N = {N}."
        )

    patterns = enumerate_patterns(N, n_down)
    if not patterns:
        raise RuntimeError(f"No patterns produced for (N={N}, n_down={n_down}).")

    P = len(patterns)
    R_batch = np.zeros((P, N, int(system.dim)), dtype=np.float64)
    perm_signs = np.zeros(P, dtype=np.float64)
    for i, sigma in enumerate(patterns):
        R_batch[i] = localized_config_for_pattern(sigma, well_centers)
        perm_signs[i] = float(permutation_sign(sigma))

    R_t = torch.from_numpy(R_batch).to(device=loaded.device, dtype=loaded.dtype)
    sign_t, logp_t = loaded.signed_log_psi(R_t)
    sign_np = sign_t.detach().cpu().numpy().astype(np.float64)
    logp_np = logp_t.detach().cpu().numpy().astype(np.float64)

    if not np.all(np.isfinite(logp_np)):
        raise RuntimeError("Non-finite log|Psi| value encountered during extraction.")

    return SpinAmplitudePayload(
        pattern=patterns,
        log_abs_psi=logp_np,
        sign_psi=sign_np,
        perm_sign=perm_signs,
        sigma_z_total=n_up - n_down,
        n_up=n_up,
        n_down=n_down,
        n_wells=n_wells,
        well_centers=well_centers,
    )


# ---------------------------------------------------------------------------
# Bipartite entanglement
# ---------------------------------------------------------------------------


def _von_neumann_entropy(probs: np.ndarray) -> float:
    safe = probs[probs > _PROB_EPS]
    return float(-np.sum(safe * np.log(safe)))


def _spin_negativity(M: np.ndarray) -> dict[str, float]:
    """Negativity of |c><c| under partial transpose on B (well-set bipartition)."""
    norm = float(np.sum(M.conj() * M).real)
    if norm < _AMPLITUDE_EPS:
        return {
            "negativity": 0.0,
            "log_negativity": 0.0,
            "min_eigenvalue": 0.0,
            "n_negative_eigenvalues": 0,
        }
    M_n = M / np.sqrt(norm)
    rho = np.einsum("ab,cd->abcd", M_n, M_n.conj())
    dim_a, dim_b = M_n.shape
    rho_pt = np.transpose(rho, (0, 3, 2, 1)).reshape(dim_a * dim_b, dim_a * dim_b)
    rho_pt = 0.5 * (rho_pt + rho_pt.conj().T)
    eigvals = np.linalg.eigvalsh(rho_pt)
    negativity = float(np.sum((np.abs(eigvals) - eigvals) / 2.0))
    log_neg = float(np.log2(2.0 * negativity + 1.0)) if negativity > _PROB_EPS else 0.0
    return {
        "negativity": negativity,
        "log_negativity": log_neg,
        "min_eigenvalue": float(np.min(eigvals)),
        "n_negative_eigenvalues": int(np.sum(eigvals < -1e-12)),
    }


def well_set_bipartite_entropy(
    payload: SpinAmplitudePayload,
    *,
    set_a: Iterable[int],
) -> dict[str, Any]:
    """Compute spin-Schmidt decomposition for the well-set bipartition ``A | B``.

    Parameters
    ----------
    payload
        Output of :func:`extract_spin_amplitudes`.
    set_a
        Indices of wells to put in subsystem A.

    Returns
    -------
    dict with:
      * ``set_a``, ``set_b``: well-index partitions.
      * ``schmidt_values``: singular values of the
        ``2^|A| x 2^|B|`` amplitude matrix ``c[sigma_A, sigma_B]``.
      * ``schmidt_probs``: squared & normalised Schmidt values.
      * ``von_neumann_entropy``: ``S = -sum p_i ln p_i``.
      * ``purity``, ``linear_entropy``, ``effective_schmidt_rank``.
      * ``negativity``, ``log_negativity``: from PT of |c><c|.
      * Sub-keys describing matrix shape and norm.
    """
    n_wells = payload.n_wells
    set_a_sorted = sorted({int(i) for i in set_a})
    if not set_a_sorted:
        raise ValueError("set_a must be non-empty.")
    if any(i < 0 or i >= n_wells for i in set_a_sorted):
        raise ValueError(f"set_a indices must lie in [0, {n_wells}); got {set_a_sorted}.")
    set_b_sorted = sorted(set(range(n_wells)) - set(set_a_sorted))
    if not set_b_sorted:
        raise ValueError("Bipartition complement is empty; A must be a strict subset.")

    # Allocate amplitude matrix M[sigma_A, sigma_B] over the FULL spin Hilbert
    # space of each subsystem (dim 2^|A|, 2^|B|). Patterns outside the trained
    # sector remain zero, which is the physically correct restriction.
    a_size = 1 << len(set_a_sorted)
    b_size = 1 << len(set_b_sorted)
    M = np.zeros((a_size, b_size), dtype=np.float64)

    c_normalised = payload.normalised()
    set_a_arr = np.asarray(set_a_sorted, dtype=np.int64)
    set_b_arr = np.asarray(set_b_sorted, dtype=np.int64)
    for sigma, amp in zip(payload.pattern, c_normalised):
        sigma_arr = np.asarray(sigma, dtype=np.int64)
        sigma_a = sigma_arr[set_a_arr]
        sigma_b = sigma_arr[set_b_arr]
        # Big-endian encoding: bit 0 of the index corresponds to the first
        # well of the partition in increasing index order.
        idx_a = int(np.dot(sigma_a, 1 << np.arange(len(set_a_sorted))[::-1]))
        idx_b = int(np.dot(sigma_b, 1 << np.arange(len(set_b_sorted))[::-1]))
        M[idx_a, idx_b] = amp

    sigma_svd = np.linalg.svd(M, compute_uv=False, full_matrices=False)
    norm_sq = float(np.sum(sigma_svd**2))
    if norm_sq < _AMPLITUDE_EPS:
        raise RuntimeError("Bipartite amplitude matrix has essentially zero norm.")
    probs = sigma_svd**2 / norm_sq
    purity = float(np.sum(probs**2))
    eff_rank = int(np.sum(probs > 0.01 * probs[0])) if probs.size else 0

    neg = _spin_negativity(M)

    return {
        "set_a": set_a_sorted,
        "set_b": set_b_sorted,
        "n_a_states": int(a_size),
        "n_b_states": int(b_size),
        "amplitude_matrix_l2_norm_sq": norm_sq,
        "schmidt_values": sigma_svd.tolist(),
        "schmidt_probs": probs.tolist(),
        "von_neumann_entropy": _von_neumann_entropy(probs),
        "purity": purity,
        "linear_entropy": 1.0 - purity,
        "effective_schmidt_rank": eff_rank,
        "negativity": neg["negativity"],
        "log_negativity": neg["log_negativity"],
        "min_eigenvalue_pt": neg["min_eigenvalue"],
        "n_negative_eigenvalues_pt": neg["n_negative_eigenvalues"],
    }


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------


def evaluate_spin_amplitude_entanglement(
    result_dir: Path | str,
    *,
    set_a: Sequence[int] | None = None,
    spin_sector: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
    return_amplitudes: bool = True,
) -> dict[str, Any]:
    """End-to-end: load checkpoint, extract amplitudes, compute entanglement."""
    loaded = load_wavefunction_from_dir(result_dir, device=device)
    payload = extract_spin_amplitudes(loaded, spin_sector=spin_sector)
    n_wells = payload.n_wells
    if set_a is None:
        half = n_wells // 2
        if half == 0:
            raise ValueError("Cannot bipartition a 1-well system.")
        set_a = list(range(half))

    bipartite = well_set_bipartite_entropy(payload, set_a=set_a)

    out: dict[str, Any] = {
        "result_dir": str(loaded.result_dir),
        "n_particles": int(loaded.system.n_particles),
        "n_wells": int(n_wells),
        "spin_sector": {"n_up": payload.n_up, "n_down": payload.n_down},
        "well_centers": payload.well_centers.tolist(),
        "set_a": list(set_a),
        "bipartite": bipartite,
    }
    if return_amplitudes:
        c = payload.normalised()
        out["amplitudes"] = {
            "patterns": [list(p) for p in payload.pattern],
            "amplitudes_normalised": c.tolist(),
            "log_abs_psi_raw": payload.log_abs_psi.tolist(),
            "sign_psi_raw": payload.sign_psi.tolist(),
            "perm_sign": payload.perm_sign.tolist(),
        }
    return out


def spin_entanglement_target(
    result_dir: Path | str,
    *,
    set_a: Sequence[int] | None = None,
    metric: str = "von_neumann_entropy",
    spin_sector: tuple[int, int] | None = None,
) -> float:
    """Reduce bipartite spin entanglement to a single scalar target.

    Available metrics include ``"von_neumann_entropy"``, ``"negativity"``,
    ``"log_negativity"``, ``"linear_entropy"``, and ``"effective_schmidt_rank"``.
    """
    payload = evaluate_spin_amplitude_entanglement(
        result_dir,
        set_a=set_a,
        spin_sector=spin_sector,
        return_amplitudes=False,
    )
    bipartite = payload["bipartite"]
    if metric in bipartite and isinstance(bipartite[metric], (int, float)):
        return float(bipartite[metric])
    raise ValueError(
        f"Unknown or non-scalar metric '{metric}'. Available scalar metrics: "
        f"{sorted(k for k, v in bipartite.items() if isinstance(v, (int, float)))}."
    )


# ---------------------------------------------------------------------------
# Sanity helpers (used by the validation script)
# ---------------------------------------------------------------------------


def expected_singlet_amplitudes(N: int = 2) -> dict[tuple[int, ...], float]:
    """Hard-coded reference: single-particle Bell singlet for N=2.

    Useful as a unit-test reference: ``c_(0,1) = +1/sqrt(2)``,
    ``c_(1,0) = -1/sqrt(2)``.
    """
    if N != 2:
        raise NotImplementedError("Reference singlet only known for N=2.")
    return {(0, 1): 1.0 / math.sqrt(2.0), (1, 0): -1.0 / math.sqrt(2.0)}


def expected_singlet_entanglement_n2() -> dict[str, float]:
    """Reference values: S=ln(2), negativity=1/2, log-negativity=1."""
    return {
        "von_neumann_entropy": math.log(2.0),
        "negativity": 0.5,
        "log_negativity": 1.0,
    }
