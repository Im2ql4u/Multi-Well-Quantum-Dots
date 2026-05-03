"""Effective Heisenberg ``J_ij`` analysis + direct spin correlators from PINN amplitudes.

Two related tools live here:

1. **Direct spin-spin correlators** ``C_{ij} = <c|S_i.S_j|c>``
   (:func:`spin_pair_correlator`, :func:`spin_correlator_target`). These are
   *unambiguous*, *single-state* observables computed directly from the
   normalised PINN amplitudes. They are the cleanest scalar to drive in
   inverse design: e.g. ``C_{0, N-1}`` measures the engineered long-range
   spin coupling end-to-end, and is the headline Phase 2B target.

2. **Parent-Hamiltonian (covariance) fit** ``H_eff = sum J_{ij} S_i.S_j``
   (:func:`fit_effective_heisenberg`). Uses the covariance method
   (Qi & Ranard 2019, Bairey-Arad-Lindner 2019): the J-vector lies in the
   null space of the covariance matrix
   ``Q[a,b] = <O_a O_b> - <O_a><O_b>``. For SU(2)-invariant Hamiltonians
   the null space is **multi-dimensional** whenever the ground state lives
   in a spin sector of dim ``> 1`` (e.g. N≥4 chains with multiple
   singlets). This means the parent Hamiltonian is genuinely
   non-unique from a single ground state, and *no method that uses only
   |c⟩ can recover the true bonds*. We resolve this by picking the
   null-space direction that makes ``|c⟩`` the *ground state* of
   ``H_eff(J)``. Use this for *qualitative* analysis ("how Heisenberg-like
   is the trained PINN?") and for the ``effective_schmidt_rank`` /
   ``relative_residual`` metrics — but **prefer** :func:`spin_pair_correlator`
   for inverse-design targets.

Public API
==========
- :func:`apply_pair_S_dot_S` — efficient action of a single ``S_i.S_j`` on
  a sector-restricted amplitude vector.
- :func:`build_pair_operator_matrix` — the same operator as a dense
  ``P x P`` matrix.
- :func:`spin_pair_correlator` — ``<c|S_i.S_j|c>`` for a list of pairs, plus
  the full ``C`` matrix.
- :func:`evaluate_spin_correlators` — load a checkpoint, return all
  ``<S_i.S_j>`` plus connected correlators ``<S_i.S_j> - <S_i>.<S_j>``.
- :func:`spin_correlator_target` — single scalar reduction for inverse
  design (with ``mode='value' | 'neg_squared_error' | 'squared_error'``).
- :func:`fit_effective_heisenberg` — the parent-Hamiltonian fit.
- :func:`evaluate_effective_heisenberg` — load + fit + JSON summary.
- :func:`effective_J_target` — scalar reduction of fitted ``J_{ij}``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from observables.checkpoint_loader import load_wavefunction_from_dir
from observables.spin_amplitude_entanglement import (
    SpinAmplitudePayload,
    extract_spin_amplitudes,
)


LOGGER = logging.getLogger("effective_heisenberg")


_AMPLITUDE_EPS = 1e-30
_DEFAULT_DEGENERATE_TOL = 1e-9


# ---------------------------------------------------------------------------
# Action of S_i . S_j on a sector-restricted vector
# ---------------------------------------------------------------------------


def apply_pair_S_dot_S(
    c: np.ndarray,
    patterns: Sequence[tuple[int, ...]],
    i: int,
    j: int,
    pattern_index: dict[tuple[int, ...], int] | None = None,
) -> np.ndarray:
    """Compute ``(S_i . S_j) |c>`` over a fixed-S^z basis.

    Acts on the same enumerated basis used by
    :mod:`observables.spin_amplitude_entanglement` and
    :mod:`observables.heisenberg_reference`. The pair operator preserves
    the magnetisation sector, so the output lives in the same space.
    """
    if i == j:
        raise ValueError("Pair (i, j) requires i != j.")
    if pattern_index is None:
        pattern_index = {sigma: idx for idx, sigma in enumerate(patterns)}
    P = len(patterns)
    if c.shape != (P,):
        raise ValueError(f"c must have shape ({P},), got {c.shape}.")
    out = np.zeros(P, dtype=np.float64)
    for idx, sigma in enumerate(patterns):
        s_a = sigma[i]
        s_b = sigma[j]
        if s_a == s_b:
            out[idx] += 0.25 * c[idx]
        else:
            out[idx] -= 0.25 * c[idx]
            flipped = list(sigma)
            flipped[i], flipped[j] = flipped[j], flipped[i]
            tgt = pattern_index.get(tuple(flipped))
            if tgt is not None:
                out[tgt] += 0.5 * c[idx]
    return out


def build_pair_operator_matrix(
    patterns: Sequence[tuple[int, ...]],
    i: int,
    j: int,
) -> np.ndarray:
    """Dense ``P x P`` matrix for ``S_i . S_j`` on the fixed-S^z basis."""
    P = len(patterns)
    pattern_index = {sigma: idx for idx, sigma in enumerate(patterns)}
    M = np.zeros((P, P), dtype=np.float64)
    for idx, sigma in enumerate(patterns):
        s_a = sigma[i]
        s_b = sigma[j]
        if s_a == s_b:
            M[idx, idx] += 0.25
        else:
            M[idx, idx] -= 0.25
            flipped = list(sigma)
            flipped[i], flipped[j] = flipped[j], flipped[i]
            tgt = pattern_index.get(tuple(flipped))
            if tgt is not None:
                M[tgt, idx] += 0.5
    return M


# ---------------------------------------------------------------------------
# Direct spin-spin correlators (preferred inverse-design observable)
# ---------------------------------------------------------------------------


def _site_sz_expectation(
    c: np.ndarray, patterns: Sequence[tuple[int, ...]], n_wells: int
) -> np.ndarray:
    """``<c|S_i^z|c>`` per site (for connected correlators)."""
    sz = np.zeros(n_wells, dtype=np.float64)
    p2 = c * c
    for idx, sigma in enumerate(patterns):
        for i in range(n_wells):
            sz[i] += p2[idx] * (0.5 if sigma[i] == 0 else -0.5)
    return sz


def spin_pair_correlator(
    payload: SpinAmplitudePayload,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Compute ``<c|S_i.S_j|c>`` for a list of well pairs.

    Returns
    -------
    dict with keys:
      * ``pairs``           — list of fitted ``(i, j)`` pairs
      * ``correlators``     — list of ``<c|S_i.S_j|c>`` values
      * ``c_matrix``        — symmetric ``n_wells x n_wells`` matrix
        of correlators (diagonal is set to ``<c|S_i^2|c> = 3/4``)
      * ``sz_per_site``     — ``<S_i^z>`` per site
      * ``connected``       — ``C_ij - <S_i^z>*<S_j^z>`` (longitudinal
        connected correlator; useful when total ``S^z`` symmetry is
        broken).
    """
    n_wells = payload.n_wells
    patterns = list(payload.pattern)
    P = len(patterns)
    pattern_index = {sigma: idx for idx, sigma in enumerate(patterns)}
    c = payload.normalised()
    pair_list = list(pairs) if pairs is not None else _default_pairs(n_wells)
    for (i, j) in pair_list:
        if not (0 <= i < n_wells and 0 <= j < n_wells and i < j):
            raise ValueError(
                f"Invalid pair (i, j) = ({i}, {j}); require 0 <= i < j < {n_wells}."
            )

    correlators = np.zeros(len(pair_list), dtype=np.float64)
    c_matrix = np.zeros((n_wells, n_wells), dtype=np.float64)
    for k, (i, j) in enumerate(pair_list):
        Oc = apply_pair_S_dot_S(c, patterns, i, j, pattern_index=pattern_index)
        v = float(c.dot(Oc))
        correlators[k] = v
        c_matrix[i, j] = v
        c_matrix[j, i] = v
    np.fill_diagonal(c_matrix, 0.75)

    sz = _site_sz_expectation(c, patterns, n_wells)
    connected = np.zeros(len(pair_list), dtype=np.float64)
    for k, (i, j) in enumerate(pair_list):
        connected[k] = correlators[k] - sz[i] * sz[j]

    return {
        "n_wells": int(n_wells),
        "pairs": [list(p) for p in pair_list],
        "correlators": correlators.tolist(),
        "c_matrix": c_matrix.tolist(),
        "sz_per_site": sz.tolist(),
        "connected": connected.tolist(),
    }


def evaluate_spin_correlators(
    result_dir: Path | str,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    spin_sector: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load checkpoint, extract amplitudes, return ``<S_i.S_j>`` correlators."""
    loaded = load_wavefunction_from_dir(result_dir, device=device)
    payload = extract_spin_amplitudes(loaded, spin_sector=spin_sector)
    out = spin_pair_correlator(payload, pairs=pairs)
    out["result_dir"] = str(loaded.result_dir)
    out["n_particles"] = int(loaded.system.n_particles)
    out["spin_sector"] = {"n_up": payload.n_up, "n_down": payload.n_down}
    out["well_centers"] = payload.well_centers.tolist()
    return out


def spin_correlator_target(
    result_dir: Path | str,
    *,
    pair: Sequence[int],
    target_value: float | None = None,
    mode: str = "value",
    spin_sector: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
) -> float:
    """Single-scalar reduction of ``<c|S_i.S_j|c>`` for inverse design.

    Modes
    -----
    ``"value"``
        Return the raw correlator (use to push ``<S_i.S_j>`` *more
        negative*: gradient ascent maximises it; for AFM correlators we
        usually want gradient *descent* on ``-value`` so flip the sign in
        the optimizer).
    ``"neg_value"``
        Return ``-<S_i.S_j>``. Use for gradient ascent on AFM strength
        (more negative correlator → more positive target).
    ``"neg_squared_error"``
        Return ``-(<S_i.S_j> - target)^2``. Use to drive the correlator
        *toward* ``target_value`` — gradient ascent.
    ``"squared_error"``
        Return ``(<S_i.S_j> - target)^2``; for explicit minimisation.
    """
    if len(pair) != 2:
        raise ValueError(f"`pair` must have length 2, got {pair}.")
    i, j = int(pair[0]), int(pair[1])
    if i == j:
        raise ValueError("pair must have i != j.")
    a, b = (i, j) if i < j else (j, i)
    out = evaluate_spin_correlators(
        result_dir,
        pairs=[(a, b)],
        spin_sector=spin_sector,
        device=device,
    )
    val = float(out["correlators"][0])
    if mode == "value":
        return val
    if mode == "neg_value":
        return -val
    if target_value is None:
        raise ValueError(f"mode='{mode}' requires target_value.")
    err = val - float(target_value)
    if mode == "neg_squared_error":
        return -float(err * err)
    if mode == "squared_error":
        return float(err * err)
    raise ValueError(
        f"Unknown mode '{mode}'. Choose 'value', 'neg_value', "
        f"'neg_squared_error', or 'squared_error'."
    )


# ---------------------------------------------------------------------------
# Fit dataclass + main fitting routine
# ---------------------------------------------------------------------------


@dataclass
class EffectiveHeisenbergFit:
    """Output of :func:`fit_effective_heisenberg`.

    Fields
    ------
    n_wells
        Number of wells / sites.
    pairs
        Ordered list of ``(i, j)`` with ``i < j`` whose couplings were fit.
    j_vector
        Best-fit ``J_{(i,j)}`` for each pair, in the same order as ``pairs``.
        Normalisation: ``j_vector`` is fixed by demanding the **nearest-
        neighbour bond ``(0, 1)`` equals 1**, with a positive sign by
        convention. To recover the natural ``J`` in Hartree, multiply by
        ``j_scale`` (which carries the absolute scale picked up from the
        fit's <H>/||J|| ratio is *not* directly available from the
        covariance method; see ``energy_split`` instead).
    j_matrix
        Symmetric ``n_wells x n_wells`` matrix with diagonal zero,
        ``J_matrix[i, j] = J_matrix[j, i] = j_vector[k]``.
    residual_variance
        Smallest eigenvalue of the covariance matrix ``Q``. With the chosen
        normalisation ``||j_vector||_2 = 1`` *before* rescaling to the
        ``J_(0,1) = 1`` convention, this equals the variance
        ``Var_c(H_eff)`` in units of (energy * ||J||)^2. A value of
        ``0`` indicates a perfect Heisenberg parent Hamiltonian. We
        report it both raw and relative to the largest eigenvalue
        (``relative_residual``).
    relative_residual
        ``residual_variance / largest_covariance_eigenvalue``. A
        dimensionless quality metric, ``< 1e-3`` indicates a near-exact
        Heisenberg description.
    energy_split
        Difference between the lowest two energy eigenvalues of the fitted
        ``H_eff`` (in units where ``J_(0,1) = 1``). Useful for setting the
        physical energy scale once one bond's value is known
        (e.g. via super-exchange).
    overlap_with_ground
        ``|<c|psi_0>|`` where ``psi_0`` is the ground state of the fitted
        ``H_eff`` in the same sector. Approaches 1 for a perfect fit.
    """

    n_wells: int
    pairs: list[tuple[int, int]]
    j_vector: np.ndarray
    j_matrix: np.ndarray
    means: np.ndarray
    covariance_eigenvalues: np.ndarray
    residual_variance: float
    relative_residual: float
    overlap_with_ground: float
    energy_split: float

    def get_pair(self, i: int, j: int) -> float:
        """Return ``J_{(min(i,j), max(i,j))}`` from the fit (raises if missing)."""
        a, b = (i, j) if i < j else (j, i)
        return float(self.j_matrix[a, b])

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_wells": int(self.n_wells),
            "pairs": [list(p) for p in self.pairs],
            "j_vector": self.j_vector.tolist(),
            "j_matrix": self.j_matrix.tolist(),
            "means": self.means.tolist(),
            "covariance_eigenvalues": self.covariance_eigenvalues.tolist(),
            "residual_variance": float(self.residual_variance),
            "relative_residual": float(self.relative_residual),
            "overlap_with_ground": float(self.overlap_with_ground),
            "energy_split": float(self.energy_split),
        }


def _default_pairs(n_wells: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_wells) for j in range(i + 1, n_wells)]


def _build_h_eff(
    patterns: Sequence[tuple[int, ...]],
    pair_list: Sequence[tuple[int, int]],
    j_vec: np.ndarray,
) -> np.ndarray:
    P = len(patterns)
    H_eff = np.zeros((P, P), dtype=np.float64)
    for k, (i, j) in enumerate(pair_list):
        H_eff += float(j_vec[k]) * build_pair_operator_matrix(patterns, i, j)
    H_eff = 0.5 * (H_eff + H_eff.T)
    return H_eff


def _ground_state_overlap(
    H_eff: np.ndarray, c: np.ndarray
) -> tuple[float, float]:
    ev, vc = np.linalg.eigh(H_eff)
    ground = vc[:, 0]
    overlap = float(abs(np.dot(c, ground)))
    split = float(ev[1] - ev[0]) if ev.size >= 2 else float("nan")
    return overlap, split


def _resolve_J_direction(
    basis: np.ndarray,
    patterns: Sequence[tuple[int, ...]],
    pair_list: Sequence[tuple[int, int]],
    c: np.ndarray,
    means: np.ndarray | None = None,
    *,
    n_angles_per_dim: int = 24,
    n_random: int = 96,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Find the unit ``J`` vector (in ``span(basis)``) that maximises the
    overlap ``|<c|psi_0(H_eff(J))>|`` between ``c`` and the ground state of
    ``H_eff(J) = sum_k J_k S_{i_k}.S_{j_k}``.

    Strategy
    --------
    * ``basis`` is a ``K x d`` matrix whose columns span the search space
      (typically: all ``K`` covariance eigenvectors when ``c`` is approximate,
      or only the ``null(Q)`` columns when ``c`` is an exact eigenstate).
    * For ``d == 1`` the answer is unique up to sign; we just check both signs.
    * For ``d >= 2`` we generate many starting points (axis directions, the
      uniform direction projected onto the basis, random unit vectors, plus a
      dense angular grid for ``d == 2``), score each by overlap with ``|c>``,
      and polish the best ``n_polish`` candidates with Nelder-Mead. The best
      polished score wins.

    Returns ``(j_vec, overlap)`` with ``||j_vec|| == 1`` (unit norm in the
    full ``K``-dim ambient space; basis is assumed orthonormal).
    """
    K, d = basis.shape
    if d <= 0:
        raise ValueError("basis must have at least one column.")

    def overlap_at(alpha: np.ndarray) -> float:
        norm = float(np.linalg.norm(alpha))
        if norm < 1e-12:
            return 0.0
        a = alpha / norm
        j_vec = basis @ a
        H_eff = _build_h_eff(patterns, pair_list, j_vec)
        ov, _ = _ground_state_overlap(H_eff, c)
        return ov

    if d == 1:
        a_pos = np.array([1.0])
        a_neg = np.array([-1.0])
        ov_pos = overlap_at(a_pos)
        ov_neg = overlap_at(a_neg)
        if ov_neg > ov_pos:
            return basis @ a_neg, ov_neg
        return basis @ a_pos, ov_pos

    rng = np.random.default_rng(seed)

    candidates: list[np.ndarray] = []
    for k in range(d):
        e = np.zeros(d, dtype=np.float64)
        e[k] = 1.0
        candidates.append(e.copy())
        candidates.append(-e)

    # Uniform-J seed: the most natural physical starting point. Project the
    # all-ones vector (length K) onto the basis to get a uniform-AFM seed.
    uni = np.ones(K, dtype=np.float64) / math.sqrt(K)
    proj_uni = basis.T @ uni
    n_uni = float(np.linalg.norm(proj_uni))
    if n_uni > 1e-12:
        candidates.append(proj_uni / n_uni)

    # Project -means onto basis: minimum-<H>_c direction (energy-lowest seed).
    if means is not None:
        proj = basis.T @ (-means)
        n = float(np.linalg.norm(proj))
        if n > 1e-12:
            candidates.append(proj / n)

    for _ in range(n_random):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v) + 1e-30
        candidates.append(v)
    if d == 2:
        thetas = np.linspace(0.0, np.pi, n_angles_per_dim, endpoint=False)
        for th in thetas:
            candidates.append(np.array([np.cos(th), np.sin(th)]))
        candidates.extend(
            [np.array([np.cos(th + np.pi), np.sin(th + np.pi)]) for th in thetas]
        )

    polished: list[tuple[float, np.ndarray]] = []
    try:
        from scipy.optimize import minimize  # type: ignore[import]

        def neg(alpha: np.ndarray) -> float:
            return -overlap_at(alpha)

        scored = sorted(
            ((overlap_at(c0), c0) for c0 in candidates), key=lambda kv: -kv[0]
        )
        n_polish = min(12, len(scored))
        for _, x0 in scored[:n_polish]:
            try:
                res = minimize(
                    neg,
                    x0,
                    method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
                )
                polished.append(
                    (float(-res.fun), np.asarray(res.x, dtype=np.float64))
                )
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        for x0 in candidates:
            polished.append((overlap_at(x0), x0))

    if not polished:
        polished = [(overlap_at(x0), x0) for x0 in candidates]

    best_score, best_alpha = max(polished, key=lambda kv: kv[0])
    norm = float(np.linalg.norm(best_alpha))
    if norm < 1e-12:
        raise RuntimeError("J-direction resolution produced zero-norm vector.")
    j_vec = basis @ (best_alpha / norm)
    return j_vec, best_score


def fit_effective_heisenberg(
    payload: SpinAmplitudePayload,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    nn_normalise: bool = True,
    enforce_positive_nn: bool = True,
    compute_overlap: bool = True,
    null_space_tol: float = 1e-10,
    null_space_seed: int = 0,
) -> EffectiveHeisenbergFit:
    """Covariance-method fit of ``J_{ij}`` to PINN Mott amplitudes.

    Methodology
    -----------
    Step 1: covariance ``Q[a,b] = <c|O_a O_b|c> - <O_a><O_b>`` over pair
    operators ``O_k = S_i.S_j``. ``J`` is in ``null(Q)`` for any parent
    Hamiltonian.

    Step 2: ``null(Q)`` is generally multi-dimensional when the ground
    state lives in a spin sector of dim ``> 1`` (e.g. 4-site singlet
    space is 2D → ``null(Q)`` for NN-only N=4 is also 2D). All null-space
    directions give ``c`` as an eigenstate of the corresponding
    ``H_eff(J)``, but only one direction makes ``c`` the **ground state**
    (lowest eigenvalue).

    Step 3: resolve the null-space ambiguity by maximising the overlap
    ``|<c|psi_0(H_eff(J))>|`` on the unit sphere in ``null(Q)``.
    """
    n_wells = payload.n_wells
    patterns = list(payload.pattern)
    P = len(patterns)
    if P == 0:
        raise ValueError("Empty pattern list; nothing to fit.")
    pattern_index = {sigma: idx for idx, sigma in enumerate(patterns)}

    pair_list = list(pairs) if pairs is not None else _default_pairs(n_wells)
    if not pair_list:
        raise ValueError("No pairs to fit.")
    for (i, j) in pair_list:
        if not (0 <= i < n_wells and 0 <= j < n_wells and i < j):
            raise ValueError(f"Invalid pair (i, j) = ({i}, {j}); require 0 <= i < j < {n_wells}.")

    c = payload.normalised()
    K = len(pair_list)

    Oc = np.zeros((P, K), dtype=np.float64)
    for k, (i, j) in enumerate(pair_list):
        Oc[:, k] = apply_pair_S_dot_S(c, patterns, i, j, pattern_index=pattern_index)

    means = c.dot(Oc)  # <c|O_k|c>
    Q = Oc.T.dot(Oc) - np.outer(means, means)
    Q = 0.5 * (Q + Q.T)

    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)

    largest = float(np.max(np.abs(eigvals))) if eigvals.size else 1.0
    abs_tol = float(null_space_tol) * max(largest, 1.0)
    null_mask = np.abs(eigvals) < abs_tol
    null_count = int(np.sum(null_mask))

    # Search the full ``K``-dim sphere. We pass eigvecs as the orthonormal
    # basis (so ``alpha = e_k`` directly probes Q-eigenvector ``k``) — this
    # gives the optimizer a natural seed grid, but does *not* restrict the
    # search to ``null(Q)``. For exact eigenstates (synthetic), the
    # null-space directions are still in this basis and give overlap = 1
    # with zero residual; for approximate eigenstates (PINN), the optimizer
    # can find better parents that lie outside ``null(Q)``.
    j_vec, overlap = _resolve_J_direction(
        eigvecs,
        patterns,
        pair_list,
        c,
        means=means,
        seed=null_space_seed,
    )

    # Residual variance AT the chosen ``j_vec``: ``Var_c(H_eff) = J^T Q J``.
    # This measures how Heisenberg-like ``c`` actually is at the fitted parent
    # (zero for exact eigenstates, larger for approximate states). The
    # smallest Q eigenvalue is a *lower bound* on what's achievable — but
    # only if the corresponding direction is the physical (ground-state)
    # parent, which is not always the case for symmetric approximate states.
    j_norm_sq = float(j_vec.dot(j_vec))
    if j_norm_sq < 1e-30:
        residual = float(eigvals[0])
    else:
        residual = float(j_vec.dot(Q.dot(j_vec)) / j_norm_sq)
    if largest < 1e-30:
        # Degenerate Q (e.g. K=1 with c an eigenstate of the single operator).
        # All directions give c as eigenstate; relative residual is ill-defined.
        relative = 0.0
    else:
        relative = residual / largest

    nn_pair = (0, 1)
    nn_idx = None
    if nn_pair in pair_list:
        nn_idx = pair_list.index(nn_pair)

    # Sign / normalisation conventions. We only flip the global sign of J if
    # doing so PRESERVES |c⟩ being the ground state of the resulting H_eff.
    # When the optimum is in a hemisphere with J_(0,1) < 0, flipping signs
    # would invert the spectrum and turn |c⟩ into the highest excited state,
    # which we must avoid. In that case we leave the sign alone and warn.
    sign_flipped = False
    if enforce_positive_nn and nn_idx is not None and j_vec[nn_idx] < 0:
        H_check = _build_h_eff(patterns, pair_list, -j_vec)
        ov_flipped, _ = _ground_state_overlap(H_check, c)
        if ov_flipped > 0.999:
            j_vec = -j_vec
            sign_flipped = True
        else:
            LOGGER.info(
                "fit_effective_heisenberg: refused to enforce J_(0,1) > 0 "
                "because flipping signs would lose ground-state property "
                "(would-be overlap %.4f).",
                ov_flipped,
            )

    if nn_normalise and nn_idx is not None:
        nn_value = float(j_vec[nn_idx])
        if abs(nn_value) > 1e-12:
            # Divide by |J_(0,1)| (NOT J_(0,1)): a negative divisor would flip
            # all signs and turn |c⟩ from ground state into the highest excited
            # state. We always preserve the spectrum sign convention chosen by
            # the (sign-aware) null-space optimizer.
            j_vec = j_vec / abs(nn_value)

    j_matrix = np.zeros((n_wells, n_wells), dtype=np.float64)
    for k, (i, j) in enumerate(pair_list):
        j_matrix[i, j] = j_vec[k]
        j_matrix[j, i] = j_vec[k]

    overlap_final = float("nan")
    energy_split = float("nan")
    if compute_overlap:
        H_eff = _build_h_eff(patterns, pair_list, j_vec)
        overlap_final, energy_split = _ground_state_overlap(H_eff, c)
    else:
        overlap_final = float(overlap)

    return EffectiveHeisenbergFit(
        n_wells=n_wells,
        pairs=pair_list,
        j_vector=np.asarray(j_vec, dtype=np.float64),
        j_matrix=j_matrix,
        means=np.asarray(means, dtype=np.float64),
        covariance_eigenvalues=eigvals,
        residual_variance=residual,
        relative_residual=relative,
        overlap_with_ground=overlap_final,
        energy_split=energy_split,
    )


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------


def evaluate_effective_heisenberg(
    result_dir: Path | str,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    spin_sector: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
    nn_normalise: bool = True,
    enforce_positive_nn: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, extract amplitudes and fit ``H_eff = sum J_ij S_i.S_j``."""
    loaded = load_wavefunction_from_dir(result_dir, device=device)
    payload = extract_spin_amplitudes(loaded, spin_sector=spin_sector)
    fit = fit_effective_heisenberg(
        payload,
        pairs=pairs,
        nn_normalise=nn_normalise,
        enforce_positive_nn=enforce_positive_nn,
    )
    out = fit.to_dict()
    out["result_dir"] = str(loaded.result_dir)
    out["n_particles"] = int(loaded.system.n_particles)
    out["spin_sector"] = {"n_up": payload.n_up, "n_down": payload.n_down}
    out["well_centers"] = payload.well_centers.tolist()
    return out


def effective_J_target(
    result_dir: Path | str,
    *,
    pair: Sequence[int],
    target_value: float | None = None,
    mode: str = "value",
    pairs: Sequence[tuple[int, int]] | None = None,
    spin_sector: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
) -> float:
    """Scalar reduction of ``J_{pair}`` for inverse design.

    Modes
    -----
    ``"value"``
        Return the raw fitted ``J_{pair}`` (in nearest-neighbour units when
        ``nn_normalise=True``). Use to *maximise* a chosen long-range
        coupling — gradient ascent.
    ``"neg_squared_error"``
        Return ``-(J_{pair} - target_value)^2``. Use to drive ``J_{pair}``
        *toward* ``target_value`` — gradient ascent (the ``-`` makes it a
        maximisation target).
    ``"squared_error"``
        Return ``(J_{pair} - target_value)^2``; for explicit minimisation.
    """
    if len(pair) != 2:
        raise ValueError(f"`pair` must have length 2, got {pair}.")
    i, j = int(pair[0]), int(pair[1])
    if i == j:
        raise ValueError("pair must have i != j.")
    a, b = (i, j) if i < j else (j, i)

    fit_dict = evaluate_effective_heisenberg(
        result_dir,
        pairs=pairs,
        spin_sector=spin_sector,
        device=device,
    )
    j_matrix = np.asarray(fit_dict["j_matrix"], dtype=np.float64)
    j_value = float(j_matrix[a, b])

    if mode == "value":
        return j_value
    if target_value is None:
        raise ValueError(f"mode='{mode}' requires target_value.")
    err = j_value - float(target_value)
    if mode == "neg_squared_error":
        return -float(err * err)
    if mode == "squared_error":
        return float(err * err)
    raise ValueError(
        f"Unknown mode '{mode}'. Choose 'value', 'neg_squared_error', or 'squared_error'."
    )


__all__ = [
    "EffectiveHeisenbergFit",
    "apply_pair_S_dot_S",
    "build_pair_operator_matrix",
    "effective_J_target",
    "evaluate_effective_heisenberg",
    "evaluate_spin_correlators",
    "fit_effective_heisenberg",
    "spin_correlator_target",
    "spin_pair_correlator",
]
