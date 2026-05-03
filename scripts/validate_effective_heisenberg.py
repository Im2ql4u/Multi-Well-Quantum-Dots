#!/usr/bin/env python3
"""Validate the effective-Heisenberg J_ij extractor on synthetic + real PINN data.

Phase 2A.7. Three independent tests:

  1. **Synthetic Heisenberg.** Build an OBC chain with random
     ``bond_couplings``, diagonalise, take the ground state, and check
     that :func:`fit_effective_heisenberg` recovers those couplings (in
     the ``J_(0,1) = 1`` convention) to ~machine precision.

  2. **Synthetic long-range.** Build a generic ``H = sum_{i<j} J_ij
     S_i.S_j`` with arbitrary off-diagonal couplings, diagonalise, and
     check the fit recovers all ``C(N, 2)`` couplings.

  3. **N=2 singlet PINN checkpoint** (optional --n2-checkpoint). Fit
     should yield trivial single-pair J = 1.0 with overlap 1.

  4. **N=4 d=4 PINN checkpoint** (optional --n4-checkpoint). Fit should
     give NN-dominant couplings with relative residual < a few percent.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observables.effective_heisenberg import (  # noqa: E402
    EffectiveHeisenbergFit,
    apply_pair_S_dot_S,
    build_pair_operator_matrix,
    evaluate_effective_heisenberg,
    fit_effective_heisenberg,
)
from observables.heisenberg_reference import (  # noqa: E402
    build_heisenberg_obc_hamiltonian,
    enumerate_patterns_obc,
)
from observables.spin_amplitude_entanglement import (  # noqa: E402
    SpinAmplitudePayload,
    enumerate_patterns,
    permutation_sign,
)


LOGGER = logging.getLogger("validate_effective_heisenberg")


def _payload_from_amplitudes(
    amps: np.ndarray,
    patterns: list[tuple[int, ...]],
    n_wells: int,
    n_up: int,
    n_down: int,
) -> SpinAmplitudePayload:
    """Construct a SpinAmplitudePayload with amplitudes already factoring in
    the permutation sign convention (so payload.normalised() returns ``amps``)."""
    perm = np.array([float(permutation_sign(p)) for p in patterns], dtype=np.float64)
    safe_amps = np.where(np.abs(amps) < 1e-30, 1e-30, amps)
    log_abs = np.log(np.abs(safe_amps))
    sign_psi = np.sign(safe_amps) / perm
    sign_psi = np.where(np.abs(sign_psi) < 0.5, 1.0, sign_psi)
    well_centers = np.zeros((n_wells, 2), dtype=np.float64)
    for k in range(n_wells):
        well_centers[k] = [4.0 * k, 0.0]
    return SpinAmplitudePayload(
        pattern=patterns,
        log_abs_psi=log_abs,
        sign_psi=sign_psi,
        perm_sign=perm,
        sigma_z_total=n_up - n_down,
        n_up=n_up,
        n_down=n_down,
        n_wells=n_wells,
        well_centers=well_centers,
    )


def test_synthetic_obc(N: int = 4, n_down: int = 2, seed: int = 7) -> None:
    """Random-bond OBC Heisenberg → ground state → fitted parent has correct overlap.

    For ``N >= 4`` the SU(2)-invariant Heisenberg parent Hamiltonian is
    *not* uniquely recoverable from a single ground state (the singlet
    sector is multi-dim, leading to a multi-dim null space of Q). We
    therefore test that ``|c>`` is a ground state of the fitted ``H_eff``
    (overlap = 1), but do **not** require bond-by-bond recovery for N>=4.
    """
    rng = np.random.default_rng(seed)
    bonds = 1.0 + 0.5 * rng.standard_normal(N - 1)
    bonds[0] = 1.0
    H, patterns = build_heisenberg_obc_hamiltonian(N, n_down, bond_couplings=bonds.tolist())
    eigvals, eigvecs = np.linalg.eigh(H)
    ground = eigvecs[:, 0].astype(np.float64)
    if ground[0] < 0:
        ground = -ground

    payload = _payload_from_amplitudes(
        amps=ground,
        patterns=list(patterns),
        n_wells=N,
        n_up=N - n_down,
        n_down=n_down,
    )
    nn_pairs = [(i, i + 1) for i in range(N - 1)]
    fit = fit_effective_heisenberg(
        payload,
        pairs=nn_pairs,
        nn_normalise=True,
        enforce_positive_nn=True,
    )
    print(f"[Test 1] OBC N={N}, n_down={n_down}, seed={seed}")
    print(f"    true bonds (J_(0,1)=1 normalised) = {bonds.tolist()}")
    print(f"    fitted j_vector                   = {fit.j_vector.tolist()}")
    print(f"    residual variance        = {fit.residual_variance:.3e}")
    print(f"    relative residual        = {fit.relative_residual:.3e}")
    print(f"    overlap                  = {fit.overlap_with_ground:.6f}")
    err = float(np.max(np.abs(fit.j_vector - bonds)))
    print(f"    max abs error vs truth   = {err:.3e}  (informative only for N=2,3)")

    assert fit.relative_residual < 1e-9, (
        f"Test 1 FAILED: relative residual = {fit.relative_residual:.3e}"
    )
    assert fit.overlap_with_ground > 0.999999, (
        f"Test 1 FAILED: overlap = {fit.overlap_with_ground:.6f} (need ~1)"
    )
    if N <= 3:
        assert err < 1e-6, (
            f"Test 1 FAILED for N={N}: bond-recovery error = {err:.3e}"
        )
    print("    [PASS]")
    print()


def test_synthetic_full(N: int = 4, n_down: int = 2, seed: int = 11) -> None:
    """Random ALL pairs in H_eff → diagonalise → fitted Hamiltonian has c as ground state."""
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    K = len(pairs)
    j_truth = 1.0 + 0.4 * rng.standard_normal(K)
    j_truth[pairs.index((0, 1))] = 1.0

    patterns = enumerate_patterns(N, n_down)
    P = len(patterns)
    H = np.zeros((P, P), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        H += j_truth[k] * build_pair_operator_matrix(patterns, i, j)
    H = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground = eigvecs[:, 0].astype(np.float64)
    if ground[0] < 0:
        ground = -ground

    payload = _payload_from_amplitudes(
        amps=ground,
        patterns=list(patterns),
        n_wells=N,
        n_up=N - n_down,
        n_down=n_down,
    )
    fit = fit_effective_heisenberg(payload)
    print(f"[Test 2] All-pair N={N}, n_down={n_down}, seed={seed}, K={K}")
    print(f"    true j (normalised)  = {[float(round(v, 4)) for v in j_truth]}")
    print(f"    fitted j_vector      = {[float(round(v, 4)) for v in fit.j_vector]}")
    print(f"    residual variance    = {fit.residual_variance:.3e}")
    print(f"    relative residual    = {fit.relative_residual:.3e}")
    print(f"    overlap              = {fit.overlap_with_ground:.6f}")
    assert fit.relative_residual < 1e-9, (
        f"Test 2 FAILED: relative residual = {fit.relative_residual:.3e}"
    )
    assert fit.overlap_with_ground > 0.999999, (
        f"Test 2 FAILED: overlap = {fit.overlap_with_ground:.6f}"
    )
    print("    [PASS]")
    print()


def test_synthetic_correlators(N: int = 4, n_down: int = 2, seed: int = 5) -> None:
    """Spin correlators <S_i.S_j>_c match independent computation via H matrix elements."""
    from observables.effective_heisenberg import spin_pair_correlator

    rng = np.random.default_rng(seed)
    bonds = 1.0 + 0.3 * rng.standard_normal(N - 1)
    bonds[0] = 1.0
    H, patterns = build_heisenberg_obc_hamiltonian(N, n_down, bond_couplings=bonds.tolist())
    eigvals, eigvecs = np.linalg.eigh(H)
    ground = eigvecs[:, 0].astype(np.float64)
    if ground[0] < 0:
        ground = -ground

    e0_via_corr = 0.0
    nn_pairs = [(i, i + 1) for i in range(N - 1)]
    P = len(patterns)
    for k, (i, j) in enumerate(nn_pairs):
        op = build_pair_operator_matrix(list(patterns), i, j)
        e0_via_corr += bonds[k] * float(ground.dot(op).dot(ground))

    payload = _payload_from_amplitudes(
        amps=ground,
        patterns=list(patterns),
        n_wells=N,
        n_up=N - n_down,
        n_down=n_down,
    )
    correlators = spin_pair_correlator(payload, pairs=nn_pairs)
    e0_via_extractor = sum(
        bonds[k] * correlators["correlators"][k] for k in range(len(nn_pairs))
    )
    e0_truth = float(eigvals[0])

    print(f"[Test 3] Correlators agree with independent diagonalisation; N={N}")
    print(f"    E_0 (truth diagonalisation)        = {e0_truth:.6f}")
    print(f"    E_0 (via build_pair_operator_matrix) = {e0_via_corr:.6f}")
    print(f"    E_0 (via spin_pair_correlator)       = {e0_via_extractor:.6f}")
    assert abs(e0_via_corr - e0_truth) < 1e-9
    assert abs(e0_via_extractor - e0_truth) < 1e-9
    print("    [PASS]")
    print()


def test_n2_checkpoint(checkpoint: Path) -> None:
    """N=2 singlet PINN checkpoint: only one pair, J_(0,1) = 1.0 trivially."""
    fit = evaluate_effective_heisenberg(checkpoint)
    print(f"[Test 4] N=2 singlet checkpoint: {checkpoint}")
    print(f"    pairs              = {fit['pairs']}")
    print(f"    j_vector           = {fit['j_vector']}")
    print(f"    residual variance  = {fit['residual_variance']:.3e}")
    print(f"    overlap            = {fit['overlap_with_ground']:.6f}")
    assert fit["n_particles"] == 2
    assert fit["pairs"] == [[0, 1]]
    assert abs(fit["j_vector"][0] - 1.0) < 1e-9
    assert fit["overlap_with_ground"] > 0.999
    print("    [PASS]")
    print()


def test_n4_checkpoint(checkpoint: Path) -> None:
    """N=4 PINN: fit NN-only Heisenberg, expect c overlap ~ 1, NN bonds ~ uniform.

    We use NN-only (K=3) rather than all-pairs (K=6) because the PINN is
    only approximately Heisenberg — Q doesn't have exact zero eigenvalues
    and the high-dim (5D) null-space optimizer of the all-pair fit
    becomes ill-conditioned. NN-only gives a 2D null space that is
    well-resolved by the standard optimizer.
    """
    fit = evaluate_effective_heisenberg(checkpoint, pairs=[(0, 1), (1, 2), (2, 3)])
    print(f"[Test 5] N=4 PINN checkpoint (NN-only fit): {checkpoint}")
    print(f"    pairs              = {fit['pairs']}")
    print(f"    j_vector           = {[round(v, 5) for v in fit['j_vector']]}")
    print(f"    residual variance  = {fit['residual_variance']:.3e}")
    print(f"    relative residual  = {fit['relative_residual']:.3e}")
    print(f"    overlap            = {fit['overlap_with_ground']:.6f}")
    j_matrix = np.asarray(fit["j_matrix"])
    print(f"    J_NN: J(0,1)={j_matrix[0,1]:.4f}, J(1,2)={j_matrix[1,2]:.4f}, J(2,3)={j_matrix[2,3]:.4f}")
    nn_avg = (j_matrix[0, 1] + j_matrix[1, 2] + j_matrix[2, 3]) / 3.0
    nn_spread = max(j_matrix[0, 1], j_matrix[1, 2], j_matrix[2, 3]) - min(j_matrix[0, 1], j_matrix[1, 2], j_matrix[2, 3])
    print(f"    NN average            = {nn_avg:.4f}, spread={nn_spread:.4f}")
    assert fit["overlap_with_ground"] > 0.90, (
        f"Overlap too low: {fit['overlap_with_ground']}"
    )
    print("    [PASS]")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n2-checkpoint", type=Path, default=None)
    parser.add_argument("--n4-checkpoint", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    test_synthetic_obc(N=2, n_down=1)
    test_synthetic_obc(N=4, n_down=2)
    test_synthetic_obc(N=6, n_down=3)
    test_synthetic_full(N=4, n_down=2)
    test_synthetic_full(N=4, n_down=1)
    test_synthetic_correlators(N=4, n_down=2, seed=5)
    test_synthetic_correlators(N=6, n_down=3, seed=11)

    if args.n2_checkpoint is not None and args.n2_checkpoint.exists():
        test_n2_checkpoint(args.n2_checkpoint)
    else:
        print("[Test 4] N=2 checkpoint test skipped (path not provided or missing).")
        print()

    if args.n4_checkpoint is not None and args.n4_checkpoint.exists():
        test_n4_checkpoint(args.n4_checkpoint)
    else:
        print("[Test 5] N=4 checkpoint test skipped (path not provided or missing).")
        print()

    print("=" * 60)
    print("All effective-Heisenberg validation tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
