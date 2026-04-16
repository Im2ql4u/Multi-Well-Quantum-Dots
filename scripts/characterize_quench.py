#!/usr/bin/env python
"""
Characterize the B-field quench: singlet/triplet transition, spin properties,
partial transpose negativity, and entanglement of pre- and post-quench GS.

Uses the shared-model exact diag (which has proper spin sectors) to analyse
the singlet-triplet crossing under Zeeman field.

Usage:
  .venv/bin/python scripts/characterize_quench.py --sep 4.0 --B-post 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from exact_diag_double_dot import (
    DiagConfig,
    build_2d_dvr,
    build_ci_hamiltonian,
    build_potential_matrix,
    build_slater_basis_sorted,
    infer_box_half_widths,
    precompute_coulomb_kernel,
    single_particle_eigenstates,
)

LOGGER = logging.getLogger("characterize_quench")


def classify_spin_of_eigenstate(
    eigvec: np.ndarray,
    slater_basis: list[tuple[int, int, str, str]],
    n_ci: int,
) -> dict[str, float]:
    """Decompose an eigenstate into spin-sector weights."""
    weights: dict[str, float] = {}
    for idx in range(min(n_ci, len(slater_basis))):
        _, _, _, spin_type = slater_basis[idx]
        weights[spin_type] = weights.get(spin_type, 0.0) + eigvec[idx] ** 2
    return weights


def build_spatial_rdm_shared(
    eigvec: np.ndarray,
    slater_basis: list[tuple[int, int, str, str]],
    single_vecs: np.ndarray,
    n_ci: int,
    n_grid: int,
) -> np.ndarray:
    """Build the spatial 1-particle reduced density matrix from a shared-model CI eigenvector.

    For each Slater determinant (a, b, spin_cfg, spin_type), the spatial part is:
      singlet:  φ_a(1)φ_b(2) + φ_b(1)φ_a(2)  (normalised, symmetric)
      triplet:  φ_a(1)φ_b(2) - φ_b(1)φ_a(2)  (normalised, antisymmetric)
      same-orbital: φ_a(1)φ_a(2)  (trivially symmetric)

    The 1-particle RDM is: ρ(r, r') = ∫ Ψ*(r, r2) Ψ(r', r2) dr2
    In the orbital basis: ρ_{ij} = Σ_k c_k² × (contribution from det k)

    For Schmidt/partial-transpose, we work with the coefficient matrix A[a,b]
    in the orbital basis, mapping the CI ground state onto the spatial part.
    """
    # Build the symmetric coefficient matrix A[a,b] such that
    # Ψ_spatial(r1, r2) = Σ_{a,b} A[a,b] φ_a(r1) φ_b(r2)
    n_orb = single_vecs.shape[1]
    A = np.zeros((n_orb, n_orb), dtype=np.float64)

    for idx in range(min(n_ci, len(slater_basis))):
        a, b, spin_cfg, spin_type = slater_basis[idx]
        c = eigvec[idx]
        if a == b:
            # Same orbital (singlet only): |aa⟩ normalised = φ_a(1)φ_a(2)
            A[a, a] += c
        elif spin_type == "singlet":
            # Singlet: (φ_a φ_b + φ_b φ_a)/√2
            A[a, b] += c / np.sqrt(2.0)
            A[b, a] += c / np.sqrt(2.0)
        elif spin_type.startswith("triplet"):
            # Triplet: (φ_a φ_b - φ_b φ_a)/√2
            A[a, b] += c / np.sqrt(2.0)
            A[b, a] -= c / np.sqrt(2.0)

    return A


def entanglement_from_A(A: np.ndarray) -> dict[str, object]:
    """Compute entanglement measures from coefficient matrix A[a,b].

    A[a,b] is the expansion coefficient in Ψ = Σ_{a,b} A[a,b] |φ_a⟩|φ_b⟩.
    SVD(A) = UΣV† gives the Schmidt decomposition directly.
    """
    _u, sigma, _vh = np.linalg.svd(A, full_matrices=False)
    norm_sq = float(np.sum(sigma**2))
    if abs(norm_sq) < 1e-15:
        return {"entropy": 0.0, "purity": 1.0, "negativity": 0.0, "log_negativity": 0.0}

    probs = sigma**2 / norm_sq
    safe_probs = probs[probs > 1e-15]
    entropy = float(-np.sum(safe_probs * np.log(safe_probs)))
    purity = float(np.sum(probs**2))
    # Negativity for pure state from Schmidt values
    sigma_norm = sigma / np.sqrt(norm_sq)
    sigma_sum = float(np.sum(sigma_norm))
    negativity = (sigma_sum**2 - 1.0) / 2.0
    log_negativity = float(np.log2(max(sigma_sum, 1e-30)))

    return {
        "entropy": entropy,
        "purity": purity,
        "linear_entropy": 1.0 - purity,
        "negativity": negativity,
        "log_negativity": log_negativity,
        "schmidt_values": (sigma_norm[:10]).tolist(),
        "schmidt_probs": (probs[:10] / probs.sum()).tolist(),
        "effective_rank": float(np.exp(entropy)),
    }


def partial_transpose_negativity(A: np.ndarray) -> dict[str, object]:
    """Compute partial transpose negativity from the coefficient matrix.

    For a pure state |Ψ⟩ = Σ A[a,b] |a⟩|b⟩, the density matrix is:
        ρ[a,b,a',b'] = A[a,b] A*[a',b']
    Partial transpose over subsystem B:
        ρ^{T_B}[a,b,a',b'] = ρ[a,b',a',b] = A[a,b'] A*[a',b]
    This is a Hermitian matrix of size (n_orb²) × (n_orb²).
    Neg eigenvalues of ρ^{T_B} witness entanglement (PPT criterion).

    For a pure state, negativity = (Σ σ_k)² - 1) / 2 analytically,
    but we compute directly from ρ^{T_B} to verify and to show the eigenvalues.
    """
    n = A.shape[0]
    # Build ρ^{T_B} as an (n²) × (n²) matrix
    # Index: I = a*n + b,  J = a'*n + b'
    # ρ^{T_B}[I, J] = A[a, b'] * A[a', b]
    # where I -> (a, b), J -> (a', b')
    rho_pt = np.zeros((n * n, n * n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            I = a * n + b
            for ap in range(n):
                for bp in range(n):
                    J = ap * n + bp
                    rho_pt[I, J] = A[a, bp] * A[ap, b]

    # Normalise so Tr(ρ) = 1
    norm = np.trace(rho_pt)
    if abs(norm) < 1e-15:
        return {"neg_eigenvalues": [], "negativity_direct": 0.0, "trace_norm": 0.0}
    rho_pt /= norm

    eigvals = np.linalg.eigvalsh(rho_pt)
    neg_eigs = eigvals[eigvals < -1e-14]
    negativity_direct = float(np.sum(np.abs(neg_eigs)))
    trace_norm = float(np.sum(np.abs(eigvals)))

    return {
        "neg_eigenvalues": sorted(neg_eigs.tolist()),
        "n_negative": len(neg_eigs),
        "negativity_direct": negativity_direct,
        "trace_norm": trace_norm,
        "log_negativity_direct": float(np.log2(max(trace_norm, 1e-30))),
        "min_eigenvalue": float(eigvals.min()),
    }


def run_characterization(
    sep: float,
    B_pre: float,
    B_post: float,
    omega: float = 1.0,
    kappa: float = 0.7,
    epsilon: float = 0.02,
    nx: int = 20,
    ny: int = 20,
    n_sp_states: int = 40,
    n_ci_compute: int = 200,
) -> dict:
    """Run full characterisation of pre- and post-quench ground states."""
    results: dict = {"sep": sep, "B_pre": B_pre, "B_post": B_post, "omega": omega, "kappa": kappa}

    x_half, y_half = infer_box_half_widths(sep, omega, n_wells=2)

    # Build single-particle basis (independent of B)
    x_grid, y_grid, w_x, w_y, t2d = build_2d_dvr(nx=nx, ny=ny, x_half_width=x_half, y_half_width=y_half)
    v2d = build_potential_matrix(x_grid=x_grid, y_grid=y_grid, sep=sep, omega=omega, smooth_t=0.2)
    single_energies, single_vecs = single_particle_eigenstates(t2d=t2d, v2d=v2d, n_sp_states=n_sp_states)
    slater_basis = build_slater_basis_sorted(n_sp_states, single_energies)
    kernel = precompute_coulomb_kernel(
        x_grid=x_grid, y_grid=y_grid, w_x=w_x, w_y=w_y,
        kappa=kappa, epsilon=epsilon, include_quadrature_weights=True,
    )

    for label, B in [("pre", B_pre), ("post", B_post)]:
        LOGGER.info("=" * 60)
        LOGGER.info("  %s-quench: B = %.3f", label.upper(), B)
        LOGGER.info("=" * 60)

        h_ci = build_ci_hamiltonian(
            slater_basis=slater_basis,
            single_energies=single_energies,
            single_vecs=single_vecs,
            kernel=kernel,
            n_ci_compute=n_ci_compute,
            b_field=B,
            g_factor=2.0,
            mu_b=1.0,
        )
        eigvals, eigvecs = eigh(h_ci)
        n_ci = min(n_ci_compute, len(slater_basis))

        # Ground state
        E0 = float(eigvals[0])
        gs_vec = eigvecs[:, 0]
        spin_weights = classify_spin_of_eigenstate(gs_vec, slater_basis, n_ci)

        LOGGER.info("  E0 = %.8f", E0)
        LOGGER.info("  Eigenspectrum (first 6): %s", np.array2string(eigvals[:6], precision=6))
        LOGGER.info("  Gap E1-E0 = %.6f", float(eigvals[1] - eigvals[0]))
        LOGGER.info("  Spin decomposition of GS:")
        for stype, weight in sorted(spin_weights.items(), key=lambda x: -x[1]):
            if weight > 1e-6:
                LOGGER.info("    %-12s: %.6f (%.2f%%)", stype, weight, 100.0 * weight)

        # Identify dominant spin sector
        dominant_spin = max(spin_weights, key=spin_weights.get)
        LOGGER.info("  => GS is predominantly: %s", dominant_spin)

        # Build spatial coefficient matrix A
        A = build_spatial_rdm_shared(gs_vec, slater_basis, single_vecs, n_ci, single_vecs.shape[0])
        LOGGER.info("  A matrix: shape=%s, ‖A‖²=%.6f", A.shape, float(np.sum(A**2)))

        # Entanglement from SVD
        ent = entanglement_from_A(A)
        LOGGER.info("  Von Neumann entropy S = %.6f", ent["entropy"])
        LOGGER.info("  Purity             = %.6f", ent["purity"])
        LOGGER.info("  Negativity (SVD)   = %.6f", ent["negativity"])
        LOGGER.info("  Log-negativity     = %.6f", ent["log_negativity"])
        sv_str = [f"{v:.4f}" for v in ent["schmidt_values"][:6]]
        LOGGER.info("  Schmidt values     = %s", sv_str)

        # Partial transpose negativity (direct eigenvalue computation)
        # Use truncated A for manageable matrix size
        n_orb_trunc = min(10, A.shape[0])  # 10 orbitals → 100×100 ρ^{T_B}
        A_trunc = A[:n_orb_trunc, :n_orb_trunc]
        # Renormalise
        A_norm = float(np.sum(A_trunc**2))
        if A_norm > 1e-15:
            A_trunc = A_trunc / np.sqrt(A_norm)

        pt = partial_transpose_negativity(A_trunc)
        LOGGER.info("  Partial transpose (n_orb=%d):", n_orb_trunc)
        LOGGER.info("    # negative eigenvalues: %d", pt["n_negative"])
        LOGGER.info("    Negativity (direct)   : %.6f", pt["negativity_direct"])
        LOGGER.info("    Min eigenvalue        : %.8f", pt["min_eigenvalue"])
        LOGGER.info("    Log-negativity        : %.6f", pt["log_negativity_direct"])
        if pt["neg_eigenvalues"]:
            neg_str = [f"{v:.6f}" for v in pt["neg_eigenvalues"][:10]]
            LOGGER.info("    Negative eigenvalues  : %s", neg_str)

        # First excited state
        E1 = float(eigvals[1])
        ex1_vec = eigvecs[:, 1]
        ex1_spin = classify_spin_of_eigenstate(ex1_vec, slater_basis, n_ci)
        ex1_dominant = max(ex1_spin, key=ex1_spin.get)
        LOGGER.info("  First excited: E1=%.6f  spin=%s", E1, ex1_dominant)

        results[label] = {
            "B": B,
            "E0": E0,
            "gap": float(eigvals[1] - eigvals[0]),
            "eigenvalues": eigvals[:8].tolist(),
            "spin_weights": spin_weights,
            "dominant_spin": dominant_spin,
            "entanglement": ent,
            "partial_transpose": {
                "n_orb_truncated": n_orb_trunc,
                "n_negative_eigenvalues": pt["n_negative"],
                "negativity": pt["negativity_direct"],
                "min_eigenvalue": pt["min_eigenvalue"],
                "log_negativity": pt["log_negativity_direct"],
                "negative_eigenvalues": pt["neg_eigenvalues"][:20],
            },
            "first_excited_spin": ex1_dominant,
        }

    # Summary comparison
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("  SUMMARY: B=%.2f → B=%.2f QUENCH", B_pre, B_post)
    LOGGER.info("=" * 60)
    pre = results["pre"]
    post = results["post"]
    LOGGER.info("  Energy:    %.6f → %.6f  (ΔE = %.6f)", pre["E0"], post["E0"], post["E0"] - pre["E0"])
    LOGGER.info("  Gap:       %.6f → %.6f", pre["gap"], post["gap"])
    LOGGER.info("  GS spin:   %s → %s", pre["dominant_spin"], post["dominant_spin"])
    LOGGER.info("  1st exc:   %s → %s", pre["first_excited_spin"], post["first_excited_spin"])
    LOGGER.info("  Entropy:   %.6f → %.6f", pre["entanglement"]["entropy"], post["entanglement"]["entropy"])
    LOGGER.info("  Negativity: %.6f → %.6f", pre["partial_transpose"]["negativity"], post["partial_transpose"]["negativity"])
    LOGGER.info("  # neg eigs: %d → %d", pre["partial_transpose"]["n_negative_eigenvalues"], post["partial_transpose"]["n_negative_eigenvalues"])

    spin_flip = pre["dominant_spin"] != post["dominant_spin"]
    if spin_flip:
        LOGGER.info("  *** SPIN FLIP DETECTED: %s → %s ***", pre["dominant_spin"], post["dominant_spin"])
    else:
        LOGGER.info("  No spin flip (GS remains %s)", pre["dominant_spin"])

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Characterize B-field quench: singlet/triplet, entanglement, negativity.")
    parser.add_argument("--sep", type=float, default=4.0, help="Well separation.")
    parser.add_argument("--B-pre", type=float, default=0.0, help="B-field before quench.")
    parser.add_argument("--B-post", type=float, default=0.5, help="B-field after quench.")
    parser.add_argument("--omega", type=float, default=1.0, help="Confinement frequency.")
    parser.add_argument("--kappa", type=float, default=0.7, help="Coulomb strength.")
    parser.add_argument("--nx", type=int, default=20, help="DVR grid points in x.")
    parser.add_argument("--ny", type=int, default=20, help="DVR grid points in y.")
    parser.add_argument("--out-json", type=str, default="", help="Write results to JSON file.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    results = run_characterization(
        sep=args.sep,
        B_pre=args.B_pre,
        B_post=args.B_post,
        omega=args.omega,
        kappa=args.kappa,
        nx=args.nx,
        ny=args.ny,
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        LOGGER.info("Saved to %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
