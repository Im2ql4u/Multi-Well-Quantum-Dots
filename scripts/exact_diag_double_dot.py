from __future__ import annotations

import argparse
import heapq
import itertools
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import eigh


LOGGER = logging.getLogger("exact_diag_double_dot")


@dataclass(frozen=True)
class DiagConfig:
    nx: int
    ny: int
    sep: float
    omega: float
    smooth_t: float
    kappa: float
    epsilon: float
    n_sp_states: int
    n_ci_compute: int
    b_field: float
    g_factor: float
    mu_b: float
    x_half_width: float
    y_half_width: float
    n_wells: int = 2
    model_mode: Literal["shared", "one_per_well"] = "one_per_well"
    confinement_mode: Literal["localized", "softmin"] = "localized"
    kinetic_prefactor: float = 0.5


def sine_dvr_1d(x0: float, x1: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return grid, quadrature weights, and kinetic matrix for 1D sine-DVR."""
    length = x1 - x0
    grid = x0 + np.arange(1, n + 1) * length / (n + 1)
    weights = np.full(n, length / (n + 1), dtype=np.float64)
    j = np.arange(1, n + 1, dtype=np.float64)
    u = np.sqrt(2.0 / (n + 1)) * np.sin(np.outer(j, j * np.pi / (n + 1)))
    kinetic = (u.T * (j * np.pi / length) ** 2) @ u
    return grid, weights, kinetic


def build_2d_dvr(
    nx: int,
    ny: int,
    x_half_width: float,
    y_half_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_grid, w_x, t_x = sine_dvr_1d(-x_half_width, x_half_width, nx)
    y_grid, w_y, t_y = sine_dvr_1d(-y_half_width, y_half_width, ny)
    t2d = np.kron(t_x, np.eye(ny)) + np.kron(np.eye(nx), t_y)
    return x_grid, y_grid, w_x, w_y, t2d


def softmin_double_well_potential(
    x: np.ndarray,
    y: np.ndarray,
    sep: float,
    omega: float,
    smooth_t: float,
) -> np.ndarray:
    """Soft-min of two harmonic wells centered at +-sep/2 along x."""
    x_l = -0.5 * sep
    x_r = 0.5 * sep
    v_l = 0.5 * omega**2 * ((x - x_l) ** 2 + y**2)
    v_r = 0.5 * omega**2 * ((x - x_r) ** 2 + y**2)
    return -smooth_t * np.logaddexp(-v_l / smooth_t, -v_r / smooth_t)


def softmin_multiwell_potential(
    x: np.ndarray,
    y: np.ndarray,
    centers: list[float],
    omega: float,
    smooth_t: float,
) -> np.ndarray:
    """Soft-min harmonic confinement over arbitrary x-axis well centers."""
    v_terms = []
    for cx in centers:
        v_terms.append(0.5 * omega**2 * ((x - cx) ** 2 + y**2))
    v_stack = np.stack(v_terms, axis=0)
    return -smooth_t * np.log(np.sum(np.exp(-v_stack / smooth_t), axis=0))


def build_potential_matrix(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sep: float,
    omega: float,
    smooth_t: float,
) -> np.ndarray:
    x2d, y2d = np.meshgrid(x_grid, y_grid, indexing="ij")
    potential = softmin_double_well_potential(x2d, y2d, sep=sep, omega=omega, smooth_t=smooth_t)
    return np.diag(potential.ravel())


def build_centered_harmonic_potential_matrix(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    omega: float,
    center_x: float,
) -> np.ndarray:
    """2D harmonic potential centered at `center_x` on the x-axis."""
    x2d, y2d = np.meshgrid(x_grid, y_grid, indexing="ij")
    potential = 0.5 * omega**2 * ((x2d - center_x) ** 2 + y2d**2)
    return np.diag(potential.ravel())


def build_softmin_multiwell_potential_matrix(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centers: list[float],
    omega: float,
    smooth_t: float,
) -> np.ndarray:
    x2d, y2d = np.meshgrid(x_grid, y_grid, indexing="ij")
    potential = softmin_multiwell_potential(
        x2d,
        y2d,
        centers=centers,
        omega=omega,
        smooth_t=smooth_t,
    )
    return np.diag(potential.ravel())


def single_particle_eigenstates(
    t2d: np.ndarray,
    v2d: np.ndarray,
    n_sp_states: int,
) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = eigh(t2d + v2d)
    return eigvals[:n_sp_states], eigvecs[:, :n_sp_states]


def build_slater_basis_sorted(
    n_basis_orbitals: int,
    single_energies: np.ndarray,
) -> list[tuple[int, int, str, str]]:
    """Energy-sorted Slater basis for singlet/triplet sectors."""
    raw: list[tuple[float, int, int, str, str]] = []
    for a in range(n_basis_orbitals):
        for b in range(a, n_basis_orbitals):
            e_ab = float(single_energies[a] + single_energies[b])
            if a == b:
                raw.append((e_ab, a, b, "ud", "singlet"))
            else:
                raw.append((e_ab, a, b, "singlet", "singlet"))
                raw.append((e_ab, a, b, "uu", "triplet_p"))
                raw.append((e_ab, a, b, "triplet_0", "triplet_0"))
                raw.append((e_ab, a, b, "dd", "triplet_m"))
    raw.sort(key=lambda item: (item[0], item[1], item[2], item[4]))
    return [(a, b, spin_cfg, spin_type) for _, a, b, spin_cfg, spin_type in raw]


def precompute_coulomb_kernel(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    w_x: np.ndarray,
    w_y: np.ndarray,
    kappa: float,
    epsilon: float,
    include_quadrature_weights: bool = True,
) -> np.ndarray:
    x2d, y2d = np.meshgrid(x_grid, y_grid, indexing="ij")
    x_flat, y_flat = x2d.ravel(), y2d.ravel()
    wx2d, wy2d = np.meshgrid(w_x, w_y, indexing="ij")
    weight_flat = (wx2d * wy2d).ravel()

    dx = x_flat[:, None] - x_flat[None, :]
    dy = y_flat[:, None] - y_flat[None, :]
    r12 = np.sqrt(dx**2 + dy**2 + epsilon**2)
    kernel = kappa / r12
    if include_quadrature_weights:
        kernel = kernel * weight_flat[:, None] * weight_flat[None, :]
    return kernel


def zeeman_shift(spin_type: str, b_field: float, g_factor: float, mu_b: float) -> float:
    if spin_type == "triplet_p":
        s_z = 1.0
    elif spin_type == "triplet_m":
        s_z = -1.0
    else:
        s_z = 0.0
    return 0.5 * g_factor * mu_b * b_field * s_z


def build_ci_hamiltonian(
    slater_basis: list[tuple[int, int, str, str]],
    single_energies: np.ndarray,
    single_vecs: np.ndarray,
    kernel: np.ndarray,
    n_ci_compute: int,
    b_field: float,
    g_factor: float,
    mu_b: float,
) -> np.ndarray:
    n_cfg = min(n_ci_compute, len(slater_basis))
    h_ci = np.zeros((n_cfg, n_cfg), dtype=np.float64)
    two_e_cache: dict[tuple[int, int, int, int], float] = {}

    def two_e(a: int, b: int, c: int, d: int) -> float:
        key = (a, b, c, d)
        if key not in two_e_cache:
            two_e_cache[key] = float((single_vecs[:, a] * single_vecs[:, c]) @ kernel @ (single_vecs[:, b] * single_vecs[:, d]))
        return two_e_cache[key]

    for idx_i in range(n_cfg):
        a_i, b_i, _, spin_i = slater_basis[idx_i]
        same_i = a_i == b_i
        for idx_j in range(idx_i, n_cfg):
            a_j, b_j, _, spin_j = slater_basis[idx_j]
            if spin_i != spin_j:
                continue

            same_j = a_j == b_j
            val = 0.0

            if idx_i == idx_j:
                val += float(single_energies[a_i] + single_energies[b_i])
                val += zeeman_shift(spin_i, b_field=b_field, g_factor=g_factor, mu_b=mu_b)

            if spin_i == "singlet":
                if same_i and same_j:
                    if a_i == a_j:
                        val += two_e(a_i, a_i, a_i, a_i)
                    else:
                        val += 2.0 * two_e(a_i, a_j, a_i, a_j)
                elif same_i:
                    val += np.sqrt(2.0) * two_e(a_i, a_j, a_i, b_j)
                elif same_j:
                    val += np.sqrt(2.0) * two_e(a_i, a_j, b_i, a_j)
                else:
                    val += two_e(a_i, a_j, b_i, b_j) + two_e(a_i, b_j, b_i, a_j)
            else:
                if (not same_i) and (not same_j):
                    val += two_e(a_i, a_j, b_i, b_j) - two_e(a_i, b_j, b_i, a_j)

            h_ci[idx_i, idx_j] += val
            if idx_i != idx_j:
                h_ci[idx_j, idx_i] = h_ci[idx_i, idx_j]

    return h_ci


def infer_box_half_widths(sep: float, omega: float, n_wells: int = 2) -> tuple[float, float]:
    ho_len = 1.0 / np.sqrt(max(omega, 1e-8))
    edge_center = 0.5 * max(n_wells - 1, 1) * sep
    x_half = max(edge_center + 4.0 * ho_len, 6.0 * ho_len)
    y_half = max(4.0 * ho_len, 3.0)
    return x_half, y_half


def well_centers_linear(n_wells: int, sep: float) -> list[float]:
    if n_wells < 2:
        raise ValueError(f"n_wells must be >= 2 for one_per_well mode, got {n_wells}.")
    origin = 0.5 * (n_wells - 1)
    return [(idx - origin) * float(sep) for idx in range(n_wells)]


def select_low_energy_product_basis(
    per_well_energies: list[np.ndarray],
    n_ci_compute: int,
) -> list[tuple[float, tuple[int, ...]]]:
    n_wells = len(per_well_energies)
    n_states = len(per_well_energies[0])
    if n_ci_compute <= 0:
        raise ValueError("n_ci_compute must be positive.")

    # Keep only n_ci_compute smallest product energies via bounded max-heap.
    heap: list[tuple[float, tuple[int, ...]]] = []
    for orb_tuple in itertools.product(range(n_states), repeat=n_wells):
        e_total = float(sum(per_well_energies[w][orb_tuple[w]] for w in range(n_wells)))
        entry = (-e_total, orb_tuple)
        if len(heap) < n_ci_compute:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    selected = [(-neg_e, orb_tuple) for neg_e, orb_tuple in heap]
    selected.sort(key=lambda item: (item[0], item[1]))
    return selected


def run_exact_diagonalization_one_per_well_multi(cfg: DiagConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-electron-per-well CI for linear multiwell systems (distinguishable well channels)."""
    x_grid, y_grid, w_x, w_y, t2d = build_2d_dvr(
        nx=cfg.nx,
        ny=cfg.ny,
        x_half_width=cfg.x_half_width,
        y_half_width=cfg.y_half_width,
    )
    t2d = cfg.kinetic_prefactor * t2d

    centers = well_centers_linear(cfg.n_wells, cfg.sep)
    per_well_energies: list[np.ndarray] = []
    per_well_vecs: list[np.ndarray] = []
    if cfg.confinement_mode == "localized":
        for center_x in centers:
            v_well = build_centered_harmonic_potential_matrix(
                x_grid=x_grid,
                y_grid=y_grid,
                omega=cfg.omega,
                center_x=center_x,
            )
            energies_w, vecs_w = single_particle_eigenstates(
                t2d=t2d,
                v2d=v_well,
                n_sp_states=cfg.n_sp_states,
            )
            per_well_energies.append(energies_w)
            per_well_vecs.append(vecs_w)
    elif cfg.confinement_mode == "softmin":
        # Parity ablation: use one shared soft-min one-body Hamiltonian for all channels.
        v_shared = build_softmin_multiwell_potential_matrix(
            x_grid=x_grid,
            y_grid=y_grid,
            centers=centers,
            omega=cfg.omega,
            smooth_t=cfg.smooth_t,
        )
        energies_w, vecs_w = single_particle_eigenstates(
            t2d=t2d,
            v2d=v_shared,
            n_sp_states=cfg.n_sp_states,
        )
        for _ in range(cfg.n_wells):
            per_well_energies.append(energies_w)
            per_well_vecs.append(vecs_w)
    else:
        raise ValueError(f"Unknown confinement_mode={cfg.confinement_mode}")

    kernel = precompute_coulomb_kernel(
        x_grid=x_grid,
        y_grid=y_grid,
        w_x=w_x,
        w_y=w_y,
        kappa=cfg.kappa,
        epsilon=cfg.epsilon,
        include_quadrature_weights=False,
    )

    basis = select_low_energy_product_basis(per_well_energies, cfg.n_ci_compute)
    n_cfg = len(basis)
    n_wells = cfg.n_wells
    h_ci = np.zeros((n_cfg, n_cfg), dtype=np.float64)
    two_e_cache: dict[tuple[int, int, int, int, int, int], float] = {}

    def pair_integral(w_l: int, w_r: int, a_i: int, a_j: int, b_i: int, b_j: int) -> float:
        key = (w_l, w_r, a_i, a_j, b_i, b_j)
        if key not in two_e_cache:
            left = per_well_vecs[w_l]
            right = per_well_vecs[w_r]
            two_e_cache[key] = float((left[:, a_i] * left[:, a_j]) @ kernel @ (right[:, b_i] * right[:, b_j]))
        return two_e_cache[key]

    zeeman_single = 0.5 * cfg.g_factor * cfg.mu_b * cfg.b_field
    for idx_i, (e_i, orb_i) in enumerate(basis):
        # one-body part is diagonal in this product basis
        h_ci[idx_i, idx_i] = e_i + zeeman_single
        for idx_j in range(idx_i, n_cfg):
            _, orb_j = basis[idx_j]
            val = 0.0
            for p in range(n_wells):
                for q in range(p + 1, n_wells):
                    # Spectator overlap: for the two-body operator V_{pq},
                    # all wells k not in {p, q} must have matching orbitals
                    # between bra and ket (orthonormality → δ_{orb_i[k], orb_j[k]}).
                    spectator_ok = all(
                        orb_i[k] == orb_j[k] for k in range(n_wells) if k != p and k != q
                    )
                    if not spectator_ok:
                        continue
                    val += pair_integral(
                        p,
                        q,
                        orb_i[p],
                        orb_j[p],
                        orb_i[q],
                        orb_j[q],
                    )
            h_ci[idx_i, idx_j] += val
            if idx_i != idx_j:
                h_ci[idx_j, idx_i] = h_ci[idx_i, idx_j]

    eigvals, eigvecs = eigh(h_ci)
    merged_sp = np.sort(np.concatenate(per_well_energies))
    return eigvals, eigvecs, merged_sp


def run_exact_diagonalization(cfg: DiagConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid, y_grid, w_x, w_y, t2d = build_2d_dvr(
        nx=cfg.nx,
        ny=cfg.ny,
        x_half_width=cfg.x_half_width,
        y_half_width=cfg.y_half_width,
    )
    v2d = build_potential_matrix(
        x_grid=x_grid,
        y_grid=y_grid,
        sep=cfg.sep,
        omega=cfg.omega,
        smooth_t=cfg.smooth_t,
    )
    single_energies, single_vecs = single_particle_eigenstates(t2d=t2d, v2d=v2d, n_sp_states=cfg.n_sp_states)
    slater_basis = build_slater_basis_sorted(cfg.n_sp_states, single_energies)
    kernel = precompute_coulomb_kernel(
        x_grid=x_grid,
        y_grid=y_grid,
        w_x=w_x,
        w_y=w_y,
        kappa=cfg.kappa,
        epsilon=cfg.epsilon,
        include_quadrature_weights=True,
    )
    h_ci = build_ci_hamiltonian(
        slater_basis=slater_basis,
        single_energies=single_energies,
        single_vecs=single_vecs,
        kernel=kernel,
        n_ci_compute=cfg.n_ci_compute,
        b_field=cfg.b_field,
        g_factor=cfg.g_factor,
        mu_b=cfg.mu_b,
    )
    eigvals, eigvecs = eigh(h_ci)
    return eigvals, eigvecs, single_energies


def run_exact_diagonalization_one_per_well(cfg: DiagConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-electron one-per-well reference with left/right localized one-body bases."""
    x_grid, y_grid, w_x, w_y, t2d = build_2d_dvr(
        nx=cfg.nx,
        ny=cfg.ny,
        x_half_width=cfg.x_half_width,
        y_half_width=cfg.y_half_width,
    )
    t2d = cfg.kinetic_prefactor * t2d
    v_left = build_centered_harmonic_potential_matrix(
        x_grid=x_grid,
        y_grid=y_grid,
        omega=cfg.omega,
        center_x=-0.5 * cfg.sep,
    )
    v_right = build_centered_harmonic_potential_matrix(
        x_grid=x_grid,
        y_grid=y_grid,
        omega=cfg.omega,
        center_x=0.5 * cfg.sep,
    )
    left_energies, left_vecs = single_particle_eigenstates(t2d=t2d, v2d=v_left, n_sp_states=cfg.n_sp_states)
    right_energies, right_vecs = single_particle_eigenstates(t2d=t2d, v2d=v_right, n_sp_states=cfg.n_sp_states)
    kernel = precompute_coulomb_kernel(
        x_grid=x_grid,
        y_grid=y_grid,
        w_x=w_x,
        w_y=w_y,
        kappa=cfg.kappa,
        epsilon=cfg.epsilon,
        include_quadrature_weights=False,
    )

    product_basis: list[tuple[float, int, int]] = []
    for a in range(cfg.n_sp_states):
        for b in range(cfg.n_sp_states):
            product_basis.append((float(left_energies[a] + right_energies[b]), a, b))
    product_basis.sort(key=lambda item: (item[0], item[1], item[2]))
    product_basis = product_basis[: cfg.n_ci_compute]

    n_cfg = len(product_basis)
    h_ci = np.zeros((n_cfg, n_cfg), dtype=np.float64)
    two_e_cache: dict[tuple[int, int, int, int], float] = {}

    def two_e(a: int, b: int, c: int, d: int) -> float:
        key = (a, b, c, d)
        if key not in two_e_cache:
            two_e_cache[key] = float((left_vecs[:, a] * left_vecs[:, c]) @ kernel @ (right_vecs[:, b] * right_vecs[:, d]))
        return two_e_cache[key]

    # One-per-well reference uses one electron in each localized well basis.
    # Zeeman shift is applied on electron-1 channel to match quench setup.
    zeeman_single = 0.5 * cfg.g_factor * cfg.mu_b * cfg.b_field

    for idx_i, (_, a_i, b_i) in enumerate(product_basis):
        h_ci[idx_i, idx_i] = float(left_energies[a_i] + right_energies[b_i] + zeeman_single)
        for idx_j in range(idx_i, n_cfg):
            _, a_j, b_j = product_basis[idx_j]
            val = two_e(a_i, b_i, a_j, b_j)
            h_ci[idx_i, idx_j] += val
            if idx_i != idx_j:
                h_ci[idx_j, idx_i] = h_ci[idx_i, idx_j]

    eigvals, eigvecs = eigh(h_ci)
    merged_sp = np.sort(np.concatenate([left_energies, right_energies]))
    return eigvals, eigvecs, merged_sp


def solve_exact_diagonalization(cfg: DiagConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg.model_mode == "shared":
        if cfg.n_wells != 2:
            raise ValueError("shared mode currently supports n_wells=2 only.")
        return run_exact_diagonalization(cfg)
    if cfg.model_mode == "one_per_well":
        if cfg.n_wells == 2:
            return run_exact_diagonalization_one_per_well(cfg)
        return run_exact_diagonalization_one_per_well_multi(cfg)
    raise ValueError(f"Unknown model_mode={cfg.model_mode}")


def run_validation(base_args: argparse.Namespace) -> int:
    checks = [
        {"name": "sep0_no_coulomb", "sep": 0.0, "kappa": 0.0, "target": 2.0},
        {"name": "sep0_with_coulomb", "sep": 0.0, "kappa": base_args.kappa, "target": 3.0},
        {"name": "sep20_with_coulomb", "sep": 20.0, "kappa": base_args.kappa, "target": 2.0},
        {"name": "sep4_with_coulomb", "sep": 4.0, "kappa": base_args.kappa, "target": 2.17},
    ]
    rel_tol = 0.05
    failures = 0
    for check in checks:
        x_half, y_half = infer_box_half_widths(check["sep"], base_args.omega, n_wells=2)
        cfg = DiagConfig(
            nx=base_args.nx,
            ny=base_args.ny,
            sep=check["sep"],
            omega=base_args.omega,
            smooth_t=base_args.smooth_t,
            kappa=check["kappa"],
            epsilon=base_args.epsilon,
            n_sp_states=base_args.n_sp_states,
            n_ci_compute=base_args.n_ci_compute,
            b_field=0.0,
            g_factor=base_args.g_factor,
            mu_b=base_args.mu_b,
            x_half_width=x_half,
            y_half_width=y_half,
            model_mode=base_args.model,
            kinetic_prefactor=base_args.kinetic_prefactor,
        )
        evals, _, _ = solve_exact_diagonalization(cfg)
        e0 = float(evals[0])
        err = abs(e0 - check["target"])
        rel_err = err / max(abs(check["target"]), 1e-12)
        ok = rel_err <= rel_tol
        status = "PASS" if ok else "FAIL"
        LOGGER.info(
            "validate %-20s e0=% .6f target=% .6f |err|=% .6f rel_err=%.2f%% tol=%.2f%% => %s",
            check["name"],
            e0,
            check["target"],
            err,
            100.0 * rel_err,
            100.0 * rel_tol,
            status,
        )
        if not ok:
            failures += 1
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact diagonalization for one-per-well DVR+CI references.")
    parser.add_argument("--nx", type=int, default=20, help="DVR grid points in x.")
    parser.add_argument("--ny", type=int, default=20, help="DVR grid points in y.")
    parser.add_argument("--n-max", type=int, default=6, help="Principal-like cutoff proxy used to set retained SP states.")
    parser.add_argument("--n-sp-states", type=int, default=40, help="Single-particle eigenstates retained for CI.")
    parser.add_argument("--n-ci-compute", type=int, default=200, help="Number of determinants retained in CI matrix.")
    parser.add_argument("--sep", type=float, default=4.0, help="Well separation along x.")
    parser.add_argument("--n-wells", type=int, default=2, help="Number of wells/electrons for one_per_well mode.")
    parser.add_argument("--omega", type=float, default=1.0, help="Confinement frequency.")
    parser.add_argument("--smooth-t", type=float, default=0.2, help="Soft-min temperature for double-well potential.")
    parser.add_argument("--kappa", type=float, default=0.7, help="Coulomb strength.")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Coulomb softening.")
    parser.add_argument("--B", type=float, default=0.0, help="Magnetic field entering Zeeman shift.")
    parser.add_argument("--g-factor", type=float, default=2.0, help="g-factor in Zeeman term.")
    parser.add_argument("--mu-b", type=float, default=1.0, help="Bohr magneton scaling in Zeeman term.")
    parser.add_argument(
        "--model",
        type=str,
        default="one_per_well",
        choices=["shared", "one_per_well"],
        help="Reference Hamiltonian mode: shared soft-min or one-per-well localized basis.",
    )
    parser.add_argument(
        "--confinement-mode",
        type=str,
        default="localized",
        choices=["localized", "softmin"],
        help="One-per-well one-body confinement used for multiwell parity checks.",
    )
    parser.add_argument(
        "--kinetic-prefactor",
        type=float,
        default=0.5,
        help="Prefactor multiplying the DVR kinetic operator.",
    )
    parser.add_argument("--validate", action="store_true", help="Run built-in limit checks and exit status by pass/fail.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if args.n_sp_states < args.n_max:
        args.n_sp_states = max(args.n_sp_states, args.n_max)
    if args.n_ci_compute < args.n_sp_states:
        args.n_ci_compute = args.n_sp_states

    if args.validate:
        return run_validation(args)

    x_half, y_half = infer_box_half_widths(args.sep, args.omega, n_wells=args.n_wells)
    cfg = DiagConfig(
        nx=args.nx,
        ny=args.ny,
        sep=args.sep,
        omega=args.omega,
        smooth_t=args.smooth_t,
        kappa=args.kappa,
        epsilon=args.epsilon,
        n_sp_states=args.n_sp_states,
        n_ci_compute=args.n_ci_compute,
        b_field=args.B,
        g_factor=args.g_factor,
        mu_b=args.mu_b,
        x_half_width=x_half,
        y_half_width=y_half,
        n_wells=args.n_wells,
        model_mode=args.model,
        confinement_mode=args.confinement_mode,
        kinetic_prefactor=args.kinetic_prefactor,
    )

    LOGGER.info(
        "Running exact diag: mode=%s confinement=%s n_wells=%d sep=%.3f omega=%.3f B=%.3f nx=%d ny=%d sp=%d ci=%d",
        cfg.model_mode,
        cfg.confinement_mode,
        cfg.n_wells,
        cfg.sep,
        cfg.omega,
        cfg.b_field,
        cfg.nx,
        cfg.ny,
        cfg.n_sp_states,
        cfg.n_ci_compute,
    )

    eigvals, _, single_energies = solve_exact_diagonalization(cfg)
    LOGGER.info("Ground-state energy E0 = %.8f", float(eigvals[0]))
    LOGGER.info("Lowest single-particle energies: %s", np.array2string(single_energies[:6], precision=6))
    LOGGER.info("Lowest many-body eigenvalues: %s", np.array2string(eigvals[:5], precision=8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())