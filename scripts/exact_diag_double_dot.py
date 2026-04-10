from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

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
) -> np.ndarray:
    x2d, y2d = np.meshgrid(x_grid, y_grid, indexing="ij")
    x_flat, y_flat = x2d.ravel(), y2d.ravel()
    wx2d, wy2d = np.meshgrid(w_x, w_y, indexing="ij")
    weight_flat = (wx2d * wy2d).ravel()

    dx = x_flat[:, None] - x_flat[None, :]
    dy = y_flat[:, None] - y_flat[None, :]
    r12 = np.sqrt(dx**2 + dy**2 + epsilon**2)
    return (kappa / r12) * weight_flat[:, None] * weight_flat[None, :]


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


def infer_box_half_widths(sep: float, omega: float) -> tuple[float, float]:
    ho_len = 1.0 / np.sqrt(max(omega, 1e-8))
    x_half = max(0.5 * sep + 4.0 * ho_len, 6.0 * ho_len)
    y_half = max(4.0 * ho_len, 3.0)
    return x_half, y_half


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


def run_validation(base_args: argparse.Namespace) -> int:
    checks = [
        {"name": "sep0_no_coulomb", "sep": 0.0, "kappa": 0.0, "target": 2.0, "tol": 0.25},
        {"name": "sep20_no_coulomb", "sep": 20.0, "kappa": 0.0, "target": 2.0, "tol": 0.25},
        {"name": "sep4_with_coulomb", "sep": 4.0, "kappa": base_args.kappa, "target": 2.17, "tol": 0.8},
    ]
    failures = 0
    for check in checks:
        x_half, y_half = infer_box_half_widths(check["sep"], base_args.omega)
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
        )
        evals, _, _ = run_exact_diagonalization(cfg)
        e0 = float(evals[0])
        err = abs(e0 - check["target"])
        ok = err <= check["tol"]
        status = "PASS" if ok else "FAIL"
        LOGGER.info(
            "validate %-18s e0=% .6f target=% .6f |err|=% .6f tol=% .6f => %s",
            check["name"],
            e0,
            check["target"],
            err,
            check["tol"],
            status,
        )
        if not ok:
            failures += 1
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact diagonalization for 2-electron double dot (DVR+CI).")
    parser.add_argument("--nx", type=int, default=20, help="DVR grid points in x.")
    parser.add_argument("--ny", type=int, default=20, help="DVR grid points in y.")
    parser.add_argument("--n-max", type=int, default=6, help="Principal-like cutoff proxy used to set retained SP states.")
    parser.add_argument("--n-sp-states", type=int, default=40, help="Single-particle eigenstates retained for CI.")
    parser.add_argument("--n-ci-compute", type=int, default=200, help="Number of determinants retained in CI matrix.")
    parser.add_argument("--sep", type=float, default=4.0, help="Well separation along x.")
    parser.add_argument("--omega", type=float, default=1.0, help="Confinement frequency.")
    parser.add_argument("--smooth-t", type=float, default=0.2, help="Soft-min temperature for double-well potential.")
    parser.add_argument("--kappa", type=float, default=1.0, help="Coulomb strength.")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Coulomb softening.")
    parser.add_argument("--B", type=float, default=0.0, help="Magnetic field entering Zeeman shift.")
    parser.add_argument("--g-factor", type=float, default=2.0, help="g-factor in Zeeman term.")
    parser.add_argument("--mu-b", type=float, default=1.0, help="Bohr magneton scaling in Zeeman term.")
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

    x_half, y_half = infer_box_half_widths(args.sep, args.omega)
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
    )

    LOGGER.info(
        "Running exact diag: sep=%.3f omega=%.3f B=%.3f nx=%d ny=%d sp=%d ci=%d",
        cfg.sep,
        cfg.omega,
        cfg.b_field,
        cfg.nx,
        cfg.ny,
        cfg.n_sp_states,
        cfg.n_ci_compute,
    )

    eigvals, _, single_energies = run_exact_diagonalization(cfg)
    LOGGER.info("Ground-state energy E0 = %.8f", float(eigvals[0]))
    LOGGER.info("Lowest single-particle energies: %s", np.array2string(single_energies[:6], precision=6))
    LOGGER.info("Lowest many-body eigenvalues: %s", np.array2string(eigvals[:5], precision=8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())