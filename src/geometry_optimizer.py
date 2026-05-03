"""Inverse design: optimise quantum-dot geometry to hit a target property.

Bilevel optimisation
--------------------
We split the problem into

  * **Inner loop** — train a variational wavefunction Ψ(θ) to ground-state
    accuracy for the *fixed* geometry parametrised by θ. We re-use the
    existing two-stage non-MCMC trainer (`scripts/run_two_stage_ground_state`)
    and trust it to converge.
  * **Outer loop** — update θ to optimise some scalar target ``T[Ψ(θ)]``
    (energy, entanglement, exchange gap, pair correlation, …).

For the **energy target** the Hellmann-Feynman theorem gives the gradient
exactly, ``∂E/∂R_k = ⟨∂V/∂R_k⟩_Ψ``, so we sample the trained Ψ and average
the harmonic force on each well centre. This is one extra evaluation per
outer step.

For **non-energy targets** we use **central finite differences on the
parametrised target**:

  ∂T/∂θ_i ≈ ( T(θ + ε_i) − T(θ − ε_i) ) / ( 2 ε_i )

This costs ``2 · dim(θ)`` extra inner-loop trainings per outer step, but
each warm-starts from the previous outer-step checkpoint so it is *much*
cheaper than training from scratch.

Generic interface
-----------------
* ``param_to_wells(theta)``  — user-supplied callable mapping a parameter
  vector ``θ ∈ R^k`` to a concrete ``[{"center": (...), "omega": ..., ...}]``
  list. For an N=2 chain the natural choice is ``theta = [d]`` with
  ``wells = [(-d/2, 0), (+d/2, 0)]``.
* ``target_fn(result_dir, wells, system)`` — user-supplied callable
  returning a scalar ``float``. Default options for ``target`` keyword:
  ``"energy"`` (read from result.json) or ``"entanglement_n2"`` (uses
  :func:`observables.checkpoint_entanglement.entanglement_target_n2`).

Usage
-----
    from geometry_optimizer import GeometryOptimizer

    opt = GeometryOptimizer(
        base_config_path="configs/one_per_well/n2_singlet_d2_s42.yaml",
        target="entanglement_n2",
        param_to_wells=lambda t: [
            {"center": [-float(t[0])/2, 0.0], "omega": 1.0, "n_particles": 1},
            {"center": [+float(t[0])/2, 0.0], "omega": 1.0, "n_particles": 1},
        ],
        param_init=[2.0],
        param_step=[0.4],
        n_outer_steps=10,
        lr_param=0.5,
        device="cuda:0",
    )
    optimal_theta, history = opt.run()
"""
from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts" / "run_two_stage_ground_state.py"


LOGGER = logging.getLogger("geometry_optimizer")


WellsLike = list[dict[str, Any]]


@dataclass
class StepRecord:
    """One outer-loop step record persisted to history.json."""

    step: int
    theta: list[float]
    target: float
    energy: float
    grad_theta: list[float]
    wells: list[list[float]]
    centre_result_dir: str | None
    perturbed_targets: list[dict[str, Any]]
    dt_sec: float
    # Multi-sector targets (e.g. exchange_gap) populate sector-resolved
    # auxiliaries here; single-sector targets leave these empty.
    sector_energies: dict[str, float] | None = None
    sector_result_dirs: dict[str, str] | None = None


@dataclass
class GeomEvalContext:
    """Result of evaluating one geometry at a single outer step.

    Captures one or more inner-loop trainings (one per spin sector) plus a
    bookkeeping primary so that single-sector targets (energy, entanglement)
    keep working unchanged. Multi-sector targets (exchange_gap, J_eff) read
    the auxiliary sectors directly from ``sector_*``.
    """

    wells: WellsLike
    primary_result_dir: Path | None
    primary_summary_path: Path | None
    primary_energy: float
    sector_result_dirs: dict[str, Path | None]
    sector_summary_paths: dict[str, Path | None]
    sector_energies: dict[str, float]
    primary_sector: str

    @property
    def all_ok(self) -> bool:
        return all(v is not None for v in self.sector_result_dirs.values())


def default_param_to_wells_n2(theta: np.ndarray, omega: float = 1.0) -> WellsLike:
    """Standard N=2 parametrisation: theta = [d], wells along x-axis."""
    d = float(theta[0])
    return [
        {"center": [-0.5 * d, 0.0], "omega": float(omega), "n_particles": 1},
        {"center": [+0.5 * d, 0.0], "omega": float(omega), "n_particles": 1},
    ]


def make_uniform_chain_param_to_wells(
    n_wells: int,
    *,
    omega: float = 1.0,
) -> Callable[[np.ndarray], WellsLike]:
    """Uniform spacing N-well chain: theta = [d], wells at -((N-1)/2)*d, ..., +((N-1)/2)*d.

    Centres lie on the x-axis with the chain centred on the origin.
    """
    if n_wells < 2:
        raise ValueError("Chain parametrisation requires n_wells >= 2.")

    def _to_wells(theta: np.ndarray) -> WellsLike:
        d = float(theta[0])
        offsets = (np.arange(n_wells) - 0.5 * (n_wells - 1)) * d
        return [
            {"center": [float(x), 0.0], "omega": float(omega), "n_particles": 1}
            for x in offsets
        ]

    return _to_wells


def make_per_bond_chain_param_to_wells(
    n_wells: int,
    *,
    omega: float = 1.0,
) -> Callable[[np.ndarray], WellsLike]:
    """Per-bond spacing chain: theta = [d_01, d_12, ..., d_{N-2,N-1}].

    The chain is centred so the centroid of the wells lies on the origin.
    """
    if n_wells < 2:
        raise ValueError("Per-bond chain parametrisation requires n_wells >= 2.")

    def _to_wells(theta: np.ndarray) -> WellsLike:
        bonds = np.asarray(theta, dtype=np.float64)
        if bonds.size != n_wells - 1:
            raise ValueError(
                f"Per-bond chain expects {n_wells - 1} parameters, got {bonds.size}."
            )
        x = np.zeros(n_wells, dtype=np.float64)
        x[1:] = np.cumsum(bonds)
        x = x - x.mean()
        return [
            {"center": [float(xi), 0.0], "omega": float(omega), "n_particles": 1}
            for xi in x
        ]

    return _to_wells


def make_dimer_chain_n4_param_to_wells(
    *,
    omega: float = 1.0,
) -> Callable[[np.ndarray], WellsLike]:
    """Symmetric N=4 chain with theta = [d_outer, d_middle].

    Wells live on the x-axis with bond layout
    ``d_outer | d_middle | d_outer`` and the chain centred on the origin.
    This bakes in the reflection symmetry expected for the N=4 ground state
    of an OBC Heisenberg chain so the inverse-design optimiser does not have
    to discover it implicitly via finite differences. Total span is
    ``2 d_outer + d_middle``.
    """

    def _to_wells(theta: np.ndarray) -> WellsLike:
        params = np.asarray(theta, dtype=np.float64)
        if params.size != 2:
            raise ValueError(
                f"Dimer N=4 chain expects 2 parameters [d_outer, d_middle], got {params.size}."
            )
        d_outer = float(params[0])
        d_middle = float(params[1])
        x = np.array(
            [
                -0.5 * d_middle - d_outer,
                -0.5 * d_middle,
                +0.5 * d_middle,
                +0.5 * d_middle + d_outer,
            ],
            dtype=np.float64,
        )
        return [
            {"center": [float(xi), 0.0], "omega": float(omega), "n_particles": 1}
            for xi in x
        ]

    return _to_wells


def make_dimer_chain_n8_param_to_wells(
    *,
    omega: float = 1.0,
) -> Callable[[np.ndarray], WellsLike]:
    """Symmetric N=8 chain with theta = [d1, d2, d3, d4].

    Builds a reflection-symmetric 8-well chain with bond layout
    ``d1 | d2 | d3 | d4 | d3 | d2 | d1`` (4 unique bond lengths from the
    end inwards). The chain is centred on the origin with all wells on
    the x-axis. Total span is ``2*(d1 + d2 + d3) + d4``.

    This bakes in the natural reflection symmetry of an isolated 8-site
    OBC chain so the optimiser does not have to rediscover it via finite
    differences, halving the gradient cost from 8 perturbed trainings per
    outer step to 4. The parametrisation can describe a wide range of
    physical geometries:

    * **Uniform**         d1 = d2 = d3 = d4
    * **Tetra-dimerised** d1, d4 small, d2, d3 large (boundary + central
      pairs decoupled from the bulk)
    * **End-to-end coupled** d2, d3 small, d1, d4 large (collapsed central
      4-site core, isolated end pair)
    """

    def _to_wells(theta: np.ndarray) -> WellsLike:
        params = np.asarray(theta, dtype=np.float64)
        if params.size != 4:
            raise ValueError(
                f"Dimer N=8 chain expects 4 parameters [d1, d2, d3, d4], got {params.size}."
            )
        d1, d2, d3, d4 = (float(v) for v in params)
        # Bond pattern (left-to-right, length 7):
        #   d1, d2, d3, d4, d3, d2, d1
        bonds = np.array([d1, d2, d3, d4, d3, d2, d1], dtype=np.float64)
        x = np.zeros(8, dtype=np.float64)
        x[1:] = np.cumsum(bonds)
        x = x - x.mean()
        return [
            {"center": [float(xi), 0.0], "omega": float(omega), "n_particles": 1}
            for xi in x
        ]

    return _to_wells


def make_dimer_pair_n8_param_to_wells(
    *,
    omega: float = 1.0,
) -> Callable[[np.ndarray], WellsLike]:
    """SSH-like dimerised N=8 chain with theta = [d_short, d_long].

    Two-parameter alternating-bond chain with the bond layout
    ``d_s | d_l | d_s | d_l | d_s | d_l | d_s``. This is the classic
    Su-Schrieffer-Heeger (SSH) dimerisation pattern restricted to OBC and
    centred on the origin. Total span: ``4 d_s + 3 d_l``.

    Why this matters for inverse design:

    * **2 parameters** (vs. 4 for ``dimer_chain_n8``) → only 2 perturbations
      per outer step in fd_forward, or 4 in fd_central. Per-step wall time
      drops by ~50 % at the price of less geometric expressivity.
    * The two-parameter SSH manifold contains the *physically interesting*
      transition between two regimes, both of which strongly modulate the
      end-to-end correlator ``<S_0 S_7>``:

      - **Trivial dimer phase** ``d_s << d_l``: 4 nearly-decoupled singlet
        pairs (0,1), (2,3), (4,5), (6,7). End-to-end correlation tends to
        the singlet-of-pair limit ``<S_0 S_7> -> 0`` (independent dimers).
      - **Topological / extended phase** ``d_s >> d_l``: a single 8-site
        chain with weakly coupled end pairs. Strong AFM correlations
        propagate the full length, ``<S_0 S_7>`` becomes maximally
        negative.

      An optimiser pushed by ``--mode neg_value`` should drive the system
      towards the topological end (large d_s / small d_l), making this
      a very interpretable demo of inverse design picking the SSH phase.
    """

    def _to_wells(theta: np.ndarray) -> WellsLike:
        params = np.asarray(theta, dtype=np.float64)
        if params.size != 2:
            raise ValueError(
                f"Dimer-pair N=8 chain expects 2 parameters [d_short, d_long], got {params.size}."
            )
        d_short, d_long = (float(v) for v in params)
        bonds = np.array(
            [d_short, d_long, d_short, d_long, d_short, d_long, d_short],
            dtype=np.float64,
        )
        x = np.zeros(8, dtype=np.float64)
        x[1:] = np.cumsum(bonds)
        x = x - x.mean()
        return [
            {"center": [float(xi), 0.0], "omega": float(omega), "n_particles": 1}
            for xi in x
        ]

    return _to_wells


def make_displacement_2d_param_to_wells(
    base_centers: np.ndarray | list[list[float]],
    *,
    omega: float = 1.0,
    dim: int = 2,
) -> Callable[[np.ndarray], WellsLike]:
    """Free per-well 2D displacements from a fixed base layout.

    The parameter vector has length ``dim * len(base_centers)`` and is laid
    out as ``[delta_0_x, delta_0_y, delta_1_x, delta_1_y, ...]`` (in 2D);
    well ``i`` is then placed at ``base_centers[i] + delta_i``. Each well
    has the same ``omega`` and ``n_particles = 1`` (one-per-well lane).

    This is the workhorse parametrisation for **disorder-pattern inverse
    design** (Phase 2C "MBL stretch"): instead of *sampling* random
    Gaussian disorder around a uniform chain and computing observables,
    the geometry optimiser searches over the full 2N-dimensional
    well-position landscape for the configuration that maximises a
    chosen observable (e.g. the bipartite well-set entanglement, or the
    end-to-end spin correlator). With ``--param-lower`` and
    ``--param-upper`` set to ``±sigma_max`` per coordinate, the search
    is constrained to the same ``|delta_i| <= sigma_max`` ball that the
    σ-sweep uses, but is now *deterministic* and *gradient-driven*
    rather than Monte-Carlo.

    Parameters
    ----------
    base_centers : (N, dim) array-like
        Base layout (e.g. uniform chain centres). Shape ``(N, dim)``.
    omega : float
        Per-well harmonic confinement (currently uniform across wells).
    dim : int
        Spatial dimension. Must match the number of columns of
        ``base_centers``.

    Notes
    -----
    * The search space scales linearly with ``N`` (``2 N`` parameters in
      2D). For ``fd_central``, gradient cost is ``2 * 2 N`` perturbed
      trainings per outer step, so this is most useful for ``N <= 8``;
      for larger systems, switch to ``fd_forward`` (``2 N`` trainings)
      or sub-sample the parameter set.
    * The base layout is recorded *as-is* (no global re-centring); if
      the original chain was centred on the origin, the geometry
      optimiser cannot exploit translation invariance to drift the
      whole chain — this is by design (translational drift would not
      change any of the inverse-design targets, but it would break
      warm-starting).
    """
    base = np.asarray(base_centers, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != int(dim):
        raise ValueError(
            f"base_centers must have shape (N, {dim}); got {base.shape}."
        )
    n_wells = int(base.shape[0])
    n_params = int(dim) * n_wells

    def _to_wells(theta: np.ndarray) -> WellsLike:
        params = np.asarray(theta, dtype=np.float64)
        if params.size != n_params:
            raise ValueError(
                f"displacement_2d expects {n_params} parameters "
                f"(dim={int(dim)} x N={n_wells}); got {params.size}."
            )
        deltas = params.reshape(n_wells, int(dim))
        out: list[dict[str, Any]] = []
        for i in range(n_wells):
            centre = (base[i] + deltas[i]).tolist()
            out.append({
                "center": [float(c) for c in centre],
                "omega": float(omega),
                "n_particles": 1,
            })
        return out

    return _to_wells


class GeometryOptimizer:
    """Bilevel inverse-design loop.

    Inner training is delegated to ``scripts/run_two_stage_ground_state.py``;
    every outer step writes a fresh YAML config (with the current geometry
    and an optional ``init_from`` warm start), launches the trainer, then
    evaluates the target on the resulting checkpoint.
    """

    def __init__(
        self,
        base_config_path: str | Path,
        *,
        target: str | Callable[[GeomEvalContext], float] = "energy",
        param_to_wells: Callable[[np.ndarray], WellsLike] | None = None,
        param_init: list[float] | None = None,
        param_step: list[float] | None = None,
        param_lower: list[float] | None = None,
        param_upper: list[float] | None = None,
        n_outer_steps: int = 15,
        lr_param: float = 0.3,
        sense: str | None = None,
        gradient_method: str = "auto",
        n_hf_samples: int = 2048,
        stage_a_epochs: int = 3000,
        stage_b_epochs: int = 2000,
        stage_a_min_energy: float = 0.5,
        stage_a_strategy: str = "auto",
        device: str = "cuda:0",
        out_dir: Path | None = None,
        seed: int = 42,
        warm_start: bool = True,
        target_kwargs: dict[str, Any] | None = None,
        spin_overrides: dict[str, dict[str, Any]] | None = None,
        primary_sector: str | None = None,
    ):
        self.base_cfg_path = Path(base_config_path).resolve()
        self.target = target
        self.param_to_wells = param_to_wells or default_param_to_wells_n2
        self.n_outer = int(n_outer_steps)
        self.lr_param = float(lr_param)
        if sense is None:
            sense = "min" if (isinstance(target, str) and target == "energy") else "max"
        if sense not in {"max", "min"}:
            raise ValueError(f"sense must be 'max' or 'min', got {sense!r}.")
        self.sense = sense
        self.gradient_method = gradient_method
        self.n_hf = int(n_hf_samples)
        self.stage_a_epochs = int(stage_a_epochs)
        self.stage_b_epochs = int(stage_b_epochs)
        self.stage_a_min_energy = float(stage_a_min_energy)
        self.stage_a_strategy = str(stage_a_strategy)
        self.device = device
        self.seed = int(seed)
        self.warm_start = bool(warm_start)
        self.target_kwargs = dict(target_kwargs or {})

        with self.base_cfg_path.open("r", encoding="utf-8") as fh:
            base_cfg = yaml.safe_load(fh)
        self._base_cfg = base_cfg
        self.omega = float(
            base_cfg.get("system", {}).get("omega", 1.0)
            if base_cfg.get("system", {}).get("type") in ("double_dot", "single_dot")
            else base_cfg["system"]["wells"][0].get("omega", 1.0)
        )

        if param_init is None:
            theta_init = self._infer_theta_from_config(base_cfg)
        else:
            theta_init = np.asarray(param_init, dtype=np.float64).copy()
        self._theta_init = theta_init
        self.theta = theta_init.copy()

        if param_step is None:
            param_step = [max(0.05, 0.1 * abs(t)) for t in theta_init]
        self.param_step = np.asarray(param_step, dtype=np.float64).copy()
        if self.param_step.shape != self.theta.shape:
            raise ValueError("param_step must match param_init in shape.")

        self.param_lower = (
            np.asarray(param_lower, dtype=np.float64) if param_lower is not None else None
        )
        self.param_upper = (
            np.asarray(param_upper, dtype=np.float64) if param_upper is not None else None
        )

        # Resolve spin overrides (multi-sector trainings per geometry).
        # When None, the optimizer trains a single inner run per geometry
        # using whatever spin block the base config supplies. When set,
        # each named sector gets its own training; ``primary_sector`` selects
        # which one is reported as the geometry's "energy".
        if isinstance(target, str) and target == "exchange_gap" and spin_overrides is None:
            spin_overrides = self._default_exchange_gap_overrides()
        self.spin_overrides = (
            None if spin_overrides is None else {k: dict(v) for k, v in spin_overrides.items()}
        )
        if self.spin_overrides:
            if primary_sector is None:
                primary_sector = "singlet" if "singlet" in self.spin_overrides else next(
                    iter(self.spin_overrides)
                )
            if primary_sector not in self.spin_overrides:
                raise ValueError(
                    f"primary_sector '{primary_sector}' not in spin_overrides keys {list(self.spin_overrides)}."
                )
        self.primary_sector = primary_sector or "primary"

        if isinstance(target, str):
            if target == "energy":
                self._target_fn = self._target_fn_energy
            elif target == "entanglement_n2":
                self._target_fn = self._target_fn_entanglement_n2
            elif target == "well_set_entanglement":
                self._target_fn = self._target_fn_well_set_entanglement
            elif target == "exchange_gap":
                self._target_fn = self._target_fn_exchange_gap
            elif target == "spin_correlator":
                self._target_fn = self._target_fn_spin_correlator
            elif target == "effective_J":
                self._target_fn = self._target_fn_effective_J
            elif target == "pair_corr":
                self._target_fn = self._target_fn_pair_corr
            else:
                raise ValueError(
                    f"Unknown built-in target '{target}'. "
                    "Pass a callable for custom targets."
                )
        else:
            self._target_fn = target  # type: ignore[assignment]

        self.out_dir = Path(out_dir) if out_dir else (
            REPO / "results" / "inverse_design" /
            f"{self.base_cfg_path.stem}_{target if isinstance(target, str) else 'custom'}"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._last_warm_start_dirs: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> tuple[np.ndarray, list[StepRecord]]:
        """Run the outer geometry optimisation loop."""
        history: list[StepRecord] = []

        for step in range(self.n_outer):
            t0 = time.time()
            LOGGER.info(
                "=== Outer step %d/%d | theta=%s ===",
                step + 1, self.n_outer, np.round(self.theta, 4).tolist(),
            )

            # --- 1. Train at current theta (centre evaluation, all sectors) ---
            wells_centre = self.param_to_wells(self.theta)
            centre_ctx = self._train_geometry(
                tag=f"step{step:03d}_centre",
                wells=wells_centre,
                init_warm_dirs=self._last_warm_start_dirs if self.warm_start else None,
            )
            if centre_ctx.primary_result_dir is None:
                LOGGER.warning("[step %d] Centre training failed; aborting.", step)
                break

            E_centre = float(centre_ctx.primary_energy)
            T_centre = self._evaluate_target(centre_ctx)
            if self.spin_overrides:
                sector_energies_str = ", ".join(
                    f"E_{name}={E:.5f}" for name, E in centre_ctx.sector_energies.items()
                )
                LOGGER.info(
                    "[step %d] centre: T=%.5f | %s",
                    step, T_centre, sector_energies_str,
                )
            else:
                LOGGER.info(
                    "[step %d] centre: E=%.5f, T=%.5f",
                    step, E_centre, T_centre,
                )

            # --- 2. Compute gradient ∂T/∂θ ---
            method = self._select_gradient_method()
            if method == "hf":
                grad = self._hellmann_feynman_gradient_param(
                    centre_ctx.primary_result_dir, wells_centre
                )
                perturbed = []
            else:
                grad, perturbed = self._fd_gradient_param(
                    step,
                    theta_centre=self.theta,
                    target_centre=T_centre,
                    init_warm_ctx=centre_ctx,
                    fd_mode=method,
                )
            LOGGER.info(
                "[step %d] grad(theta) = %s  (|grad|=%.4f)",
                step, np.round(grad, 5).tolist(), float(np.linalg.norm(grad)),
            )

            # --- 3. Update theta ---
            sign = 1.0 if self.sense == "max" else -1.0
            theta_new = self.theta + sign * self.lr_param * grad
            theta_new = self._clip_theta(theta_new)
            theta_change = theta_new - self.theta
            LOGGER.info(
                "[step %d] theta update: %s -> %s  (Δ=%s)",
                step,
                np.round(self.theta, 4).tolist(),
                np.round(theta_new, 4).tolist(),
                np.round(theta_change, 4).tolist(),
            )

            # --- 4. Record & advance ---
            dt = time.time() - t0
            sector_dirs_str = (
                {k: (str(v) if v is not None else None)
                 for k, v in centre_ctx.sector_result_dirs.items()}
                if self.spin_overrides else None
            )
            record = StepRecord(
                step=step,
                theta=self.theta.tolist(),
                target=float(T_centre),
                energy=float(E_centre),
                grad_theta=grad.tolist(),
                wells=[list(w["center"]) for w in wells_centre],
                centre_result_dir=(
                    str(centre_ctx.primary_result_dir)
                    if centre_ctx.primary_result_dir else None
                ),
                perturbed_targets=perturbed,
                dt_sec=float(dt),
                sector_energies=(
                    dict(centre_ctx.sector_energies) if self.spin_overrides else None
                ),
                sector_result_dirs=sector_dirs_str,
            )
            history.append(record)
            self._save_history(history)

            self.theta = theta_new
            # Warm-start each sector independently from its own centre run.
            for sector_name, rd in centre_ctx.sector_result_dirs.items():
                if rd is not None:
                    self._last_warm_start_dirs[sector_name] = rd

        return self.theta, history

    # ------------------------------------------------------------------
    # Built-in targets
    # ------------------------------------------------------------------

    def _target_fn_energy(self, ctx: GeomEvalContext) -> float:
        result_dir = ctx.primary_result_dir
        if result_dir is None:
            return float("nan")
        try:
            with (result_dir / "result.json").open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if "final_energy" in payload:
                return float(payload["final_energy"])
            return float("nan")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to read energy from %s: %s", result_dir, exc)
            return float("nan")

    def _target_fn_entanglement_n2(self, ctx: GeomEvalContext) -> float:
        from observables.checkpoint_entanglement import entanglement_target_n2

        result_dir = ctx.primary_result_dir
        if result_dir is None:
            return float("nan")
        kwargs = {
            "metric": self.target_kwargs.get("metric", "dot_label_negativity"),
            "max_ho_shell": int(self.target_kwargs.get("max_ho_shell", 2)),
        }
        return entanglement_target_n2(result_dir, **kwargs)

    def _target_fn_well_set_entanglement(self, ctx: GeomEvalContext) -> float:
        """Mott-projected bipartite spin entanglement (N >= 2 in the Mott regime).

        The bipartition defaults to the first ``n_wells // 2`` wells (set A)
        versus the rest (set B), which corresponds to a half/half split for
        chains and a left/right split for double dots. Override via
        ``target_kwargs={"set_a": [...]}``.

        Available metrics (via ``target_kwargs={"metric": ...}``):
          * ``"von_neumann_entropy"`` (default)
          * ``"negativity"``
          * ``"log_negativity"``
          * ``"linear_entropy"``
          * ``"effective_schmidt_rank"``
        """
        from observables.spin_amplitude_entanglement import spin_entanglement_target

        result_dir = ctx.primary_result_dir
        if result_dir is None:
            return float("nan")
        kwargs: dict[str, Any] = {
            "metric": self.target_kwargs.get("metric", "von_neumann_entropy"),
        }
        if "set_a" in self.target_kwargs:
            kwargs["set_a"] = list(self.target_kwargs["set_a"])
        if "spin_sector" in self.target_kwargs:
            kwargs["spin_sector"] = tuple(self.target_kwargs["spin_sector"])
        return spin_entanglement_target(result_dir, **kwargs)

    def _target_fn_exchange_gap(self, ctx: GeomEvalContext) -> float:
        """Singlet-triplet exchange gap target.

        ``T = E_triplet - E_singlet`` by default (positive for AFM ground state).
        Set ``target_kwargs={"signed": False}`` to optimise ``|E_T - E_S|`` instead.
        Set ``target_kwargs={"target_J": J0}`` to optimise ``-(E_T - E_S - J0)^2``,
        which dialled with ``sense="max"`` drives the gap towards a *specific*
        target value J0 (Phase 2B-style J_eff engineering).

        Sector names default to ``"singlet"`` / ``"triplet"``; override via
        ``target_kwargs={"sector_low": ..., "sector_high": ...}`` if you used
        custom override names.
        """
        sector_low = self.target_kwargs.get("sector_low", "singlet")
        sector_high = self.target_kwargs.get("sector_high", "triplet")
        signed = bool(self.target_kwargs.get("signed", True))
        target_J = self.target_kwargs.get("target_J", None)

        E_low = ctx.sector_energies.get(sector_low, float("nan"))
        E_high = ctx.sector_energies.get(sector_high, float("nan"))
        if not (np.isfinite(E_low) and np.isfinite(E_high)):
            return float("nan")
        gap = float(E_high) - float(E_low)
        if target_J is not None:
            return -float((gap - float(target_J)) ** 2)
        return gap if signed else abs(gap)

    def _target_fn_spin_correlator(self, ctx: GeomEvalContext) -> float:
        """Direct spin-spin correlator ``C_{ij} = <c|S_i.S_j|c>`` target.

        This is the **cleanest** Heisenberg-style inverse-design observable —
        a single, unambiguous, ground-state expectation value that we
        compute directly from the PINN Mott amplitudes. There is no fitting
        ambiguity (unlike effective ``J_ij``) and no need for excited-state
        sectors (unlike ``exchange_gap``). One inner-loop training per
        geometry suffices.

        ``target_kwargs`` (passed via the CLI):
          * ``pair``: ``[i, j]`` — the well pair whose correlator is the
            scalar target. Required.
          * ``mode``: one of
              - ``"value"``           — return raw ``<S_i.S_j>``.
                With ``--sense=max`` this *maximises* the correlator
                (drives toward ``+0.25``, FM-aligned). With
                ``--sense=min`` it minimises (drives toward ``-0.75``,
                singlet limit).
              - ``"neg_value"``       — return ``-<S_i.S_j>``.
                Use with ``--sense=max`` to *strengthen AFM correlations*
                (drive correlator toward ``-0.75``).
              - ``"neg_squared_error"`` — return ``-(<S_i.S_j> - target)^2``.
                With ``--sense=max`` this drives the correlator *toward*
                ``target_value`` (engineer-to-spec).
              - ``"squared_error"``     — explicit minimisation form.
            Default: ``"value"``.
          * ``target_value``: required for ``neg_squared_error`` /
            ``squared_error`` modes. The correlator value to engineer.
          * ``spin_sector``: ``[n_up, n_down]`` sector to extract from
            (defaults to whatever the trained checkpoint contains).
        """
        from observables.effective_heisenberg import spin_correlator_target

        result_dir = ctx.primary_result_dir
        if result_dir is None:
            return float("nan")
        pair = self.target_kwargs.get("pair")
        if pair is None:
            raise ValueError(
                "spin_correlator target requires target_kwargs={'pair': [i, j], ...}."
            )
        kwargs: dict[str, Any] = {
            "pair": list(pair),
            "mode": str(self.target_kwargs.get("mode", "value")),
        }
        if "target_value" in self.target_kwargs:
            kwargs["target_value"] = float(self.target_kwargs["target_value"])
        if "spin_sector" in self.target_kwargs:
            kwargs["spin_sector"] = tuple(self.target_kwargs["spin_sector"])
        return float(spin_correlator_target(result_dir, **kwargs))

    def _target_fn_effective_J(self, ctx: GeomEvalContext) -> float:
        """Effective Heisenberg ``J_{(i, j)}`` target (covariance fit).

        Fits ``H_eff = sum_{(p, q)} J_{pq} S_p.S_q`` to the PINN Mott
        amplitudes and reports a scalar reduction of one fitted bond
        ``J_{(i,j)}``. Supports the same ``mode`` options as
        :meth:`_target_fn_spin_correlator`.

        **Caveats**: For ``N >= 4`` the parent Hamiltonian fit is generically
        non-unique (multi-dimensional null space of the covariance matrix);
        we resolve the ambiguity by picking the direction that makes the
        PINN amplitudes the *ground state* of ``H_eff``, but the recovered
        ``J_{ij}`` matrix is genuinely degenerate under symmetry. For
        inverse design we **strongly recommend** ``spin_correlator`` instead.
        This target is provided mainly for analysis-by-optimisation use
        cases.

        ``target_kwargs``:
          * ``pair``: ``[i, j]`` — pair whose fitted ``J`` is the target.
          * ``mode``: ``"value"`` / ``"neg_squared_error"`` / ``"squared_error"``.
          * ``target_value``: required for the ``squared_error`` modes.
          * ``pairs``: optional list of ``(i, j)`` pairs to include in the fit
            basis. Default: nearest-neighbour bonds for chains, or all
            ``C(N, 2)`` pairs.
          * ``spin_sector``: optional ``[n_up, n_down]``.
        """
        from observables.effective_heisenberg import effective_J_target

        result_dir = ctx.primary_result_dir
        if result_dir is None:
            return float("nan")
        pair = self.target_kwargs.get("pair")
        if pair is None:
            raise ValueError(
                "effective_J target requires target_kwargs={'pair': [i, j], ...}."
            )
        kwargs: dict[str, Any] = {
            "pair": list(pair),
            "mode": str(self.target_kwargs.get("mode", "value")),
        }
        if "target_value" in self.target_kwargs:
            kwargs["target_value"] = float(self.target_kwargs["target_value"])
        if "pairs" in self.target_kwargs:
            kwargs["pairs"] = [tuple(p) for p in self.target_kwargs["pairs"]]
        if "spin_sector" in self.target_kwargs:
            kwargs["spin_sector"] = tuple(self.target_kwargs["spin_sector"])
        return float(effective_J_target(result_dir, **kwargs))

    def _target_fn_pair_corr(self, ctx: GeomEvalContext) -> float:
        """Pair-correlation ``g_sigma(r0)`` engineer-to-spec target.

        Estimates the Gaussian-broadened pair density at ``r0`` from MCMC
        samples of ``|psi|^2`` on the trained checkpoint and reports a
        scalar reduction. Phase 1C deliverable.

        ``target_kwargs`` (from CLI):
          * ``r0``: required, the pair separation we care about (Bohr).
          * ``sigma``: optional Gaussian broadening width. ``None`` →
            ``0.15 * r0``.
          * ``mode``: ``"value"`` / ``"neg_value"`` /
            ``"neg_squared_error"`` / ``"squared_error"``. Same semantics
            as :meth:`_target_fn_spin_correlator`.
          * ``target_value``: required for the squared-error modes.
          * ``n_samples``: MCMC sample count (default 4096).
          * ``mh_warmup``, ``mh_decorrelation``: MH controls.
          * ``seed``: MCMC seed for reproducible FD gradients
            (default 42 → identical noise pattern for theta+eps and
            theta-eps perturbations, halving FD-gradient variance).
        """
        from observables.pair_correlation import pair_corr_target

        result_dir = ctx.primary_result_dir
        if result_dir is None:
            return float("nan")
        r0 = self.target_kwargs.get("r0")
        if r0 is None:
            raise ValueError(
                "pair_corr target requires target_kwargs={'r0': float, ...}."
            )
        kwargs: dict[str, Any] = {
            "r0": float(r0),
            "mode": str(self.target_kwargs.get("mode", "value")),
        }
        if "sigma" in self.target_kwargs and self.target_kwargs["sigma"] is not None:
            kwargs["sigma"] = float(self.target_kwargs["sigma"])
        if "target_value" in self.target_kwargs:
            kwargs["target_value"] = float(self.target_kwargs["target_value"])
        if "n_samples" in self.target_kwargs:
            kwargs["n_samples"] = int(self.target_kwargs["n_samples"])
        if "mh_warmup" in self.target_kwargs:
            kwargs["mh_warmup"] = int(self.target_kwargs["mh_warmup"])
        if "mh_decorrelation" in self.target_kwargs:
            kwargs["mh_decorrelation"] = int(self.target_kwargs["mh_decorrelation"])
        if "seed" in self.target_kwargs:
            kwargs["seed"] = (
                None if self.target_kwargs["seed"] is None
                else int(self.target_kwargs["seed"])
            )
        return float(pair_corr_target(result_dir, **kwargs))

    # ------------------------------------------------------------------
    # Spin-sector overrides
    # ------------------------------------------------------------------

    def _default_exchange_gap_overrides(self) -> dict[str, dict[str, Any]]:
        """N-aware default singlet/triplet sector specs for ``exchange_gap``.

        Defaults:
          * Singlet  S^z=0 sector: ``n_up = N/2 (round down)``, ``n_down = N - n_up``
          * Triplet  S^z=1 sector: ``n_up = singlet_n_up + 1``, ``n_down = singlet_n_down - 1``

        For N=2 this gives singlet=(1,1) and triplet=(2,0) (fully polarised
        m=+1 triplet, a single Slater determinant — much easier to optimise
        than the m=0 triplet). For N=4 it gives singlet=(2,2) and triplet=(3,1).

        Both sectors disable the N=2 ``architecture.singlet`` permanent
        ansatz unless the sector explicitly opts back in.
        """
        n = self._infer_n_particles()
        n_up_singlet = n // 2
        n_down_singlet = n - n_up_singlet
        if n_down_singlet < 1:
            raise ValueError(
                f"Cannot construct a default singlet/triplet pair with N={n}; "
                "supply spin_overrides explicitly."
            )
        n_up_triplet = n_up_singlet + 1
        n_down_triplet = n_down_singlet - 1
        singlet_override: dict[str, Any] = {
            "n_up": int(n_up_singlet),
            "n_down": int(n_down_singlet),
            "force_no_singlet_arch": False,
        }
        # For N=2, the legacy ``singlet_self_residual`` recipe (which forces the
        # singlet permanent ansatz, ``architecture.singlet=True``) is
        # *substantially* more stable than ``improved_self_residual`` under the
        # warm-starting regime used by the inverse-design loop. Without this
        # auto-default, the singlet sector at small ``d`` slides into a
        # non-physical minimum (E -> -35 Ha at d=1.97 has been observed).
        # Triplet (m=+1) cannot use the (1,1) singlet permanent and continues
        # to use whatever the user / outer optimiser supplies.
        if n == 2:
            singlet_override["stage_a_strategy"] = "singlet_self_residual"
        return {
            "singlet": singlet_override,
            "triplet": {
                "n_up": int(n_up_triplet),
                "n_down": int(n_down_triplet),
                "force_no_singlet_arch": True,
            },
        }

    def _infer_n_particles(self) -> int:
        sys_cfg = self._base_cfg.get("system", {})
        if sys_cfg.get("type") == "double_dot":
            return int(sys_cfg.get("n_left", 1)) + int(sys_cfg.get("n_right", 1))
        wells = sys_cfg.get("wells", [])
        if wells:
            return sum(int(w.get("n_particles", 1)) for w in wells)
        raise ValueError("Could not infer particle count from base config.")

    # ------------------------------------------------------------------
    # Training launcher
    # ------------------------------------------------------------------

    def _build_config(
        self,
        wells: WellsLike,
        run_name: str,
        init_from: Path | None,
        spin_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = copy.deepcopy(self._base_cfg)
        # Override system to the explicit "custom" form for full geometry control.
        old_system = cfg.get("system", {})
        new_system = {
            "type": "custom",
            "wells": [
                {
                    "center": list(w["center"]),
                    "omega": float(w.get("omega", self.omega)),
                    "n_particles": int(w.get("n_particles", 1)),
                }
                for w in wells
            ],
            "dim": int(old_system.get("dim", 2)),
            "coulomb": bool(old_system.get("coulomb", True)),
        }
        for key in (
            "smooth_T", "B_magnitude", "B_direction", "g_factor", "mu_B",
            "coulomb_strength", "zeeman_electron1_only", "zeeman_particle_indices",
        ):
            if key in old_system:
                new_system[key] = old_system[key]
        cfg["system"] = new_system

        cfg["run_name"] = run_name
        cfg["allow_missing_dmc"] = True

        training = dict(cfg.get("training", {}))
        training["device"] = self.device
        training["seed"] = self.seed
        cfg["training"] = training

        if init_from is not None:
            cfg["init_from"] = {
                "result_dir": str(Path(init_from).resolve()),
                "strict": False,
            }
        else:
            cfg.pop("init_from", None)

        # Inject explicit spin sector when requested. We always use the
        # n_up/n_down form; the wavefunction layer handles the rest.
        if spin_override is not None:
            spin_block: dict[str, Any] = {
                "n_up": int(spin_override["n_up"]),
                "n_down": int(spin_override["n_down"]),
            }
            cfg["spin"] = spin_block
            # The N=2 singlet permanent ansatz is hardwired to (1,1); a
            # triplet sector must fall back to the generic spin-determinant
            # path.
            if spin_override.get("force_no_singlet_arch", False):
                arch = dict(cfg.get("architecture", {}))
                if arch.get("singlet"):
                    arch["singlet"] = False
                # Generic chain ansatz needs at least one of multi_ref or a
                # backflow to carry correlation; default multi_ref=True is
                # sound for N≤8. Don't touch arch_type / hidden sizes.
                arch.setdefault("multi_ref", True)
                cfg["architecture"] = arch
            # Override-driven Stage A strategy (e.g. force improved_self_residual
            # for triplet sectors that can't use singlet_self_residual).
            if "stage_a_strategy" in spin_override and spin_override["stage_a_strategy"]:
                cfg.setdefault("_invdes_meta", {})["stage_a_strategy"] = str(
                    spin_override["stage_a_strategy"]
                )

        return cfg

    def _train(
        self,
        *,
        tag: str,
        wells: WellsLike,
        init_from: Path | None,
        spin_override: dict[str, Any] | None = None,
    ) -> tuple[Path | None, Path | None]:
        """Train at the given geometry and return ``(result_dir, summary_path)``.

        Returns ``(None, None)`` on failure.
        """
        cfg = self._build_config(
            wells,
            run_name=f"invdes_{tag}",
            init_from=init_from,
            spin_override=spin_override,
        )
        # Optional per-sector Stage A strategy override.
        meta = cfg.pop("_invdes_meta", {})
        sector_stage_a_strategy = meta.get("stage_a_strategy", None)
        cfg_path = self.out_dir / f"cfg_{tag}.yaml"
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)

        log_path = self.out_dir / f"train_{tag}.log"
        summary_path = self.out_dir / f"summary_{tag}.json"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO / "src") + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        # Stream trainer stdout/stderr unbuffered so train_{tag}.log fills in real
        # time. Without this, Python block-buffers output to the redirected file
        # and the log only flushes at process exit, hiding hours-long runs
        # behind a zero-byte log file.
        env["PYTHONUNBUFFERED"] = "1"
        if self.device.startswith("cuda:"):
            env["CUDA_MANUAL_DEVICE"] = self.device.replace("cuda:", "")

        stage_a_strategy = sector_stage_a_strategy or self.stage_a_strategy
        cmd = [
            "python3.11",
            "-u",
            str(RUNNER),
            "--config", str(cfg_path),
            "--stage-a-strategy", str(stage_a_strategy),
            "--stage-a-epochs", str(self.stage_a_epochs),
            "--stage-b-epochs", str(self.stage_b_epochs),
            "--seed-override", str(self.seed),
            "--stage-a-min-energy", f"{self.stage_a_min_energy:g}",
            "--summary-json", str(summary_path),
        ]
        LOGGER.info("[%s] launching trainer (log: %s)", tag, log_path.name)
        with log_path.open("w") as fh:
            ret = subprocess.run(cmd, stdout=fh, stderr=fh, env=env)
        if ret.returncode != 0:
            LOGGER.warning(
                "[%s] trainer exited with code %d (see %s)", tag, ret.returncode, log_path
            )
            return None, None

        if not summary_path.exists():
            LOGGER.warning("[%s] expected summary JSON missing at %s", tag, summary_path)
            return None, None

        result_dir = self._read_result_dir_from_summary(summary_path)
        if result_dir is None:
            LOGGER.warning("[%s] could not resolve result_dir from %s", tag, summary_path)
        return result_dir, summary_path

    def _train_geometry(
        self,
        *,
        tag: str,
        wells: WellsLike,
        init_warm_dirs: dict[str, Path] | None,
    ) -> GeomEvalContext:
        """Train every required spin sector at this geometry; return a context.

        For single-sector targets (``spin_overrides=None``) this calls
        ``_train`` once and packages the result under sector key ``"primary"``.
        For multi-sector targets it spawns one inner training per spin sector,
        each warm-started from its own previous-step centre when available.
        """
        sector_dirs: dict[str, Path | None] = {}
        sector_summaries: dict[str, Path | None] = {}
        sector_energies: dict[str, float] = {}

        if self.spin_overrides is None:
            warm_dir = (init_warm_dirs or {}).get(self.primary_sector)
            rd, sp = self._train(tag=tag, wells=wells, init_from=warm_dir)
            sector_dirs[self.primary_sector] = rd
            sector_summaries[self.primary_sector] = sp
            sector_energies[self.primary_sector] = self._read_energy_from_summary(sp)
        else:
            for sector_name, sector_spec in self.spin_overrides.items():
                sector_tag = f"{tag}__{sector_name}"
                warm_dir = (init_warm_dirs or {}).get(sector_name)
                rd, sp = self._train(
                    tag=sector_tag,
                    wells=wells,
                    init_from=warm_dir,
                    spin_override=sector_spec,
                )
                sector_dirs[sector_name] = rd
                sector_summaries[sector_name] = sp
                sector_energies[sector_name] = self._read_energy_from_summary(sp)

        primary_dir = sector_dirs.get(self.primary_sector)
        primary_summary = sector_summaries.get(self.primary_sector)
        primary_energy = sector_energies.get(self.primary_sector, float("nan"))

        return GeomEvalContext(
            wells=wells,
            primary_result_dir=primary_dir,
            primary_summary_path=primary_summary,
            primary_energy=float(primary_energy),
            sector_result_dirs=sector_dirs,
            sector_summary_paths=sector_summaries,
            sector_energies=sector_energies,
            primary_sector=self.primary_sector,
        )

    @staticmethod
    def _read_result_dir_from_summary(summary_path: Path) -> Path | None:
        try:
            with summary_path.open("r", encoding="utf-8") as fh:
                summary = json.load(fh)
            for stage in ("stage_b", "stage_a"):
                stage_payload = summary.get(stage)
                if not stage_payload:
                    continue
                rd = stage_payload.get("result_dir", "")
                if rd and Path(rd).exists():
                    return Path(rd)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to parse summary %s: %s", summary_path, exc)
        return None

    @staticmethod
    def _read_energy_from_summary(summary_path: Path | None) -> float:
        if summary_path is None:
            return float("nan")
        try:
            with summary_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            for stage in ("stage_b", "stage_a"):
                stage_payload = payload.get(stage)
                if not stage_payload:
                    continue
                E = stage_payload.get("result", {}).get("final_energy")
                if E is not None:
                    return float(E)
        except Exception:
            pass
        return float("nan")

    def _evaluate_target(self, ctx: GeomEvalContext) -> float:
        try:
            return float(self._target_fn(ctx))
        except Exception as exc:
            LOGGER.warning("Target evaluation failed: %s", exc)
            return float("nan")

    # ------------------------------------------------------------------
    # Gradient methods
    # ------------------------------------------------------------------

    def _select_gradient_method(self) -> str:
        if self.gradient_method != "auto":
            method = self.gradient_method
            if method not in {"hf", "fd_central", "fd_forward", "fd_backward"}:
                raise ValueError(
                    f"Unknown gradient_method={method!r}; expected one of "
                    "{'auto', 'hf', 'fd_central', 'fd_forward', 'fd_backward'}."
                )
            return method
        if isinstance(self.target, str) and self.target == "energy":
            return "hf"
        return "fd_central"

    def _is_diverged(self, summary_path: Path | None) -> bool:
        """Heuristic: a training is 'diverged' if final_energy is below a sane bound.

        For our trap+Coulomb systems final_energy >= 0 always; we use 0.1 as a
        soft floor, which is well below any reasonable ground state but well
        above the ``-O(10)`` Hartree negatives we get when training collapses.
        """
        if summary_path is None or not summary_path.exists():
            return True
        try:
            with summary_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            best_energy: float | None = None
            for stage in ("stage_b", "stage_a"):
                stage_payload = payload.get(stage)
                if not stage_payload:
                    continue
                E = stage_payload.get("result", {}).get("final_energy")
                if E is None:
                    continue
                if best_energy is None or E < best_energy:
                    best_energy = float(E)
            if best_energy is None:
                return True
            return best_energy < 0.1
        except Exception:
            return True

    def _ctx_is_diverged(self, ctx: GeomEvalContext | None) -> bool:
        """A multi-sector context is healthy iff every sector trained cleanly."""
        if ctx is None:
            return True
        for name, sp in ctx.sector_summary_paths.items():
            if self._is_diverged(sp):
                return True
            if ctx.sector_result_dirs.get(name) is None:
                return True
        return False

    def _fd_gradient_param(
        self,
        step: int,
        *,
        theta_centre: np.ndarray,
        target_centre: float,
        init_warm_ctx: GeomEvalContext | None,
        fd_mode: str = "fd_central",
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Finite-difference gradient on the parametrised target.

        ``fd_mode`` selects the FD scheme:

        * ``"fd_central"`` — train both ``theta + eps`` and ``theta - eps`` per
          parameter (2K perturbations for K parameters). Most accurate; default.
        * ``"fd_forward"`` — train only ``theta + eps`` per parameter and use
          ``(T_+ - T_centre) / eps``. Halves the gradient cost (K perturbations
          instead of 2K), at the price of first-order rather than
          second-order accuracy in ``eps``. Recommended for expensive (large
          N) systems where the centre training is already a good warm start
          and the per-step variation in ``T`` is well above PINN noise.
        * ``"fd_backward"`` — symmetric: only train ``theta - eps``.

        For all modes, the implementation falls back to one-sided differences
        when a perturbation diverges or hits a bound, and returns 0 (no
        update) if both sides fail.

        Each perturbation warm-starts every spin sector independently from its
        own centre training in this outer step.
        """
        grad = np.zeros_like(theta_centre)
        perturbed: list[dict[str, Any]] = []
        # Per-sector warm-start map for this step's perturbations.
        warm_dirs: dict[str, Path] = {}
        if init_warm_ctx is not None and self.warm_start:
            for sector_name, rd in init_warm_ctx.sector_result_dirs.items():
                if rd is not None:
                    warm_dirs[sector_name] = rd

        for i in range(theta_centre.size):
            eps = float(self.param_step[i])
            theta_plus = theta_centre.copy()
            theta_minus = theta_centre.copy()
            theta_plus[i] += eps
            theta_minus[i] -= eps

            do_plus = fd_mode in {"fd_central", "fd_forward"}
            do_minus = fd_mode in {"fd_central", "fd_backward"}

            plus_in_bounds = self._inside_bounds(theta_plus, i) if do_plus else False
            minus_in_bounds = self._inside_bounds(theta_minus, i) if do_minus else False

            wells_plus = self.param_to_wells(theta_plus) if do_plus else None
            wells_minus = self.param_to_wells(theta_minus) if do_minus else None

            tag_plus = f"step{step:03d}_dir{i}_plus"
            tag_minus = f"step{step:03d}_dir{i}_minus"

            ctx_plus: GeomEvalContext | None = None
            ctx_minus: GeomEvalContext | None = None
            if plus_in_bounds:
                ctx_plus = self._train_geometry(
                    tag=tag_plus,
                    wells=wells_plus,
                    init_warm_dirs=warm_dirs if self.warm_start else None,
                )
            if minus_in_bounds:
                ctx_minus = self._train_geometry(
                    tag=tag_minus,
                    wells=wells_minus,
                    init_warm_dirs=warm_dirs if self.warm_start else None,
                )

            plus_ok = (
                plus_in_bounds
                and ctx_plus is not None
                and ctx_plus.primary_result_dir is not None
                and not self._ctx_is_diverged(ctx_plus)
            )
            minus_ok = (
                minus_in_bounds
                and ctx_minus is not None
                and ctx_minus.primary_result_dir is not None
                and not self._ctx_is_diverged(ctx_minus)
            )

            T_plus = self._evaluate_target(ctx_plus) if plus_ok else float("nan")
            T_minus = self._evaluate_target(ctx_minus) if minus_ok else float("nan")

            mode = "central"
            if plus_ok and minus_ok and not np.isnan(T_plus) and not np.isnan(T_minus):
                grad[i] = (T_plus - T_minus) / (2.0 * eps)
            elif plus_ok and not np.isnan(T_plus):
                grad[i] = (T_plus - target_centre) / eps
                mode = "forward"
            elif minus_ok and not np.isnan(T_minus):
                grad[i] = (target_centre - T_minus) / eps
                mode = "backward"
            else:
                grad[i] = 0.0
                mode = "stalled"

            LOGGER.info(
                "[step %d dir %d] FD %-9s | T+=%.5g (ok=%s) T-=%.5g (ok=%s) -> grad[%d]=%+.5g",
                step, i, mode, T_plus, plus_ok, T_minus, minus_ok, i, grad[i],
            )

            perturbed_record: dict[str, Any] = {
                "direction": int(i),
                "eps": eps,
                "fd_mode": mode,
                "theta_plus": theta_plus.tolist(),
                "theta_minus": theta_minus.tolist(),
                "T_plus": float(T_plus),
                "T_minus": float(T_minus),
                "plus_in_bounds": bool(plus_in_bounds),
                "minus_in_bounds": bool(minus_in_bounds),
                "plus_ok": bool(plus_ok),
                "minus_ok": bool(minus_ok),
                "result_dir_plus": (
                    str(ctx_plus.primary_result_dir)
                    if ctx_plus and ctx_plus.primary_result_dir else None
                ),
                "result_dir_minus": (
                    str(ctx_minus.primary_result_dir)
                    if ctx_minus and ctx_minus.primary_result_dir else None
                ),
            }
            if self.spin_overrides:
                perturbed_record["sector_energies_plus"] = (
                    dict(ctx_plus.sector_energies) if ctx_plus else {}
                )
                perturbed_record["sector_energies_minus"] = (
                    dict(ctx_minus.sector_energies) if ctx_minus else {}
                )
            perturbed.append(perturbed_record)

        return grad, perturbed

    def _inside_bounds(self, theta: np.ndarray, dim: int) -> bool:
        if self.param_lower is not None and theta[dim] < self.param_lower[dim]:
            return False
        if self.param_upper is not None and theta[dim] > self.param_upper[dim]:
            return False
        return True

    def _hellmann_feynman_gradient_param(
        self,
        result_dir: Path,
        wells: WellsLike,
    ) -> np.ndarray:
        """Approximate ``∂E/∂θ`` via the Hellmann-Feynman theorem.

        For a parametrised set of wells ``R_k(θ)`` we have

            ∂E/∂θ_i = sum_k (∂R_k/∂θ_i) · ⟨∂V/∂R_k⟩_Ψ
                    = sum_k (∂R_k/∂θ_i) · ⟨-ω² (r_k - R_k)⟩_Ψ      (HO trap)

        ``∂R_k/∂θ_i`` is approximated by central differences on
        :func:`param_to_wells`.
        """
        n_wells = len(wells)
        dim = len(wells[0]["center"])

        # Average HF force per well (samples may not exist; approximate by Mott Gaussian).
        force_per_well = self._mott_force_per_well(result_dir, wells)

        grad = np.zeros_like(self.theta)
        for i in range(self.theta.size):
            eps = max(1e-3, float(self.param_step[i]) * 0.1)
            theta_plus = self.theta.copy()
            theta_minus = self.theta.copy()
            theta_plus[i] += eps
            theta_minus[i] -= eps

            wells_plus = self.param_to_wells(theta_plus)
            wells_minus = self.param_to_wells(theta_minus)
            for k in range(n_wells):
                dRk = (
                    np.asarray(wells_plus[k]["center"], dtype=np.float64)
                    - np.asarray(wells_minus[k]["center"], dtype=np.float64)
                ) / (2.0 * eps)
                grad[i] += float(np.dot(dRk, force_per_well[k]))
        return grad

    def _mott_force_per_well(
        self,
        result_dir: Path,
        wells: WellsLike,
    ) -> list[np.ndarray]:
        """``⟨-ω²(r_k - R_k)⟩_Ψ`` per well, sampled from the Mott Gaussian.

        Falls back to per-well Gaussian sampling when actual VMC samples
        are not available; this is exact in the deep-Mott limit and a
        controlled approximation away from it.
        """
        rng = np.random.default_rng(self.seed)
        omega = self.omega
        sigma = 1.0 / (omega ** 0.5)
        forces: list[np.ndarray] = []
        for well in wells:
            R_k = np.asarray(well["center"], dtype=np.float64)
            r_k = R_k + rng.normal(0.0, sigma, size=(self.n_hf, R_k.size))
            forces.append(-(omega ** 2) * (r_k - R_k).mean(axis=0))
        return forces

    # ------------------------------------------------------------------
    # State-management helpers
    # ------------------------------------------------------------------

    def _infer_theta_from_config(self, base_cfg: dict[str, Any]) -> np.ndarray:
        sys_cfg = base_cfg["system"]
        if sys_cfg.get("type") == "double_dot":
            return np.asarray([float(sys_cfg["separation"])], dtype=np.float64)
        wells = sys_cfg.get("wells")
        if wells is None or len(wells) < 2:
            raise ValueError(
                "Cannot infer theta from base config; please pass param_init explicitly."
            )
        c0 = np.asarray(wells[0]["center"], dtype=np.float64)
        c1 = np.asarray(wells[1]["center"], dtype=np.float64)
        return np.asarray([float(np.linalg.norm(c1 - c0))], dtype=np.float64)

    def _clip_theta(self, theta: np.ndarray) -> np.ndarray:
        out = theta.copy()
        if self.param_lower is not None:
            out = np.maximum(out, self.param_lower)
        if self.param_upper is not None:
            out = np.minimum(out, self.param_upper)
        return out

    def _save_history(self, history: list[StepRecord]) -> None:
        path = self.out_dir / "history.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                [vars(rec) for rec in history],
                fh,
                indent=2,
                default=str,
            )


__all__ = [
    "GeometryOptimizer",
    "GeomEvalContext",
    "StepRecord",
    "default_param_to_wells_n2",
    "make_uniform_chain_param_to_wells",
    "make_per_bond_chain_param_to_wells",
    "make_dimer_chain_n4_param_to_wells",
    "make_dimer_chain_n8_param_to_wells",
    "make_dimer_pair_n8_param_to_wells",
    "make_displacement_2d_param_to_wells",
]
