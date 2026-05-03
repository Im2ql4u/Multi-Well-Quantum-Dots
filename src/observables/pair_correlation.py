"""Pair-correlation observables ``g_sigma(r0)`` and inverse-design target.

This module implements the *Phase 1C* deliverable: a scalar pair-correlation
observable ``g_sigma(r0) = <sum_{i<j} K_sigma(|r_i-r_j| - r0)> / [N(N-1)/2]``
estimated from MCMC samples of ``|psi|^2`` on a saved checkpoint, plus a
target wrapper ``pair_corr_target`` consumable by
``GeometryOptimizer``.

Why a Gaussian-broadened delta? The continuous-space pair-distance
distribution is a smooth function of geometry, so a finite-width kernel is
an unbiased estimator (in the limit of small sigma) and gives a low-noise
finite-N estimator at ``sigma > 0``. The width ``sigma`` is the resolution
at which we compare the geometric structure across runs.

The estimator uses the existing :func:`mcmc_resample` sampler, which
performs a Metropolis-Hastings random walk over the network's
``log|psi|`` with sensible defaults inherited from the training pipeline.

Usage
-----

>>> from observables.pair_correlation import pair_correlation_at_r0
>>> out = pair_correlation_at_r0(
...     "results/inverse_design/n2_smoke_p1e/step007_centre",
...     r0=3.5, sigma=0.4, n_samples=4096,
... )
>>> out["g_r0"]
0.234...

For inverse-design use:

>>> from observables.pair_correlation import pair_corr_target
>>> T = pair_corr_target(result_dir, r0=4.0, mode="neg_squared_error",
...                      target_value=0.3, n_samples=4096, seed=42)

The ``seed`` kwarg makes the sampler reproducible at a fixed geometry,
which is important for finite-difference gradients in the bilevel
optimisation loop.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from observables.checkpoint_loader import load_wavefunction_from_dir
from training.sampling import mcmc_resample, sample_multiwell_init


def _representative_omega(loaded) -> float:  # type: ignore[no-untyped-def]
    """Pick a representative ``omega`` for the MCMC step scale.

    For the multi-well systems we currently train, every well shares the
    same ``omega``. We just take the first one and warn if the system has
    heterogeneous wells.
    """
    omegas = [float(w.omega) for w in loaded.system.wells]
    if not omegas:
        return 1.0
    if max(omegas) - min(omegas) > 1e-6:
        # Use harmonic mean as a compromise; this affects only the MH
        # step scale, not correctness.
        return float(len(omegas) / sum(1.0 / max(o, 1e-8) for o in omegas))
    return float(omegas[0])


def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Return the upper-triangular pairwise distances ``|r_i - r_j|`` for ``i<j``.

    Parameters
    ----------
    x : torch.Tensor
        Particle positions, shape ``(B, N, D)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, K)`` with ``K = N(N-1)/2``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (B, N, D), got {tuple(x.shape)}.")
    n_part = x.shape[1]
    diffs = x.unsqueeze(2) - x.unsqueeze(1)
    distances = diffs.pow(2).sum(dim=-1).clamp_min(0.0).sqrt()
    triu = torch.triu_indices(n_part, n_part, offset=1, device=x.device)
    return distances[:, triu[0], triu[1]]


def pair_correlation_at_r0(
    result_dir: Path | str,
    *,
    r0: float,
    sigma: float | None = None,
    n_samples: int = 4096,
    mh_warmup: int = 400,
    mh_decorrelation: int = 4,
    mh_step_scale: float = 0.35,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Estimate the Gaussian-broadened pair correlation ``g_sigma(r0)``.

    The estimator is

    .. math::

        g_\\sigma(r_0) = \\frac{1}{N(N-1)/2}
            \\Big\\langle \\sum_{i<j} K_\\sigma(|r_i-r_j|-r_0) \\Big\\rangle_{|\\psi|^2}

    where ``K_sigma(d) = (1/(sigma*sqrt(2*pi))) * exp(-d^2/(2*sigma^2))``.

    The factor ``1/[N(N-1)/2]`` makes the estimator scale-free in the
    number of particles: ``g_sigma(r0)`` is the *per-pair* probability
    density at separation ``r0``.

    Parameters
    ----------
    result_dir
        Directory containing ``config.yaml`` and ``model.pt``.
    r0 : float
        Target pair separation (in Bohr). Must be positive.
    sigma : float or None
        Gaussian broadening width. ``None`` -> ``0.15 * r0`` (so larger
        ``r0`` -> proportionally larger window). Smaller ``sigma`` -> more
        local but noisier.
    n_samples : int
        Number of MCMC samples used in the estimator. Use 4–16 k for
        development, 64 k+ for production-quality numbers.
    mh_warmup : int
        Number of MH steps for warm-up before measurement.
    mh_decorrelation : int
        MH decorrelation factor for the production sweep.
    mh_step_scale : float
        Single-particle step scale for the random walk; scaled internally
        by ``omega^{-1/2}``. Typical value ``0.3-0.5``.
    seed : int or None
        If set, calls ``torch.manual_seed(seed)`` before sampling so the
        estimator is reproducible at fixed geometry. Recommended when
        used in finite-difference gradient loops.
    device : str / torch.device / None
        Override the device. Defaults to the checkpoint's recorded device.

    Returns
    -------
    dict
        Keys: ``g_r0``, ``r0``, ``sigma``, ``n_samples``, ``n_pairs``,
        ``mean_pair_distance``, ``per_sample_mean``, ``per_sample_std``,
        ``mh_acceptance``.
    """
    if r0 <= 0:
        raise ValueError(f"r0 must be positive, got {r0!r}.")
    if seed is not None:
        torch.manual_seed(int(seed))

    sigma_eff = float(sigma) if sigma is not None else max(0.15 * float(r0), 1e-3)

    loaded = load_wavefunction_from_dir(result_dir, device=device)
    system = loaded.system
    omega = _representative_omega(loaded)

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        _, log_abs = loaded.signed_log_psi(x)
        return log_abs

    x_init = sample_multiwell_init(
        n_samples,
        system=system,
        device=loaded.device,
        dtype=loaded.dtype,
    )

    x, acc_warm, _ = mcmc_resample(
        psi_log_fn=psi_log_fn,
        x_prev=x_init,
        n_keep=n_samples,
        n_elec=system.n_particles,
        dim=system.dim,
        omega=omega,
        device=loaded.device,
        dtype=loaded.dtype,
        system=system,
        sigma_fs=(0.5,),
        mh_steps=int(mh_warmup),
        mh_step_scale=float(mh_step_scale),
        mh_decorrelation=1,
    )

    x, acc_prod, _ = mcmc_resample(
        psi_log_fn=psi_log_fn,
        x_prev=x,
        n_keep=n_samples,
        n_elec=system.n_particles,
        dim=system.dim,
        omega=omega,
        device=loaded.device,
        dtype=loaded.dtype,
        system=system,
        sigma_fs=(0.5,),
        mh_steps=2,
        mh_step_scale=float(mh_step_scale),
        mh_decorrelation=int(mh_decorrelation),
    )

    pair_d = _pairwise_distances(x)
    n_pairs = int(pair_d.shape[1])
    if n_pairs == 0:
        raise ValueError("Cannot compute pair correlation for a 1-particle system.")

    sigma_t = torch.tensor(sigma_eff, device=x.device, dtype=x.dtype).clamp_min(1e-12)
    log_norm = -math.log(float(sigma_t)) - 0.5 * math.log(2.0 * math.pi)
    log_kernel = log_norm - 0.5 * ((pair_d - r0) / sigma_t).pow(2)
    kernel = log_kernel.exp()

    per_sample = kernel.sum(dim=1) / float(n_pairs)
    g_mean = float(per_sample.mean())
    g_std = float(per_sample.std(unbiased=False))
    mean_pair_distance = float(pair_d.mean())

    return {
        "g_r0": g_mean,
        "r0": float(r0),
        "sigma": float(sigma_eff),
        "n_samples": int(n_samples),
        "n_pairs": int(n_pairs),
        "mean_pair_distance": mean_pair_distance,
        "per_sample_mean": g_mean,
        "per_sample_std": g_std,
        "mh_acceptance_warmup": float(acc_warm),
        "mh_acceptance_production": float(acc_prod),
    }


def pair_corr_target(
    result_dir: Path | str,
    *,
    r0: float,
    sigma: float | None = None,
    mode: str = "value",
    target_value: float | None = None,
    n_samples: int = 4096,
    mh_warmup: int = 400,
    mh_decorrelation: int = 4,
    mh_step_scale: float = 0.35,
    seed: int | None = 42,
    device: str | torch.device | None = None,
) -> float:
    """Reduce the pair-correlation observable to a scalar target.

    Modes
    -----
    ``"value"``
        Return ``g_sigma(r0)`` as-is. Pair with ``--sense max`` to drive
        the pair-density at ``r0`` *up*, or ``--sense min`` to drive it
        *down*.
    ``"neg_value"``
        Return ``-g_sigma(r0)``. With ``--sense max`` this drives the
        density *down* (e.g. push electrons apart at ``r0``). Equivalent
        to ``--sense min`` on ``"value"``, provided as an explicit
        affordance for symmetry with the spin-correlator target.
    ``"neg_squared_error"``
        Return ``-(g_sigma(r0) - target_value)^2``. With ``--sense max``,
        drive the pair density *toward* ``target_value`` — the
        engineer-to-spec mode.
    ``"squared_error"``
        Same as above but flipped sign for explicit minimisation.

    The ``seed`` kwarg defaults to ``42`` so finite-difference gradient
    pairs see the same MCMC trajectory between perturbations (modulo the
    geometry change). Bump it (or set to ``None``) for stochastic noise
    estimates.
    """
    out = pair_correlation_at_r0(
        result_dir,
        r0=r0,
        sigma=sigma,
        n_samples=n_samples,
        mh_warmup=mh_warmup,
        mh_decorrelation=mh_decorrelation,
        mh_step_scale=mh_step_scale,
        seed=seed,
        device=device,
    )
    g = out["g_r0"]
    mode_l = mode.lower()
    if mode_l == "value":
        return float(g)
    if mode_l == "neg_value":
        return -float(g)
    if mode_l == "neg_squared_error":
        if target_value is None:
            raise ValueError("mode='neg_squared_error' requires target_value.")
        return -float((g - float(target_value)) ** 2)
    if mode_l == "squared_error":
        if target_value is None:
            raise ValueError("mode='squared_error' requires target_value.")
        return float((g - float(target_value)) ** 2)
    raise ValueError(
        f"Unknown mode={mode!r}. Expected one of "
        "{'value', 'neg_value', 'neg_squared_error', 'squared_error'}."
    )


__all__ = ["pair_correlation_at_r0", "pair_corr_target"]
