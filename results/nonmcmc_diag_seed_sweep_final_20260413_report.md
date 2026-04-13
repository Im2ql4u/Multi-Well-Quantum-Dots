# Non-MCMC Residual Seed Sweep Report (2026-04-13)

## Scope
- Systems: N2, N3, N4 one-per-well
- Seeds: 42, 314, 901 (all fully converged)
- Source data: `results/nonmcmc_diag_seed_sweep_final_20260413.json`

## Important Reference-Quality Note
- The diagonalization value is a finite-basis CI reference, not a mathematically exact bound.
- In this code path (`scripts/exact_diag_double_dot.py`), defaults are `n_sp_states=40` and `n_ci_compute=200` (truncated basis).
- Per your instruction, `E_model < E_diag` is not treated as an error; it is treated as possible diagonalization under-convergence.

## Metric Convention Used Here
- Symmetric difference (for transparency): `delta = E_model - E_diag`
- One-sided exceedance (primary for this report): `exceedance = max(E_model - E_diag, 0)`
- Percent exceedance: `100 * max((E_model - E_diag)/|E_diag|, 0)`

## Figures
1. Final energies vs diag reference: `results/figures/nonmcmc_seed_sweep_energy_vs_diag_20260413.png`
2. One-sided exceedance percent: `results/figures/nonmcmc_seed_sweep_one_sided_exceedance_pct_20260413.png`
3. Seed spread (mean±sd) with diag marker: `results/figures/nonmcmc_seed_sweep_seed_spread_20260413.png`

## Per-Run Table
| System | Seed | E_diag | E_final | Delta (E_final-E_diag) | One-sided exceedance % |
|---|---:|---:|---:|---:|---:|
| N2 | 42 | 2.25442431 | 2.25390015 | -0.00052416 | 0.000000 |
| N2 | 314 | 2.25442431 | 2.25403474 | -0.00038957 | 0.000000 |
| N2 | 901 | 2.25442431 | 2.25323214 | -0.00119217 | 0.000000 |
| N3 | 42 | 3.63700158 | 3.63625982 | -0.00074176 | 0.000000 |
| N3 | 314 | 3.63700158 | 3.63724785 | +0.00024627 | 0.006771 |
| N3 | 901 | 3.63700158 | 3.63685775 | -0.00014383 | 0.000000 |
| N4 | 42 | 5.10444947 | 5.10360638 | -0.00084309 | 0.000000 |
| N4 | 314 | 5.10444947 | 5.10258785 | -0.00186162 | 0.000000 |
| N4 | 901 | 5.10444947 | 5.10292713 | -0.00152234 | 0.000000 |

## System-Level Summary
| System | Mean energy | SD energy | Mean exceedance % | Max exceedance % |
|---|---:|---:|---:|---:|
| N2 | 2.25372234 | 0.00042983 | 0.000000 | 0.000000 |
| N3 | 3.63678847 | 0.00049765 | 0.002257 | 0.006771 |
| N4 | 5.10304045 | 0.00051864 | 0.000000 | 0.000000 |

## Interpretation
- Under the one-sided metric, most runs have 0 exceedance (i.e., they do not sit above the finite-basis diagonalization value).
- Non-zero exceedance appears only when a run energy is above the diagonalization reference; these are small in magnitude (order 1e-2% or less).
- Because the reference is truncated-basis CI, any below-reference result should be treated as “inconclusive vs CI quality,” not automatically as model failure.

## Next Validation Step (if needed)
- Tighten the diagonalization basis (increase `n_sp_states` and `n_ci_compute`) and check monotone CI convergence of E_diag for N4 first.