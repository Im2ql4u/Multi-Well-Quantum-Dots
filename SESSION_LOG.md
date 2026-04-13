# Session Log

Last session: [2026-04-13] — Converged 3-Seed Non-MCMC Benchmark, Reporting Update, and Cleanup
See ARCHIVE.md for full history.

## Next session
**Project objective:** Advance one-per-well N=2/N=3/N=4 quality with non-MCMC training while tightening reference-quality evidence for publication-grade claims.
**Active plan file:** [plans/2026-04-12_nonmcmc-residual-collocation-multiwell.md](plans/2026-04-12_nonmcmc-residual-collocation-multiwell.md)
**Recommended starting point:** Run a CI convergence ladder for N4 (`n_sp_states`, `n_ci_compute`) and re-evaluate interpretation of sub-reference energies.
**Open questions:**
- Are current diagonalization references sufficiently converged to support sign-sensitive claims?
- Do one-sided exceedance conclusions hold under tighter CI settings?
**Unverified assumptions:** Below-reference model energies currently assumed to be mostly CI-truncation effects.
**Active workarounds:** [configs/one_per_well/seed_sweep/n4_nonmcmc_residual_anneal_s901.yaml](configs/one_per_well/seed_sweep/n4_nonmcmc_residual_anneal_s901.yaml) uses `device: cuda:7` to bypass external contention on cuda:5.
**Foundation status:** Non-MCMC training is stable across three seeds for N=2/N=3/N=4; reporting now distinguishes finite-basis reference limits via one-sided exceedance.
**Context freshness:** fresh
**Contradiction flags:** none

## Session metrics (latest)
**Steps completed:** 4 of 4 planned
**Material deviations:** 2
**Evaluation gates triggered:** 2 (convergence + interpretation)
**Unresolved uncertainties:** 1
