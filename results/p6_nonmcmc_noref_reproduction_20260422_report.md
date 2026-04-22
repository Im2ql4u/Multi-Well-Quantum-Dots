# Reference-Free Non-MCMC Reproduction Report (2026-04-22)

## Scope
- Systems: `N=2`, `N=3`, `N=4`
- Seeds: `42`, `314`, `901`
- Training mode: `loss_type: residual`, `residual_objective: residual`
- Sampler: stratified non-MCMC
- Goal: test whether the fully reference-free residual objective reproduces the earlier reference-targeted lane

## Key Result
- The reference-free runs are stable and low-variance, but they do **not** reproduce the earlier reference-targeted energies.
- All three systems converge to lower energies than the earlier lane by a systematic amount:
  - `N=2`: mean shift `-1.20609%` vs prior CI target
  - `N=3`: mean shift `-0.56956%` vs prior CI target
  - `N=4`: mean shift `-0.38078%` vs prior CI target
- Relative to the earlier `seed=42` runs, the shifts are similarly systematic:
  - `N=2`: `-1.07874%`
  - `N=3`: `-0.50065%`
  - `N=4`: `-0.39862%`

## Per-System Summary

### N=2
- CI/reference-targeted benchmark used previously: `2.25442431`
- New seed results:
  - `s42`: `2.22953481`
  - `s314`: `2.23185570`
  - `s901`: `2.22031117`
- Mean: `2.22723389`
- Seed SD: `0.00498596`
- Mean delta vs prior target: `-0.02719042` (`-1.20609%`)

### N=3
- CI/reference-targeted benchmark used previously: `3.63700158`
- New seed results:
  - `s42`: `3.61769242`
  - `s314`: `3.61341204`
  - `s901`: `3.61775579`
- Mean: `3.61628675`
- Seed SD: `0.00203289`
- Mean delta vs prior target: `-0.02071483` (`-0.56956%`)

### N=4
- CI/reference-targeted benchmark used previously: `5.10444947`
- New seed results:
  - `s42`: `5.08351049`
  - `s314`: `5.08366338`
  - `s901`: `5.08786397`
- Mean: `5.08501261`
- Seed SD: `0.00201718`
- Mean delta vs prior target: `-0.01943686` (`-0.38078%`)

## Interpretation
- The objective is numerically well-behaved:
  - all runs finished,
  - ESS remained fixed at `512`,
  - final losses match final energy variances,
  - last-window energy fluctuations are small.
- But the result is **not** “same answer without the reference”.
- The current evidence is more consistent with “pure variance minimization finds a different low-variance eigenstate/lane” than with “the reference was unnecessary.”

## Immediate Conclusion
- We should **not** yet use this exact reference-free setup as the production replacement for the current reference-targeted pipeline.
- The next thing to test should be a hybrid reference-free schedule that restores ground-state bias without CI input, for example:
  - short direct-energy warm start with fixed-proposal `sampler: is`, then
  - switch to stratified `residual_objective: residual`.

## Artifacts
- Configs: `configs/one_per_well/no_ref_seed_sweep/`
- Runs:
  - `results/p6_n2_nonmcmc_residual_noref_*`
  - `results/p6_n3_nonmcmc_residual_noref_*`
  - `results/p6_n4_nonmcmc_residual_noref_*`
