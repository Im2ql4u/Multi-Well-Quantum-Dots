# Constraints

Durable project truths promoted from repeated sessions.
Only include constraints that have been validated by evidence.

## Verified Constraints

- Use fixed virial evaluation protocol for cross-run comparison: FD evaluator + MH steps/warmup parity across runs. Mixed protocols produced false regressions.
- Keep MCMC and non-MCMC result lanes separated (`results/mcmc_training/` vs `results/nonmcmc_training/`) to avoid evidence contamination.
- For non-MCMC residual/collocation training, per-batch MAD clipping of local energy is required for stability.
- Treat finite-basis CI diagonalization references as approximate unless convergence in `n_sp_states` and `n_ci_compute` has been explicitly checked.

## Suspected (Needs More Evidence)

- Small below-reference model energies are primarily due to CI truncation, not model overfitting or Hamiltonian mismatch.
