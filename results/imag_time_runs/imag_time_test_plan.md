# Imaginary-Time TDSE Test Plan

## Physics checks
1. Non-interacting d=0, coulomb=off: verify E0~2 and gap~1.
2. Interacting d=0, coulomb=on: verify Kohn mode gap~1.
3. Intermediate d=4: compare VMC vs PINN consistency.
4. Large separation d=8: check slower decay and stable E(tau).

## Numerical checks
1. No NaN/Inf in trajectory energies and fit outputs.
2. Acceptance ratios remain in a reasonable range for eval MCMC.
3. Fit methods agree within uncertainty where successful.
4. Runtime and memory are tracked for each strategy.

## Suggested progression
1. Run smoke profile on d=0 for both strategies.
2. Run tiny profile on d=0,4 for both strategies.
3. Run baseline profile on d=0,4,8.
4. Run production profile after baseline diagnostics are clean.
