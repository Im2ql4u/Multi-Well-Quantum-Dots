# No-Reference Full Reproduction Blueprint (2026-04-22)

## Goal

Recreate the previous successful `E_ref`-guided result lanes with a fully CI-independent training workflow while preserving the robustness already achieved in the legacy campaign.

This is not a single benchmark. The old success story had multiple layers:

1. one-per-well ground states
2. singlet permanent separation sweep with entanglement
3. fixed-spin magnetic sector energies
4. downstream quench and entanglement products that depend on the reproduced ground states

The no-reference program should rebuild those layers in the same order.

## Why Order Matters

The ground-state runs are upstream of everything else.

- If the no-ref ground states are unstable, then singlet, magnetic, lambda, and quench conclusions are contaminated.
- If the ground-state lane is good only for some systems, later products should be restricted to those systems rather than treated as global validation.

So the correct sequence is:

1. reproduce the `N=2/3/4` one-per-well seed sweep
2. reproduce the `N=2` singlet separation sweep
3. reproduce the fixed-spin magnetic sector ladders
4. only then resume lambda and quench campaigns on reproduced no-ref ground states

## Canonical Legacy Success Lanes

### A. Ground-State Seed Sweep

Legacy success source:

- `results/nonmcmc_diag_seed_sweep_final_20260413_report.md`

Canonical configs:

- `configs/one_per_well/n2_nonmcmc_residual_anneal_s42.yaml`
- `configs/one_per_well/seed_sweep/n2_nonmcmc_residual_anneal_s314.yaml`
- `configs/one_per_well/seed_sweep/n2_nonmcmc_residual_anneal_s901.yaml`
- `configs/one_per_well/n3_nonmcmc_residual_anneal_s42.yaml`
- `configs/one_per_well/seed_sweep/n3_nonmcmc_residual_anneal_s314.yaml`
- `configs/one_per_well/seed_sweep/n3_nonmcmc_residual_anneal_s901.yaml`
- `configs/one_per_well/n4_nonmcmc_residual_anneal_s42.yaml`
- `configs/one_per_well/seed_sweep/n4_nonmcmc_residual_anneal_s314.yaml`
- `configs/one_per_well/seed_sweep/n4_nonmcmc_residual_anneal_s901.yaml`

Minimum validation:

- training convergence
- independent FD energy
- generalized virial
- occupancy fractions

New mandatory validation:

- compare train-vs-FD gap
- compare seed spread before and after FD re-evaluation
- classify each system as robust / mixed / unstable

### B. Singlet Permanent Separation Sweep

Legacy success source:

- `JOURNAL.md` entry `2026-04-16`

Canonical configs:

- `configs/one_per_well/n2_singlet_d2_s42.yaml`
- `configs/one_per_well/n2_singlet_d4_s42.yaml`
- `configs/one_per_well/n2_singlet_d6_s42.yaml`
- `configs/one_per_well/n2_singlet_d8_s42.yaml`
- `configs/one_per_well/n2_singlet_d12_s42.yaml`
- `configs/one_per_well/n2_singlet_d20_s42.yaml`

Minimum validation:

- training convergence
- final energy vs legacy lane
- particle and dot-projected entanglement
- L/R sector probabilities

New mandatory validation:

- independent FD energy and component split
- compare entanglement against the CI-calibrated interpretation, not only the old model output
- separate “right energy” from “right entanglement”

### C. Fixed-Spin Magnetic Sector Ladders

Legacy success source:

- `configs/magnetic/n3_*`
- `configs/magnetic/n4_*`
- `results/p5_n3_mag_*`
- `results/p5_n4_mag_*`

Canonical configs:

- `configs/magnetic/n3_0up3down_b0p5_s42.yaml`
- `configs/magnetic/n3_1up2down_b0p5_s42.yaml`
- `configs/magnetic/n3_2up1down_b0p5_s42.yaml`
- `configs/magnetic/n3_3up0down_b0p5_s42.yaml`
- `configs/magnetic/n4_0up4down_b0p5_s42.yaml`
- `configs/magnetic/n4_2up2down_b0p5_s42.yaml`
- `configs/magnetic/n4_4up0down_b0p5_s42.yaml`

Minimum validation:

- training convergence
- independent FD energy
- sector ranking by total energy

Important interpretation:

- these runs document fixed-spin sector energies
- they do not validate nontrivial uniform-`B` magnetic dynamics in the generalized ansatz

### D. Lambda Sweep

Legacy source:

- `configs/magnetic/n2_singlet_d4_lam0p00_s42.yaml`
- `configs/magnetic/n2_singlet_d4_lam0p25_s42.yaml`
- `configs/magnetic/n2_singlet_d4_lam0p50_s42.yaml`
- `configs/magnetic/n2_singlet_d4_lam0p75_s42.yaml`

Important note:

- these are already effectively CI-free in the legacy lane because `alpha_end=0`
- this means lambda should be treated as the first extension after the reproduction gates, not as part of the initial proof that the no-ref replacement is trustworthy

## Recommended No-Reference Training Protocol

Pure variance minimization alone is not robust enough across systems and seeds.

Current best candidate:

1. Stage A: direct variational warm start
   - `loss_type: weak_form`
   - `sampler: is`
   - fixed-proposal non-MCMC

2. Stage B: pure variance refinement
   - `loss_type: residual`
   - `residual_objective: residual`
   - `sampler: stratified`
   - initialize from Stage A checkpoint

Reasoning:

- Stage A supplies ground-state bias without CI input.
- Stage B restores the low-variance non-MCMC residual lane we actually want to scale.

## Acceptance Gates

### Ground States

Promote only if all of the following hold:

- stable convergence across seeds
- no occupancy-collapse signature
- generalized virial no worse than legacy, ideally materially better
- FD energy not worse than legacy beyond the expected evaluation noise

System-specific expectation based on the current audit:

- `N=3` should pass first
- `N=4` is plausible but needs an FD-energy tie or win
- `N=2` needs basin stabilization before promotion

### Singlet Sweep

Promote only if:

- energies stay near the legacy lane
- `d>=6` still lands in the LR singlet sector
- entanglement is stable under the same measurement settings

### Magnetic Sectors

Promote only if:

- sector ordering is reproduced
- fixed-spin energy offsets remain consistent
- no one mistakes this for proof of shared-spin magnetic physics

## New Infrastructure Added

To support this program:

- `src/run_ground_state.py` now accepts `init_from` so staged no-ref training can warm-start from a previous checkpoint
- `scripts/run_two_stage_ground_state.py` runs the generic two-stage CI-free protocol
- `scripts/run_noref_reproduction_suite.py` emits the canonical reproduction manifest for the legacy success lanes

## Immediate Next Action

Run the two-stage no-ref protocol on the ground-state seed sweep first.

That is the smallest reproduction set that still answers the core question:

can the CI-free workflow faithfully recover the legacy trusted lane without sacrificing robustness?
