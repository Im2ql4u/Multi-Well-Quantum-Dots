# Plan: Magnetic Reference Campaign

Date: 2026-04-16
Status: in progress

## Objective
Answer the magnetic-field questions for N=2, N=3, and N=4 with an evidence chain that separates:
- zero-field one-per-well ground-state facts,
- magnetic exact-diag/shared-model reference physics,
- current one-per-well constrained-model magnetic behavior,
- and any future shared-model VMC/PINN magnetic implementation.

## Current State
- Zero-field one-per-well non-MCMC ground states are validated for N=2, N=3, and N=4 to within about 0.1% of the current CI targets.
- N=2 Löwdin/dot-projected entanglement is calibrated strongly enough to distinguish shared versus one-per-well reference lanes.
- N=3/N=4 block-partition entanglement infrastructure now exists and the Hermite-local quadrature path is materially more stable than the old local Legendre box path.
- Uniform-B one-per-well magnetic evolution is currently believed to be spatially trivial in the fixed-spin one-per-well model.
- We have one exact-diag/shared-model magnetic characterization artifact for N=2 at sep=4 and one one-per-well magnetic PINN artifact for N=3, but no systematic magnetic reference suite yet.

## Questions To Answer
1. After turning on a magnetic field, what state does each system evolve toward for N=2, N=3, and N=4?
2. Are those post-field states entangled, and if so in what sense?
3. Are the wavefunctions converged and numerically stable?
4. Are the post-field states genuinely different from the initial states, or is the change only an energy offset?
5. Do the results align with CI/exact diag, and if not, why not?

## Implementation Phases

### Phase 1 — Magnetic Reference Sweep
Build a reusable shared-model exact-diag magnetic sweep over separation and B.

Deliverables:
- `scripts/run_magnetic_reference_sweep.py`
- summary JSON in `results/diag_sweeps/`
- sweep tables for GS spin sector, gap, entropy, and negativity as functions of `(sep, B)`

Acceptance gate:
- Reproduce the known sep=4, B=0→0.5 singlet-to-triplet switch for N=2.
- Produce at least one small magnetic grid over `sep in {2,4,8}` and `B in {0.0,0.25,0.5,0.75,1.0}`.

### Phase 2 — Zero-Field Summary Freeze
Lock the current zero-field statements for N=2, N=3, N=4 from the validated artifacts.

Deliverables:
- one summary table for energy agreement, convergence status, and entanglement status
- explicit note that N=3 partition dependence is part of the result

Acceptance gate:
- every number in the summary table links to a saved artifact

### Phase 3 — Constrained Magnetic Baseline
Compare the current one-per-well magnetic evolution against the reference lane.

Deliverables:
- one comparison note per available N showing whether the post-field state differs spatially from the initial state
- for N=2 and N=3, state clearly whether uniform B is trivial in the current ansatz

Acceptance gate:
- direct reference-to-baseline comparison exists at matched `sep` and `B`

### Phase 4 — Decide Magnetic Modeling Upgrade
Choose whether to proceed with:
- shared-model/spin-sector-aware VMC/PINN magnetic implementation, or
- a documented stop where the constrained one-per-well model is declared insufficient for the magnetic question

Acceptance gate:
- decision is tied to evidence from Phases 1–3, not preference

## Risks
1. Shared-model exact diag may become too expensive for larger basis sizes or higher-N extensions.
2. Magnetic reference physics may be clear for N=2 but computationally incomplete for N=3/N=4 in the current shared-model tooling.
3. The current one-per-well magnetic baseline may remain spatially trivial by construction, which is useful diagnostically but not a full answer.

## Immediate Next Steps
1. Implement and run the magnetic reference sweep script for N=2.
2. Use that sweep to identify the first clean spin-sector transition map.
3. Then compare the existing one-per-well magnetic artifacts against that map before any new PINN magnetic run.