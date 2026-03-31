# Plan: Generalized Observables Phase

Date: 2026-03-31
Status: draft

## Objective
Add trustworthy generalized observables on top of the validated generalized ground-state workflow, with success defined as reproducible outputs for density-related observables and at least one known-limit or symmetry-based validation per observable family.

## Context
The generalized ground-state path is now credible: corrected virial validation shows N=2 improved to about 3% residual and N=4 remains about 4%, while improved campaign energies are stable across seeds. The cleanup plan identified that the next best expansion is not richer training or magnetic-field complexity, but observables that can expose whether the generalized wavefunction is physically sensible beyond energy and virial checks. Sampling remains the main unresolved structural risk, so this phase must prioritize observables that can be cross-checked against simple limits or symmetries.

## Approach
Implement observables in the generalized path in increasing difficulty. Start with one- and two-body quantities that are directly derived from sampled configurations and easiest to validate, then add reduced diagnostics such as pair correlation and density moments before considering entanglement-style quantities. Reuse the current collocation/importance-resampling path instead of introducing a separate sampler. Only observables with a concrete validation check are allowed in this phase.

## Foundation checks (must pass before new code)
- [ ] Data pipeline known-input check
- [ ] Split/leakage validity check
- [ ] Baseline existence or baseline-creation step identified
- [ ] Relevant existing implementation read and understood

## Scope
**In scope:** generalized one-body density, pair correlation / radial correlation diagnostics, low-order density moments, observable-runner integration for generalized checkpoints, and tests/known-limit checks for each observable family.
**Out of scope:** magnetic-field extensions, time-dependent dynamics, FD-heavy benchmarking, architecture changes, and entanglement metrics that cannot be validated in this phase.

## Steps

### Step 1 — Audit existing observable code and notebook-era logic
**What:** Read current observable-related modules and any notebook or script logic that already computes densities, pair correlations, or summaries, then identify what can be ported versus what should be retired.
**Files:** `src/observables/`, `src/imaginary_time_pinn.py`, `scripts/generate_density_gifs.py`, relevant notebooks under `src/`
**Acceptance check:** A concrete list exists of observable computations to reuse, port, or avoid.
**Risk:** Duplicating inconsistent logic from legacy notebook-era paths.

### Step 2 — Define a minimal authoritative observable API
**What:** Add a small reusable API under `src/observables/` for evaluating sampled observables from generalized models. Keep it narrow and config-driven.
**Files:** `src/observables/`, `src/run_ground_state.py`, possibly a new generalized observable runner under `scripts/`
**Acceptance check:** There is one clear callable path from model checkpoint plus config to observable outputs.
**Risk:** Observable logic gets embedded into scripts and becomes hard to test.

### Step 3 — Implement one-body density and low-order spatial moments
**What:** Compute one-body density summaries and low-order moments such as `<r^2>` from generalized samples. Include output formats that are useful both numerically and for plotting.
**Files:** `src/observables/`, `scripts/`
**Acceptance check:** For non-interacting or symmetric cases, densities and moments obey the expected symmetry and produce stable repeated estimates.
**Risk:** Sampling noise produces visually plausible but numerically unstable density estimates.

### Step 4 — Implement pair-correlation observables
**What:** Add pair-distance histograms or `g(r)`-style diagnostics for the generalized wavefunction, with same-spin/opposite-spin separation when feasible.
**Files:** `src/observables/`, `scripts/`
**Acceptance check:** Pair-correlation outputs are finite, normalized consistently, and show physically sensible behavior in non-interacting and interacting double-dot cases.
**Risk:** Poor normalization or binning silently produces misleading physical interpretation.

### Step 5 — Add observable validation tests and smoke runs
**What:** For every observable family, add at least one known-limit or symmetry-based test and one end-to-end smoke run using an existing checkpoint/config.
**Files:** `tests/`, `configs/generalized/`, `scripts/`
**Acceptance check:** `pytest -v` passes and the smoke runner produces outputs under a dated `results/` directory without NaN/Inf.
**Risk:** Outputs “look fine” but are not actually validated.

### Step 6 — Decide whether entanglement-style observables are ready
**What:** Evaluate whether the now-implemented observable base is sufficient to support entanglement entropy or reduced-density-matrix style quantities. If not, explicitly defer them.
**Files:** `src/observables/`, current plans under `plans/`
**Acceptance check:** The phase ends with an explicit go/no-go decision on entanglement observables.
**Risk:** Overreaching into quantities that lack trustworthy validation.

## Risks and mitigations
- Sampling noise contaminates observables: validate each observable with a known symmetry or known-limit check before interpretation.
- Legacy notebook logic conflicts with generalized implementation: audit before porting and keep one authoritative API.
- Pair-correlation normalization errors lead to false physical claims: define normalization conventions explicitly in code and tests.
- Observable outputs create large result sprawl: write to dated run folders and keep output scope minimal.

## Success criteria
- There is one authoritative generalized observable API under `src/observables/`.
- One-body density and pair-correlation outputs exist for generalized checkpoints and pass concrete validation checks.
- The observable pipeline has tests and at least one end-to-end smoke run.
- The end of the phase includes an explicit decision on whether entanglement-style observables are ready or deferred.

## Current State
**Active step:** 1 / Audit existing observable code and notebook-era logic
**Last evidence:** Generalized cleanup plan execution identified generalized observables as the first approved next phase, with sampling still the main unresolved structural risk.
**Current risk:** Legacy observable logic may be fragmented across notebooks and scripts, making it easy to duplicate inconsistent implementations.
**Next action:** Audit `src/observables/`, legacy scripts, and notebooks for reusable observable logic.
**Blockers:** None.