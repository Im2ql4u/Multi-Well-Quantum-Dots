# Plan: Generalized Cleanup and Next-Phase Advancement

Date: 2026-03-31
Status: in-progress

## Objective
Stabilize and clean the generalized multi-well codepath so the repository has one trustworthy generalized ground-state workflow, then advance to the next physics phases only if cleanup and validation gates pass.

## Context
Recent generalized validation changed the interpretation of the project substantially. The virial checker was wrong and is now fixed in `scripts/run_virial_check.py`, so prior 26-71% virial failures were overstated. With the corrected checker and the improved training setup (`fd_colloc`, larger backflow, warmup+cosine LR), N=2 double-dot virial residual improved from about 10-12% to about 2.7-3.6%, while N=4 remained in the roughly 4% band. That means the generalized path is now viable, but not yet cleanly hardened: ESS is still uneven, repo state is cluttered, and there are still overlapping legacy vs generalized workflows.

The latest `SESSION_LOG.md` still points to quench/FD follow-up as the next session, while the current session established that generalized ground-state validation is now credible enough to justify cleanup before extending scope. `JOURNAL.md` also records that prior provisional conclusions were vulnerable to weak references and fragile runtime paths; this plan is designed to avoid repeating that pattern by tightening the generalized foundation before moving into richer observables or magnetic-field extensions.

## Approach
Use a two-stage plan. Stage A cleans and hardens the generalized ground-state workflow: repository loose-thread audit, validation hardening, sampling-risk assessment, and workflow consolidation. Stage B is conditional: only if Stage A produces a clean, reproducible, and interpretable generalized path do we advance to the next phases. The recommended next phases are not “train more,” but generalized observables and generalized physics support built on the now-validated core. This route is preferred over immediately adding more architectures, because the current evidence says the main unresolved issues are workflow quality, sampling confidence, and codebase coherence rather than a total architecture failure.

## Foundation checks (must pass before new code)
- [x] Data pipeline known-input check
- [x] Split/leakage validity check
- [x] Baseline existence or baseline-creation step identified
- [x] Relevant existing implementation read and understood

## Baseline evidence

| Case | Baseline corrected virial | Improved virial | Final energy | Seed spread | Verified | Uncertain |
|---|---:|---:|---:|---:|---|---|
| N=2 CTNN/Unified double-dot | 10.4-12.2% | 2.7-3.6% | 2.24830 mean | std about 7.8e-05 | Corrected virial check is consistent across 4 seeds | ESS remains uneven in some runs |
| N=4 CTNN/Unified double-dot | 3.5-4.0% | 4.0-4.4% | 6.99890 mean | std about 9.5e-06 | Energies are very stable and virial remains in marginal band | Improvement vs baseline is not demonstrated |
| Non-interacting N=2 known limit | exact 2.0 | 2.00442 | 2.00442 | single run | Known-limit pipeline check passes on current generalized path | Energy variance remains finite because this is not a strict eigenvalue proof |

Verified inputs for the table:
- improved campaign 6/6 jobs exit code 0 in `results/20260330_193737_generalized_multigpu_campaign/`
- corrected virial checker in `scripts/run_virial_check.py`
- targeted tests passed: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_training.py tests/test_wavefunction.py -q` -> `11 passed`
- known-limit rerun: `results/20260331_083709_validate_nonint_n2_double_1_1/result.json` reports `final_energy=2.00442097014668`

## Scope
**In scope:** generalized ground-state workflow cleanup, validation hardening, ESS/sampling audit, result-artifact policy cleanup, generalized config/workflow consolidation, and planning the next generalized phases behind explicit gates.
**Out of scope:** new external dependencies, large-scale architecture search, FD-heavy campaigns, major README rewrite, and magnetic/quench claims based on unconverged FD references.

## Steps

### Step 1 — Freeze the generalized baseline and evidence
**What:** Create a single authoritative baseline summary for the generalized ground-state path from the improved campaign and corrected virial validation. Record the corrected before/after numbers for N=2 and N=4, and explicitly separate what is verified from what is still uncertain.
**Files:** `results/20260330_193737_generalized_multigpu_campaign/`, `results/validation_20260330/virial_results_v2.json`, `scripts/run_virial_check.py`, `plans/2026-03-31_generalized-cleanup-next-phases.md`
**Acceptance check:** A compact evidence table exists in the plan or companion session note with: baseline corrected virial, improved virial, final energies, seed spread, and current uncertainty bullets.
**Risk:** The team continues using stale or pre-sign-fix interpretations.

### Step 2 — Audit loose threads in the repo
**What:** Identify repository loose threads created by the generalized work: unrelated generated-result deletions, stale logs, obsolete configs, duplicated runners, old validation scripts still encoding outdated assumptions, and dirty-worktree artifacts that block clean commits.
**Files:** `.gitignore`, `results/`, `configs/generalized/`, `scripts/`, `plans/2026-03-29_generalized-quantum-dot.md`
**Acceptance check:** A concrete audit list exists with each loose thread labeled as one of: keep, archive, delete, ignore, consolidate, or defer.
**Risk:** Cleanup accidentally removes still-needed artifacts or mixes user changes with generated output policy.

Current audit list:
- `results/` tracked legacy artifacts currently deleted in the worktree: `defer` unless user explicitly wants restoration or permanent removal committed.
- timestamped generalized run directories under `results/20*`: `ignore` for future runs, `keep` for current evidence-bearing outputs, `archive` only by explicit cleanup pass.
- `results/validation_20260330/` and `results/20260330_193737_generalized_multigpu_campaign/`: `keep` as current authoritative evidence.
- `scripts/run_generalized_multigpu_campaign.py`: `keep` as authoritative generalized campaign launcher.
- `src/run_ground_state.py`: `keep` as authoritative generalized ground-state entry point.
- `scripts/run_virial_check.py`: `keep` as authoritative generalized physics validation script.
- `scripts/run_n2_simple_sweep.py`, `scripts/run_n2_simple_sweep_multigpu.sh`, `scripts/run_multiseed_imag_time_campaign.py`, `scripts/run_n12_focus_campaign.py`, `scripts/run_residual_campaign.py`: `defer` for legacy/imag-time classification; do not extend for generalized ground-state.
- `.gitignore` generated-artifact policy: `consolidate` so future timestamped runs and campaign logs are not mixed into normal source-control work.
- stale mixed-purpose logs under `results/*.log`: `ignore` for future generation, `defer` current deletion/restoration decisions.

Sampling trustworthiness verdict:
- Current generalized results are trustworthy enough for workflow advancement, but not yet strong enough to declare sampling “solved.”
- Evidence from `results/validation_20260331/improved_campaign_convergence.json`:
	- all six improved runs plateaued
	- N=2 late energy spread is very tight and virial residuals improved to about 3%
	- energy variance improved strongly in all runs
	- ESS drops below half-median are frequent across runs, so uncertainty is now concentrated in sampling robustness rather than optimizer instability
- Practical interpretation: use current generalized workflow for next-phase observables, but do not claim sampling robustness is complete and do not treat N=4 as fully saturated.

### Step 3 — Harden correctness checks around the generalized path
**What:** Add or tighten targeted checks for the exact pieces that recently caused false conclusions: virial sign correctness, scheduler behavior, and known-limit behavior. At minimum, define tests or scripted checks for non-interacting energy, virial theorem sign convention, and LR schedule endpoints.
**Files:** `scripts/run_virial_check.py`, `src/training/vmc_colloc.py`, `tests/`, `configs/generalized/validate_noninteracting_n2.yaml`, `configs/generalized/validate_noninteracting_n4.yaml`
**Acceptance check:** There is an executable check path that would fail if the virial sign regressed or if scheduler endpoints are wrong; known-limit validation still passes.
**Risk:** The code remains numerically correct today but fragile against future edits.

### Step 4 — Audit sampling robustness and ESS failure modes
**What:** Treat ESS variability as the main unresolved structural risk. Quantify ESS distributions across completed improved runs, compare low-ESS cases against final virial/energy quality, and determine whether current proposal design is merely noisy or methodologically limiting.
**Files:** `src/training/sampling.py`, `src/training/vmc_colloc.py`, `results/20260330_193737_generalized_multigpu_campaign/logs/`, `scripts/analyze_convergence.py`
**Acceptance check:** A short verdict exists answering: “Are current generalized results trustworthy despite ESS variability?” with evidence from completed runs, not intuition.
**Risk:** Low ESS may be producing apparently stable energies while silently weakening confidence in wavefunction quality.

### Step 5 — Consolidate the generalized workflow as the preferred ground-state path
**What:** Make the generalized path the clean default for future ground-state work by clarifying which files are authoritative and which are legacy. This includes deciding what remains in legacy/imaginary-time code, what is generalized ground-state only, and where future work should attach.
**Files:** `src/run_ground_state.py`, `src/training/`, `src/wavefunction.py`, `scripts/run_generalized_multigpu_campaign.py`, legacy runner scripts under `scripts/`
**Acceptance check:** There is a clear map of “use this path for generalized ground-state training” with at least one authoritative launcher, one authoritative validation path, and identified legacy scripts to stop extending.
**Risk:** Future work continues to fork across overlapping entry points and duplicated logic.

Authoritative workflow map:
- authoritative ground-state entry point: `src/run_ground_state.py`
- authoritative multi-GPU launcher: `scripts/run_generalized_multigpu_campaign.py`
- authoritative physics validation: `scripts/run_virial_check.py`
- authoritative training core: `src/training/vmc_colloc.py`
- authoritative wavefunction assembly: `src/wavefunction.py`
- legacy / do-not-extend for generalized ground-state: `scripts/run_n2_simple_sweep.py`, `scripts/run_n2_simple_sweep_multigpu.sh`, `scripts/run_multiseed_imag_time_campaign.py`, `scripts/run_n12_focus_campaign.py`, `scripts/run_residual_campaign.py`

### Step 6 — Clean result-artifact and dirty-worktree policy
**What:** Resolve generated-output sprawl and git hygiene without losing real artifacts. Decide which results belong in version control, which should be ignored, and how future campaign outputs are stored so implementation can commit code changes without colliding with large result deletions.
**Files:** `.gitignore`, `results/`, campaign output directories, any cleanup scripts already present
**Acceptance check:** `git status` can cleanly distinguish source/config changes from generated artifacts after the cleanup policy is applied.
**Risk:** A messy worktree keeps blocking atomic commits and makes future reviews harder to trust.

Current status:
- future timestamped generalized outputs are now ignored via `.gitignore`
- remaining `git status` noise is dominated by pre-existing tracked result deletions and unrelated user/worktree changes
- this is a real blocker to fully satisfying Step 6 without explicit user approval, because resolving it would require restoring or permanently deleting tracked result artifacts outside the safe scope of this task

### Step 7 — Define the gated next phases for the generalized codebase
**What:** Decide what “next phase” should mean now that generalized ground-state is credible. Recommended ordering:
1. Generalized observables phase: density matrices, pair correlation, entanglement entropy, and diagnostics from the generalized wavefunction.
2. Generalized physics phase: custom multi-well layouts and magnetic-field generalization only after observables are stable.
3. Generalized time-dependent phase: only after the generalized ground-state code is the trusted initialization path.
Include explicit gate conditions for moving from one phase to the next.
**Files:** `src/observables/`, `src/config.py`, `src/potential.py`, `src/run_ground_state.py`, current plans under `plans/`
**Acceptance check:** The plan names the next phase to implement first and states why it outranks the alternatives.
**Risk:** The project jumps into richer physics before the generalized base is clean enough to interpret outputs.

### Step 8 — Produce a follow-on implementation-ready plan for the selected next phase
**What:** After cleanup/audit conclusions are in hand, write a second plan dedicated to the first approved next phase. If cleanup fails its gates, write a remediation-only plan instead.
**Files:** `plans/`
**Acceptance check:** A successor plan exists with atomic steps and acceptance checks, and it does not assume unresolved cleanup risks are already solved.
**Risk:** The next implementation starts with unresolved ambiguity and repeats the current repo sprawl.

## Risks and mitigations
- ESS variability hides sampling bias: require a dedicated ESS/trustworthiness verdict before accepting the generalized workflow as “done.”
- Dirty worktree and result churn obscure code changes: separate generated-artifact policy cleanup from source-code cleanup, and do not revert user-owned result changes implicitly.
- Legacy/generalized duplication causes future divergence: explicitly mark one launcher and one validation path as authoritative.
- Premature expansion into magnetic/quench phases reintroduces weak-reference conclusions: keep FD-heavy or unconverged-reference work out of this plan.
- N=4 may still be architecture-limited even if workflow is clean: keep that as an open question, but only revisit architecture after sampling and workflow risks are assessed.

## Success criteria
- There is one trustworthy, clearly designated generalized ground-state workflow with corrected validation and reproducible campaign evidence.
- Loose threads in the repo are enumerated and dispositioned, with result-artifact policy no longer blocking clean source commits.
- Generalized validation has targeted correctness checks guarding the recent failure modes.
- ESS risk is assessed explicitly enough to know whether current results are trustworthy or only provisional.
- A follow-on plan exists for the first next phase, with explicit gate conditions for advancing further.

## Current State
**Active step:** 8 / Produce a follow-on implementation-ready plan for the selected next phase
**Last evidence:** Hardening checks passed (`20 passed` in full test suite); known-limit rerun reports `final_energy=2.00442097014668`; convergence audit saved to `results/validation_20260331/improved_campaign_convergence.json`; `.gitignore` now ignores future timestamped run artifacts.
**Current risk:** The generalized workflow is now clean enough to proceed, but full dirty-worktree cleanup is blocked by pre-existing tracked result deletions outside safe autonomous scope.
**Next action:** Write the successor plan for the first approved next phase: generalized observables.
**Blockers:** Full Step 6 completion requires explicit user direction on tracked result deletions already present in the worktree.