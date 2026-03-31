# Plan: Generalized Cleanup and Next-Phase Advancement

Date: 2026-03-31
Status: draft

## Objective
Stabilize and clean the generalized multi-well codepath so the repository has one trustworthy generalized ground-state workflow, then advance to the next physics phases only if cleanup and validation gates pass.

## Context
**Updated 2026-03-31 after full evidence remediation (see `plans/2026-03-31_fix-evidence-and-fair-comparison.md`).**

The virial checker sign was indeed fixed in `scripts/run_virial_check.py`; prior 26–71% virial failures were an artifact of the wrong sign convention. However, the earlier claim that "N=2 double-dot virial residual improved from about 10–12% to about 2.7–3.6%" is **not supported by evidence** and has been retracted. The fix-evidence investigation found:

**N=4 (reliable IS comparison, same sep=4.0):**
- Old training (reinforce_hybrid, bf_hidden=32): virial 3.5–4.0% (IS, ESS~3400, HIGH confidence)
- Improved training (fd_colloc, bf_hidden=64): virial 13.0–13.1% (MH burn300, LOW confidence)
- Energy improved ~0.75%, but virial is WORSE with the improved setup

**N=2 (unreliable comparison; cannot determine direction):**
- Old and improved models trained on different separations — a matching old baseline at sep=4.0 now exists (s401/s402) but MH evaluator is inconsistently calibrated between flat-ψ² old models (accept~0.74) and peaked-ψ² improved models (accept~0.50)
- MH numbers: old sep=4 virial ~19%, improved virial ~8.8% — direction favors improved, but evaluator bias is unresolved

**Root cause of degraded N=4 virial and unstable N=2 comparison:**
The fd_colloc + bf_hidden=64 combination causes **training-time ESS collapse**: improved N=2 training ran at ESS_median=13/step, improved N=4 at ESS_median=22/step. Old models trained at ESS 856–1638/step. The improved training gradients were computed from near-degenerate samples at nearly every epoch. The resulting wavefunctions may have converged to a local optimum of a noise-dominated objective.

**Consequence for this plan:** The generalized path is NOT cleanly validated as of this writing. The improved training setup does not reliably improve virial quality, and the training procedure itself has a fundamental sampling flaw. Step 1 of this plan (which asks to "freeze the generalized baseline and evidence") must use the numbers above, not the previously claimed 2.7–3.6%. Steps 3 and 4 of this plan now have a concrete root cause to fix: training-time ESS collapse in the fd_colloc regime.

The `SESSION_LOG.md` still points to quench/FD follow-up as the next session. `JOURNAL.md` records that prior conclusions were vulnerable to weak references. This plan's original intent — to harden the foundation before extending scope — is still correct, but the foundation is weaker than previously believed.

## Approach
Use a two-stage plan. Stage A cleans and hardens the generalized ground-state workflow: repository loose-thread audit, validation hardening, sampling-risk assessment, and workflow consolidation. Stage B is conditional: only if Stage A produces a clean, reproducible, and interpretable generalized path do we advance to the next phases. The recommended next phases are not “train more,” but generalized observables and generalized physics support built on the now-validated core. This route is preferred over immediately adding more architectures, because the current evidence says the main unresolved issues are workflow quality, sampling confidence, and codebase coherence rather than a total architecture failure.

## Foundation checks (must pass before new code)
- [ ] Data pipeline known-input check
- [ ] Split/leakage validity check
- [ ] Baseline existence or baseline-creation step identified
- [ ] Relevant existing implementation read and understood

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

### Step 6 — Clean result-artifact and dirty-worktree policy
**What:** Resolve generated-output sprawl and git hygiene without losing real artifacts. Decide which results belong in version control, which should be ignored, and how future campaign outputs are stored so implementation can commit code changes without colliding with large result deletions.
**Files:** `.gitignore`, `results/`, campaign output directories, any cleanup scripts already present
**Acceptance check:** `git status` can cleanly distinguish source/config changes from generated artifacts after the cleanup policy is applied.
**Risk:** A messy worktree keeps blocking atomic commits and makes future reviews harder to trust.

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
**Active step:** 2 / Audit loose threads in the repo
**Last evidence:** The first hardening pass is complete. The virial sign convention now lives in a reusable helper, the generalized LR schedule now applies at the start of each epoch rather than after the optimizer step, and `pytest tests/test_training.py -q` passes with the new regression checks.
**Current risk:** Repository churn is dominated by generated artifacts rather than source instability. Current `git status` shows hundreds of tracked deletions under `results/` and a smaller active generalized source/config tree, so cleanup is now mainly a policy and classification problem.
**Audit snapshot:**
- `results/`: dominant churn source, currently about 236 tracked deletions and 65 untracked additions. Candidate disposition is ignore/archive policy rather than source-by-source review.
- generalized source/config additions under `src/`, `scripts/`, and `configs/generalized/`: candidate keep/consolidate set, not cleanup noise.
- legacy validation removal (`src/observables/validation.py`, `tests/test_validation.py`): currently appears to be an unreferenced consolidation, but should still be reviewed before final cleanup.
- stale session guidance (`SESSION_LOG.md`): still points to quench/FD follow-up and now lags the generalized-ground-state evidence.
**Next action:** Finish the loose-thread classification list, then decide whether `.gitignore` policy changes should be applied now or staged as a separate cleanup commit.
**Blockers:** None.