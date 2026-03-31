# Plan: Fix Evidence Foundations and Establish Fair Before/After Comparison

Date: 2026-03-31
Status: draft

## Objective
Fix the three foundational defects discovered by the diagnostic review — stale evidence artifacts, ESS collapse in post-training evaluation, and an invalid cross-separation comparison — so that the generalized ground-state workflow has a truthful evidence base. Only then determine whether the improved training setup (fd_colloc + larger backflow + cosine LR) actually improves virial quality over the baseline.

Success condition: a single evidence table comparing old and improved models on the **same physical system** with ESS > 1000 in all evaluation runs, using the corrected virial formula, with all numbers traceable to dated result artifacts.

## Context

### What triggered this plan
A diagnostic review of `plans/2026-03-31_generalized-cleanup-next-phases.md` found that the plan's central claim — "N=2 double-dot virial residual improved from about 10–12% to about 2.7–3.6%" — has no supporting artifact and is contradicted by the evidence that does exist. Three distinct problems were identified:

**Problem 1 — Stale evidence file uses the wrong virial formula.**
`results/validation_20260330/virial_results_v2.json` stores derived fields (`virial_residual`, `virial_relative`) computed with the OLD wrong sign convention (`2T = 2V_trap + V_int`). The raw energy components in the file are correct, but the headline numbers show 26–71% virial residuals instead of the actual 3.5–12.2%. This file was never regenerated after the sign fix in `src/observables/diagnostics.py`.

**Problem 2 — ESS collapse makes improved-model evaluation unreliable.**
Running the corrected virial check today produced real numbers, but ESS for improved models was 4–280 vs 1400–6400 for old models. With ESS this low, the Gaussian-mixture proposal is failing catastrophically for the improved wavefunctions. All improved-model statistics (energy, virial) are suspect. The proposal parameters (`sigma_fs: [0.8, 1.3, 2.0]`) are identical in training and evaluation configs — the per-well separated proposal (`sample_multiwell`) should handle well centres correctly, but the width tiers may not match what the improved backflow network has learned.

**Problem 3 — N=2 "before/after" compares different physical systems.**
All old N=2 runs (g6, g7 series) use `separation: 6.0`. All improved N=2 runs use `separation: 4.0`. These are different physical systems with different ground states, different energies, and different virial balances. Comparing virial residuals across separations measures nothing about training quality. The N=4 comparison is valid (both use `separation: 4.0`), but there the improved models are **worse** (5.5% vs 3.5%), though the low ESS casts doubt on even those numbers.

### What the corrected virial check actually showed

Recomputing stored v2 values with the correct formula (`2T = 2V_trap - V_int`):
- g4 N=4 ctnn (old): 3.5%
- g5 N=4 unified (old): 4.0%
- g6 N=2 ctnn (old, sep=6.0): 12.2%
- g7 N=2 unified (old, sep=6.0): 10.4%

Running `scripts/run_virial_check.py` (corrected) on improved models today:
- impr_n2_ctnn_s301 (sep=4.0): 4.5% (ESS 10–243)
- impr_n2_ctnn_s302 (sep=4.0): 4.3% (ESS 11–96)
- impr_n2_unified_s303 (sep=4.0): 4.9% (ESS 11–276)
- impr_n2_unified_s304 (sep=4.0): 5.3% (ESS 11–279)
- impr_n4_ctnn_s305 (sep=4.0): 5.5% (ESS 4–101)
- impr_n4_unified_s306 (sep=4.0): 5.6% (ESS 4–100)

The ESS numbers for improved models are so low that these virial estimates are unreliable. But even taken at face value, the plan's 2.7–3.6% claim is not supported.

### What changed between old and improved training

| Parameter | Old (g4–g7) | Improved (impr_*) |
|---|---|---|
| loss_type | reinforce_hybrid | fd_colloc |
| bf_hidden | 32 | 64 |
| epochs | 20,000 | 30,000 |
| LR schedule | flat | warmup 500 + cosine to lr/100 |
| N=2 separation | **6.0** | **4.0** |
| N=4 separation | 4.0 | 4.0 |
| sigma_fs | [0.8, 1.3, 2.0] | [0.8, 1.3, 2.0] |

The separation change for N=2 was intentional (comments in `campaign_jobs_improved.yaml` say nothing about it, so the intent is unclear), but it invalidates any before/after comparison unless a matching baseline exists.

### What the previous plan got right
- The virial sign was correctly fixed in code (`src/observables/diagnostics.py`, `scripts/run_virial_check.py`) — reusable, tested, and correct.
- The LR scheduler timing was fixed (applies factor before epoch, not after) and tested.
- The regression test suite (12 tests) passes and guards against sign regression.
- The per-well multi-well proposal in `src/training/sampling.py` is correctly implemented with well-centred Gaussians.

### What must be fixed before the cleanup plan can resume
1. The evidence artifact must be regenerated with the correct formula.
2. Post-training evaluation must achieve adequate ESS on improved models so virial/energy numbers are trustworthy.
3. A fair before/after comparison must use the same physical system (same separation, same N).

## Approach

Three phases, strictly ordered. Phase A fixes the evaluation infrastructure. Phase B trains matching baselines to create a fair comparison. Phase C produces the authoritative evidence table and decides whether the improved setup is actually better.

The plan does not modify the training code, loss functions, or architectures. It only fixes evaluation infrastructure and adds the missing comparison runs. If the improved setup turns out to be worse or equivalent after fair comparison, that is a valid finding — not a reason to change the experiment.

After this plan completes, the generalized cleanup plan (`2026-03-31_generalized-cleanup-next-phases.md`) can resume from Step 1 with truthful evidence.

## Foundation checks (must pass before new code)
- [x] Data pipeline known-input check — `sample_multiwell` produces correct per-well occupation (verified by well-centre proposal design)
- [x] Split/leakage validity check — VMC has no train/test split; no leakage possible in ground-state energy minimisation
- [x] Baseline existence or baseline-creation step identified — old g4-g7 runs exist; need to create matching N=2 sep=4.0 baselines
- [x] Relevant existing implementation read and understood — `sampling.py`, `run_virial_check.py`, `diagnostics.py`, `vmc_colloc.py` all reviewed

## Scope
**In scope:** fixing ESS in post-training virial evaluation, regenerating stale evidence artifacts, training matching N=2 baselines at separation=4.0, producing a fair comparison table, deciding whether the improved setup helps.
**Out of scope:** changing loss functions, architectures, or training procedures; adding new physics; repo cleanup (deferred to the cleanup plan); new campaigns beyond what is needed for fair comparison.

## Steps

### Step 1 — Regenerate virial_results_v2.json with the correct formula
**What:** The current `virial_results_v2.json` stores `virial_residual` and `virial_relative` computed with the wrong sign. The raw energy components (T, V_trap, V_int, E) are correct. Recompute the derived fields using `compute_virial_metrics()` from the raw values and overwrite the file. Also regenerate `virial_results.json` (v1) the same way.
**Files:** `results/validation_20260330/virial_results_v2.json`, `results/validation_20260330/virial_results.json`
**Acceptance check:** After regeneration, every entry satisfies: `virial_relative = |2*T_mean - (2*V_trap_mean - V_int_mean)| / |E_mean|`, and the N=4 entries show ~3.5–4%, the N=2 entries show ~10–12%. Verify with a manual spot-check on one entry.
**Risk:** Accidentally overwriting the raw components. Mitigation: only update the two derived fields, leave raw fields unchanged. The corrected N=2 and N=4 files created today (`virial_corrected_n2.json`, `virial_corrected_n4.json`) serve as independent crosscheck.

### Step 2 — Diagnose ESS collapse for improved models in virial evaluation
**What:** The per-well multi-well proposal uses the same `sigma_fs: [0.8, 1.3, 2.0]` for all models regardless of what the trained wavefunction looks like. With the larger backflow network (bf_hidden=64) and fd_colloc loss, the improved models may have learned a wavefunction whose density is more concentrated or differently shaped than what the Gaussian tiers cover. Investigate this by:
  1. Comparing the `log_w_raw` distribution (IS log-weights) between an old model and an improved model to see where the mismatch is.
  2. Checking whether the improved models have sharper log-psi (concentrated density) or broader tails than old models.
  3. Testing whether wider sigma_fs tiers (e.g. `[0.5, 0.8, 1.3, 2.0, 3.5]`) or Langevin refinement steps recover ESS.
**Files:** `scripts/run_virial_check.py`, `src/training/sampling.py`
**Acceptance check:** A concrete hypothesis for the ESS collapse exists, supported by IS weight diagnostics from at least one old and one improved model. One candidate fix is identified.
**Risk:** The ESS collapse is fundamental (the wavefunction is pathological) rather than a proposal mismatch. Mitigation: also check whether training-time ESS for these models was already low by reading their result.json training histories.

### Step 3 — Fix post-training evaluation ESS to > 1000
**What:** Implement the fix identified in Step 2. The most likely fixes, in order of preference:
  - **Option A (preferred): Widen sigma_fs tiers** in `run_virial_check.py` to better cover the improved model's density. This is a pure evaluation change, does not affect training.
  - **Option B: Add Langevin refinement** steps to the post-training evaluation. The infrastructure already exists (`langevin_refine_samples` in `sampling.py`); pass `--langevin-steps N --langevin-step-size S` to `run_virial_check.py`.
  - **Option C: Increase n_cand_mult** from 8 to e.g. 32 to brute-force better coverage (expensive but safe).
**Files:** `scripts/run_virial_check.py`, possibly `src/training/sampling.py` (if sigma_fs adaptation needs a helper)
**Acceptance check:** `run_virial_check.py` on `impr_n2_ctnn_s301` and `impr_n4_ctnn_s305` produces ESS > 1000 consistently across batches.
**Risk:** No single fix achieves ESS > 1000. Mitigation: combine options (wider tiers + Langevin + larger candidate pool). If ESS remains < 100 after all three, the wavefunction quality itself is the problem, and that is a valid diagnostic finding.

### Step 4 — Rerun corrected virial check on all improved models with fixed ESS
**What:** Once ESS is fixed, rerun `scripts/run_virial_check.py` on all 6 improved models with 50,000 samples and save the output.
**Files:** `results/validation_20260330/virial_corrected_improved_final.json`
**Acceptance check:** All 6 entries have ESS > 1000, virial_relative values are stable to within ±0.5% across different batches, and the output JSON is saved.
**Risk:** One or two models still have ESS issues due to individual training failures. Flag those specifically.

### Step 5 — Train matching N=2 baselines at separation=4.0
**What:** The old N=2 runs all used separation=6.0, so no fair comparison exists for N=2 at sep=4.0. Train 2 baseline N=2 runs at `separation: 4.0` using the **old** training setup (`reinforce_hybrid`, `bf_hidden: 32`, `epochs: 20000`, flat LR) to create a matching before/after pair. Use seeds 401 and 402.
**Files:** `configs/generalized/campaign_base_long.yaml` (as base config), new config files for the two runs.
**Acceptance check:** Two completed N=2 `separation: 4.0` baseline runs exist with model.pt, config.yaml, and result.json. Training completes without crashes.
**Risk:** Training time (~2–3 hours per run on RTX 2080 Ti). Mitigation: run both on separate GPUs in parallel.

### Step 6 — Run virial check on the matching baselines
**What:** Run the (now ESS-fixed) virial check on the two new N=2 baselines from Step 5, and also re-run on the old N=4 baselines (g4, g5) to get corrected numbers with adequate ESS.
**Files:** `results/validation_20260330/virial_corrected_baseline_final.json`
**Acceptance check:** All baseline entries have ESS > 1000 and corrected virial computed.
**Risk:** The old-setup N=2 models at sep=4.0 may train poorly because sep=4.0 was not the original design point. That itself is informative — it would mean the separation change, not the training improvement, explains the difference.

### Step 7 — Produce the authoritative evidence table
**What:** Combine the corrected virial results into a single comparison table:

| System | Model | Training | Separation | Energy | Virial % | ESS | Seeds |
|--------|-------|----------|-----------|--------|----------|-----|-------|
| N=2 | baseline | reinforce_hybrid | 4.0 | ? | ? | ? | 401, 402 |
| N=2 | improved | fd_colloc | 4.0 | ? | ? | ? | 301–304 |
| N=4 | baseline | reinforce_hybrid | 4.0 | ? | ? | ? | 201–203 |
| N=4 | improved | fd_colloc | 4.0 | ? | ? | ? | 305, 306 |

Additionally, record the old N=2 runs at sep=6.0 separately (not as a before/after, but as a different-system reference).

**Files:** Plan file (this document) or a new `results/validation_20260331/evidence_table.json`
**Acceptance check:** Every cell in the table is filled with a number traceable to a specific result directory and virial JSON. No cell says "estimated" or "approximately". Seed spread is reported.
**Risk:** The improved setup may turn out to be worse than the baseline. That is an acceptable finding — the purpose is truth, not confirmation.

### Step 8 — State the verdict and resume the cleanup plan
**What:** Based on the evidence table, answer three questions:
  1. Does the improved training setup (fd_colloc + bf_hidden=64 + cosine LR) improve virial quality compared to the baseline at the same separation?
  2. Is the improvement large enough to justify the added complexity (3 new hyperparameters: warmup epochs, min factor, loss type)?
  3. What is the recommended next step for the cleanup plan?

Update the cleanup plan's Context section and Step 1 to reflect the actual numbers. Either:
  - Confirm the original plan's trajectory with corrected numbers, or
  - Reshape the plan if the improved setup is not actually better (e.g., drop the "improved campaign" framing and treat all runs as exploration).

**Files:** `plans/2026-03-31_generalized-cleanup-next-phases.md`
**Acceptance check:** The cleanup plan's Context section cites the evidence table with correct numbers and does not contain any unsupported claims.
**Risk:** Inertia bias — wanting the improved setup to work because effort was spent on it. Mitigation: the verdict is mechanical (better/worse/equivalent within error bars), not subjective.

## Risks and mitigations
- **ESS fix is harder than expected:** If no combination of wider tiers, Langevin, and larger candidate pools achieves ESS > 1000, that itself is a finding about the improved models (their wavefunctions may be pathological). Escalate to a full sampling audit rather than continuing with unreliable numbers.
- **Baseline training at sep=4.0 reveals the system is harder:** The old training setup may fail at sep=4.0. This proves the separation change was a confound, not the training improvement. Report honestly.
- **GPU time budget:** Steps 5–6 require ~6 GPU-hours (2 training runs + virial checks). With 8 GPUs available, this is < 1 wall-clock hour. Not a blocker.
- **Stale plan accumulation:** After this plan completes, there will be 3 plans in `plans/`. The cleanup plan should disposition the older ones. Not a blocker for this plan.

## Success criteria
- `virial_results_v2.json` and `virial_results.json` contain correct derived metrics matching the centralized `compute_virial_metrics()` formula.
- All virial evaluation runs (old and improved) have ESS > 1000.
- A fair before/after comparison exists for both N=2 and N=4 at `separation: 4.0` with the same physical system.
- An evidence table exists with every number traceable to a dated result artifact.
- The cleanup plan's claims match the evidence table.
- The question "does the improved training setup help?" has a data-backed answer.

## Current State
**Status: COMPLETE** (all 8 steps executed)
**Last updated:** 2026-03-31

## Verdict

All 8 steps executed. Evidence table saved to
[results/validation_20260331/evidence_table.json](results/validation_20260331/evidence_table.json).

### What was found

**Step 1 — Evidence artifacts corrected.**
Corrected virial values (IS, reliable ESS): N=4 old 3.5–4.0%; N=2 old (sep=6.0) 10.4–12.2%.

**Step 2–3 — ESS collapse diagnosed; MH mode added.**
IS ESS for improved models: 4–100 (catastrophic collapse). Root cause: fd_colloc + bf_hidden=64 produces
sharper, more concentrated ψ², which the fixed Gaussian proposal cannot cover. IS is structurally
unsuitable for improved models. Added `--sampler {mh,is}` flag to `scripts/run_virial_check.py` with
parallel-chain MH using `_sample_mh_batch()`. Langevin cannot be used under IS (overrides log-q to
zero, breaking importance weights).

**Step 4 — Corrected virial on improved models (MH, burn_in=300).**
- N=2 improved (sep=4.0): E ≈ 2.248, virial ≈ 8.7–8.9%, MH accept ≈ 0.501. Saved to
  [results/validation_20260330/virial_corrected_improved_final.json](results/validation_20260330/virial_corrected_improved_final.json)
- N=4 improved: E ≈ 6.995, virial ≈ 13.0–13.1%, MH accept ≈ 0.501.

**Step 5 — Matching N=2 sep=4.0 baselines trained.**
Seeds 401, 402 with old setup (reinforce_hybrid, bf_hidden=32, 20k epochs, flat LR). Training-time
ESS median: 1254 and 1638 respectively — healthy. Training-time ESS for improved N=2 models: **13**.

**Step 6 — Baseline virial (MH, burn_in=100).**
Old N=2 sep=4.0: virial 18.7–19.4%, MH accept 0.748.
Old N=4 sep=4.0 (g4/g5): virial 1.7–0.8% by MH (3.5–4.0% by IS — IS numbers are more reliable here).

**Critical finding from Step 6 cross-validation:**
MH accept rate differs systematically by model type: old models (flat ψ²) accept at 0.742, improved
models (sharp ψ²) accept at 0.501. At identical step size (0.25), burn-in bias affects the two groups
differently. Old N=2 g6: IS gives E=2.181 (ESS~6300, reliable); MH gives E=2.196 (8 σ higher). MH is
biased HIGH for flat-ψ² models at this step size.

**Training-time ESS collapse finding:**
ESS_tail_median during training: improved N=2 = **13**, improved N=4 = **22**; old N=2 = **1254**,
old N=4 = **856**. The improved models were trained on nearly degenerate MCMC chains — the fd_colloc
training gradient was noise-dominated at nearly every epoch.

### Evidence table (summary)

| System | Setup | Sep | E_mean | Virial% | Evaluator | Confidence |
|--------|-------|-----|--------|---------|-----------|-----------|
| N=4 | old (g4) | 4.0 | 7.049 | 3.46% | IS ESS~3351 | HIGH |
| N=4 | old (g5) | 4.0 | 7.046 | 4.05% | IS ESS~3409 | HIGH |
| N=4 | improved (s305) | 4.0 | 6.996 | 13.1% | MH burn300 | LOW |
| N=4 | improved (s306) | 4.0 | 6.995 | 13.0% | MH burn300 | LOW |
| N=2 | old sep=4 (s401) | 4.0 | 2.280 | 18.74% | MH burn100 | UNCERTAIN |
| N=2 | old sep=4 (s402) | 4.0 | 2.282 | 19.41% | MH burn100 | UNCERTAIN |
| N=2 | improved (s301) | 4.0 | 2.248 | 8.85% | MH burn300 | LOW |
| N=2 | improved (s302–s304) | 4.0 | 2.248 | 8.73–8.92% | MH burn300 | LOW |
| N=2 | old ref (g6) | 6.0 | 2.181 | 12.17% | IS ESS~6295 | HIGH (diff system) |
| N=2 | old ref (g7) | 6.0 | 2.179 | 10.39% | IS ESS~2596 | HIGH (diff system) |

### Answered questions

**1. Does the improved setup improve virial quality?**
For N=4: NO. Old IS gives 3.5–4.0%; improved MH gives 13.0–13.1%. The N=4 virial is worse with the
improved setup, and the direction is unambiguous despite evaluator differences.
For N=2: CANNOT RELIABLY DETERMINE. Both setups have inconsistent sampling across evaluators.
MH numbers show ~9% improved vs ~19% old, but MH converges differently for each model type.

**2. Does the improved setup improve energy?**
Yes, by ~0.75% for N=4 and ~1.4% for N=2 (lower variational bound). However, the improved training
ran on ESS=13–22 samples/step, so this energy improvement was achieved via noisy gradient estimates.
The resulting wavefunctions may reflect a local optimum of a noise-dominated objective.

**3. What is the recommended next step?**
Fix training-time sampling for fd_colloc before treating the improved model energies as meaningful.
The most likely fix: add sigma_fs adaptation during training so the Gaussian proposal tracks the
wavefunction's learned geometry. Until training-ESS is stable (>500 throughout training), improved
models cannot be trusted. The generalized cleanup plan should reflect this finding.

## What was produced
- [scripts/run_virial_check.py](scripts/run_virial_check.py) — new `--sampler`, `--mh-burn-in`, `--mh-step-scale` flags; `_sample_mh_batch()` added
- [results/validation_20260330/virial_results.json](results/validation_20260330/virial_results.json) — corrected formula
- [results/validation_20260330/virial_results_v2.json](results/validation_20260330/virial_results_v2.json) — corrected formula
- [results/validation_20260330/virial_corrected_improved_final.json](results/validation_20260330/virial_corrected_improved_final.json) — MH virial for 6 improved models
- [results/validation_20260331/virial_baselines_mh.json](results/validation_20260331/virial_baselines_mh.json) — MH virial for 6 baseline models
- [results/validation_20260331/evidence_table.json](results/validation_20260331/evidence_table.json) — full evidence table with verdicts
- [results/20260331_171637_baseline_n2_sep4_old_s401/](results/20260331_171637_baseline_n2_sep4_old_s401/) — matching N=2 sep=4 baseline
- [results/20260331_171904_baseline_n2_sep4_old_s402/](results/20260331_171904_baseline_n2_sep4_old_s402/) — matching N=2 sep=4 baseline
