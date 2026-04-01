# Plan: MCMC Validation Through Production

Date: 2026-04-01
Status: confirmed

## Project objective
Produce publication-quality VMC ground-state energies and virial-validated wavefunctions for N=2 and N=4 double quantum dots with Coulomb interaction, using the LCAO Slater-Jastrow-backflow ansatz.

## Objective
Execute the remaining MCMC training validation (Phases 1 full, 2.3–2.4, 3, 4 of the prior MCMC plan) on GPU, clean the repo state so code changes are committable, and produce an evidence table that either confirms a working pipeline or identifies the next structural blocker. Success condition: one configuration achieves virial < 5% for N=4 double-dot with Coulomb, with ESS/accept_rate healthy throughout training, across 2+ seeds.

## Context
The review (2026-04-01) identified that the critical-path bottleneck is unfilled: the MH sampler is implemented (Phase 2.1–2.2 of `plans/2026-03-31_mcmc-training-and-validation.md`), but the actual experiments that answer the scientific questions (Phases 2.3–4) have not been run. The only model with acceptable virial numbers (3.5–4% for N=4) is the old reinforce_hybrid + bf_hidden=32 + IS setup. All "improved" models are garbage (trained on ESS 13–22). The working tree has 236 result-artifact deletions burying 11 meaningful code changes. SESSION_LOG.md is stale.

Prior negative findings (from JOURNAL.md):
- IS + fd_colloc + bf_hidden=64 → ESS collapse → garbage models. Do not repeat.
- N=2 cross-separation comparison (sep=6 vs sep=4) is invalid. All comparisons must use same system.
- FD dipole-gap GT is not converged. Do not use as authoritative reference yet.
- Langevin-IS path is scientifically invalid (now blocked with ValueError).

## Approach
Four phases, strictly ordered. Phase 1 unblocks everything else (git hygiene, GPU check). Phase 2 validates the MH sampler on non-interacting problems where exact answers are known—if it fails here, nothing downstream matters. Phase 3 runs controlled comparisons with Coulomb to answer the four open hypotheses (H2–H5 from the MCMC plan). Phase 4 produces production results if Phase 3 succeeds. All runs use tmux for persistence; all results go under dated directories. 8× RTX 2080 Ti available; GPUs 0,1,5,6 have ~7 GB free, suitable for N=2 runs; GPUs with ~4 GB used are less loaded.

## Foundation checks (must pass before new code)
- [x] Data pipeline known-input check — `sample_multiwell` verified for per-well occupation
- [x] Split/leakage validity check — VMC has no train/test split
- [x] Baseline existence — old g4/g5 N=4 models exist with virial 3.5–4.0%
- [x] Relevant existing implementation read and understood — sampling.py, vmc_colloc.py, run_noninteracting_validation.py, campaign_mcmc.yaml all reviewed

## Scope
**In scope:** git commit separation, SESSION_LOG update, non-interacting MH validation on GPU (full epochs), controlled IS-vs-MH and fd_colloc-vs-reinforce comparisons with Coulomb, production runs of the winning configuration, evidence synthesis.
**Out of scope:** new architectures, new loss functions, quench/magnetic physics, FD ground-truth convergence, observables (density matrices/entanglement), README changes, cleanup-plan Steps 5–8.

---

## Phase 1 — Housekeeping (single session, ~30 min)
**Goal:** Clean working tree so code changes are isolatable, update SESSION_LOG, verify GPU availability.
**Estimated scope:** 2 files edited, 2 git commits, no new code.

### Step 1.1 — Commit code changes separately from result deletions
**What:** Stage and commit only the 11 code/config/doc files that changed (sampling.py patch, test additions, vmc_colloc.py MH integration, diagnostics, etc.). Leave result-artifact deletions as a separate commit or defer them.
**Files:** `src/training/sampling.py`, `src/training/vmc_colloc.py`, `tests/test_training.py`, `src/observables/__init__.py`, `src/observables/validation.py`, `tests/test_validation.py`, `src/config.py`, `src/imaginary_time_pinn.py`, `src/imaginary_time_vmc.py`, `scripts/generate_time_evolution_report.py`, `plans/2026-03-31_generalized-observables-phase.md`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && git --no-pager log --oneline -1` → shows new commit message starting with `fix(sampling):` or similar
**Risk:** Accidentally staging result files. Mitigation: use `git add` with explicit paths only.

### Step 1.2 — Update SESSION_LOG.md to reflect actual state
**What:** Replace stale quench/FD guidance with: active plan is this plan, active phase is Phase 2, bottleneck is running MCMC experiments, foundation status is MH implemented but untested on full runs.
**Files:** `SESSION_LOG.md`
**Acceptance check:** `head -20 /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots/SESSION_LOG.md` → contains reference to `plans/2026-04-01_mcmc-validation-to-production.md` and "Phase 2"
**Risk:** None.

### Step 1.3 — Verify GPU and environment
**What:** Confirm `.venv/bin/python` can import torch with CUDA, confirm at least 4 GPUs have < 8 GB used, confirm tmux is available.
**Files:** none
**Acceptance check:** `.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(f'GPUs: {torch.cuda.device_count()}')"` → prints `GPUs: 8`; `which tmux` → prints path
**Risk:** Another user may have loaded GPUs. Check `nvidia-smi` before launching.

---

## Phase 2 — Non-Interacting Validation on GPU (single session, ~2 hours wall time)
**Depends on:** Phase 1 complete
**Goal:** Confirm the ansatz + MH sampler produces correct energies for non-interacting systems where exact answers are known. This validates the machinery before adding Coulomb complexity.
**Estimated scope:** 0 new files; run existing scripts with full epochs.

### Step 2.1 — Full Phase 1 non-interacting baselines (IS, reinforce_hybrid)
**What:** Run all 4 non-interacting cases (1.1_n2, 1.1_n4, 1.2_n2, 1.2_n4) with the old setup (IS, reinforce_hybrid, bf_hidden=32, 10k epochs) on GPU. This establishes the baseline the MH runs must match.
**Files:** `scripts/run_noninteracting_validation.py`
**Acceptance check:**
```bash
PYTHONPATH=src .venv/bin/python scripts/run_noninteracting_validation.py \
  --all-phase1 --epochs 10000 --device cuda:1 \
  --output-dir results/validation_20260401_p1_is && \
python -c "
import json, pathlib
for d in sorted(pathlib.Path('results/validation_20260401_p1_is').iterdir()):
    r = json.loads((d/'result.json').read_text())
    E = r['energy_history'][-1]
    print(f\"{d.name}: E={E:.4f}\")
"
```
→ Expected: 1.1_n2 E ≈ 2.00 (±0.05), 1.1_n4 E ≈ 6.00 (±0.15), 1.2_n2 E ≈ 2.00 (±0.05), 1.2_n4 E ≈ 6.00 (±0.15). All complete without error.
**Risk:** Single-well N=4 shell-filling may differ from naive 6ω if the basis/LCAO setup is incorrect for 4 electrons in one well. If energy is wrong, this is a foundation bug — stop and debug before continuing.

### Step 2.2 — Phase 1 with fd_colloc + IS (reproduce ESS failure mode)
**What:** Run 1.2_n2 with fd_colloc + bf_hidden=64 + IS, 10k epochs. Purpose: confirm whether ESS collapses even for non-interacting case (diagnostic from MCMC plan Step 1.3).
**Files:** `scripts/run_noninteracting_validation.py`
**Acceptance check:**
```bash
PYTHONPATH=src .venv/bin/python scripts/run_noninteracting_validation.py \
  --phase1-fd --epochs 10000 --device cuda:5 \
  --output-dir results/validation_20260401_p1_fd && \
python -c "
import json, pathlib
for d in sorted(pathlib.Path('results/validation_20260401_p1_fd').iterdir()):
    r = json.loads((d/'result.json').read_text())
    E = r['energy_history'][-1]
    ess = r.get('ess_history',[[0]])[-1]
    print(f\"{d.name}: E={E:.4f}, ESS_last={ess}\")
"
```
→ Expected: Either ESS stays healthy (≥500) proving ESS collapse is Coulomb-specific, or ESS collapses proving fd_colloc+IS is fundamentally broken. Both are valid findings.
**Risk:** fd_colloc may NaN. If so, that's a finding — record and continue.

### Step 2.3 — MH sampler on non-interacting double-well (head-to-head)
**What:** Run 1.2_n2 and 1.2_n4 (double-well, coulomb=false) with MH sampler, reinforce_hybrid, bf_hidden=32, 10k epochs. Compare to Step 2.1 IS results on the same system.
**Files:** `scripts/run_noninteracting_validation.py`
**Acceptance check:**
```bash
PYTHONPATH=src .venv/bin/python scripts/run_noninteracting_validation.py \
  --phase2-double-mh --epochs 10000 --device cuda:6 \
  --output-dir results/validation_20260401_p2_mh && \
python -c "
import json, pathlib
for d in sorted(pathlib.Path('results/validation_20260401_p2_mh').iterdir()):
    r = json.loads((d/'result.json').read_text())
    E = r['energy_history'][-1]
    print(f\"{d.name}: E={E:.4f}\")
"
```
→ **Acceptance gate:** Energy within 1% of exact (2.0 for N=2, 6.0 for N=4). MH and IS results agree within noise. Accept rate in 0.3–0.7 range.
**Risk:** MH chains get stuck in one well. Check per-well occupation in output. If stuck, the `sample_multiwell` initialization is failing — debug before continuing.

### Step 2.4 — MH + fd_colloc on non-interacting double-well
**What:** Run 1.2_n2 with MH + fd_colloc + bf_hidden=64, 10k epochs. The critical test: does MH fix the ESS collapse that killed fd_colloc under IS?
**Files:** `scripts/run_noninteracting_validation.py` (may need a new `--phase2-fd-mh` flag or direct `run_ground_state.py` call with custom config)
**Acceptance check:**
```bash
# If run_noninteracting_validation.py doesn't support this combo, use run_ground_state.py directly:
cat > /tmp/p2_fd_mh.yaml << 'EOF'
run_name: p2_fd_mh_n2_doublewell
allow_missing_dmc: true
system:
  type: double_dot
  n_left: 1
  n_right: 1
  separation: 4.0
  omega: 1.0
  dim: 2
  coulomb: false
architecture:
  arch_type: ctnn
  bf_hidden: 64
  bf_layers: 2
  pinn_hidden: 32
  pinn_layers: 2
  use_backflow: true
training:
  epochs: 10000
  lr: 0.001
  n_coll: 256
  n_cand_mult: 8
  loss_type: fd_colloc
  fd_h: 0.01
  sampler: mh
  mh_steps: 10
  mh_step_scale: 0.25
  sigma_fs: [0.8, 1.3, 2.0]
  grad_clip: 1.0
  seed: 1
  device: cuda:0
  dtype: float64
EOF
PYTHONPATH=src .venv/bin/python src/run_ground_state.py --config /tmp/p2_fd_mh.yaml
```
→ **Acceptance gate:** E converges to within 1% of 2.0. Accept rate 0.3–0.7. No training instability (no NaN, loss decreasing).
→ **If fails:** fd_colloc has a problem beyond sampling. The loss function itself needs investigation. Record finding and use reinforce_hybrid for Phase 3.
**Risk:** fd_colloc + MH may train slower. Allow full 10k epochs before judging.

### Step 2.5 — Phase 2 evidence synthesis
**What:** Compile a comparison table of all Phase 2 results. Decide: (a) does MH produce correct answers? (b) does fd_colloc work with MH? (c) which combinations proceed to Phase 3?
**Files:** This plan (update Current State), new `results/validation_20260401_phase2_summary.json`
**Acceptance check:**
```bash
python -c "
import json, pathlib
dirs = ['results/validation_20260401_p1_is', 'results/validation_20260401_p1_fd', 'results/validation_20260401_p2_mh']
for root in dirs:
    p = pathlib.Path(root)
    if not p.exists(): continue
    for d in sorted(p.iterdir()):
        rp = d / 'result.json'
        if not rp.exists(): continue
        r = json.loads(rp.read_text())
        E = r['energy_history'][-1]
        print(f'{d.name}: E_final={E:.4f}')
"
```
→ Expected: printed table with all runs, all energies close to exact values for IS and MH.
**Risk:** One combination may fail completely. That is a valid finding — flag it and exclude from Phase 3.

---

## Phase 3 — Controlled Comparisons With Coulomb (single session, ~6 hours wall time)
**Depends on:** Phase 2 complete with at least MH + reinforce_hybrid validated on non-interacting
**Goal:** Run the 2×2 matrix of {reinforce_hybrid, fd_colloc} × {bf_hidden=32, bf_hidden=64} with MH sampler on N=2 and N=4 double-dots with Coulomb. Find the best combination. Answer H2–H5.
**Estimated scope:** 1 new campaign variants YAML, 1 script invocation per N, 0 new code.

### Step 3.1 — Create campaign variants YAML for Phase 3
**What:** Write a variants YAML for `run_generalized_multigpu_campaign.py` that defines 8 jobs: 4 loss×capacity combos × 2 seeds, for N=2 double-dot sep=4.0 with Coulomb. Use `campaign_mcmc.yaml` as the base config template.
**Files:** `configs/generalized/phase3_n2_variants.yaml` (new), `configs/generalized/phase3_n4_variants.yaml` (new)
**Acceptance check:** `python -c "import yaml; d=yaml.safe_load(open('configs/generalized/phase3_n2_variants.yaml')); print(len(d['jobs']), 'jobs')"` → prints `8 jobs`
**Risk:** Config field names may not match what `run_generalized_multigpu_campaign.py` expects. Verify against the campaign runner's override-merge logic.

### Step 3.2 — Launch N=2 Phase 3 campaign on GPU
**What:** Run the N=2 campaign across 8 GPUs (1 job per GPU), 20k epochs, in tmux.
**Files:** `scripts/run_generalized_multigpu_campaign.py`, `configs/generalized/phase3_n2_variants.yaml`
**Acceptance check:**
```bash
tmux new-session -d -s phase3_n2 \
  "cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && \
   PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python \
   scripts/run_generalized_multigpu_campaign.py \
   --variants configs/generalized/phase3_n2_variants.yaml \
   --gpu-indices 0,1,2,3,4,5,6,7 2>&1 | tee results/phase3_n2.log"
```
Then verify launch: `tmux has-session -t phase3_n2 && echo 'running'` → prints `running`.
After completion: all 8 `result.json` files exist in the campaign output directory.
**Risk:** GPU memory contention with other users' processes. Monitor first 2 minutes; if OOM, reduce to 4 GPUs and run in 2 batches.

### Step 3.3 — Launch N=4 Phase 3 campaign on GPU
**What:** Same as 3.2 but for N=4 (4 electrons, 2 per well). Run after N=2 completes (or in parallel if GPUs are free). 20k epochs.
**Files:** `configs/generalized/phase3_n4_variants.yaml`
**Acceptance check:** Same pattern as 3.2 — all 8 `result.json` files exist.
**Risk:** N=4 with Coulomb may be significantly more expensive (more forward passes per MH step). Profile first 100 epochs and extrapolate total time. If > 12 hours, reduce to 10k epochs.

### Step 3.4 — Post-training virial check on all Phase 3 models
**What:** Run `scripts/run_virial_check.py` with `--sampler mh` on all 16 trained models (8 N=2 + 8 N=4).
**Files:** `scripts/run_virial_check.py`
**Acceptance check:**
```bash
PYTHONPATH=src .venv/bin/python scripts/run_virial_check.py \
  --device cuda:0 \
  results/<phase3_n2_campaign_dir>/*/  \
  results/<phase3_n4_campaign_dir>/*/ && \
echo "Virial check complete"
```
→ Expected: JSON output with virial percentages. At least one N=4 configuration has virial < 5%.
**Risk:** MH evaluator acceptance rate may differ from training acceptance rate. Record both.

### Step 3.5 — Phase 3 evidence synthesis and hypothesis closure
**What:** Build the definitive comparison table. Close hypotheses H2–H5:

| Config | N | Loss | bf_hidden | Seed | E_mean | E_var | Virial% | Accept | Epochs |
|--------|---|------|-----------|------|--------|-------|---------|--------|--------|
| (fill from results) | | | | | | | | | |

**H2** (IS causes training failure for fd_colloc): Compare Phase 2 IS-ESS vs Phase 3 MH-accept.
**H3** (fd_colloc better than reinforce when properly sampled): Compare virial across loss types.
**H4** (bf_hidden=64 helps): Compare across capacity at same loss.
**H5** (publishable results): Any config with N=4 virial < 5%?

**Files:** `results/validation_20260401_phase3_evidence.json` (new)
**Acceptance check:**
```bash
python -c "
import json
t = json.load(open('results/validation_20260401_phase3_evidence.json'))
for row in t: print(f\"{row['name']}: virial={row['virial_pct']:.1f}%, E={row['energy']:.4f}\")
"
```
→ Expected: all 16 rows printed with numbers.
**Risk:** No configuration achieves virial < 5% for N=4. If so, this is a real finding. The ansatz or training may need deeper investigation (architecture expert escalation). Do not inflate results.

---

## Phase 4 — Production Runs (single session, ~8 hours wall time)
**Depends on:** Phase 3 complete with at least one configuration meeting virial < 5% for N=4
**Goal:** Generate publication-quality results with the validated pipeline. 3 seeds, 30k epochs, comprehensive evaluation.
**Estimated scope:** 1 new variants YAML per N, 0 new code.

### Step 4.1 — Select winning configuration
**What:** From Phase 3 evidence table, select the configuration with: (a) lowest N=4 virial, (b) N=2 virial also acceptable, (c) consistent across seeds. If multiple configs are within noise, prefer the simpler one (reinforce_hybrid, bf_hidden=32).
**Files:** This plan (update Current State with selection rationale)
**Acceptance check:** A concrete config is named. The selection rationale is stated. Not "we chose the best one" — the numbers that justify the choice are cited.
**Risk:** Close call between configs. Use the conservative choice.

### Step 4.2 — Production campaign (3 seeds × {N=2, N=4})
**What:** Train 6 models (3 seeds each for N=2 and N=4 double-dot, sep=4.0, Coulomb=true) with 30k epochs, cosine LR with warmup if Phase 3 showed it helps, using the winning config.
**Files:** `configs/generalized/production_variants.yaml` (new)
**Acceptance check:** 6 `result.json` + `model.pt` + `config.yaml` files exist in dated output directory. All training completed without NaN.
**Risk:** 30k epochs may take ~3 hours per model. With 6 jobs across 6 GPUs, ~3 hours wall time.

### Step 4.3 — Comprehensive virial evaluation
**What:** Run MH virial check with 50k samples, burn_in=500 on all 6 production models.
**Files:** `scripts/run_virial_check.py`
**Acceptance check:** All N=4 virial < 5%. All N=2 virial < 5%. Cross-seed spread < 2 percentage points.
**Risk:** One seed may be an outlier. Flag it specifically rather than averaging it away.

### Step 4.4 — Compare to FD ground truth (where available)
**What:** Run `scripts/compute_fd_ground_truth.py` for N=2 sep=4.0 Coulomb. Compare VMC energy to FD.
**Files:** `scripts/compute_fd_ground_truth.py`
**Acceptance check:** VMC energy within 2% of FD for N=2. For N=4, note that FD may not be available — record as "pending GT."
**Risk:** FD convergence is a known issue. Use best available grid; flag convergence status.

### Step 4.5 — Final evidence document and git tag
**What:** Produce `results/validation_20260401_production/evidence_table.json` with all production numbers. Close all 5 hypotheses (H1–H5). Update this plan to Status: completed. Commit and tag.
**Files:** Plan file, evidence JSON, git tag `result/2026-04-01-mcmc-validated`
**Acceptance check:** `git --no-pager log --oneline -1` → shows commit. `git tag -l 'result/2026-04-01*'` → shows tag.
**Risk:** Tagging before results are truly final. Only tag after virial gates pass.

---

## Risks and mitigations
- **MH sampler produces wrong answers on non-interacting test:** Stop at Phase 2 — do not proceed to Coulomb. Debug the sampler, not the comparisons.
- **fd_colloc fails even with MH:** Drop fd_colloc from Phase 3; run Phase 3 with reinforce_hybrid only (2 configs instead of 4). This halves Phase 3 time.
- **No configuration achieves virial < 5% for N=4:** This is a real finding about the ansatz, not a failure of the plan. Escalate to architecture expert — the LCAO Slater-Jastrow-backflow may have insufficient variational freedom for the N=4 interacting double-dot. Record honestly in JOURNAL.
- **GPU contention:** Other users may load GPUs. Check `nvidia-smi` before each phase launch. Fall back to sequential runs on fewer GPUs if needed.
- **Wall time exceeds expectations:** The 20k-epoch Phase 3 runs are the most expensive. If per-epoch time is > 1s (profile in first 100 epochs), reduce to 10k epochs — still sufficient for convergence comparison, just not production quality.
- **Stale working tree distracts:** Phase 1.1 separates the code commit from result artifacts. Do not attempt to clean up all 236 result deletions before running experiments.

## Success criteria
- MH sampler validated on non-interacting systems (energy within 1% of exact)
- At least one loss×capacity combination produces N=4 virial < 5% with Coulomb
- Evidence table with all numbers traceable to dated result directories
- All 5 hypotheses from the MCMC plan closed with data, not assumptions
- Git tag marking the validated state

## Current State
**Active phase:** 2 — Non-Interacting Validation on GPU
**Active step:** 2.1 — Full Phase 1 non-interacting baselines (IS, reinforce_hybrid)
**Last evidence:** `/usr/bin/time -f 'SANITY_ELAPSED=%E' env PYTHONPATH=src .venv/bin/python scripts/run_noninteracting_validation.py --all-phase1 --epochs 5 --device cuda:1 --output-dir results/validation_20260401_p1_is_sanity` -> completed in `SANITY_ELAPSED=0:12.44`; 1.1_n2 stayed finite with ESS about 824→852 and energy about 2.16→2.13; 1.1_n4 stayed finite with energy about 6.17→6.30; 1.2_n2 stayed finite with energy about 2.05→2.06; 1.2_n4 stayed finite but sat near energy about 4.26→4.24 with ESS collapsing as low as 2.4 before recovering to about 230.
**Current risk:** The plan's acceptance target for non-interacting N=4 double-well appears to be wrong. The current plan expects about 6.0, but the system definition `double_dot(N_L=2, N_R=2, sep=4.0, coulomb=false)` physically suggests two doubly occupied ground orbitals (2 per well), i.e. total energy about 4.0, not 6.0. If that is true, the plan would falsely classify a correct run as failure.
**Next action:** Resolve the N=4 double-well analytical target and update the Phase 2 acceptance criterion before launching the full 10k-epoch baseline run.
**Blockers:** Phase 2.1 acceptance is ambiguous because the exact target for 1.2_n4 in the confirmed plan is likely incorrect.
