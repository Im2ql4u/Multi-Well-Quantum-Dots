# Plan: Post-Recovery Validation and Production

Date: 2026-04-02
Status: confirmed

## Project objective
Produce publication-quality VMC ground-state energies and virial-validated wavefunctions for N=2 and N=4 double quantum dots with Coulomb interaction, using the LCAO Slater-Jastrow-backflow ansatz.

## Context — What happened
A `git clean -fd` deleted ~80 untracked working files. All critical source modules were recovered from `.pyc` bytecode caches using `pycdc` decompiler + manual `dis`/`marshal` analysis. 16 files reconstructed, test suite restored to 22 passing / 6 skipped.

### What is recovered and working
- **Core pipeline:** `run_ground_state.py --config <yaml>` → builds system → trains model → saves results. Verified end-to-end (smoke test).
- **Training:** `training/vmc_colloc.py` (tracked), `training/sampling.py` (recovered: IS + MH sampler), `training/collocation.py` (recovered: fd_colloc, weak_form, reinforce_hybrid losses).
- **Wavefunction:** `wavefunction.py` (recovered: GroundStateWF with Slater-Jastrow-like ansatz — Gaussian envelope + one-body NN + pair NN + Jastrow).
- **Architectures:** `architectures/backflow.py` (OrbitalBackflowNet), `architectures/jastrow.py` (CTNNJastrow), `architectures/unified_ctnn.py` (UnifiedCTNN). All pass forward+backward tests.
- **Observables:** `observables/diagnostics.py`, `observables/validation.py` (virial metrics).
- **Potential:** `potential.py` (compute_potential + legacy wrapper). Matches legacy `imaginary_time_pinn.compute_potential` bitwise.
- **Tests:** 22 passing, 6 skipped (for refactored-out functions: sample_mixture, sample_multiwell, compute_grad_logpsi, eval_multiwell_logq).

### What is lost and NOT recoverable
- `scripts/run_noninteracting_validation.py` — convenience wrapper for Phase 2 validation runs. **Replaced by:** direct `run_ground_state.py --config` calls with per-case YAML configs.
- `scripts/run_generalized_multigpu_campaign.py` — multi-GPU job orchestrator. **Replaced by:** bash loops + tmux or a new thin wrapper.
- `configs/generalized/campaign_mcmc.yaml` — MH base config. **Replaced by:** new configs based on `campaign_base_improved.yaml` patterns.
- All P1-P8 plan files and unified roadmap.
- `src/imag_time_thesis_adapter.py` (non-critical path).

### Known limitations of recovered code
1. **GroundStateWF ignores `arch_type` parameter.** The bytecode-recovered wavefunction uses a fixed architecture (Gaussian envelope + one-body MLP + pair MLP + Jastrow). The original likely dispatched between PINN/CTNN/unified based on `arch_type`. For now, all runs use this fixed ansatz. Architecture selection must be re-implemented if CTNN vs PINN comparisons are needed.
2. **Architecture modules (backflow, jastrow, unified_ctnn) are reconstructions.** Core message-passing logic was inferred from bytecode metadata. Interface-compatible but internal computation may differ from original. Tests pass but subtle numerical differences are possible.
3. **6 sampling/gradient functions were refactored out.** The codebase evolved past what `test_training.py` originally tested.

## Objective
Starting from the recovered codebase: (1) commit the recovery as a defensive checkpoint, (2) validate the pipeline on non-interacting systems with known answers, (3) if validated, run controlled Coulomb comparisons, (4) produce publication results.

Success condition: one configuration achieves virial < 5% for N=4 double-dot with Coulomb, across 2+ seeds.

## Foundation checks
- [x] Data pipeline known-input check — `importance_resample` and `mcmc_resample` tested with known Gaussian targets
- [x] Split/leakage validity check — VMC has no train/test split
- [x] Baseline existence — old N=4 models exist with virial 3.5–4.0% (per JOURNAL.md)
- [x] Test suite — 22 passing, 6 skipped
- [x] End-to-end smoke test — `run_ground_state.py` completes on CPU
- [ ] GPU verification — needs confirm before Phase 2

## Scope
**In scope:** git commit, GPU verification, non-interacting validation, IS-vs-MH comparison with Coulomb, production runs, evidence synthesis.
**Out of scope:** re-implementing architecture dispatch (arch_type), new architectures, quench/magnetic physics, FD ground-truth convergence, README changes.

---

## Phase 1 — Commit Recovery and Verify Infrastructure (~15 min)

### Step 1.1 — Commit all recovered code
**What:** Stage all recovered/fixed files. Commit as defensive checkpoint.
**Files to stage:**
```
src/architectures/__init__.py
src/architectures/backflow.py
src/architectures/jastrow.py
src/architectures/unified_ctnn.py
src/training/__init__.py
src/training/sampling.py
src/training/collocation.py
src/observables/diagnostics.py
src/observables/validation.py
src/potential.py
src/wavefunction.py
src/run_ground_state.py
tests/conftest.py
tests/test_architectures.py
tests/test_potential.py
tests/test_training.py
tests/test_validation.py
tests/test_wavefunction.py
```
**Commit message:** `fix(recovery): reconstruct 16 modules from .pyc bytecode after git clean`
**Acceptance:** `git log --oneline -1` shows the commit, `pytest tests/ -v` still 22 passed.

### Step 1.2 — Verify GPU and environment
**What:** Confirm CUDA available, check which GPUs have headroom, confirm tmux.
**Acceptance:** `PYTHONPATH=src .venv/bin/python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"` → 8 GPUs. `nvidia-smi` shows available memory.

---

## Phase 2 — Non-Interacting Validation on GPU (~2 hours wall time)
**Depends on:** Phase 1 complete
**Goal:** Confirm the ansatz + samplers produce correct energies for non-interacting systems where exact answers are known.

### Step 2.1 — Create validation YAML configs
**What:** Create 4 YAML config files for non-interacting validation:

| Case | System | N | Exact E | Config file |
|------|--------|---|---------|-------------|
| 2.1a | single_dot, ω=1 | 2 | 2.0 | `configs/validation/ni_n2_single_is.yaml` |
| 2.1b | double_dot, sep=4, ω=1 | 2 | 2.0 | `configs/validation/ni_n2_double_is.yaml` |
| 2.1c | single_dot, ω=1 | 4 | 6.0 | `configs/validation/ni_n4_single_is.yaml` |
| 2.1d | double_dot, sep=4, ω=1 | 4 | 4.0 | `configs/validation/ni_n4_double_is.yaml` |

All configs: `coulomb: false`, `sampler: is`, `loss_type: weak_form`, `epochs: 10000`, `n_coll: 256`.

**Acceptance:** All 4 YAML files parse without error.

### Step 2.2 — Run IS baselines on GPU
**What:** Run all 4 cases on separate GPUs via tmux.
```bash
for cfg in configs/validation/ni_*.yaml; do
  name=$(basename "$cfg" .yaml)
  tmux new-session -d -s "$name" \
    "cd $(pwd) && PYTHONPATH=src .venv/bin/python src/run_ground_state.py --config $cfg 2>&1 | tee results/validation_${name}.log"
done
```
**Acceptance gate:** All energies within 1% of exact. All runs complete without NaN.

### Step 2.3 — Run MH sampler on double-well cases
**What:** Create 2 more YAML configs identical to 2.1b/2.1d but with `sampler: mh`, `mh_steps: 10`, `mh_step_scale: 0.25`. Run on GPU.
**Acceptance gate:** MH energies agree with IS energies within noise. Accept rate in 0.3–0.7.

### Step 2.4 — MH + fd_colloc diagnostic
**What:** Run case 2.1b with `sampler: mh`, `loss_type: fd_colloc`. Does MH prevent the ESS collapse that killed fd_colloc under IS?
**Acceptance gate:** Training completes without NaN. Energy converges within 2% of exact (2.0). If fails, record finding and use reinforce_hybrid/weak_form for Phase 3.

### Step 2.5 — Phase 2 evidence synthesis
**What:** Compile comparison table. Decide which sampler × loss combinations proceed to Phase 3.
**Files:** `results/validation_phase2_summary.json`

---

## Phase 3 — Controlled Comparisons With Coulomb (~6 hours wall time)
**Depends on:** Phase 2 complete, at least MH validated on non-interacting
**Goal:** Find best sampler × loss combination for Coulomb systems. Target: virial < 5% for N=4.

### Step 3.1 — Create Coulomb comparison configs
**What:** For each validated Phase 2 combination, create YAML configs for N=2 and N=4 double-dot (sep=4.0, coulomb=true), 2 seeds each, 20k epochs.
**Files:** `configs/phase3/` directory with configs.

### Step 3.2 — Launch campaigns
**What:** Run all Phase 3 configs across available GPUs in tmux sessions. Monitor first 100 epochs for stability.
**Acceptance:** All runs complete. No NaN crashes.

### Step 3.3 — Virial evaluation
**What:** Run `scripts/run_virial_check.py --sampler mh` on all trained models.
**Acceptance gate:** At least one N=4 config has virial < 5%.

### Step 3.4 — Evidence synthesis and hypothesis closure
**What:** Build comparison table. Address:
- H1: Does MH fix ESS collapse for fd_colloc?
- H2: Is fd_colloc better than reinforce_hybrid/weak_form when properly sampled?
- H3: Any config achieves publishable virial for N=4?

---

## Phase 4 — Production Runs (~8 hours wall time)
**Depends on:** Phase 3 complete with winner identified
**Goal:** Publication-quality results. 3 seeds × {N=2, N=4}, 30k epochs.

### Step 4.1 — Select winning configuration
**What:** From Phase 3 evidence, select config with lowest N=4 virial. If tied, prefer simpler config.

### Step 4.2 — Production campaign
**What:** 6 models (3 seeds × 2 systems), 30k epochs with cosine LR if Phase 3 showed it helps.

### Step 4.3 — Comprehensive virial evaluation
**What:** MH virial check with 50k samples on all 6 models.
**Acceptance:** All N=4 virial < 5%, all N=2 virial < 5%, cross-seed spread < 2 pp.

### Step 4.4 — Final evidence document and git tag
**What:** Produce evidence JSON, close hypotheses, commit and tag `result/2026-04-XX-mcmc-validated`.

---

## Risks and mitigations
- **Recovered GroundStateWF ignores arch_type:** All runs use the same fixed ansatz. If the original code used different architectures for different configs, results may differ. Mitigation: if Phase 2 fails (wrong energies), the wavefunction architecture is suspect.
- **Bytecode reconstruction errors:** Architecture modules are approximations. If forward pass produces wrong answers, check the recovered code against known physics. Mitigation: non-interacting validation (Phase 2) catches this.
- **fd_colloc may fail even with MH:** Drop it, proceed with weak_form/reinforce_hybrid only.
- **No config achieves virial < 5%:** This is a real finding. The simplified GroundStateWF may lack expressiveness from the original architecture dispatch. Record honestly.
- **GPU contention:** Monitor `nvidia-smi`. Fall back to sequential runs.

## Current State
**Active phase:** 1 — Commit Recovery
**Active step:** 1.1 — Stage and commit recovered files
**Last evidence:** 22 tests passing, 6 skipped. `run_ground_state.py` smoke test completed.
**Blockers:** None.
