# Plan: Non-MCMC Residual-Collocation Training for Multi-Well Runs

Date: 2026-04-12
Status: confirmed

## Project objective
Produce publication-quality VMC ground-state energies for N=2, N=3, N=4 one-per-well quantum dots with Coulomb interaction; hard gate is <1% vs exact diag.

## Objective
Implement and validate a non-MCMC training pipeline (residual/collocation style from Thesis) adapted to generalized multi-well systems, then launch N=3 and N=4 trainings with this scheme; success means training uses no MCMC in its inner loop and reaches stable finite energies with reproducible configs/results.

## Context
Current training in `src/imaginary_time_pinn.py` uses MCMC for both ground-state warm start and precompute pool generation, while the requested direction is explicitly non-MCMC for training. The Thesis method specifies residual-based stage-I training with i.i.d. stratified collocation batches (center/tails/mixed/shells/dimers), adaptive mixture weighting, and optional stage-II SR refinement. We need a multi-well-compatible implementation and launch runs under this new regime.

Relevant prior negative history from `DECISIONS.md` / `JOURNAL.md`:
- 2026-04-10: cross-run conclusions were invalid under mixed evaluation settings; this plan enforces fixed evaluation checks before claiming gains.
- 2026-04-12: major time was lost chasing a phantom model gap caused by a bad reference; this plan adds foundation parity checks (Hamiltonian/reference/sampler parity) before interpreting outcomes.

`CONSTRAINTS.md`: no relevant constraints file present.

## Approach
Port the Thesis residual training pattern into the generalized multi-well codepath by introducing an explicit non-MCMC collocation sampler and residual objective schedule, while keeping MCMC available only for post-training validation if needed. Implement this in session-sized phases: design and wiring first, then sampler implementation, then objective integration, then controlled experiment launch. Use strict config flags to prevent accidental fallback to MCMC. Keep acceptance checks executable and evidence-first.

## Foundation checks (must pass before new code)
- [ ] Data pipeline known-input check
- [ ] Split/leakage validity check
- [ ] Baseline existence or baseline-creation step identified
- [ ] Relevant existing implementation read and understood

## Scope
**In scope:** non-MCMC collocation sampler for training; residual objective with moving target `E_eff`; multi-well compatibility; launch configs and training runs for N=3/N=4; fixed evaluation commands.
**Out of scope:** rewriting exact diagonalization; architecture redesign (new NN blocks); broad quench-physics reinterpretation; destructive cleanup/history edits.

## Phase 1 - Spec lock and interface design (session-sized)
**Goal:** Freeze the exact non-MCMC training contract and config API before coding.
**Estimated scope:** 3-4 files, low risk

### Step 1.1 - Map Thesis sampling/objective into repo interfaces
**What:** Translate Thesis stage-I residual formulation and stratified sampler into explicit implementation requirements for this repo.
**Files:** `Thesis/method.tex`, `src/imaginary_time_pinn.py`, `src/training/vmc_colloc.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && rg -n "Residual-based pretraining|Configuration-space sampling|train_pinn|mcmc_sample|loss_type" Thesis/method.tex src/imaginary_time_pinn.py src/training/vmc_colloc.py` -> expected: lines showing residual objective and current MCMC touchpoints
**Risk:** Mis-mapping Thesis notation to generalized multi-well code paths.

### Step 1.2 - Define config schema for non-MCMC-only training
**What:** Specify new config keys (example): `training.sampler_mode=non_mcmc`, `training.nonmcmc_components`, `training.eeff_schedule`, `training.allow_mcmc_fallback=false`.
**Files:** `src/config.py`, `configs/one_per_well/*.yaml`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && rg -n "sampler_mode|allow_mcmc_fallback|eeff|nonmcmc" src/config.py configs/one_per_well` -> expected: keys present in schema and at least one config
**Risk:** Config drift across old and new launch scripts.

### Step 1.3 - Add explicit non-MCMC guardrail
**What:** Design hard-fail checks so training aborts if any training path calls MCMC while `sampler_mode=non_mcmc`.
**Files:** `src/imaginary_time_pinn.py`, `src/training/vmc_colloc.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && rg -n "allow_mcmc_fallback|raise ValueError|sampler_mode" src/imaginary_time_pinn.py src/training/vmc_colloc.py` -> expected: explicit guard conditions
**Risk:** Silent fallback to legacy MCMC path.

## Phase 2 - Implement non-MCMC stratified collocation sampler (session-sized)
**Depends on:** Phase 1 complete
**Goal:** Training batches are generated i.i.d. without MCMC, with multi-well-aware geometry.
**Estimated scope:** 2-3 files, medium risk

### Step 2.1 - Create sampler module with stratified components
**What:** Implement sampler components from Thesis (center/tails/mixed/shells/dimers), rotation/permutation augmentation, and capped-simplex mixing.
**Files:** `src/training/nonmcmc_sampler.py` (new), optionally `src/training/__init__.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python -c "from training.nonmcmc_sampler import sample_batch; import torch; x,meta=sample_batch(batch_size=64,n_particles=4,dim=2,omega=1.0,well_centers=[(-6,0),(-2,0),(2,0),(6,0)]); print(tuple(x.shape), sorted(meta.keys())[:3])"` -> expected: `(64, 4, 2)` and metadata keys without error
**Risk:** Dimers/shell logic introduces non-finite points near coalescence.

### Step 2.2 - Add adaptive mixture updates (EG + caps)
**What:** Implement per-epoch component difficulty scoring and exponentiated-gradient updates with cap projection/floor.
**Files:** `src/training/nonmcmc_sampler.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python -c "from training.nonmcmc_sampler import update_mixture_weights; import numpy as np; w=np.array([0.2]*5); d=np.array([1,2,3,4,5.],float); w2=update_mixture_weights(w,d,eta=0.1,cap=0.3,floor=0.02); print(w2.sum(), w2.min(), w2.max())"` -> expected: sum ~1.0, min >= floor, max <= cap
**Risk:** Instability if difficulty estimates are too noisy early on.

### Step 2.3 - Wire sampler into training paths with strict mode
**What:** Replace MCMC batch generation in target training loops with non-MCMC sampler when configured.
**Files:** `src/imaginary_time_pinn.py`, `src/training/vmc_colloc.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python -c "import yaml; c=yaml.safe_load(open('configs/one_per_well/n3_nonmcmc_residual_s42.yaml')); print(c['training']['sampler_mode'])"` -> expected: prints `non_mcmc`
**Risk:** Partial wiring leaves hidden MCMC calls in precompute or warm-start utilities.

## Phase 3 - Integrate residual/collocation objective schedule (session-sized)
**Depends on:** Phase 2 complete
**Goal:** Stage-I residual objective with moving target `E_eff` is functional and configurable for multi-well.
**Estimated scope:** 2-3 files, medium risk

### Step 3.1 - Implement `E_eff` target schedule
**What:** Add residual objective modes: self-target (`mu`), fixed reference, and annealed blend with cosine schedule.
**Files:** `src/training/vmc_colloc.py` and/or `src/imaginary_time_pinn.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python -c "from training.vmc_colloc import compute_eeff; import numpy as np; print([round(compute_eeff(mu=3.8,e_ref=3.6,epoch=e,total=100,mode='energy_var',alpha_start=0.0,alpha_end=1.0,alpha_decay_frac=0.5),4) for e in (0,25,50,100)])"` -> expected: monotone transition from mu toward e_ref
**Risk:** Wrong schedule semantics causing objective jumps.

### Step 3.2 - Preserve numerical robustness from current training
**What:** Keep/port clipping, finite checks, and quantile trimming in residual loss pipeline.
**Files:** `src/training/vmc_colloc.py`, `src/training/collocation.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python -c "import torch; from training.vmc_colloc import safe_trim; x=torch.tensor([1.,2.,3.,100.]); y=safe_trim(x,0.25); print(y.numel(), float(y.max()))"` -> expected: trims extreme outlier and returns finite tensor
**Risk:** Over-aggressive trimming can hide real failures.

### Step 3.3 - Add reproducible run metadata proving non-MCMC training
**What:** Persist sampler diagnostics (component usage, weights, non-MCMC flag) into result JSON/logs.
**Files:** `src/imaginary_time_pinn.py`, `src/run_ground_state.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && rg -n "sampler_mode|component_usage|mixture_weights|non_mcmc" results -g "*.json"` -> expected: fields present in new run outputs
**Risk:** Missing metadata makes post-hoc audit impossible.

## Phase 4 - Launch N=3 and N=4 non-MCMC trainings (session-sized)
**Depends on:** Phase 3 complete
**Goal:** Execute at least one stable non-MCMC residual training run for each of N=3 and N=4.
**Estimated scope:** 2-4 config files + run logs, medium risk

### Step 4.1 - Create launch configs (N=3, N=4)
**What:** Add dedicated configs using `sampler_mode=non_mcmc`, residual objective schedule, and explicit multi-well geometry.
**Files:** `configs/one_per_well/n3_nonmcmc_residual_s42.yaml`, `configs/one_per_well/n4_nonmcmc_residual_s42.yaml`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && python -c "import yaml; [print(yaml.safe_load(open(p))['training']['sampler_mode']) for p in ['configs/one_per_well/n3_nonmcmc_residual_s42.yaml','configs/one_per_well/n4_nonmcmc_residual_s42.yaml']]"` -> expected: both print `non_mcmc`
**Risk:** Config defaults unintentionally re-enable legacy behavior.

### Step 4.2 - Launch non-MCMC training runs
**What:** Start N=3 and N=4 runs with fixed seeds and capture logs/results.
**Files:** `src/run_ground_state.py`, `results/p4_n3_nonmcmc_*`, `results/p4_n4_nonmcmc_*`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python src/run_ground_state.py --config configs/one_per_well/n3_nonmcmc_residual_s42.yaml` -> expected: training starts, logs show non-MCMC sampler active
**Risk:** Training instability from sampler distribution mismatch.

### Step 4.3 - Fixed-protocol evaluation versus corrected diag
**What:** Evaluate final energies under fixed protocol and compare to post-fix diag references.
**Files:** `scripts/eval_ground_state_components.py`, `scripts/exact_diag_double_dot.py`, run result dirs
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python scripts/eval_ground_state_components.py --result-dir results/<new_run_dir> --n-samples 4096` -> expected: finite component report; then run exact diag command and compute percent error
**Risk:** Claiming improvement without protocol parity.

## Risks and mitigations
- Residual objective can be stiff/ill-conditioned: start with self-target (`alpha=0`) warmup and gradual anneal; keep clipping and outlier trimming.
- Non-MCMC sampler may undersample rare cusp regions: include dimer component + hard-example injection + minimum component floors.
- Silent MCMC fallback: enforce hard fail when `sampler_mode=non_mcmc` and MCMC path is reached.
- Cross-run comparability errors: use fixed eval commands and explicitly record evaluator settings in outputs.

## Anticipated expert invocations
None anticipated - standard implementation path.

## Success criteria
- Training path for selected runs uses no MCMC calls (verified by logs/flags/guards).
- N=3 and N=4 non-MCMC runs complete with finite losses/energies.
- Results include sampler diagnostics proving non-MCMC usage.
- Energy comparison against corrected diag can be reproduced by documented commands.

## Current State
**Active phase:** Phase 4 - Launch N=3 and N=4 non-MCMC trainings
**Active step:** Step 4.2 - Launch non-MCMC training runs
**Last evidence:** `PYTHONPATH=src .venv/bin/python src/run_ground_state.py --config configs/one_per_well/n2_nonmcmc_residual_warmup_s42.yaml` -> training started on `cuda:5`, epoch logs emitted with `sampler: stratified` and `ess=256.0` (i.i.d. non-MCMC batch).
**Current risk:** Early residual-only warmup shows high variance and negative mean local energy; schedule/sampler tuning may be needed before scaling to N=3/N=4.
**Next action:** Let N=2 warmup run continue to collect stability trend, then tune objective (annealed `energy_var`) and launch N=3 with the same non-MCMC path.
**Blockers:** None.
