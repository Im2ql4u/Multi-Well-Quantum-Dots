# Plan: Non-MCMC Collocation-Based Quench Pipeline

Date: 2026-04-13
Status: draft

## Project objective
Produce publication-quality imaginary-time spectroscopy (magnetic quench dynamics / gap extraction) for N=2, N=3, N=4 one-per-well quantum dots, using non-MCMC collocation-based sampling throughout the entire pipeline (ground state training + quench evolution).

## Context and diagnosed issues

### Diagnosis of existing quench failures
All existing quench results in `results/imag_time_pinn/` show E(τ) increasing instead of decaying. Root causes identified:

1. **PRIMARY — Wrong E_ref (Layer 4):** The PDE residual is R = ∂_τ g + (E_L - E_ref). For multi-well systems loaded via `ground_state_dir`, `cfg.well_sep` stays at default `0.0`, so the E_ref selection logic at `run_single()` line ~1617 uses `cfg.E_ref=3.0` (N=2 single-dot value) instead of the actual E_vmc≈5.09 for N=4. This forces the network to learn a non-physical linearly-growing g, collapsing importance weights.

2. **SECONDARY — MCMC init for multi-well (Layer 2):** `mcmc_sample()` initializes all particles at origin and only offsets for legacy 2-well (`well_sep > 0`). Multi-well systems (well centers at ±2, ±6) rely on 500 warmup steps to find distant wells — marginal.

3. **TERTIARY — No well_sep override:** When loading generalized GS artifacts, `well_sep` is never set from `system_override`, yet multiple code paths branch on it.

### What already works
- Non-MCMC GS training validated for N=2 (0.019%), N=3 (0.020%), N=4 (0.017%) vs exact diag
- GS artifacts ready in `results/nonmcmc_training/` and `results/mcmc_training/`
- Stratified sampler (`src/training/sampling.py`) with 5-component mixture proven
- SpectralG architecture with analytical τ-derivative
- Quench pipeline structure (Phases 1-4) is sound, just buggy in configuration

### Relevant negative history
- Non-MCMC without MAD clipping was unstable (variance explosion). Must apply clipping to precompute E_L0 values when generated from off-distribution stratified samples.
- Cross-run conclusions under mixed evaluation settings were invalid. Must use consistent protocol.

## Approach
Fix bugs first, validate with known-answer test (N=2 single dot, gap=ω=1.0), then progressively replace MCMC with stratified sampling in the quench pipeline. Each phase is session-sized and independently useful.

## Phase 1 — Fix bugs and validate MCMC quench (this session)
**Goal:** Get a working, honest quench result using MCMC + correct E_ref, so we have a baseline before switching to non-MCMC.
**Estimated scope:** ~4 edits to `imaginary_time_pinn.py`, 1-2 validation runs.

### Step 1.1 — Fix E_ref selection for generalized systems
**What:** When `ground_state_dir` is set, always use `E_ref = E_vmc` regardless of `well_sep`. The current logic `E_ref = E_vmc if cfg.well_sep > 0.01 else cfg.E_ref` is wrong for multi-well.
**Files:** `src/imaginary_time_pinn.py` (line ~1617)
**Fix:** Replace the E_ref selection block with:
```python
if cfg.ground_state_dir is not None:
    # Generalized systems: use measured VMC energy as reference
    E_ref = E_vmc
elif cfg.no_vmc_train:
    if cfg.coulomb:
        E_ref = 3.0 if cfg.well_sep <= 1e-10 else 2.0 + 1.0 / max(cfg.well_sep, 1e-10)
    else:
        E_ref = 2.0
else:
    E_ref = E_vmc if cfg.well_sep > 0.01 else cfg.E_ref
```
**Acceptance:** `grep -n "ground_state_dir is not None" src/imaginary_time_pinn.py` shows the new branch.
**Risk:** If E_vmc has noise, E_ref will be slightly off. Acceptable — the PDE residual is robust to small E_ref offsets.

### Step 1.2 — Fix MCMC initialization for multi-well systems
**What:** When `system_override` is available, initialize MCMC walkers near their assigned well centers instead of at origin.
**Files:** `src/imaginary_time_pinn.py` — `precompute_ground_state()` and `estimate_energy()`
**Fix:** After calling `mcmc_sample()`, if `system_override` has multi-well geometry, re-center each particle near its well. Or: add a `well_centers` parameter to `mcmc_sample()`.
**Acceptance:** Print mean particle positions after warmup; they should cluster near well centers.
**Risk:** Over-engineering the MCMC init when Phase 2 will replace it entirely.

### Step 1.3 — Known-answer validation: N=2 single-dot quench (B=0→0)
**What:** Run `--tiny` mode (N=2 interacting single dot, no magnetic field, no quench). This has exact answer: E₀≈3.0, gap=ω=1.0. Verify E(τ) decays and gap extraction works.
**Files:** CLI `--tiny` already exists.
**Acceptance:** E(τ) monotonically decreases. Extracted gap within 20% of ω=1.0. n_eff stays above 500 at τ_max.
**Risk:** Existing `--tiny` path is unaffected by our E_ref fix (uses legacy path), so this validates the base pipeline integrity.

### Step 1.4 — Validation: N=4 multi-well quench with MCMC + correct E_ref
**What:** Run quench with locked GS artifact from `results/nonmcmc_training/p4_n4_*`, B=0→0.5, with the fixed E_ref logic.
**Acceptance:** E(τ) decays. n_eff > 200 at τ_max. PDE loss converges below 0.01.
**Risk:** MCMC init may still be marginal for N=4 multi-well. If so, Step 1.2 fix becomes critical.

## Phase 2 — Non-MCMC precompute (replace Phase 2 MCMC)
**Goal:** Replace `mcmc_sample()` in `precompute_ground_state()` with `stratified_resample()` from `src/training/sampling.py`.
**Depends on:** Phase 1 (confirmed working MCMC baseline).
**Estimated scope:** ~50 lines new code in `imaginary_time_pinn.py`.

### Step 2.1 — Add stratified precompute function
**What:** Create `precompute_ground_state_stratified()` that:
1. Calls `stratified_resample()` with `system_override` well geometry
2. Computes E_L0, ∇logψ₀, ΔV on those samples (same math as MCMC version)
3. Applies MAD clipping to E_L0 (critical for off-distribution stability)
4. Returns same dict format as `precompute_ground_state()`
**Files:** `src/imaginary_time_pinn.py`
**Acceptance:**
```python
# Precomputed pool should have:
# - x.shape = (n_precompute, n_particles, dim)
# - E_L0 finite, no NaN/Inf
# - |E_L0.mean() - E_vmc| < 0.5 (samples are reasonable)
```
**Risk:** Stratified samples x ∼ mixture-of-Gaussians ≠ x ∼ |ψ₀|². E_L0 variance will be higher. MAD clipping should handle this, but may need wider clip_width than GS training.

### Step 2.2 — Wire stratified precompute into run_single()
**What:** Add config flag `precompute_sampler: str = "mcmc"` to `PINNConfig`. When `"stratified"`, call `precompute_ground_state_stratified()` instead of `precompute_ground_state()`.
**Files:** `src/imaginary_time_pinn.py` — `PINNConfig`, `run_single()`
**Acceptance:** Run with `precompute_sampler="stratified"` — training starts, E_L0 stats printed, no NaN.
**Risk:** Config drift between legacy and new paths.

### Step 2.3 — Validate: N=2 non-MCMC precompute + MCMC eval
**What:** Run N=2 single-dot quench using stratified precompute but standard (MCMC-based) evaluation. Compare trajectory to Phase 1 result.
**Acceptance:** Gap estimate within 15% of MCMC-only result from Step 1.3. E(τ) shape qualitatively similar.
**Risk:** Higher E_L0 variance from stratified sampling may increase PDE training noise.

## Phase 3 — Non-MCMC evaluation (replace Phase 4 MCMC)
**Goal:** Replace the MCMC-sampled evaluation pool with stratified samples + proper importance weighting.
**Depends on:** Phase 2 (non-MCMC precompute validated).
**Estimated scope:** ~40 lines new code.

### Step 3.1 — Add stratified evaluation path
**What:** In `evaluate_trajectory()`, when precomputed pool was generated by stratified sampler, importance weights must account for the sampling distribution:
```
w ∝ |ψ(x,τ)|² / q(x)  where q(x) = stratified mixture density
```
Currently: `w ∝ exp(2g)` which assumes x ∼ |ψ₀|² (MCMC).
For stratified: `w ∝ |ψ₀(x)|² · exp(2g) / q(x)`.
**Files:** `src/imaginary_time_pinn.py` — `evaluate_trajectory()`
**Acceptance:** n_eff stays above 100 for N=2 through full τ range.
**Risk:** Computing q(x) for the mixture requires knowing component assignment. Alternative: use self-normalized importance weights with |ψ₀|²/q as base weight stored in precomputed dict.

### Step 3.2 — Validate: Fully non-MCMC N=2 quench
**What:** Run complete non-MCMC pipeline: locked GS (non-MCMC trained) → stratified precompute → PINN training → stratified evaluation. No MCMC anywhere.
**Acceptance:** Gap within 20% of known ω=1.0 for N=2 single dot. E(τ) decays monotonically.
**Risk:** Compounding errors from off-distribution sampling in both precompute and eval.

## Phase 4 — Production runs: N=2, N=3, N=4 quench campaigns
**Goal:** Run calibrated non-MCMC quench experiments for all three system sizes.
**Depends on:** Phase 3 (fully non-MCMC pipeline validated on N=2).
**Estimated scope:** 6-12 config files, multi-GPU runs.

### Step 4.1 — N=2 quench campaign
**What:** B=0→0 (no quench, gap=ω validation), B=0→0.5 (magnetic quench). Two seeds each.
**GS artifacts:** `results/nonmcmc_training/p4_n2_nonmcmc_residual_anneal_s42_20260412_232259/`
**Acceptance:** B=0→0 gap within 10% of ω. B=0→0.5 produces sensible Zeeman-split trajectory.

### Step 4.2 — N=3 quench campaign
**What:** B=0→0.5, B=0.5→0 (reverse quench). Two seeds each.
**GS artifacts:** `results/nonmcmc_training/p4_n3_nonmcmc_residual_anneal_s42_20260413_001421/`
**Acceptance:** E(τ) decays, gap extraction succeeds, n_eff stable.

### Step 4.3 — N=4 quench campaign
**What:** B=0→0.5, B=0.5→0, varying Zeeman particle subsets. Two seeds each.
**GS artifacts:** `results/nonmcmc_training/p4_n4_nonmcmc_residual_anneal_s42_20260413_001824/`
**Acceptance:** Same as N=3. Compare against MCMC baseline from Step 1.4.

### Step 4.4 — Cross-comparison: MCMC vs non-MCMC quench
**What:** For N=4 B=0→0.5 case, run matched MCMC and non-MCMC quench. Compare trajectories, gap estimates, n_eff profiles.
**Acceptance:** Non-MCMC gap estimate within 20% of MCMC. If not, diagnose which phase (precompute or eval) contributes most variance.

## Risks and mitigations
- **Stratified sampling distribution mismatch:** MAD clipping + wider precompute pool (2× n_precompute).
- **n_eff collapse at large τ:** Monitor g_rms; if >3 at τ_max, reduce τ_max or add g regularization.
- **E_ref noise from E_vmc:** Use exact diag reference when available instead of E_vmc estimate.
- **MCMC init for multi-well:** Phase 2 eliminates this entirely by replacing MCMC.

## Success criteria
1. N=2 non-MCMC quench reproduces gap=ω within 15%
2. N=3 and N=4 quench trajectories decay monotonically with finite n_eff
3. Gap estimates are consistent across 2 seeds (spread < 30% of mean)
4. No MCMC calls in the final non-MCMC pipeline path (verified by log output)

## Current State
**Active phase:** Phase 1 — Fix bugs and validate
**Active step:** Step 1.1 — Fix E_ref selection
**Last evidence:** Diagnosis complete (E_ref=3.0 used for N=4 system with E_vmc≈5.09)
**Blockers:** None
