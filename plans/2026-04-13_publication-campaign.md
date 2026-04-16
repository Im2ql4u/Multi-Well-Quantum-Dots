# Plan: Non-MCMC Neural VMC Publication Campaign

Date: 2026-04-13
Status: draft

## Paper thesis
Non-MCMC neural VMC for multi-well quantum dot arrays: ground-state accuracy matching exact diagonalization, spectral gap extraction via imaginary-time quench, continuous parameter response functions inaccessible to MCMC, and scaling to N=6–8 electrons in a 1D dot chain — all from a single differentiable framework with no Markov chains in the optimization loop.

## Target narrative
Three interlocking stories:
1. **Methods:** Non-MCMC (stratified i.i.d.) training is stable, accurate, and reproducible for up to N=8 correlated fermions in multi-well geometry.
2. **Spectroscopy:** The fully differentiable pipeline extracts spectral gaps from imaginary-time quench dynamics without MCMC anywhere in the pipeline.
3. **New observables:** Because the training is non-MCMC, the energy is differentiable w.r.t. Hamiltonian parameters — enabling forces between dots, magnetic susceptibility, and adiabatic-connection correlation energy decomposition that MCMC-based VMC cannot efficiently compute.

## What already exists
- Non-MCMC GS validated: N=2 (0.019%), N=3 (0.020%), N=4 (0.017%) vs exact diag, 3 seeds each
- GS artifacts in `results/` for N=2,3,4
- Quench pipeline (`src/imaginary_time_pinn.py`) with diagnosed bugs and clear fixes (see `plans/2026-04-13_nonmcmc-quench-pipeline.md`)
- Stratified sampler with 5-component mixture (`src/training/sampling.py`)
- Pair correlation / double-well sweep infrastructure (`results/double_well/`)
- Particle-selective Zeeman field (`zeeman_particle_indices` in `SystemConfig`)
- Exact diag reference code (`scripts/exact_diag_double_dot.py`)

## Verified constraints (from CONSTRAINTS.md)
- Non-MCMC requires MAD clipping for local-energy stability
- CI references are finite-basis — below-reference energies are not automatic proof of superiority
- Cross-run comparison requires locked evaluation protocol
- MCMC and non-MCMC results must stay in separate lanes

## Dependency graph

```
Phase 0 (foundation)
  ├── 0A: Fix quench pipeline bugs
  └── 0B: N=6 scaling run
        │
        ├── [if N=6 holds] ──► Phase 1 (scaling)
        │                        ├── 1A: N=8 scaling run
        │                        └── 1B: CI convergence ladder for N=4,6,8
        │
        └── [if N=6 degrades] ─► Diagnose, adjust narrative to N=2–4 depth
              │
Phase 2 (spectroscopy) [depends on 0A]
  ├── 2A: N=2 quench validation (known answer: gap=ω=1.0)
  ├── 2B: N=2,3,4 magnetic quench (B: 0→0.5)
  └── 2C: Non-MCMC quench (replace MCMC in precompute+eval)
        │
Phase 3 (differentiable observables) [depends on Phase 1]
  ├── 3A: Adiabatic connection (λ-sweep, 5-line code change)
  ├── 3B: Parameter derivatives (dE/dd, dE/dB)
  └── 3C: Addition energy spectra ΔE₂(N)
        │
Phase 4 (physics depth — pick 1–2)
  ├── 4A: Entanglement entropy across wells (SWAP/ratio trick)
  ├── 4B: Well-separation sweep → Wigner crossover
  └── 4C: Spin-selective quench (particle-mask Zeeman)
```

---

## Phase 0 — Foundation (1–2 sessions)

### 0A: Fix quench pipeline bugs
**Goal:** Get imaginary-time spectroscopy producing correct E(τ) decay curves.
**What:** Apply the 3 fixes diagnosed in `plans/2026-04-13_nonmcmc-quench-pipeline.md`:
1. Fix E_ref selection for generalized systems (use E_vmc when `ground_state_dir` is set)
2. Fix MCMC initialization for multi-well (init near well centers)
3. Fix well_sep override when loading generalized GS artifacts
**Files:** `src/imaginary_time_pinn.py` (~4 targeted edits)
**Acceptance:** N=2 single-dot `--tiny` quench produces monotonically decaying E(τ) and gap within 20% of ω=1.0.
**Risk:** Low — bugs are identified, fixes are clear.

### 0B: N=6 scaling run
**Goal:** Determine whether non-MCMC VMC maintains accuracy at N=6 (6-well chain, 1 electron per well).
**What:**
1. Run exact diag for N=6 one-per-well (wells at x = -10, -6, -2, +2, +6, +10, sep=4.0). Note: CI basis will be large — may need `n_ci_compute=500+` or reduced `n_sp_states`.
2. Create config `configs/one_per_well/n6_nonmcmc_residual_anneal_s42.yaml` — same architecture/training as N=4 but 6 wells, `n_coll` possibly increased to 768 or 1024.
3. Run training, compare against diag.
**Files:** `scripts/exact_diag_double_dot.py`, new config, `src/run_ground_state.py`
**Acceptance:** Relative error < 0.1% vs exact diag. If exact diag is too slow/unconverged, compare against product-state energy as upper bound.
**Risk:** Medium — the combinatorial CI basis for N=6 may not converge with current code. The stratified sampler may need wider coverage (6 wells span x ∈ [-10, +10]). Sample budget may need increase.
**Critical decision point:** If N=6 error > 1%, diagnose before proceeding. If > 5%, the scaling story needs adjustment.

---

## Phase 1 — Scaling (2–3 sessions, depends on 0B)

### 1A: N=8 scaling run
**Goal:** Push to 8-well chain.
**What:**
1. Exact diag for N=8 (if computationally feasible — likely needs DMRG or aggressive CI truncation).
2. Config `n8_nonmcmc_residual_anneal_s42.yaml` — wells at x = -14, -10, -6, -2, +2, +6, +10, +14.
3. Possible adjustments: increase `n_coll` to 1024–2048, widen sampler coverage, increase PINN hidden to 128.
4. Run 2 seeds minimum.
**Acceptance:** Energy within 0.5% of best available reference. Stable training (no variance explosion).
**Risk:** High — CI reference may be unavailable. Sampler coverage across 8 wells is uncharted. Determinant evaluation for 8×8 is fine but backflow/PINN cost grows.
**Fallback:** If training is unstable, this is diagnostic information worth reporting ("the framework reaches its scaling limit at N=X for these specific reasons").

### 1B: CI convergence ladder
**Goal:** Establish trustworthy references by sweeping `n_sp_states` and `n_ci_compute`.
**What:** For N=4, 6 (and 8 if feasible), run exact diag with increasing basis:
- `n_sp_states` ∈ {20, 30, 40, 60}
- `n_ci_compute` ∈ {100, 200, 500, 1000}
Plot E₀ vs basis size. Determine convergence within stated accuracy.
**Files:** `scripts/exact_diag_double_dot.py`
**Acceptance:** E₀ changes by < 0.01% between last two basis sizes for N=4. For N=6+, document the convergence trajectory even if not fully converged.
**Risk:** For N≥6, the Hilbert space explodes. May need to state "VMC energy is below our best CI approximation" — which is itself a publishable finding.

### 1C: Multi-seed robustness for N=6, N=8
**Goal:** 3-seed sweep for each system size, matching the N=2/3/4 protocol.
**What:** Seeds 42, 314, 901 for each. Report mean, SD, and one-sided exceedance.
**Acceptance:** SD < 0.5% of mean energy. Exceedance < 0.1%.

---

## Phase 2 — Spectroscopy (2–3 sessions, depends on 0A)

### 2A: N=2 quench validation (known answer)
**Goal:** Validate the quench pipeline on a system with known spectral gap.
**What:** N=2 single dot, B=0→0, Coulomb on. Known gap = ω = 1.0 (to first order).
- Use fixed quench pipeline from Phase 0A.
- Run with MCMC first (baseline), then non-MCMC.
**Acceptance:** Extracted gap within 15% of ω=1.0. E(τ) monotonically decreasing. n_eff > 500 at τ_max.

### 2B: Multi-well magnetic quench campaign
**Goal:** Produce publication-quality spectral gap curves for N=2,3,4 under magnetic field quench.
**What:** For each system size:
- B: 0 → 0.5 (forward quench)
- B: 0.5 → 0 (reverse quench, if time permits)
- 2 seeds each
**GS artifacts:** Use the validated non-MCMC ground states from the seed sweep.
**Acceptance:** E(τ) decays for all runs. Gap extraction succeeds. n_eff stays above 200.
**Deliverable:** Figures showing E(τ) curves, extracted gaps vs N, and gap vs B.

### 2C: Fully non-MCMC quench pipeline
**Goal:** Replace MCMC in precompute and evaluation with stratified sampling.
**What:** Implement stratified precompute (as designed in `plans/2026-04-13_nonmcmc-quench-pipeline.md` Phases 2–3):
1. `precompute_ground_state_stratified()` using `stratified_resample()` + MAD clipping
2. Importance-weighted evaluation with q(x) correction
3. Config flag `precompute_sampler: "stratified"`
**Acceptance:** N=2 gap within 20% of MCMC baseline from Step 2A.
**Significance:** "Zero MCMC anywhere in the pipeline" is the headline claim.

---

## Phase 3 — Differentiable Observables (2–3 sessions, depends on Phase 1)

This is the unique selling point of non-MCMC. MCMC training makes the energy a stochastic function of parameters (because the sampling distribution depends on the wavefunction which depends on training). Non-MCMC training with i.i.d. samples makes the energy deterministically differentiable w.r.t. Hamiltonian parameters.

### 3A: Adiabatic connection (λ-sweep)
**Goal:** Compute correlation energy as a function of Coulomb coupling strength.
**What:**
1. **Code change (~5 lines):** Replace `coulomb: bool` with `coulomb_strength: float` in `SystemConfig` and `compute_potential()`. When `coulomb_strength=1.0`, recover current behavior. When `0.0`, non-interacting. Intermediate values scale the Coulomb term.
2. Run N=2,3,4 (and N=6 if available) at λ = 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0.
3. At λ=0: verify against non-interacting analytic solution (E = N × ω for 1D HO, or look up 2D).
4. Compute: $E_c(\lambda) = E(\lambda) - E_{\text{HF}}(\lambda)$, and the adiabatic connection integrand $\langle V_{ee} \rangle_\lambda$.
5. Integrate: $E_c = \int_0^1 d\lambda \, [\langle V_{ee} \rangle_\lambda - \langle V_{ee} \rangle_{\text{HF}}]$.
**Acceptance:** E(λ=0) matches analytic non-interacting result. E(λ=1) matches existing ground-state results. Integrand is smooth.
**Deliverable:** Figure showing correlation energy buildup vs λ for different N. This is a DFT-relevant result.
**Risk:** Low for code change. Training at small λ should be easier (less correlation). The interesting question is whether the correlation energy per particle has a clear trend with N.

### 3B: Parameter derivatives
**Goal:** Demonstrate that non-MCMC VMC enables differentiable observables.
**What:**
1. **Inter-dot force:** $F(d) = -dE/dd$ where $d$ is well separation. Train E(d) at d = 2.0, 3.0, 4.0, 5.0, 6.0, 8.0 for N=2,3,4. Compute force by finite difference. Then demonstrate that we can also compute it via automatic differentiation through the energy evaluation.
2. **Magnetic susceptibility:** $\chi = -d^2E/dB^2$. Train at B = 0.0, 0.1, 0.2, 0.3, 0.5 and compute χ by finite difference.
3. Compare FD derivatives with autograd derivatives (if feasible — requires making the potential parameters torch tensors and differentiating through the energy expectation).
**Acceptance:** Forces are smooth, physically reasonable (repulsive at small d, vanishing at large d). χ has correct sign and magnitude.
**Deliverable:** Force curves F(d) and susceptibility χ(B) for N=2,3,4. If autograd derivatives work, a direct comparison showing agreement.
**Risk:** The autograd path through the full energy evaluation may have memory issues for large N. FD derivatives are always available as fallback.

### 3C: Addition energy spectra
**Goal:** Compute Coulomb blockade addition energies.
**What:** For a fixed 8-well chain (or 6-well if N=8 doesn't work):
1. Train E(N) for N = 1 through N = max_wells, placing electrons one per well progressively (1→2→3→…→8).
2. Compute $\Delta_2(N) = E(N+1) + E(N-1) - 2E(N)$.
3. N=1 is trivial (single particle in a well, E = ω). N=2 is already done.
**Acceptance:** $\Delta_2$ shows expected shell/correlation structure. Even-odd oscillation from spin pairing.
**Caveat:** This requires training new configs for N=1, 5, 7 (and N=6 from Phase 1). The under-occupied chain configs (e.g., 3 electrons in 8 wells) are untested — orbital assignment and sampler may need adjustment.
**Risk:** Medium — the orbital round-robin assignment in `wavefunction.py` distributes electrons across all wells. For 3 electrons in 8 wells, the SD construction needs to handle sparse occupancy correctly. May need a session of foundation work.

---

## Phase 4 — Physics Depth (pick 1–2, each ~1–2 sessions)

### 4A: Entanglement entropy across wells
**Goal:** Measure bipartite Rényi-2 entropy between left and right halves of the dot chain.
**What:** Use the SWAP trick: $\text{Tr}(\rho_A^2) = \langle \text{SWAP}_A \rangle$ evaluated by sampling two independent copies of the wavefunction and swapping particles in subsystem A.
- Partition: left half of wells vs right half.
- Measure $S_2(d)$ as a function of well separation for N=4.
- Measure $S_2(N)$ for N=2,4,6,8 at fixed separation.
**Implementation:** New script (~100 lines) that loads a trained wavefunction, generates paired samples, computes SWAP ratio.
**Risk:** Variance of the SWAP estimator can be exponential in system size. May need large sample counts. For N=4 this should be fine; for N=8 it might be marginal.

### 4B: Well-separation sweep → Wigner crossover
**Goal:** Map the delocalized-to-localized transition as wells are brought together or pushed apart.
**What:** For N=4, sweep d = 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0.
- Compute: E(d), pair correlation g(r), one-body density ρ(x).
- Look for: crossover from delocalized (particles spread across wells) to localized (one electron pinned per well).
**Implementation:** Pair correlation code exists in `src/functions/Analysis.py`. Density is straightforward.
**Risk:** At very small d, the multi-well potential merges into something like a single wider well. The SD orbital assignment may not adapt well to this.

### 4C: Spin-selective quench
**Goal:** Demonstrate particle-selective Zeeman dynamics — a capability unique to this framework.
**What:** For N=4 (1+1+1+1):
- Quench B on particles {0,2} only (every other dot) while keeping {1,3} at B=0.
- Compare against uniform B quench.
- This simulates selective spin manipulation in a QD chain — directly relevant to qubit readout protocols.
**Implementation:** Already supported via `zeeman_particle_indices` in `SystemConfig`. Just needs a config.
**Risk:** Low for ground state. Medium for quench (quench pipeline must correctly propagate the particle mask).

---

## Phase 5 — Ambitious extensions (future paper territory, but start prototyping)

### 5A: Excited-state wavefunctions via penalty method
**Goal:** Train not just E₀ but also E₁, E₂ explicitly.
**What:** Add orthogonality penalty: train ψ₁ with loss = E[ψ₁] + μ |⟨ψ₀|ψ₁⟩|². This gives access to transition matrix elements → optical spectra.
**Risk:** High research risk. Penalty methods for excited states in VMC are known to be finicky.

### 5B: Transfer learning across system sizes
**Goal:** Train on N=4, fine-tune for N=6 and N=8.
**What:** Save PINN + backflow weights from N=4. Initialize N=6 run with those weights (expanding the SD/orbital layer). Measure: how many epochs to converge vs training from scratch?
**Risk:** Architectural mismatch (different number of pairs/orbitals). May need careful weight mapping.

### 5C: 2D dot arrays (2×2 plaquette)
**Goal:** Move from 1D chain to 2D array — 4 dots at corners of a square.
**What:** The config already supports 2D well positions. Just set centers to (0,0), (d,0), (0,d), (d,d).
**Risk:** Medium — the LCAO orbital construction and stratified sampler assume roughly 1D geometry. 2D requires testing the orbital assignment with 2D well centers.

---

## Execution timeline (ambitious but realistic)

### Week 1: Foundation
- **Session 1:** Phase 0A (fix quench bugs) + Phase 0B (launch N=6 run)
- **Session 2:** Phase 2A (N=2 quench validation) + assess N=6 results → go/no-go on scaling

### Week 2: Scaling + quench
- **Session 3:** Phase 1A (N=8 run) + Phase 1B (CI convergence ladder)
- **Session 4:** Phase 2B (magnetic quench campaign for N=2,3,4)

### Week 3: Differentiable observables
- **Session 5:** Phase 3A (adiabatic λ-sweep — code change + N=2,3,4 runs)
- **Session 6:** Phase 3B (parameter derivatives: force curves + susceptibility)

### Week 4: Depth + campaign completion
- **Session 7:** Phase 2C (fully non-MCMC quench) + Phase 1C (multi-seed scaling)
- **Session 8:** Phase 3C (addition spectra) or Phase 4A (entanglement) — pick based on results so far

### Week 5: Extensions + writing
- **Session 9:** Phase 4B or 4C (Wigner sweep or spin-selective quench)
- **Session 10:** Phase 5B (transfer learning prototype) + figure/table generation
- Begin paper draft with established results

### Ongoing throughout:
- Every new system size gets exact diag reference (or documented convergence limit)
- Every run gets 2+ seeds before any claim
- Results go into the correct lane (`results/nonmcmc_training/`)
- Figures accumulate in `results/figures/`

---

## Decision gates

| Gate | Condition | Action if fails |
|------|-----------|-----------------|
| G1: N=6 scaling | Error < 0.1% vs diag | Diagnose (sampler? determinant? training?). If unfixable, paper focuses on N≤4 with Phase 3/4 depth |
| G2: N=8 scaling | Error < 0.5% vs best ref | Report as "regime where VMC outpaces CI" if VMC < CI. If training unstable, report scaling frontier |
| G3: Quench validation | N=2 gap within 15% of ω | Do not proceed to multi-well quench until fixed |
| G4: λ=0 check | E(λ=0) matches analytic | Do not trust correlation energy decomposition until this passes |
| G5: Force curve | F(d) is repulsive and smooth | If noisy, increase seeds/samples before claiming |

---

## Paper outline (tentative)

1. **Introduction:** Neural VMC for quantum dots; why non-MCMC matters
2. **Method:** Wavefunction ansatz (SD + PINN correlator + backflow), stratified i.i.d. sampling, residual/collocation loss, MAD clipping
3. **Ground states:** N=2 through N=6(8), accuracy vs exact diag, scaling analysis, seed robustness
4. **Spectral gaps:** Imaginary-time PDE solver, magnetic quench for N=2,3,4, validation against known gaps
5. **Differentiable observables:**
   - Adiabatic connection and correlation energy
   - Inter-dot forces and magnetic susceptibility
   - (Addition spectra if available)
6. **Discussion:** What non-MCMC enables that MCMC cannot; scaling limits; future directions (2D, excited states, transport)
7. **Conclusion**

---

## Success criteria for the full campaign
- [ ] Ground-state accuracy < 0.1% for N=2–6, < 0.5% for N=8
- [ ] At least one spectral gap extracted from non-MCMC pipeline
- [ ] Adiabatic connection λ-sweep for N=2,3,4 with smooth integrand
- [ ] At least one parameter derivative demonstrated (force or susceptibility)
- [ ] 3-seed robustness for flagship results
- [ ] All results reproducible from commit hash + config file
