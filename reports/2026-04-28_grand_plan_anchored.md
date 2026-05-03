# Grand research plan — every step has a ground-truth anchor

**Author:** Aleksander Skogen
**Date drafted:** 2026-04-28 ~12:00 CEST
**Horizon:** 5 months (target: 2026-09-30)
**Companion documents:**
- [`reports/2026-04-28_research_roadmap.md`](2026-04-28_research_roadmap.md) — the tactical 5-week plan; this file extends and re-frames it.
- [`reports/2026-04-28_supervisor_update.md`](2026-04-28_supervisor_update.md) — running update.
- [`reports/2026-04-27_supervisor_report.md`](2026-04-27_supervisor_report.md) — frozen snapshot.

---

## 0. The principle

**Every result we publish must be triangulated against at least one independent ground-truth anchor.** No result moves to the next phase until its anchors pass. No phase begins until its anchors are operational.

Three rules:

1. **No floating findings.** The d-sweep "PINN diverges from Heisenberg at large d" finding sat for two weeks without an external benchmark. We will not repeat that. Every observable claim has at least one of {ED, analytical limit, RHF, Heisenberg-toy, prior PINN run, published experimental data} as a check.
2. **Anchors are pre-registered with thresholds.** Each anchor has a quantitative pass/fail criterion *defined before we run the comparison*. If we hand-tune the threshold after seeing the result, the anchor doesn't count.
3. **Anchor failure stops forward progress.** A failed anchor either reveals a bug (fix and retry), forces a model-class change (replan), or — best case — is a real scientific discovery (write up *as such*, not as nuisance).

---

## 1. Anchor inventory (what we have, what we need to build)

| # | Anchor | Status in repo | Where / what's missing |
|---|---|---|---|
| 1 | **CI-DVR ED at N=2 (shared DVR)** | ✅ exists | `scripts/exact_diag_double_dot.py:684+`. Static spectrum via `eigh`. |
| 2 | **CI-DVR ED at N=2-N=8 (one per well)** | ✅ exists | Same file, `run_exact_diagonalization_one_per_well_multi:318-399`. |
| 3 | **⟨S_i·S_j⟩ on a CI-DVR eigenvector** | ❌ **GAP** | Spin sectors classified in `scripts/characterize_quench.py:36-46`, but full N×N spin-correlator on the CI eigenvector not implemented. |
| 4 | **ED-Trotter real-time evolution** | ❌ **GAP** | No `expm(-iHt)` in `exact_diag_double_dot.py`. Existing `time_evolve_trained` in `src/train_double_well.py:419` is Langevin diffusion in `\|ψ\|²`, *not* Schrödinger. |
| 5 | **Heitler-London analytical N=2** | ❌ **GAP** | Mentioned only in `JOURNAL.md`. ~50 LoC to add. |
| 6 | **Heisenberg OBC reference (uniform J)** | ✅ exists | `src/observables/heisenberg_reference.py`. Uniform J only; bond-couplings hook exists. |
| 7 | **Schrieffer-Wolff: extract t,U from continuum, predict J=4t²/U** | ❌ **GAP** | Only narrative in `heisenberg_reference.py:16` comment. ~150 LoC. |
| 8 | **Hartree-Fock / RHF baseline** | ✅ exists | `src/functions/Slater_Determinant.py:hartree_fock_closed_shell`, `hartree_fock_2d`. Not wired as default validation. |
| 9 | **Free-particle Slater limit (d→∞)** | ⚠️ partial | Slater builders exist; the d→∞ workflow (one HO orbital per well, no Coulomb) needs ~30 LoC of glue. |
| 10 | **Mott spin-amplitude extractor + ⟨S·S⟩** | ✅ exists | `src/observables/spin_amplitude_entanglement.py`, `src/observables/effective_heisenberg.py`. |
| 11 | **N=2 PINN-vs-ED energy/density/⟨S·S⟩ glue script** | ❌ **GAP** | Existing `scripts/validate_spin_amplitude_n2.py` checks only spin structure, not E and density vs ED. ~80 LoC glue. |
| 12 | **Heisenberg cross-check on chain** | ✅ exists | `scripts/heisenberg_cross_check.py` and `analyse_one` in `scripts/n_chain_d_sweep.py`. |
| 13 | **Symmetry assertions (Sz, N) post-training** | ⚠️ soft only | `same_dot_occupancy_penalty` in trainer is a soft penalty, not an assertion. Hard assert costs <10 LoC. |
| 14 | **Post-training virial theorem check** | ✅ exists | `scripts/check_virial_multiwell.py`. Cheap, powerful — should be standard. |
| 15 | **Independent second NQS implementation** | ❌ **GAP** | None. Optional but powerful — could use `netket` as third-party cross-check at small N. ~1 day to wire. |
| 16 | **DMRG / MPS via tenpy or quimb** | ❌ **GAP** | None. Useful for 1D Mott-limit cross-check. ~3 days to wire if we accept the tenpy dep. |
| 17 | **PySCF FCI / CCSD(T) reference** | ❌ **GAP** | None. Best for closed-shell N≤6 in a localized basis. ~3-5 days to wire. Optional. |
| 18 | **Si / GaAs effective-mass material presets** | ❌ **GAP** | No `m_eff`, `epsilon_r` in any config. Trivial to add but defines the experimental bridge. |
| 19 | **Lieb-Robinson velocity bound (analytical)** | ❌ **GAP** | Pure formula, no implementation needed; just a comparison script. |
| 20 | **Calabrese-Cardy quench (analytical)** | ❌ **GAP** | CFT prediction for 1D-critical, ~20 LoC closed form. |

**Construction schedule for the anchors:** see Phase 0 below (~3 days).

---

## 2. Phase 0 — Anchor infrastructure (Days 1-3)

**Goal:** before doing *any* new physics, build the missing anchors. None of this is glamorous; all of it is the "always know if results are legit" requirement.

| # | Task | File | LoC | Days |
|---|---|---|---|---|
| 0.1 | Hard symmetry asserts after every Stage A and Stage B training: ⟨Sz⟩ = sector_Sz to 1e-6, ⟨N⟩ = N_target to 1e-6. Fail-loud (raise) if violated. Wire into `vmc_colloc.py` and `imaginary_time_pinn.py`. | `src/training/vmc_colloc.py`, `src/training/qhe_collocation.py` | 80 | 0.5 |
| 0.2 | `compute_spin_correlators_ci(eigvec, basis) -> ndarray[N,N]` — full ⟨S_i·S_j⟩ on a CI-DVR eigenvector. Validate at N=2: singlet must give −¾, triplet +¼. | `src/observables/exact_diag_reference.py`, `tests/test_spin_correlators_ci.py` | 200 | 1 |
| 0.3 | `compare_pinn_ed_n2.py` — load N=2 PINN checkpoint, compute E, density, ⟨S·S⟩ on CI-DVR ED, write JSON+PNG comparison. (Phase-1 anchor A1.2.) | `scripts/compare_pinn_ed_n2.py` | 150 | 0.5 |
| 0.4 | `heitler_london_n2.py` — closed-form 2-electron-2-well analytical solution. Returns E_S, E_T, J_HL = E_T - E_S, ⟨S·S⟩, density. | `src/observables/heitler_london.py`, `tests/test_heitler_london.py` | 200 | 0.5 |
| 0.5 | `ed_trotter_evolve.py` — given an ED Hamiltonian and a starting CI vector, evolve under either `scipy.linalg.expm(-iH·dt)` (gold standard at N≤6) or Suzuki-Trotter (for larger N). Returns time-resolved observables. | `src/observables/ed_trotter.py`, `tests/test_ed_trotter.py` | 250 | 1 |
| 0.6 | `extract_tight_binding.py` — given a multi-well config, build localised Wannier-like orbitals (Löwdin orthogonalisation of the HO orbitals at each well) and extract on-site U_ii, hopping t_ij, NN exchange J_SW = 4t²/U. | `src/observables/tight_binding_extract.py`, `tests/test_tight_binding_extract.py` | 350 | 1 |
| 0.7 | `materials.py` — Si MOS (m* = 0.19 m_e, ε ≈ 11.7), GaAs (m* = 0.067 m_e, ε ≈ 12.9), Si:P donor (ε ≈ 11.7, m* ≈ 0.2). Effective Bohr radius and Hartree set the conversion to `kappa`/`omega` units used in our configs. Auto-emit a "preset" config block. | `src/observables/materials.py`, `configs/materials/si_mos_default.yaml`, `configs/materials/gaas_default.yaml` | 200 | 0.5 |
| 0.8 | Full virial-theorem auto-check (already exists in `scripts/check_virial_multiwell.py`) wired as a default post-training assertion: |2T + V| / |E| < 0.05. | `src/training/vmc_colloc.py` | 30 | 0.25 |

**Phase 0 deliverable:** `tests/test_phase0_anchors.py` exercising every new anchor in unit form. **Gate G0: all 7 unit tests pass; runtime < 5 minutes.**

**Without Phase 0 done, Phase 1 cannot start.** This is the discipline we said we wanted.

---

## 3. Phase 1 — Lock-down static validation (Weeks 1-3)

**Goal:** all existing static-state PINN claims at N=2, N=4, N=8 are validated against multiple anchors. No "novel finding" gets stamped publishable until every anchor is green.

### Anchor table for Phase 1

| Anchor ID | What we compare | Threshold | If it fails |
|---|---|---|---|
| **A1.1** | N=2 PINN E_GS vs Heitler-London analytical E_GS at d ∈ {2, 3, 4, 6, 8, 14} | <0.5% relative | Fundamental bug: re-investigate trainer. |
| **A1.2** | N=2 PINN E, density(x), ⟨S₀·S₁⟩ vs CI-DVR ED at d=4 | <0.1% on E, <2% L1 on density, <2% on ⟨S·S⟩ | Either ED basis truncated too low (cross-check by raising basis) or PINN bug. |
| **A1.3** | N=4 PINN E_GS, ⟨S·S⟩ matrix, gap, density vs CI ED at d ∈ {2, 3, 4, 6, 8, 10, 14} | <1% on E, <5% on full ⟨S·S⟩ matrix, <10% on gap | If only a *subset* of d fails: likely correlation-strength regime issue. If all fail: bug. *This anchor is the gate that decides the d-sweep "PINN diverges from Heisenberg" story.* |
| **A1.4** | N=2 and N=4 PINN E_GS at d → ∞ vs free-Slater Hartree-Fock baseline | E_GS asymptotes to N · E_single_well; the gap between PINN and HF closes (correlation→0) | If PINN doesn't reach HF asymptote: ansatz can't represent the non-interacting limit. Investigate. |
| **A1.5** | N=4 PINN J_extracted (from `effective_heisenberg.py`) vs J_SW = 4t²/U from `extract_tight_binding.py` at d ∈ {3, 4, 6, 8} | <30% (SW is 2nd-order; higher orders contribute) | If the *trend* in d disagrees: Schrieffer-Wolff truncation breaks down at some d, and we have a real result. |
| **A1.6** | Symmetry: ⟨Sz⟩ — sector_Sz; ⟨N⟩ — N_target | <1e-6 each | Hard fail. Don't publish anything from a checkpoint that violates this. |
| **A1.7** | Virial theorem 2⟨T⟩ + ⟨V⟩ + extra terms ≈ 0 (modulo confinement) | <5% relative | Likely indicates incomplete sampling or training collapse. |
| **A1.8** | Multi-seed reproducibility (3 seeds): every observable should agree to within 1σ of its variational variance | seeds within 2σ | Single-seed result not robust. |
| **A1.9** | RHF energy as upper bound on PINN E_GS | E_PINN ≤ E_RHF + 0 | Hard fail (variational principle violated). Bug. |

### Sub-tasks

| # | Task | Day |
|---|---|---|
| 1.1 | A1.1 — write `compare_pinn_heitler_london_n2.py`, run on existing N=2 d-sweep checkpoints. | 4 |
| 1.2 | A1.2 — already exists post-Phase-0 (`compare_pinn_ed_n2.py`), run on existing N=2 checkpoints. | 4 |
| 1.3 | A1.3 — `n4_ed_d_sweep.py` + `compare_pinn_ed_n4_d_sweep.py`. Decision gate **G1**: real or artefact for the d-sweep. | 5-7 |
| 1.4 | A1.4 — wire the d→∞ free-Slater baseline into the d-sweep comparison. | 8 |
| 1.5 | A1.5 — Schrieffer-Wolff cross-check at d ∈ {3, 4, 6, 8} for N=2, N=4. | 9-10 |
| 1.6 | A1.6, A1.7, A1.9 — automated assertions in trainer (Phase 0.1, 0.8 already lay the groundwork). | (continuous) |
| 1.7 | A1.8 — re-run N=8 SSH flagship at 2 additional seeds (existing config; 2 GPU-days). | 11-12 |

**Gate G1 (end of week 1):** all of A1.1, A1.2, A1.3 pass on existing static checkpoints OR we have an honest writeup of which fail and why. No softer outcome.

**Phase 1 deliverable:** `reports/benchmark_static_n2_n4_n8.md` listing every anchor with its actual measured value, threshold, and pass/fail. This file IS the validation evidence we cite in any later paper.

---

## 4. Phase 2 — Excited states with anchors (Weeks 3-5)

**Goal:** the excited-state NQS lane (Track C from previous roadmap) lands with rigorous validation, unblocking gap engineering.

### Anchor table

| Anchor ID | What we compare | Threshold | If it fails |
|---|---|---|---|
| **A2.1** | N=2 NQS singlet-triplet gap vs analytical Heitler-London J_HL | <2% at d ∈ {3, 4, 6, 8} | Orthogonality penalty broken. Re-derive. |
| **A2.2** | N=4 NQS first-excited E_1 vs CI-DVR ED first eigenvalue | <5% on gap E_1 - E_0 | Re-tune λ_ortho, check excited-state collapse. |
| **A2.3** | N=2 NQS triplet ⟨S·S⟩ = +¼ exactly | <2% | Wrong sector / mixing. |
| **A2.4** | Excited-state ⟨Sz⟩, ⟨N⟩ conserved | <1e-6 | Bug. |
| **A2.5** | E_1_NQS ≥ E_0_NQS (no level inversion) | strict ≥ | Bug. |
| **A2.6** | At N=4, NQS gap energy vs ED scales correctly with d (decreases as d increases for AFM Heisenberg-like regime) | qualitative match plus <10% at each d | Either NQS or ED has a problem; cross-check with second seed. |

**Phase 2 deliverable:** `reports/benchmark_excited_n2_n4.md` plus `src/wavefunction.py:ExcitedStateWF`, `scripts/run_two_stage_excited_state.py`. **Gate G2: A2.1 passes (the cleanest test).**

---

## 5. Phase 3 — Network geometry generalisation (Weeks 4-6)

**Goal:** generalise the chain/ring/SSH parametrisation to arbitrary 2D networks (graph topology). Required for both Option α and Option β.

### Anchor table

| Anchor ID | What we compare | Threshold | If it fails |
|---|---|---|---|
| **A3.1** | New graph factory at N=8 chain reproduces existing N=8 SSH flagship results bit-identically (same seed, same recipe) | <0.01% on all observables | Refactor introduced bug. |
| **A3.2** | 2-node graph reproduces N=2 d-sweep exactly | <0.01% | Bug. |
| **A3.3** | Triangular 3-site plaquette N=3 PINN GS vs CI-DVR ED | <2% on E, <5% on ⟨S·S⟩ matrix | Frustrated geometry breaks something in the recipe. |
| **A3.4** | 2×2 square lattice N=4 PINN GS vs CI-DVR ED | <1% on E, <5% on ⟨S·S⟩ matrix | Bug. |
| **A3.5** | Honeycomb 6-site PINN GS vs CI-DVR ED | <2% on E, <10% on ⟨S·S⟩ matrix | Bug or regime issue. |

**Phase 3 deliverable:** `src/parametrisations/network_factory.py`, `scripts/network_smoke_tests.py`, `reports/benchmark_networks.md`. **Gate G3: A3.1 + A3.3 both pass.**

---

## 6. Phase 4 — Static designer entanglement networks (Option α, Weeks 6-9)

**Goal:** demonstrate inverse-design of *non-local* entanglement structure on a frustrated network. Headline result if the rest of the plan slips.

### Anchor table

| Anchor ID | What we compare | Threshold | If it fails |
|---|---|---|---|
| **A4.1** | Engineered observable at N=4 vs ED for the optimised geometry | <5% relative | Optimiser found an unphysical region; revisit landscape. |
| **A4.2** | 3-seed reproducibility of the optimised geometry and observable | <10% spread | Optimisation noise dominates the engineered signal. Result is not robust enough to publish. |
| **A4.3** | At N=8: variance < 1e-3, ESS > 16, PINN E_GS below RHF baseline | hard | Underconvergence; re-tune training. |
| **A4.4** | Engineered observable trend (e.g., entanglement vs frustration angle) is monotonic / smooth | qualitative | If chaotic: the inverse-design landscape is too rugged; restart from multiple initial geometries. |
| **A4.5** | At the optimised geometry, ⟨S·S⟩ matrix qualitatively matches *some* known frustrated-spin-model prediction (e.g., 120° AFM order on triangular) | qualitative | Result is novel — write it up. |
| **A4.6** | Topological entanglement entropy proxy (Kitaev-Preskill bipartition) is non-zero and reproducible | "non-zero" within 3 seeds, sign consistent | Topological signature not robust. |

**Phase 4 deliverable:** `results/inverse_design/network_alpha_*/` with the engineered network state, `reports/option_alpha_designer_entanglement.md`. **Gate G4: A4.1 + A4.2 + A4.6 all pass.**

---

## 7. Phase 5 — Real-time NQS scaffolding (Weeks 8-12)

**Goal:** implement variance-stable real-time NQS evolution. **Highest-risk phase — most anchors, lowest priors.**

### Anchor table — these are the "always know" anchors that decide whether t-NQS is real

| Anchor ID | What we compare | Threshold | If it fails |
|---|---|---|---|
| **A5.1** | Energy conservation under static H over a unit time | \|ΔE\|/\|E\| < 1e-3 | Time stepping is wrong. Likely Suzuki-Trotter order issue or time-step instability. |
| **A5.2** | ⟨Sz⟩, ⟨N⟩ exact conservation under spin-conserving static H | <1e-6 | Bug. |
| **A5.3** | N=2 closed-form Rabi oscillation (small detuning) vs NQS | <2% on amplitude, <2% on frequency | Bug. |
| **A5.4** | N=2 quench vs ED-Trotter (Phase 0.5 anchor) over T = 5/Δ | <2% on every observable | Bug. |
| **A5.5** | N=4 quench vs ED-Trotter over T = 5/J | <5% on E, ⟨S·S⟩(t), entanglement(t) | Real-time NQS doesn't scale beyond N=2. Major problem. |
| **A5.6** | Trotter step Δt convergence: results converge as Δt → 0 (4 step sizes) | <5% spread between Δt and Δt/2 | Stiff problem; reduce Δt. |
| **A5.7** | Reversibility: forward T then backward T returns to initial state | <5% L2 distance from |ψ(0)⟩ | Phase-loss problem or stochastic error accumulation. |
| **A5.8** | At N=8: variance over the entire trajectory stays < 1e-2, ESS > 8 throughout | hard | Variance instability is the moat-collapse scenario. **If A5.8 fails, real-time at scale doesn't work; pivot to Option α as headline.** |
| **A5.9** | Lieb-Robinson cone width at N=8: information cone slope vs analytical bound | <2× the LR upper bound | Either non-physical superluminal info propagation (bug) or genuine continuum-vs-lattice effect. |

**Phase 5 deliverable:** `src/training/realtime_nqs.py`, `scripts/run_realtime_quench.py`, `reports/benchmark_realtime_n2_n4.md`. **Gate G5: A5.1, A5.4, A5.5, A5.8 all pass. If A5.8 fails, the grand goal is restricted to Option α.**

---

## 8. Phase 6 — Real-time entanglement transfer (Option β headline, Weeks 12-16)

**Goal:** inverse-designed control pulse moves a Bell pair across an 8-dot continuum network with ED-validated fidelity.

### Anchor table

| Anchor ID | What we compare | Threshold | If it fails |
|---|---|---|---|
| **A6.1** | At N=4: NQS-engineered protocol's terminal fidelity vs ED-Trotter prediction for the same (geometry, pulse) | <5% on terminal fidelity | The optimiser converged on a NQS artefact. Re-design loss. |
| **A6.2** | Adiabatic limit: a slow-enough pulse (T → ∞) hits the adiabatic limit; deviations follow Landau-Zener scaling | qualitative + <30% on Landau-Zener slope | Non-adiabatic processes dominate. Either physics insight (write it up) or bug. |
| **A6.3** | Trotter convergence in the engineered protocol | <5% spread across Δt | Bug. |
| **A6.4** | 3-seed reproducibility of the engineered (geometry, pulse) | <15% on terminal fidelity | Optimisation landscape too rugged; multi-start. |
| **A6.5** | Random-pulse counterfactual: random Fourier pulse with same time and bandwidth gives much lower fidelity | engineered fidelity > 3× random | If not: optimisation signal is in the noise. |
| **A6.6** | Lieb-Robinson upper bound on transfer time | engineered T ≥ T_LR_min | Strict — superluminal protocol = bug. |

**Phase 6 deliverable:** `results/inverse_design/network_beta_*/`, `reports/option_beta_realtime_transfer.md`. **Gate G6: A6.1 + A6.5 both pass.**

---

## 9. Phase 7 — Experimental connection (Weeks 16-20)

**Goal:** every result in Phases 4-6 is mapped to a specific experimental QD platform with realistic noise, and predicted experimental observables match published curves.

### Anchor table

| Anchor ID | What we compare | Threshold | If it fails |
|---|---|---|---|
| **A7.1** | Si MOS J(d) curve vs Watson et al. 2018, Camenzind et al. 2022 | OOM agreement (within factor 3) | Material parameters differ; revisit `materials.py` defaults. |
| **A7.2** | GaAs J(d) curve vs Petta 2005, Tarucha review | OOM agreement | Same. |
| **A7.3** | Charge-stability diagram (CSD) for 2-dot system: charge transition lines in correct positions | qualitative | Continuum picture vs Hubbard CSD differ in known ways; flag as physics insight if so. |
| **A7.4** | Predicted gate fidelity vs experimentally-published 2-qubit fidelities for Si MOS QDs | predicted upper bound > experimental fidelity | Either we're underestimating decoherence (likely) or experimental fidelity is sub-optimal. |
| **A7.5** | Fabrication tolerance σ_max (from Track E) consistent with published positional uncertainty | within factor 2 | Either we miss decoherence channels or there's a real opportunity in better fabrication. |

**Phase 7 deliverable:** `reports/experimental_bridge.md` with one paragraph per anchor, citing specific papers. **Gate G7: A7.1 OR A7.2 passes.**

---

## 10. Anchor matrix (one-page summary)

| Phase | Anchors | Key gates | Risk if grand-plan-blocking failure |
|---|---|---|---|
| 0 — infrastructure | 8 anchor builds (HL, ED-Trotter, SW, materials, etc.) | G0: unit tests pass | None (infrastructure only) |
| 1 — static validation | A1.1-A1.9 | G1: A1.1+A1.2+A1.3 pass | Major: invalidates existing claims, forces rewrite |
| 2 — excited states | A2.1-A2.6 | G2: A2.1 passes | Drop excited-state lane, use ED-only at N≤4 |
| 3 — networks | A3.1-A3.5 | G3: A3.1+A3.3 pass | Drop network generalisation, stick to chains |
| 4 — static designer (α) | A4.1-A4.6 | G4: A4.1+A4.2+A4.6 pass | Demote α to "preliminary"; rely on β/γ/δ |
| 5 — real-time NQS | A5.1-A5.9 | **G5: A5.1+A5.4+A5.5+A5.8 pass** | **Major: grand goal restricted to Option α; β/γ/δ blocked** |
| 6 — real-time transfer (β) | A6.1-A6.6 | G6: A6.1+A6.5 pass | β fails; α is headline. |
| 7 — experimental | A7.1-A7.5 | G7: A7.1 OR A7.2 passes | Paper missing the experimental bridge; lower-impact venue |

---

## 11. Time-ordered execution

```
Days 1-3       Phase 0 anchor infrastructure (Days 1-3)
Week 1         Phase 1 starts; G1 by EOD week 1
Week 2         Phase 1 continued; A1.4-A1.9
Week 3         Phase 1 wraps; Phase 2 starts; Phase 3 starts in parallel
Week 4         G2 (excited-state); G3 (network factory); Phase 4 starts
Week 5-7       Phase 4 (designer entanglement on networks); G4
Week 6         Phase 5 (real-time NQS) starts in parallel
Week 8-12      Phase 5 main effort; G5 around week 11-12
Week 12        If G5 passes: Phase 6 starts. Else: write up Option α and stop.
Week 13-16     Phase 6 (real-time transfer); G6
Week 17-20     Phase 7 (experimental bridge); G7
Week 21-22     Paper writing
```

Critical-path: **Phase 0 → A1.3 (G1) → real-time NQS (G5) → real-time transfer (G6).** Everything else is parallelisable.

---

## 12. Risk register at the anchor level

| Risk | Probability | Mitigation |
|---|---|---|
| **A1.3 (N=4 ED) reveals d-sweep is artefact** | 60% | Reframe as methodology paper. Existing 5-track roadmap captures this. |
| **A1.5 (Schrieffer-Wolff) reveals deeper PINN-Heisenberg disagreement** | 40% | Genuine physics insight; pivot d-sweep paper to "where lattice models fail in continuum QDs". |
| **A2.1 (excited-state gap) > 5%** | 30% | Tune λ_ortho upward; alternatively penalise overlap of *amplitudes* not just probability. |
| **A4.2 (multi-seed reproducibility) fails for designer networks** | 25% | The engineered observable is variational noise. Try simpler observables (dimerization > winding number); use more seeds for multi-start. |
| **A5.8 (real-time variance instability)** | **35%** | Pivot to Option α. The 5-month plan still produces a publishable Phase 4 result. |
| **A6.5 (random-pulse counterfactual) — engineered ≯ 3× random** | 30% | The optimiser landscape is too flat; expand pulse parametrisation; use Pareto multi-objective with explicit fidelity-vs-time. |
| **A7.x — experimental bridge fails for all platforms** | 20% | Drop to weaker claim ("relevant to the class of microscopic-Coulomb continuum QD models") and submit to a methodology venue instead. |

---

## 13. What this plan changes vs the previous roadmap

The 5-track roadmap (`reports/2026-04-28_research_roadmap.md`) is **subsumed** by Phases 0-4 of this plan, with two key changes:

1. **Track A is upgraded to a Phase 0 + Phase 1 prerequisite** for everything else. It's no longer "the d-sweep gating item"; it's the anchor for *every other* result we'll produce, because (a) the ED scaffold becomes the validation infrastructure for real-time, (b) the ⟨S·S⟩-on-ED extraction is reused throughout.
2. **Track G — real-time NQS — is added as Phase 5**, with a deeper anchor stack than the original sketch.

Tracks B (topological), D (Pareto), E (fabrication) remain as written but are now anchored:
- Track B's winding-number target now requires A4.6 (topological entanglement entropy reproducibility across seeds).
- Track D's Pareto frontier requires gap from Track C (Phase 2) which requires A2.2.
- Track E's fabrication tolerance requires the engineered network to first pass A4.1, A4.2, A4.6.

---

## 14. Right-now actions (today, before close of business)

In priority order:

1. **Phase 0.1 (today, 2 hours)**: add hard `assert ⟨Sz⟩ ≈ sector_Sz` and `assert ⟨N⟩ ≈ N_target` post-training in `src/training/vmc_colloc.py`. Run on all existing checkpoints to retroactively validate. (Cheap, high-value: any silent-failure checkpoints surface immediately.)

2. **Phase 0.2 (today, 4 hours)**: implement `compute_spin_correlators_ci` in `src/observables/exact_diag_reference.py`. Validate at N=2 (singlet → −¾, triplet → +¼). Unit test.

3. **Phase 0.3 + Phase 0.4 (today/tomorrow)**: write `compare_pinn_ed_n2.py` AND `heitler_london_n2.py`. **Run the N=2 anchor A1.1 + A1.2 on existing N=2 d-sweep checkpoints — we may already have the validation we need without a single new training run.**

4. **Phase 0.5 (tomorrow)**: implement `ed_trotter_evolve.py`. This is the validation infra for *every* downstream real-time result. Doing it now makes sure we don't reach Phase 5 and discover the anchor doesn't exist.

5. **Phase 1.3 (tomorrow afternoon)**: kick off `n4_ed_d_sweep.py` over d ∈ {2, 3, 4, 6, 8, 10, 14}. **Decision gate G1 by EOD tomorrow.**

Items 1-3 are all CPU-bound and don't compete with the seed-17 sweep on cuda:3.

---

## 15. Definition of success at each level

- **Strong success — full grand goal lands:** all Phase 7 anchors green. Headline paper: "Inverse-designed real-time entanglement transfer in continuum 2D QD networks", Nature Physics or PRX-Quantum.
- **Moderate success — Option α only:** Phases 0-4 + 7 anchors green; Phase 5/6 blocked. Headline paper: "Inverse-designed entanglement structures in continuum 2D QD networks", PRX or PRB-Letter.
- **Methodology success — Phase 1 reveals d-sweep is artefact:** clean methodology paper "Limits of variational ansätze in flat-correlation regimes for 2D continuum QDs", Phys. Rev. Research.
- **Failure mode — A1.3 fails AND A5.8 fails:** strong technical paper on the recipe (`self_residual` + `multi_ref=False`), framed as "Variance-stable training of large-N continuum NQS", Phys. Rev. E or J. Chem. Phys.

In every outcome, we have *something* publishable, and *every* result has rigorous anchors.

---

*Live document. Each anchor outcome will be appended below as it is measured.*
