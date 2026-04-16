# Archive

Compressed session history. Maintained by session-close. When this file exceeds 10 entries, the oldest 5 are compressed into ## Older History.

Read the last 2–3 entries during session open. Read the full file only when reconstructing history.

---

## Format

```
## [YYYY-MM-DD] — <session title>

### Technical summary
[what was done, what was concluded, what is open]

### Session metrics
- Steps completed: <n of m planned>
- Material deviations: <count>
- Evaluation gates triggered: <count + verdict>
- Unresolved uncertainties: <count>

### Human reflection
**Understood this session:** ...
**Still unclear:** ...
**Skeptic's view:** ...
**Would do differently:** ...

---
```

---

## [2026-04-10] — Virial Gap Corrective Sweep and Evaluation Protocol Diagnosis

### Technical summary
- Central goal: diagnose why corrected backflow/cusp changes appeared to catastrophically worsen virial and determine whether the failure was architectural or evaluational.
- Accomplished: implemented structural fixes (SD evaluated on backflowed coordinates, cusp on physical coordinates), added FD/autograd Laplacian backend support, launched and completed 8-GPU factorial sweep, and re-evaluated all runs under consistent MH virial settings.
- Attempted but failed: initial virial reads (38%–55%) suggested severe regression; this did not hold after protocol-aligned re-evaluation and was traced to inconsistent evaluation settings (short MH warmup/steps and mismatched evaluator configuration).
- Decisions made: use Full Close evidence standard, lock virial comparison protocol for fair cross-run ranking (FD evaluator, MH steps=40, warmup batches=20) before drawing conclusions.
- Workarounds in place: evaluator compatibility shim for legacy checkpoints missing `backflow.w_intra`/`backflow.w_inter` keys.
- Unverified: seed-to-seed robustness for corrected best variant (only seed 901 evaluated in corrected sweep).
- Skeptic view: current evidence shows no progress toward <10%, and no claim of improvement is credible without fixed protocol plus 2-seed confirmation.
- Single most important carry-forward: the "catastrophic regression" diagnosis was mostly an evaluation artifact; corrected models remain in the ~13%–15% virial regime.
- Recommended next action: run a 2-seed confirm on corrected best variant with locked virial protocol, then decide whether to continue architecture exploration or pivot.

### Human reflection
**Understood this session:** not provided at close (session closed on user request).
**Still unclear:** not provided at close (session closed on user request).
**Skeptic's view:** not provided at close (session closed on user request).
**Would do differently:** not provided at close (session closed on user request).

---

## [2026-04-13] — Non-MCMC Training Stabilization and Dual-Lane Session Close

### Technical summary
- Central goal: establish non-MCMC residual/collocation training as a stable path and close the session with explicit separation between MCMC and non-MCMC evidence.
- Accomplished: completed non-MCMC runs for N=2/N=3/N=4 with robust clipping and low variance, then created explicit result lanes to separate MCMC-trained outputs from non-MCMC-trained outputs.
- Run outcomes:
	- N=2 final `E=2.253998` vs diag `2.25442431` (0.019% error).
	- N=3 final `E=3.636260` vs diag `3.63700158` (0.020% error).
	- N=4 final `E=5.103606` vs diag `5.10444947` (0.017% error).
- Attempted but failed earlier in this direction: non-MCMC without local-energy clipping produced unstable variance and gradient domination by outliers.
- Decisions made:
	- Use non-MCMC as the main training-development lane.
	- Keep MCMC as a separate validation/baseline lane.
	- Enforce lane distinction in result storage (`results/nonmcmc_training/` vs `results/mcmc_training/`).
- Workarounds in place: none active; robust clipping is now treated as required stabilization, not temporary workaround.
- Unverified: multi-seed robustness for N=3/N=4 under this exact non-MCMC setup is not yet established.
- Skeptic view: single-seed success can still hide regime fragility; stronger fields/quench objectives may require retuning `n_coll` and clipping parameters.
- Single most important carry-forward: architecture is not the blocker; training regime separation and stable non-MCMC optimization are now the core direction.
- Recommended next action: run 2-seed non-MCMC confirmations for N=3 and N=4 in the non-MCMC lane before extending to magnetic-quench workflows.

### Session metrics
- Steps completed: 5 of 5 planned (N=2 completion, N=3/4 references, N=3/4 configs, N=3/4 runs, close consolidation)
- Material deviations: 1 (added lane-structure migration during close to preserve methodological clarity)
- Evaluation gates triggered: 1 (stability gate: known-input + post-fix full-run evidence)
- Unresolved uncertainties: 1 (multi-seed robustness for non-MCMC N=3/N=4)

### Human reflection
**Understood this session:** that the architectures seemingly are capable, but that our results so far are only mcmc training
**Still unclear:** no
**Skeptic's view:** that our results so far are only mcmc training
**Would do differently:** residual/collocation based training

---

## [2026-04-13] — Quick Close: Non-MCMC Artifact Review

### Technical summary
- Goal: keep advancing one-per-well ground-state development with non-MCMC training as the main path, while keeping MCMC results distinct as a validation lane.
- Reviewed the new non-MCMC result artifacts rather than only the console logs.
- Confirmed the good runs are numerically strong: N=2/N=3/N=4 non-MCMC annealed runs remain within about 0.02% of their exact-diag targets.
- Confirmed the implementation path that made non-MCMC work: stratified i.i.d. sampling, residual/collocation objective, fixed exact-diag targets, and per-batch MAD clipping of local-energy outliers.
- Errors found in the saved artifacts:
	- `reference_energy.resolved` is wrong in the successful result files and reflects the auto fallback, not the exact-diag residual target.
	- `final_energy` is the clipped training-time statistic from the residual branch, not an independent post-training evaluation metric.
	- Failed bring-up runs and accepted runs are still mixed in the same non-MCMC lane with no explicit success/failure marker.
- What remains open: the runs are good evidence that non-MCMC training can work, but the artifact format is still not trustworthy enough for downstream reporting without fixes.
- Recommended next action: fix result serialization first, then add a post-training evaluation pass, then run multi-seed non-MCMC confirmations.

### Session metrics
- Steps completed: 3 of 3 planned (artifact review, issue identification, close update)
- Material deviations: 0
- Evaluation gates triggered: 1 (artifact-integrity review found metadata issues)
- Unresolved uncertainties: 1 (single-seed non-MCMC robustness still unverified)

### Human reflection
**Understood this session:** not provided at close (quick close on review note).
**Still unclear:** not provided at close (quick close on review note).
**Skeptic's view:** not provided at close (quick close on review note).
**Would do differently:** not provided at close (quick close on review note).

---

## [2026-04-13] — Converged 3-Seed Non-MCMC Benchmark, Reporting Update, and Cleanup

### Technical summary
- Central goal: complete a fully converged 3-seed non-MCMC benchmark for N=2/N=3/N=4, produce decision-grade reporting artifacts, and reduce repository clutter from generated outputs.
- Accomplished: ran and completed all missing seed-sweep jobs (s314/s901) to final epochs for N=2/N=3/N=4; aggregated latest converged runs into a single final report JSON.
- Accomplished: generated report visuals and a written benchmark note using a one-sided exceedance metric relative to finite-basis diagonalization (count only E_model > E_diag as exceedance).
- Accomplished: cleaned untracked generated artifacts under results while preserving the final benchmark deliverables (final JSON, report markdown, and three figures).
- Accomplished: reviewed changed core training/sampling code paths and validated with tests (`23 passed, 6 skipped`).
- Attempted but failed: initial N4 s901 launch on cuda:5 failed with CUDA OOM due to another user process; rerun on cuda:7 completed.
- Decisions made: finite-basis CI reference is treated as approximate in reporting, and below-reference model energies are not auto-labeled as model error.
- Workarounds in place: seed-sweep config for N4 s901 is pinned to cuda:7 to bypass external GPU contention.
- Unverified: whether small below-reference margins are due to CI truncation only, or include residual Hamiltonian/parity mismatch effects.
- Skeptic view: despite convergence, reference-quality claims still require a tighter CI convergence check (larger `n_sp_states` / `n_ci_compute`) before treating below-reference outcomes as physical improvement.
- Single most important carry-forward: non-MCMC training is stable across three seeds for N=2/N=3/N=4, but interpretation quality is bounded by finite-basis diagonalization fidelity.
- Recommended next action: run a focused CI convergence ladder for N4 (`n_sp_states`, `n_ci_compute`) and verify monotone stabilization of E_diag before promoting sub-CI model energies as improvements.

### Session metrics
- Steps completed: 4 of 4 planned (complete runs, aggregate final report, generate figures/report note, cleanup)
- Material deviations: 2 (GPU reassignment for N4 s901, one-sided reporting metric replacing symmetric error framing)
- Evaluation gates triggered: 2 (full-convergence gate passed; artifact/interpretation gate passed with finite-basis caveat)
- Unresolved uncertainties: 1 (reference convergence adequacy of finite-basis CI)

### Human reflection
**Understood this session:** that non-mcmc works well
**Still unclear:** whether the results are 100% legit
**Skeptic's view:** not sure
**Would do differently:** not much

---

## [2026-04-16] — Löwdin Sector Fix, Singlet Separation Sweep, and Entanglement Measurement

### Technical summary
- Central goal: replace broken Voronoi-masked sector analysis with Löwdin S^{-1/2} orthogonalization, compute correct shared-CI reference energies for d=6/12/20, train singlet permanent ansatz across full separation sweep, and measure entanglement on all checkpoints.
- Accomplished: implemented Löwdin basis (global HO evaluation + symmetric orthogonalization) in `src/observables/entanglement.py`; verified Gram orthonormality <3e-15 at d=2 and d=4 (high-overlap regime). Computed shared-CI reference energies with correct kinetic_prefactor=0.5 and dx≈0.8 bohr box scaling: d=6 E₀=1.83320031, d=12 E₀=1.75531218, d=20 E₀=1.74902235. Created and launched singlet configs for all six separations. Ran Löwdin entanglement measurement on all six checkpoints.
- Attempted but failed (fixed): (1) `exact_diag_double_dot.py --model shared` path was missing `kinetic_prefactor=0.5` multiplication on t2d, giving wrong CI energies (d=20 gave E₀=0.64). Fixed by computing directly via Python API. (2) d=20 with nx=20 gave dx=1.4 bohr (too coarse), causing sp_e[0]=0.988 instead of 1.0. Fixed: nx scales with d to maintain dx≈0.8 bohr. (3) WellSpec API required positional `n_particles` arg — fixed in smoke test. (4) d=6 run first launched with no logs/ directory — re-launched after mkdir.
- Decisions made: Löwdin S^{-1/2} is now the canonical sector decomposition method; Voronoi masking is deprecated for all separation values.
- Workarounds in place: (1) singlet permanent ansatz uses raw (non-Löwdin) HO orbitals — structurally broken at d≤4 because permanent with non-orthogonal orbitals produces LL+RR≈50% regardless of PINN training. (2) dot_label_negativity ≈0.26–0.30 vs CI reference 0.50 at d=8 — multi-mode PINN correlator dilutes the 2×2 density matrix beyond pure singlet.
- Unverified: whether dot-label negativity discrepancy at d≥6 is an ansatz limitation or Löwdin basis incompleteness (~5% of norm outside basis). Whether d=2/4 could be fixed by using Löwdin-orthogonalized permanent orbitals in the ansatz.
- Skeptic view: d≥6 results correctly recover LR singlet sector structure and energies within 0.05%, but the entanglement metric (negativity=0.26–0.30) is substantially below what a pure singlet would give (0.50). This underestimate has not been explained and could indicate that the ansatz or the measurement is wrong.
- Single most important carry-forward: the Löwdin sector analysis gives correct L/R decomposition, but the entanglement quantity itself is still not calibrated against a known pure singlet — this is the outstanding validation gap.
- Recommended next action: compute Löwdin-basis entanglement for the CI reference wavefunction itself (not just network) to calibrate the measurement and establish whether 0.26–0.30 is a measurement artifact or a genuine ansatz deficit.

### Human reflection
**Understood this session:** "we had some issues"
**Still unclear:** "whether the fixes are legit"
**Skeptic's view:** "not sure"
**Would do differently:** "i dont know"

### Session metrics
- Steps completed: 4 of 4 planned (Löwdin fix, CI energies, singlet configs+runs, entanglement measurement)
- Material deviations: 2 (exact_diag kinetic_prefactor bug required Python API workaround; nx scaling fix for large d)
- Evaluation gates triggered: 2 (Gram orthonormality check passed; sector structure passes for d≥6 but not d≤4)
- Unresolved uncertainties: 3 (negativity underestimate, d≤4 permanent ansatz structural limitation, measurement not calibrated against CI reference)

---

## [2026-04-15] — Entanglement Reference Audit and Non-MCMC Variational Estimator Fix

### Technical summary
- Goal: determine whether the CI reference or network entanglement measurement was wrong, then implement a proper non-MCMC variational path if needed.
- Accomplished: validated the shared-CI sector structure, showed one-per-well CI was the wrong reference for the d=2/d=4/d=8 entanglement question, diagnosed symmetry-broken network states under shared-CI targeting, and implemented a correct fixed-proposal non-MCMC variational estimator for `sampler: is` while blocking invalid direct-energy use with `sampler: stratified`.
- Not done / blocked: the corrected non-MCMC variational estimator is numerically stable but still does not reach the shared-CI singlet manifold; d8 remains around `E≈2.11` instead of the shared CI `1.760`.
- Recommended next action: test whether a wider or better-matched importance proposal changes the variational endpoint before concluding the ansatz itself is the dominant limitation.

### Human reflection
**Carry forward:** that we have multiple atrange issues
**Still uncertain:** the entire thing

### Session metrics
- Steps completed: 6 of 7 planned
- Material deviations: 3
- Evaluation gates triggered: 3 (reference audit passed, measurement audit passed, estimator validation passed with performance caveat)
- Unresolved uncertainties: 3

---
