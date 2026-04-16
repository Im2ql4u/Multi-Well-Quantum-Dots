# Journal

Research journal. Each entry documents an experiment, a significant result, or a meaningful shift in understanding. Written as if read by a technically capable person who has not been following the project.

Entries older than 8 are compressed into ## Earlier Experiments by session-close. Preserve conclusions, discard step-by-step detail.

---

## Format

```
### [YYYY-MM-DD] — <experiment title>
**Motivation:** <what question were we trying to answer>
**Method:** <what was done — concisely but precisely>
**Results:** <numbers, with units always>
**What the numbers actually mean:** <interpretation separate from the numbers>
**What we cannot explain:** <anomalies or uncertainties>
**Caveats:** <what might be wrong with this interpretation>
**What a skeptic would say:** <honest critique>
**Output reference:** results/YYYY-MM-DD_<n>/
**Next question:** <what this makes us want to investigate>
```

## Negative / Failed / Inconclusive format

```
### [YYYY-MM-DD] — NEGATIVE: <what was tried>
**Hypothesis tested:** <specific claim under test>
**Method:** <what was done>
**Expected result:** <what would have confirmed the hypothesis>
**Actual result:** <what actually happened>
**Why it failed:** <root cause, or best current understanding>
**What this rules out:** <directions this failure eliminates>
**What this does NOT rule out:** <what remains plausible>
**Severity:** dead-end | needs-rethink | minor-setback
**Lessons for future work:** <what to remember next time>
**Output reference:** results/YYYY-MM-DD_<n>/ or n/a
```

## Comparison format

Use this when 2+ experiments address the same question and a cross-run verdict is needed.

```
## Comparison: <question being answered>
Date: YYYY-MM-DD
Experiments compared: <entry refs>

| Dimension       | Experiment A | Experiment B | Experiment C |
|-----------------|--------------|--------------|--------------|
| Method          | <short>      | <short>      | <short>      |
| Key metric      | <value>      | <value>      | <value>      |
| Secondary metric| <value>      | <value>      | <value>      |
| Training cost   | <value>      | <value>      | <value>      |
| Failure modes   | <short>      | <short>      | <short>      |

**Winner and why:** <evidence-based verdict>
**What this does NOT settle:** <remaining uncertainty>
**What a skeptic would say:** <critique of the comparison itself>
**Recommended next experiment:** <next most informative step>
```

---

### [2026-04-10] — NEGATIVE: Corrected backflow/cusp plus FD-vs-autograd sweep did not close virial gap
**Hypothesis tested:** Fixing SD/backflow coupling and cusp-coordinate handling, plus trying autograd Laplacian, would materially reduce N=4 double-dot virial residual (target <10%, stretch <5%).
**Method:**
- Implemented structural fixes in wavefunction/correlator paths.
- Added Laplacian backend switch (`fd`/`autograd`) in training and virial diagnostics.
- Ran 8 parallel 6000-epoch trainings across GPUs 0–7: base, well-PINN, well-BF, both × FD/autograd training modes.
- Re-evaluated all runs with locked virial protocol (FD evaluator, MH steps=40, warmup batches=20).
**Expected result:** At least one corrected variant would push virial below 10% and show robust separation from baseline.
**Actual result:**
- Initial inconsistent evaluations showed ~38%–55% virial (later diagnosed as protocol artifact).
- Protocol-aligned re-evaluation produced ~12.73%–15.34% virial across all 8 runs.
- Best run: `wellpinn_fd` at ~12.73%; still above target.
**Why it failed:** Structural fix corrected a modeling inconsistency but did not remove the remaining virial bottleneck; additionally, inconsistent early evaluation obscured true behavior and created a false catastrophic-regression narrative.
**What this rules out:** "Simple SD/backflow target correction alone" as sufficient to close the virial gap.
**What this does NOT rule out:**
- modest but real gains around ~1% absolute virial from specific variants,
- deeper architecture/objective limitations,
- seed dependence of the corrected best variant.
**Severity:** needs-rethink
**Lessons for future work:**
- Never compare virial across runs without a fixed evaluator protocol.
- Treat low-MH quick diagnostics as triage only, not decision evidence.
- Require at least 2 seeds before promoting a variant decision.
**Output reference:** [results/p3fix_n4_dd_base_fd_s901_20260410_085510](results/p3fix_n4_dd_base_fd_s901_20260410_085510), [results/p3fix_n4_dd_wellpinn_fd_s901_20260410_085327](results/p3fix_n4_dd_wellpinn_fd_s901_20260410_085327), [results/p3fix_n4_dd_wellbf_fd_s901_20260410_085309](results/p3fix_n4_dd_wellbf_fd_s901_20260410_085309), [results/p3fix_n4_dd_both_fd_s901_20260410_085457](results/p3fix_n4_dd_both_fd_s901_20260410_085457), [results/p3fix_n4_dd_base_autograd_s901_20260410_085549](results/p3fix_n4_dd_base_autograd_s901_20260410_085549), [results/p3fix_n4_dd_wellpinn_autograd_s901_20260410_084755](results/p3fix_n4_dd_wellpinn_autograd_s901_20260410_084755), [results/p3fix_n4_dd_wellbf_autograd_s901_20260410_085704](results/p3fix_n4_dd_wellbf_autograd_s901_20260410_085704), [results/p3fix_n4_dd_both_autograd_s901_20260410_085915](results/p3fix_n4_dd_both_autograd_s901_20260410_085915)

### [2026-04-12] — Critical bug: missing spectator overlap in N≥3 exact diag CI matrix
**Motivation:** All NN approaches (PINN, CTNN, backflow, all hyperparameter sweeps) converged to identical N=3 energy (E≈3.634), supposedly 11% above diag reference (3.272). This universality was suspicious — if the NN were limited, different architectures should find different local optima.
**Method:**
- Ran CTNN backflow on SD coordinates for N=3 (1+1+1): 56,690 params, 6000 epochs. Result: E=3.634 identical to baseline.
- Printed CI Hamiltonian matrix from `run_exact_diagonalization_one_per_well_multi`. Found massive off-diagonal Coulomb couplings (0.13–0.33) between states differing in spectator orbitals.
- Root cause: when computing `<Ψ_i|V_{pq}|Ψ_j>` for pair (p,q), spectator particles k∉{p,q} were not checked for `orb_i[k] == orb_j[k]`. With orthonormal orbitals, nonmatching spectators give zero overlap — the code was computing a nonsensical sum.
- Fixed by adding `spectator_ok = all(orb_i[k] == orb_j[k] for k in range(n_wells) if k != p and k != q)`.
- N=2 code was unaffected (no spectators with exactly 2 wells).
**Results:**
- Fixed N=3 diag (localized, nx=22, n_ci=200): E0 = 3.637 (was 3.272 before fix)
- Fixed N=4 diag (localized, nx=18, n_ci=200): E0 = 5.105 (was ~4.3 before fix)
- N=3 VMC = 3.634 vs fixed diag 3.637 → **-0.08%** error (VMC slightly below CI)
- N=4 VMC = 5.100 vs fixed diag 5.105 → **-0.08%** error (consistent)
- N=2 diag unchanged at E0 = 2.254 (no spectators, not affected by bug)
**What the numbers actually mean:** The VMC was working correctly all along. The "11% gap" was entirely a diag bug. The PINN correlator + single-Gaussian SD captures slightly more correlation than the finite CI expansion, which is why VMC is fractionally below CI.
**What we cannot explain:** VMC gives E ≈ 0.003 below the CI-diag for both N=3 and N=4. This is either: (a) PINN captures more dynamic correlation than the truncated CI, or (b) a small Coulomb softening mismatch (VMC eps=0.01/√ω, diag epsilon=0.01 directly).
**Caveats:** CI convergence at n_sp=20/n_ci=200 not rigorously verified beyond grid-size checks. The small VMC-below-CI discrepancy could indicate CI is not fully converged.
**What a skeptic would say:** The VMC values being slightly below diag is suspicious — variational principle says VMC should be above exact. Either the CI is not converged, or there's a Hamiltonian mismatch.
**Output reference:** [scripts/exact_diag_double_dot.py](scripts/exact_diag_double_dot.py) (fix at line 370), commit `9d5b57f`; [results/mcmc_training/p4_n3_ctnn_bf_reinforce_s42_20260412_100358](results/mcmc_training/p4_n3_ctnn_bf_reinforce_s42_20260412_100358)
**Next question:** Resolve VMC-below-CI puzzle (check epsilon parity). Then proceed to magnetic quench time evolution for N=3/N=4.

## Comparison: One-per-well ground state across N=2, N=3, N=4
Date: 2026-04-12
Experiments compared: Phase 2 N=2 CTNN, Phase 4 N=3 baseline, Phase 4 N=4 GS

| Dimension          | N=2 (1+1)       | N=3 (1+1+1)     | N=4 (1+1+1+1)   |
|--------------------|------------------|------------------|------------------|
| VMC energy         | 2.237 (κ=0.7)   | 3.634 (κ=1.0)   | 5.100 (κ=1.0)   |
| Diag CI energy     | 2.179 (κ=0.7)   | 3.637 (κ=1.0)   | 5.105 (κ=1.0)   |
| Diag product-state | 2.260 (κ=1.0)   | 3.648 (κ=1.0)   | 5.121 (κ=1.0)   |
| VMC vs CI          | ~2.7% (κ=0.7)   | -0.08%           | -0.08%           |
| Architecture       | CTNN+backflow    | PINN (no BF)     | PINN (no BF)     |
| Epochs             | 30,000           | 6,000            | 1,100            |
| Well separation    | 4.0              | 4.0              | 4.0              |

**Winner and why:** All VMC runs match or beat their respective CI references. N=3 and N=4 are validated at <0.1% of the corrected diag; N=2 comparison at κ=0.7 shows 2.7% gap which may improve with κ parity alignment.
**What this does NOT settle:** (1) N=2 needs rerunning at κ=1.0 for apples-to-apples comparison. (2) VMC-below-CI puzzle needs epsilon-parity check.
**What a skeptic would say:** N=4 ran only 1100 epochs — is it converged? And the κ mismatch between N=2 and N=3/N=4 makes cross-N comparison unreliable.
**Recommended next experiment:** (1) Run N=2 at κ=1.0 for parity. (2) Check epsilon parity between VMC and diag. (3) Proceed to magnetic quench.

### [2026-04-13] — Non-MCMC residual/collocation training validated for N=2, N=3, N=4
**Motivation:** Validate that training can be done without MCMC and still hit exact-diag-quality energies, with MCMC reserved only for optional post-training validation.
**Method:**
- Root-cause diagnosis for non-MCMC instability used a Layer 1/2 check: known-input FD local-energy test first (analytic Gaussian), then training-branch audit.
- Added robust per-batch local-energy MAD clipping in all training loss branches and enabled it in non-MCMC configs.
- Ran three stratified i.i.d. non-MCMC training jobs with residual objective and fixed exact-diag targets.
**Results:**
- N=2 (target 2.25442431): final `E=2.253998`, relative error `0.019%`.
- N=3 (target 3.63700158): final `E=3.636260`, relative error `0.020%`.
- N=4 (target 5.10444947): final `E=5.103606`, relative error `0.017%`.
- Variance remained low and finite through full training for all runs; no divergence after clipping was introduced.
**What the numbers actually mean:** Non-MCMC training is now numerically stable and reaches the same quality regime as prior MCMC-trained models for one-per-well ground states.
**What we cannot explain:** Whether this remains stable under stronger magnetic/quench settings without retuning clip width and sample budget.
**Caveats:** These are single-seed runs with fixed architecture; this validates viability, not final production optimum.
**What a skeptic would say:** The gain may come from robust clipping rather than intrinsic sampler quality; multi-seed evidence is still needed.
**Output reference:** [results/nonmcmc_training/p4_n2_nonmcmc_residual_anneal_s42_20260412_232259](results/nonmcmc_training/p4_n2_nonmcmc_residual_anneal_s42_20260412_232259), [results/nonmcmc_training/p4_n3_nonmcmc_residual_anneal_s42_20260413_001421](results/nonmcmc_training/p4_n3_nonmcmc_residual_anneal_s42_20260413_001421), [results/nonmcmc_training/p4_n4_nonmcmc_residual_anneal_s42_20260413_001824](results/nonmcmc_training/p4_n4_nonmcmc_residual_anneal_s42_20260413_001824)
**Next question:** For N=4+, should the next robustness budget go to `n_coll` scaling or to deeper architecture under the same non-MCMC sampler?

### [2026-04-16] — Singlet permanent ansatz: separation sweep d={2,4,6,8,12,20} with Löwdin entanglement
**Motivation:** Does the singlet permanent ansatz φ_L(r1)φ_R(r2)+φ_R(r1)φ_L(r2) correctly learn the LR singlet state across a range of dot separations? Does the entanglement (dot-label negativity) track the CI reference?
**Method:**
- Trained singlet permanent ansatz (log-domain, PINN correlator) for N=2 at d=2,4,6,8,12,20 using stratified non-MCMC training with 3000 epochs, residual objective, shared-CI target energies.
- Measured L/R sector probabilities and von Neumann entropy using new Löwdin S^{-1/2} basis; measured dot-label negativity from projected 2×2 reduced density matrix.
**Results:**
- d=2: E=1.6464 (ΔE=−0.65% vs CI), S_vN=0.593, neg=0.000, LL=25.3%, LR=24.6%, proj_w=0.982
- d=4: E=1.8812 (ΔE=+0.20%), S_vN=0.944, neg=0.000, LL=20.4%, LR=31.0%, proj_w=0.950
- d=6: E=1.8354, S_vN=0.768, neg=0.288, LL=0.0%, LR=50.0%, proj_w=0.941
- d=8: E=1.7609 (ΔE=+0.05%), S_vN=0.762, neg=0.264, LL=0.0%, LR=50.0%, proj_w=0.954
- d=12: E=1.7566, S_vN=0.759, neg=0.299, LL=0.0%, LR=50.0%, proj_w=0.972
- d=20: E=1.7498, S_vN=0.760, neg=0.289, LL=0.0%, LR=50.0%, proj_w=0.978
**What the numbers actually mean:** For d≥6 the network converges to the correct LR singlet sector (LL=RR=0%, LR=RL=50%) with energies within 0.05% of the CI reference. For d≤4 the sector structure is wrong (LL+RR≈50%), which is a structural property of the permanent ansatz with non-orthogonal raw HO orbitals: at d=2 the orbital overlap is S_LR=exp(−1)≈0.37, which means the permanent after Löwdin projection inherently contains ~50% double-occupancy amplitude regardless of training.
**What we cannot explain:** dot-label negativity is 0.26–0.30 at d≥6, substantially below the pure singlet expectation of 0.50. This discrepancy is not explained: it could be (a) PINN correlator adds intra-well correlations that dilute the 2×2 density matrix, (b) ~5% of norm outside the Löwdin basis is lost, or (c) the measurement itself is wrong because it has not been validated against the CI reference wavefunction.
**Caveats:** Single seed only. Measurement not calibrated against CI reference wavefunction in the same Löwdin basis.
**What a skeptic would say:** Energy convergence is clean but the entanglement metric has never been shown to be correct even for a known input. Until the measurement is validated on the CI ground state, the negativity numbers are uninterpretable.
**Output reference:** [results/diag_sweeps/singlet_entanglement_d{2,4,6,8,12,20}_lowdin_s42.json](results/diag_sweeps/), [results/p4_n2_singlet_d*](results/)
**Next question:** Compute Löwdin-basis entanglement for the CI wavefunction at d=8 to calibrate. If CI gives neg≈0.50, the deficit is entirely in the ansatz. If CI also gives neg<0.50, the measurement underestimates due to basis truncation.

### [2026-04-13] — Converged 3-seed non-MCMC benchmark with finite-basis-aware reporting
**Motivation:** Convert single-seed non-MCMC success into multi-seed evidence and remove misleading interpretation when model energy falls below finite-basis CI reference.
**Method:**
- Completed full seed sweep for N=2/N=3/N=4 with seeds 42, 314, 901 using non-MCMC residual anneal configs.
- Aggregated latest converged runs into a unified report JSON and generated three figures (energy vs CI, one-sided exceedance bars, seed spread).
- Used one-sided exceedance metric: `max(E_model - E_diag, 0)` as primary benchmark error.
**Results:**
- N2: mean relative exceedance `0.000000%`, max `0.000000%`.
- N3: mean relative exceedance `0.002257%`, max `0.006771%`.
- N4: mean relative exceedance `0.000000%`, max `0.000000%`.
- Final artifacts: [results/nonmcmc_diag_seed_sweep_final_20260413.json](results/nonmcmc_diag_seed_sweep_final_20260413.json), [results/nonmcmc_diag_seed_sweep_final_20260413_report.md](results/nonmcmc_diag_seed_sweep_final_20260413_report.md), figures in [results/figures](results/figures).
**What the numbers actually mean:** Across three seeds, non-MCMC training is stable and generally does not exceed the finite-basis CI reference; where it does, exceedance is tiny.
**What we cannot explain:** Whether below-reference energies are entirely CI truncation artifacts or partially reflect remaining setup mismatch.
**Caveats:** CI reference remains finite-basis/truncated (`n_sp_states`, `n_ci_compute`); these results are high-confidence training evidence but not proof of physical overperformance.
**What a skeptic would say:** Without a CI convergence ladder, below-reference energies should be treated as inconclusive, not as confirmed improvements.
**Output reference:** [results/nonmcmc_diag_seed_sweep_final_20260413.json](results/nonmcmc_diag_seed_sweep_final_20260413.json)
**Next question:** How fast does N4 CI ground energy stabilize as `n_sp_states` and `n_ci_compute` are increased under fixed Hamiltonian settings?
