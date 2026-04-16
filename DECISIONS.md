# Decisions

Permanent, append-only record of architectural and methodological decisions. Never delete or rewrite. If a decision was reversed, add a new entry explaining why.

Only write entries for genuine decisions. Not every small implementation choice. Quality over completeness.

---

## Format

```
### [YYYY-MM-DD] — <short title>
**Decision:** <what was chosen>
**Alternatives considered:** <what else was on the table>
**Reasoning:** <why this, not the alternatives>
**Constraints introduced:** <what this makes harder going forward>
**Confidence:** high / medium / low
```

---

### [2026-04-10] — Lock Virial Comparison Protocol
**Decision:** Use a fixed virial evaluation protocol for cross-run comparisons: FD Laplacian evaluator, MH sampler with `mh_steps=40`, `mh_warmup_batches=20`, and matched sample budgets.
**Alternatives considered:** Compare runs using mixed evaluator settings (different MH depth/warmup and/or autograd evaluator) based on whichever command was most recent.
**Reasoning:** Mixed evaluation settings produced large apparent regressions (up to ~55% virial) that disappeared under protocol-aligned re-evaluation (~13%–15%), so fair ranking requires fixed evaluation conditions.
**Constraints introduced:** Adds evaluation cost; quick low-MH diagnostics are no longer acceptable for decision-grade claims.
**Confidence:** high

## Negative Memory

### [2026-04-10] — FAILED: Treating early p3fix virial as catastrophic regression
**What:** Interpreted initial corrected-run virial outputs (~38%–55%) as direct evidence that the structural backflow/cusp fix broke the model.
**Why it failed:** Those numbers were produced under a different and weaker evaluation setup (short MH schedule / inconsistent evaluator settings), making them non-comparable to prior baselines.
**Evidence:** Re-evaluating old baseline and corrected runs with matched settings (`mh_steps=40`, warmup 20, FD evaluator) yielded ~13%–16% across runs, not ~38%–55%.
**What to do instead:** Freeze evaluation protocol before any cross-run claim and rerun diagnostics under identical settings.
**Severity:** needs-rethink

### [2026-04-10] — FAILED: Corrective architecture sweep to reduce virial below 10%
**What:** 8-run factorial sweep (FD/autograd training × base/well-PINN/well-BF/both) after correcting SD/backflow and cusp-coordinate handling.
**Why it failed:** None of the corrected variants achieved the plan gate; all remained in the ~12.7%–15.3% virial range under fair FD evaluation.
**Evidence:** `scripts/run_virial_check.py` summary on p3fix runs with locked protocol showed best case 12.73% (wellpinn_fd), worst 15.34% (both_autograd).
**What to do instead:** Perform 2-seed confirmation on best corrected variant and then revisit architecture assumptions or objective design.
**Severity:** minor-setback

### [2026-04-12] — N≥3 exact diag was wrong due to missing spectator overlap
**Decision:** Fixed `run_exact_diagonalization_one_per_well_multi` in `scripts/exact_diag_double_dot.py` to enforce spectator orbital orthogonality in the CI Hamiltonian. All N≥3 diag results before commit `9d5b57f` are invalid.
**Alternatives considered:** None — this was a clear bug, not a design choice.
**Reasoning:** When computing `<Ψ_i|V_{pq}|Ψ_j>` for a two-body pair (p,q) in an N-particle product basis, particles k∉{p,q} must satisfy `orb_i[k] == orb_j[k]` (spectator overlap). The code was missing this check, injecting spurious off-diagonal Coulomb coupling. N=2 was unaffected (no spectators). N=3 CI energy was 3.272 (wrong) vs 3.637 (correct). This explains the entire "11% gap" that drove multiple sessions of architecture/hyperparameter investigation.
**Constraints introduced:** Must re-derive any N≥3 diag reference from scratch using post-fix code.
**Confidence:** high

## Negative Memory

### [2026-04-12] — FAILED: All N=3 architecture/hyperparameter investigations were chasing a phantom gap
**What:** Ran CTNN backflow, MH sweep (10→40 steps), decorrelation sweep, batch size sweep, LR sweep, loss-type ablation — none improved N=3 energy beyond 3.634.
**Why it failed:** The "11% gap" was due to a bug in the diag code, not a limitation of the NN. The VMC was actually performing correctly all along.
**Evidence:** After fixing the diag spectator overlap, corrected N=3 diag = 3.637 vs VMC = 3.634 (VMC is -0.08% vs CI). All sweep variants gave identical energy because the true gap is <0.4%.
**What to do instead:** Always verify the reference before blaming the model. Cross-check diag with known limits (kappa=0 → N×ω, large sep → sum of point charges).
**Severity:** needs-rethink

## Decisions (continued)

### [2026-04-13] — Split result lanes by training regime (MCMC vs non-MCMC)
**Decision:** Maintain two explicit result lanes: `results/mcmc_training/` for MCMC-trained runs and `results/nonmcmc_training/` for i.i.d. stratified residual/collocation training runs.
**Alternatives considered:** Keep mixed run folders under `results/` with naming-only separation.
**Reasoning:** Mixed folders caused ambiguity about whether a result supports non-MCMC training viability or only MCMC-based training. Lane separation makes interpretation and next-session planning unambiguous.
**Constraints introduced:** Existing analysis scripts and references must use lane-aware paths; future runs should be moved or written into the correct lane.
**Confidence:** high

### [2026-04-13] — Non-MCMC as primary training direction, MCMC as validation lane
**Decision:** Continue development primarily on non-MCMC residual/collocation training; keep MCMC runs as a distinct baseline/validation track.
**Alternatives considered:** Continue training primarily with MCMC and treat non-MCMC as optional.
**Reasoning:** After robust clipping fix, non-MCMC runs reached exact-diag-level accuracy for N=2/N=3/N=4 while satisfying the project direction to avoid MCMC during optimization.
**Constraints introduced:** Non-MCMC stability controls (clip width, sample budget, diagnostic checks) become mandatory for new configs.
**Confidence:** high

### [2026-04-13] — Treat finite-basis diagonalization as approximate reference in benchmark reporting
**Decision:** For benchmark summaries against CI diagonalization, use one-sided exceedance (`max(E_model - E_diag, 0)`) as the primary error metric and report symmetric deltas separately for transparency.
**Alternatives considered:** Treat absolute/symmetric deviation from CI as model error regardless of sign.
**Reasoning:** CI reference here is finite-basis/truncated (`n_sp_states`, `n_ci_compute`), so `E_model < E_diag` can reflect CI under-convergence rather than model failure.
**Constraints introduced:** Any claim that model energies "beat reference" now requires explicit CI-convergence evidence before being promoted as physical improvement.
**Confidence:** medium

## Negative Memory (continued)

### [2026-04-13] — FAILED: Raw non-MCMC residual training without local-energy clipping
**What:** Early non-MCMC residual/collocation runs were executed without robust clipping of local-energy outliers.
**Why it failed:** i.i.d. stratified batches produced heavy-tail local-energy samples that dominated gradients and caused unstable variance/exploding dynamics.
**Evidence:** Pre-fix runs showed high and erratic `e_var` (up to O(10^2-10^4)) and large energy drift; post-fix clipped runs converged stably to <0.03% error vs exact diag for N=2/N=3/N=4.
**What to do instead:** Apply per-batch MAD clipping before loss construction in all non-MCMC branches and keep known-input FD operator checks in the bring-up flow.
**Severity:** needs-rethink

### [2026-04-13] — FAILED: Interpreting sub-CI energies as automatic model error
**What:** Used symmetric error framing where energies below finite-basis CI reference were treated as model/reporting errors.
**Why it failed:** CI reference is truncated-basis and not guaranteed converged; sub-CI outcomes may indicate reference bias instead of model pathology.
**Evidence:** In the converged 3-seed N2/N3/N4 sweep, most runs landed slightly below CI while maintaining stable training diagnostics and low seed spread.
**What to do instead:** Use one-sided exceedance as the primary benchmark metric and require explicit CI convergence ladders before classifying below-reference energies.
**Severity:** needs-rethink

## Decisions (continued)

### [2026-04-16] — Löwdin S^{-1/2} as canonical sector decomposition (replace Voronoi masking)
**Decision:** All L/R sector analysis uses global Löwdin-orthogonalized HO basis. HO functions are evaluated at all quadrature points without Voronoi masking; the full cross-well overlap matrix S is symmetrically orthogonalized via S^{-1/2}. Voronoi-masked QR is deprecated.
**Alternatives considered:** Voronoi masking + QR (old approach); natural orbital decomposition from 1-RDM.
**Reasoning:** At small separations (d≤8), HO orbital tails extend deep into the other well's Voronoi region, so masked QR discards amplitude that physically belongs to that orbital. Löwdin evaluates everything globally and lets the linear algebra handle cross-well orthogonality — Gram error <3e-15 verified. Natural orbitals would require computing the full 1-RDM from the network, which is expensive and harder to interpret in a fixed L/R assignment.
**Constraints introduced:** Basis quality depends on max_ho_shell parameter. At d=2, shell=2 captures 98.2% of norm; shell=4 recommended for precision measurements. Computationally heavier than Voronoi.
**Confidence:** high

## Negative Memory (continued)

### [2026-04-16] — FAILED: exact_diag_double_dot.py --model shared path for CI energies
**What:** Used `run_exact_diagonalization` via CLI with `--model shared` to compute CI reference energies for d=6, 12, 20.
**Why it failed:** The `--model shared` code path (line ~399) does not apply `kinetic_prefactor=0.5` to the t2d matrix, while the correct `solve_shared_ci_reference` function in `compare_ci_vmc_dot_entanglement.py` does. Result: d=20 gave E₀≈0.64 (physically wrong; correct value 1.749).
**Evidence:** Python API call with explicit `t2d = 0.5 * t2d` gave d=20 E₀=1.74902235; CLI gave ~0.64.
**What to do instead:** Always use Python API (`build_2d_dvr`, `build_potential_matrix`, etc.) with explicit kinetic_prefactor=0.5, not the CLI --model shared path, for shared-CI reference energies.
**Severity:** needs-rethink

### [2026-04-16] — FAILED: Voronoi sector analysis at small separations
**What:** Old `_build_localized_ho_basis` used per-well Voronoi masking (nearest-center assignment) followed by QR orthonormalization of masked columns.
**Why it failed:** At d≤8 bohr, HO orbital tails extend significantly into the adjacent well's Voronoi region. Masking discards this amplitude, producing a numerically orthogonal but physically incorrect basis. Result: sector structure at d=2 showed LL≈25%, LR≈25%, RL≈25%, RR≈25% (flat) even for a network with near-singlet energy.
**Evidence:** After Löwdin fix, d≥6 gives LL=0%, LR=50%, consistent with singlet. d=2/4 remaining issue is attributed to the ansatz (permanent with non-orthogonal orbitals), not the measurement.
**What to do instead:** Use Löwdin S^{-1/2} (see decision above).
**Severity:** dead-end
