# Research Roadmap — push to groundbreaking

**Author:** Aleksander Skogen
**Date drafted:** 2026-04-28 ~10:50 CEST
**Horizon:** 5 weeks (target wrap-up: 2026-06-02)
**Companion documents:**
- [`reports/2026-04-27_supervisor_report.md`](2026-04-27_supervisor_report.md) — frozen project snapshot through 2026-04-27.
- [`reports/2026-04-28_supervisor_update.md`](2026-04-28_supervisor_update.md) — running update for 2026-04-28.

---

## TL;DR

Honest assessment of the project so far is that the **infrastructure is excellent but the science is mid-tier**, with two findings (PINN-Heisenberg divergence at large `d`, today's non-monotonic `E_orbital(Sz)`) that are at high risk of being numerical artefacts. The five-track plan below converts the existing tooling into results that can survive an external benchmark and that target *non-trivial physics* rather than scalar correlator demonstrations.

The single highest-value move is **Track A (ED benchmark at N=4 d-sweep)** — it gates whether the d-sweep finding is publishable or is a known limitation of variational ansätze in flat-landscape regimes. Tracks B, C, D, E run mostly in parallel and depend on Track A only for cross-validation.

---

## Sequencing map

```
                ┌────────────────────────────────────────────┐
                │  Track A — ED benchmark at N=4 (d-sweep)   │  Week 1 (gate)
                │  Decision gate G1: real or artefact?       │
                └─────────────┬──────────────────────────────┘
                              │
     ┌────────────────────────┼─────────────────────────┐
     ▼                        ▼                         ▼
┌─────────┐           ┌────────────────┐         ┌──────────────┐
│ Track B │           │ Track C        │         │  Track E     │
│ Topo    │  Wk 1-3   │ Excited-state  │ Wk 1-4  │  Fabrication │ Wk 4-5
│ invar.  │           │ NQS            │         │  tolerance   │
└────┬────┘           └────────┬───────┘         └──────────────┘
     │                         │
     └─────────────┬───────────┘
                   ▼
         ┌────────────────────────┐
         │ Track D — Pareto       │ Wk 3-4
         │ (T vs gap, N=4 then N=8)│
         └────────────────────────┘
```

**Decision gates:**
- **G1 (end of Week 1)** — Track A: ED vs PINN at N=4. If PINN tracks ED ⇒ real physics (double down on d-sweep paper); if PINN drifts from both ED and Heisenberg ⇒ honest negative result, reframe as methodological paper.
- **G2 (end of Week 2)** — Track C: excited-state NQS at N=2 must reproduce singlet-triplet gap to 1% of analytical. If not, fall back to ED-only gap calculations everywhere.
- **G3 (end of Week 3)** — Track B: winding number module must reproduce sign flip on existing N=8 SSH checkpoints (theta = [4.0,4.0] vs [4.49,3.46]). If not, debug or pivot to dimerization-only metrics.
- **G4 (end of Week 4)** — Track D: Pareto frontier must show non-degenerate trade-off between `T` and gap. If frontier is flat / degenerate, refocus remaining budget on Track E robustness study.

---

## Track A — Resolve the d→∞ PINN-Heisenberg divergence (Week 1, **gating**)

### Goal
Decide once and for all whether the "PINN diverges from Heisenberg at large d" finding is real physics or a variational pathology. Without this, the d-sweep is undefendable.

### Existing infrastructure (verified)
- `scripts/exact_diag_double_dot.py:318-399` — `run_exact_diagonalization_one_per_well_multi`: linear multi-well CI with one electron per well, DVR kinetic + soft-min confinement + Coulomb, full `eigh` spectrum.
- `scripts/exact_diag_double_dot.py:163-207` — `precompute_coulomb_kernel`: validated Coulomb matrix elements (the same kernel that `tests/test_shared_ci_coulomb_kernel.py` regression-tests).
- `tests/test_shared_ci_coulomb_kernel.py` — guards against the eps=0.01 singularity bug we hit in Phase 0.

### What's missing (small, ~1.5 days)
1. ⟨S_i·S_j⟩ on the CI ground-state vector for the linear-chain CI sector. CI gives a determinant-by-determinant eigenvector; need to (i) for each pair (i,j) build the operator `S_i·S_j` in the CI basis (it's diagonal in particle positions and has matrix elements in spin), (ii) compute ⟨Ψ_GS|S_i·S_j|Ψ_GS⟩.
2. Similarly: total `S²` (sanity check, should equal `S(S+1)` for the GS sector), per-bond NN dimerization, and (small) total spin entropy.

### Concrete sub-tasks
| # | Task | File | Day |
|---|---|---|---|
| A1 | Read `exact_diag_double_dot.py:318-450`; document its CI basis, spin block structure, eigenvector format. | (notes only) | 1 |
| A2 | Add `compute_spin_correlators_ci(eigvec, basis)` to `src/observables/exact_diag_reference.py`. Returns NxN matrix of ⟨S_i·S_j⟩. Validated by hand at N=2 (singlet ⇒ ⟨S₀·S₁⟩ = −¾). | `src/observables/exact_diag_reference.py` | 1-2 |
| A3 | Write `scripts/n4_ed_d_sweep.py`. Loops d ∈ {2, 3, 4, 6, 8, 10, 14}; for each d runs the existing CI driver in the Sz=0 sector (and Sz=2 for the gap), extracts E_GS, E_first_excited, full ⟨S_i·S_j⟩ matrix, total spin entropy. Saves `results/ed/n4_d_sweep_ed.json`. | `scripts/n4_ed_d_sweep.py` | 2 |
| A4 | Write `scripts/compare_pinn_ed_n4_d_sweep.py`. Loads existing PINN d-sweep results + new ED results; for each d computes Δ(E), Δ(⟨S·S⟩_NN), Δ(spin entropy), Heisenberg overlap. Emits 4-panel PNG. | `scripts/compare_pinn_ed_n4_d_sweep.py` | 3 |
| A5 | Decision gate G1; write outcome paragraph in supervisor update. | (report) | 3 |

### Deliverable
- `results/ed/n4_d_sweep_ed.json`
- `results/ed/n4_d_sweep_ed_vs_pinn.png` — 4 panels: (E vs d), (⟨S₀·S₁⟩ vs d), (spin entropy vs d), (Heisenberg overlap vs d), each showing PINN, ED, Heisenberg lines.

### Definition of "groundbreaking achieved" for this track
- **Real physics outcome:** ED tracks PINN, Heisenberg drifts away ⇒ identify the analytical mechanism (longer-range exchange, ring exchange, or off-diagonal interaction). Write up as primary headline result. Estimated additional 1-2 weeks to characterize mechanism.
- **Artefact outcome:** ED tracks Heisenberg at all d, PINN drifts ⇒ document loss-noise floor vs J. Reframe Phase 3 as "limits of variational ansätze in flat-landscape regimes" — still publishable, but as methodology not discovery.

---

## Track B — Topological order parameters and topological-target inverse design (Weeks 1-3)

### Goal
Convert the SSH dimerization story from a scalar correlator demonstration into a topological-invariant engineering result. Distinguish the multi-well QD chain from any other AFM chain by engineering a non-local order parameter.

### Existing infrastructure
- Per-bond `⟨S_i·S_j⟩` already accessible via `src/observables/effective_heisenberg.py:153-227` (full N×N correlator matrix).
- Density extraction already exists for edge-localization measurements.
- `GeometryOptimizer` accepts custom callables as targets (`src/geometry_optimizer.py:532-533`), so adding a topological target only requires writing the observable, not modifying the optimizer.

### What to build
1. `src/observables/topological_invariants.py`:
   - `dimerization_order_parameter(corr_matrix, n_wells)` — `D = (1/N) Σᵢ (-1)ⁱ ⟨S_i·S_{i+1}⟩`. Pure scalar but a **non-local** order parameter (the *staggering* of NN correlators is the signature of broken translation symmetry).
   - `ssh_winding_number(spin_amplitude_eigenstructure)` — uses the dimer-eigenmode structure from `src/observables/spin_amplitude_entanglement.py`. Maps the chain to a two-band tight-binding model (sub-lattice = even/odd bonds), computes the Zak phase via the Wilson loop of the Bloch eigenstates. Returns ∈ {0, 1}.
   - `edge_localization(density, n_wells, edge_window=2)` — fraction of total density in the first/last `edge_window` wells minus the bulk-uniform expectation.
2. CLI wiring in `scripts/run_inverse_design.py`:
   - Add `winding_number`, `edge_localization`, `dimerization` as built-in targets.
3. Validation:
   - Run forward computation on the existing N=8 SSH flagship checkpoints at θ = [4.0, 4.0] (uniform, expect winding = 0) and θ = [4.49, 3.46] (dimerized, expect winding = 1). **Decision gate G3.**
4. Inverse-design demonstration:
   - Re-run N=8 SSH inverse design with `target = winding_number` (binary) — use a sigmoid-relaxed proxy in the loss to make it differentiable. Show the optimizer drives winding from 0 → 1 across the optimization steps.

### Sub-tasks
| # | Task | Day |
|---|---|---|
| B1 | Implement `dimerization_order_parameter` + unit test (synthetic uniform vs alternating ⟨S·S⟩ inputs). | 5-6 |
| B2 | Implement `ssh_winding_number` + unit test on toy SSH model with analytical winding 0/1. | 7-9 |
| B3 | Implement `edge_localization` + unit test. | 9 |
| B4 | Forward-validate winding number on existing N=8 SSH checkpoints. **G3.** | 10-11 |
| B5 | Wire as inverse-design targets; smoke test on N=4. | 12-13 |
| B6 | N=8 inverse design with `winding_number` target. Re-engineer the SSH flagship state. | 14-17 |

### Deliverable
- `src/observables/topological_invariants.py` (~150 LoC + tests)
- New built-in target options visible in `scripts/run_inverse_design.py --help`
- `results/inverse_design/n8_ssh_winding_invdes/` — trajectory showing winding 0 → 1, with per-step PINN check that the engineered geometry is internally consistent.

### Definition of "groundbreaking achieved"
The winding number is engineered from 0 to 1 by tuning (θ₁, θ₂), and the engineered topological state is robust against ±1% well-position perturbation (cross-validates with Track E). This is a clean topological inverse-design result that is **not a 3.8% correlator improvement** — it's a binary (or near-binary) topological transition driven by geometric tuning.

---

## Track C — Excited-state NQS lane (Weeks 1-4, parallel build)

### Goal
Add a variational first-excited-state lane to the NQS pipeline so that the gap can be computed at any N (not just N≤4 where ED is tractable). Unlocks Track D at N=8.

### Existing infrastructure
- `src/run_ground_state.py` + `src/training/vmc_colloc.py` — single-state training driver.
- `src/geometry_optimizer.py:713-738` — `_target_fn_exchange_gap`: bilevel sector training (different Sz sectors), not a true variational excited state.
- No orthogonal-projection loss term exists.

### What to build
1. `src/wavefunction.py` — add `ExcitedStateWF(GroundStateWF, gs_checkpoint, lambda_ortho)` wrapper that holds a frozen GS reference and exposes:
   - Forward: same as the underlying ansatz.
   - Loss (extra term): `λ · |⟨ψ_excited|ψ_GS⟩|² / (⟨ψ_GS|ψ_GS⟩ ⟨ψ_excited|ψ_excited⟩)` evaluated by Monte Carlo on the same collocation samples.
2. `src/training/vmc_colloc_excited.py` — extends the existing trainer with the orthogonality penalty (alternative: just monkey-patch the loss in the existing trainer).
3. `scripts/run_two_stage_excited_state.py` — orchestrates GS training, then excited-state training initialized from the GS checkpoint with the ortho penalty.
4. Validation chain:
   - **N=2 singlet-triplet gap** vs analytical (≈ J for AFM dimer; J extracted from existing `effective_J`). **Decision gate G2.**
   - **N=4 first-excited energy** vs Track A's ED result.
   - **N=8 first-excited energy** as the eventual deliverable (no external benchmark, but variance + ESS health checks).

### Sub-tasks
| # | Task | Day |
|---|---|---|
| C1 | Read `vmc_colloc.py` loss path, identify the cleanest extension point. | 4 |
| C2 | Implement orthogonality penalty as a free-standing function with unit tests on toy 2-state systems (analytic projection). | 5-7 |
| C3 | Wire into trainer; add `--ortho-lambda`, `--gs-checkpoint` to `run_two_stage_excited_state.py`. | 8-10 |
| C4 | Validate at N=2: train GS, train ES with λ ∈ {0.1, 1.0, 10.0}, gap vs J ratio. **G2.** | 11-12 |
| C5 | Validate at N=4: gap vs ED gap from Track A. | 13-15 |
| C6 | N=8 first-excited energy + variance health check. | 16-19 |
| C7 | Add `target = gap` (= E_excited − E_GS) to `GeometryOptimizer` callables. | 20 |

### Deliverable
- `src/wavefunction.py` updated with `ExcitedStateWF`
- `scripts/run_two_stage_excited_state.py`
- `results/excited/n2_gap_vs_J.json`, `results/excited/n4_gap_vs_ed.json`, `results/excited/n8_gap.json`

### Definition of "groundbreaking achieved"
- The excited-state lane reproduces N=4 ED gap to <5%.
- N=8 first-excited energy is variationally consistent (variance < 10⁻³, ESS > 16).
- Gap is wired as a target in `GeometryOptimizer` for direct use in Track D.

---

## Track D — Multi-objective Pareto frontier (T vs gap) (Weeks 3-4)

### Goal
Convert single-scalar inverse design into a multi-objective trade-off study: maximize entanglement subject to a minimum gap (or equivalently, sweep the Pareto frontier).

### Existing infrastructure
- `GeometryOptimizer` does single-scalar; multi-objective via custom callable is possible but Pareto-aware constrained optimization is not implemented.
- For N=4: gap from Track A (cheap ED), entanglement from existing `entanglement_n2` / `well_set_entanglement`.
- For N=8: gap from Track C, entanglement from existing observables.

### What to build
1. `scripts/pareto_sweep.py`:
   - Approach 1 (simple): weighted sum `L(θ) = -T(θ) + α · max(0, gap_min - gap(θ))²` with `α` swept over {1, 10, 100} and `gap_min` swept. Each combination runs an inverse-design optimization, gives a (T*, gap*) point. Collect all such points and plot the Pareto envelope.
   - Approach 2 (more rigorous, time permitting): NSGA-II or a simple 2D θ grid scan if dim(θ) is small (≤2 for SSH parametrization).
2. Plotting: `scripts/plot_pareto_frontier.py` — collects the (T, gap) cloud, computes the Pareto-dominant subset, plots it with each θ-state annotated.

### Sub-tasks
| # | Task | Day |
|---|---|---|
| D1 | Build `scripts/pareto_sweep.py` with weighted-sum approach. | 18-20 |
| D2 | N=4 Pareto sweep using ED gap (cheap). 5×5 (α, gap_min) grid ⇒ 25 inverse-design runs at N=4 (~1 hour each on a single GPU; ~2-3 GPU-days). | 21-23 |
| D3 | N=4 Pareto plot + interpretation. **G4.** | 23-24 |
| D4 (conditional on G2/G4 pass) | N=8 Pareto sweep using NQS gap (Track C). 3×3 (α, gap_min) grid ⇒ 9 runs at N=8 (~3 hours each; ~3-4 GPU-days). | 25-28 |
| D5 | N=8 Pareto plot + interpretation. | 28 |

### Deliverable
- `results/inverse_design/n4_pareto_t_vs_gap.{csv,png}`
- `results/inverse_design/n8_pareto_t_vs_gap.{csv,png}` (if Track C lands)
- A short paragraph identifying the "knee" of the Pareto frontier — the geometry that gives the best entanglement-per-gap trade-off.

### Definition of "groundbreaking achieved"
The N=8 Pareto frontier shows a non-trivial trade-off (i.e., the entanglement-maximizing geometry has measurably smaller gap than the gap-maximizing geometry, with a continuous frontier between them). The "knee" geometry is identified and characterized — that is the geometry an experimentalist would actually want to fabricate.

---

## Track E — Fabrication-tolerance robustness study (Weeks 4-5)

### Goal
Quantify how robust the engineered states (SSH dimerization, topological winding, Pareto-knee geometry) are to realistic fabrication noise. Connects theory to experimental QD platforms.

### Existing infrastructure
- `scripts/gen_disorder_configs.py:1-73` — generates positionally-jittered well configs for MBL studies. Direct precedent for fabrication noise.
- `src/geometry_optimizer.py:340-342` — disorder-pattern inverse-design hooks (parametrisation only, not random sampling).

### What to build
1. `scripts/fabrication_noise_sweep.py`:
   - Inputs: a baseline geometry (SSH flagship θ*, or any other engineered state), a perturbation protocol (`gaussian_position_jitter(σ_pos)`, `gaussian_depth_jitter(σ_depth)`), and `M` (number of MC samples per σ).
   - For each σ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0} (relative to the natural well-spacing length scale), draws M=20 perturbed geometries, trains each from the same seed (or warm-started from the unperturbed checkpoint), evaluates the relevant observable.
   - Emits mean ± std curves vs σ.
2. Realistic σ values from the literature:
   - Si MOS QDs: position jitter ~0.5-1 nm in arrays (Watson et al. 2018; Camenzind et al. 2022). Depth jitter from gate voltage noise ~0.5% of barrier height.
   - GaAs gate-defined QDs: position ~5-10 nm, depth ~2-3% of barrier.
   - Donor-based: position ±1 atomic plane (~0.4 nm), depth ±10 meV.
3. Tolerance curves for: SSH dimerization (Track B), winding number (Track B), Pareto-knee state (Track D).

### Sub-tasks
| # | Task | Day |
|---|---|---|
| E1 | Read `gen_disorder_configs.py`; extract reusable jitter sampling. | 22-23 |
| E2 | Add depth-jitter (well-omega) to the perturbation protocol. | 23-24 |
| E3 | Build `scripts/fabrication_noise_sweep.py` orchestrator. | 24-26 |
| E4 | Run M=20 σ-sweep on SSH flagship. ~6 GPU-days (parallelizable over 3-4 GPUs). | 27-30 |
| E5 | (Optional, if Track B lands) Run on winding-number-engineered state. | 31-33 |
| E6 | Tolerance curves + interpretation. Identify σ_max (largest σ at which observable stays >50% of unperturbed value). | 33-35 |

### Deliverable
- `scripts/fabrication_noise_sweep.py`
- `results/robustness/ssh_fabrication_tolerance.{csv,png}` — mean ± std curves of dimerization, winding number, gap, T vs σ_pos and σ_depth.
- A paragraph in the final report linking σ_max to one or two specific experimental QD platforms with citations.

### Definition of "groundbreaking achieved"
"The engineered SSH dimerization survives positional jitter up to σ = 0.X (Y% of well-spacing), corresponding to fabrication tolerance achievable in [specific experimental platform]". This sentence — or its negation — is what makes the theoretical result citable by experimentalists.

---

## Right-now actions (today, before close of business)

### Immediately, no GPU contention
1. Run `amplitude_evolution.py` on each of the 5 N=8 sector checkpoints from today's seed-42 sweep. Pure CPU, ~10 min total. Goal: spin-resolved spectroscopy of each Sz sector at B=0. (Existing pending todo `p3b-amplitude-per-sector`.)
2. Generate the scaling-curve plot for N ∈ {2, 3, 4, 8, 12, 16}. Pure CPU, ~2 min. (Existing pending todo `p4-scaling-plot`.)

### When seed-17 sweep finishes (cuda:3 frees up, ~12:30 CEST)
3. Compare seed-42 and seed-17 sector-aware sweeps. If E_orbital(Sz=±1) below E_orbital(Sz=0) **persists** ⇒ note as possibly real, add to roadmap. If it **flips** ⇒ confirms variational under-convergence, note in supervisor update.

### Same evening (overnight if needed)
4. Track A start: read `scripts/exact_diag_double_dot.py:318-400`, scope the ⟨S·S⟩-on-CI extension. Write `src/observables/exact_diag_reference.py:compute_spin_correlators_ci`. Validate at N=2 (singlet must give −¾).

### Tomorrow morning
5. Track A continued: run `scripts/n4_ed_d_sweep.py` over d ∈ {2, 3, 4, 6, 8, 10, 14}. Compare with PINN d-sweep results at the same d-values. **Decision gate G1 by tomorrow EOD.**

---

## Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Track A ED says PINN is artefact at d→∞** | Medium-High (my prior: 65%) | Headline d-sweep finding dies. | Reframe d-sweep as methodological/limits paper; this is still valuable and publishable in a methods journal. |
| **Track C excited-state NQS doesn't converge** (collapse, λ-tuning never stable) | Medium (my prior: 30%) | N=8 gap unobtainable; Track D N=8 dies. | Fall back to ED for N=4 Pareto only. Drop N=8 Pareto from scope. |
| **Track B winding number doesn't sign-flip on existing checkpoints** | Low-Medium (my prior: 25%) | Topological story is harder. | Re-examine: maybe the SSH flagship didn't reach the topologically nontrivial phase. May need to run with a stronger dimerization target. Or pivot to dimerization-only metrics. |
| **GPU contention slows everything 2x** | Medium (consistent issue) | All deadlines shift right by 30%. | Already factored in (dates above are "best-case but realistic"). 6/8 GPUs available steady-state. |
| **Shell unresponsiveness blocks launches** | Low-Medium (intermittent) | Wastes 5-10 min per launch. | Already established workaround: read terminal files / log files directly via Read tool, retry shell after 30-60s. |
| **Excited-state pipeline reveals N=8 GS itself is sub-converged** | Low (my prior: 15%) | All published N=8 numbers shift. | Track A's N=4 ED + a single N=8 multi-seed run will catch this. |
| **N=4 ED disagrees with PINN at d=4 too** | Low (my prior: 10%) | Even the d=4 Heisenberg cross-check (existing 0.939 overlap) is in question. | Re-validate with explicit eigenvector overlap computation, not just energy. |

---

## Definition of "groundbreaking" for the project as a whole

A reasonable success criterion at the end of week 5: **at least three of the five tracks land** with the "groundbreaking" outcome described in their respective sections, AND at least one of the following is true:

- **(a) The d-sweep finding survives ED benchmark and has an identified analytical mechanism** (Track A real-physics outcome). This is sufficient on its own for a high-impact methodology+physics paper.
- **(b) Topological inverse design with winding-number engineering and fabrication-robust tolerance** (Tracks B + E). This is sufficient on its own as a "designer topological QD chain" paper.
- **(c) Multi-objective Pareto frontier with experimentally-actionable knee geometry** (Tracks C + D + E). This is sufficient on its own as an applied quantum-information paper.

**If only one of (a), (b), (c) lands**: still a strong PhD chapter, not a high-impact paper.
**If two of (a), (b), (c) land**: high-impact paper at the methodology level.
**If three land**: a thesis with a publishable trilogy (methodology, topological, applied).

The bottleneck for all three is Track A. If A says artefact, we lose (a) but the methodology paper "limits of variational ansätze" is still reasonable; (b) and (c) are unaffected. If A says real, we accelerate.

---

## Calendar snapshot

```
       Mon 28 Apr | Tue 29 Apr | Wed 30 Apr | Thu 01 May | Fri 02 May
Track A:  scoping    code+test    run+gate G1   write-up    (rest)
Track B:    -          -            -           start      build
Track C:    -          -            -           scoping    impl
Track D:    -          -            -            -          -
Track E:    -          -            -            -          -

       Wk 2 (5-9 May) | Wk 3 (12-16) | Wk 4 (19-23) | Wk 5 (26-30 May) | wrap (1-2 Jun)
Track A:  (mech?)        write-up      (in paper)      (in paper)        --
Track B: build+G3        invdes run    fab cross       (in paper)        --
Track C: impl+G2         N=4 valid     N=8 + targets   (in paper)        --
Track D:    -            scope         N=4 sweep+G4    N=8 sweep         --
Track E:    -              -            scope          σ-sweep+write     --
```

---

*Live document, updated as decision gates fire. Each gate outcome will be appended as a "G_N — outcome" stanza below.*
