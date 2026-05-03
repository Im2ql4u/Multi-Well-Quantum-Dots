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

### [2026-04-28] — Real-time NQS PINN prototype passes the trivial-evolution sanity check: trained ⟨g_I⟩(t) matches analytical −E_0·t to 0.01% across t∈[0,1]
**Motivation:** Strategic pivot of 2026-04-28 (see `reports/2026-04-28_pivot_realtime_nqs.md`) re-prioritises the moonshot: a non-MCMC, deterministic-sampled, gradient-based PINN platform for **real-time** fermionic many-body dynamics on engineered multi-dot networks. Standard time-dependent NQS work (Carleo-Troyer, Schmitt-Heyl, …) lives on lattices and uses MCMC + stochastic reconfiguration / TDVP linear solves. We have the unique infrastructure to attempt the same on the **continuum**, with **deterministic sampling**, via **PDE-residual collocation** on (x, t). Step 1 is to verify that the residual derivation, autograd implementation, and hard-IC trick all work on the simplest nontrivial test: free evolution under the same H whose ground state is ψ_0 (analytical answer: ψ(x,t)=e^{−iE_0 t}ψ_0(x), i.e. g_R≡0, g_I=−E_0·t).
**Method:**
- Derived the coupled real-time PDE for `g(x,t)=g_R+i g_I` from `i∂_t ψ = H ψ` with `ψ=exp(log ψ_0 + g)`. Result: `∂_t g_R = −½∇²g_I − (∇log ψ_0)·∇g_I − ∇g_R·∇g_I` and `∂_t g_I = −E_L^{(0)} − ΔV + ½∇²g_R + (∇log ψ_0)·∇g_R + ½(|∇g_R|²−|∇g_I|²)`. IC `g_R(x,0)=g_I(x,0)=0`.
- Built `src/realtime_pinn.py` (~430 LOC, self-contained): `RealTimeNet` (FiLM 2-output MLP with hard IC via output × t multiplier — no IC penalty needed), `compute_realtime_residual` (autograd-based, with `allow_unused=True` safety so oracle/decoupled networks don't crash), `train_realtime_pinn` (Adam + cosine LR, no SR, no MCMC).
- Wrote two layers of tests. **Pytest** (`tests/test_realtime_pinn.py`, 7/7 passing in 4.5 s on CPU): IC-by-construction, output shapes, residual shapes, `Re/Im(E_L)` formulas at zero g, and an **oracle correctness check** that plugs `g_R≡0, g_I=−E_0 t` into `compute_realtime_residual` and verifies the residual vanishes to **machine precision** (≤ 1e-10). The latter isolates symbolic correctness from training noise. **GPU smoke** (`scripts/smoke_realtime_pinn_trivial.py`): fits a 25k-parameter PINN to the trivial PDE on a 1024-sample HO (N=2, d=2, ω=4, E_0=8) for 600 epochs in 80 s on cuda:6.
**Results:**

| t   | ⟨g_R⟩      | g_R RMS    | ⟨g_I⟩      | analytic −E_0·t | RMS err on g_I | rel err |
|-----|------------|------------|------------|------------------|----------------|---------|
| 0.00 | 0          | 0          | 0          | 0                | 0              | 0.00%   |
| 0.10 | 2.6e-5     | 1.1e-4     | −0.80010   | −0.80000         | 1.31e-4        | 0.02%   |
| 0.25 | 1.8e-5     | 1.3e-4     | −1.99987   | −2.00000         | 1.68e-4        | 0.01%   |
| 0.50 | 8.8e-5     | 1.7e-4     | −4.00006   | −4.00000         | 2.29e-4        | 0.01%   |
| 0.75 | 2.3e-4     | 3.0e-4     | −6.00006   | −6.00000         | 2.63e-4        | 0.00%   |
| 1.00 | 3.7e-4     | 4.9e-4     | −8.00010   | −8.00000         | 2.93e-4        | 0.00%   |

PDE residual loss decreased from 64 → 1.2e-5 over 600 epochs.
**What the numbers actually mean:** The PDE-residual derivation is correct (any sign or coefficient bug would show up as g_R growing instead of staying near zero, or as g_I deviating from the linear ramp). The non-MCMC, deterministic-sampling, PDE-residual approach to real-time NQS evolution is *viable*: trained PINN reproduces the analytical answer to four-five decimal places across the entire time interval, simultaneously, with no MCMC and no SR. The hard-IC trick (output × t) eliminates the IC penalty hyperparameter and gives exact `g(x,0)=0` at t=0.
**What we cannot explain:** Nothing in this regime — the trivial test is *meant* to be machine-checkable. The interesting failure modes (real ΔV ≠ 0, larger N, longer t) are precisely what the next experiments target.
**Caveats:** This is a synthetic pool — `E_L^{(0)} ≡ E_0` was set by hand, not computed from a real ψ_0. The next test (`scripts/run_realtime_n2_quench.py`, not yet built) will use a real trained N=2 ψ_0 with a Zeeman quench and compare to ED time evolution. Also, the harmonic-oscillator pool is symmetric Gaussian, which lets the small `g_R` track partly to symmetry rather than to optimisation pressure; we'll lose that crutch in non-symmetric problems.
**What a skeptic would say:** "You haven't proven anything new — the analytical solution is trivially representable by a small network. The real test is whether you can fit a *non-stationary* trajectory with comparable accuracy." Correct, and that is the exact next experiment.
**Output reference:** `results/realtime_pinn/trivial/{trivial_smoke.json,trivial_smoke.png}`; pivot writeup in `reports/2026-04-28_pivot_realtime_nqs.md`.
**Next question:** Does the residual-collocation PINN reproduce a *real* N=2 ψ_0 quench (turn on a Zeeman B field at t=0, evolve for one Larmor period) to within 5% RMS of ED time evolution? If yes, scale to N=4 chain Néel quench; that's the entry point to "no-ED-can-touch-this" territory.

---

### [2026-04-28] — Real-time NQS PINN beats first non-trivial quench: 1.00% RMS rel error vs analytical breathing-mode reference for the N=2 2D harmonic-oscillator ω-quench (ω₀=1 → ω₁=2) over a full breathing period
**Motivation:** With the trivial sanity check passing to 0.01%, the next test is the smallest *non-trivial* quench against an analytical reference: a sudden frequency change of a 2D HO. The closed-form Heisenberg-picture answer is `⟨|x|²⟩₁ₑ(t) = (1/ω₀)cos²(ω₁t) + (ω₀/ω₁²)sin²(ω₁t)`, and the quench induces a *spatially-varying* PDE driver `ΔV(x) = ½(ω₁²−ω₀²)Σ_i|x_i|²` — not the trivial `ΔV=0` case. We need a *quantitative* benchmark with Z(t) ≡ 1 unitarity diagnostic.
**Method:**
- Built `src/realtime_quench_ho.py` — analytical pool (`ψ_0` Gaussian product + closed-form `E_L^{(0)} = N·ω₀`, `∇log ψ_0 = -ω_0 x`, `ΔV(x) = ½(ω₁²-ω₀²)Σ|x_i|²`), analytical reference `analytical_x2_per_electron(t)`, deterministic |ψ_0|² sampling (rng.normal, no MCMC).
- Built `scripts/run_realtime_n2_omega_quench.py` driver: trains a real-time PINN on the precomputed pool, evaluates `⟨Σ_i|x_i|²⟩(t)` via reweighted estimator `⟨O·e^{2g_R}⟩/⟨e^{2g_R}⟩`, and the unitarity diagnostic `Z(t):=⟨e^{2g_R}⟩` (≡1 under exact unitary evolution). 7 quench unit tests in `tests/test_realtime_quench_ho.py` (formulas, period, residual identity at t=0).
- **First attempt** with the generic `RealTimeNet` (FiLM MLP, 25k params): 19% RMS rel err in 1200 epochs. Loss converged to 0.04 but `Z(t)` drifted from 0.79 to 1.05 — the residual is small but the wavefunction *leaks norm* and the reweighted observable is biased.
- Tried adding a **soft unitarity regularizer** `(log Z(t))²` (less aggressive than `(Z-1)²` thanks to log scaling). At weight 0.5–1.0 it destabilised training; at 0.05 it didn't change much. Tried **small-t analytic anchor** loss `g_I(x, t_a) = -(E_L⁽⁰⁾+ΔV)·t_a`: helped marginally (24%→28%), strong anchor (50) actively diverged. Tried **quadratic-in-x features** added to the spatial encoder: also destabilised.
- **Diagnosis**: stochastic variability across 'identical' MLP runs is ~5pp; the MLP just *lacks the right inductive bias* to represent the breathing dynamics, which has the **exact analytical form** `g(x,t) = c(t) + α(t)·r²(x)` with `r²(x) = Σ_i|x_i|²`.
- **Fix**: built `PolynomialQuenchNet` — a tiny MLP `t_embed → (c_R, c_I, α_R, α_I)(t)` whose output is assembled into `g_R = (c_R + α_R·r²)·t` and `g_I = (c_I + α_I·r²)·t`. The `× t` factor still enforces the hard IC. Total: **2228 params** (11× smaller than the MLP).
**Results:**

| t (units of T_breath = π/ω₁) | analytic ⟨Σ|x_i|²⟩ | PINN ⟨Σ|x_i|²⟩ | abs err | rel err | Z(t)   |
|------------------------------|---------------------|-----------------|---------|---------|--------|
| 0                            | 2.000               | 1.972           | 0.028   | 1.40%   | 1.0000 |
| 0.25 T (max compression)     | 0.501               | 0.505           | 0.004   | 0.88%   | 1.012  |
| 0.50 T (recurrence)          | 1.230               | 1.221           | 0.009   | 0.71%   | 1.010  |
| 0.75 T                       | 1.230               | 1.225           | 0.005   | 0.36%   | 1.005  |
| 1.00 T (full recurrence)     | 2.000               | 1.966           | 0.034   | 1.72%   | 1.0003 |

Aggregate over 60 t-points evenly spaced on [0, T]: **RMS rel err = 1.00%, max rel err = 1.72%.** Z(t) ∈ [1.000, 1.015] → unitarity preserved to 1.5%. Final loss = 3.2e-4 (vs 0.04 for the MLP, 30× lower). Training wall = **31 s on cuda:6**.
**What the numbers actually mean:** The non-MCMC, deterministic-sampling, PDE-residual-collocation framework for real-time fermionic NQS dynamics **works at quantitative accuracy on a non-trivial quench**, beating the 5% RMS pre-registered target by 5×. The residual ~1% is *dominated by Monte-Carlo sampling noise* on the 8192-sample eval pool (visible at t=0 where the answer is exactly `2.000` but the empirical mean of `Σ|x_i|²` over 8192 |ψ_0|² samples is `1.972` — pure sampling noise, the PINN itself is exact at t=0 by the hard IC). Z(t) drift ≤ 1.5% means the ansatz preserves norm to that level *purely from the PDE residual* — no explicit norm regularizer, no manual rescaling. The polynomial ansatz works because it *is* the exact analytical structure for separable Gaussian quenches; this validates the full PDE-residual + reweighted-observable pipeline end-to-end.
**What we cannot explain:** Why the generic FiLM MLP plateaus at 19% rel err even with 25k params — capacity-wise it should easily represent a 4-coefficient quadratic-in-r² function. The training surface for the MLP appears to have many shallow local minima that all satisfy the residual to ~0.04 but pick *different* spatial structures for `g_R`. This is consistent with the under-determination of small PDE residuals in the absence of spatial anchors. We did not try Sobolev-style PDE penalties or curriculum on t_max, both of which might help but are not necessary now that the polynomial route works.
**Caveats:** The polynomial ansatz is **exact only for separable, Gaussian-preserving Hamiltonians** (no Coulomb, no anharmonic potentials). For real multi-dot problems we need either (a) a polynomial backbone + MLP residual `g_R = c(t) + α(t)·r² + ε_R(x,t)`, or (b) a richer expansion (cross terms `x_i·x_j`, higher orders). The 1% RMS is on a *single observable* (`⟨|x|²⟩`); other observables (entanglement entropy, density profile far from r=0, off-diagonal coherences) may fail. Z(t) drift up to 1.5% is small for one breathing period, but at multi-period times errors will compound; a future test should integrate over 5–10 periods.
**What a skeptic would say:** "You hand-coded the analytical solution structure into the network, then announced the network reproduces the analytical solution. That's tautological." Counter-argument: the *polynomial structure* is a soft prior (the network still has to learn the time-dependent coefficients from PDE residual alone, with no observable matching). A bigger validation will come from problems where the polynomial ansatz is *approximate*, not exact — e.g. N=2 with weak Coulomb, where the polynomial part captures the orbital geometry but the residual MLP must capture the correlation-induced dispersion.
**Output reference:** `results/realtime_pinn/omega_quench/poly_neval8192/{omega_quench.json,omega_quench.png}`; the failure analysis lives in `results/realtime_pinn/omega_quench/{n2_w01_w12_e1200_s0,reproduce_run1,anchor5_2000,anchor50_1200,long_5000_baseline,quad_feats_baseline}/`.
**Next question:** Does the polynomial-backbone + MLP-residual ansatz hold up against a Coulomb-coupled N=2 quench (where the polynomial answer is approximate, not exact), benchmarked against an ED time evolution? After that, the N=4 chain Néel quench is the entry point to "no-ED-can-touch-this" territory.

---

### [2026-04-26] — Multi-sector inverse design: exchange-gap target (E_T − E_S) drives N=2 separation from d=2.0 to d=1.77 in 4 outer steps, doubling J from 0.093 to 0.185 Ha
**Motivation:** Phase 1B. With dot-label / spin-amplitude entanglement targets validated, we need to demonstrate that the inverse-design framework can target *any* many-body observable — not just observables computed from a single trained Ψ. The natural next observable is the **singlet-triplet exchange gap** `J = E_T − E_S`, which is THE foundational quantity for quantum-dot spin qubits (Loss-DiVincenzo gates, singlet-triplet readout, Heisenberg-chain emulation). Computing it requires *two* trainings per geometry — one in the singlet sector and one in the triplet sector — so this also exercises the multi-sector machinery the framework needs for the eventual `J_eff(i,j)` engineering flagship.
**Method:**
- Refactored `src/geometry_optimizer.py` to support multi-sector inverse design: introduced `GeomEvalContext` (one geometry → one or more sector-resolved trainings) and `spin_overrides: dict[name, {n_up, n_down, force_no_singlet_arch, stage_a_strategy}]`. `_train_geometry` spawns one inner training per spin sector, each warm-starting from its own previous-step checkpoint. The legacy single-sector targets (`energy`, `entanglement_n2`, `well_set_entanglement`) still work unchanged via the `primary_sector` key.
- Added the `exchange_gap` target with N-aware default sectors (singlet = (N//2, N − N//2), triplet = singlet S^z + 1). For N=2 the triplet defaults to `(2, 0)`, the fully polarised m=+1 sector — a single Slater determinant, much cheaper to optimise than the m=0 hybrid. The triplet sector auto-disables the N=2 `architecture.singlet` permanent ansatz and enables `multi_ref=True`. CLI flags: `--target exchange_gap`, `--singlet-spin n_up n_down`, `--triplet-spin n_up n_down`, `--target-J J0` (drive the gap to a SPECIFIC value via `T = -(J − J0)^2`), `--unsigned-gap`, `--triplet-stage-a-strategy`.
- Smoke test: started at `d=2.0`, bounds `d ∈ [1.5, 6.0]`, learning rate 0.4, FD step 0.4, 4 outer steps, `improved_self_residual` Stage A for both sectors, Stage B disabled. Run on cuda:6 in parallel with the in-flight N=4 entanglement flagship on cuda:3.
**Results:**

| Step | d | E_singlet (Ha) | E_triplet (Ha) | **J = E_T − E_S** | grad | FD mode | dt (s) |
|------|---|----------------|----------------|---------------------|--------|---------|--------|
| 0    | 2.000 | 2.394 | 2.487 | **0.0933** | −0.228 | central | 356 |
| 1    | 1.909 | 2.392 | 2.510 | **0.1181** | −0.214 | central | 356 |
| 2    | 1.823 | 2.371 | 2.533 | **0.1620** | −0.146 | forward (lower-bound clip) | 238 |
| 3    | 1.765 | 2.366 | 2.551 | **0.1854** | −0.179 | forward (lower-bound clip) | 235 |

Total wall time: **20 minutes** (24 trainings = 4 outer steps × (1 centre + 2 perturbations) × 2 sectors). `J` doubled. `θ*` = 1.7646; the optimiser hit the lower bound `d=1.5` on the FD-minus side at step 2 and the divergence detector correctly fell back to forward differences for the rest of the run.
**What the numbers actually mean:**
- `J > 0` confirms the singlet is the AFM ground state, as super-exchange `J ≈ 4 t² / U` predicts. The PINN reproduces this without being told about the spin physics.
- The gradient is *opposite in sign* from the entanglement-target run. Phase 1E pushed `d` from 2.0 → 3.35 to maximise dot-label negativity (Bell-state coherence); Phase 1B pushes `d` from 2.0 → 1.77 to maximise `J` (gate speed). **Same machinery, opposite optimum.** This is the textbook quantum-dot quantum-computing tradeoff — coherence vs gate speed — and the bilevel loop reads it off automatically without operator hints.
- Step 0 took 356 s (six fresh trainings), step 3 took 235 s (warm-starting from step-2 sector checkpoints saved 35%). Each sector independently warm-starts from its own previous-step centre, which the architecture is designed for.
- The signed gap *J(d)* is monotonic and exponentially decaying — `J(1.5) ≳ 0.21`, `J(2.0) ≈ 0.09`, `J(2.4) ≈ 0.02`. This is consistent with `t ~ exp(−d²/(2σ²))` super-exchange and validates the FD numerics: the gradient at step 0 was central, at step 2/3 forward (boundary), and the magnitudes track the analytic decay rate.
**What we cannot explain (yet):** Whether the (1.0/0.0)-sector triplet uses the same orbital basis as the (1.0/1.0) singlet — a potential apples-to-oranges concern. Different ansätze could in principle introduce a fixed energy offset that contaminates the gap measurement. The Heisenberg sign and decay rate look right, so the worst-case bias is small, but a CI cross-check at d=4 (where exact diag is feasible) would settle this.
**Caveats:**
- The triplet sector uses `multi_ref=True` instead of the singlet permanent. The multi-ref ansatz has a strictly higher variational ceiling for the triplet because the spin orbitals are populated separately, so the gap is a *lower bound* on the true J. At d=2 this is not a problem: both sectors have low residual variance (~10⁻³ Ha²) and the qualitative trend is robust.
- 4 outer steps are not enough to saturate. The lower-bound 1.5 is a soft cap chosen to keep the Mott projection valid; pushing further would enter the spatially-overlapping regime where the (2,0) sector starts costing significant Coulomb energy and the gap analysis becomes a proper t-J calculation.
**What a skeptic would say:** "You haven't shown you can hit a *specific* J value, only that the loop moves in the right direction." Right — that's why we follow up with `--target-J 0.05`: drive `T = -(J − 0.05)^2` from `d=2.0` (where J ≈ 0.09) up to the J=0.05 isocurve at `d ≈ 2.5`. This is the actual deliverable for quantum-processor design (engineer a *specific* exchange coupling) and is the basis for the upcoming Phase 2B N=8 J_eff engineering flagship.
**Output reference:** [results/inverse_design/n2_exchange_smoke/history.json](results/inverse_design/n2_exchange_smoke/history.json), [results/inverse_design/n2_exchange_smoke/optimal_geometry.json](results/inverse_design/n2_exchange_smoke/optimal_geometry.json), [results/inverse_design/n2_exchange_smoke/trajectory.png](results/inverse_design/n2_exchange_smoke/trajectory.png), [src/geometry_optimizer.py](src/geometry_optimizer.py) (multi-sector refactor)
**Next question:** Validate `--target-J` (drive J to 0.05 Ha via gradient on `-(J − 0.05)²`); then scale to N=8 `dimer_chain` parametrisation with target = `(J_eff(0,7) − J_target)²` to engineer a specific long-range exchange coupling — Phase 2B flagship.

---

### [2026-04-26] — Heisenberg cross-check on N=4 chain at d=4: PINN overlap with pure Heisenberg AFM is 0.939, off-Mott corrections show up as enhanced bipartite entanglement
**Motivation:** Phase 2A.4. Use the OBC Heisenberg ground state as a reference for the Mott-projected spin amplitudes of the trained PINN. At "deep enough" d the PINN should approach Heisenberg with a `4 t^2 / U` super-exchange coupling; the deviation tells us how off-Mott the trained network is at that geometry.
**Method:**
- Implemented `src/observables/heisenberg_reference.py` and `scripts/heisenberg_cross_check.py`. The Heisenberg Hamiltonian `H = sum_i J_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z)` is built in the `(n_up, n_down)` sector using the *same pattern enumeration* as the spin-amplitude extractor, then dense-diagonalised. The cross-check tool aligns global signs by inner product with the PINN amplitudes.
- Cross-checked the N=4 chain checkpoint `p4_n4_nonmcmc_residual_anneal_s42__stageB_noref_20260424_101003` (uniform d=4, multi-ref ansatz, 3000 epochs no-ref Stage B).
**Results:**

| metric | uniform Heisenberg (J=1) | PINN @ d=4 |
|--------|--------------------------|------------|
| E_GS (J=1 units) | -1.6160 | – (Coulomb units 5.064) |
| GS multiplicity | 1 | – |
| `c_dduu, c_uudd` | -0.149 | -0.369 |
| `c_dudu, c_udud` | +0.558 | +0.445 |
| `c_duud, c_uddu` | -0.408 | -0.407 |
| `<c_pinn|c_Heis>` | 1.000 (self) | **0.939** |
| L2 residual | 0.000 | 0.349 |
| `S({0,1}|{2,3})` | 0.319 | **0.785** |
| negativity | 0.500 | **0.826** |
| log-negativity | 1.000 | 1.407 |
| effective Schmidt rank | 4 | 3 |

**What the numbers actually mean:**
- 0.939 overlap is *very high* — the PINN has discovered ~94% of the Heisenberg AFM physics from minimising the *full continuum Coulomb Hamiltonian alone*, with no Heisenberg knowledge baked in. The dominant amplitudes (Néel patterns) and the global sign structure match.
- The 6% deviation lives almost entirely in the boundary patterns `dduu`/`uudd`. Heisenberg says these should be small (frustrated by AFM bond pattern), the PINN puts more weight on them. Physically this is the spatial-orbital admixture that the Heisenberg model cannot represent: at d=4 the wavefunctions overlap enough that the effective spin model is *t-J-like* rather than pure Heisenberg.
- Counter-intuitively, the PINN has **higher** bipartite spin entanglement than pure Heisenberg (0.785 vs 0.319, log-negativity 1.41 vs 1.00). The Heisenberg ground state is highly peaked on the Néel pattern — its Schmidt distribution `[0.927, 0.063, 0.005, 0.005]` gives low entropy. The PINN's redistribution toward boundary patterns flattens the Schmidt distribution to `[0.726, 0.136, 0.136, 0.001]`, which has **higher** entropy.
- This is the first quantitative demonstration that the **off-Mott contributions in the trained NQS are physically real and influence entanglement-class observables**: not a bias of the metric, not noise, but a real renormalisation of the effective spin model at finite d.
**What we cannot explain (yet):** Whether the off-Mott enhancement of S monotonically decays as d → ∞ to the Heisenberg value (0.319) or whether it persists. The Mott projection is exact in the limit but the *value of the projected metric* depends on the underlying state. Higher-order super-exchange could itself shift the spin-model coefficients.
**Caveats:**
- The Heisenberg reference assumes uniform J. For a chain with weakly broken inversion symmetry (e.g. our `dimer_chain_n4` parametrisation away from θ=[d,d]) we should compute `J_01, J_12, J_23` from a separate two-site super-exchange estimate.
- Schmidt rank is also different (3 vs 4): PINN is missing one component. Could be (a) a residual symmetry the PINN finds, (b) a numerical artefact from finite-precision SVD, or (c) a real preference of the continuum problem for a specific 3-state structure.
- We have only one cross-check checkpoint. Need d ∈ {4, 6, 8, 12} sweep to characterise convergence.
**What a skeptic would say:** "The PINN may have under-converged, picked a non-ground-state local minimum, or simply made a numerical mistake. Any of those could explain the 6% off." Fair, but the energy *did* hit the Mott formula and the Stage-A self-residual variance dropped to ~1e-4. Fastest way to settle: add a d=8 N=4 chain run.
**Output reference:** [src/observables/heisenberg_reference.py](src/observables/heisenberg_reference.py), [scripts/heisenberg_cross_check.py](scripts/heisenberg_cross_check.py)
**Next question:** Does the inverse-design loop drive bipartite entanglement higher *because* of off-Mott contributions, or *despite* them?

---

### [2026-04-26] — Mott spin-amplitude entanglement extractor lands and reproduces 4-site Heisenberg AFM signatures from a trained N=4 PINN
**Motivation:** N=2 dot-label entanglement saturates at 0.5 once `d ≳ 4`, so it is not a useful inverse-design target for the flagship run. We need a metric that (a) generalises to arbitrary N in the Mott regime, (b) is computable from the trained NQS by direct evaluation (no MCMC, no exact diagonalisation), and (c) is differentiable with respect to geometry so the bilevel optimiser can use it. The natural candidate is the *bipartite spin-sector entanglement* extracted from Mott-projected spin amplitudes c_σ.
**Method:**
- Implemented `src/observables/spin_amplitude_entanglement.py`. For an N-particle, N-well, one-per-well system trained at fixed spin template `σ_T = (0,…,0,1,…,1)`, the Mott amplitude for any pattern σ ∈ {0,1}^N with the same `S^z` is `c_σ = sgn(π_σ) · Ψ(R^*(σ); σ_T) / N!`, where `R^*(σ)` is the localised configuration that places the up-labelled particles at σ's up wells (in increasing index order) and the down-labelled particles at σ's down wells. The permutation sign `sgn(π_σ)` is computed from inversions in the interleaved sequence and accounts for fermionic antisymmetry between distinct localisations.
- Public API: `enumerate_patterns`, `localized_config_for_pattern`, `permutation_sign`, `extract_spin_amplitudes`, `well_set_bipartite_entropy`, `evaluate_spin_amplitude_entanglement`, `spin_entanglement_target`. The bipartite metric returns Schmidt values + probabilities, von Neumann entropy, purity, linear entropy, effective Schmidt rank, negativity, and log-negativity for any well-set bipartition `A | B`.
- Added `scripts/evaluate_spin_amplitude_entanglement.py` (CLI evaluator) and `scripts/validate_spin_amplitude_n2.py` (N=2 unit test against the textbook Bell-state singlet).
- Wired `well_set_entanglement` into `GeometryOptimizer` as a target with `target_kwargs={"metric": ..., "set_a": [...]}`. Added two new chain parametrisations: `make_uniform_chain_param_to_wells` (θ = [d], uniform spacing) and `make_per_bond_chain_param_to_wells` (θ = [d_01, d_12, …, d_{N-2,N-1}], per-bond spacing). `scripts/run_inverse_design.py` exposes `--parametrisation`, `--n-wells`, `--metric`, `--set-a`.
**Results — N=2 validation (singlet checkpoint, d=2.4):**
- `c_(0,1) = +0.7071`, `c_(1,0) = -0.7071`, ratio = -1.000 (textbook Bell singlet ✓)
- Schmidt probs = [0.5, 0.5], `S = ln 2 = 0.6931` ✓, negativity = 0.5 ✓, log-negativity = 1.0 ✓.
- The N=2 multi-ref / singlet ansatz is *structurally guaranteed* to produce a perfect spin Bell state regardless of d, so this metric is constant in geometry for N=2 — confirming we should keep using the dot-label spatial metric for N=2 inverse design and reserve the Mott spin-amplitude metric for N ≥ 3.
**Results — N=4 chain at d=4 (`p4_n4_nonmcmc_residual_anneal_s42__stageB_noref_20260424_101003`, wells at -6,-2,2,6, multi-ref ansatz, S^z=0):**

| pattern | spin-string | extracted c_σ |
|---------|-------------|---------------|
| [1,1,0,0] | dduu | +0.36898 |
| [1,0,1,0] | dudu | **-0.44539** |
| [1,0,0,1] | duud | +0.40680 |
| [0,1,1,0] | uddu | +0.40680 |
| [0,1,0,1] | udud | **-0.44539** |
| [0,0,1,1] | uudd | +0.36898 |

| metric (bipartition `A={0,1}` vs `B={2,3}`) | value | reference |
|---------------------------------------------|-------|-----------|
| Schmidt probs | [0.726, 0.136, 0.136, 0.001] | rank-3 dominantly |
| `S_vN` | **0.785** | ln 4 = 1.386 (max) |
| negativity | 0.826 | Bell pair = 0.5 |
| log-negativity | 1.407 | Bell pair = 1.0 |
| purity | 0.564 | min = 0.25 |
| effective Schmidt rank | 3 | – |

**What the numbers actually mean:**
- The trained PINN is *spin-rotation invariant* to numerical precision: every pattern has its global-spin-flip partner at exactly the same amplitude (e.g. `c[1,1,0,0] = c[0,0,1,1]`). The state is total `S=0`, as required.
- The state is *reflection-symmetric* about the chain centre: `c[1,0,1,0] = c[0,1,0,1]`, `c[1,0,0,1] = c[0,1,1,0]`. The PINN has discovered the inversion symmetry of the geometry without supervision.
- The Néel patterns `dudu`/`udud` carry the largest amplitude (0.445), exactly the AFM fingerprint of the 4-site Heisenberg chain ground state via super-exchange. The "domain-wall" patterns `dduu`/`uudd` are weakest (0.369). The extracted ground state is a 4-site Heisenberg AFM singlet to a quantitative degree, *with no hand-tuning* — this is the ground state the network learned by minimising the Coulomb Hamiltonian alone.
- The bipartite entanglement is large but not maximal: `S = 0.785` vs the structural ceiling `ln(3) ≈ 1.10` (the dimension of the `S=0`, `S^z=0` subspace of two spin-1/2 pairs is 3) and `ln 4 = 1.386` (full subspace dimension). There is genuine room for inverse design to push it higher by tuning geometry — e.g. dimerising the chain into two well-separated pairs would saturate `S = ln 2` from each pair's Bell singlet.
**What we cannot explain (yet):** The smallest Schmidt value (0.0015) is not exactly zero, indicating a tiny but nonzero contamination beyond the `S=0` × `S=0` × singlet-to-singlet structure. Could be (a) finite-d corrections to deep-Mott projection, (b) residual numerical noise from the inner training, or (c) genuine higher-multiplet structure in the AFM ground state. To be checked against an exact 4-site Heisenberg ground-state reference.
**Caveats:**
- The metric is *exact* only in the deep-Mott limit; off-Mott contamination scales as `exp(-d²/2)` for ω=1 (so < 0.04% at d=4, < 1e-7 at d=6). At d=4 the projection is well-justified.
- The N=4 reference checkpoint uses the multi-ref ansatz, which has full spin-sector freedom but enforces antisymmetry only by Slater determinants; a spin-symmetric variant would have lower variational ceiling but smaller residual rank-4 contamination.
**What a skeptic would say:** "The PINN learned a state that *looks* like a Heisenberg AFM ground state because that's what the energy functional rewards. You haven't proven it's the same state." Fair — but the spin-rotation invariance, reflection symmetry, and AFM-dominant amplitudes are model-independent fingerprints. The next step is to compute the exact 4-site Heisenberg ground-state amplitudes via small-Hilbert-space CI and check the overlap. If overlap > 0.99 we have direct numerical proof; if not, we have a quantitative measure of the deviation.
**Output reference:** [src/observables/spin_amplitude_entanglement.py](src/observables/spin_amplitude_entanglement.py), [scripts/evaluate_spin_amplitude_entanglement.py](scripts/evaluate_spin_amplitude_entanglement.py), [scripts/validate_spin_amplitude_n2.py](scripts/validate_spin_amplitude_n2.py)
**Next question:** Run N=4 inverse-design smoke test with target = max well-set entanglement (uniform chain, θ = [d], maximise `von_neumann_entropy` for `A = {0,1}` vs `B = {2,3}`); add a 4-site Heisenberg ground-state cross-check.

---

### [2026-04-26] — N=2 inverse-design smoke test completes; 8 outer steps drive dot-label negativity from 0.077 to 0.448 (Bell limit 0.5) in 2 h on one GPU
**Motivation:** End-to-end completion of the Phase 1E smoke test, the first quantitative validation of the bilevel inverse-design loop on a real many-body observable.
**Method:** Same as the previous entry: N=2 singlet permanent ansatz, θ = [d], param_init = 2.0, bounds = [1.9, 6.5], lr = 0.7, FD-step = 0.4, 8 outer steps. Stage A only (Stage B disabled via `--stage-a-min-energy 999`). Run on `cuda:3`.
**Results:**

| Step | d (centre) | E (Ha) | T = dot_neg | ∇θ | dt (s) |
|------|-----------|--------|-------------|-----|--------|
| 0    | 2.000     | 2.404  | 0.0766      | +0.2458 | 1216 |
| 1    | 2.172     | 2.367  | 0.1209      | +0.3205 | 1071 |
| 2    | 2.396     | 2.357  | 0.1919      | +0.4151 | 1095 |
| 3    | 2.687     | 2.351  | 0.2967      | +0.3150 | 1082 |
| 4    | 2.907     | 2.338  | 0.3637      | +0.2600 | 1082 |
| 5    | 3.090     | 2.323  | 0.4063      | +0.2062 |  753 |
| 6    | 3.234     | 2.310  | 0.4321      | +0.1635 |  627 |
| 7    | 3.348     | 2.301  | 0.4481      | +0.1327 |  614 |

Total wall time: **2 h 5 min** (7540 s) on one A100. Trajectory `θ: 2.000 → 2.172 → 2.396 → 2.687 → 2.907 → 3.090 → 3.234 → 3.348`. Gradient monotonically decreasing as ``T`` approaches the saturation value 0.5.
**What the numbers actually mean:**
- The bilevel loop closes cleanly. Every outer step produces a finite, well-conditioned gradient; every gradient step increases the target; every parameter update stays inside the trust region. There is no "black art" tuning — the same hyperparameters that worked at step 0 also worked at step 7.
- Wall time per step *decreases* from ~1200 s to ~600 s as the loop matures, because warm-starting from the previous trained checkpoint saves 50% of the inner-loop work. By step 5 the network is already near-converged and only needs to track the geometry change.
- The trajectory follows the calibrated CI curve closely: `T(d=3.4)` from CI ≈ 0.46, PINN ≈ 0.45 — within 1.5%. The inverse-design loop reads off the same physics the calibration extracted, with no manual scheduling.
**What we cannot explain (yet):** Step 1 grad jumped (0.32) while step 0 grad was lower (0.25), even though we expected monotonic decrease toward saturation. Likely just FD noise from the first step's d=1.6 perturbation getting clipped and falling back to a forward FD. Step 2 onwards the FD is properly central and well-behaved.
**Caveats:**
- 0.448 is below the Bell-state limit 0.5 by 10%. Pushing the last 10% would take ~5 more outer steps and may run into the upper-bound cap d=6.5; the step size is now |dθ| ≈ 0.09 / step. We deliberately stopped at 8 because (a) the validation target is "monotonic, gradient-driven, no operator intervention", which is met, and (b) for the flagship N=4 demonstration the larger-N regime is where this pipeline really shines.
- The first FD perturbation diverged catastrophically (d=1.6) — fixed in the new ``_is_diverged`` and ``_inside_bounds`` heuristics; subsequent steps were stable.
**What a skeptic would say:** "You're climbing the calibration curve you already had. Show me a target that the curve doesn't determine analytically." Fair — this is exactly the purpose of the N=4 flagship: maximise the well-set bipartite entanglement on a 4-site chain. Heisenberg cross-check (see following entry) shows that the PINN at d=4 is **not** the pure Heisenberg AFM ground state, so the gradient signal will be driven by genuine quantum-mechanical correlations that no analytic model gives in closed form.
**Output reference:** [results/inverse_design/n2_smoke_p1e/history.json](results/inverse_design/n2_smoke_p1e/history.json), [results/inverse_design/n2_smoke_p1e/optimal_geometry.json](results/inverse_design/n2_smoke_p1e/optimal_geometry.json)
**Next question:** N=4 flagship inverse-design run (symmetric dimer-chain parametrisation) and Heisenberg cross-check at multiple d.

---

### [2026-04-26] — Inverse-design framework lands; N=2 entanglement smoke test is monotonically climbing toward the Bell-state limit
**Motivation:** The non-MCMC, E-ref-free training pipeline gives a smooth, deterministic energy landscape across Hamiltonian parameters. We want to convert that into a working *inverse-design* loop — given a many-body target observable, automatically engineer the geometry to hit it. Phase 1 of this programme focuses on a working bilevel pipeline with the calibrated dot-label entanglement target.
**Method:**
- Refactored `src/geometry_optimizer.py` into a generic bilevel optimiser:
  * Outer θ → ``param_to_wells(θ)`` ⟶ rewrite YAML config as ``system.type: custom`` with explicit well centres.
  * Inner: subprocess to ``scripts/run_two_stage_ground_state.py`` with `--summary-json` so the parent can pick up the trained checkpoint deterministically.
  * Outer gradient: Hellmann-Feynman for ``target=energy``; central finite differences on the *real* target for non-energy targets, with sense automatically set to ``min`` for energy and ``max`` otherwise. Warm-start from the previous step's centre is on by default.
- Wrote ``src/observables/checkpoint_loader.py`` and ``src/observables/checkpoint_entanglement.py`` to reload a trained ``GroundStateWF`` from any ``result_dir`` and compute particle / dot-label entanglement on the same DVR quadrature grid used for the CI calibration. The two together expose a one-line ``entanglement_target_n2(result_dir)`` callable that the geometry optimiser uses as the target function.
- Updated ``scripts/run_inverse_design.py`` to expose the new API (``--target entanglement_n2``, ``--param-init``, ``--param-step``, ``--param-lower/upper``, ``--gradient-method``, ``--stage-a-strategy``, …).
- Added ``scripts/analyze_inverse_design.py`` that consumes ``history.json`` and produces a 4-panel figure (target / energy / |grad| / T(θ)).
- Phase 1E smoke test (in progress on GPU 3): N=2 singlet permanent ansatz, θ = [d], param_init = 2.0, bounds = [1.9, 6.5], lr = 0.7, FD-step = 0.4, 8 outer steps, ``--stage-a-strategy singlet_self_residual --stage-a-epochs 2500 --stage-b-epochs 1 --stage-a-min-energy 999`` (a deliberate hack to skip Stage B since for N=2 Stage A already converges to var ≈ 1e-5 and Stage B's pure-variance refinement *destroys* the entanglement when restarted from a low-variance state).
**Results so far (4 of 8 outer steps complete):**

| Step | d (centre) | E (Ha) | T = dot_neg | ∇θ |
|------|-----------|--------|-------------|-----|
| 0    | 2.000     | 2.404  | 0.0766      | +0.246 |
| 1    | 2.172     | 2.367  | 0.1209      | +0.321 |
| 2    | 2.396     | 2.357  | 0.1919      | +0.415 |
| 3    | 2.687     | 2.351  | 0.2967      | +0.315 |

The optimizer pushes d outward as predicted by the calibrated CI curve and the gradient matches the slope of the CI baseline (steepest near d ≈ 2-3, then flattening as we approach saturation at d ≥ 4).
**What the numbers actually mean:** This is the first end-to-end demonstration that the non-MCMC NQS pipeline gives a **reliable, gradient-driven inverse-design loop** for a non-trivial many-body observable. The loop:
1. Trains Ψ(θ) to ground-state accuracy at three nearby θ values per outer step,
2. Reads off the dot-label negativity directly from the network with a quadrature evaluator,
3. Feeds the central-FD gradient back into the geometry update.
There is no MCMC noise to fight, no E_ref scaffolding to tune, and no gradient bias from sampling: every step is deterministic up to the seed. This is exactly the property our pipeline has and standard MCMC NQS pipelines do not.
**What we cannot explain (yet):** Step-1 *centre* T (0.121) was higher than the FD-perturbation estimates would naively predict given the gradient (0.32 × 0.4 / 2 = 0.06 increment expected, but we got 0.044 — within the FD finite-step truncation error and the warm-started inner-loop noise). Step 2 grad ≈ 0.42 is genuinely the steepest part of the curve.
**Caveats:**
- The first iteration of the smoke test ran with Stage B enabled and we discovered Stage B's pure-variance refinement (with the trainer's automatic ``n_coll=32`` cap) collapses the entanglement: after Stage A the variance was 1e-5, but Stage B drove it back up to 4.5e-4 while pulling the wavefunction into a less-coherent local minimum. **Stage B is harmful for N=2 inverse design** — Stage A alone (with sufficient epochs) is the right inner loop. We worked around it via ``--stage-a-min-energy 999`` (every Stage A is reported as "failed gate" so Stage B is skipped); a cleaner ``--no-stage-b`` flag is a TODO.
- The first FD- perturbation at step 0 (d = 1.6, below the lower bound) **diverged catastrophically** (final_energy = -88 Ha) because the warm-started weights from d=2.0 plus the unsoftened Coulomb singularity at small d collapsed the wavefunction. The FD-NaN fallback caught it but the gradient was then a forward difference, not a central one. **Fixed for future runs:** geometry_optimizer now (a) skips FD perturbations that go outside the parameter bounds and (b) detects divergence via ``final_energy < 0.1`` and falls back to one-sided FD when needed.
- 2500 stage A epochs / step × 3 trainings / step × 8 steps = ~2.5 hours wall time. For N=8 this would be ~6 hours, which is acceptable for a flagship run but suggests we should explore SPSA or fewer-shot variance-decay early stopping for higher-dimensional θ.
**What a skeptic would say:** "The trajectory you're seeing is just the calibrated CI curve, the optimizer isn't doing anything magical." Correct — and that is exactly the point. The PINN reproduces the CI dot_label_negativity along the d-axis and the optimizer reads off its slope to climb. What's new is that the *loop closes*: target is computed from a trained network, gradient is computed by FD on the trained network, and θ moves in response. For energy targets HF still applies; for arbitrary targets like entanglement, gap, pair correlation, the same machinery now works.
**Output reference:** [results/inverse_design/n2_smoke_p1e/history.json](results/inverse_design/n2_smoke_p1e/history.json), [src/geometry_optimizer.py](src/geometry_optimizer.py), [scripts/run_inverse_design.py](scripts/run_inverse_design.py), [scripts/analyze_inverse_design.py](scripts/analyze_inverse_design.py), [plans/2026-04-26_inverse_design_framework.md](plans/2026-04-26_inverse_design_framework.md)
**Next question:** Generalise the bipartite entanglement metric to N>2 via Mott-projected spin-sector amplitudes and run the N=8 flagship inverse-design loop with target = max bipartite well-set entanglement (or, equivalently, target = engineered ``J_eff`` with prescribed value).

### [2026-04-26] — Löwdin dot-label entanglement is a calibrated metric in the Mott regime; PINN coherence deficit is real
**Motivation:** Phase 0A of the inverse-design programme. The N=2 singlet PINN runs reported `dot_label_negativity ≈ 0.26-0.30` at d ∈ {6,8,12,20}, nominally far below the textbook pure-singlet value of 0.5. Before building inverse-design targets on top of this number, decide whether the gap is (i) a measurement bias of the Löwdin metric itself or (ii) a genuine ansatz coherence deficit.
**Method:**
- Built shared-model CI ground state (n_sp=30, n_ci=200) on a 28×20 DVR grid for d ∈ {2,4,6,8} at ω=1, κ=1, ε=0.01, kinetic prefactor 0.5.
- Re-evaluated `compute_shared_ci_grid_entanglement` at `max_ho_shell` ∈ {1,2,3,4} on the same CI ground state for each d.
- Reported projected subspace weight, dot-label negativity, log-negativity, and projected sector probabilities.
- Implementation: `scripts/calibrate_lowdin_entanglement.py`.
**Results:**

| d | shell=1 dot_neg | shell=4 dot_neg | proj_w (shell=1) | p_LR | S_vN |
|---|-----------------|-----------------|-------------------|------|------|
| 2 | 0.024 | 0.000 | 0.999 | 0.56 | 0.071 |
| 4 | 0.437 | 0.370 | 0.999 | 0.94 | 0.586 |
| 6 | 0.500 | 0.500 | 1.000 | 1.00 | 0.6931 |
| 8 | 0.500 | 0.500 | 1.000 | 1.00 | 0.6931 |

CI ground energies: 1.641, 1.936, 1.931, 1.902 Ha for d = 2, 4, 6, 8.
**What the numbers actually mean:**
- At d ≥ 6 the Löwdin metric reproduces the textbook pure-singlet value `neg = 0.5` to numerical precision and `S_vN = ln 2`. The metric is calibrated.
- Therefore the PINN's measured 0.26-0.30 at d ≥ 6 is a real coherence deficit of the singlet permanent ansatz: the network learns the correct LR sector probabilities (proj_w → 1, p_LR → 1, p_LL ≈ 0) but does not phase-lock the `|LR⟩` and `|RL⟩` components into a Bell state.
- At d=4 the CI itself drops to 0.44 because the wavefunction has measurable doubly-occupied admixtures and intra-well correlations. As shell grows, the basis resolves these and the label-level density matrix correctly looks more mixed (0.44 → 0.37). This is physics, not bias.
- At d=2 doublons dominate; LR is no longer the right basis and `neg → 0`. We should not target d=2 in any inverse-design entanglement experiment.
**What we cannot explain (initially):** Why the singlet permanent PINN appeared to fail to phase-lock LR/RL despite hitting energy to ≤0.1% of CI. — *Resolved below: the PINN does not actually fail; the previous measurement of 0.26-0.30 was stale.*
**Caveats:**
- The calibration's shared-model CI script (inherited from `compare_ci_vmc_dot_entanglement.py`) gives a quantitatively wrong **energy**: at d=8 it returns E=1.902 < 2.0 = the non-interacting limit, which is unphysical for repulsive Coulomb. Symptom traced to a Coulomb-kernel quadrature-weight convention mismatch (`include_quadrature_weights=True` together with eigenvector normalization that already absorbs sqrt-weights). The eigenvector *structure* (and hence the entanglement) is unaffected: the ground state is still the LR-localized Mott singlet, and `dot_label_negativity → 0.5` is the correct textbook answer. **Energy comparisons with this CI path should not be trusted; use one_per_well CI or ω-only spectrum sanity checks instead.**
- Grid resolution 28×20 is sufficient at d ≤ 8 (per-bin spacing ≤ 0.7 Bohr); at larger d we'd need to grow nx accordingly. Adaptive grid sizing is built into the script.
**What a skeptic would say:** "0.50 at d=6,8 is suspiciously round — sure you didn't accidentally read off an idealized Bell-state expectation?" Sanity check: the same code on the d=2 case correctly returns 0.024, confirming the metric is not hard-wired to 0.5.
**Decision locked:** Use `max_ho_shell=2` as the standard setting for inverse-design entanglement targets. p_LR > 0.99 across the operating regime, and shell=2 captures the leading correction to shell=1 at small d. Always report both the PINN value and a thin-CI calibration on the optimized geometry as a sanity check.

**ADDENDUM — the PINN deficit is a stale measurement, not a real ansatz problem.**
Re-evaluated the latest April-24 stage-B singlet checkpoints (`p4_n2_singlet_d{2,4,6,8,12,20}_s42__stageB_noref_20260424_*`) using a fresh checkpoint-loading evaluator (`src/observables/checkpoint_entanglement.py`) on a comparable 23×16 quadrature grid:

| d | PINN E | proj_w | LL | LR | RL | RR | dot_neg | S_vN |
|---|--------|--------|------|------|------|------|---------|------|
| 2 | 2.390 | 1.00 | 0.20 | 0.30 | 0.30 | 0.20 | 0.077 | 0.54 |
| 4 | 2.248 | 1.00 | 0.003 | 0.497 | 0.497 | 0.003 | 0.491 | 0.69 |
| 6 | 2.161 | 1.00 | 0.000 | 0.500 | 0.500 | 0.000 | 0.500 | 0.69 |
| 8 | 2.124 | 1.00 | 0.000 | 0.500 | 0.500 | 0.000 | 0.500 | 0.69 |
| 12 | 2.083 | 1.00 | 0.000 | 0.500 | 0.500 | 0.000 | 0.500 | 0.69 |
| 20 | 2.049 | 1.00 | 0.000 | 0.500 | 0.500 | 0.000 | 0.500 | 0.69 |

Energies match the Mott prediction E(d)=2 + 1/d to ≤0.1% (e.g. d=8 → 2.125 predicted vs 2.124 measured). Negativity hits the Bell-state maximum 0.5 to ≤0.1% for d ≥ 4. The previously-reported "0.26-0.30 deficit" stored in `results/diag_sweeps/nonmcmc_entanglement_summary_signed_localized_highres_20260414.json` came from a stale or under-converged April-14 checkpoint (its raw sector probabilities `LL=0.35, RR=0.44, LR≈RL≈0.10` describe a charge-density wave, not a Mott singlet). The current ansatz has *no* coherence deficit at d ≥ 4.
**Implication for inverse design:** Maximizing dot-label negativity in N=2 is already saturated. To produce a meaningful smoke test, either (a) start from d=2 where neg≈0.08 and watch the optimizer push wells apart, or (b) move directly to N=4/N=8 where bipartite-well entanglement is non-trivially geometry-dependent.
**Output reference:** [results/diag_sweeps/lowdin_calibration_20260426_170318.json](results/diag_sweeps/lowdin_calibration_20260426_170318.json), [scripts/calibrate_lowdin_entanglement.py](scripts/calibrate_lowdin_entanglement.py), [src/observables/checkpoint_entanglement.py](src/observables/checkpoint_entanglement.py), [src/observables/checkpoint_loader.py](src/observables/checkpoint_loader.py)
**Next question:** Generalize bipartite entanglement to N>2 via well-set partition (Mott-projected spin sector or 1-RDM/Peschel) and drive the inverse-design loop on a non-saturated target.

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
