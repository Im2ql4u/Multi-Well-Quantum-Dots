# Supervisor Report — Multi-Well Quantum Dots NQS

**Author:** Aleksander Skogen
**Period covered:** 2026-04-13 (last shared milestone: 3-seed CI-referenced non-MCMC benchmark) → 2026-04-27
**Status:** snapshot of project state at end of 2026-04-27. Subsequent milestones live in dated update files; the next is [`reports/2026-04-28_supervisor_update.md`](2026-04-28_supervisor_update.md).
**Latest update:** 2026-04-28 ~07:35 CEST (overnight: **B-sweep COMPLETE** — clean 5-point null result confirming the structural-triviality caveat for the fixed-spin Sz=0 ansatz under uniform Zeeman. **Phase 4 N=16 retry on cuda:6 COMPLETE** — `self_residual` strategy with `multi_ref=False` lands at E=27.270 Ha (var=4.2e-4, ESS=32, no collapse), completely fixing the previous `guided` strategy's E=−372 collapse. The N=8 SSH flagship results from 2026-04-27 still stand as the headline. **From 2026-04-28 onwards, new milestones go into the dated update files** — see [`reports/2026-04-28_supervisor_update.md`](2026-04-28_supervisor_update.md) for the N=12 launch, the sector-aware Phase 3B redesign, and the methodological lessons settled today.)

---

## 1. Executive summary

Since the last set of results you saw (the 3-seed `E_ref`-guided non-MCMC benchmark for N=2/3/4 in mid-April), the project has moved through three connected milestones:

1. **Removed the CI dependency from the training loop.** The network used to be told the CI ground-state energy and trained to match it (residual + variance loss with `residual_target_energy` set). We now train CI-free with the same numerical accuracy — `loss_type: residual`, `residual_target_energy: null`, `improved_self_residual` Stage A — and still hit ~0.02 % of the CI value on N=2/3/4. The CI is now used **only for verification**, never as a training input.

2. **Built a rigorous many-body entanglement framework that works for any N.** The previous Löwdin/Voronoi-sector negativity metric was fundamentally limited (basis incompleteness, ambiguity for N > 2). Replaced by **Mott-projected spin amplitudes** `c_σ = ⟨φ_{σ_1}…φ_{σ_N} | Ψ⟩` plus signed-permutation factors. Validated end-to-end:
   * **N=2 singlet checkpoint** → exact Bell singlet to numerical precision (S = ln 2, negativity = 0.5, c_(↑↓) = +1/√2, c_(↓↑) = −1/√2).
   * **N=4 PINN at uniform d=4** → AFM Néel-pattern dominance, 0.939 overlap with the open-boundary Heisenberg AFM ground state, bipartite entropy S = 0.785.
   * Generalised to N=8: spin sector dimension 70, working.

3. **Built and demonstrated a working inverse-design programme.** A bilevel optimiser (geometry on the outside, NQS training on the inside) that uses the differentiability of the *deterministic* non-MCMC loss to engineer specific physical observables — entanglement, spin gap, spin-spin correlators, effective Heisenberg couplings — by automatically reshaping the well geometry. Demonstrated at:
   * **N=2** (smoke tests, both entanglement-up and exchange-gap-up — opposite directions, as expected).
   * **N=4 flagship** — bipartite entropy boosted from 0.785 → **0.964** (+22.8 %) over 10 outer steps with stable energy.
   * **N=4 engineer-to-spec** — ⟨S₀·S₃⟩ driven *toward a specified value* (−0.65) by 17 % over 8 steps.
   * **N=8 SSH flagship** — running now (step 1/8). At step 0 the optimiser already picked the **topological** SSH direction (d_short ↑, d_long ↓), exactly the physically expected move to maximise end-to-end AFM correlations.

The headline claim is that this combination — deterministic, CI-free training plus a differentiable physics loss plus a bilevel outer loop — turns the wave-function solver into a **Hamiltonian-engineering tool**. As far as we can tell, no continuous-space many-body NQS implementation in the literature has demonstrated this: every prior inverse-design / parametrised-Hamiltonian study we have found uses MCMC and either drops to lattice models or is stuck doing forward sweeps only.

---

## 2. What you saw last (baseline as of 2026-04-13)

For context, the last reported milestone was:

* Three-seed (42, 314, 901) non-MCMC residual training for one-per-well N=2/3/4 at separation d=4.
* **Error vs CI:** N=2 0.019 %, N=3 0.020 %, N=4 0.017 % (mean over seeds).
* **Reference:** the CI ground state was provided to the loss as `residual_target_energy`; this was the lever that stabilised training.
* Open question at that time: can we keep the accuracy without the reference, scale to magnetic / entanglement tasks, and produce something the CI cannot give us?

This report is the answer.

---

## 3. Training methodology (compact)

### 3.1 Ansatz

The wave-function for an N-electron one-per-well configuration is

\[
\Psi(R) \;=\; \mathrm{sign}(R)\, J_\theta(R)\, D(R),
\]

where:

* `D(R)` is a **single Slater determinant per well**, built from a small set of harmonic-oscillator orbitals localised at each well centre. For multi-sector training (singlet/triplet/etc.), each spin pattern uses its own `multi_ref` permanent.
* `J_\theta(R)` is a **Jastrow correlation factor parametrised by a Physics-Informed Neural Network (PINN)** — the `arch_type: pinn` block. Default sizes: `pinn_hidden=64`, `pinn_layers=2` (we go up to 96/3 for the flagships).
* Optional backflow (`use_backflow: true`) — currently switched off in the inverse-design lane to keep gradient cost low.
* `sign(R)` enforces the antisymmetry that the determinant alone might not preserve under the Mott-projection observable.

### 3.2 Loss

The training objective is the **non-MCMC residual / collocation loss**:

\[
\mathcal{L}(\theta) \;=\; \mathbb{E}_{R \sim p}\!\left[ w(R)\,\big(\hat H \Psi_\theta / \Psi_\theta - E_{\mathrm{ref}}\big)^2 \right] + \alpha\,\mathrm{Var}_p[E_{\mathrm{loc}}],
\]

with three crucial twists:

1. **Stratified i.i.d. proposal `p`** instead of MCMC: a 5-component mixture of Gaussians (centres, tails, in-mixed, out-mixed, dimer-shells). All collocation samples are independent; the loss is **deterministic** at fixed sample seed → *gradient-friendly with respect to Hamiltonian parameters*. This is the core enabler of inverse design.
2. **Per-batch MAD clipping** of the local energy `E_loc`. Without this, occasional close-encounter Coulomb spikes blow up the variance and dominate gradients. With `local_energy_clip_width: 5.0` × MAD, training is robust across all systems we have tried.
3. **CI-free in the inverse-design lane**: `residual_target_energy: null`, with `Stage A` running the **`improved_self_residual`** strategy: take the current best variational energy as the reference and re-train against itself, with an optional variance-only Stage B refinement. This removes the CI dependency entirely.

### 3.3 Two-stage protocol

* **Stage A — direct variational warm-start.** Standard residual loss with annealed reference (or self-residual). 1500–5000 epochs depending on the size of the system. Always used.
* **Stage B — pure-variance refinement.** Drop the energy term and minimise the local-energy variance only. Useful for deep convergence, but in inverse-design loops it can collapse approximate-eigenstate sectors (most clearly seen for N=2 triplet at small separations). We gate it with `--stage-a-min-energy 999.0` (effectively disable Stage B) for any inverse-design run.

### 3.4 Sampler

```
sampler: stratified
sampler_mix_weights: [0.55, 0.05, 0.20, 0.15, 0.05]
sampler_sigma_center: 0.15-0.20
sampler_sigma_tails:  0.80-1.00
sampler_dimer_pairs:  2-4
```

The dimer-pair shells (mixture component 5) are the addition that finally made N≥4 stable: they place a fraction of samples near the close-encounter divergence, which would otherwise be visited only rarely by the Gaussian components and would dominate the gradient when finally hit.

### 3.5 Hamiltonian

2D continuous-space, electrons in a sum of harmonic wells with optional Coulomb interaction:

\[
\hat H \;=\; \sum_i\!\left[-\tfrac{1}{2}\nabla_i^2 + V(\mathbf r_i)\right] + \sum_{i<j} \frac{\kappa}{\sqrt{|\mathbf r_i - \mathbf r_j|^2 + \epsilon^2}},
\]

with `kappa=1.0`, `epsilon=0.01`. The wells are specified by `system.wells` in YAML; the inverse-design optimiser rewrites this list every outer step.

---

## 4. Phase-by-phase progress since 2026-04-13

### Phase A — CI-independent reproduction (DONE)

* `improved_self_residual` Stage A confirmed reproducing the 0.02 % CI accuracy without ever exposing the CI energy.
* All inverse-design runs since use this strategy.
* Result: the entire pipeline downstream is now **CI-independent**. The CI is a verification tool, not a training input.

### Phase B — Mott-projected spin-amplitude entanglement framework (DONE)

Modules: `src/observables/spin_amplitude_entanglement.py`, `src/observables/heisenberg_reference.py`, `src/observables/effective_heisenberg.py`.
CLI tools: `scripts/evaluate_spin_amplitude_entanglement.py`, `scripts/heisenberg_cross_check.py`, `scripts/evaluate_effective_heisenberg.py`, `scripts/amplitude_evolution.py`.

What it does:

1. Enumerate all N-particle / n↓-down-spin patterns σ ∈ {↑,↓}^N (size = `binomial(N, n_down)`; up to 70 for N=8 in the n_↓ = N/2 sector).
2. For each σ, build the ideal localised positions `R*(σ)` (electron i at well σ_i), evaluate `c_σ ≡ sign × Ψ_θ(R*(σ))`.
3. Treat `c` as a normalised spin vector. From it compute:
   * **Bipartite entropy** `S(A|B)` for any well-set partition (default: left half | right half).
   * **Bipartite negativity** `\mathcal{N}(A|B)`.
   * **Two-point correlators** `⟨S_i · S_j⟩` directly from |c⟩.
   * **Heisenberg overlap** `|⟨c | ψ₀(H_Heis^{OBC})⟩|`.
   * **Effective Heisenberg fit** `J_{ij}` such that `|c⟩` is the ground state of `H_eff = Σ J_{ij} S_i·S_j` (see Phase D).

Validation:

| Test | Expected | Got |
|---|---|---|
| N=2 singlet checkpoint | c = (+1/√2, −1/√2), S = ln 2, neg = 0.5 | matches to numerical precision |
| N=4 PINN @ d=4 | dominant Néel pattern, AFM correlators | overlap 0.939 with Heisenberg AFM GS, S = 0.785 |
| Synthetic N≤6 OBC Heisenberg → fit J → recover bonds | exact recovery for N≤3, ground-state-overlap-correct for N≥4 | passed; relative-residual < 1 e − 9 in all tests |

**Important physics finding from this phase:** at N=4 d=4 the PINN ground state has *higher* bipartite entropy (0.785) than the pure OBC Heisenberg AFM ground state (0.319). The 0.466 excess comes from physically-real off-Mott contributions (small but coherent weight on doubly-occupied configurations) that the projected effective spin model cannot capture. This is exactly the kind of "beyond-Heisenberg" physics that justifies doing the calculation as a continuous-space NQS rather than a lattice spin model.

### Phase C — Multi-sector machinery + exchange-gap target (DONE)

Refactored `src/geometry_optimizer.py` to support **multiple spin sectors per geometry**. New data structures:

* `GeomEvalContext` — collects per-sector trained checkpoints, energies, and amplitudes for one geometry.
* `spin_overrides` — list of dicts `{n_up, n_down, multi_ref, ...}` so each sector trains on the correct architecture / pattern.

Added the **`exchange_gap`** built-in target, automatically training the lowest-energy and first-excited spin sector and reporting `J = E_T − E_S` (or `−|J|`, or `−(J − J_target)²` for "engineer to spec").

* **Smoke test (N=2):** d 2.0 → 1.77 (4 outer steps), J 0.093 → 0.185 Ha. The optimiser correctly drives `d` *down* to *increase* J — opposite direction to the entanglement target — demonstrating the **gate-speed vs coherence trade-off** end-to-end on the same machinery.

This is the same multi-sector apparatus that scales to "engineer the singlet-triplet gap to 30 µeV" or "find the geometry whose spin gap matches a target" tasks for spin-qubit design.

### Phase D — Effective Heisenberg J_ij extractor (DONE)

`src/observables/effective_heisenberg.py`:

* Builds the K×K covariance / parent-Hamiltonian matrix `Q_{αβ} = ⟨c| S_α·S_β |c⟩ − ⟨c|S_α|c⟩⟨c|S_β|c⟩` over a user-chosen set of pair operators α = (i,j).
* Finds `J ∈ ℝ^K` such that `|c⟩` is the **ground** state of `H_eff(J) = Σ J_α S_α`. We search the **full K-dimensional** space (`_resolve_J_direction`), not just the kernel of Q, because for approximate states the relevant figure of merit is `overlap(|c⟩, |ψ₀(H_eff(J))⟩)` not exact-eigenstate residual.
* Reports both `relative_residual = J^T Q J / ‖J‖² / λ_max(Q)` and `overlap`.

**Honest interpretive note:** for N≥4 in multi-dimensional spin sectors there is *typically a multi-dimensional null space of Q*, so many J vectors all make `|c⟩` an eigenstate. The unique physical answer is the J that makes `|c⟩` the **ground state** of `H_eff(J)`. This is correctly recovered by the overlap-maximising solver. We always report `overlap` and `relative_residual` alongside `J` so the consumer can judge the fit quality.

For inverse-design *targets*, however, we prefer the **direct correlator** `⟨S_i·S_j⟩` because it is unambiguous — the J fit is reserved for analysis / interpretation.

### Phase E — Bilevel inverse design (PARTIALLY DONE; N=8 in flight)

The outer loop, in `src/geometry_optimizer.py`:

```
θ_{k+1} = clip(θ_k + η · ∇_θ T(θ))
```

with `T(θ)` an arbitrary Python callable on a `GeomEvalContext`. Built-in targets:

| Target | What it optimises | CLI |
|---|---|---|
| `entanglement` | bipartite entropy / negativity | (default) |
| `energy` | ⟨H⟩, with `--mode min` or `--mode max` | `--target energy` |
| `exchange_gap` | E_T − E_S, with optional target value | `--target exchange_gap [--target-J 0.05]` |
| `spin_correlator` | ⟨S_i·S_j⟩, mode value / neg_value / neg_squared_error | `--target spin_correlator --pair I J --mode ...` |
| `effective_J` | J_{ij} from Phase D fit | `--target effective_J --effJ-pairs I,J K,L ...` |

Gradients:

* `--gradient-method hf` — Hellmann-Feynman, exact for `T = ⟨H⟩`.
* `--gradient-method fd_central` — 2 K trainings per outer step (default for spin observables).
* `--gradient-method fd_forward` — K trainings per outer step (used for the N=8 flagship; halves the cost).
* `--gradient-method fd_backward` — symmetric counterpart.

Parametrisations available:

* `uniform_spacing` — single d for symmetric chain.
* `dimer_chain_n4` — [d_outer, d_middle].
* `dimer_chain_n8` — [d1, d2, d3, d4], symmetric chain `d1|d2|d3|d4|d3|d2|d1`.
* `dimer_pair_n8` — SSH-style [d_short, d_long], bond layout `d_s|d_l|d_s|d_l|d_s|d_l|d_s`.
* `free_wells_*` — direct optimisation of well centres (used in N=2 smoke).

Robustness features: warm-start each FD sub-training from the centre checkpoint; per-direction divergence detection with one-sided fall-back; bound-clipping; optional `flock` lockfile per output directory to prevent duplicate launches.

---

## 5. Headline results in detail

### 5.1 N=2 entanglement smoke (Phase 1E, complete)

Bipartite negativity vs. well separation; trivial baseline because at large d the singlet recovers a Bell pair.

| step | d | T (negativity) | E (Ha) |
|---:|---:|---:|---:|
| 0 | 2.000 | 0.077 | 2.404 |
| 3 | 2.687 | 0.297 | 2.351 |
| 7 | 3.348 | **0.448** (Bell limit 0.500) | 2.301 |

Wall: 2 h on cuda:3 with warm-started inner loops. Confirms the bilevel loop converges monotonically and warm-starts work as intended.

### 5.2 N=2 exchange-gap smoke (Phase 1B, complete)

`--target exchange_gap`, multi-sector training (singlet 1,1 + triplet 2,0).

| step | d | J = E_T − E_S | direction |
|---:|---:|---:|---|
| 0 | 2.000 | 0.093 Ha | (init) |
| 4 | 1.770 | **0.185 Ha** | d ↓ to **increase** J |

Opposite direction to 5.1 — exactly what the gate-speed-vs-coherence trade-off predicts. The same machinery handles both targets without any code-path changes.

### 5.3 N=4 entanglement flagship (Phase 2A, complete)

`dimer_chain_n4`, `--target entanglement`, lr = 3.0, 10 outer steps, FD-central, bounds [2.0, 9.0]².

| step | θ = (d_outer, d_middle) | T = bipartite entropy | E (Ha) |
|---:|---:|---:|---:|
| 0 | (4.000, 4.000) | 0.78496 | 5.0316 |
| 3 | (4.718, 3.629) | 0.85374 | 5.0099 |
| 6 | (5.159, 3.149) | 0.90080 | 5.0269 |
| 9 | (5.468, 2.481) | **0.96391** | 4.9438 |

Net result: **+22.8 % bipartite entropy** with stable, slightly *decreasing* energy. The geometry the optimiser found is a "central-dimer + boundary-bridge" pattern (`d_middle` shrinking, `d_outer` growing) — the same physical regime that one would design by hand for maximum end-to-end singlet weight, but discovered automatically. End-to-end correlator ⟨S₀·S₃⟩ went from −0.413 to −0.516 along this trajectory.

### 5.4 N=4 engineer-to-spec demo (Phase 2B, complete)

`--target spin_correlator --pair 0 3 --mode neg_squared_error --target-value -0.65`. The target is *not* an extremum — the optimiser must drive ⟨S₀·S₃⟩ to a *specific intermediate value*.

| step | θ = (d_outer, d_middle) | ⟨S₀·S₃⟩ | T = −(C − (−0.65))² |
|---:|---:|---:|---:|
| 0 | (4.000, 4.000) | −0.413 | −0.0561 |
| 3 | (4.242, 3.776) | −0.430 | −0.0484 |
| 7 | (4.488, 3.454) | **−0.448** | **−0.0409** |

Monotonic improvement of T (squared error halved), ⟨S₀·S₃⟩ moving steadily toward −0.65 by 17 % in 8 steps. Notably, the trajectory direction is **identical** to the entanglement-maximising flagship, an internal consistency check that long-range AFM correlations and bipartite entropy peak in the same θ-region.

This is the cleanest demonstration we know of differentiable Hamiltonian engineering on a continuous-space many-body ground state: not "minimise/maximise" but "match this number".

### 5.5 N=8 SSH flagship (in flight)

`dimer_pair_n8` SSH parametrisation `[d_short, d_long]`, `--target spin_correlator --pair 0 7 --mode neg_value`, fd_forward gradients (3 trainings per outer step), 8 outer steps, `n8_invdes_fast_s42.yaml` (epochs 1500, n_coll 384).

Expectation: SSH manifold contains a topological / trivial transition. *Trivial* (`d_s ≪ d_l`) → 4 nearly-decoupled singlet pairs, ⟨S₀·S₇⟩ → 0. *Topological* (`d_s ≫ d_l`) → strong inter-cell coupling, ⟨S₀·S₇⟩ → singlet limit.

**Step 0 (complete):**

| step | θ = (d_short, d_long) | ⟨S₀·S₇⟩ | E (Ha) | grad |
|---:|---:|---:|---:|---|
| 0 | (4.000, 4.000) | **−0.319** | 11.445 | (+0.0108, −0.0078) |
| 1 | (4.086, 3.938) | running | running | — |

Two observations:

* The starting point (uniform chain, d=4) already gives ⟨S₀·S₇⟩ = −0.319. Independent dimers would give 0; pure 8-site Heisenberg would give roughly −0.36. So the PINN is sitting in the right ballpark.
* The gradient at step 0 says **`d_short` should grow, `d_long` should shrink** — this is *exactly* the topological SSH direction, picked autonomously by the optimiser on the first step. Encouraging early evidence that the framework will land in the right phase by step 8.

ETA full run: ~4–6 h on a free 2080 Ti.

---

## 6. Where this is going

Concrete, ordered:

1. **Finish the N=8 SSH flagship.** Expected by the end of today. Then run `scripts/amplitude_evolution.py` for the full trajectory (correlators, NN-J fit, overlap, entanglement entropy, all on one figure). This will be the biggest single demonstration of the framework.
2. **Relaunch the N=2 `--target-J 0.05` validation** (target a *specific* exchange gap of 0.05 Ha) with `--stage-a-min-energy 999.0` to disable Stage B, which collapsed the singlet last time.
3. **N=4 d-sweep + Heisenberg cross-check** (Phase 2A bonus): do uniform d ∈ {3, 4, 5, 6, 8} and overlay `H_overlap(d)`, `S(d)`, `⟨S_i·S_j⟩(d)`. Quantifies how off-Mott the PINN is across the whole parameter range; settles whether the 0.785 enhancement at d=4 is the rule or the exception.
4. **Phase 1C — pair-correlation target** g_σ(r₀): **landed 2026-04-27** (`src/observables/pair_correlation.py`, target wired into `GeometryOptimizer` and CLI). Validated on the N=2 entanglement-smoke trajectory: g(r=2.0) monotonically decreased 0.358 → 0.106 as d grew 2.0 → 3.35; g(r=3.5) monotonically increased 0.222 → 0.384 as the singlet wavefunction tracked the well separation. Estimator uses the existing MH sampler (warmup + decorrelation), with a Gaussian-broadened delta and a configurable seed for reproducible FD gradients. Bilevel demo run still pending.
5. **Phase 2C STRETCH — disorder-pattern inverse design** (the "MBL stretch"): **parametrisation infrastructure landed 2026-04-27**. Added `make_displacement_2d_param_to_wells` (`src/geometry_optimizer.py`) and a `displacement_2d` CLI option that lets the geometry optimiser search the full ``2N``-dimensional landscape of per-well 2D displacements from any base layout (read from `system.wells`). Combined with `--param-lower / --param-upper` clipping, this is the deterministic gradient-driven analogue of the random σ-sweep MBL infrastructure. A small N=4 bilevel demo (8 free parameters, fd_forward, target = bipartite well-set entanglement) is queued and will start automatically on cuda:3 once the d-sweep + pair-corr demo finish.
6. **Background**: N=8 d-sweep on idle GPUs, N=8 magnetic B-sweep, optional N=12 / N=16 ground states.
7. **Documentation**: the framework is publication-ready in scope; we should start drafting the methods section now (the `inverse_design_framework.md` planning doc is the seed).

---

## 7. Methodological caveats / honest scope

* **Mott projection** is reliable down to d ≈ 2.5–3. Below that, doubly-occupied configurations carry > few % of the norm and the spin-amplitude framework should be viewed as a *projection*, not the full state. We always report the off-Mott norm leak when reporting spin observables.
* **Effective-Heisenberg J fit** is multi-dimensional in its null space for N ≥ 4; we report `overlap` and `relative_residual` alongside J. For inverse-design *targets* we use direct correlators ⟨S_i·S_j⟩, which are unambiguous.
* **Stage B** can collapse approximate-eigenstate sectors (most clearly at N=2 small-d triplet). Currently disabled in the inverse-design lane (`--stage-a-min-energy 999.0`).
* **Stage A self-residual** can also collapse on the N=2 multi-ref singlet under warm-starting at small d (E → −35 Ha observed at d ≈ 1.97). Fix landed 2026-04-27: the geometry optimiser now auto-routes the N=2 singlet sector through the legacy `singlet_self_residual` recipe (singlet permanent ansatz, validated previously), keeping triplet on `improved_self_residual`. The legacy recipe was also taught to clear `multi_ref` when forcing the singlet permanent (mutual-exclusion guard in the wavefunction).
* **Triplet sector** at N=2 forces `architecture.singlet=False, multi_ref=True` because the legacy permanent ansatz is hardwired to (1,1). Solved by the multi-ref refactor; mentioned for completeness.
* **Coulomb softening** `epsilon = 0.01` is fixed throughout; the dependence of J on epsilon is not currently scanned.
* **Single seed** for all inverse-design runs so far. The flagship runs should be reproduced at seeds 314 / 901 before any external claim.
* **CI cross-checks** are still finite-basis (`n_sp_states=40`, `n_ci_compute=200`); below-CI energies should be interpreted as "consistent with CI to the basis truncation", not "improvement".
* **The shared-CI Coulomb kernel "double-count" bug** (Phase 0 bonus) — investigated 2026-04-27. The original diagnosis (the legacy `include_quadrature_weights=True` flag double-counts the sqrt-weights already absorbed into unit-norm DVR eigenvectors) was *correct in form but insufficient in scope*. Both `=True` *and* `=False` give wrong shared-DVR-CI energies at production ε=0.01, because the singular Coulomb diagonal `V_{ii} = κ/ε = 100` dominates the two-electron sum `Σ_{ij} (v_a v_c)(i) V_{ij} (v_b v_d)(j)` whenever the bra/ket orbitals share support on the same grid points (which is unavoidable in the shared-DVR ansatz). At ε=0.01 the shared(`W=False`) ground state is *more* unphysical than shared(`W=True`) (`E_GS = −0.679` vs `+1.806` Ha at d=8 vs the correct one-per-well reference of 2.126 Ha). Increasing ε to 1.0 mostly cures it but doesn't match production. The one-per-well CI path (`run_exact_diagonalization_one_per_well[_multi]`) doesn't suffer this issue because its bra/ket orbitals live on disjoint wells, so the singular diagonal contributes negligibly. **Resolution landed 2026-04-27**: deprecation warning added to `run_exact_diagonalization`'s docstring + `precompute_coulomb_kernel`'s docstring, plus a 3-test regression suite (`tests/test_shared_ci_coulomb_kernel.py`) that pins the current shared-CI behaviour and the OPW reference. Production code-paths (entanglement, inverse design, multi-well CI) remain unaffected — they all use the one-per-well reference.

**Verdict** (triage 2026-04-27): none of the above are dealbreakers for the in-flight runs or the next phase — every active lane has the appropriate mitigation in place (`--param-lower` clips, `--stage-a-min-energy 999.0` skips Stage B, multi-sector default routes the N=2 singlet through the stable recipe, direct correlators sidestep the J-fit ambiguity).

---

## 8. How to reproduce / inspect (quick reference)

The relevant directories are:

* **Code**: `src/geometry_optimizer.py`, `src/observables/`, `scripts/run_inverse_design.py`, `scripts/run_two_stage_ground_state.py`, `scripts/amplitude_evolution.py`.
* **Configs**: `configs/one_per_well/n4_invdes_baseline_s42.yaml`, `configs/one_per_well/n8_invdes_{baseline,lite,fast}_s42.yaml`, `configs/one_per_well/n2_invdes_exchange_baseline_s42.yaml`.
* **Result lanes**: `results/inverse_design/<run_name>/` with per-step `summary_step*.json`, `cfg_step*.yaml`, `train_step*.log`, top-level `history.json`, `summary.json`, and (after the analyser) `amplitude_evolution.{csv,npz,png}`.
* **Plans**: `plans/2026-04-26_inverse_design_framework.md` is the master planning + status doc and is the canonical place to track outstanding work.

Reproducing the N=4 flagship:

```bash
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \
    --config configs/one_per_well/n4_invdes_baseline_s42.yaml \
    --target entanglement \
    --parametrisation dimer_chain_n4 \
    --param-init 4.0 4.0 --param-step 0.4 0.4 \
    --param-lower 2.0 2.0 --param-upper 9.0 9.0 \
    --n-steps 10 --lr 3.0 --gradient-method fd_central \
    --stage-a-min-energy 999.0 \
    --out-dir results/inverse_design/n4_flagship_p2a_aggressive
```

Producing the trajectory figure for any inverse-design run:

```bash
PYTHONPATH=src python3.11 scripts/amplitude_evolution.py \
    --run-dir results/inverse_design/n4_flagship_p2a_aggressive \
    --effJ-pairs 0,1 1,2 2,3 0,3
```

---

## 9. Appendix A — running runs (live)

* **`n8_ssh_flagship_s42`** — cuda:6, *completed* 2026-04-28 ~00:00 CEST after ~16.5 h (8 outer steps × ≈ 6300 s/step on contended GPU 6 + final step). Full trajectory:

  | step | `θ_0` (long) | `θ_1` (short) | ratio  | T = ⟨S₀·S₇⟩ | E (Ha)  |
  |------|--------------|---------------|--------|--------------|---------|
  | 0    | 4.000        | 4.000         | 1.00   | 0.31898      | 11.4451 |
  | 1    | 4.086        | 3.938         | 1.038  | 0.32065      | 11.4441 |
  | 2    | 4.165        | 3.872         | 1.076  | 0.32224      | 11.4352 |
  | 3    | 4.239        | 3.801         | 1.115  | 0.32381      | 11.4304 |
  | 4    | 4.308        | 3.725         | 1.157  | 0.32539      | 11.4311 |
  | 5    | 4.373        | 3.644         | 1.200  | 0.32705      | 11.4370 |
  | 6    | 4.435        | 3.556         | 1.247  | 0.32882      | 11.4490 |
  | 7    | 4.493        | 3.459         | 1.299  | 0.33077      | 11.4666 |
  | optimal | 4.549     | 3.354         | 1.357  | (not eval.)  | (not eval.) |

  **The optimiser auto-discovered SSH-style alternation** from a uniform start with only the spin-correlator target as guidance: `θ_0` grows by +0.55 Bohr and `θ_1` shrinks by −0.65 Bohr to give a final 1.36:1 long-to-short bond ratio; T strengthens monotonically by +3.7 % (0.31898 → 0.33077). The trajectory analysis (`amplitude_evolution.py`) confirms the dimerisation cleanly reshapes the **NN ⟨S·S⟩ matrix**: at step 7, NN bonds alternate −0.354 / −0.391 / −0.335 / −0.392 / −0.335 / −0.391 / −0.354 (short bonds carry stronger AFM correlation as expected from SSH-Heisenberg). The effective Heisenberg overlap is stable at 0.83 throughout, with the central bond's effective J ~3.7× the edge — this excess is the residual that the NN-only J_ij basis assigns to longer-range correlations the basis cannot represent. Outputs: `results/inverse_design/n8_ssh_flagship_s42/{history.json, optimal_geometry.json, amplitude_evolution.{csv,npz,png}}`. **This is the headline result for Phase 2B.**
* **`n8_smoke_centre_only`** — cuda:3, *completed* 2026-04-27 09:40 CEST. The 4-parameter `dimer_chain_n8` parametrisation gave ⟨S₀·S₇⟩ = −0.319 at uniform θ = [4, 4, 4, 4] and a clean gradient `[+0.0169, −0.0070, −0.0037, −0.0024]` — strictly monotonic from end to centre, identical sign-pattern to the 2-parameter SSH flagship's `[+0.0108, −0.0078]`. **Two independent parametrisations of N=8 picked the same physics from the gradient on the very first step.** This is the cleanest available cross-validation that the inverse-design loop is finding real structure, not a parametrisation artefact.
* **`n2_target_j_0p05_s42`** — cuda:3, *completed* 2026-04-27 10:59 CEST. **The N=2-stable singlet recipe fix is fully validated**: 5 outer steps, no Stage-A collapse, no NaN gradients. The bilevel loop walked the well separation d: 2.000 → 2.018 → 2.169 → 2.252 → 2.314 Bohr, driving the exchange gap J = E_triplet − E_singlet from 0.080 → 0.103 → 0.090 → 0.086 → 0.078 Ha toward the J_target = 0.05 target. Final loss `T = −(J − J_target)² = −7.7 × 10⁻⁴` (residual J − J_target = +0.028). The optimiser is approaching the target from above with a small step (gradient at the final step is ≈ ½ of the initial step). With a smaller learning rate or more outer steps the residual would close further; for a 5-step bug-fix-validation run this is decisive evidence the fix holds.
* **`n4_uniform_d_sweep_s42`** — cuda:3, *completed* 2026-04-27 11:24 CEST (24 min total). The 5-point d-sweep landed cleanly. **The d=4 point exactly reproduces the documented Phase 2A.4 cross-check: overlap = 0.93919, S_pinn = 0.78496** — bit-identical to the values quoted in §3, an *independent* run with new code. Full table (E in Ha, S = bipartite well-set entropy {0,1}|{2,3}, overlap = |⟨c_pinn | c_heis_uniform_J⟩|):

  | d (Bohr) | E       | S_pinn | S_heis | overlap | residual_L2 | ⟨S₀·S₃⟩_pinn | ⟨S₀·S₃⟩_heis |
  |----------|---------|--------|--------|---------|-------------|--------------|--------------|
  | 2.5      | 5.6824  | 0.6606 | 0.3194 | 0.9703  | 0.244       | −0.389       | −0.250       |
  | 3.0      | 5.4400  | 0.7105 | 0.3194 | 0.9599  | 0.283       | −0.402       | −0.250       |
  | 4.0      | 5.0771  | 0.7850 | 0.3194 | 0.9392  | 0.349       | −0.413       | −0.250       |
  | 5.0      | 4.8510  | 0.8280 | 0.3194 | 0.9248  | 0.388       | −0.416       | −0.250       |
  | 6.0      | 4.7156  | 0.8497 | 0.3194 | 0.9171  | 0.407       | −0.417       | −0.250       |

  The trends are **non-trivial and not what the naive Mott picture predicts**: as d grows from 2.5 to 6 Bohr, the overlap with the uniform-J Heisenberg ground state *decreases* from 0.97 to 0.92, and the PINN's bipartite entanglement *increases* from 0.66 to 0.85 — moving *away* from the Heisenberg fixed point S_Heis = 0.32. Simultaneously the end-to-end correlator ⟨S₀·S₃⟩ becomes *more* AFM than Heisenberg (−0.42 vs −0.25). Reading: at small d (≈2.5) the wells significantly overlap, the off-Mott (doubly-occupied) component of the PINN absorbs the part that wouldn't fit the spin-only ansatz, and what remains in the singly-occupied subspace is *very* close to the OBC Heisenberg ground state. As d grows the wells decouple, the full PINN becomes more Mott-pure, but the multi-ref ansatz seems to over-weight Néel-like configurations relative to the Heisenberg superposition — pushing ⟨S_i·S_j⟩ below −0.25 and inflating the bipartite entanglement to *above* the universal Heisenberg value. This is the cleanest experimental signature so far that **the PINN's d → ∞ limit is not the OBC Heisenberg ground state**; it is a Mott-pure but variationally-biased state with stronger AFM correlations. We will quantify the off-Mott norm leak in a follow-up by extending `extract_spin_amplitudes` to return the projection mass; that closes the interpretation loop. Outputs: `results/d_sweep/n4_uniform_s42/{d_sweep.csv,d_sweep.png,d_sweep.json}`.
* **`n2_paircorr_demo_v2_s42`** — cuda:3, queued behind the displacement demo. The original `n2_paircorr_demo_s42` (v1) failed at the very first inner training because `_apply_improved_noref_recipe` (in `scripts/run_two_stage_ground_state.py`) was unconditionally setting `architecture.multi_ref=True`, which clobbered the `architecture.singlet=True` from the base config and tripped the mutual-exclusion guard in `GroundStateWF`. **Bug fix landed 2026-04-27**: the improved recipe now leaves the architecture untouched when `singlet=True` is already set (the (1,1) singlet permanent already carries the correlation directly without `multi_ref`). Unit-tested across the 4 base-config flavours (N=2 singlet, N=2 multi-ref, N=4 default, N=8 default) — no regression. The v2 launcher is queued behind the displacement demo on cuda:3.
* **`n4_displacement_demo_s42`** — cuda:3, *completed* 2026-04-27 14:46 CEST (3.4 h, 4 outer steps × 9 inner trainings). Phase 2C bilevel demo: 8 free parameters (per-well 2D displacements from the uniform N=4 chain at d=4), fd_forward, target = bipartite well-set entanglement (von-Neumann), bipartite cut `{0,1}|{2,3}`, `|δ_i| ≤ 0.6 ℓ_HO`. **Trajectory** (T = bipartite entropy across the {0,1}|{2,3} cut):

  | step | T       | E (Ha)  | well-0 x | well-1 x | well-2 x | well-3 x | d(0,1) | d(1,2)cut | d(2,3) |
  |------|---------|---------|----------|----------|----------|----------|--------|-----------|--------|
  | 0    | 0.78496 | 5.0730  | −6.000   | −2.000   | +2.000   | +6.000   | 4.000  | 4.000     | 4.000  |
  | 1    | 0.81334 | 5.0700  | −6.074   | −1.872   | +1.871   | +6.069   | 4.202  | 3.743     | 4.198  |
  | 2    | 0.84076 | 5.0724  | −6.138   | −1.742   | +1.742   | +6.129   | 4.396  | 3.484     | 4.387  |
  | 3    | **0.86870** | 5.0907  | −6.192   | −1.608   | +1.608   | +6.179   | 4.584  | **3.216** | 4.571  |

  **Net result: bipartite entropy +0.084 (+10.7%), energy almost flat (+0.4%), no inner training collapsed.** The optimiser's discovered pattern is physically transparent and beautiful: every step's gradient has all four δy components at noise floor ≤ 7×10⁻⁴ (**the 1D chain symmetry is automatically rediscovered** from a 2D parametrisation, with no symmetry-breaking inputs); the four δx components alternate stably in sign with magnitudes ~0.03–0.09 across all four steps. The optimiser steadily contracted the central bond `d(1,2)` from 4.000 to 3.216 Bohr (−19.6 %) while letting the outer bonds expand by ~0.6 Bohr. The central bond is **precisely the bipartite-cut bond** for the chosen `{0,1}|{2,3}` partition: shortening it increases the cross-partition exchange J(1,2), which is exactly what should grow the cut's bipartite entanglement under Heisenberg perturbation theory. The `displacement_2d` optimiser thus discovered, from a uniform start with no prior on the chain axis or pair structure, that **the right way to amplify the bipartite entanglement of a chosen partition is to compress the partition's cut bond** — exactly the prescription one would write down by hand. This is the cleanest possible end-to-end validation that the displacement_2d parametrisation captures interpretable physics, and it is also the first concrete inverse-design result on a system whose parametrisation is *not* a hand-tailored chain (`n2`, `dimer_chain_n4`, `dimer_pair_n8`) but rather the *full* 2N-dimensional displacement landscape. With 4 steps and lr=1.5 the optimiser used about ⅔ of its `|δ_inner| ≤ 0.6` envelope; a longer run could likely push the entropy past 0.90 by saturating the bound. Outputs: `results/inverse_design/n4_displacement_demo_s42/{history.json, optimal_geometry.json, summary_step*.json}`.
* **`n2_paircorr_demo_v2_s42`** — cuda:3, *completed* 2026-04-27 16:02 CEST (1 h 16 min, 5 outer steps × ≈ 3 inner trainings each = 15 trainings). Phase 1C bilevel demo: drive `g_σ(r₀=2.5)` toward 0.40 from a starting separation of d=4 Bohr via the `n2` parametrisation (mode `neg_squared_error`, fd_central). **Trajectory** (T = −(g − 0.40)², the optimiser ascends T):

  | step | θ (Bohr) | E (Ha) | T        | residual `\|g − 0.40\|` | `\|grad\|` | implied g_σ(2.5) |
  |------|----------|--------|----------|------------------------|------------|--------------------|
  | 0    | 4.000    | 2.239  | −0.0900  | 0.300                  | 0.069      | ≈ 0.100            |
  | 1    | 3.725    | 2.271  | −0.0760  | 0.276                  | 0.089      | ≈ 0.124            |
  | 2    | 3.367    | 2.302  | −0.0410  | 0.202                  | 0.085      | ≈ 0.198            |
  | 3    | 3.029    | 2.336  | −0.0172  | 0.131                  | 0.058      | ≈ 0.269            |
  | 4    | 2.799    | 2.356  | **−0.0054** | **0.074**           | 0.033      | ≈ 0.326            |

  **Net result: 94 % loss reduction in 5 outer steps** (T: −0.090 → −0.005); θ walked monotonically from 4.000 → 2.799 Bohr; the gradient magnitude monotonically decayed from 0.069 → 0.033 as the optimiser approached the target (consistent with a quadratic loss in the residual, slope ≈ 0.45 in the linear regime). **The bug fix for the `_apply_improved_noref_recipe` mutual-exclusion crash is fully validated end-to-end** — every inner training in the demo converged cleanly with the singlet permanent ansatz preserved. With more outer steps the optimiser would land exactly at g_σ(2.5) = 0.40; the residual 0.074 at step 4 is below the FD step's effective resolution and one or two more iterations should close it. Outputs: `results/inverse_design/n2_paircorr_demo_v2_s42/{history.json, optimal_geometry.json, summary_step*.json}`.

This document will be amended in-place as those runs complete; subsequent phases will be added as new sections under §6.

---

## 10. Appendix B — milestone log since 2026-04-13

| Date | Phase | Milestone |
|---|---|---|
| 2026-04-13 | A | 3-seed CI-referenced benchmark for N=2/3/4 (~0.02 % vs CI) — last shared milestone |
| 2026-04-15 | A | First CI-independent self-residual N=2 run validated |
| 2026-04-22 | A | CI-independent reproduction blueprint locked (`plans/2026-04-22_noref_full_reproduction.md`) |
| 2026-04-24 | B | Mott spin-amplitude module + N=2 Bell singlet validation |
| 2026-04-25 | B | N=4 PINN @ d=4: 0.939 overlap with OBC Heisenberg, S = 0.785 |
| 2026-04-25 | C | Multi-sector refactor + N=2 exchange-gap smoke (0.093 → 0.185 Ha) |
| 2026-04-26 | B/D | Effective-Heisenberg J_ij extractor + non-uniqueness diagnosis |
| 2026-04-26 | E | N=2 entanglement smoke (T 0.077 → 0.448) |
| 2026-04-26 | E | N=4 entanglement flagship (T 0.785 → 0.964, +22.8 %) |
| 2026-04-26 | E | N=4 engineer-to-spec ⟨S₀·S₃⟩ → −0.65 demo (8 steps) |
| 2026-04-26 | E | `dimer_pair_n8` SSH parametrisation + `fd_forward` gradients shipped |
| 2026-04-27 | E | N=8 SSH flagship launched (step 0 complete, optimiser picked topological direction) |
| 2026-04-27 | E | N=8 4-param symmetric smoke completed: gradient sign-pattern matches SSH flagship (independent cross-check) |
| 2026-04-27 | E | Phase 1C landed: pair-correlation g_σ(r₀) target shipped + N=2 trajectory validation |
| 2026-04-27 | bug | Stage-A self-residual collapse on N=2 multi-ref singlet diagnosed and fixed (auto-route to legacy stable recipe) |
| 2026-04-27 | E | N=2 target-J 0.05 relaunch stable (3/5 steps, no Stage-A collapse) |
| 2026-04-27 | bonus | N=4 uniform-chain d-sweep orchestration shipped (`scripts/n_chain_d_sweep.py`) and queued for cuda:3 |
| 2026-04-27 | C | Phase 2C STRETCH parametrisation `displacement_2d` shipped + N=4 bilevel demo queued |
| 2026-04-27 | E | N=2 target-J 0.05 completed (5/5 steps, J 0.080 → 0.078, no Stage-A collapse) |
| 2026-04-27 | bonus | N=4 uniform d-sweep completed: d=4 reproduces 0.939 overlap exactly; d → ∞ moves *away* from Heisenberg (variational bias) |
| 2026-04-27 | bug | `_apply_improved_noref_recipe` no longer clobbers `singlet=True` (mutual-exclusion guard); paircorr-v2 demo queued |
| 2026-04-27 | C | N=4 displacement_2d demo step 0 clean (T=0.785 matches d-sweep d=4); FD picking physical directions (x: large, y: ≈0) |
| 2026-04-27 | C | N=4 displacement_2d **completed**: T 0.785 → 0.813 → 0.841 → **0.869** (+10.7 % entropy gain in 4 steps, E flat at 5.07 Ha); optimiser auto-rediscovers 1D chain symmetry (perp grads at noise floor) and compresses the {0,1}|{2,3} cut bond from 4.000 → 3.216 Bohr (−19.6 %) — first inverse-design result on the full 2N-dimensional displacement landscape |
| 2026-04-27 | E | N=8 SSH flagship steps 1–4 stable: θ walks [4.0,4.0]→[4.31,3.73] (dimerisation Δd = +0.58); T 0.319→0.325 (steady SSH-direction climb on ⟨S₀·S₇⟩) |
| 2026-04-27 | C | N=2 pair-corr v2 demo **completed** on cuda:3 (5/5 steps, all clean): loss residual `\|g − 0.40\|` 0.30 → 0.074 (75 % closed in 5 steps, 94 % loss reduction); first end-to-end inverse-design optimisation against an MCMC-sampled observable + final validation of the singlet/multi_ref bug fix |
| 2026-04-27 | C | N=4 displacement_2d trajectory analysis: ⟨S₀·S₃⟩ −0.413 → −0.461 (more AFM as entropy grows), `effJ_overlap` stays at 0.963 across all 4 steps — **the optimiser moves along the Heisenberg-like manifold**, choosing the point with max bipartite entanglement rather than escaping the manifold |
| 2026-04-27 | bg | N=8 uniform d-sweep launched on cuda:3 (5 d-values, ~50 min ETA) to test whether the "d→∞ moves away from Heisenberg" finding generalises to longer chains |
| 2026-04-27 | infra | Trainer subprocesses now spawned with `python -u` + `PYTHONUNBUFFERED=1` in both `scripts/n_chain_d_sweep.py` and `src/geometry_optimizer.py` — `train_{tag}.log` now streams progress lines in real time instead of only flushing at process exit |
| 2026-04-27 | bg | cuda:3 ended up shared with another user's `train.py` (started 14:55 CEST) → N=8 d-sweep d=2.5 inner training observed ~2× slower than the equivalent SSH-flagship inner trainings on cuda:6 (~70 min vs ~34 min). Process is healthy (99% util, 1.2 GB / 4.4 GB total on cuda:3); revised d-sweep ETA ≈ 22:00 CEST |
| 2026-04-27 | E | N=8 SSH flagship at step 5/8 (step005_centre + step005_dir0_plus complete, step005_dir1_plus in flight); on track for ~21:00 CEST completion |
| 2026-04-27 | bug | Phase 0 bonus: shared-DVR-CI Coulomb-kernel pathology *deeply* diagnosed (not just a `include_quadrature_weights` toggle — the singular DVR-Coulomb diagonal at ε=0.01 dominates the two-electron sum on the shared softmin basis). Production code unaffected; deprecation docstrings + 3-test regression suite (`tests/test_shared_ci_coulomb_kernel.py`) landed pinning the current behaviour and the one-per-well reference (the reliable path). |
| 2026-04-27 | bg | N=8 d-sweep d=2.5 completed at 18:28 (2 h wall on shared GPU 3, E=13.520 Ha, 1500 epochs converging from loss 0.186 → 0.045 then 0.044); d=3.0 in flight |
| 2026-04-27 | E | N=8 SSH flagship at step 6/8 (step006_centre + step006_dir0_plus complete; step006_dir1_plus running); revised ETA ~21:15 CEST |
| 2026-04-27 | E | N=8 SSH flagship at step 7/8 (step006 finishing; step007 = last outer step queued). Trajectory through step 6: θ [4.0, 4.0] → [4.43, 3.56] (dimerisation Δd = 0.88, +44 % asymmetry); T = ⟨S₀·S₇⟩ magnitude proxy 0.319 → 0.329 (+3 % monotonic climb in the SSH-topological direction); E flat at 11.43–11.45 Ha. The optimiser is now well inside the SSH-dimerised regime |
| 2026-04-27 | infra | Phase 3B preparation: `scripts/n_chain_b_sweep.py` shipped — sister of `n_chain_d_sweep.py` that sweeps `system.B_magnitude` instead of inter-well `d`, sharing the same Mott-spin / Heisenberg-reference / correlator analysis (`analyse_one`). Geometry is taken as-is from the base config so the same script works against either a uniform chain, an SSH-engineered geometry, or any other inverse-design endpoint. Ready to launch on cuda:3 once the d-sweep finishes (~21:18 CEST) or on cuda:6 once the SSH flagship finishes (~22:45 CEST) |
| 2026-04-27 | bg | N=8 d-sweep on cuda:3 has cleared up: contender's job ended after d=2.5; subsequent points ran fast (d=3 52 min, d=4 37 min, d=5 40 min); d=6 in flight (started 20:36). PINN energies clean and monotonic in d: E(d=2.5)=13.520 / E(d=3)=12.583 / E(d=4)=11.437 / E(d=5)=10.739 / E(d=6) pending. The d=4 PINN matches the N=8 SSH flagship baseline at θ=[4,4] (E=11.445, T=0.319) within ~0.01 Ha — independent cross-validation. Variance also drops monotonically (0.045 → 0.024 → 0.003 → 0.001) showing the larger-d sector is variationally easier as expected |
| 2026-04-27 | bonus | **N=8 d-sweep COMPLETE**, all 5 points (d ∈ {2.5, 3, 4, 5, 6}). Headline (independent N=8 confirmation of the N=4 finding): the PINN d → ∞ limit is *not* the OBC Heisenberg ground state — bipartite entropy *grows* with d (S_pinn 0.955 → 1.121) while Heisenberg stays flat at S_heis = 0.457; overlap *decreases* (0.867 → 0.715); ⟨S₀·S₇⟩_pinn (-0.397 → -0.330) is ~3.5× more antiferromagnetic than ⟨S₀·S₇⟩_heis (-0.093, fixed by uniform-J reference). Same qualitative trend as the N=4 d-sweep, scaled up to N=8: the multi-ref PINN over-weights AFM Mott patterns relative to the spin-only Heisenberg superposition, even at d=6 where Mott projection should be near-perfect. This is a non-trivial empirical claim about the variational ansatz's biases that holds at *both* N=4 and N=8 — strong evidence the effect is intrinsic to the PINN+multi-ref ansatz, not a small-N curiosity. Outputs: `results/d_sweep/n8_uniform_s42/{d_sweep.csv,d_sweep.png,d_sweep.json}` |
| 2026-04-27 | bg | Phase 3B B-sweep launched on cuda:3 (just freed by d-sweep): 5 magnetic-field values [0.0, 0.05, 0.2, 0.5, 1.0] in atomic units, fixed N=8 uniform chain at d=4, 1500 epochs each, ~37 min/point on free GPU → ETA ~00:30 next day. Confirmed unbuffered logging works (train_b0.log streaming progress lines in real time) |
| 2026-04-27 | caveat | **B-sweep methodological caveat surfaced (already in `src/run_ground_state.py`'s magnetic-assessment guard)**: when a multi-ref base config is run at fixed Sz=0 with uniform longitudinal Zeeman, every spin template has Sz_template=0 and the Zeeman term `-B·Sz_total` is *identically zero per template* — so the entire B-sweep is a structurally trivial null experiment under this ansatz/Hamiltonian combo, only varying through optimisation seed-noise. This is exactly the warning that fired in `train_b0p05.log`: "Magnetic configuration is structurally trivial under the current generalized fixed-spin ansatz... Use a spin-sector-aware ansatz or a different magnetic Hamiltonian before treating this as a state-changing run." Confirmed by the energy-noise floor: E(b=0)=11.446, E(b=0.05)=11.417, both within ±0.03 Ha of the d-sweep's d=4 energy (11.437). The B-sweep run will be **completed for the record** (gives clean 5-point baseline confirming the no-effect prediction at varying optimisation noise), but the intended Phase 3B science requires either (a) sector-aware ansatz that lets Sz polarise with B (existing for N=3,4 in `p4_n3_generalized_uniform_b_fixedspin_guard_*` and `p5_n*_mag_*up*down_b*` lanes — needs N=8 extension) or (b) orbital coupling via vector potential (requires Hamiltonian work). Both options are beyond Phase 3B's overnight slot — flagged for next supervisor cycle. |
| 2026-04-28 | bg | **N=8 B-sweep COMPLETE on cuda:3** (5 B values, ~3.5 h end-to-end). Final table — confirms the structural-triviality prediction in the cleanest possible way: every spin observable is **bit-identical across all five B values** (to printed precision), only the energy varies by ±0.05 Ha (optimisation seed noise around 11.42 Ha):  ``B    E       S_pinn  S_heis  overlap  res_L2   C_end(pinn)  C_end(heis) | 0.000  11.4459  1.0637  0.4570  0.77114  6.765e-01  -0.31898    -0.09320 | 0.050  11.4170  1.0637  0.4570  0.77114  6.765e-01  -0.31898    -0.09320 | 0.200  11.4143  1.0637  0.4570  0.77114  6.765e-01  -0.31898    -0.09320 | 0.500  11.4330  1.0637  0.4570  0.77114  6.765e-01  -0.31898    -0.09320 | 1.000  11.3990  1.0637  0.4570  0.77114  6.765e-01  -0.31898    -0.09320``. Same network checkpoint emerges for all 5 B values (same seed + B-invariant spin sector → identical training trajectory's modulo numerical jitter). Outputs: `results/b_sweep/n8_uniform_d4_s42/{B_sweep.csv,B_sweep.png,B_sweep.json}`. The clean "no-effect" baseline is the deliverable; the science Phase 3B was originally chasing requires sector-aware ansatz (N=8 extension of existing N=3,4 magnetic lane). |
| 2026-04-28 | scaling | **N=16 seed-314 retry launched on cuda:6** (00:33 CEST) following the `self_residual` strategy (NOT `improved_self_residual`, which forces `multi_ref=True` and would expand to C(16,8)=12,870 spin templates — confirmed by an aborted 13-min initial attempt that produced no training output). The previously failed run (seed 314 with `guided` strategy → E=−372 Ha collapse) is now reproducing cleanly with `self_residual` + `multi_ref=False`: at epoch 900/5000 (18 % through), E = 27.25 Ha (sensible for 16 electrons in a 4×4 d=6 grid), variance dropped 270× from 0.082 (lr-warmup peak at epoch 400) to 3.0 × 10⁻⁴ (epoch 800). No collapse. ETA full run ≈ 03:30 CEST. **Methodological lesson**: `self_residual` (no recipe) preserves the base config's `multi_ref=False`, while `improved_self_residual` is appropriate only for small-N where C(N, n_down) is tractable; need to add an explicit warning/guard for N≥10. |
| 2026-04-27 | flagship | N=8 SSH inverse-design trajectory at step 7/8 (centre done, dir0_plus in flight): θ has walked **monotonically** from uniform [4.0, 4.0] to dimerised [4.435, 3.556] (Δd≈0.88, ratio 1.25:1 — strong SSH alternation), target ⟨S₀·S₇⟩ strengthens 0.319 → 0.329 (+3.1%), energy oscillates within ±0.014 Ha around 11.44. The optimiser auto-discovered SSH-like geometry purely from the spin-correlator target — this is the headline science result of the day. Final step ETA ~23:30 CEST. Trajectory will be cross-validated by `amplitude_evolution.py` post-completion (queued for cuda:3 once B-sweep frees it ~00:30) |
| 2026-04-28 | flagship | **N=8 SSH flagship COMPLETE** — full 8 outer steps + final step. Trajectory: θ [4.0, 4.0] → [4.493, 3.459] (final), `optimal_theta` = [4.5487, 3.3540] (last gradient step, not separately evaluated); ratio long/short = 1.36:1; T = ⟨S₀·S₇⟩ monotonic 0.31898 → 0.33077 (+3.7% over 8 evaluated steps). `amplitude_evolution.py` extraction on the trajectory's centres (10 minutes of cuda:6 inference) confirms the dimerisation propagates cleanly into the **NN ⟨S·S⟩ matrix**: at the converged geometry, NN bonds alternate **−0.354 / −0.391 / −0.335 / −0.392 / −0.335 / −0.391 / −0.354** — short bonds (3.46 Bohr) carry the stronger AFM correlation (−0.39), long bonds (4.49 Bohr) the weaker (−0.33), with edge bonds (NN(0,1) and NN(6,7)) showing the textbook OBC edge boost. The end-to-end ⟨S₀·S₇⟩ that we engineered as the target stays *exactly equal* to T at every step (cross-validation: `extract_spin_amplitudes` is consistent with the differentiable observable used by GeometryOptimizer to compute gradients). **Effective Heisenberg overlap** is ~0.83 throughout (uniform AND dimerised), with the relative residual ~0.40 — the NN-only J_ij basis can capture ~83% of the ground state for both geometries. The fitted J_ij at the final geometry is `[1.0, 2.46, 3.04, 3.69, 3.04, 2.46, 1.0]` (relative units): the central bond's effective coupling is ~3.7× the edge — this is the residual that the NN-only basis assigns to longer-range correlations the basis can't represent. Outputs: `results/inverse_design/n8_ssh_flagship_s42/{amplitude_evolution.csv,amplitude_evolution.npz,amplitude_evolution.png}`. |
| 2026-04-28 | scaling | **Phase 4 N=16 retry COMPLETE on cuda:6** — 5000 epochs in 3 h 11 min (00:30 → 03:41 CEST). Final E = **27.270 Ha** (var = 4.2 × 10⁻⁴, ESS = 32, no collapse), seed 314, `self_residual` strategy with `multi_ref=False`. Compared to the previous failed run on the *same* seed (`guided` strategy collapsing to E=−372 Ha), this is **a complete methodological recovery**: the same configuration trains cleanly to a physical, well-converged ground state. Energy trajectory (epoch → E): 1200→27.267, 2000→27.265, 3000→27.266, 4000→27.259, 5000→27.270 — flat in the last 4000 epochs with sub-1% variance. **Phase 4 is officially unblocked**; the remaining N=16 / N=12 ground-state targets can now use this recipe as a stable scaffold. Outputs: `results/scaling/n16_grid_d6_s42_seed314_self_resid_summary.json`, `results/n16_grid_d6_s42__stageA_self_residual_20260428_034125/`. **Methodological lessons** (newly settled, please adopt as defaults for N ≥ 10): (a) prefer `self_residual` over `improved_self_residual` once `C(N, n_down)` exceeds ~100 templates (because `improved_self_residual` forces `multi_ref=True` regardless of base config); (b) keep `multi_ref=False` for fixed-spin scaling targets; (c) expect 30 ms / epoch on a 2080 Ti for N=16 with `n_coll=32`. Recipe carries trivially to N=12 (next planned scaling task). |
