# 2026-04-26 — Inverse Design Framework

## Why now

The non-MCMC, E-ref-free training pipeline turned out to give a smooth,
deterministic energy landscape over Hamiltonian parameters (well centres,
omega, B, σ disorder) — a property MCMC NQS pipelines do not have. This
is the core enabler of the **inverse design** flagship: optimise the
geometry (or any continuous Hamiltonian parameter) to hit a target
many-body property.

This document captures the framework as built on 2026-04-26 and the
roadmap to the N=8 flagship target.

## Architecture

### Bilevel optimisation

```
outer θ → train Ψ(θ)  →  evaluate target T[Ψ(θ)]  →  ∇_θ T  →  θ ← θ + α·∇_θ T
                                                                ^---- repeat
```

* **Inner**: ``scripts/run_two_stage_ground_state.py`` — Stage A guided
  warm start, Stage B pure variance refinement. Same protocol that
  already gives Mott physics at d≥4 for N≤8.
* **Outer**: ``src/geometry_optimizer.GeometryOptimizer`` — finite-difference
  gradient on the *real* target (entanglement, gap, …) for non-energy
  targets, Hellmann-Feynman for energy.

### Parametrisation

The user supplies ``param_to_wells(theta) -> [{center, omega, n_particles}]``
that maps a low-dimensional parameter vector ``θ ∈ R^k`` to a concrete
well configuration. The default for an N=2 chain is
``theta = [d]``, ``wells = [(-d/2, 0), (+d/2, 0)]``.

For an N=8 chain the natural choice is one of:
* ``theta = [d]`` (uniform spacing) — 1 parameter.
* ``theta = [d_01, d_12, …, d_67]`` (per-bond spacing) — N-1 parameters.
* ``theta = [d, ε]`` (breathing dimerisation: alternating d±ε) — 2 parameters.

### Targets

Built-in target strings:

| ``--target``            | Sense | Cost / outer step | Notes |
| ----------------------- | ----- | ----------------- | ----- |
| ``energy``              | min   | 1 inner train     | HF gradient (cheap) |
| ``entanglement_n2``     | max   | (1+2k) trains     | dot-label negativity, calibrated against CI in Phase 0A |
| ``exchange_gap_n2``  *  | max   | (2+4k) trains     | requires two spin sectors per geometry — Phase 1B |
| ``pair_corr_r0`` *      | min/max| (1+2k) trains    | g(r) at fixed r, used as Wigner witness — Phase 1C |
| ``bipartite_ent_general`` * | max | (1+2k) trains | Mott-projected spin amplitudes, N≥4 — Phase 2A |

`*` Pending implementation; rows in italics are scheduled in the roadmap below.

Custom targets are also supported by passing a callable
``target_fn(result_dir, wells) -> float`` to ``GeometryOptimizer``.

### Warm-starting

Each outer step's ``init_from`` is set to the previous step's result_dir,
which loads ``model.pt`` with ``strict=False``. Because the ansatz
shape is fixed and the geometry change per step is small, this should
substantially accelerate convergence of the inner trainer and keep the
outer loop deterministic across geometries.

For finite-difference perturbations within a single outer step, both
the ``+ε`` and ``-ε`` trainings warm-start from the *centre* of that step.

## Status snapshot (2026-04-26 night)

| Phase | Status | Result |
|-------|--------|--------|
| 1E   | ✅ done   | N=2 smoke test, T 0.077 → 0.448 in 8 outer steps, ~2 h on cuda:3 |
| 2A.1 | ✅ done   | Mott spin-amplitude extractor (`src/observables/spin_amplitude_entanglement.py`) |
| 2A.2 | ✅ done   | N=2 validation: c_(0,1)=+0.707, c_(1,0)=−0.707, S=ln 2, neg=0.5 |
| 2A.3 | ✅ done   | CLI evaluator + GeometryOptimizer integration (`well_set_entanglement` target) |
| 2A.4 | ✅ done   | Heisenberg cross-check tool (`scripts/heisenberg_cross_check.py`); N=4 @ d=4 overlap = 0.939 |
| 2A.5 | 🚧 in progress | N=4 flagship inverse-design run on cuda:3; step 4/10, T = 0.870 (from 0.785) |
| 1B.1 | ✅ done   | Multi-sector refactor: `GeomEvalContext`, `spin_overrides`, per-sector warm-starting |
| 1B.2 | ✅ done   | `exchange_gap` target with N-aware sector defaults + `--target-J`/`--unsigned-gap` |
| 1B.3 | ✅ done   | N=2 smoke test: J = E_T − E_S, 0.093 → 0.185 Ha in 4 steps (d 2.0 → 1.77), 20 min |
| 1B.4 | 🚧 in flight | `--target-J 0.05` validation run on cuda:6 (drive J *to* a specific value) |
| 1C   | ⏳ pending | Pair-correlation target g(r₀) |
| 2B   | ⏳ pending | N=8 J_eff(0,7) engineering — uses 1B.2 multi-sector machinery |
| 2C   | ⏳ stretch | MBL disorder-pattern inverse design |

## Phase 1B (completed) — Multi-sector inverse design / exchange-gap target

The framework now supports **multi-sector inverse design**: each outer-loop
geometry evaluation can spawn one inner-loop training per spin sector, with
each sector independently warm-starting from its own previous-step centre.
This is the architectural prerequisite for *every* observable that depends
on a comparison between two (or more) symmetry sectors:

* singlet-triplet exchange gap `J = E_T − E_S` (Phase 1B, done)
* effective long-range exchange `J_eff(i,j)` via two flipped-spin patterns (Phase 2B)
* triplet-triplet splitting / D-E parameters via three-sector training
* spin-charge gap `Δ = E(N+1) − E(N)` via different-particle-count training

### Architecture

`GeometryOptimizer` accepts `spin_overrides: dict[name, sector_spec]`. Each
sector_spec is a dict with `n_up`, `n_down`, optional `force_no_singlet_arch`
and optional `stage_a_strategy`. The default `primary_sector` is `"singlet"`
when the key exists; the primary's energy is reported as the geometry's "E"
in `history.json`, while the auxiliary sectors land in `sector_energies`.

```python
opt = GeometryOptimizer(
    base_config_path="configs/one_per_well/n2_invdes_exchange_baseline_s42.yaml",
    target="exchange_gap",        # auto-fills spin_overrides for N=2/4/...
    target_kwargs={"target_J": 0.05},   # optional: hit a *specific* J value
    ...
)
```

### N=2 smoke test (completed)

```
CUDA_MANUAL_DEVICE=6 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \
    --config configs/one_per_well/n2_invdes_exchange_baseline_s42.yaml \
    --target exchange_gap \
    --param-init 2.0 --param-step 0.4 \
    --param-lower 1.5 --param-upper 6.0 \
    --n-steps 4 --lr 0.4 \
    --stage-a-epochs 1000 --stage-b-epochs 500 \
    --stage-a-min-energy 999 \
    --stage-a-strategy improved_self_residual \
    --triplet-stage-a-strategy improved_self_residual \
    --gradient-method fd_central \
    --out-dir results/inverse_design/n2_exchange_smoke
```

| step | d | E_S | E_T | J = E_T−E_S | grad | dt (s) |
|------|---|------|------|-------|--------|--------|
| 0 | 2.000 | 2.394 | 2.487 | **0.0933** | −0.228 | 356 |
| 1 | 1.909 | 2.392 | 2.510 | **0.1181** | −0.214 | 356 |
| 2 | 1.823 | 2.371 | 2.533 | **0.1620** | −0.146 | 238 |
| 3 | 1.765 | 2.366 | 2.551 | **0.1854** | −0.179 | 235 |

**Total wall time:** ~20 min on a single 2080 Ti, 24 trainings (4 outer
steps × 3 evals × 2 sectors). The gap doubled and the optimiser hit the
lower bound at step 2; the divergence/bounds heuristics gracefully degraded
to forward FD for steps 2–3.

**Phase 1B.4 in flight:** `--target-J 0.05` run on cuda:6 starting from
`d=2.0` (J ≈ 0.09); expect monotonic motion *up* in d to drive J *down*
toward 0.05, then settle on the J=0.05 isocurve.

## Phase 2A.5 flagship — N=4 inverse design

We engineered a symmetric `dimer_chain_n4` parametrisation θ = [d_outer, d_middle]
that bakes in the inversion symmetry of the chain, so the optimiser does not
have to discover it via FD.

**Target:** maximise the von-Neumann bipartite entropy `S({0,1} | {2,3})`.

**Physics expectation:**

* Uniform chain (d_outer = d_middle): S ≈ 0.79 (measured); already substantial
  via super-exchange-driven AFM correlations.
* Dimerised intra-bipartition (small d_outer, large d_middle):
  |Ψ⟩ → |singlet_{01}⟩ ⊗ |singlet_{23}⟩, factorisable across the bipartition,
  S → 0.
* Inverse-dimerised cross-bipartition (large d_outer, small d_middle):
  |Ψ⟩ → |singlet_{12}⟩ ⊗ |singlet_{03}⟩ — both pairs cross the bipartition,
  S → 2 ln 2 ≈ 1.386.

**Initial conditions:** θ = [4.0, 4.0] (uniform). Bounds d_i ∈ [2.0, 8.0].
LR = 0.5, FD-step = 0.4, 6 outer steps. Stage A 1500 epochs / Stage B
disabled (`--stage-a-min-energy 999`). Wall-time budget ~2.5 h.

**Step 0 result (running):**

| step | θ | E (Ha) | T = S_vN | ∇_θ T |
|------|---|--------|----------|-------|
| 0 | [4.000, 4.000] | 5.0635 | 0.7850 | [+0.0963, −0.0383] |

Both gradient signs match the physics prediction (d_outer ↑ helps, d_middle ↓ helps).

```
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \
    --config configs/one_per_well/n4_invdes_baseline_s42.yaml \
    --target well_set_entanglement \
    --parametrisation dimer_chain_n4 \
    --metric von_neumann_entropy --set-a 0 1 \
    --param-init 4.0 4.0 --param-step 0.4 0.4 \
    --param-lower 2.0 2.0 --param-upper 8.0 8.0 \
    --n-steps 6 --lr 0.5 \
    --stage-a-epochs 1500 --stage-b-epochs 800 --stage-a-min-energy 999.0 \
    --stage-a-strategy improved_self_residual --gradient-method fd_central \
    --out-dir results/inverse_design/n4_flagship_p2a
```

Diagnostic plot is produced by `scripts/analyze_n4_inverse_design.py`,
which tracks Mott amplitudes per step, the Schmidt distribution, and
the Heisenberg overlap.

## Phase 1E smoke test (completed)

```
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \
    --config configs/one_per_well/n2_singlet_d2_s42.yaml \
    --target entanglement_n2 \
    --param-init 2.0 --param-step 0.4 \
    --param-lower 1.5 --param-upper 6.5 \
    --n-steps 8 --lr 0.7 \
    --stage-a-epochs 1500 --stage-b-epochs 1000 \
    --stage-a-strategy singlet_self_residual \
    --stage-a-min-energy 0.5 \
    --out-dir results/inverse_design/n2_smoke_p1e
```

**Final trajectory:**

| step | d | E (Ha) | T = dot_neg | ∇θ | dt (s) |
|------|---|--------|-------------|-----|--------|
| 0 | 2.000 | 2.404 | 0.0766 | +0.246 | 1216 |
| 1 | 2.172 | 2.367 | 0.1209 | +0.321 | 1071 |
| 2 | 2.396 | 2.357 | 0.1919 | +0.415 | 1095 |
| 3 | 2.687 | 2.351 | 0.2967 | +0.315 | 1082 |
| 4 | 2.907 | 2.338 | 0.3637 | +0.260 | 1082 |
| 5 | 3.090 | 2.323 | 0.4063 | +0.206 |  753 |
| 6 | 3.234 | 2.310 | 0.4321 | +0.164 |  627 |
| 7 | 3.348 | 2.301 | 0.4481 | +0.133 |  614 |

T monotonically increases by 5.85x; ∇T decreases as θ approaches saturation;
warm-starting halves wall time per step from 1200 s (step 0) to 614 s (step 7).
Total elapsed: 7541 s ≈ 2 h 5 min on a single A100.

Diagnostic plots are produced by:

```
PYTHONPATH=src python3.11 scripts/analyze_inverse_design.py \
    --run-dir results/inverse_design/n2_smoke_p1e \
    --ideal-target 0.5
```

## Phase 2A.5 N=4 entanglement flagship (completed)

Successfully drove ``S_vN({0,1}|{2,3})`` from 0.7850 → **0.9639** in 10 outer
steps (+22.8 %) on cuda:3 in ~3 hours. The optimiser discovered a
"central-dimer + boundary-bridge" geometry (``d_outer`` increases,
``d_middle`` decreases) that strengthens long-range AFM correlations:

| step | θ = (d_outer, d_middle) | E (Ha) | S_vN | ⟨S₀·S₃⟩ |
|------|-------------------------|--------|------|---------|
| 0    | (4.000, 4.000)          | 5.063  | 0.785 | −0.413 |
| 9    | (5.50,  3.27)           | 5.073  | **0.9639** | **−0.516** |

Energy was nearly flat throughout — the gain in entanglement comes from a
pure correlation-structure rearrangement at fixed total binding energy,
*exactly* the thing a useful inverse-design pipeline should discover.

## Phase 2A.6 + 2A.7 — Effective Heisenberg J_ij + spin correlators (completed)

`src/observables/effective_heisenberg.py` exposes two related but
distinct readouts for any trained PINN checkpoint:

* **Direct spin correlator** ``C_{ij} = ⟨c|S_i·S_j|c⟩`` — ``c`` is the
  Mott-projected spin amplitude vector. Unambiguous, deterministic, no
  fitting. **Recommended target for inverse design.**
* **Covariance / parent-Hamiltonian fit**: find ``J = (J_{ij})`` such that
  ``H_eff(J) = E_0 + Σ J_{ij} S_i·S_j`` has ``c`` as ground state. For
  ``N ≥ 4`` the null space of the covariance matrix ``Q`` is multi-
  dimensional, so the fit picks the ``J`` direction in ``span(eigvecs(Q))``
  that maximises ``|⟨c|ψ_0(H_eff(J))⟩|`` — i.e., among all Heisenberg
  Hamiltonians compatible with ``c``, return the one for which ``c`` is
  the ground state. NN-only fits are robust; all-pair fits exhibit
  unavoidable parameter ambiguity for approximate states. Treat ``J_ij``
  as a *diagnostic*, not a target.

Validation (``scripts/validate_effective_heisenberg.py``):
N=2 singlet (closed form), N=3 OBC Heisenberg synthetic (exact bond
recovery), N=4 OBC synthetic (overlap > 1−1e−12, residual < 1e−9), and
N=4 PINN checkpoint at d=4 (NN-only fit: overlap > 0.99, residual ≲ 0.2).

## Phase 2A.8 — `amplitude_evolution.py` trajectory analyser (completed)

Generic per-N inverse-design trajectory inspector. For each step it
extracts (i) the full Mott spin-amplitude vector, (ii) the spin-spin
correlator matrix ``C_{ij}``, (iii) the NN-only Heisenberg fit
``J_{i,i+1}`` plus overlap and relative residual, and (iv) any user-
specified extra pair correlators. Outputs CSV summary, NPZ archive of
per-step arrays, and 6-panel PNG. Used for both the N=4 entanglement
flagship and the N=4 engineer-to-spec demo below.

## Phase 2B core — `spin_correlator` engineer-to-spec target (completed)

`GeometryOptimizer._target_fn_spin_correlator` exposes
``T = ⟨c|S_i·S_j|c⟩`` (mode ``value``) or its negative
(``neg_value``, drives toward singlet limit −0.75) or
``−(C − C_target)²`` (``neg_squared_error``, drives **to** a specific
spec value). All wired through ``run_inverse_design.py`` via
``--pair I J``, ``--mode``, ``--target-value``, ``--corr-spin-sector``.

### N=4 engineer-to-spec demo (completed)

Target: ``⟨S₀·S₃⟩ = −0.65`` on the ``dimer_chain_n4`` chain. Optimiser
correctly drove the system *toward* the target by ~17 % over 8 steps,
with monotonic improvement in the squared-error and stable energy:

| step | θ = (d_outer, d_middle) | E (Ha) | ⟨S₀·S₃⟩ | T = −(C − (−0.65))² |
|------|-------------------------|--------|---------|---------------------|
| 0    | (4.000, 4.000)          | 5.063  | −0.413  | −0.0561 |
| 7    | (4.488, 3.454)          | 5.069  | **−0.448** | **−0.0409** |

Notably, the trajectory direction (``d_outer ↑``, ``d_middle ↓``) is
identical to the entanglement-maximising flagship — an internal
consistency check that the long-range AFM correlator and the bipartite
entropy peak in the same region of θ-space.

## Phase 2B flagship — N=8 SSH-style chain (in flight)

Two complementary N=8 runs:

1. **Validation smoke** on cuda:3 (`results/inverse_design/n8_smoke_centre_only`):
   `dimer_chain_n8` 4-parameter chain (theta = [d1, d2, d3, d4]) with
   `n8_invdes_lite_s42.yaml` (epochs 2500, n_coll 512, fd_central). Single
   outer step, used to validate the workflow and produce reference
   gradients on a contested GPU. Centre at uniform d=4 gave
   **E = 11.453 Ha, ⟨S₀·S₇⟩ = −0.319** — significantly more AFM than I
   would naively guess from independent dimers; long-range correlations
   in the OBC chain are robust.

2. **SSH flagship** on cuda:6 (`results/inverse_design/n8_ssh_flagship_s42`):
   `dimer_pair_n8` 2-parameter SSH-style chain
   ``theta = [d_short, d_long]``, bond layout
   ``d_s | d_l | d_s | d_l | d_s | d_l | d_s``. Uses
   `n8_invdes_fast_s42.yaml` (epochs 1500, n_coll 384) and
   `--gradient-method fd_forward` for a per-step cost of only 3 trainings
   (centre + 2 forward perturbations). Target: ``--mode neg_value`` on
   ``⟨S₀·S₇⟩`` (drive end-to-end correlation toward the singlet limit).

   Physics interpretation: the SSH manifold contains a clean
   topological/trivial transition. *Trivial* phase
   (``d_s ≪ d_l``): nearly-decoupled singlet pairs (0,1), (2,3), (4,5),
   (6,7); ``⟨S₀·S₇⟩ → 0``. *Topological* phase (``d_s ≫ d_l``): strong
   inter-cell coupling propagates AFM correlations down the chain;
   ``⟨S₀·S₇⟩ → −∞``. The optimiser should drive the system into the
   topological end of the SSH manifold.

   Launch:

   ```
   CUDA_MANUAL_DEVICE=6 PYTHONPATH=src python3.11 scripts/run_inverse_design.py \
       --config configs/one_per_well/n8_invdes_fast_s42.yaml \
       --target spin_correlator --pair 0 7 --mode neg_value \
       --parametrisation dimer_pair_n8 \
       --param-init 4.0 4.0 --param-step 0.4 0.4 \
       --param-lower 2.5 2.5 --param-upper 10.0 10.0 \
       --n-steps 8 --lr 8.0 --gradient-method fd_forward \
       --stage-a-epochs 1500 --stage-b-epochs 1 \
       --stage-a-min-energy 999.0 --stage-a-strategy improved_self_residual \
       --out-dir results/inverse_design/n8_ssh_flagship_s42
   ```

   Each outer step costs (1 + 2 forward FD) = 3 trainings ≈ 30–45 min
   on a free 2080 Ti; full 8-step run budget ≈ 4–6 h.

## Roadmap

1. **Phase 1B** — exchange-gap target. Train two spin sectors (S=0 / S=1)
   per geometry; report ``|E_T - E_S|``. Internal target callable
   launches a second training in the triplet pattern.
2. **Phase 1C** — pair-correlation target g(r₀). MC over |Ψ|² with
   the standard sampler; cheap (no extra training).
3. **Phase 2A FLAGSHIP** — generalise bipartite entanglement to N≥4
   via Mott-projected spin amplitudes ``c_σ``. Implementation plan:
   - Sample electron positions from per-well Gaussians + permutation
     symmetrisation.
   - Compute ``c_σ = ⟨φ_{σ_1}…φ_{σ_N} | Ψ⟩`` for σ ∈ {↑,↓}^N.
   - Reduced density matrix ρ_A = Tr_B[|c⟩⟨c|]; entanglement entropy.
   - Validate against CI for N=4 (16 spin configs).
   - Run inverse design at N=8: target = max bipartite entanglement
     between halves; expected outcome — non-uniform spacing engineered
     to maximise XY-coupling between halves.
4. **Phase 2B** — quantitative engineering: target = ``J_eff(0,7) =
   J_target``. Goal: demonstrate we can dial a specific exchange.
5. **Phase 2C STRETCH** — disorder-pattern inverse design. Pair with
   the existing σ-sweep MBL infrastructure to find disorder realisations
   that maximise the MBL window.

## Risks and mitigations

* **Inner-loop fluctuations**: residual seed-to-seed variance in the
  trained energy is ~5 mHa for N=2 at d=2, which translates to noise in
  the FD gradient of order (5e-3 / 0.4) ~1e-2 per dimension. The
  smoothness of T(θ) and warm-starting both help keep the gradient
  signal well above this noise floor. If this becomes a problem, fall
  back to SPSA, or to gradient smoothing across a few outer steps.
* **Warm-start mismatch**: when changing geometry the orbital block
  changes shape continuously but the network may still see a small
  step. Mitigated by ``strict=False`` and empirically negligible at
  per-step changes Δd ≲ 0.5.
* **N>2 spin amplitudes off-diagonal contamination**: the Mott
  projection neglects double-occupied configurations. At d≥4, the
  off-diagonal weight is < exp(-d²/2) < 1%. We will report it
  alongside every estimate.
