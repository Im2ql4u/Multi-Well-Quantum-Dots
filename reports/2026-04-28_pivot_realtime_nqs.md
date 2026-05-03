# Strategy pivot — back to the ambitious plan
**Date:** 2026-04-28
**Status:** Locked. Supersedes ``2026-04-28_grand_plan_anchored.md`` (now demoted to a publication-time QA appendix).

---

## What changed

The earlier "Grand Plan with Anchors" structured every research direction around
external ground-truth validation (Heitler-London for N=2, ED for N=4, materials
calibration, virial assertion, tight-binding extraction). That structure was
strangling exploration: every novel direction was gated on building yet another
anchor, and the cumulative effect was to spend the next 6–8 weeks on
infrastructure rather than on the genuinely interesting science.

We now treat validation as **late-stage publication QA**, not as a phase gate.

### What we *keep* as cheap insurance

These three anchors run in milliseconds and cost nothing at training time. They
stay because they are virtually free; they do **not** gate any further work:

| Anchor | Status | Cost | Why we keep it |
|---|---|---|---|
| `training/symmetry_asserts.py` (Phase 0.1) | done | free at training | catches `n_up + n_down ≠ N` config errors at train time; audit found 0 violations across 1023 historical checkpoints |
| `observables/heitler_london.py` (Phase 0.4) | done, 15/15 tests | sub-second | analytical N=2 closed form; agrees with `2ω + κ/d` far-field limit to 4 sig figs |
| `observables/spin_correlators_ci.py` (Phase 0.2) | done, 15/15 tests | milliseconds | `⟨S_1·S_2⟩ = -3/4` (singlet) / `+1/4` (triplet) on a CI eigenvector |

### What we *demote* to publication-time QA

These are useful but are no longer prerequisites. They get done when we are
writing up, not before:

* Phase 0.3 PINN ↔ ED N=2 cross-comparison
* Phase 0.5 ED time evolution (Trotter / `expm`)
* Phase 0.6 tight-binding `t/U` extraction → Schrieffer-Wolff `J_SW`
* Phase 0.7 materials calibration (Si/MOS, GaAs)
* Phase 0.8 virial-theorem assertion `|2T+V|/|E| < 0.05`

They will appear as a **single late-stage QA notebook** in the publication
package, run once on the final results, not as a recurring blocker.

---

## The grand goal, restated

A **non-MCMC, deterministic-sampled, gradient-based PINN platform for
time-dependent fermionic many-body physics on engineered multi-dot networks**.
The unique selling proposition is the *intersection*:

* 2D continuum (not lattice) — captures real device physics, not toy models
* Many-body fermionic (not single-particle) — captures interactions
* Deterministic stratified sampling (not MCMC) — gives smooth gradients across
  Hamiltonian parameters, which is what unlocks gradient-based design
* Differentiable Hamiltonian — enables both inverse design and TDVP
* Multi-well topology — chains, rings, ladders, plaquettes, frustration

Almost every published NQS code-base lives outside at least three of those
constraints (lattice-only, single-particle, MCMC-only, or non-differentiable).
The intersection itself is essentially uninhabited territory.

---

## Three ambitious threads (parallel development, validate at end)

### T1 — Real-time NQS quench dynamics (the moonshot)

**Why:** You explicitly flagged "time evolutionary non-mcmc stability" as
high-potential. Most NQS time-evolution work (Carleo-Troyer, Schmitt-Heyl, …)
lives on lattices and uses MCMC + stochastic reconfiguration. We have the
ingredients to do it on the **continuum**, with **deterministic sampling**, via
**PDE-residual collocation** on (x, t).

**Existing infrastructure:**

* `src/imaginary_time_pinn.py` — already does τ-evolution from PDE residual
  `∂_τ g + (E_L − E_ref) = 0`. The *only* substantive deltas to make it
  real-time:
  1. Promote `g(x, τ) ∈ ℝ` to `g(x, t) = g_R(x, t) + i g_I(x, t)` (two-channel
     real network).
  2. Replace the PDE residual with the coupled real-time pair derived below.

#### Real-time PDE residual derivation

Setup: `ψ(x, t) = exp(log ψ_0(x) + g(x, t))` with `g` complex, `ψ_0` the trained
ground state of `H_0`, and the system evolved under `H = H_0 + ΔV(x)` (sudden
quench at `t = 0`).

The TDSE `i ∂_t ψ = H ψ` becomes `∂_t g = -i E_L`, with

```
E_L = -½[∇² log ψ + (∇ log ψ)²] + V_0 + ΔV
    = E_L^(0) + ΔV - ½∇²g - (∇ log ψ_0) · ∇g - ½(∇g)²
```

Splitting `g = g_R + i g_I` and writing `a := ∇log ψ_0`, the *real* and
*imaginary* parts give two coupled real PDEs:

```
∂_t g_R = -½ ∇²g_I - a·∇g_I - ∇g_R·∇g_I            (Im[E_L] residual)
∂_t g_I = -E_L^(0) - ΔV
          + ½ ∇²g_R + a·∇g_R
          + ½(|∇g_R|² - |∇g_I|²)                    (-Re[E_L] residual)
```

Initial condition: `g_R(x, 0) = g_I(x, 0) = 0`.

#### Trivial smoke test (ΔV = 0)

If we evolve under the same `H_0` whose ground state is `ψ_0`, then
`E_L^(0)(x) → E_0` (constant) up to numerical fluctuations. The exact answer is
the global phase rotation `ψ(x, t) = e^{-i E_0 t} ψ_0(x)`, i.e.

```
g_R(x, t) ≡ 0,    g_I(x, t) = -E_0 t.
```

Plugging into the PDE pair: both LHS = RHS trivially. So a working PINN must
recover `g_R ≈ 0` and `g_I ≈ -E_0 t` with floor near machine precision when
trained on the exact ground state. This is the smoke test we stand up first.

#### N=2 quench (the headline result)

Once the trivial test passes, we run a real quench: prepare `ψ_0` as the GS at
`B = 0`, then turn on a uniform Zeeman field `B ≠ 0` at `t = 0` and evolve
forward. ED time evolution is cheap at N=2, so we have a tight benchmark:

* PINN observable `⟨S_z⟩(t)` vs ED `⟨S_z⟩(t)`.
* PINN observable `⟨S_1 · S_2⟩(t)` vs ED — should oscillate between singlet
  (−3/4) and triplet (+1/4) at frequency `g μ_B B`.
* Norm conservation `‖ψ(t)‖ = 1` to within `10⁻³`.

**Targets to scale toward:** N=4 chain Néel quench, N=8 chain ring quench,
multi-dot networks where ED is impossible (N=12+).

### T2 — Inverse design at scale (pragmatic flagship)

**Status:** We forgot we already finished two flagships:

* `results/inverse_design/n4_flagship_p2a_aggressive/` — 10 outer steps,
  von-Neumann entropy maximisation across `{0,1}|{2,3}` bipartition,
  `T = 0.9639`, `E = 4.9438`. **Target effectively saturated** (max for that
  partition is `S = ln 4 ≈ 1.386`, so we are at ~70% of theoretical max).
* `results/inverse_design/n8_ssh_flagship_s42/` — 8 outer steps, spin
  correlator maximisation on a dimer-pair parametrisation,
  `T = 0.331`, `E = 11.467`.

Neither of those has been deeply analysed. Re-mining them is essentially free
science and gives us at least one paper-worthy result before the moonshot
matures.

**Next runs:**

* Multi-target N=8 inverse design: jointly optimise entanglement *and* spin
  gap; characterise the Pareto front.
* N=12 inverse design (gradient-based design at sizes ED cannot reach) —
  unique to this codebase.

### T3 — Network topology phase diagram (visual flagship)

**Why:** "this network of quantum multi-dots" was your phrasing. Most studies
fix a topology (chain) and sweep parameters; we can sweep *topology itself*.

Six topologies to map:

| Topology | Wells | Frustration | Status |
|---|---|---|---|
| Chain | linear | no | extensively studied |
| Ring (closed chain) | ring | no | exists for QHE only — port to spin |
| 2×2 plaquette | square | no | exists at N=4 |
| Ladder (2×L) | rect grid | no | not yet |
| T-junction | 3 arms | no | not yet |
| Triangular (frustrated) | triangle | yes | not yet |

For each topology and each system size `N ∈ {4, 6, 8}`, train the GS, extract
Mott amplitudes via `observables/spin_amplitude_entanglement.py`, compute the
multi-partite entanglement structure, and plot a graph poster. The "dot" in
each topology graph is colour-coded by local density; edges are weighted by
spin-correlator magnitude.

This is **maximally visual** and immediately shareable, even before we have
the results to publish.

---

## Execution order

1. **Now** — T1 prototype (real-time PINN, trivial smoke + N=2 ΔV ≠ 0 quench).
2. **In parallel, when GPU is free** — T2 deep dive on existing N=4 / N=8
   flagships, plus a multi-target N=8 run.
3. **Once T1 has a working N=2 quench** — T3 topology sweep at N=4 (cheap, fast).
4. **Late** — Publication-time QA notebook running everything in
   `reports/2026-04-28_grand_plan_anchored.md` Section "Phase 0".

Validation is the *last* phase, not the first.
