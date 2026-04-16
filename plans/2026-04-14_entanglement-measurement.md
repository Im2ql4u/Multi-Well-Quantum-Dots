# Plan: Ground State Characterization and Entanglement Measurement

Date: 2026-04-14
Status: draft

## Project objective
Produce publication-quality VMC ground-state energies and wavefunctions for multi-well quantum dots, validated against exact diagonalization; characterize entanglement properties and demonstrate entanglement dynamics under quenches.

## Diagnosis: Why the B-field quench is spatially trivial

**Finding:** The existing B=0→0.5 quench result (`results/imag_time_pinn/pinn_quench_single_fast_B0p50.json`) converged to E=4.137, which matches E_ref + ΔV_zeeman = 3.634 + 0.5 = 4.134 within VMC noise. The spatial wavefunction is unchanged.

**Root cause:** Our VMC uses the **one-per-well model** (distinguishable particles, fixed spin assignment ↑↓↑↓...). The Zeeman term ½gμ_B B Σs_z is a **constant** for a fixed spin configuration — it does not depend on particle positions. Therefore:
- ΔV(x) = constant for all x
- The spatial Hamiltonian is unchanged
- The ground-state wavefunction is identical before and after quench
- All eigenvalues shift by the same constant (confirmed by exact diag: E₀=2.179 → 2.679 at B=0.5)

**Contrast with shared model (identical fermions):** The shared-basis exact diag DOES show nontrivial B-field physics:
- B=0: singlet GS (E=2.656), triplet excited (E=2.801)
- B>0: triplet_m is lowered by B, crossing singlet at B_c ≈ 0.145
- This singlet-triplet transition involves spin entanglement changes

But our VMC cannot represent this because it has fixed spin assignments and no superposition of spin sectors.

## What CAN create interesting spatial dynamics

For the one-per-well VMC, quenches that change the **spatial** Hamiltonian create real dynamics:

| Quench type | What changes | Entanglement effect |
|---|---|---|
| Well separation d₁ → d₂ | Tunneling/Coulomb | d↓ → more interaction → more entanglement |
| Coulomb coupling κ₁ → κ₂ | Interaction strength | κ↑ → more correlation → more entanglement |
| Confinement ω₁ → ω₂ | Wavefunction extent | ω↓ → wider WF → more overlap → more entanglement |
| Position-dependent B(x) | Zeeman gradient | Breaks spatial symmetry (requires code extension) |

**Recommendation:** Use a **well-separation quench** (d_large → d_small) as the primary demonstration. This is:
- Physically motivated (tunable tunnel coupling in real experiments)
- Already supported by the code (`d` parameter in config)
- Produces clear entanglement signal (product state → entangled state)
- Validatable against exact diag

## Approach

### Phase 1 — Exact diag eigenspectrum analysis (~1h)
**Goal:** Map the full eigenspectrum including spin sectors as a function of B-field and well separation. This gives (a) the reference for what entanglement the system should have, and (b) publication-quality phase diagram.

- **Step 1.1:** B-field sweep in shared model: E₀(B) for B=0..2.0 at sep=4, identify singlet-triplet crossing
- **Step 1.2:** Separation sweep: E₀(d) for d=2..20 in both models, quantify the singlet-triplet gap Δ(d)
- **Step 1.3:** Exact diag entanglement: compute the Schmidt decomposition of the exact diag GS wavefunction as a function of d and compare singlet vs triplet

### Phase 2 — Build entanglement measurement for VMC wavefunctions (~2-3h)
**Goal:** Create `scripts/measure_entanglement.py` that takes a VMC checkpoint and computes spatial entanglement.

**Method — Schmidt decomposition (SVD) approach:**
For ψ(x₁, x₂) where x₁, x₂ ∈ ℝ² (2D particles):
1. Create a 2D grid for each particle: x-grid centered on each well, y-grid symmetric
2. Evaluate ψ(x₁, x₂) on the full 4D grid (x₁_grid × x₂_grid)
3. Reshape as matrix M(i₁, i₂) where i₁ indexes particle-1 grid points, i₂ indexes particle-2
4. Include quadrature weights: M̃(i₁, i₂) = √w₁ × √w₂ × M(i₁, i₂)
5. SVD: M̃ = UΣV†
6. Schmidt coefficients: σₖ → normalized probabilities pₖ = σₖ²/Σσₖ²
7. **Von Neumann entropy:** S = -Σ pₖ log pₖ
8. **Schmidt rank:** number of non-negligible pₖ (effective dimension of entanglement)

**Method — Partial transpose negativity:**
1. Build reduced density matrix ρ_A(i₁, i₁') = Σ_{i₂} M̃(i₁, i₂) M̃*(i₁', i₂) (this is UΣ²U†)
2. For the partial transpose: reshape ρ into (d_A, d_B, d_A, d_B), transpose subsystem B
3. Diagonalize ρ^{T_B}
4. **Negativity:** N = Σ (|λₖ| - λₖ)/2
5. **Log-negativity:** E_N = log₂(2N + 1)

For pure states, both methods give equivalent yes/no entanglement detection. The SVD is numerically cleaner. The negativity provides a standard quantifier used in the literature.

**Validation:**
- Product state (d=20, weak Coulomb): S ≈ 0, N ≈ 0
- Interacting state (d=4, κ=0.7): S > 0, N > 0
- Bell-like limit (d→0, strong interaction): S → log(d_eff)

**Grid requirements estimate:**
- Per dimension: Nₓ = 25-30 points (covers well ±4 HO lengths)
- Total wavefunction evaluations: Nₓ⁴ = 25⁴ ≈ 390K (feasible on GPU in ~seconds)
- Matrix for SVD: Nₓ² × Nₓ² = 625×625 (trivial to diagonalize)

### Phase 3 — Characterize existing ground states (~1h)
**Goal:** Measure spatial entanglement of the validated VMC ground states.

- **Step 3.1:** Load N=2 one-per-well GS checkpoints at sep=4.0 (from MCMC and non-MCMC training)
- **Step 3.2:** Run entanglement measurement: compute S, negativity, Schmidt spectrum
- **Step 3.3:** Separation sweep: load or train GS at d=2, 4, 8, 12, 20; plot S(d) and N(d)
- **Step 3.4:** Compare VMC entanglement vs exact diag entanglement at the same d values
- **Step 3.5:** Report: what are the entanglement properties the GS has? How does it differ from a product state?

### Phase 4 — Well-separation quench with entanglement dynamics (~2h)
**Goal:** Demonstrate entanglement generation via a spatially non-trivial quench.

**Protocol:** Train GS at d_initial=12 (weakly interacting, near-product state). Quench to d_final=4 (strongly interacting). Time-evolve via imaginary time. Measure entanglement at each τ step.

- **Step 4.1:** Modify `imaginary_time_pinn.py` to support well-separation quench (ΔV = V(d_final) - V(d_initial), which IS position-dependent)
- **Step 4.2:** Run quench d=12→4 for N=2
- **Step 4.3:** Evaluate E(τ) and S(τ), N(τ) at multiple τ values
- **Step 4.4:** Compare final-state entanglement to exact diag GS at d=4
- **Step 4.5:** If E(τ) converges correctly, extend to N=3 and N=4

### Phase 5 (stretch) — Spin-sector analysis from exact diag
**Goal:** For the paper, produce exact-diag results showing the singlet-triplet transition under B-field, complementing the VMC spatial entanglement results.

- Sweep B from 0 to 2.0 in shared model
- For each B: identify GS spin sector, compute spin-sector entanglement
- Plot the singlet-triplet phase diagram
- This provides the theoretical context for why magnetic fields matter in quantum dots, even though our VMC only captures one sector at a time

## Scope
**In scope:**
- Entanglement measurement implementation (Schmidt + negativity)
- Ground state entanglement characterization as a function of well separation
- Well-separation quench with entanglement dynamics
- Exact diag phase diagram for context

**Out of scope:**
- Extending VMC to dynamic spin (spin-orbital VMC) — significant architecture change
- Position-dependent magnetic field — code extension for future work
- New training runs for N≥6
- Thesis beamer modifications

## Foundation checks
- [x] VMC GS accuracy verified to <0.1% of CI for N=2/N=3/N=4
- [x] Exact diag code works in both shared and one_per_well modes
- [x] B-field in exact diag confirmed to split singlet/triplet correctly
- [x] One-per-well Zeeman is a constant shift (confirmed numerically)
- [ ] Grid-based wavefunction evaluation at scale (Phase 2 bring-up)
- [ ] Exact diag entanglement reference for validation (Phase 1)

## Risk
1. **Grid resolution for entanglement:** If Nₓ=25 is insufficient, entanglement values may have discretization error. Mitigation: convergence check at Nₓ=20,25,30,40.
2. **Wavefunction normalization:** SVD requires properly weighted integrals. If quadrature weights are wrong, entanglement values are wrong. Mitigation: verify ∫|ψ|²=1 on the grid before computing entanglement.
3. **Separation quench code changes:** The imaginary-time PDE currently only supports ΔV from Zeeman. Need to extend to support ΔV from any potential difference. Moderate code change in `precompute_ground_state()`.
