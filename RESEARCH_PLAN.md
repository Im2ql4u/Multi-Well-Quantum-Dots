# Research Directions: Novel Quantum Dot Physics
**Updated: 2026-04-24**

## Core Advantage
CI-free PINN training that scales polynomially with N. CI scales as N! and is dead at N≈20.
We can reach systems that are **literally computationally impossible** by any other continuous-space method.

---

## Direction S: Scaling to Large N

### Goal
Demonstrate that the method works at N = 8, 12, 16, 32. This is the foundational claim.
Measure E(N), energy per particle E/N, and verify Mott-like physics persists.

### Key implementation constraint
Multi-ref Slater determinant scales as C(N, n_up):
- N=8:  C(8,4)  =       70 dets → use multi_ref=True
- N=12: C(12,6) =      924 dets → use multi_ref=False (too slow otherwise)
- N=16: C(16,8) =   12,870 dets → multi_ref=False mandatory
- N=32: C(32,16)= 601M dets    → multi_ref=False mandatory

For multi_ref=False, the PINN Jastrow factor carries the correlation burden.
Architecture must grow: pinn_hidden=128 (N=8), 256 (N=12-16), 512 (N=32).

### Grid geometries
Use 2D rectangular grids, d=6 spacing (Mott-insulating regime), half-filling:
- N=8:  2×4 grid, wells at {(-3,-9),(3,-9),(-3,-3),(3,-3),(-3,3),(3,3),(-3,9),(3,9)} → 4up4down
- N=12: 3×4 grid, 12 wells in 3 columns × 4 rows, d=6 spacing → 6up6down
- N=16: 4×4 grid → 8up8down
- N=32: 4×8 grid → 16up16down

### FD Laplacian cost for large N
Cost per batch: 2×N×d forward passes. For N=32, d=2: 128 passes/sample.
Solution: reduce batch size n_coll from 512 to 128 for N≥16. Use `laplacian_mode: autograd`
for N≥16 (one backward pass vs 128 forward passes, faster for large N).

### Files to create
- `scripts/gen_scaling_configs.py` — generates grid YAML configs for any N×M
- `scripts/launch_scaling_sweep.sh` — 4 GPU groups (one per N)
- `scripts/analyze_scaling.py` — E(N), E/N table, compare to Mott prediction

### Expected physics
- E/N should converge as N→∞ (thermodynamic limit)
- Dominant term: ω per electron (HO energy) + Coulomb from nearest-neighbor pairs
- Mott prediction: E/N ≈ ω + z/(2d) where z = coordination number, d = well spacing
  - 2D square lattice: z=4, so E/N ≈ 1 + 4/(2×6) = 1 + 0.333 = 1.333 Ha/electron
- Any deviation from this = quantum correction (exchange, correlation)

---

## Direction D: Many-Body Localization via Disorder

### Goal
Add quenched positional disorder to well centers. Sweep disorder strength σ.
Measure bipartite entanglement entropy S as function of σ.
Observe: S ~ const (MBL/Mott, low σ) vs S growing (extended/metallic, non-interacting limit).

### Why this is novel
MBL in continuous-space 2D Coulomb systems is almost completely unstudied numerically.
DMRG requires a 1D geometry. ED requires discretization. We work in native 2D continuous space.

### Setup
- Base system: N=8 electrons in 2×4 grid, d=6 (clean Mott state known from Direction S)
- Disorder: well center i → center_i + δ_i, where δ_i ~ N(0, σ²) in both x and y
- Disorder strengths σ: 0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0 (in units of ℓ_HO = 1/√ω)
- Realizations: 4 per σ value (disorder averaging), 2 seeds each = 8 runs per σ = 56 total
- Analysis: S_ent(σ), variance of S across realizations, E(σ)

### Key observable: entanglement entropy vs disorder
- σ=0 (clean): E_ent from Mott physics — low, area-law-like
- σ >> d (strong disorder): Anderson localization — very low S (each electron trapped in one well)
- Intermediate σ: non-trivial interplay between Coulomb interactions and disorder

### Files to create
- `scripts/gen_disorder_configs.py` — takes base config + σ + seed → disordered YAML
- `scripts/launch_mbl_sweep.sh` — disorder sweep launcher
- `scripts/analyze_mbl.py` — S_ent(σ) curves, participation ratio, disorder averaging

---

## Direction QH: Quantum Hall / Laughlin Physics

### Goal
Find signatures of fractional quantum Hall (Laughlin) physics by applying large orbital
magnetic field B to N=6 electrons. At filling ν = N/N_Φ = 1/3 (N_Φ = 3N flux quanta),
the Laughlin state should emerge as the ground state.

### Key implementation: complex wavefunction
Current PINN is **real-valued**. Orbital B field requires complex ψ because the
minimal-coupling kinetic operator T = (p - eA)²/2m mixes real and imaginary parts.

The local energy with orbital B (symmetric gauge A = B/2(-y, x)):
  E_L = -½ [∇²ψ/ψ - 2iA·∇logψ + |A|²] + V
      = -½ ∇²ψ/ψ + iB/2 L_z/m + B²r²/8 + V_Coulomb + V_trap

where L_z = x∂_y - y∂_x is the angular momentum operator.

### Wavefunction ansatz for Laughlin-type state
Factor the wavefunction: Ψ(r₁,...,r_N) = Φ_L(z₁,...,z_N) × exp(J_PINN(r₁,...,r_N))

where:
- Φ_L = ∏_{i<j}(z_i - z_j)^m × exp(-B/4 × Σ|r_i|²) is the Laughlin wavefunction (m=3 for ν=1/3)
  with z_i = x_i + iy_i as complex coordinates
- J_PINN is the real PINN Jastrow factor (learnable correlation)
- The product is complex-valued

This avoids modifying the core PINN architecture: the PINN learns deviations from Laughlin.

### Observable signatures of Laughlin physics
1. Energy gap: at ν=1/3, there should be a gap above the GS (incompressible state)
2. Pair correlation g(r): characteristic suppression at short distance, g(r) ~ r⁶ for m=3
3. Entanglement spectrum: counting structure {1,1,2,3,5,...} (edge modes, Li-Haldane spectrum)
4. Angular momentum: L_z = m × N(N-1)/2 for Laughlin state

### Files to create
- `src/wavefunction_complex.py` — complex Ψ = Laughlin_base × exp(J_PINN), local energy
- `src/training/complex_collocation.py` — complex local energy computation
- `configs/qhe/` — configs for N=6,8,10 at ν=1/3
- `scripts/gen_qhe_configs.py` — generate QHE configs
- `scripts/analyze_qhe.py` — angular momentum, pair correlation g(r), entanglement spectrum

---

## Direction I: Inverse Design

### Goal
Given a **target property**, automatically find the well geometry that produces it.
Examples:
  (a) Maximize entanglement between two specific wells (for quantum information)
  (b) Find geometry with specific spin gap ΔE = E_triplet - E_singlet
  (c) Find frustrated geometry (triangle-like) with maximum frustration parameter

### Why this is novel
No one has done differentiable geometry optimization for continuous-space quantum dot arrays.
This directly connects to experimental semiconductor quantum dot design.

### Architecture: bi-level optimization

**Inner loop (for each geometry proposal):**
  Train PINN to convergence for fixed geometry → get E(geometry) and ψ(geometry)

**Outer loop (geometry optimization):**
  gradient = ∂E/∂geometry computed via Hellmann-Feynman:
  ∂E/∂R_k = ⟨∂H/∂R_k⟩_ψ  (no need for meta-gradients through training)
  = ⟨∂V/∂R_k⟩_ψ  (only the potential depends on geometry)

This is exact and does NOT require differentiating through the training process.
The expectation value is estimated by MC sampling from the trained ψ.

**Outer optimizer:** Adam or L-BFGS on geometry parameters.

### Implementation
- `src/geometry_optimizer.py` — wraps training + HF gradient computation
- `scripts/run_inverse_design.py` — specify target, initial geometry, run optimization
- Start with: maximize bipartite entanglement for N=4 (tractable test case)
  Then scale to: N=8 with entanglement target or specific correlation pattern

### Files to create
- `src/geometry_optimizer.py`
- `scripts/run_inverse_design.py`
- `scripts/analyze_inverse_design.py`

---

## Execution Order

| Priority | Direction | Status | GPU cost | New code |
|---|---|---|---|---|
| 1 | S: N=8 scaling | **Launch now** | 1 GPU, ~2h | gen_scaling_configs.py |
| 2 | D: MBL N=8 disorder | **Launch now** | 2 GPUs | gen_disorder_configs.py |
| 3 | S: N=12, N=16 | After N=8 validates | 2 GPUs | already have gen script |
| 4 | QH: complex ψ | Implement this week | 2 GPUs | wavefunction_complex.py |
| 5 | I: inverse design | Implement next | 1 GPU | geometry_optimizer.py |
| 6 | S: N=32 | After N=16 works | 4 GPUs | need autograd Laplacian |

---

## What Makes This Publishable

The combination of all four directions in one paper:
*"Continuous-space neural quantum states beyond CI: scaling, localization, topological order,
and inverse design of quantum dot arrays"*

The key claims:
1. **First** neural quantum state results for N=32 continuous-space 2D Coulomb systems
2. **First** MBL phase diagram in realistic continuous-space quantum dots
3. **First** Laughlin state discovered by neural network without a priori ansatz
4. **First** inverse design of quantum dot geometry via Hellmann-Feynman gradient

Each direction is independently publishable in PRL/PRB. Together = Nature Physics level.
