# Plan: One-Per-Well Ground States and Magnetic Quench Time Evolution

Date: 2026-04-10
Status: confirmed

## Project objective
Produce publication-quality VMC ground-state energies and wavefunctions for multi-well quantum dots with Coulomb interaction, validated against exact diagonalization; extend to imaginary-time evolution under magnetic field quench for entanglement studies.

## Objective
Establish the one-particle-per-well regime as the primary physics focus: (1) fix/verify the virial formula for multi-well systems, (2) locate or rebuild the exact diagonalization reference, (3) produce a validated N=2 (1+1) ground state, (4) implement magnetic field quench protocol and time-evolve, (5) compare against diagonalization, and (6) generalize to N=3 and N=4 (one per well). Success condition: VMC ground-state energy for N=2 (1+1) agrees with exact diagonalization to within 1%, and imaginary-time evolution under B-field quench converges to the correct new ground state.

## Context

### What triggered this
User questioning whether the virial theorem applies to multi-well systems. The last several sessions (April 8–10) chased a ~13–15% virial residual for N=4 (2+2) double dot that never closed despite 8-run architecture sweeps. The virial formula `2T = 2V_trap - V_int` assumes a single-center harmonic trap; for a multi-well system where `V_trap = Σ_i ½ω²(r_i - R_i)²`, the virial relation has additional cross-terms involving well centers. The persistent residual may be partly or fully a formula artifact.

Separately, the project direction is shifting from "two per well" to "one per well, generalize to N wells" with magnetic field quench + time evolution.

### What exists
- **Working ground-state VMC**: `src/run_ground_state.py` with PINN/CTNN/Unified architectures, MH sampling, FD/autograd Laplacian
- **Working imaginary-time evolution**: `src/imaginary_time_pinn.py` with magnetic field (Zeeman) support already implemented (`magnetic_B_initial`, `magnetic_B`, `zeeman_electron1_only`), spectral-G architecture
- **N=2 (1+1) results**: CTNN runs converged well (E≈2.17 at sep=4.0), PINN 50k-epoch weekend runs failed catastrophically (E~15–196)
- **Well-separation sweep**: per-d N=2 1+1 sweep gives E=3.00 (d=0) → 2.05 (d=20), physically sensible
- **Magnetic field in config**: `SystemConfig` has `B_magnitude`, `B_direction`, `g_factor`, `mu_B`, `zeeman_electron1_only`
- **Diagonalization notebook**: `qdsensingmarch17b.ipynb` was referenced in a log (`results/imag_time_pinn/nb_diagonalization_exec_tmux.log`) but the file is missing from the repo. Must locate or rebuild.
- **`SystemConfig.custom()`**: supports arbitrary N wells with arbitrary particle counts — generalization infrastructure exists

### Prior negative findings (do not repeat)
- **FAILED**: Weekend 50k-epoch PINN runs for N=2 1+1 with `fd_colloc` loss diverged to E~15–196. The CTNN architecture converged fine (E≈2.17). Use CTNN or REINFORCE loss, not `fd_colloc` for 1+1.
- **FAILED**: 8-run corrective sweep could not get N=4 (2+2) virial below 12.7%; virial formula itself may be wrong for multi-well.
- **FAILED**: IS sampler diverges. Use MH only.
- Virial cross-run comparisons **must** use locked protocol (FD evaluator, MH steps=40, warmup 20).

### Constraint check
No `CONSTRAINTS.md` yet — no semantic memory.

## Approach

**Phase 0 — Audit**: Derive the correct multi-well virial relation and check whether the current formula has an inherent bias for separated wells. This is 30 minutes of math + a code check, and it may explain the entire 13–15% plateau.

**Phase 1 — Diagonalization reference**: Locate `qdsensingmarch17b.ipynb` or rebuild exact diagonalization for N=2 (1+1) double dot with and without magnetic field. This gives the ground truth to validate against.

**Phase 2 — N=2 ground state**: Train a validated N=2 (1+1) ground state using the CTNN architecture (which worked before). Verify against diagonalization.

**Phase 3 — Magnetic quench**: Use `imaginary_time_pinn.py` to: (a) start from B=0 ground state, turn on B for one particle, evolve to new ground state; (b) start from B≠0 ground state, turn off B, evolve to true ground state. Compare final energies against diagonalization.

**Phase 4 — Generalize to N=3, N=4**: Extend to 3 wells (1+1+1) and 4 wells (1+1+1+1) using `SystemConfig.custom()`. Run ground states, then magnetic quench.

---

## Foundation checks (must pass before new code)
- [x] Data pipeline known-input check — VMC sampling verified
- [x] Split/leakage validity — VMC has no train/test split
- [x] Baseline existence — N=2 (1+1) CTNN runs give E≈2.17 at sep=4.0
- [x] **Virial formula verified for multi-well** — generalized virial check shows large correction on N=4 while matching legacy behavior on single-well
- [ ] **Exact diagonalization reference available** — Phase 1
- [x] Relevant implementation read — `imaginary_time_pinn.py`, `potential.py`, `config.py`, `run_ground_state.py` reviewed

## Scope
**In scope:**
- Multi-well virial theorem derivation and formula correction
- Locating or rebuilding exact diagonalization (N=2, with/without B-field)
- N=2 (1+1) ground state with CTNN, validated against exact diag
- Magnetic field quench time evolution for N=2 (1+1)
- Generalization to N=3 (1+1+1) and N=4 (1+1+1+1) ground states
- Comparison against exact diag at each N

**Out of scope:**
- N=4 (2+2) double dot virial chase (pausing this)
- Entanglement measures (future work)
- New architectures beyond what exists
- Thesis writing
- Well-separation sweeps
- README updates

---

## Phase 0 — Multi-Well Virial Audit (~30 min)
**Goal:** Determine whether the current virial formula `2T = 2V_trap - V_int` is correct for multi-well systems. If not, derive and implement the correct formula.

### Step 0.1 — Derive multi-well virial relation
**What:** For a system with particles in wells centered at positions $\mathbf{R}_i$, the trap potential is $V_{\text{trap}} = \sum_i \frac{1}{2}\omega^2 |\mathbf{r}_i - \mathbf{R}_i|^2$. The virial theorem states $2\langle T \rangle = \sum_i \langle \mathbf{r}_i \cdot \nabla_i V \rangle$. Compute $\nabla_i V_{\text{trap}} = \omega^2(\mathbf{r}_i - \mathbf{R}_i)$, so:

$$\sum_i \langle \mathbf{r}_i \cdot \nabla_i V_{\text{trap}} \rangle = \omega^2 \sum_i \langle |\mathbf{r}_i|^2 - \mathbf{r}_i \cdot \mathbf{R}_i \rangle = 2\langle V_{\text{trap}} \rangle + \omega^2 \sum_i |\mathbf{R}_i|^2 - 2\omega^2 \sum_i \langle \mathbf{r}_i \cdot \mathbf{R}_i \rangle$$

Wait — expand: $\mathbf{r}_i \cdot \omega^2(\mathbf{r}_i - \mathbf{R}_i) = \omega^2(|\mathbf{r}_i|^2 - \mathbf{r}_i \cdot \mathbf{R}_i)$, and $V_{\text{trap}} = \frac{1}{2}\omega^2 \sum_i (|\mathbf{r}_i|^2 - 2\mathbf{r}_i \cdot \mathbf{R}_i + |\mathbf{R}_i|^2)$. So:

$$\sum_i \mathbf{r}_i \cdot \nabla_i V_{\text{trap}} = 2V_{\text{trap}} + \omega^2 \sum_i (\mathbf{r}_i \cdot \mathbf{R}_i - |\mathbf{R}_i|^2)$$

The full virial: $2T = 2V_{\text{trap}} + \omega^2 \sum_i \langle \mathbf{r}_i \cdot \mathbf{R}_i - |\mathbf{R}_i|^2 \rangle - V_{\text{int}}$

The current code uses $2T = 2V_{\text{trap}} - V_{\text{int}}$, missing the cross-term $\omega^2 \sum_i (\langle \mathbf{r}_i \cdot \mathbf{R}_i \rangle - |\mathbf{R}_i|^2)$.

For well-localized particles, $\langle \mathbf{r}_i \rangle \approx \mathbf{R}_i$, so $\langle \mathbf{r}_i \cdot \mathbf{R}_i \rangle \approx |\mathbf{R}_i|^2$ and the correction is small. But for particles that can tunnel or are not well-localized, the correction can be significant.

**However**, note the code uses a `logsumexp` smooth-min potential, NOT `V = ½ω²|r - R_assigned|²`. Each particle sees a mix of wells. This makes the virial derivation more complex because $\nabla_i V_{\text{trap}}$ involves softmax weights. Need to compute this numerically.

**This step is analytical + verification.** Write a small script that computes the correct virial for multi-well systems by numerically evaluating $\sum_i \langle \mathbf{r}_i \cdot \nabla_i V \rangle$ on existing samples.

**Files:** Create `scripts/check_virial_multiwell.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python scripts/check_virial_multiwell.py --result-dir results/20260329_134224_g6_n2_double_1_1_ctnn --device cuda:0` → expected: prints both old virial (formula-based) and new virial (numerically correct $\sum_i \langle r_i \cdot \nabla_i V \rangle$). If they differ significantly, we've found the formula bug.
**Risk:** The smooth-min potential makes analytical virial messy. Numerical evaluation via autograd on V is the cleanest path.

### Step 0.2 — Cross-check on single-well (should match)
**What:** Run the same script on a single-well result to confirm old and new virial formulas agree when there's only one well center at origin.
**Acceptance check:** `PYTHONPATH=src .venv/bin/python scripts/check_virial_multiwell.py --result-dir results/20260329_134141_g0_n2_single_pinn --device cuda:0` → expected: old and new virial agree to < 0.1%.
**Risk:** None.

### Step 0.3 — Cross-check on N=4 (2+2) double dot
**What:** Run on the N=4 (2+2) results that showed ~13–15% virial. If the corrected formula gives significantly lower virial, the plateau was a formula artifact.
**Acceptance check:** `PYTHONPATH=src .venv/bin/python scripts/check_virial_multiwell.py --result-dir results/p2fix2_n4_pinn_s901_cusp_eps_2h_20260409_104115 --device cuda:0` → expected: corrected virial printed alongside old formula. Document the difference.
**Risk:** If the correction is negligible, the 13–15% virial is real and the problem is elsewhere.

**Phase 0 Gate:** If the corrected virial formula is significantly better (drops residual by >3%), update `compute_virial_metrics` and reinterpret all prior virial results. If correction is negligible, the virial question is settled but the residual remains unexplained. Either way, record finding and proceed.

---

## Phase 1 — Exact Diagonalization Reference (~2 hours)
**Depends on:** Phase 0 complete (nice to have, not blocking)
**Goal:** Have a working exact diagonalization that produces ground-state energies for N=2 (1+1) double dot with and without magnetic field. This is the ground truth for all subsequent VMC validation.

### Step 1.1 — Locate or confirm missing diagonalization notebook
**What:** The notebook `qdsensingmarch17b.ipynb` was executed per the log at `results/imag_time_pinn/nb_diagonalization_exec_tmux.log` but is not in the repo. Check `~/.ipynb_checkpoints/`, `~/.local/`, git history (`git log --all --diff-filter=D -- '*.ipynb'`), and ask user if it's in another location.
**Files:** N/A
**Acceptance check:** `git log --all --diff-filter=D -- '*qdsensing*'` → expected: either finds the deletion commit or confirms it was never committed. Also: `find /itf-fi-ml/home/aleksns/ -maxdepth 4 -name '*qdsensing*' -o -name '*diag*nb*' 2>/dev/null | head -5`
**Risk:** Notebook may be permanently lost. If so, rebuild in Step 1.2.

### Step 1.2 — Build exact diagonalization for N=2 double dot
**What:** Create a standalone script/notebook that performs exact diagonalization for N=2 particles (one per well) in a 2D double harmonic potential with Coulomb interaction and optional Zeeman splitting. Use a product Fock-Darwin basis (single-particle HO eigenstates centered at each well), build the Hamiltonian matrix in this basis, and diagonalize with `numpy.linalg.eigh`.

Basis: For each well, use the lowest $M$ harmonic oscillator eigenstates (2D: quantum numbers $(n_x, n_y)$ with $n_x + n_y \leq n_{\max}$). Two-particle basis: $|m_L, m_R\rangle$ where $m_L$ is the state of particle in left well, $m_R$ is the state in right well. Antisymmetrize for identical fermions (but note: for 1+1 with opposite spin, no antisymmetrization needed between wells).

Matrix elements: $\langle m'_L m'_R | H | m_L m_R \rangle$ with:
- $H = T_1 + T_2 + V_{\text{trap},1} + V_{\text{trap},2} + V_{\text{Coulomb}}(r_{12}) + V_{\text{Zeeman}}$
- Single-particle parts are diagonal in the HO basis
- Coulomb matrix elements computed numerically via Gauss-Hermite quadrature

Outputs: ground-state energy $E_0$, first few excited-state energies, ground-state wavefunction coefficients. For B≠0: include Zeeman term $\frac{1}{2}g\mu_B B s_z$ per particle.

**Files:** Create `scripts/exact_diag_double_dot.py`
**Acceptance check:** `PYTHONPATH=src .venv/bin/python scripts/exact_diag_double_dot.py --n-max 6 --sep 4.0 --omega 1.0 --B 0.0` → expected: prints E_0 and first 4 eigenvalues. E_0 should be close to ~2.17 (matching VMC result for N=2 1+1 at sep=4). Then with `--B 0.5`: energy shifts due to Zeeman.
**Risk:** Coulomb matrix elements in HO basis can be slow for large basis. Mitigation: start with n_max=4 (cheap, ~100 basis states), increase to n_max=8 for convergence check.

### Step 1.3 — Validate exact diag against known limits
**What:** Check exact diag results in known limits:
- `sep=0, B=0, no Coulomb`: ground-state energy = $2 \times \hbar\omega = 2.0$ (two particles in same HO)
- `sep=0, B=0, Coulomb`: should match single-dot E≈3.0 for N=2
- `sep=20, B=0, Coulomb`: should approach $2 \times 1.0 = 2.0$ (two independent particles)
- `sep=4, B=0, Coulomb`: should match VMC E≈2.17

**Acceptance check:** `PYTHONPATH=src .venv/bin/python scripts/exact_diag_double_dot.py --validate` → expected: all 4 limits within 5% of expected values.
**Risk:** Basis truncation at small separation can be poor. n_max=8 should suffice for sep≥4 where particles are well-localized.

**Phase 1 Gate:** Exact diag reproduces known limits and VMC energy at sep=4. If not, debug diag before proceeding.

---

## Phase 2 — Validated N=2 (1+1) Ground State (~1 hour)
**Depends on:** Phase 1 gate passed
**Goal:** Train a fresh N=2 (1+1) ground state using CTNN (the architecture that worked before), validate energy against exact diag to within 1%.

### Step 2.1 — Create config for N=2 (1+1) ground state
**What:** Create YAML config based on the successful CTNN run template but with REINFORCE loss (the `fd_colloc` runs for 1+1 diverged).
Config: `system.type=double_dot, n_left=1, n_right=1, sep=4.0, omega=1.0, coulomb=true`, `arch_type=pinn` (since CTNN converged for this before, and PINN also worked with REINFORCE), `loss_type=reinforce`, 6000 epochs, seed 42.
**Files:** Create `configs/one_per_well/n2_1_1_gs_s42.yaml`
**Acceptance check:** `cat configs/one_per_well/n2_1_1_gs_s42.yaml | grep -E 'n_left|n_right|loss_type|arch_type'` → expected: n_left: 1, n_right: 1, loss_type: reinforce, arch_type: pinn
**Risk:** Low — we have prior successful runs with this setup.

### Step 2.2 — Train and compare to exact diag
**What:** Run ground-state training on GPU 0. After completion, compare final energy to exact diag E_0.
**Acceptance check:** `PYTHONPATH=src .venv/bin/python src/run_ground_state.py --config configs/one_per_well/n2_1_1_gs_s42.yaml` → expected: completes in ~10–30 min, final_energy within 1% of exact diag E_0 (expected ~2.17).
**Risk:** PINN with REINFORCE should work based on the `wk2_p1_n2_double` results (E=6.91 for 2+2 which is reasonable). If 1+1 diverges with PINN, switch to CTNN.

### Step 2.3 — Run virial with corrected formula (if Phase 0 found correction)
**What:** If Phase 0 determined a corrected virial formula, run it on this result. If not, skip.
**Acceptance check:** Virial residual printed. If corrected formula is in place, expect <5%.
**Risk:** None.

**Phase 2 Gate:** Ground-state energy matches exact diag to within 1%.

---

## Phase 3 — Magnetic Quench Time Evolution (~3 hours)
**Depends on:** Phase 2 gate passed
**Goal:** Demonstrate imaginary-time evolution under magnetic field quench for N=2 (1+1):
  - **Protocol A:** Start from B=0 ground state → turn on B for one particle → evolve to new GS
  - **Protocol B:** Start from B≠0 ground state → turn off B → evolve to true GS
Compare final energies against exact diag with the corresponding B value.

### Step 3.1 — Verify exact diag with B-field
**What:** Run exact diag for sep=4, ω=1 at B=0 and B=0.5 (and B=1.0 for range). Record E_0(B=0), E_0(B=0.5), E_0(B=1.0). These are the targets.
**Acceptance check:** `PYTHONPATH=src .venv/bin/python scripts/exact_diag_double_dot.py --sep 4.0 --omega 1.0 --B 0.0 --B 0.5 --B 1.0` → expected: table of E_0 vs B.
**Risk:** Zeeman shifts for 1+1 (opposite spin) partially cancel. May need to use `zeeman_electron1_only=True` for a visible effect.

### Step 3.2 — Protocol A: B=0 ground state → turn on B → evolve
**What:** Use `imaginary_time_pinn.py` with `magnetic_B_initial=0.0` (VMC phase finds B=0 GS) and `magnetic_B=0.5` (PDE phase evolves toward B=0.5 GS). The existing code already supports this via the `PINNConfig` fields.

Create a config or command-line invocation. The key settings:
- `n_particles=2, well_sep=4.0, omega=1.0, coulomb=True`
- `magnetic_B_initial=0.0, magnetic_B=0.5`
- `zeeman_electron1_only=True` (so the quench only affects one particle for maximum effect)
- `tau_max=5.0, n_epochs_pde=8000`

**Files:** This can be run directly via command-line args or by creating a launcher script.
**Acceptance check:** Run completes and `E(tau=tau_max)` converges to exact diag `E_0(B=0.5)` within 2%.
**Risk:** If the PDE phase does not converge (known issue: inconsistent-results quality for gap extraction), increasing `tau_max` or `n_epochs_pde` may help. The ground state convergence (E at late tau) should be more robust than gap extraction.

### Step 3.3 — Protocol B: B≠0 ground state → turn off B → evolve
**What:** Reverse: `magnetic_B_initial=0.5, magnetic_B=0.0`. VMC phase finds B=0.5 GS, PDE phase evolves toward B=0 GS.
**Acceptance check:** `E(tau=tau_max)` converges to exact diag `E_0(B=0)` within 2%.
**Risk:** Same as 3.2.

### Step 3.4 — Compare both protocols against exact diag
**What:** Create a comparison table:
| Protocol | E_initial (VMC) | E_final (tau→∞) | E_exact (diag) | Error |
**Acceptance check:** Table printed to stdout. Both errors < 2%.
**Risk:** PDE phase convergence may be slow. If tau_max=5 is insufficient, try tau_max=10.

**Phase 3 Gate:** Both protocols converge to the correct exact diag energy within 2%.

---

## Phase 4 — Generalize to N=3, N=4 One-Per-Well (~3 hours)
**Depends on:** Phase 3 gate passed
**Goal:** Extend to 3 and 4 particles in 3 and 4 wells. Validate ground states against exact diag where feasible.

### Step 4.1 — Create N=3 (1+1+1) system config
**What:** Use `SystemConfig.custom()` with 3 wells (e.g., spacing=4.0, centers at -4, 0, +4 or equidistant). Create a ground-state config.
**Files:** Create `configs/one_per_well/n3_1_1_1_gs_s42.yaml`
**Acceptance check:** `cat configs/one_per_well/n3_1_1_1_gs_s42.yaml` → shows system.type: custom, 3 wells each with n_particles: 1
**Risk:** The `run_ground_state.py` already supports `type: custom`. The wavefunction setup via `setup_closed_shell_system()` must handle 3 wells — verify that the Slater determinant, spin assignment, and well_id work for odd N (N=3 means 2 up + 1 down or vice versa).

### Step 4.2 — Train N=3 ground state
**What:** Train and check energy is physically reasonable (should be close to 3×1.0 = 3.0 for well-separated wells, lower with Coulomb at finite separation).
**Acceptance check:** Training completes, energy is in range [2.5, 4.0] for sep=4.0.
**Risk:** Odd particle number may cause spin-assignment issues. Check `setup_closed_shell_system` handles N=3.

### Step 4.3 — Exact diag for N=3 (if feasible)
**What:** Extend exact diag to N=3. This requires antisymmetrizing the 3-particle basis. For opposite spins (2 up + 1 down), the spatial wavefunction for the 2 up-spin particles must be antisymmetrized. Basis size grows as $M^3$ where $M$ is the single-particle basis size, so n_max=4 gives ~$15^3=3375$ states — feasible.
**Acceptance check:** `PYTHONPATH=src .venv/bin/python scripts/exact_diag_double_dot.py --n-wells 3 --n-max 4 --sep 4.0 --omega 1.0` → expected: E_0 printed.
**Risk:** Implementation complexity for 3-body antisymmetrization. If too complex for this session, skip and validate N=3 against the large-separation limit only.

### Step 4.4 — N=3 magnetic quench (Protocol A)
**What:** Adapt `imaginary_time_pinn.py` for N=3 system. Run Protocol A: B=0 → B=0.5.
**Acceptance check:** Training completes, E(tau_max) converges.
**Risk:** Imaginary-time code may assume N=2 in places. Need to verify generality.

### Step 4.5 — Create and train N=4 (1+1+1+1) if N=3 works
**What:** Same workflow as N=3 but with 4 wells. Only attempt if N=3 ground state + quench succeeded.
**Files:** Create `configs/one_per_well/n4_1_1_1_1_gs_s42.yaml`
**Acceptance check:** Training completes, energy is physically reasonable.
**Risk:** 4 wells → 4 particles → basis size $M^4$ for exact diag may be large. Use n_max=3 for diag.

**Phase 4 Gate:** N=3 and N=4 ground states converge. At least N=3 matches exact diag or large-separation limit.

---

## Risks and mitigations
- **Multi-well virial formula correction is negligible**: Even if virial correction is small, we still learn something definitive. Move on to the core physics.
- **Diagonalization notebook truly lost**: Rebuild in Phase 1.2. The physics is standard; a clean implementation takes ~2 hours.
- **Imaginary-time PDE doesn't converge for quench**: Increase tau_max, n_epochs_pde. If still fails, use direct VMC at the target B-field as a fallback (train two separate ground states instead of evolving).
- **N=3 odd-particle spin issues**: Check `setup_closed_shell_system` carefully. May need a 2-up-1-down configuration with a partial Slater determinant.
- **`fd_colloc` loss diverges for 1+1**: Already known from weekend campaign. Use REINFORCE.
- **`imaginary_time_pinn.py` only supports 2-particle double dot**: It uses legacy `well_sep` parameter, not `SystemConfig.custom()`. May need adapter code for N>2 / >2 wells.

## Anticipated expert invocations
None anticipated — standard implementation path. The exact diag is textbook quantum mechanics; the VMC/imaginary-time infrastructure already exists.

## Success criteria
- Multi-well virial formula audited and corrected if needed
- Exact diag reference for N=2 (1+1) with and without B-field
- N=2 (1+1) VMC ground state validated to within 1% of exact diag
- Magnetic quench protocols A and B converge to correct energies within 2%
- N=3 (1+1+1) ground state trained and validated
- N=4 (1+1+1+1) ground state trained (stretch goal)

## Current State
**Active phase:** Phase 1 — Exact Diagonalization Reference
**Active step:** Step 1.2 — Build exact diagonalization for N=2 double dot
**Last evidence:** 
- Legacy target run now executes through a compatibility path: `PYTHONPATH=src .venv/bin/python scripts/check_virial_multiwell.py --result-dir results/20260329_134224_g6_n2_double_1_1_ctnn --device cuda:0` -> `E≈967.25`, old virial `199.92%`, new virial `213.06%` (numerically valid execution but physically implausible).
- Modern control run is physically consistent and shows strong virial-formula effect: `PYTHONPATH=src .venv/bin/python scripts/check_virial_multiwell.py --result-dir results/p2fix2_n4_pinn_s901_cusp_eps_2h_20260409_104115 --device cuda:0` -> `E≈7.0226`, old virial `14.58%`, new virial `1.71%`.
- Step 0.2 acceptance check passed on single-well legacy PINN checkpoint after mapped-legacy loader support: `PYTHONPATH=src .venv/bin/python scripts/check_virial_multiwell.py --result-dir results/20260329_134141_g0_n2_single_pinn --device cuda:0` -> `E≈3.0199`, old virial `1.67%`, new virial `1.66%`, `Delta residual=-0.000373` (old/new formulas agree as expected for single-well).
- Step 0.3 acceptance check re-run (current evidence): `PYTHONPATH=src .venv/bin/python scripts/check_virial_multiwell.py --result-dir results/p2fix2_n4_pinn_s901_cusp_eps_2h_20260409_104115 --device cuda:0` -> `E≈6.9841`, old virial `13.43%`, new virial `2.79%`, `Delta residual=1.1331`.
- Shared evaluator updated and verified: `PYTHONPATH=src .venv/bin/python scripts/run_virial_check.py --result-dirs results/p2fix2_n4_pinn_s901_cusp_eps_2h_20260409_104115 --device cuda:0 --n-samples 2048` -> generalized virial `1.11%` vs legacy comparator `14.60%`.
- Phase 1 Step 1.1 acceptance checks executed:
  - `git log --all --diff-filter=D -- '*qdsensing*'` -> no output (no deletion record in repo history).
  - `find /itf-fi-ml/home/aleksns/ -maxdepth 4 \( -name '*qdsensing*' -o -name '*diag*nb*' \) 2>/dev/null | head -5` -> only unrelated notebook outside repo; missing notebook not found.
  - Provenance check: `grep -n 'qdsensingmarch17b.ipynb' results/imag_time_pinn/nb_diagonalization_exec_tmux.log` shows execution start/finish for `src/qdsensingmarch17b.ipynb`, confirming it existed at runtime but is now absent.
**Current risk:** Legacy CTNN-jastrow replay for `results/20260329_134224_g6_n2_double_1_1_ctnn` remains scientifically untrusted (implausible energy), but the generalized virial computation path is validated on modern and legacy-PINN checkpoints.
**Next action:** Implement Step 1.2 by creating a new exact-diagonalization script for N=2 double-dot with optional Zeeman term and run its acceptance checks.
**Blockers:** Missing `src/qdsensingmarch17b.ipynb` cannot be recovered from current repo or nearby filesystem indices; rebuilding exact diagonalization is required.
