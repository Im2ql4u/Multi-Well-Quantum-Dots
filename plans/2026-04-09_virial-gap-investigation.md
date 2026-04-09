# Plan: Close the 15% Virial Gap — Well-Aware Architecture + Sanity Checks

Date: 2026-04-09
Status: in-progress

## Project objective
Produce publication-quality VMC ground-state energies and virial-validated wavefunctions for N=2 and N=4 double quantum dots with Coulomb interaction. The hard gate: virial residual < 5% for N=4 double-dot Coulomb.

## Objective
Reduce the virial residual from the current ~15% plateau to < 5% for the N=4 double-dot Coulomb system, primarily through architecture changes that make the wavefunction aware of well structure. Secondary: rule out sampling and loss as contributors via controlled sanity checks. Success condition: at least one architecture variant achieves virial < 5% across 2 seeds.

## Context

### What we know
Three physics bugs were fixed on 2026-04-08 (Kato cusp constants, cusp decay length, Coulomb softening). This brought virial from **38.4% → ~15%** — a large improvement, but it plateaued there and training longer (3k→6k epochs) did not help.

Two 6000-epoch runs with different seeds (901, 902) both plateau at virial ~15% with energy ~7.0. This rules out:
- **Longer training alone** (Layer 5) — already tested, no improvement from 3k→6k
- **Seed sensitivity** — both seeds give same virial
- **Previous physics bugs** — those were at Layer 2 and are now fixed

### What we suspect
The remaining 15% virial gap is likely a **Layer 3 (architecture)** problem. The fundamental issue: in a double-well system, the physics of particle correlations has two distinct regimes:

1. **Intra-well pairs** — particles in the SAME well: close proximity, strong Coulomb repulsion, strong exchange effects, need to satisfy Kato cusp
2. **Inter-well pairs** — particles in DIFFERENT wells: large separation (~4.0 a.u. = well separation), weak Coulomb (≈1/4), primarily dipole-like coupling

The current architecture treats **all particle pairs identically** (aside from spin). The PINN's ψ branch computes the same safe pair features for a pair of particles 0.5 a.u. apart in the same well as for a pair 4.0 a.u. apart in different wells. The backflow similarly has no notion of which well a particle belongs to.

This is analogous to using a single-site model to describe a diatomic molecule — it will get the energy roughly right but miss the distinct local-vs-nonlocal correlation structure needed for virial accuracy.

### Prior negative findings (do not repeat)
- IS sampler diverges. Use MH only.
- Flat Gaussian (no Slater determinant) fails catastrophically.
- Pre-fix cusp constants were wrong by factor of 2.
- Coulomb eps=1.0 was catastrophically too strong.

### Virial comparison baseline (from `results/p2fix2_virial_comparison_20260409.json`)

| Run | Epochs | Virial % | E_mean | E_std | Notes |
|-----|--------|----------|--------|-------|-------|
| pre-fix smoke | 3000 | 38.4% | 7.198 | 1.362 | Before cusp+eps fix |
| post-fix smoke | 3000 | 14.8% | 7.028 | 0.456 | After fix |
| post-fix 2h s901 | 6000 | 14.9% | 7.021 | 0.301 | Plateau confirmed |
| post-fix 2h s902 | 6000 | 15.3% | 6.996 | 1.220 | Plateau confirmed |

## Approach

**Primary focus: Architecture** (Phases 2–3). Introduce well-awareness into the wavefunction so the network naturally separates intra-well from inter-well correlations. Test three complementary ideas with increasing complexity, each compared A/B against the current baseline.

**Secondary: Sanity checks** (Phase 1). Before touching architecture, verify that sampling and loss are not silently degrading the virial. This is a quick "clear the floor" step — the non-interacting results suggest these are fine, but we must confirm for the interacting case.

**Design principle**: Every experiment is an A/B comparison against the current baseline config (the 6k-epoch seed 901 run). We change ONE thing at a time. Each experiment runs 6000 epochs on a single GPU (~36 min), so we can run 4 in parallel on GPUs 0,1,2,4.

## Foundation checks (must pass before new code)
- [x] Data pipeline known-input check — multi-well init verified
- [x] Split/leakage validity — VMC has no train/test split
- [x] Baseline existence — p2fix2 runs serve as baseline (virial ~15%, E ~7.0)
- [x] Physics fixes committed — **NOT YET: src/PINN.py and src/training/collocation.py have unstaged changes**
- [ ] **Commit physics fixes** — Step 0.1

## Scope
**In scope:**
- Commit existing physics fixes
- Sampling sanity check (MH walker distribution visualization)
- Loss function sanity check (REINFORCE vs weak_form vs fd_colloc A/B)
- Well-aware pair features (PINN modification)
- Well-aware backflow (BackflowNet modification)
- Well-aware CTNN backflow (CTNNBackflowNet modification)
- Fair A/B comparisons at 6000 epochs
- Virial evaluation of all runs

**Out of scope:**
- Longer training alone (already tested, doesn't help)
- IS sampler (known broken)
- Imaginary-time / spectroscopy
- Well-separation sweeps
- New loss function design (virial penalty etc.) — only if architecture alone fails
- README / thesis updates

---

## Phase 0 — Commit and Baseline Lock (15 min)
**Goal:** Commit the three physics fixes, establish the exact baseline config as a reference point.

### Step 0.1 — Commit physics fixes
**What:** Stage and commit the three code changes in `src/PINN.py` (cusp constants, cusp decay length) and `src/training/collocation.py` (Coulomb eps).
**Files:** `src/PINN.py`, `src/training/collocation.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && git add src/PINN.py src/training/collocation.py && git diff --cached --stat`
→ expected: shows 2 files modified, then commit with descriptive message.
**Risk:** None.

### Step 0.2 — Record baseline config as canonical reference
**What:** Copy the 2h seed 901 config to a `configs/baseline/` directory for explicit reference. All future A/B tests will use this config as the "B" (control).
**Files:** Create `configs/baseline/n4_dd_pinn_s901_baseline.yaml` (copy of `configs/phase2_fix_20260408/n4_pinn_s901_cusp_epsfix_2h.yaml`)
**Acceptance check:** `diff configs/baseline/n4_dd_pinn_s901_baseline.yaml configs/phase2_fix_20260408/n4_pinn_s901_cusp_epsfix_2h.yaml` → expected: identical (or differs only in run_name)
**Risk:** None.

---

## Phase 1 — Sampling & Loss Sanity Checks (~2 hours wall-clock)
**Depends on:** Phase 0 complete
**Goal:** Confirm that the ~15% virial gap is NOT caused by sampling bias or loss function choice. This clears Layers 4–5 so we can focus on Layer 3.
**Estimated scope:** 2 diagnostic scripts + 3 training runs (parallel). ~2 hours.

### Step 1.1 — MH walker distribution check
**What:** Run the trained baseline model (seed 901 checkpoint) with 50k MH samples and visualize the 2D marginal density. Check that:
  (a) walkers populate both wells roughly equally (n_left ≈ 2, n_right ≈ 2 per sample)
  (b) the per-well marginal density has the expected ~Gaussian shape centered on each well
  (c) there is no "stuck" mode where walkers don't cross between wells

This requires a small script that loads the checkpoint, runs MH sampling with many steps, and saves histograms.

**Files:** Create `scripts/check_mh_distribution.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python scripts/check_mh_distribution.py --result-dir results/p2fix2_n4_pinn_s901_cusp_eps_2h_20260409_104115 --n-samples 50000 --mh-steps 100 --device cuda:0` → expected: prints per-well particle counts (should be ~2.0 each for 4 particles, 2 per well), mean positions near ±2.0, and saves a density plot.
**Risk:** If walkers are trapped in one well, the virial gap is a sampling problem (Layer 4), not architecture. Mitigation: increase MH step size or use parallel tempering.

### Step 1.2 — Loss function A/B: REINFORCE vs fd_colloc vs weak_form
**What:** The baseline uses `loss_type: reinforce` with MAD clipping. Run two additional 6000-epoch experiments changing ONLY the loss type:
  - **Experiment A**: `loss_type: fd_colloc` (direct mean of local energy, gradient through everything)
  - **Experiment B**: `loss_type: weak_form` (identical to fd_colloc in this codebase — confirm)

All other settings identical to baseline. If the virial gap is caused by REINFORCE's stop-gradient on E_L, one of these should show meaningful improvement.

**Files:** Create `configs/phase1_sanity/n4_dd_pinn_s901_fdcolloc.yaml`, `configs/phase1_sanity/n4_dd_pinn_s901_weakform.yaml`
**Acceptance check:** `for f in configs/phase1_sanity/*.yaml; do echo "$f:"; grep loss_type "$f"; done` → expected: one says `fd_colloc`, one says `weak_form`
**Risk:** If fd_colloc or weak_form crash with NaN (they differentiate through the FD Laplacian which is expensive), reduce n_coll to 256. This is a known issue — REINFORCE avoids this cost by stop-gradient.

### Step 1.3 — Run loss A/B experiments + baseline rerun
**What:** Launch 3 runs in parallel: fd_colloc (GPU 0), weak_form (GPU 1), baseline re-run with different seed 903 (GPU 2) as a control.
**Files:** Create `configs/phase1_sanity/n4_dd_pinn_s903_reinforce.yaml` (baseline with seed 903)
**Acceptance check:** After ~40 min, all three complete. `ls results/p1san_*/result.json | wc -l` → expected: 3
**Risk:** fd_colloc/weak_form may be 3–4× slower per epoch due to differentiating through the Laplacian. If so, reduce epochs to 2000 and note.

### Step 1.4 — Virial comparison for Phase 1
**What:** Run `scripts/run_virial_check.py` on all Phase 1 results + baseline.
**Acceptance check:** `PYTHONPATH=src .venv/bin/python scripts/run_virial_check.py --result-dirs results/p2fix2_n4_pinn_s901_cusp_eps_2h_20260409_104115 results/p1san_*` → expected: virial numbers for all runs. If all are ~15%, loss is not the problem. If one is < 10%, that loss type helps and we adopt it.

**Phase 1 Gate:** If all loss types give virial ~15% AND sampling is verified, proceed to Phase 2 (architecture changes). If a loss type helps, adopt it as the new baseline for Phase 2.

---

## Phase 2 — Well-Aware Architecture: Pair Features (~3 hours wall-clock)
**Depends on:** Phase 1 gate passed
**Goal:** Modify the PINN correlator to distinguish intra-well from inter-well particle pairs. Test whether this improves virial.

### Background: The Physics Motivation

In a double-dot with separation d=4.0, the N=4 electrons (2 per well) experience two qualitatively different types of correlation:

```
Well L (center at -2.0)           Well R (center at +2.0)
  e₀ ←→ e₁                         e₂ ←→ e₃
  (intra-well: strong               (intra-well: strong
   Coulomb, ~1/r at r~0.5)          Coulomb)
          ↕                                  ↕
   ← ← ← ← inter-well: weak Coulomb ~1/4 → → → →
```

The intra-well correlation is dominated by the Kato cusp (short-range, r→0 behavior) and exchange. The inter-well correlation is a longer-range dipole-like effect. A single set of pair features cannot easily represent both.

### Step 2.1 — Add well-identity infrastructure to SystemConfig and wavefunction
**What:** Each particle needs a `well_id` attribute. In `SystemConfig`, the wells already track which particles belong to which well via `WellSpec.n_particles`. We need to:
  (a) Create a `well_id` tensor of shape (N,) with integer labels (0 for left-well particles, 1 for right-well, etc.)
  (b) Pass it through `setup_closed_shell_system()` into `GroundStateWF` as a buffer
  (c) Make it available to the PINN and backflow modules

**Concrete implementation:**
```python
# In setup_closed_shell_system():
well_ids = []
for well_idx, well in enumerate(system.wells):
    well_ids.extend([well_idx] * int(well.n_particles))
well_id = torch.tensor(well_ids, device=device, dtype=torch.long)
# Return it in params dict:
params["well_id"] = well_id

# In GroundStateWF.__init__():
well_id = params.get("well_id", torch.zeros(system.n_particles, dtype=torch.long))
self.register_buffer("well_id", well_id, persistent=False)
```

**Files:** `src/config.py` (if needed), `src/wavefunction.py`, `src/PINN.py`
**Acceptance check:** `PYTHONPATH=src .venv/bin/python -c "
import torch
from config import SystemConfig
from wavefunction import GroundStateWF, setup_closed_shell_system
sys = SystemConfig.double_dot(N_L=2, N_R=2, sep=4.0, omega=1.0, dim=2)
C, s, p = setup_closed_shell_system(sys, device='cpu', dtype=torch.float64, E_ref='auto', allow_missing_dmc=True)
print('well_id:', p['well_id'])
m = GroundStateWF(sys, C, s, p, arch_type='pinn', pinn_hidden=64, pinn_layers=2, bf_hidden=64, bf_layers=2)
print('model well_id buffer:', m.well_id)
"` → expected: `well_id: tensor([0, 0, 1, 1])`, model buffer matches
**Risk:** Low. Pure plumbing change.

### Step 2.2 — Add well-aware pair features to PINN
**What:** Modify `PINN.forward()` to compute a `same_well` indicator for each i<j pair, analogous to how `same_spin` is already handled. Then use it to:

(a) **Gate the NN pair features**: scale the ψ-branch output by a learned factor depending on whether the pair is intra-well or inter-well. Concretely, add a small 2-output MLP that takes `same_well` (and optionally the raw pair distance) and produces two scaling factors — one for intra-well, one for inter-well.

(b) **Separate pooling**: compute `psi_mean_intra` (average over same-well pairs) and `psi_mean_inter` (average over cross-well pairs) separately, then concatenate them into the ρ input. This doubles the pair information going into the readout.

The key API change is that `PINN.forward()` now accepts an optional `well_id` tensor:
```python
def forward(self, x, spin=None, well_id=None):
    ...
    if well_id is not None:
        wi = well_id[self.idx_i]  # (P,)
        wj = well_id[self.idx_j]  # (P,)
        same_well = (wi == wj).unsqueeze(0).unsqueeze(-1).to(x.dtype)  # (1,P,1)
    else:
        same_well = torch.ones(1, n_pairs, 1, device=x.device, dtype=x.dtype)

    # Split psi_out into intra and inter pools
    psi_intra = (psi_out * same_well).sum(1) / same_well.sum(1).clamp(1)
    psi_inter = (psi_out * (1 - same_well)).sum(1) / (1 - same_well).sum(1).clamp(1)
    # rho input now includes both: [phi_mean, psi_intra, psi_inter, extras]
```

This changes the ρ input from `2*dL + 2` to `3*dL + 2`. The ρ MLP must be rebuilt with this new input size.

**Files:** `src/PINN.py` (modify `PINN.__init__`, `PINN.forward`), `src/wavefunction.py` (pass `well_id` through)
**Acceptance check:** `PYTHONPATH=src .venv/bin/python -c "
import torch
from config import SystemConfig
from wavefunction import GroundStateWF, setup_closed_shell_system
sys = SystemConfig.double_dot(N_L=2, N_R=2, sep=4.0, omega=1.0, dim=2)
C, s, p = setup_closed_shell_system(sys, device='cpu', dtype=torch.float64, E_ref='auto', allow_missing_dmc=True)
m = GroundStateWF(sys, C, s, p, arch_type='pinn', pinn_hidden=64, pinn_layers=2, bf_hidden=64, bf_layers=2)
x = torch.randn(4, 4, 2, dtype=torch.float64)
lp = m(x)
print('log_psi shape:', lp.shape, 'values:', lp)
loss = lp.mean(); loss.backward()
gn = sum(p.grad.norm().item()**2 for p in m.parameters() if p.grad is not None)**0.5
print('grad_norm:', gn)
assert torch.isfinite(lp).all(), 'Non-finite log_psi'
print('PASS')
"` → expected: PASS with finite values and gradients
**Risk:** The ρ MLP dimension change breaks checkpoint loading from old runs. This is fine — we are starting fresh experiments. Mitigate: make well-aware features opt-in via a config flag `use_well_features: true` so old configs still work.

### Step 2.3 — Config flag and backward compatibility
**What:** Add a config field `architecture.use_well_features: true/false` (default: false). When false, the PINN behaves exactly as before (all pairs pooled together). When true, activates the well-aware split pooling from Step 2.2. This lets us do a clean A/B comparison.

**Files:** `src/run_ground_state.py` (parse new config field), `src/wavefunction.py` (pass to PINN), `src/PINN.py` (conditional logic)
**Acceptance check:** `PYTHONPATH=src .venv/bin/python -c "
import torch
from config import SystemConfig
from wavefunction import GroundStateWF, setup_closed_shell_system
sys = SystemConfig.double_dot(N_L=2, N_R=2, sep=4.0, omega=1.0, dim=2)
C, s, p = setup_closed_shell_system(sys, device='cpu', dtype=torch.float64, E_ref='auto', allow_missing_dmc=True)
# Without well features (backward compat)
m_old = GroundStateWF(sys, C, s, p, arch_type='pinn', pinn_hidden=64, pinn_layers=2, bf_hidden=64, bf_layers=2, use_well_features=False)
# With well features
m_new = GroundStateWF(sys, C, s, p, arch_type='pinn', pinn_hidden=64, pinn_layers=2, bf_hidden=64, bf_layers=2, use_well_features=True)
x = torch.randn(4, 4, 2, dtype=torch.float64)
print('old:', m_old(x))
print('new:', m_new(x))
print('PASS — both produce finite output')
"` → expected: both produce finite values
**Risk:** None.

### Step 2.4 — A/B experiment: well-aware PINN vs baseline
**What:** Create two configs:
  - **A (treatment):** Same as baseline + `use_well_features: true`
  - **B (control):** Exact baseline (seed 901, 6000 epochs)
Also run treatment with seed 902 for seed variation.

Run 3 experiments in parallel on GPUs 0, 1, 2.

**Files:** `configs/phase2_wellaware/n4_dd_wellpinn_s901.yaml`, `configs/phase2_wellaware/n4_dd_wellpinn_s902.yaml`, `configs/phase2_wellaware/n4_dd_baseline_s901_rerun.yaml`
**Acceptance check:** After ~40 min, `ls results/p2wa_*/result.json | wc -l` → expected: 3
**Risk:** Well-aware split may hurt energy if the architecture is over-parameterized for 6 pairs (N=4 has only 6 pairs: 2 intra-L, 2 intra-R, 4 cross... wait, let me count: with N=4, i<j pairs = 6 total. If 2 particles per well: intra-L = C(2,2)=1, intra-R = 1, cross = 2×2=4. So 2 intra pairs and 4 inter pairs). The split pooling averages over different-sized groups — ensure proper normalization.

### Step 2.5 — Virial evaluation for Phase 2
**What:** Run virial check on Phase 2 results + baseline.
**Acceptance check:** `PYTHONPATH=src .venv/bin/python scripts/run_virial_check.py --result-dirs results/p2fix2_n4_pinn_s901_cusp_eps_2h_20260409_104115 results/p2wa_*` → expected: virial numbers. If treatment < baseline by > 2%, well-awareness helps.

**Phase 2 Gate:** If well-aware PINN shows virial improvement > 2% absolute over baseline, proceed to Phase 3 (extend to backflow). If no improvement, proceed to Phase 3 anyway but shift focus to backflow well-awareness.

---

## Phase 3 — Well-Aware Backflow + Combined (~3 hours wall-clock)
**Depends on:** Phase 2 complete
**Goal:** Extend well-awareness to the backflow network. The backflow determines the "virtual coordinates" x_eval = x + Δx, which are then fed to the PINN. If backflow doesn't know about well structure, it may produce Δx displacements that inappropriately mix particles across wells.

### Background: Why Backflow Matters

The backflow transformation x → x + Δx(x) is supposed to capture "beyond-Slater-determinant" correlation by allowing the effective coordinates in the correlator to depend on all particle positions. For a double-well system:
- **Good backflow:** Particles in the same well push each other's effective positions apart (cusp avoidance). Particles in different wells barely affect each other.
- **Bad backflow:** All particles affect each other equally regardless of well, producing a generic "smoothing" that misses the distinct local/nonlocal structure.

### Step 3.1 — Add well-aware message weighting to BackflowNet
**What:** Modify `BackflowNet.forward()` to accept `well_id` and use it to modulate message weights. Specifically:
  - Add two learnable scalar weights: `w_intra` and `w_inter` (initialized to 1.0 each)
  - For each i,j pair, multiply the message by `w_intra` if same well, `w_inter` if different well
  - This gives the network a knob to independently control the strength of intra-well vs inter-well backflow influence

```python
# In BackflowNet.__init__:
self.w_intra = nn.Parameter(torch.ones(1))
self.w_inter = nn.Parameter(torch.ones(1))

# In BackflowNet.forward:
if well_id is not None:
    wi = well_id.view(1, N, 1, 1).expand(B, N, N, 1)
    wj = well_id.view(1, 1, N, 1).expand(B, N, N, 1)
    same_well = (wi == wj).to(x.dtype)
    well_weight = same_well * self.w_intra + (1 - same_well) * self.w_inter
    m_ij = m_ij * well_weight
```

**Files:** `src/PINN.py` (modify `BackflowNet.__init__`, `BackflowNet.forward`), `src/wavefunction.py` (pass `well_id`)
**Acceptance check:** `PYTHONPATH=src .venv/bin/python -c "
import torch
from PINN import BackflowNet
bf = BackflowNet(d=2, msg_hidden=64, msg_layers=2, hidden=64, layers=2)
x = torch.randn(4, 4, 2, dtype=torch.float64)
spin = torch.tensor([0, 0, 1, 1])
well_id = torch.tensor([0, 0, 1, 1])
dx = bf(x, spin=spin, well_id=well_id)
print('dx shape:', dx.shape, 'mean:', dx.abs().mean().item())
assert torch.isfinite(dx).all()
print('PASS')
"` → expected: PASS
**Risk:** The well weights may start at 1.0/1.0 (no distinction) and never diverge if gradient signal is weak. Mitigate: initialize `w_intra=1.0, w_inter=0.1` to bias toward local backflow.

### Step 3.2 — Same modification for CTNNBackflowNet
**What:** Apply the same well-aware weighting to CTNNBackflowNet's edge features. The CTNN has an explicit edge tensor `h_e: (B,N,N,H)` — multiply it by the same `same_well * w_intra + (1-same_well) * w_inter` before aggregation.
**Files:** `src/PINN.py` (modify `CTNNBackflowNet`)
**Acceptance check:** Same pattern as 3.1 but with `CTNNBackflowNet`.
**Risk:** Same as 3.1.

### Step 3.3 — A/B experiments: 4 architecture variants
**What:** Run 4 experiments (parallel on 4 GPUs), all 6000 epochs:

| Name | PINN well-aware | BF well-aware | Arch type | GPU | Seed |
|------|-----------------|---------------|-----------|-----|------|
| A: baseline | no | no | pinn | 0 | 901 |
| B: well-PINN only | yes | no | pinn | 1 | 901 |
| C: well-BF only | no | yes | pinn | 2 | 901 |
| D: well-PINN + well-BF | yes | yes | pinn | 4 | 901 |

All other hyperparameters identical. This is a 2×2 factorial design that tells us whether the benefit comes from PINN pair awareness, backflow awareness, or both.

**Files:** Create 4 YAML configs in `configs/phase3_wellaware/`
**Acceptance check:** `ls configs/phase3_wellaware/*.yaml | wc -l` → expected: 4. After ~40 min running: `ls results/p3wa_*/result.json | wc -l` → expected: 4.
**Risk:** Running 4 experiments at once may cause GPU contention. Monitor with `nvidia-smi`.

### Step 3.4 — Run factorial + seed confirmation
**What:** After the 4-run factorial, take the best variant and run it with seed 902 to confirm.
**Acceptance check:** Virial check on 5 runs (4 factorial + 1 seed confirmation).
**Risk:** None beyond training time.

### Step 3.5 — Virial evaluation and factorial analysis
**What:** Run virial check on all Phase 3 results. Compute the 2×2 interaction effect.

Expected outcome table (to fill in):
```
                    BF well-aware=no    BF well-aware=yes
PINN well-aware=no  virial=15% (A)      virial=?% (C)
PINN well-aware=yes virial=?% (B)       virial=?% (D)
```

If there's an additive benefit: D < min(B, C) < A ≈ 15%.
If there's an interaction: the combined effect is non-additive.

**Phase 3 Gate:** If ANY variant achieves virial < 10%: strong signal, proceed to Phase 4 (CTNN comparison). If best is still > 12%: the well-awareness idea alone is insufficient, need to consider Phase 4 alternatives.

---

## Phase 4 — CTNN Architecture Comparison + Best-of Protocol (~3 hours)
**Depends on:** Phase 3 complete
**Goal:** Compare the best well-aware PINN from Phase 3 against the CTNN architecture (also with well-awareness). The CTNN has a fundamentally different message-passing structure (explicit edge tensor, linear transport maps) that may handle the intra/inter distinction differently.

### Step 4.1 — CTNN well-aware A/B
**What:** Run 4 experiments:

| Name | Arch | Well-aware | Seed | GPU |
|------|------|------------|------|-----|
| E: CTNN baseline | ctnn | no | 901 | 0 |
| F: CTNN well-aware | ctnn | yes | 901 | 1 |
| G: CTNN well-aware | ctnn | yes | 902 | 2 |
| H: Best PINN (from P3) | pinn | yes | 902 | 4 |

**Files:** `configs/phase4_ctnn/` with 4 YAML configs
**Acceptance check:** After ~40 min: `ls results/p4ct_*/result.json | wc -l` → expected: 4
**Risk:** CTNN may be slower per epoch (O(N²H) dense ops). If >2× slower, reduce epochs to 3000.

### Step 4.2 — Comprehensive virial comparison table
**What:** Compile ALL virial results from Phases 1–4 into a single comparison table. This is the primary deliverable.

**Acceptance check:** A JSON file at `results/virial_investigation_summary_20260409.json` with all runs, plus a human-readable summary printed to stdout.

### Step 4.3 — Select winner and document decision
**What:** Based on the comparison table:
  - If any variant achieves virial < 5%: SUCCESS. Document in `DECISIONS.md`.
  - If best is 5–10%: Partial success. Identify what's still missing.
  - If best is still > 10%: Architecture alone is insufficient. Need to consider loss redesign (virial penalty) or fundamentally different ansatz.

**Files:** `DECISIONS.md`
**Acceptance check:** `grep -c "virial" DECISIONS.md` → expected: > 0

---

## Risks and mitigations
- **Well-awareness doesn't help virial**: The 15% gap may be due to finite-difference Laplacian accuracy rather than architecture. Mitigation: in Phase 1, check if reducing `fd_h` from 0.01 to 0.005 changes the virial measurement. This is a pure post-hoc check on the existing checkpoint — no retraining needed.
- **fd_colloc/weak_form crash with NaN**: These loss types differentiate through the FD Laplacian, which is expensive and sometimes unstable. Mitigation: reduce n_coll to 256, add gradient clipping.
- **CTNN/unified bugs from bytecode recovery**: Phase 4 tests CTNN. If forward/backward fails, skip and note. This is a valid finding.
- **GPU OOM for N=4 with well-aware features**: The extra features increase memory. Mitigation: reduce n_coll to 384 if needed.
- **The virial check itself is inaccurate**: If the FD step h used in post-hoc evaluation is too coarse, the virial decomposition (T, V_trap, V_int) may be noisy. Mitigation: run virial check with both h=0.01 and h=0.005 to check convergence.

## Success criteria
- Phase 0: Physics fixes committed to git
- Phase 1: Sampling verified healthy (walkers in both wells); loss type doesn't explain the gap
- Phase 2: Well-aware PINN implemented with backward-compatible flag; A/B comparison shows direction of effect
- Phase 3: 2×2 factorial identifies which component (PINN or BF) benefits from well-awareness
- Phase 4: Best configuration identified across all architecture variants
- Overall: At least one variant with virial < 10% (stretch: < 5%) across 2 seeds

## Current State
**Active phase:** Phase 0 - Commit and Baseline Lock
**Active step:** 0.2 - Record baseline config as canonical reference
**Last evidence:** `git add src/PINN.py src/training/collocation.py && git diff --cached --stat` -> 2 files changed (6 insertions, 4 deletions); commit `5de3c35` created
**Current risk:** The well-awareness idea may not be sufficient; unknown if problem is architectural or numerical
**Next action:** Create `configs/baseline/n4_dd_pinn_s901_baseline.yaml` from the phase2 2h seed-901 config and validate with `diff`
**Blockers:** None — all GPUs free (0,1,2,4,5,6,7 available), code is functional
