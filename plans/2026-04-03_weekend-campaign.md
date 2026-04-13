# Plan: Weekend GPU Campaign — Fix Pipeline → Architecture Comparison → Production

Date: 2026-04-03
Status: in-progress

## Project objective
Produce publication-quality VMC ground-state energies and virial-validated wavefunctions for N=2 and N=4 double quantum dots with Coulomb interaction, comparing PINN, CTNN, and unified architectures for the first time with proper dispatch.

## Objective
Over 48–72 hours, iteratively: (1) fix the critical wavefunction bug, (2) validate on non-interacting systems, (3) run the first real architecture comparison (PINN vs CTNN vs unified with proper dispatch), and (4) produce production Coulomb results with virial < 5%.
Success condition: at least 2 architecture×system combinations with Coulomb achieve virial < 5% for N=4 double-dot across 2+ seeds.

## Context
The MH validation run (April 3, `p23_retry_n2_single_mh`) completed 10k epochs on the refactored wavefunction and **catastrophically failed**: energy ≈ 14 for N=2 non-interacting (target: E=2.0), variance ≈ 50. Meanwhile, the **pre-refactor** April 1 runs on the same system converged to E ≈ 1.95 with variance < 0.02.

**Root cause identified (Layer 2 — Implementation):**
The refactored `GroundStateWF` (commit `2f79940`) wraps `PINN.forward()` which returns only the correlator `f_NN(x) + cusp(x)`. But the thesis ansatz is `Ψ = SD(x) · exp(f(x))`, so `log|Ψ| = log|SD| + f`. The Gaussian/Slater envelope (`log|SD| = -ω/2 · Σ rᵢ²`) is **entirely missing**. Without it, the model starts from Ψ ≈ 1 (flat) instead of Ψ ≈ Gaussian (correct shape), and cannot recover in 10k epochs.

**Additional discovery:** At commit `6203fd8` (pre-clean), `wavefunction.py` was **never tracked**. The old campaigns (March 30–31 evidence table) used the **toy wavefunction** (Gaussian + MLP + Jastrow) which discarded `arch_type`. This means the "CTNN vs unified" comparisons in `results/validation_20260331/evidence_table.json` were **all running the same toy model** — no real architecture comparison has ever been done.

**Prior negative findings (do not repeat):**
- IS sampler diverges with current codebase. Use MH only.
- IS + fd_colloc + bf_hidden=64 → ESS collapse → garbage.
- Langevin-IS path is scientifically invalid.

## Approach
Four phases, dependency-ordered. Phase 0 is a ~2h blocking fix (add Gaussian envelope + validate). Phase 1 launches overnight non-interacting validation on 4 GPUs. Phase 2 runs the first genuine architecture comparison with Coulomb. Phase 3 scales the best configuration to production. Each phase gates on the previous one's acceptance checks. All runs use tmux + MH sampler. 8× RTX 2080 Ti available; GPUs 2,3,4,7 are fully free (~11 GB); GPUs 0,1,5,6 have ~7 GB free.

## Foundation checks (must pass before new code)
- [x] Data pipeline known-input check — `sample_multiwell_init` verified per-well occupation
- [x] Split/leakage validity check — VMC has no train/test split; no leakage possible
- [x] Baseline existence — old toy-model results exist (E≈1.95 for N=2 NI, virial 3.5% for N=4 Coulomb)
- [x] Relevant existing implementation read and understood — PINN.py forward returns correlator only, needs external envelope
- [ ] **Gaussian envelope addition** — BLOCKING. Must be done in Phase 0.

## Scope
**In scope:**
- Fix GroundStateWF Gaussian/Slater envelope (Phase 0)
- Non-interacting validation: N=2, N=4, single/double dot (Phase 1)
- Architecture comparison: PINN vs CTNN vs unified, N=2 and N=4 Coulomb (Phase 2)
- Production runs: best architecture, multiple ω, virial validation (Phase 3)
- Commit all fixes and results at each phase boundary

**Out of scope:**
- IS sampler debugging (broken, use MH)
- Imaginary-time spectroscopy (separate plan if Phase 3 succeeds)
- Well-separation sweeps (separate plan)
- Normalizing flow integration
- Magnetic field / Zeeman features
- README or thesis updates

---

## Phase 0 — Fix Gaussian Envelope (BLOCKING)
**Goal:** GroundStateWF produces physically correct log_psi = log|SD| + PINN_correlator. N=2 NI energy < 2.5 within 500 epochs on CPU.
**Estimated scope:** 1 file edit + 1 config + 1 smoke test. ~2 hours.

### Step 0.1 — Add Gaussian envelope to GroundStateWF.forward()
**What:** In `src/wavefunction.py`, add the harmonic oscillator ground-state envelope before the PINN correlator. For closed-shell N-particle system in D dimensions, the Slater determinant of the lowest HO orbitals is a product of single-particle Gaussians: `log|SD| = -ω/2 · Σᵢ |rᵢ|²` (up to a constant). This is the single dominant term that makes Ψ peak at the origin and decay at large r.

The forward method should become:
```python
# Gaussian envelope (HO ground state for all particles)
r2_sum = (x * x).sum(dim=(-1, -2))  # (B,)
log_envelope = -0.5 * self.system.omega * r2_sum

# PINN correlator (learns corrections: pair correlations, backflow, etc.)
x_eval = x
if self.backflow is not None:
    dx = self.backflow(x, spin=spin)
    x_eval = x + dx
correlator = self.pinn(x_eval, spin=spin).squeeze(-1)

log_psi = log_envelope + correlator
```

**Files:** `src/wavefunction.py` — modify `GroundStateWF.forward()`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python -c "
import torch; from config import SystemConfig; from wavefunction import GroundStateWF, setup_closed_shell_system
sys = SystemConfig.single_dot(N=2, omega=1.0, dim=2)
C, s, p = setup_closed_shell_system(sys, device='cpu', dtype=torch.float64, E_ref='auto', allow_missing_dmc=True)
m = GroundStateWF(sys, C, s, p, arch_type='ctnn')
x = torch.randn(4, 2, 2, dtype=torch.float64)
lp = m(x); print('log_psi:', lp)
# At origin, log_psi should be ~0 (envelope=0), far away should be very negative
x0 = torch.zeros(1, 2, 2, dtype=torch.float64); x_far = 5*torch.ones(1, 2, 2, dtype=torch.float64)
print('origin:', m(x0).item(), 'far:', m(x_far).item())
assert m(x0).item() > m(x_far).item(), 'Envelope must make psi decay away from origin'
print('PASS')
"` → expected: `PASS`, origin > far
**Risk:** PINN's NN part may produce large values that fight the envelope at init. Mitigate: verify rho output scale is small (< 1.0) at initialization.

### Step 0.2 — Smoke test on GPU: N=2 NI, 500 epochs
**What:** Run a quick 500-epoch MH training on GPU. With correct envelope, energy should drop toward 2.0 quickly.
**Files:** Use `configs/validation/ni_n2_single_mh.yaml` (already exists, adjust epochs to 500)
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src .venv/bin/python src/run_ground_state.py --config configs/validation/ni_n2_single_mh.yaml 2>&1 | tail -5` → expected: final energy < 4.0 (trending toward 2.0), no NaN/Inf
**Risk:** If energy > 4.0 after 500 epochs, the PINN init may be too noisy. Mitigate: add `0.01 *` scaling on PINN output at init, or zero-init the rho output layer.

### Step 0.3 — Commit fix
**What:** Git commit the wavefunction fix + any IS/collocation fixes still unstaged.
**Files:** `src/wavefunction.py`, `src/training/sampling.py`, `src/training/collocation.py`, `src/training/vmc_colloc.py`, `tests/test_training.py`
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && git diff --stat HEAD` → expected: shows modified files; then `git add -p && git commit` → clean working tree for modified tracked files
**Risk:** None.

---

## Phase 1 — Non-Interacting Validation Grid (overnight)
**Depends on:** Phase 0 complete (energy < 4.0 in smoke test)
**Goal:** All 4 non-interacting cases converge to within 2% of analytic energy after 10k epochs with MH.
**Estimated scope:** 4 YAML configs, 4 tmux sessions. ~4–8 hours wall-clock (parallel on 4 GPUs).

### Step 1.1 — Create/update 4 validation configs
**What:** Create YAML configs for N=2 single, N=2 double, N=4 single, N=4 double. All use: MH sampler, 10k epochs, fd_colloc loss, lr=0.001, no warmup, n_coll=512, grad_clip=1.0, coulomb=false.

| Config | N | Type | E_exact | GPU |
|--------|---|------|---------|-----|
| ni_n2_single_mh.yaml | 2 | single_dot | 2.0 | cuda:2 |
| ni_n2_double_mh.yaml | 2 | double_dot (sep=4) | 2.0 | cuda:3 |
| ni_n4_single_mh.yaml | 4 | single_dot | 6.0 | cuda:4 |
| ni_n4_double_mh.yaml | 4 | double_dot (sep=4) | 4.0 | cuda:7 |

**Files:** `configs/validation/ni_n2_single_mh.yaml` (update), create 3 new configs
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && for f in configs/validation/ni_n*_mh.yaml; do echo "$f:"; PYTHONPATH=src .venv/bin/python -c "import yaml; c=yaml.safe_load(open('$f')); print('  sampler:', c['training']['sampler'], 'loss:', c['training']['loss_type'])"; done` → expected: all show `sampler: mh loss: fd_colloc`
**Risk:** N=4 double dot target is 4.0 only if wells are far enough apart. If wells overlap, use N×ω=4.0 as rough target and accept wider tolerance (5%).

### Step 1.2 — Launch 4 tmux sessions
**What:** One tmux session per GPU. Run `run_ground_state.py --config <yaml>` in each.
**Files:** None (runtime)
**Acceptance check:** `tmux ls | wc -l` → expected: at least 4 sessions; `for s in p1_n2s p1_n2d p1_n4s p1_n4d; do echo "$s:"; tmux capture-pane -t $s -p 2>/dev/null | tail -1; done` → expected: all show epoch progress
**Risk:** GPU OOM for N=4 with n_coll=512. Mitigate: reduce n_coll to 256 for N=4 configs.

### Step 1.3 — Morning check: convergence validation
**What:** After ~8h, check logs. Extract final energy and variance for each run.
**Acceptance check:** `cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && for log in results/phase1_ni_*/result.json; do echo "$log:"; PYTHONPATH=src .venv/bin/python -c "import json; d=json.load(open('$log')); print(f'  E={d[\"final_energy\"]:.4f} var={d[\"final_variance\"]:.6f}')"; done` → expected: E within 2% of analytic for all 4 cases.

**Gate:** If any case has >5% error or variance >1.0, STOP and diagnose before Phase 2.

---

## Phase 2 — Architecture Comparison with Coulomb (Saturday)
**Depends on:** Phase 1 gate passed (all 4 NI cases within 2%)
**Goal:** First genuine PINN vs CTNN vs unified comparison on N=2 and N=4 double-dot Coulomb systems. Identify best architecture by virial metric.
**Estimated scope:** 12 YAML configs (3 arch × 2 systems × 2 seeds), 12 runs across 4 GPUs (3 batches of 4). ~12–18 hours total.

### Step 2.1 — Create architecture comparison configs
**What:** Create configs for 3 architectures × 2 systems × 2 seeds. All use: MH sampler, fd_colloc loss, 30k epochs, lr=0.0005, warmup=500, cosine to lr_min_factor=0.01, bf_hidden=64, n_coll=512.

| Run name | N | arch | seed | GPU batch |
|----------|---|------|------|-----------|
| arch_n2_pinn_s1 | 2 | pinn | 301 | Batch A |
| arch_n2_ctnn_s1 | 2 | ctnn | 301 | Batch A |
| arch_n2_unified_s1 | 2 | unified | 301 | Batch A |
| arch_n2_pinn_s2 | 2 | pinn | 302 | Batch A |
| arch_n4_pinn_s1 | 4 | pinn | 401 | Batch B |
| arch_n4_ctnn_s1 | 4 | ctnn | 401 | Batch B |
| arch_n4_unified_s1 | 4 | unified | 401 | Batch B |
| arch_n4_pinn_s2 | 4 | pinn | 402 | Batch B |
| arch_n2_ctnn_s2 | 2 | ctnn | 302 | Batch C |
| arch_n2_unified_s2 | 2 | unified | 302 | Batch C |
| arch_n4_ctnn_s2 | 4 | ctnn | 402 | Batch C |
| arch_n4_unified_s2 | 4 | unified | 402 | Batch C |

System: double_dot, sep=4, ω=1.0, coulomb=true for all.
GPUs: cuda:2, cuda:3, cuda:4, cuda:7 (all ~11 GB free)

**Files:** Create `configs/arch_comparison/` directory with 12 YAML files
**Acceptance check:** `ls configs/arch_comparison/*.yaml | wc -l` → expected: 12
**Risk:** Unified or CTNN backflow may have bugs from bytecode recovery. Mitigate: run a 100-epoch smoke test for each arch_type before committing to 30k.

### Step 2.2 — Smoke test all 3 architectures (100 epochs each)
**What:** Quick GPU smoke test for each arch_type to verify forward/backward passes work under training.
**Files:** None (runtime)
**Acceptance check:** `for arch in pinn ctnn unified; do echo "$arch:"; PYTHONPATH=src .venv/bin/python -c "
import torch; from config import SystemConfig; from wavefunction import GroundStateWF, setup_closed_shell_system
sys = SystemConfig.double_dot(N_L=1, N_R=1, sep=4.0, omega=1.0, dim=2)
C, s, p = setup_closed_shell_system(sys, device='cuda:2', dtype=torch.float64, E_ref='auto', allow_missing_dmc=True)
m = GroundStateWF(sys, C, s, p, arch_type='$arch', bf_hidden=64).to('cuda:2')
x = torch.randn(16, 2, 2, dtype=torch.float64, device='cuda:2')
lp = m(x); loss = lp.mean(); loss.backward()
print(f'  log_psi mean={lp.mean().item():.2f} grad_norm={sum(p.grad.norm().item()**2 for p in m.parameters() if p.grad is not None)**0.5:.4f}')
"; done` → expected: all 3 produce finite values and finite gradients
**Risk:** CTNNBackflowNet or UnifiedCTNN may have recovered-code bugs. If smoke test fails for an architecture, exclude it from comparison and note.

### Step 2.3 — Launch Batch A (4 runs on 4 GPUs)
**What:** Start 4 runs simultaneously. Each runs 30k epochs (~4–6h for N=2).
**Files:** None (runtime)
**Acceptance check:** `tmux ls | grep arch_ | wc -l` → expected: 4 sessions running
**Risk:** 30k epochs may be too long if model diverges early. Mitigate: check at epoch 5000 — if energy > 20 for N=2 or > 40 for N=4, kill and investigate.

### Step 2.4 — Launch Batch B (after Batch A frees GPUs or on remaining GPUs)
**What:** Start N=4 runs and remaining N=2 runs. Schedule sequentially if GPUs are limited.
**Acceptance check:** Same as 2.3.

### Step 2.5 — Launch Batch C (remaining runs)
**What:** Complete the 12-run grid.
**Acceptance check:** Same as 2.3.

### Step 2.6 — Compile architecture comparison table
**What:** Extract energy, variance, virial for all 12 runs. Compute mean±std across seeds. Create results/arch_comparison_20260404/comparison_table.json.
**Acceptance check:** `PYTHONPATH=src .venv/bin/python -c "
import json, glob
results = []
for f in sorted(glob.glob('results/arch_*/result.json')):
    d = json.load(open(f))
    results.append(f'{f}: E={d[\"final_energy\"]:.4f}')
print('\n'.join(results))
"` → expected: 12 result lines, energies in physically reasonable range (1–10 for N=2, 5–20 for N=4)
**Risk:** Some architectures may not converge. This IS a valid result — it tells us which architecture works.

---

## Phase 3 — Production Runs (Sunday)
**Depends on:** Phase 2 complete. Best architecture identified (lowest virial, most consistent across seeds).
**Goal:** Production-quality results for the best architecture across N=2,4 with Coulomb. Virial < 5% for all cases.
**Estimated scope:** 8–12 runs, ~24h total. Exact grid depends on Phase 2 findings.

### Step 3.1 — Select best architecture from Phase 2
**What:** Rank architectures by: (1) virial residual, (2) energy variance, (3) consistency across seeds. Pick top for production.
**Files:** Analysis script or manual inspection
**Acceptance check:** Write decision to `DECISIONS.md` with specific numbers from Phase 2.
**Risk:** No architecture achieves virial < 10%. Mitigate: fall back to the toy wavefunction (Phase 0 code pre-refactor) as emergency baseline — it produced virial 3.5% in March.

### Step 3.2 — Create production config grid
**What:** Best architecture × [N=2, N=4] × [ω=0.5, ω=1.0] × [2 seeds] = 8 runs.
All use: MH sampler, fd_colloc, 50k epochs, lr=0.0005, warmup=1000, cosine lr, bf_hidden=64, n_coll=512.
**Files:** `configs/production/` directory with 8 YAML files
**Acceptance check:** `ls configs/production/*.yaml | wc -l` → expected: 8

### Step 3.3 — Launch production runs (2 batches of 4)
**What:** 4 GPUs × 2 batches, each ~12h.
**Acceptance check:** After completion, all 8 result.json files exist under `results/production_*/`
**Risk:** 50k epochs may OOM on N=4. Mitigate: reduce to 30k if memory issues.

### Step 3.4 — Virial validation and evidence table
**What:** Compute virial for all production runs. Create `results/production_20260405/evidence_table.json`.
**Acceptance check:** `PYTHONPATH=src .venv/bin/python -c "
import json, glob
for f in sorted(glob.glob('results/production_*/result.json')):
    d = json.load(open(f))
    print(f'{f}: E={d[\"final_energy\"]:.4f} virial={d.get(\"virial_pct\", \"N/A\")}')
"` → expected: virial < 5% for at least 2 configurations
**Risk:** Virial computation may not be integrated into training loop. Mitigate: run post-hoc virial check using `src/observables/validation.py`.

### Step 3.5 — Commit results and tag
**What:** Commit all production configs and result summaries. Tag: `result/2026-04-05-weekend-production`.
**Acceptance check:** `git tag -l 'result/2026-04-05*'` → expected: tag exists
**Risk:** None.

---

## Automation: Campaign Runner Script
**Needed for Phases 1–3.** Create a lightweight bash script that:
1. Takes a directory of YAML configs + GPU list
2. Assigns configs round-robin to GPUs
3. Launches each in a named tmux session
4. Provides a `--status` flag to check all running sessions

**Files:** `scripts/run_campaign.sh`
**Acceptance check:** `bash scripts/run_campaign.sh --help` → expected: prints usage

This avoids manual tmux management for 12+ runs.

---

## Risks and mitigations
- **Gaussian envelope doesn't fix the problem (Phase 0 fails):** Fall back to toy wavefunction (Gaussian+MLP+Jastrow from commit 1708b0a). It worked in April 1 runs. But then architecture comparisons are impossible.
- **CTNN/unified backflow have bugs from bytecode recovery (Phase 2 fails for those arches):** Accept that only PINN arch works. Still a valid result — report with explanation.
- **N=4 runs OOM or take too long:** Reduce n_coll to 256, reduce bf_hidden to 32, or reduce epochs. N=4 with 2 spatial dims is not that large.
- **Virial > 10% for all Coulomb runs:** This would indicate a systematic problem in the local energy computation or the ansatz. Diagnose at Layer 3 (architecture) — possibly the PINN correlator capacity is too small. Try hidden_dim=128.
- **GPU contention from other users:** GPUs 2,4,7 are currently empty. Prefer those. Monitor with nvidia-smi.

## Success criteria
- Phase 0: N=2 NI energy ≈ 2.0 (±5%) after 500 GPU epochs
- Phase 1: All 4 NI cases within 2% of analytic energy
- Phase 2: At least 2 of 3 architectures produce finite results for N=4 Coulomb; comparison table compiled
- Phase 3: At least 1 configuration with virial < 5% for N=4 double-dot Coulomb
- Overall: Committed code and results, tagged in git, reproducible from config+commit

## Current State
**Active phase:** Phase 2 (interacting architecture comparison)
**Active step:** Execute bounded long-run set with new cusp+eps fixes (<=2h wall-clock total) and collect real interacting results.
**Last evidence:** Fixed smoke run `p2fix2_n4_pinn_s901_cusp_eps_smoke_20260408_154419` gives `E=7.015898`, `virial=13.67%` vs pre-fix smoke `E=7.176194`, `virial=38.45%` under identical evaluator settings.
**Current risk:** Unknown whether improvements persist over longer training and across seed variation within the 2-hour campaign budget.
**Next action:** Run a timing-calibrated sanity check, launch two parallel long runs (seed 901/902), then evaluate both with parity virial diagnostics.
**Blockers:** None
