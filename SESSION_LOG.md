# Session Log

Last session: [2026-05-03] — GPU tmux: real-time Coulomb quench (hybrid) + N=8 inverse-design lite (end-to-end ⟨S₀·S₇⟩); repository snapshot committed and pushed.
Prior highlight: [2026-04-26] — Inverse-design framework, N=2 smoke test completion, N=4 spin-amplitude flagship infrastructure
See ARCHIVE.md for full history.

## Current session

**Project objective:** Build out an ambitious inverse-design programme on top of the non-MCMC NQS pipeline. Target: end-to-end demonstrations that the deterministic, E-ref-free training enables gradient-driven Hamiltonian engineering.

**Active plan file:** [plans/2026-04-26_inverse_design_framework.md](plans/2026-04-26_inverse_design_framework.md)

**2026-05-03 — GPU jobs (tmux) + git**

Detached sessions (attach with `tmux attach -t <name>`). Logs under `runlogs/`.

| Session | GPU | Command / outputs |
|--------|-----|-------------------|
| `mwpd-rt-coulomb` | cuda:0 | `PYTHONUNBUFFERED=1 PYTHONPATH=src python3.11 -u scripts/run_realtime_n2_coulomb_quench.py …` (same flags as below) → `results/realtime_pinn/coulomb_quench/hybrid_l0p5_e2500_20260503/`; tee `runlogs/coulomb_hybrid_e2500_20260503.log` — `-u` so logs stream during training |
| `mwpd-invdes-n8` | cuda:1 (`CUDA_MANUAL_DEVICE=1`) | `run_inverse_design.py --config configs/one_per_well/n8_invdes_lite_s42.yaml --parametrisation dimer_chain_n8 --param-init 4 4 4 4 --target spin_correlator --pair 0 7 --mode neg_value --sense max --n-steps 4 --lr 0.3 --gradient-method fd_forward --stage-a-min-energy 999.0 --out-dir results/inverse_design/n8_lite_pair07_negval_20260503` → tee `runlogs/invdes_n8_lite_20260503.log` |

**Git:** Source tree snapshot (real-time PINN + quench drivers, inverse-design stack, configs, reports, tests). Large generated `results/` trees remain local unless explicitly tracked already.


**Completed (this session):**
- ✅ N=2 smoke test on dot-label negativity: 8 outer steps, T 0.077 → 0.448 (Bell limit 0.5), 2 h 5 min on cuda:3.
- ✅ Mott spin-amplitude extractor (`src/observables/spin_amplitude_entanglement.py`): handles arbitrary N>=2 in one-per-well regime, computes signed amplitudes, well-set bipartite entropy / negativity.
- ✅ Validation on N=2 singlet checkpoint: c_(0,1) = +0.707, c_(1,0) = −0.707, S = ln 2 = 0.693, neg = 0.5 (exact Bell singlet).
- ✅ N=4 PINN extraction at uniform d=4: spin-rotation invariant ground state with AFM Néel-pattern dominance, S({0,1}|{2,3}) = 0.785, neg = 0.826.
- ✅ Heisenberg cross-check tool (`src/observables/heisenberg_reference.py`, `scripts/heisenberg_cross_check.py`): N=4 PINN @ d=4 has overlap 0.939 with the pure OBC Heisenberg AFM ground state. Off-Mott boundary corrections enhance bipartite spin entanglement.
- ✅ CLI evaluator `scripts/evaluate_spin_amplitude_entanglement.py` and trajectory analyser `scripts/analyze_n4_inverse_design.py`.
- ✅ Symmetric N=4 dimer-chain parametrisation `make_dimer_chain_n4_param_to_wells` exposed via `--parametrisation dimer_chain_n4` in CLI.
- ✅ **Phase 1B — Multi-sector inverse design / exchange-gap target.** Refactored `src/geometry_optimizer.py` to support multiple spin sectors per geometry (`GeomEvalContext`, `spin_overrides`, per-sector warm-starting). Added `exchange_gap` built-in target with N-aware sector defaults and `--target-J` / `--unsigned-gap` CLI flags. Built a unified N=2 baseline config `configs/one_per_well/n2_invdes_exchange_baseline_s42.yaml` (multi_ref ansatz so both singlet (1,1) and triplet (2,0) train through the same recipe). Smoke test: J = E_T − E_S, 0.093 → 0.185 Ha in 4 outer steps (d 2.0 → 1.77), 20 min on a 2080 Ti — opposite direction to the entanglement target (gate speed vs coherence tradeoff demonstrated end-to-end).
- ✅ **Phase 2A.6 — Effective Heisenberg J_ij extractor** (`src/observables/effective_heisenberg.py`, `scripts/evaluate_effective_heisenberg.py`, `scripts/validate_effective_heisenberg.py`). Covariance / parent-Hamiltonian fit: build the K×K Gram matrix Q_(αβ) = ⟨c|S_α·S_β|c⟩ − ⟨c|S_α|c⟩⟨c|S_β|c⟩, find J in the kernel direction so that |c⟩ is the ground state of H_eff(J)=Σ J_α S_α. Includes direct ⟨S_i·S_j⟩ correlator extraction. Validated against synthetic OBC Heisenberg up to N=6 (overlap 1.0, residual < 1e-9) and N=4 d=4 PINN (NN-only fit, overlap 0.963, J_NN=[1.00, 1.16, 0.27], spread 0.89 — boundary-enhanced inhomogeneity consistent with super-exchange ~ exp(−d/L)).
- ✅ **Phase 2A.7 — J_ij ambiguity diagnosis & resolution.** Discovered that for N≥4 multi-dimensional spin sectors there is a (typically 1+ dimensional) null space of Q where many J vectors all make |c⟩ an eigenstate. The unique physical answer is the J that makes |c⟩ the *ground* state of H_eff(J) — selected via overlap-maximisation (`_resolve_J_direction`, full-K-dim search with random + axis seeds + SciPy/coordinate polishing). For exact eigenstates this recovers truth. For approximate states the relevant figure of merit is `relative_residual = J^T Q J / ‖J‖² / λ_max(Q)` and `overlap(|c⟩, |gs(H_eff)⟩)`. Pivoted: for inverse design, prefer **direct correlator targets** (unambiguous) and reserve J_ij as an analysis observable.
- ✅ **Phase 2A.8 — Trajectory analyser** `scripts/amplitude_evolution.py` (generic for any N): per outer step extracts Mott amplitudes, full ⟨S_i·S_j⟩ matrix, NN-only J_ij fit, Heisenberg-overlap, relative-residual; emits CSV summary, NPZ array dump, and a 6-panel matplotlib figure.
- ✅ **Phase 2B prep** — `make_dimer_chain_n8_param_to_wells` in `src/geometry_optimizer.py` (symmetric 4-parameter N=8 chain `d1|d2|d3|d4|d3|d2|d1`, halves FD-gradient cost via reflection symmetry); `configs/one_per_well/n8_invdes_baseline_s42.yaml` (pinn_hidden=96, pinn_layers=3, bf_hidden=48, epochs=4000, n_coll=1024, sampler_dimer_pairs=4); `--parametrisation dimer_chain_n8`, `--target spin_correlator`, `--target effective_J`, `--pair`, `--mode`, `--target-value`, `--corr-spin-sector`, `--effJ-pairs` CLI surface in `scripts/run_inverse_design.py`.

**In-flight:**
- 🚧 N=4 entanglement flagship — `results/inverse_design/n4_flagship_p2a_aggressive`. lr=3.0, 10 steps, FD-step=0.4, bounds [2.0, 9.0]^2 on cuda:3. As of last poll: step 9/10 (final centre eval running), T monotonic 0.785 → 0.918 by step 7, θ evolving [4, 4] → [5.16, 3.15] (soft-dimer regime, as predicted).
- 🚧 N=4 Phase 2B engineer-to-spec demo — `results/inverse_design/n4_phase2b_corr_target_demo` on cuda:6: drive ⟨S_0·S_3⟩ → −0.65 (target) using `dimer_chain_n4`, lr=5.0, 8 steps, `--mode neg_squared_error --target-value -0.65 --stage-a-min-energy 999.0`. Just launched (step 0 in progress).
- 🚧 (Failed) N=2 `--target-J 0.05` validation on cuda:6 *finished but diverged* — Stage B variance refinement collapsed the singlet energy at step 1; root cause: `--stage-a-min-energy` was not high enough to disable Stage B. Best result was step 0 (J ≈ 0.083, close to target 0.05). Should be relaunched with `--stage-a-min-energy 999.0` once GPUs free up.

**Recommended starting point:** Watch for `n4_flagship_p2a_aggressive` to finish (step 9/10 in flight on cuda:3). When it completes, run `PYTHONPATH=src python3.11 scripts/amplitude_evolution.py --run-dir results/inverse_design/n4_flagship_p2a_aggressive --pairs 0,1 1,2 2,3 0,3` to produce the 6-panel trajectory figure (correlators + NN-J + overlap + entanglement). Then watch for the Phase 2B demo on cuda:6 (~3–4 h, 8 steps × 5 inner trainings each). After both finish, **the immediate next ambitious move is N=8 inverse design**: `--parametrisation dimer_chain_n8 --param-init 4.0 4.0 4.0 4.0 --target spin_correlator --pair 0 7 --mode neg_value` on cuda:6, with the `n8_invdes_baseline_s42.yaml` config. This is the biggest demonstration of the end-to-end pipeline (16-fold spin sector, deterministic gradients, no MCMC) we have ever attempted.

**Open questions:**
- Does the N=4 inverse-design loop drive S past the Heisenberg limit (0.319) by exploiting the off-Mott enhancement we already observed at uniform d=4 (0.785), or does it simply move to the asymptotic dimer geometry where S → ln 4? *(Latest: T = 0.918 at step 7 with θ = soft dimer — consistent with dimer interpretation but `amplitude_evolution.py` is needed to show whether the Heisenberg overlap drops monotonically.)*
- At the optimum θ*, what is the spin-amplitude pattern? We expect dominance of the cross-bipartition singlet pair structure `|singlet_{12}⟩ ⊗ |singlet_{03}⟩`.
- Is the Heisenberg overlap monotonically decreasing along the trajectory (because the optimum lives in non-Heisenberg territory) or non-monotonic?
- For the exchange-gap target: how does J extracted from `E_T − E_S` compare to J extracted from the Mott-projected spin amplitudes (covariance fit)? Two independent estimators of the same physical quantity — agreement strengthens both.
- For the Phase 2B demo: can the optimiser truly *engineer* a specific intermediate ⟨S_0·S_3⟩ = −0.65 (between the singlet-pair limit −0.75 and the AFM-Néel value −0.20)? If so, this is the cleanest demonstration of differentiable Hamiltonian engineering on a many-body ground state in the literature.
- For N=8: what is the gradient SNR at uniform d=4 with the `dimer_chain_n8` 4-D parametrisation? The deterministic loss is differentiable, but each FD-central step costs 8 inner trainings (4 params × 2 sides) — we should measure wall-time per outer step before committing to a 10-step run.

**Unverified assumptions:**
- The PINN's enhanced bipartite entanglement at uniform d=4 (S=0.785 vs Heisenberg 0.319) is robust against further training and not a residual variational error.
- The Mott projection is reliable down to d_middle ≈ 2; below that off-Mott contamination may dominate.
- The `dimer_chain_n4` parametrisation captures the relevant inversion symmetry; an asymmetric optimum is unlikely.
- The N=2 fully-polarised triplet (n_up=2, n_down=0) and the singlet (1,1) sectors trained with the same `multi_ref` ansatz give a faithful gap (no apples-to-oranges variational bias). At d=4 a CI cross-check will settle this.

**Active workarounds:**
- Stage B disabled in inverse-design inner loop (`--stage-a-min-energy 999`); for N=2 it was confirmed harmful, for N=4 we're being cautious until further evidence.
- Shared-CI Coulomb kernel quadrature double-count bug remains documented but unfixed (Phase 0 bonus); does not affect entanglement metrics, only CI energy comparisons.
- Triplet sector forces `architecture.singlet=False` and `multi_ref=True` because the legacy N=2 singlet permanent ansatz is hardwired to (1,1).

**Foundation status:**
- Mott spin-amplitude extraction is validated (perfect singlet on N=2, AFM-dominant Néel pattern on N=4).
- Bilevel inverse-design loop is correctness-checked: single-sector entanglement target (N=2 smoke), single-sector spin entanglement target (N=4 flagship in flight), multi-sector exchange-gap target (N=2 smoke complete).
- Heisenberg cross-check is a working comparator at any N up to ~12 in dense diag form.
- Multi-sector machinery (`GeomEvalContext`, `spin_overrides`) is generic — handles any number of independently-warm-started sectors.
- **Effective Heisenberg machinery validated:** covariance fit recovers truth at zero residual on synthetic OBC chains up to N=6, and gives a physically-sensible NN J pattern on the N=4 d=4 PINN. The fit is explicitly known to be non-unique for N≥4 in multi-D spin sectors (multiple kernel directions), so we report `relative_residual` and `overlap` alongside J for honest interpretation.
- **Direct correlator pipeline:** `spin_pair_correlator` and `spin_correlator_target` give un-ambiguous ⟨S_i·S_j⟩ values from the same Mott amplitudes used for entanglement; this is the preferred inverse-design observable.
- **N=8 infrastructure ready:** parametrisation, baseline config, CLI, and trajectory analyser are all in place. No N=8 inverse-design run yet attempted.

**Context freshness:** fresh

**Contradiction flags:** The earlier journal entry (now superseded) reported a "PINN coherence deficit" at N=2 d≥6. This was a stale measurement; the current evaluator gives `dot_neg = 0.5` to numerical precision, in agreement with CI. The PINN is *not* coherence-deficient for N=2.

## Session metrics (latest)
**Steps completed (this session):** 11 major (Phase 1E smoke; Phase 2A extractor + validation + CLI + cross-check tool + dimer parametrisation; Phase 1B multi-sector refactor + exchange-gap target + smoke test; Phase 2A.6 effective-J extractor + validate; Phase 2A.8 trajectory analyser; Phase 2B prep — N=8 parametrisation + config + spin_correlator target + CLI) + 2 in flight (N=4 entanglement flagship at step 9/10; N=4 Phase 2B engineer-to-spec demo at step 0).
**Material deviations:** 2 (first N=4 flagship run aborted due to too-small lr=0.5; restarted with lr=3.0. `target-J 0.05` validation diverged at step 1 due to Stage B variance refinement collapsing the singlet — needs relaunch with `--stage-a-min-energy 999.0`.)
**Evaluation gates triggered:** 5 (N=2 spin-amplitude validation against Bell singlet; Heisenberg cross-check at N=4 d=4; N=2 exchange-gap monotonicity + sign correctness vs super-exchange prediction; effective-J synthetic round-trip on OBC N≤6; N=4 PINN NN-only J fit reproducing super-exchange decay).
**Unresolved uncertainties:** 4 (Heisenberg overlap d-dependence along the N=4 flagship trajectory; Mott projection floor below d≈2.5; gap validation against CI at d=4; N=8 gradient SNR + per-step wall time).
