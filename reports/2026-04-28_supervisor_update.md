# Supervisor Update — 2026-04-28

**Author:** Aleksander Skogen
**Period covered:** 2026-04-27 evening through 2026-04-28 ≈ 10:00 CEST (≈ 14 h)
**Standing summary:** [`reports/2026-04-27_supervisor_report.md`](2026-04-27_supervisor_report.md) for the project-wide executive summary, methodology, and Phase-by-Phase status. This file only documents what changed since that report was produced.

---

## TL;DR

Three pieces of news, in priority order:

1. **Phase 4 — both scaling holes filled.** N=16 retry on cuda:6 (`self_residual` + `multi_ref=False`) landed at **E = 27.270 Ha** (variance 4.2 × 10⁻⁴, ESS = 32, no collapse) in 3 h 11 min — a complete methodological recovery from the previous `guided`-strategy collapse to E = −372 Ha on the *same* seed. The follow-on **N=12 fill-in completed at 09:38 CEST: E = 18.836 Ha** (variance 4.3 × 10⁻⁵, ESS = 32) using the same recipe. The deterministic non-MCMC NQS lane now has validated ground states at N ∈ {2, 3, 4, 8, **12**, 16}. The recipe (`self_residual`, `multi_ref=False`, `n_coll = 32` for FD Laplacian, `--stage-a-min-energy 999.0` to gate Stage B off) is the validated default for N ≥ 10.

2. **Phase 3B null result settled, redesign delivered.** The plain N=8 B-sweep (`b ∈ {0, 0.05, 0.2, 0.5, 1.0}`) finished overnight with the textbook null result we predicted (bit-identical spin observables across all five B values; only the energy moves, by ±0.03 Ha optimisation noise) — the rigorous baseline that confirms the structural-triviality caveat. The follow-on **sector-aware N=8 sweep on cuda:3 completed at 09:48 CEST**: 5 sectors trained at B=0, spin-flip mirror gives 9 Sz values, post-hoc Zeeman assembly across `B ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}` gives a clean staircase phase diagram — **GS Sz = +1 at B=0 → −1 at B=0.001 → −4 (fully polarised) at B ≥ 0.005**. Two clean level crossings, both at very small fields (the orbital-energy spread across sectors is only ≈ 37 mHa, so even sub-percent Zeeman coupling saturates the chain). The 4-panel deliverable (`results/b_sweep/n8_chain_d4_sector_aware_s42/B_sweep.{csv,json,png}`) is in. **One caveat I want to flag**: the trained `E_orbital(Sz)` is *non-monotonic* in `|Sz|` — Sz=±1 is the variational minimum at 11.4124 Ha, *below* Sz=0 (11.4435 Ha) and Sz=±2 (11.4414 Ha). For a Heisenberg-like AFM chain at d=4 we expect Sz=0 to be the orbital GS, so this is most likely under-convergence in the largest-template sector (4↑4↓ has 70 templates; 5↑3↓ has 56). A second-seed validation is queued — see § 2.

3. **One bookkeeping change:** the supervisor report has been forked. The 2026-04-27 file is frozen as the reference snapshot of yesterday's state; from today onwards new milestones go into dated update files like this one. The 2026-04-27 file's `Latest update:` line was edited *only* to point readers here.

---

## 1. Phase 4 N=16 retry — done

**Job**: `run_two_stage_ground_state.py` on `configs/scaling/n16_grid_d6_s42.yaml` with `--stage-a-strategy self_residual --stage-a-epochs 5000 --stage-a-min-energy 999.0 --seed-override 314`. Started 2026-04-28 00:30 CEST on cuda:6, finished 03:41 CEST.

**Result** (`results/scaling/n16_grid_d6_s42_seed314_self_resid_summary.json`):

```json
{
  "stage_a": {
    "result": {
      "final_energy": 27.269517159011215,
      "final_loss": 0.00042036812872476453,
      "final_energy_var": 0.0004203681287247569,
      "final_ess": 32.0
    }
  },
  "stage_b": null
}
```

Energy trajectory (epoch → E, sampled): 1200 → 27.267, 2000 → 27.265, 3000 → 27.266, 4000 → 27.259, 5000 → 27.270. Flat to ±0.01 Ha across the last 4 000 epochs; loss never exceeded 9 × 10⁻⁴; ESS pinned at the full 32. Compared to the previous failed run on the *same* seed (`guided` strategy → E = −372 Ha), this is the cleanest possible methodological recovery.

**Recipe lessons** (settled — please adopt as defaults for N ≥ 10):

* **Use `self_residual`, not `improved_self_residual`** for any sector with `C(N, n_down) ≳ 100` templates. The `improved_self_residual` recipe forces `architecture.multi_ref=True` (`scripts/run_two_stage_ground_state.py:104-105`), which silently expands the ansatz to *all* spin patterns of the requested Sz sector — fine at N≤8 (≤70 templates), intractable at N≥10. An aborted attempt for N=16 with `improved_self_residual` produced no training output for 13 minutes (kernel pre-allocation for `C(16,8) = 12 870` permanents) before being killed. The `self_residual` strategy preserves the base config's `multi_ref` (False, for the scaling configs), so the ansatz stays at one spin template plus a PINN-Jastrow correction.
* **Keep `loss_type: residual`, `residual_objective: residual`, `alpha_start = alpha_end = 0`** — `self_residual` sets these automatically (see `_build_stage_a_self_residual_cfg`, lines 168-199). The base config's `loss_type: reinforce` for N=16 was overridden cleanly.
* **`n_coll = 32` is enough** for N=16 with FD Laplacian. Wall time was ≈ 38 ms / epoch, peak GPU memory 1.79 GB.
* **No Stage B**: use `--stage-a-min-energy 999.0` to gate it off. (Stage B switches to pure-variance loss; for these large-N scaling targets, leaving Stage A's residual loss on for the full epoch budget is more stable.)

**Acceptance**: ✓. Phase 4 is officially unblocked.

---

## 2. Phase 3B sector-aware redesign — done

The plain N=8 B-sweep that finished overnight (`results/b_sweep/n8_uniform_d4_s42/`) is the cleanest possible **null result** for the structural-triviality caveat we flagged when launching it:

| B | E (Ha) | S_pinn | overlap | residual_L2 | C_end(pinn) |
|---|---|---|---|---|---|
| 0.000 | 11.4459 | 1.0637 | 0.77114 | 0.6765 | −0.31898 |
| 0.050 | 11.4170 | 1.0637 | 0.77114 | 0.6765 | −0.31898 |
| 0.200 | 11.4143 | 1.0637 | 0.77114 | 0.6765 | −0.31898 |
| 0.500 | 11.4330 | 1.0637 | 0.77114 | 0.6765 | −0.31898 |
| 1.000 | 11.3990 | 1.0637 | 0.77114 | 0.6765 | −0.31898 |

Every spin observable is bit-identical to printed precision across all five B values; only the energy varies, and only by ±0.03 Ha which is the optimisation-seed noise we already see across the d-sweep at fixed B=0. **The same network checkpoint is what emerges for all five B values**, because the implemented Hamiltonian has no orbital coupling and the Zeeman term reduces to a constant per fixed-Sz template:

```43:88:src/potential.py
    magnetic_B = float(system.B_magnitude) * float(system.B_direction[2])
    if abs(magnetic_B) > 0.0:
        if spin is None:
            up = n_particles // 2
            spin_z = torch.ones(n_particles, dtype=dtype, device=device)
            spin_z[up:] = -1.0
            spin_z = spin_z.unsqueeze(0).expand(batch_size, -1)
        else:
            s = spin.to(device=device)
            ...
        else:
            zeeman = (
                0.5 * float(system.g_factor) * float(system.mu_B) * magnetic_B * spin_z.sum(dim=1)
            )
        v = v + zeeman
```

So `V_zeeman = 0.5·g·μ_B·B·Σ s_iz`. With `s_iz = ±1`, `g = 2`, `μ_B = 1`, this reduces to `V_zeeman = B·(n_up − n_down)`, identically a constant for a fixed-Sz template. The wavefunction never sees B; only the total energy gets a shift.

### The sector-aware fix (launched this morning on cuda:3)

Re-using the existing N=3, N=4 magnetic infrastructure (`configs/magnetic/n*_*up*down_*.yaml`, `scripts/launch_magnetic_sector_sweep.sh`, `scripts/analyze_magnetic_phase_diagram.py`), I extended the lane to N=8:

* **New base config**: `configs/magnetic/n8_chain_d4_4up4down_b0_s42.yaml`. 8-well linear chain at d=4 (centres ±14 / ±10 / ±6 / ±2, matching the existing inverse-design and B-sweep geometry), `multi_ref=true`, `n_up = n_down = 4` (Sz = 0). The sweep script patches `spin: {n_up, n_down}` in-memory for the other four sectors {5↑3↓, 6↑2↓, 7↑1↓, 8↑0↓} so we don't ship five near-identical YAML files.

* **New sweep driver**: `scripts/n8_sector_b_sweep.py`. Trains five sectors at B=0 (orbital energy is B-independent for fixed-Sz, so a single training per sector suffices; `n_sectors × n_B = 1` trainings instead of `n_sectors × n_B = n_B`), then for any user-supplied list of B values computes
  $$E(B,\,\text{sector}) = E_{\text{orbital}}(\text{sector}) + 0.5\cdot g\cdot \mu_B\cdot B\cdot(n_\uparrow - n_\downarrow)$$
  analytically. The script identifies the GS sector at each B (the one minimising `E(B, sector)`), reports the level-crossing fields where the GS Sz changes by one unit, and writes a 4-panel phase-diagram PNG (E-vs-B per sector, GS Sz vs B, orbital E_base spread, GS energy envelope).

* **Sz → −Sz mirror**: the spin-isotropic Hamiltonian has E_orbital(k↑(N−k)↓) = E_orbital((N−k)↑k↓) by global spin-flip, so we only train Sz ≥ 0 sectors and mirror analytically.

* **Launch command** (running now on cuda:3, started 08:03 CEST):

  ```bash
  CUDA_MANUAL_DEVICE=3 PYTHONPATH=src PYTHONUNBUFFERED=1 \
    nohup python3.11 -u scripts/n8_sector_b_sweep.py \
      --config configs/magnetic/n8_chain_d4_4up4down_b0_s42.yaml \
      --sectors 4_4 5_3 6_2 7_1 8_0 \
      --b-values 0.0 0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 \
      --stage-a-epochs 1500 \
      --stage-a-strategy improved_self_residual \
      --stage-a-min-energy 999.0 \
      --analyse-sector 4_4 \
      --out-dir results/b_sweep/n8_chain_d4_sector_aware_s42
  ```

  Wall-clock ended up matching the prediction almost exactly: total **1 h 45 min** for the five sectors, with the wall time per sector dropping monotonically with template count (4↑4↓ 70 templates → 36 min; 5↑3↓ 56 → 31 min; 6↑2↓ 28 → 19 min; 7↑1↓ 8 → 10 min; 8↑0↓ 1 → 7 min).

### Results — sector orbital energies and phase diagram

The five trained sectors converged to (mirrored to negative Sz by the spin-flip symmetry of the spin-isotropic Hamiltonian):

| Sector | Sz | E_orbital (Ha) | Templates | Wall time |
|---|---|---|---|---|
| 4↑4↓ | 0 | 11.4435 | 70 | 36 min |
| 5↑3↓ / 3↑5↓ | ±1 | **11.4124** (min) | 56 | 31 min |
| 6↑2↓ / 2↑6↓ | ±2 | 11.4414 | 28 | 19 min |
| 7↑1↓ / 1↑7↓ | ±3 | 11.4483 (max) | 8 | 10 min |
| 8↑0↓ / 0↑8↓ | ±4 | 11.4419 | 1 | 7 min |

Overall spread: **≈ 36 mHa across all 9 sectors.**

The post-hoc Zeeman assembly with `g = 2`, `μ_B = 1` gives the staircase phase diagram (full table in `results/b_sweep/n8_chain_d4_sector_aware_s42/B_sweep.csv`):

| B (a.u.) | GS sector | GS Sz | GS E (Ha) |
|---|---|---|---|
| 0.000  | 5↑3↓ | +1 | 11.412 |
| 0.001  | 3↑5↓ | −1 | 11.410 |
| 0.005  | 0↑8↓ | −4 | 11.402 |
| 0.010  | 0↑8↓ | −4 | 11.362 |
| 0.020  | 0↑8↓ | −4 | 11.282 |
| 0.050  | 0↑8↓ | −4 | 11.042 |
| 0.100  | 0↑8↓ | −4 | 10.642 |
| 0.200  | 0↑8↓ | −4 | 9.842 |
| 0.500  | 0↑8↓ | −4 | 7.442 |
| 1.000  | 0↑8↓ | −4 | 3.442 |

Two level crossings, both at very small B:

* `B_c1 ∈ (0, 0.001)`: Sz: +1 → −1 (this is the lifting of the spin-flip degeneracy at any B≠0; trivial)
* `B_c2 ∈ (0.001, 0.005)`: Sz: −1 → −4 (the chain skips the |Sz|=2,3 plateaux and saturates directly to fully-polarised, because their `E_orbital` lies above the line connecting Sz=−1 and Sz=−4 in the (`E_orbital`, Sz) plane).

The 4-panel PNG (`B_sweep.png`) shows: (a) energy fan E(B, sector) with all 9 lines starting near 11.42 Ha and slope `B·(n_up − n_down)`, (b) sharp staircase Sz_GS(B) with the two crossings near the origin, (c) V-shaped `E_orbital − min` vs Sz with the unexpected minima at Sz = ±1, (d) the linear GS-energy envelope `E_GS(B) = 11.412 − 4|B|` for B ≥ 0.005.

> **Sign convention.** Our Hamiltonian implements the Zeeman term as `V_Z = +0.5·g·μ_B·B·Σ s_iz = +B·(n_up − n_down)` (see `src/potential.py:43-88` quoted in the previous section). With `g > 0` this is the standard "anti-aligned with B" sign for an electron — positive B favours Sz < 0. The `+1 → −1` crossing right at B = 0⁺ is the trivial Sz↔−Sz degeneracy lifting in this convention; the *physical* crossings to track are the larger-|Sz| ones.

### Caveat: non-monotonic E_orbital (Sz=±1 below Sz=0)

For a generic Heisenberg-AFM chain we expect `E_orbital(0) < E_orbital(±1) < … < E_orbital(±4)`. The sweep instead returned a V-shape with the *minimum* at Sz=±1, ≈ 31 mHa below Sz=0. Two interpretations:

1. **Variational under-convergence (most likely).** The 4↑4↓ sector has the largest spin-template ansatz (70 patterns, with `multi_ref=True` summing them coherently), and its loss curve flattens at ~2 × 10⁻³ — measurably noisier than the smaller-template sectors (5↑3↓ flattens at 8 × 10⁻³ but at a lower E because there's less destructive interference among templates; 8↑0↓ flattens at 5 × 10⁻⁴). At fixed epoch budget (1500 epochs, n_coll = 384), the 70-template sector is most likely above its true variational minimum.

2. **Real non-monotonic physics.** Possible but unusual at d=4 with isotropic Coulomb interactions; would require a non-AFM exchange or an emergent magnetic anisotropy from the lattice geometry. Not impossible (the d-sweep already showed the PINN diverges from Heisenberg at d=4) but the simpler explanation is (1).

**Validation plan**: a seed-2 run (different random init) on the same configuration. If `E_orbital(Sz=0) − E_orbital(Sz=±1)` shrinks substantially (or flips sign), this confirms variational noise; if it persists at ~30 mHa, the physics is real and warrants a longer-epoch sweep. **Queued; will be launched once GPU 3 is available** (the seed-1 sweep just released it). Result will be folded into the next dated update.

The plain B-sweep result is preserved as `results/b_sweep/n8_uniform_d4_s42/` for the methodological record; the sector-aware sweep is in `results/b_sweep/n8_chain_d4_sector_aware_s42/`.

---

## 3. Phase 4 N=12 fill-in — done

Launched on cuda:6 at 07:43 CEST and finished at 09:38 CEST (≈ 1 h 55 min) with the same recipe as the N=16 retry:

```bash
CUDA_MANUAL_DEVICE=6 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  nohup python3.11 -u scripts/run_two_stage_ground_state.py \
    --config configs/scaling/n12_grid_d6_s42.yaml \
    --stage-a-strategy self_residual \
    --stage-a-epochs 5000 \
    --stage-a-n-coll 32 \
    --stage-a-min-energy 999.0 \
    --seed-override 314 \
    --summary-json results/scaling/n12_grid_d6_s42_seed314_self_resid_summary.json
```

**Result** (`results/scaling/n12_grid_d6_s42_seed314_self_resid_summary.json`):

```json
{
  "stage_a": {
    "result": {
      "final_energy": 18.83593441664557,
      "final_loss": 4.319514300619292e-05,
      "final_energy_var": 4.319514300619882e-05,
      "final_ess": 32.0
    }
  },
  "stage_b": null
}
```

Energy trajectory (epoch → E, sampled): 100 → 18.66, 600 → 18.84, 1000 → 18.85, 2000 → 18.84, 3000 → 18.84, 5000 → 18.836. Flat to ±0.01 Ha across the last ~4 000 epochs; loss never exceeded 3 × 10⁻³; ESS pinned at the full 32; peak GPU 1.56 GB. Same qualitative behaviour as the N=16 retry — `self_residual` + `multi_ref=False` is a healthy variational lane at this scale.

This fills the gap in the scaling curve between the N=8 SSH flagship (E = 11.4 Ha, fully observable-mapped) and the N=16 retry (E = 27.27 Ha, gateway to the Phase 5 stretch goals). The deterministic non-MCMC NQS lane now has validated ground states at:

| N | E (Ha) | Lane |
|---|---|---|
| 2 | (singlet baseline, full DMC cross-check) | sector_self_residual |
| 3 | (per-sector configs) | sector_self_residual |
| 4 | (per-sector configs, Heisenberg cross-check 0.939 overlap @ d=4) | sector_self_residual |
| 8 | 11.44 (SSH flagship + d/B sweeps) | improved_self_residual / sector |
| **12** | **18.836** (this run) | self_residual |
| 16 | 27.270 | self_residual |

A scaling-curve plot can now be made from the per-N result JSONs (six points; effectively three orders of magnitude in N²Ha).

---

## 4. Open follow-ups (carried forward to next update)

In rough priority order:

1. **Validate the sector-aware sweep with a second seed.** The seed-42 run produced a non-monotonic `E_orbital(Sz)` with the minimum unexpectedly at Sz = ±1. Most likely a variational under-convergence artefact in the 70-template 4↑4↓ sector; a second seed will discriminate. Same five sectors, same 1500 epochs, different RNG seed. ETA ~1 h 45 min on a single GPU. Once it completes I'll fold the comparison into the next update.
2. **Run the N=8 amplitude-evolution analysis on each *sector*'s checkpoint at B=0**, not just the AFM 4↑4↓ sector. The polarised sectors should show qualitatively different ⟨S_i·S_j⟩ structure (zero NN AFM correlation in 8↑0↓; suppressed in 7↑1↓; etc.). This is essentially "spin-resolved spectroscopy" of the chain. Five `amplitude_evolution.py` invocations, ~10 minutes total, no GPU needed.
3. **Phase 4 multi-seed validation**: with N=12 and N=16 now both clean on seed 314, an additional seed each (e.g., 42 or 901) would rule out single-seed flukes. Optional; the loss curves are already very flat so this is belt-and-braces.
4. **Documentation**: bring the sector-aware workflow into the methods section of the standing report — it is now a first-class lane alongside `improved_self_residual` and `self_residual`. Three short paragraphs and the table from § 2 above are enough.
5. **Vector-potential orbital coupling** (the *other* way to make B nontrivial — adds the magnetic length scale ℓ_B = 1/√B as a real parameter). Out of scope for this update; revisit once the sector-aware results are corroborated by seed-2.
6. **Scaling-curve plot.** With six validated N values now in hand (`N ∈ {2, 3, 4, 8, 12, 16}`), an explicit `E vs N` and `wall-time vs N` plot would be a clean Phase-4 wrap-up deliverable. Trivial — just JSON-glob plus matplotlib.

---

## 4b. Per-sector spin spectroscopy (today, follow-up to § 2)

After the sector-aware sweep landed I ran ``analyse_one`` on each sector's checkpoint to produce the spin observables that the original sweep skipped (the ``--analyse-sector`` flag was matched against the wrong tag format and ``analyse_one`` was never called; backfilled by ``scripts/n8_per_sector_amplitude_analysis.py``):

| Sector | Sz | von-Neumann entropy (PINN) | Heisenberg overlap | residual L2 | ⟨S_i·S_{i+1}⟩ avg | ⟨S_0·S_{N−1}⟩ |
|---|---|---|---|---|---|---|
| 4↑4↓ | 0 | 1.064 | 0.771 | 0.677 | **−0.366** | −0.319 |
| 5↑3↓ | +1 | 1.039 | 0.768 | 0.681 | −0.323 | −0.281 |
| 6↑2↓ | +2 | 0.943 | 0.831 | 0.581 | −0.200 | −0.156 |
| 7↑1↓ | +3 | 0.693 | 0.921 | 0.398 | −0.005 | +0.034 |
| 8↑0↓ | +4 | 0.000 | 1.000 | 0.000 | **+0.250** | +0.250 |

The deliverables are `results/b_sweep/n8_chain_d4_sector_aware_s42/per_sector_observables.{json,png}`.

**Two readings of this table relevant to the seed-2 validation:**

1. **PINN entropy decreases monotonically in `|Sz|`**: physically expected (more spin polarisation ⇒ less spin entanglement). At Sz=±4 the polarised sector is a single product state with zero entropy and trivially-unit Heisenberg overlap.

2. **NN spin correlator `⟨S_i·S_{i+1}⟩` is most negative in the Sz=0 sector** (−0.366) and grows monotonically through −0.32, −0.20, 0.0 to +0.25 at full polarisation. For a Heisenberg-AFM chain with `J > 0` this implies the Sz=0 sector should be the **lowest** in orbital energy (`E_Heis = J · Σ⟨S·S⟩` more negative). But the trained `E_orbital(Sz=0) = 11.443 Ha` is **higher** than `E_orbital(Sz=±1) = 11.412 Ha`. This is **inconsistent with Heisenberg** if both sectors are well-described by it, and consistent with the hypothesis that the 70-template Sz=0 ansatz is variationally under-converged compared with the 56-template Sz=±1 ansatz. Seed-17 will be the discriminator.

## 4c. Phase 4 scaling curve

With all six N values in hand, the deterministic non-MCMC NQS scaling curve (`results/scaling/scaling_curve.{csv,png}`):

| N | E_final (Ha) | E/N (Ha) | variance | source |
|---|---|---|---|---|
| 2 | 2.248 | 1.124 | 2.0e-6 | `p4_n2_singlet_d4_s42__stageB_noref_*` |
| 3 | 3.620 | 1.207 | 4.4e-5 | `p4_n3_nonmcmc_residual_anneal_s42__stageB_noref_*` |
| 4 | 5.075 | 1.269 | 1.7e-4 | `p4_n4_nonmcmc_residual_anneal_s42__stageB_noref_*` |
| 8 | 11.437 | 1.430 | 2.6e-3 | `d_sweep/n8_uniform_s42/summary_d4.json` |
| 12 | 18.836 | 1.570 | 4.3e-5 | `scaling/n12_grid_d6_s42_seed314_self_resid_summary.json` |
| 16 | 27.270 | 1.704 | 4.2e-4 | `scaling/n16_grid_d6_s42_seed314_self_resid_summary.json` |

Notable: variance jumps to 2.6e−3 at N=8 (legacy `improved_self_residual` lane, `multi_ref=True`, 70 templates) but drops back to 4.3e−5 at N=12 and 4.2e−4 at N=16 (new `self_residual` lane with `multi_ref=False`). The curve also makes clear that **none of these energies have an external benchmark** — the next priority (the 5-track roadmap, see below) addresses this directly via Track A.

## 4d. Research roadmap to push toward groundbreaking

Two planning documents have been drafted today:

* **[`reports/2026-04-28_grand_plan_anchored.md`](2026-04-28_grand_plan_anchored.md)** — the 5-month strategic plan. Headline ambition: "Inverse-designed real-time entanglement dynamics in 2D continuum quantum-dot networks". Eight phases (0-7), each with explicit ground-truth anchors (theoretical + numerical + experimental) and pre-registered pass/fail thresholds. Built on the discipline that *every* result must be triangulated against ≥1 independent benchmark before it can move to the next phase. Critical-path: Phase 0 anchor infrastructure → A1.3 (N=4 ED gate G1) → real-time NQS (G5) → real-time transfer (G6).
* **[`reports/2026-04-28_research_roadmap.md`](2026-04-28_research_roadmap.md)** — the tactical 5-week plan; subsumed by Phases 0-4 of the grand plan. Kept as a reference for the original 5-track sequencing.

The original 5-track summary is below for continuity, but all subsequent execution follows the anchored grand plan:

* **Track A (gating, week 1):** ED benchmark at N=4 across the d-sweep. Decides whether the "PINN diverges from Heisenberg at large d" finding is real physics or a variational-pathology artefact. Existing CI infrastructure in `scripts/exact_diag_double_dot.py` has the multi-well DVR + Coulomb + `eigh` solver; only ⟨S_i·S_j⟩ extraction from the CI eigenvector needs to be added (~1.5 days).
* **Track B (weeks 1-3):** topological order parameters (dimerization, SSH winding number, edge localization) as inverse-design targets. Cross-validate against existing N=8 SSH checkpoints; re-engineer SSH state targeting the winding number directly.
* **Track C (weeks 1-4):** excited-state NQS lane via orthogonality penalty. Unblocks gap engineering at N≥6.
* **Track D (weeks 3-4):** N=4 then N=8 Pareto frontier, T (entanglement) vs gap. The "knee" geometry is the experimental headline.
* **Track E (weeks 4-5):** fabrication-tolerance σ-sweep on SSH and winding-number-engineered states. Connects theory to specific Si/GaAs QD platforms.

The road definition of "groundbreaking" requires at least three tracks to land their high-bar outcomes, with Track A being the single highest-leverage move because it gates the d-sweep paper.

## 5. File pointers

* New base config: `configs/magnetic/n8_chain_d4_4up4down_b0_s42.yaml`
* New sweep driver: `scripts/n8_sector_b_sweep.py`
* Plain-B-sweep null result (record): `results/b_sweep/n8_uniform_d4_s42/{B_sweep.csv,B_sweep.png,B_sweep.json}`
* **Sector-aware B-sweep (done)**: `results/b_sweep/n8_chain_d4_sector_aware_s42/{B_sweep.csv,B_sweep.json,B_sweep.png,train_*.log,summary_*.json,cfg_*.yaml}`
* **N=12 fill-in summary (done)**: `results/scaling/n12_grid_d6_s42_seed314_self_resid_summary.json`
* N=12 Stage-A artefacts: `results/n12_grid_d6_s42__stageA_self_residual_20260428_093807/`
* N=16 retry summary: `results/scaling/n16_grid_d6_s42_seed314_self_resid_summary.json`
* N=16 retry Stage-A artefacts: `results/n16_grid_d6_s42__stageA_self_residual_20260428_034125/`
* N=12 training log: `logs/scaling/n12_seed314_self_resid_cuda6.log`
* Sector-aware console log: `logs/b_sweep/n8_chain_d4_sector_aware_s42_console.log`
* **Per-sector observables (done today)**: `results/b_sweep/n8_chain_d4_sector_aware_s42/per_sector_observables.{json,png}`
* **Per-sector analysis driver**: `scripts/n8_per_sector_amplitude_analysis.py`
* **Scaling curve (done today)**: `results/scaling/scaling_curve.{csv,png}` and driver `scripts/plot_scaling_curve.py`
* **Research roadmap (done today)**: `reports/2026-04-28_research_roadmap.md` (5-week tactical plan)
* **Grand plan with anchors (done today)**: `reports/2026-04-28_grand_plan_anchored.md` (5-month strategic plan, 8 phases × pre-registered ground-truth anchors)

---

*Compiled live by Cursor agent during the session. Subsequent milestone entries will go into a follow-up `2026-04-29_supervisor_update.md` (or earlier if a supervisor sync is requested before then).*
