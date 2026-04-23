# N=2 No-Reference Reproduction Review (April 23, 2026)

## Scope

This report reviews the current attempt to reproduce the previously successful `E_ref`-guided `N=2` double-dot results without using `residual_target_energy`.

The concrete question is narrower than "does no-ref converge":

1. Can the no-reference lane recover the old high-quality `N=2` ground-state branch?
2. Can it recover the old entanglement behavior?
3. Can it recover the downstream magnetic-quench behavior?
4. Which implementation changes were necessary, and which ones actually mattered physically?

The report covers the code in:

- [src/run_ground_state.py](../src/run_ground_state.py)
- [scripts/run_two_stage_ground_state.py](../scripts/run_two_stage_ground_state.py)
- [src/imaginary_time_pinn.py](../src/imaginary_time_pinn.py)
- [scripts/characterize_quench.py](../scripts/characterize_quench.py)
- [scripts/run_magnetic_reference_sweep.py](../scripts/run_magnetic_reference_sweep.py)
- [scripts/run_one_per_well_multi_magnetic_reference.py](../scripts/run_one_per_well_multi_magnetic_reference.py)
- [scripts/launch_noref_robustness_campaign.sh](../scripts/launch_noref_robustness_campaign.sh)
- [scripts/run_noref_reproduction_suite.py](../scripts/run_noref_reproduction_suite.py)

and the main result artifacts:

- [results/diag_sweeps/n2_seed42__two_stage_summary_20260423_141148.json](./diag_sweeps/n2_seed42__two_stage_summary_20260423_141148.json)
- [results/diag_sweeps/n2_seed314__two_stage_summary_20260423_141148.json](./diag_sweeps/n2_seed314__two_stage_summary_20260423_141148.json)
- [results/diag_sweeps/n2_seed901__two_stage_summary_20260423_141148.json](./diag_sweeps/n2_seed901__two_stage_summary_20260423_141148.json)
- [results/diag_sweeps/magnetic_reference_n2_matched_fix_20260423_170014.json](./diag_sweeps/magnetic_reference_n2_matched_fix_20260423_170014.json)
- [results/diag_sweeps/n2_seed42__two_stage_summary_singlet_noref_20260423_180423.json](./diag_sweeps/n2_seed42__two_stage_summary_singlet_noref_20260423_180423.json)
- [results/diag_sweeps/n2_seed42__stageA_singlet_ent_n28_20260423_180423.json](./diag_sweeps/n2_seed42__stageA_singlet_ent_n28_20260423_180423.json)
- [results/diag_sweeps/n2_seed42__stageB_singlet_ent_n28_20260423_180423.json](./diag_sweeps/n2_seed42__stageB_singlet_ent_n28_20260423_180423.json)
- [results/diag_sweeps/n2_seed42__stageA_singlet_quench_tau1_ent_20260423_180423.json](./diag_sweeps/n2_seed42__stageA_singlet_quench_tau1_ent_20260423_180423.json)
- [results/imag_time_pinn/pinn_quench_single_fast_B0p50_p4_n2_nonmcmc_residual_anneal_s42__stageA_singlet_self_residual_20260423_181245.json](./imag_time_pinn/pinn_quench_single_fast_B0p50_p4_n2_nonmcmc_residual_anneal_s42__stageA_singlet_self_residual_20260423_181245.json)

## Executive Summary

- The original April 22, 2026 no-reference `N=2` lane failed outright. Stage A importance sampling collapsed and no seed advanced.
- The first April 23, 2026 fix removed `E_ref` guidance and switched `N=2` to self-residual warm start. That solved robustness in the narrow optimizer sense: all three seeds passed Stage A and reached Stage B with `ESS=512`.
- That first fix did not reproduce the old physics. Ground-state entanglement was moderate and seed-sensitive, and the magnetic quench stayed far from the matched CI target. The post-quench state for seed `42` even collapsed into a mostly `RR` sector.
- The decisive missing ingredient was not `E_ref` itself. It was the old `N=2` singlet permanent ansatz plus the broader stratified sampler recipe used by the successful `E_ref` lane.
- After transplanting that old recipe into a new `singlet_self_residual` no-reference Stage A, seed `42` recovered the target entanglement regime. Stage A and Stage B both produce nearly ideal `LR/RL` weight with dot entropy about `0.694`.
- The downstream magnetic quench from that new no-reference singlet state now lands very close to the matched CI post-quench entanglement target: CI `S = 0.693147`, new no-ref seed `42` `S = 0.687834`, dot-label negativity `0.490225` versus CI `0.5`.
- Exact parity is not yet fully established. We still need the same singlet no-ref path rerun on seeds `314` and `901`, and the quench gap extraction remains inconsistent across estimators.

## Implementation Review

### 1. Stage continuation support

The two-stage workflow depends on being able to start Stage B from a saved Stage A checkpoint. That is now supported directly in [src/run_ground_state.py](../src/run_ground_state.py):

- `_resolve_init_from_model_path()` resolves `init_from` as either a checkpoint path or a result directory.
- `_maybe_load_initial_state()` loads `model.pt` before training and records missing or unexpected keys.
- `run_training_from_config()` now attaches initialization metadata to the result payload.

This was necessary infrastructure. Without it, Stage B was not a real continuation lane.

### 2. Two-stage no-reference driver

[scripts/run_two_stage_ground_state.py](../scripts/run_two_stage_ground_state.py) now encodes the policy decision explicitly:

- `guided`: use the warm start already present in the source config.
- `self_residual`: remove `residual_target_energy` and run pure residual minimization.
- `singlet_self_residual`: same no-reference objective, but with the old successful `N=2` singlet recipe transplanted in.

The important physics choice is `_apply_legacy_n2_singlet_recipe()`:

- `architecture.singlet = true`
- `architecture.use_backflow = false`
- stratified sampler kept
- wider sampler widths:
  - `sigma_center = 0.20`
  - `sigma_tails = 1.00`
  - `sigma_mixed_in = 0.25`
  - `sigma_mixed_out = 0.70`
  - `shell_radius = 1.20`
  - `shell_radius_sigma = 0.06`
  - `dimer_pairs = 1`
  - `dimer_eps_max = 0.06`

This matches the old successful `E_ref` singlet config at [results/p4_n2_singlet_d4_s42_20260415_154551/config.yaml](./p4_n2_singlet_d4_s42_20260415_154551/config.yaml). The only thing deliberately removed in the new path is `residual_target_energy`.

### 3. Quench reproducibility fixes

[src/imaginary_time_pinn.py](../src/imaginary_time_pinn.py) received three changes that matter for review quality:

- locked ground-state geometry is now applied before freezing `well_sep_initial` and `well_sep_final`
- quench outputs now include the source ground-state artifact name, so runs no longer overwrite each other
- PINN Adam uses `foreach=False`, which avoids the GPU optimizer-state crash seen in the PyTorch 2.1 float64 quench workload

These are methodological fixes, not physics improvements. They remove bookkeeping ambiguity and an implementation bug.

### 4. Parameter-matched CI reference

The magnetic reference scripts now default to the same Hamiltonian as the no-reference `N=2` lane:

- `kappa = 1.0`
- `epsilon = 0.01`

That change is in:

- [scripts/characterize_quench.py](../scripts/characterize_quench.py)
- [scripts/run_magnetic_reference_sweep.py](../scripts/run_magnetic_reference_sweep.py)
- [scripts/run_one_per_well_multi_magnetic_reference.py](../scripts/run_one_per_well_multi_magnetic_reference.py)

The launcher [scripts/launch_noref_robustness_campaign.sh](../scripts/launch_noref_robustness_campaign.sh) now uses those matched values for the CPU sidecar CI sweep.

## Result Review

### 1. Baseline failure on April 22, 2026

The last fully guided no-reference reproduction run on April 22, 2026 failed for `N=2`. All three seeds failed the Stage A gate and never reached Stage B. That was a real failure mode, not noise.

This is why the April 23 work split into two tasks:

1. make `N=2` numerically robust without `E_ref`
2. then make it physically match the old good lane

### 2. First no-reference stabilization: generic self-residual

The first successful April 23, 2026 fix changed `N=2` Stage A to pure self-residual warm start. The summaries are:

- [seed 42](./diag_sweeps/n2_seed42__two_stage_summary_20260423_141148.json)
- [seed 314](./diag_sweeps/n2_seed314__two_stage_summary_20260423_141148.json)
- [seed 901](./diag_sweeps/n2_seed901__two_stage_summary_20260423_141148.json)

All three seeds passed Stage A and all three reached Stage B with `ESS=512`.

| Seed | Stage B energy | FD energy | New virial rel. |
| --- | ---: | ---: | ---: |
| `42` | `2.235111` | `2.361491` | `20.48%` |
| `314` | `2.237783` | `2.345028` | `6.02%` |
| `901` | `2.237984` | `2.347861` | `4.60%` |

The corresponding audits are:

- [seed 42 components](./diag_sweeps/n2_seed42__components_20260423_141148.json)
- [seed 314 components](./diag_sweeps/n2_seed314__components_20260423_141148.json)
- [seed 901 components](./diag_sweeps/n2_seed901__components_20260423_141148.json)
- [seed 42 virial](./diag_sweeps/n2_seed42__virial_20260423_141148.json)
- [seed 314 virial](./diag_sweeps/n2_seed314__virial_20260423_141148.json)
- [seed 901 virial](./diag_sweeps/n2_seed901__virial_20260423_141148.json)

Interpretation:

- the catastrophic Stage A failure was fixed
- the optimizer now finds a stable branch for every seed
- but this branch is not the old high-entanglement singlet branch

Ground-state entanglement in that generic self-residual lane was still modest and seed-sensitive:

| Seed | Particle entropy `npts=24 -> 28` | Dot entropy `npts=24 -> 28` |
| --- | ---: | ---: |
| `42` | `0.119 -> 0.105` | `0.020 -> 0.018` |
| `314` | `0.222 -> 0.207` | `0.104 -> 0.100` |
| `901` | `0.185 -> 0.170` | `0.064 -> 0.061` |

So this was a robustness fix, not a parity fix.

### 3. Matched CI target for the magnetic quench

The matched exact-diagonalization sidecar is [results/diag_sweeps/magnetic_reference_n2_matched_fix_20260423_170014.json](./diag_sweeps/magnetic_reference_n2_matched_fix_20260423_170014.json).

For `d = 4`, `B_pre = 0.0`, `B_post = 0.5`, `kappa = 1.0`, `epsilon = 0.01`, it predicts:

- pre-quench dominant spin: `singlet`
- post-quench dominant spin: `triplet_m`
- post-quench gap: `0.345611`
- post-quench entropy: `0.693147`
- post-quench negativity: `0.5`

That is the correct target for the current no-reference reproduction question.

### 4. Generic self-residual no-reference quench did not match

The first no-reference quench runs were stable after the implementation fixes, but the physics was wrong.

The post-quench `tau = 1.0` entanglement files are:

- [seed 42](./diag_sweeps/n2_seed42__quench_tau1_ent_matchedfix_20260423_170014.json)
- [seed 314](./diag_sweeps/n2_seed314__quench_tau1_ent_matchedfix_20260423_170014.json)
- [seed 901](./diag_sweeps/n2_seed901__quench_tau1_ent_matchedfix_20260423_170014.json)

| Seed | Particle entropy | Dot entropy |
| --- | ---: | ---: |
| `42` | `0.170296` | `0.032725` |
| `314` | `0.270401` | `0.073031` |
| `901` | `0.243740` | `0.053765` |

These are nowhere near the matched CI post-quench target `0.693147`.

The worst example is seed `42`: [results/diag_sweeps/n2_seed42__quench_tau1_ent_matchedfix_20260423_170014.json](./diag_sweeps/n2_seed42__quench_tau1_ent_matchedfix_20260423_170014.json) shows sector probabilities

- `LL = 0.00123`
- `LR = 0.04425`
- `RL = 0.05202`
- `RR = 0.90250`

That is a strong localization into `RR`, not the old Bell-like response we were trying to recover.

The gap extraction was also internally inconsistent. For the three generic self-residual Stage B quench runs:

| Seed | `direct_gaps[0]` | `fit_best` | `fit_log_linear` |
| --- | ---: | ---: | ---: |
| `42` | `0.596705` | `0.065578` | `0.840078` |
| `314` | `0.652189` | `0.067356` | `0.820812` |
| `901` | `0.623081` | `0.068457` | `0.836938` |

That is not a trustworthy reproduction of the matched CI gap `0.345611`.

### 5. What the old good lane actually used

The old successful `E_ref` lane for `N=2` did not just differ by `residual_target_energy`. It also used a different inductive bias:

- dedicated singlet permanent ansatz
- no backflow
- broader stratified sampler geometry

That old recipe is explicit in [results/p4_n2_singlet_d4_s42_20260415_154551/config.yaml](./p4_n2_singlet_d4_s42_20260415_154551/config.yaml).

This matters because the old high-quality behavior was not recoverable by merely setting `residual_target_energy = null` inside the generic fixed-spin lane.

### 6. Singlet self-residual no-reference proof run

The new proof artifact is [results/diag_sweeps/n2_seed42__two_stage_summary_singlet_noref_20260423_180423.json](./diag_sweeps/n2_seed42__two_stage_summary_singlet_noref_20260423_180423.json).

Stage A:

- strategy: `singlet_self_residual`
- final energy: `2.234725`
- final variance: `6.86e-05`
- final ESS: `512`

Stage B:

- final energy: `2.249912`
- final variance: `4.69e-06`
- final ESS: `512`

The important result is the entanglement structure, not the raw energy.

Ground-state entanglement:

| Artifact | Particle entropy | Dot entropy | Dot-label negativity |
| --- | ---: | ---: | ---: |
| Stage A | `0.694177` | `0.693227` | `0.497723` |
| Stage B | `0.695797` | `0.694589` | `0.495416` |

Files:

- [Stage A entanglement](./diag_sweeps/n2_seed42__stageA_singlet_ent_n28_20260423_180423.json)
- [Stage B entanglement](./diag_sweeps/n2_seed42__stageB_singlet_ent_n28_20260423_180423.json)

Both artifacts are nearly ideal `LR/RL` states:

- Stage A sectors: `LL 0.001255`, `LR 0.498733`, `RL 0.498733`, `RR 0.001279`
- Stage B sectors: `LL 0.001034`, `LR 0.498969`, `RL 0.498969`, `RR 0.001028`

This is the main result of the campaign. The old good `N=2` structure can be recovered without `E_ref`, but only if the old singlet ansatz and sampler recipe are preserved.

### 7. Singlet self-residual magnetic quench

The new quench artifact is [results/imag_time_pinn/pinn_quench_single_fast_B0p50_p4_n2_nonmcmc_residual_anneal_s42__stageA_singlet_self_residual_20260423_181245.json](./imag_time_pinn/pinn_quench_single_fast_B0p50_p4_n2_nonmcmc_residual_anneal_s42__stageA_singlet_self_residual_20260423_181245.json).

The post-quench `tau = 1.0` entanglement is [results/diag_sweeps/n2_seed42__stageA_singlet_quench_tau1_ent_20260423_180423.json](./diag_sweeps/n2_seed42__stageA_singlet_quench_tau1_ent_20260423_180423.json):

- particle entropy: `0.687834`
- dot entropy: `0.685910`
- dot-label negativity: `0.490225`

Sector probabilities:

- `LL = 0.000124`
- `LR = 0.433189`
- `RL = 0.558071`
- `RR = 0.008616`

This is qualitatively and quantitatively close to the matched CI target:

| Observable | Matched CI target | New no-ref singlet seed `42` |
| --- | ---: | ---: |
| Post-quench entropy | `0.693147` | `0.687834` |
| Post-quench negativity | `0.500000` | `0.490225` |

This is the first strong evidence in this repo that the old magnetic-entanglement behavior can be recovered in a no-reference ground-state lane.

## What Is Reproduced vs Not Yet Reproduced

### Reproduced

- stable `N=2` Stage A and Stage B without `residual_target_energy`
- parameter-matched CI magnetic reference
- old singlet-like ground-state entanglement structure for at least one no-reference seed
- post-quench entanglement very close to the matched CI target for that same seed

### Not yet reproduced exactly

- three-seed parity for the new singlet no-reference lane
- a trustworthy quench gap that matches the CI value `0.345611`
- a direct post-quench spin-sector observable in the PINN workflow

The singlet proof run still has inconsistent gap estimators:

| Estimator | Gap |
| --- | ---: |
| `direct_gaps[0]` | `0.611181` |
| `fit_best` | `0.059625` |
| `fit_log_linear` | `0.812458` |
| matched CI | `0.345611` |

So the entanglement parity is strong, but the gap inference remains unresolved.

## Limitations

### 1. Single-seed proof on the singlet no-reference path

The strongest positive result currently exists for seed `42`. Seeds `314` and `901` still need to be rerun with `singlet_self_residual`.

### 2. Entanglement numbers are not always apples-to-apples

Some older singlet reports used different measurement settings, for example [results/diag_sweeps/singlet_entanglement_d4_lowdin_s42.json](./diag_sweeps/singlet_entanglement_d4_lowdin_s42.json) used `npts = 32` and `max_ho_shell = 2`. The new no-reference singlet checks here use `npts = 28` and `max_ho_shell = 1`.

That means exact numerical equality across all historic entanglement reports should not be demanded unless the measurement settings are also matched.

### 3. Gap extraction is the weakest part of the quench analysis

The different fit methods still disagree badly, even when the post-quench state looks physically right by entanglement. This is now the main unresolved technical issue in the magnetic reproduction stack.

## Implications

The main conclusion is clear:

`E_ref` was not the essential ingredient behind the old successful `N=2` magnetic-entanglement lane.

The essential ingredients were:

- the dedicated singlet permanent ansatz
- the old wider stratified sampler geometry
- a stable continuation path into the quench stack

Removing `E_ref` is viable if those inductive biases are preserved.

The failed generic no-reference branch and the successful singlet no-reference branch make that point directly. The difference between them is not the optimizer or the downstream analysis scripts. The difference is the ansatz class and the sampling geometry.

## Recommended Next Steps

1. Rerun seeds `314` and `901` with `stage_a_strategy=singlet_self_residual`.
2. For each new seed, repeat the same bundle:
   - two-stage summary
   - ground-state entanglement at `npts=24` and `28`
   - single-`B` quench from the Stage A singlet artifact
   - post-quench entanglement at `tau=1.0`
3. Add a direct post-quench spin-sector observable to the PINN workflow so the quench is not inferred only from gap fits and entanglement.
4. Revisit the quench gap estimator. The current ensemble fit is not physically persuasive when `fit_log_linear`, `fit_best`, and `direct_gaps` differ by an order of magnitude.
5. Once seeds `314` and `901` confirm the same behavior, promote `singlet_self_residual` from a special-case proof path to the default no-reference `N=2` reproduction lane.

## Bottom Line

As of April 23, 2026, the repo no longer supports the claim that "`N=2` no-reference cannot reproduce the old magnetic-entanglement results."

What the evidence supports is narrower and more precise:

- generic fixed-spin no-reference training does not reproduce them
- singlet no-reference training very likely can
- seed `42` already demonstrates near-parity in the entanglement observables that matter most
- full exact reproduction still requires the remaining singlet seed sweep and a cleaner quench-gap readout
