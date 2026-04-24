# Reference-Free Ground-State Confidence Audit (2026-04-22)

## Question

Do the new reference-free runs

- actually converge,
- converge to the ground state rather than an arbitrary eigenstate,
- show collapse or pathology,
- remain usable for entanglement and quench workflows,
- and change anything about the current singlet/triplet or magnetic limitations?

This report audits the `p6_*_nonmcmc_residual_noref_*` runs against the earlier trusted `p4_*_nonmcmc_residual_anneal_s42_*` lane.

## Executive Verdict

- Numerical convergence is strong for all 9 no-reference runs.
- Spatial collapse is not supported by the evidence. Occupancies remain one-per-well to within sampling noise for `N=3,4`, and nearly balanced for `N=2`.
- Pure variance minimization is not yet a guaranteed ground-state finder in this repo. The loss is `Var(E_L)` only, so it can converge to different eigenstate basins.
- `N=3` is the cleanest success case. All three no-ref seeds beat the old lane on independent FD energy and generalized virial.
- `N=4` is mixed. The no-ref seeds look physically reasonable and often improve virial, but the independent FD energy is slightly higher than the old lane.
- `N=2` is not production-safe yet. Two seeds look promising, one seed is clearly bad, and the entanglement spread is too large.
- The no-ref checkpoints are compatible with the entanglement and quench tooling, but they do not remove the current fixed-spin magnetic limitation. Uniform longitudinal `B` is still structurally trivial in the generalized ansatz.

## What Was Audited

Training stability:

- training-time final energy
- tail-window energy fluctuations
- final variance and ESS

Independent physics checks:

- finite-difference re-evaluated energy and component split via `scripts/eval_ground_state_components.py`
- generalized multiwell virial via `scripts/check_virial_multiwell.py`
- `N=2` particle and dot-projected entanglement via `scripts/measure_entanglement.py`

Supporting artifacts:

- [results/diag_sweeps/noref_components_audit_20260422.json](results/diag_sweeps/noref_components_audit_20260422.json)
- [results/diag_sweeps/noref_virial_audit_20260422.json](results/diag_sweeps/noref_virial_audit_20260422.json)
- [results/diag_sweeps/noref_n2_entanglement_20260422.json](results/diag_sweeps/noref_n2_entanglement_20260422.json)

## Important Method Note

The no-reference configs still serialize a `reference_energy.resolved` field, because the wavefunction setup path resolves `E_ref: auto` during model construction. That value is not used by the loss here.

What matters for training is:

- `loss_type: residual`
- `residual_objective: residual`
- `residual_target_energy: null`

In [src/training/vmc_colloc.py](../src/training/vmc_colloc.py), this makes `compute_eeff()` return the batch mean local energy `mu`, so the actual objective is

`mean[(E_L - mu)^2] = Var(E_L)`.

That is an eigenstate objective, not a guaranteed ground-state objective.

## 1. Training Convergence

All no-ref runs finished cleanly with fixed `ESS = 512` throughout.

| Run | Final train energy | Tail rel. sd (%) | Final variance |
| --- | ---: | ---: | ---: |
| `N2 s314` | `2.23185570` | `0.0085` | `3.80e-06` |
| `N2 s42`  | `2.22953481` | `0.0081` | `4.85e-06` |
| `N2 s901` | `2.22031117` | `0.0213` | `8.10e-05` |
| `N3 s314` | `3.61341204` | `0.0176` | `6.67e-05` |
| `N3 s42`  | `3.61769242` | `0.0177` | `4.93e-05` |
| `N3 s901` | `3.61775579` | `0.0260` | `7.76e-05` |
| `N4 s314` | `5.08366338` | `0.0166` | `7.84e-05` |
| `N4 s42`  | `5.08351049` | `0.0162` | `5.73e-05` |
| `N4 s901` | `5.08786397` | `0.0173` | `7.54e-05` |

Conclusion:

- Numerically, the optimizer is not wandering.
- The solution family is stable enough to audit physically.
- `N2 s901` is already visibly weaker than the other seeds even at the training-log level.

## 2. Independent Energy Audit

The crucial check is whether the lower training energy survives an independent FD re-evaluation.

### N=2

| Run | Train E | FD E | FD delta vs old FD | Generalized virial (%) | `V_ee` | Occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Old `p4 s42` | `2.25384805` | `2.35674388` | `0.00000000` | `23.15` | `0.34813` | `0.469 / 0.531` |
| No-ref `s314` | `2.23185570` | `2.35371519` | `-0.00302870` | `9.95` | `0.33233` | `0.463 / 0.537` |
| No-ref `s42` | `2.22953481` | `2.35030153` | `-0.00644236` | `13.09` | `0.33620` | `0.468 / 0.532` |
| No-ref `s901` | `2.22031117` | `2.42176158` | `+0.06501770` | `34.36` | `0.40540` | `0.450 / 0.550` |

Interpretation:

- `s314` and `s42` look physically better than the old lane by both FD energy and virial.
- `s901` is clearly a bad basin: much worse FD energy, much worse virial, and much lower entanglement.
- This means `N=2` no-ref is not robust across seeds. The method can find a good branch, but it can also fail badly.

### N=3

| Run | Train E | FD E | FD delta vs old FD | Generalized virial (%) | `V_ee` | Occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Old `p4 s42` | `3.63589549` | `3.64734974` | `0.00000000` | `26.76` | `0.64864` | `0.333 / 0.334 / 0.333` |
| No-ref `s314` | `3.61341204` | `3.63412333` | `-0.01322641` | `5.92` | `0.63592` | `0.333 / 0.333 / 0.333` |
| No-ref `s42` | `3.61769242` | `3.63272978` | `-0.01461995` | `7.05` | `0.63268` | `0.334 / 0.332 / 0.334` |
| No-ref `s901` | `3.61775579` | `3.63040170` | `-0.01694804` | `5.13` | `0.63462` | `0.333 / 0.334 / 0.333` |

Interpretation:

- This is the strongest case for the no-reference lane.
- All three seeds improve FD energy by `13` to `17` mHa relative to the old lane.
- All three seeds slash the generalized virial error from `26.76%` down to `5.13% - 7.05%`.
- Occupancy is perfectly one-per-well within noise.
- The lower energy is not coming from collapse; it comes from a better balance of lower kinetic plus lower interaction energy, with a slightly higher confinement cost.

### N=4

| Run | Train E | FD E | FD delta vs old FD | Generalized virial (%) | `V_ee` | Occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Old `p4 s42` | `5.10385570` | `5.09754131` | `0.00000000` | `10.89` | `1.09841` | `0.250 / 0.250 / 0.250 / 0.250` |
| No-ref `s314` | `5.08366338` | `5.10010100` | `+0.00255969` | `7.08` | `1.09598` | `0.250 / 0.250 / 0.250 / 0.250` |
| No-ref `s42` | `5.08351049` | `5.09973862` | `+0.00219730` | `3.59` | `1.09671` | `0.250 / 0.250 / 0.250 / 0.250` |
| No-ref `s901` | `5.08786397` | `5.09908875` | `+0.00154743` | `16.13` | `1.10380` | `0.250 / 0.250 / 0.250 / 0.250` |

Interpretation:

- All three no-ref seeds are extremely tight on FD energy.
- But they are all slightly higher than the old FD benchmark by `1.5` to `2.6` mHa.
- Two seeds improve virial substantially (`s314`, `s42`), while one seed is worse (`s901`).
- So `N=4` no-ref is physically plausible and often cleaner by virial, but it is not yet a clear energy win.

## 3. Train-Energy Trustworthiness

The training estimator is not equally trustworthy across all systems.

### Gap between final training energy and independent FD energy

| Run family | Typical gap `FD - train` |
| --- | ---: |
| Old `N=2` | `+0.1029` |
| No-ref `N=2` good seeds | `+0.1208` to `+0.1219` |
| No-ref `N=2` bad seed | `+0.2015` |
| Old `N=3` | `+0.0115` |
| No-ref `N=3` | `+0.0126` to `+0.0207` |
| Old `N=4` | `-0.0063` |
| No-ref `N=4` | `+0.0112` to `+0.0164` |

Takeaway:

- `N=2` training energy is not a reliable absolute metric here.
- `N=3` is much better behaved.
- `N=4` training energy looks systematically optimistic relative to FD re-evaluation.

This is why the production decision should be based on the FD/virial audit, not on `result.json.final_energy` alone.

## 4. Entanglement Audit for N=2

The no-ref checkpoints do work with the entanglement pipeline.

### `N=2` entanglement results

| Run | Particle entropy | Particle negativity | Dot entropy | Dot negativity | Dot-label negativity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old `p4 s42` | `0.388689` | `0.652106` | `0.345953` | `0.573069` | `0.000000` |
| No-ref `s314` | `0.356168` | `0.622733` | `0.315188` | `0.532582` | `0.000000` |
| No-ref `s42` | `0.413195` | `0.686142` | `0.367841` | `0.594235` | `0.000000` |
| No-ref `s901` | `0.159208` | `0.363869` | `0.127082` | `0.298177` | `0.000000` |

Interpretation:

- Entanglement is not broken by no-ref training.
- The good `N=2` seeds give entanglement near the old lane, with `s42` even slightly higher.
- The bad `N=2 s901` seed has much lower entanglement, matching its poor FD/virial behavior.
- Dot-label negativity remains zero in every case, so the no-ref lane is not producing a qualitatively new Bell-like dot sector here.

## 5. Did the System Collapse?

Current evidence says no.

Why:

- `N=3` and `N=4` occupancies are essentially exactly one electron per well.
- `N=2` occupancies remain balanced between left and right wells.
- The lower-energy no-ref states do not show the occupancy signature of all particles crowding one side.
- The energy shifts are better explained by a different correlation pattern:
  - lower `V_ee` for the good `N=2` seeds,
  - lower kinetic and lower `V_ee` for `N=3`,
  - near-canceling `T`, `V_conf`, `V_ee` shifts for `N=4`.

So the main risk is not collapse. The main risk is branch selection.

## 6. Ground-State Confidence by System

### N=2

Confidence: low to moderate

- Two seeds look genuinely better than the old lane.
- One seed is clearly wrong.
- Seed spread becomes much larger after independent FD evaluation.
- Conclusion: no-ref `N=2` can find the right kind of state, but the basin is not robust enough yet.

### N=3

Confidence: high

- All three seeds agree tightly in train energy, FD energy, occupancy, and virial.
- All three improve on the old lane in the same direction.
- This is the best evidence that the no-ref method can recover or even improve the ground-state branch.

### N=4

Confidence: moderate

- All three seeds are stable and physically reasonable.
- Two seeds improve virial, one worsens it.
- Independent FD energy is slightly above the old lane for every seed.
- Conclusion: this is not a collapse or catastrophe, but it is not yet a clean ground-state improvement either.

## 7. Magnetic / Quench / Singlet-Triplet Implications

### Quench compatibility

Yes, the no-ref ground states are compatible with the quench stack.

Why:

- the imaginary-time code loads a locked ground state from `ground_state_dir` by reading `model.pt + config.yaml`
- this is independent of whether training used an external reference energy

Relevant code:

- `src/imaginary_time_pinn.py`
- `scripts/measure_entanglement.py`

### Entanglement compatibility

Yes. The measurement code loads the same saved checkpoint format and works on the no-ref artifacts directly.

### Uniform magnetic-field response

No change in capability.

The current generalized ansatz still uses one fixed spin template, and the repo explicitly marks uniform longitudinal Zeeman coupling as a constant energy shift only under that ansatz.

Relevant code:

- `src/run_ground_state.py`
- `src/wavefunction.py`

So no-ref training does not suddenly make the existing generalized lane capable of nontrivial uniform-`B` singlet-triplet physics.

### Singlet/triplet support

Still limited.

- There is a dedicated `N=2`, two-well singlet permanent path in `GroundStateWF`.
- There is spin-sector scanning support.
- But full shared-spin superposition physics is still not present in the generalized training lane.

So:

- no-ref helps ground-state training independence,
- but it does not solve the ansatz-level spin-physics limitation.

## 8. Bottom-Line Production Decision

Use the current no-ref lane as follows:

- `N=3`: yes, as a strong proof-of-concept and likely production candidate
- `N=4`: maybe, but only after one more validation step
- `N=2`: no, not without a guard against bad basins
- `N=6+`: not yet with pure variance alone

## 9. Recommended Next Steps

1. Add a hybrid no-ref schedule.

   Use a short direct-energy stage first, then switch to pure variance minimization.  
   Goal: keep the CI-free workflow but bias the optimizer toward the ground-state basin.

2. Use `N=3` as the first promotion target.

   `N=3` is the best current evidence that no-ref training works physically, not just numerically.

3. Re-test `N=4` under the hybrid schedule.

   The target is simple:

   - keep the good virial behavior,
   - recover the slight FD-energy loss relative to the old lane,
   - eliminate the `s901`-style mixed-quality seed behavior.

4. Do a dedicated `N=2` basin-disambiguation campaign.

   Minimum package:

   - more seeds,
   - spin-sector scan where relevant,
   - `d=0` Kohn-style validation,
   - compare pure variance versus hybrid warm-start.

5. Only after that, scale to `N=6,8+`.

   The current pure-variance result is promising, but not yet trustworthy enough by itself for production scaling.

## Final Assessment

Reference-free training is real. It is not a fluke, it is not obviously collapsing, and it already looks very strong for `N=3`.

But the evidence also says something important and non-negotiable:

pure `Var(E_L)` minimization is a branch-selection problem, not a solved ground-state problem.

That means the right next move is not to abandon the direction. The right move is to keep the CI-free idea and add just enough ground-state bias to make the good branch reproducible across seeds and systems.
