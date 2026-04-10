# Journal

Research journal. Each entry documents an experiment, a significant result, or a meaningful shift in understanding. Written as if read by a technically capable person who has not been following the project.

Entries older than 8 are compressed into ## Earlier Experiments by session-close. Preserve conclusions, discard step-by-step detail.

---

## Format

```
### [YYYY-MM-DD] — <experiment title>
**Motivation:** <what question were we trying to answer>
**Method:** <what was done — concisely but precisely>
**Results:** <numbers, with units always>
**What the numbers actually mean:** <interpretation separate from the numbers>
**What we cannot explain:** <anomalies or uncertainties>
**Caveats:** <what might be wrong with this interpretation>
**What a skeptic would say:** <honest critique>
**Output reference:** results/YYYY-MM-DD_<n>/
**Next question:** <what this makes us want to investigate>
```

## Negative / Failed / Inconclusive format

```
### [YYYY-MM-DD] — NEGATIVE: <what was tried>
**Hypothesis tested:** <specific claim under test>
**Method:** <what was done>
**Expected result:** <what would have confirmed the hypothesis>
**Actual result:** <what actually happened>
**Why it failed:** <root cause, or best current understanding>
**What this rules out:** <directions this failure eliminates>
**What this does NOT rule out:** <what remains plausible>
**Severity:** dead-end | needs-rethink | minor-setback
**Lessons for future work:** <what to remember next time>
**Output reference:** results/YYYY-MM-DD_<n>/ or n/a
```

## Comparison format

Use this when 2+ experiments address the same question and a cross-run verdict is needed.

```
## Comparison: <question being answered>
Date: YYYY-MM-DD
Experiments compared: <entry refs>

| Dimension       | Experiment A | Experiment B | Experiment C |
|-----------------|--------------|--------------|--------------|
| Method          | <short>      | <short>      | <short>      |
| Key metric      | <value>      | <value>      | <value>      |
| Secondary metric| <value>      | <value>      | <value>      |
| Training cost   | <value>      | <value>      | <value>      |
| Failure modes   | <short>      | <short>      | <short>      |

**Winner and why:** <evidence-based verdict>
**What this does NOT settle:** <remaining uncertainty>
**What a skeptic would say:** <critique of the comparison itself>
**Recommended next experiment:** <next most informative step>
```

---

### [2026-04-10] — NEGATIVE: Corrected backflow/cusp plus FD-vs-autograd sweep did not close virial gap
**Hypothesis tested:** Fixing SD/backflow coupling and cusp-coordinate handling, plus trying autograd Laplacian, would materially reduce N=4 double-dot virial residual (target <10%, stretch <5%).
**Method:**
- Implemented structural fixes in wavefunction/correlator paths.
- Added Laplacian backend switch (`fd`/`autograd`) in training and virial diagnostics.
- Ran 8 parallel 6000-epoch trainings across GPUs 0–7: base, well-PINN, well-BF, both × FD/autograd training modes.
- Re-evaluated all runs with locked virial protocol (FD evaluator, MH steps=40, warmup batches=20).
**Expected result:** At least one corrected variant would push virial below 10% and show robust separation from baseline.
**Actual result:**
- Initial inconsistent evaluations showed ~38%–55% virial (later diagnosed as protocol artifact).
- Protocol-aligned re-evaluation produced ~12.73%–15.34% virial across all 8 runs.
- Best run: `wellpinn_fd` at ~12.73%; still above target.
**Why it failed:** Structural fix corrected a modeling inconsistency but did not remove the remaining virial bottleneck; additionally, inconsistent early evaluation obscured true behavior and created a false catastrophic-regression narrative.
**What this rules out:** "Simple SD/backflow target correction alone" as sufficient to close the virial gap.
**What this does NOT rule out:**
- modest but real gains around ~1% absolute virial from specific variants,
- deeper architecture/objective limitations,
- seed dependence of the corrected best variant.
**Severity:** needs-rethink
**Lessons for future work:**
- Never compare virial across runs without a fixed evaluator protocol.
- Treat low-MH quick diagnostics as triage only, not decision evidence.
- Require at least 2 seeds before promoting a variant decision.
**Output reference:** [results/p3fix_n4_dd_base_fd_s901_20260410_085510](results/p3fix_n4_dd_base_fd_s901_20260410_085510), [results/p3fix_n4_dd_wellpinn_fd_s901_20260410_085327](results/p3fix_n4_dd_wellpinn_fd_s901_20260410_085327), [results/p3fix_n4_dd_wellbf_fd_s901_20260410_085309](results/p3fix_n4_dd_wellbf_fd_s901_20260410_085309), [results/p3fix_n4_dd_both_fd_s901_20260410_085457](results/p3fix_n4_dd_both_fd_s901_20260410_085457), [results/p3fix_n4_dd_base_autograd_s901_20260410_085549](results/p3fix_n4_dd_base_autograd_s901_20260410_085549), [results/p3fix_n4_dd_wellpinn_autograd_s901_20260410_084755](results/p3fix_n4_dd_wellpinn_autograd_s901_20260410_084755), [results/p3fix_n4_dd_wellbf_autograd_s901_20260410_085704](results/p3fix_n4_dd_wellbf_autograd_s901_20260410_085704), [results/p3fix_n4_dd_both_autograd_s901_20260410_085915](results/p3fix_n4_dd_both_autograd_s901_20260410_085915)
