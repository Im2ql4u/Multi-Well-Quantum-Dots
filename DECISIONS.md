# Decisions

Permanent, append-only record of architectural and methodological decisions. Never delete or rewrite. If a decision was reversed, add a new entry explaining why.

Only write entries for genuine decisions. Not every small implementation choice. Quality over completeness.

---

## Format

```
### [YYYY-MM-DD] — <short title>
**Decision:** <what was chosen>
**Alternatives considered:** <what else was on the table>
**Reasoning:** <why this, not the alternatives>
**Constraints introduced:** <what this makes harder going forward>
**Confidence:** high / medium / low
```

---

### [2026-04-10] — Lock Virial Comparison Protocol
**Decision:** Use a fixed virial evaluation protocol for cross-run comparisons: FD Laplacian evaluator, MH sampler with `mh_steps=40`, `mh_warmup_batches=20`, and matched sample budgets.
**Alternatives considered:** Compare runs using mixed evaluator settings (different MH depth/warmup and/or autograd evaluator) based on whichever command was most recent.
**Reasoning:** Mixed evaluation settings produced large apparent regressions (up to ~55% virial) that disappeared under protocol-aligned re-evaluation (~13%–15%), so fair ranking requires fixed evaluation conditions.
**Constraints introduced:** Adds evaluation cost; quick low-MH diagnostics are no longer acceptable for decision-grade claims.
**Confidence:** high

## Negative Memory

### [2026-04-10] — FAILED: Treating early p3fix virial as catastrophic regression
**What:** Interpreted initial corrected-run virial outputs (~38%–55%) as direct evidence that the structural backflow/cusp fix broke the model.
**Why it failed:** Those numbers were produced under a different and weaker evaluation setup (short MH schedule / inconsistent evaluator settings), making them non-comparable to prior baselines.
**Evidence:** Re-evaluating old baseline and corrected runs with matched settings (`mh_steps=40`, warmup 20, FD evaluator) yielded ~13%–16% across runs, not ~38%–55%.
**What to do instead:** Freeze evaluation protocol before any cross-run claim and rerun diagnostics under identical settings.
**Severity:** needs-rethink

### [2026-04-10] — FAILED: Corrective architecture sweep to reduce virial below 10%
**What:** 8-run factorial sweep (FD/autograd training × base/well-PINN/well-BF/both) after correcting SD/backflow and cusp-coordinate handling.
**Why it failed:** None of the corrected variants achieved the plan gate; all remained in the ~12.7%–15.3% virial range under fair FD evaluation.
**Evidence:** `scripts/run_virial_check.py` summary on p3fix runs with locked protocol showed best case 12.73% (wellpinn_fd), worst 15.34% (both_autograd).
**What to do instead:** Perform 2-seed confirmation on best corrected variant and then revisit architecture assumptions or objective design.
**Severity:** minor-setback
