# Archive

Compressed session history. Maintained by session-close. When this file exceeds 10 entries, the oldest 5 are compressed into ## Older History.

Read the last 2–3 entries during session open. Read the full file only when reconstructing history.

---

## Format

```
## [YYYY-MM-DD] — <session title>

### Technical summary
[what was done, what was concluded, what is open]

### Session metrics
- Steps completed: <n of m planned>
- Material deviations: <count>
- Evaluation gates triggered: <count + verdict>
- Unresolved uncertainties: <count>

### Human reflection
**Understood this session:** ...
**Still unclear:** ...
**Skeptic's view:** ...
**Would do differently:** ...

---
```

---

## [2026-04-10] — Virial Gap Corrective Sweep and Evaluation Protocol Diagnosis

### Technical summary
- Central goal: diagnose why corrected backflow/cusp changes appeared to catastrophically worsen virial and determine whether the failure was architectural or evaluational.
- Accomplished: implemented structural fixes (SD evaluated on backflowed coordinates, cusp on physical coordinates), added FD/autograd Laplacian backend support, launched and completed 8-GPU factorial sweep, and re-evaluated all runs under consistent MH virial settings.
- Attempted but failed: initial virial reads (38%–55%) suggested severe regression; this did not hold after protocol-aligned re-evaluation and was traced to inconsistent evaluation settings (short MH warmup/steps and mismatched evaluator configuration).
- Decisions made: use Full Close evidence standard, lock virial comparison protocol for fair cross-run ranking (FD evaluator, MH steps=40, warmup batches=20) before drawing conclusions.
- Workarounds in place: evaluator compatibility shim for legacy checkpoints missing `backflow.w_intra`/`backflow.w_inter` keys.
- Unverified: seed-to-seed robustness for corrected best variant (only seed 901 evaluated in corrected sweep).
- Skeptic view: current evidence shows no progress toward <10%, and no claim of improvement is credible without fixed protocol plus 2-seed confirmation.
- Single most important carry-forward: the "catastrophic regression" diagnosis was mostly an evaluation artifact; corrected models remain in the ~13%–15% virial regime.
- Recommended next action: run a 2-seed confirm on corrected best variant with locked virial protocol, then decide whether to continue architecture exploration or pivot.

### Human reflection
**Understood this session:** not provided at close (session closed on user request).
**Still unclear:** not provided at close (session closed on user request).
**Skeptic's view:** not provided at close (session closed on user request).
**Would do differently:** not provided at close (session closed on user request).

---
