# Session Log

Last session: [2026-04-10] — Virial Gap Corrective Sweep and Evaluation Protocol Diagnosis
See ARCHIVE.md for full history.

## Next session
**Project objective:** Produce publication-quality VMC ground-state energies and virial-validated wavefunctions for N=2 and N=4 double quantum dots with Coulomb interaction; hard gate is virial residual < 5% for N=4.
**Active plan file:** [plans/2026-04-09_virial-gap-investigation.md](plans/2026-04-09_virial-gap-investigation.md)
**Recommended starting point:** Run 2-seed confirmation for the corrected best variant (`p3fix wellpinn_fd`) under locked virial protocol, then compare against old baseline with identical evaluator settings.
**Open questions:**
- Is the apparent ~1% absolute virial gain of corrected `wellpinn_fd` over corrected base real across seeds or noise?
- After protocol lock, is remaining ~13%–15% virial gap primarily an architecture limit or an objective/evaluator limit?
**Unverified assumptions:** Single-seed corrected sweep is representative.
**Active workarounds:** Legacy checkpoint compatibility in diagnostics (`strict=False` with allowed missing `backflow.w_intra`/`backflow.w_inter`).
**Foundation status:** Sampling fairness and FD evaluator sign convention verified; comparison protocol now fixed (FD evaluator + MH steps=40 + warmup 20); multi-seed confirmation still missing.
**Context freshness:** fresh
**Contradiction flags:** Initial 38%–55% virial interpretation contradicted by protocol-aligned reevaluation (~12.7%–15.3%); treat earlier high numbers as evaluation-artifact diagnostics, not final evidence.

## Session metrics (latest)
**Steps completed:** 3 of 3 planned (structural fix, 8-run sweep, protocol-aligned diagnosis)
**Material deviations:** 1 (added corrective subphase before Phase 4 CTNN comparison)
**Evaluation gates triggered:** 2 (phase gate and diagnosis gate; verdict: iterate)
**Unresolved uncertainties:** 2
