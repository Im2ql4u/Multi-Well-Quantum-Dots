# Session Log

Last session: [2026-04-16] — Löwdin Sector Fix, Singlet Separation Sweep, and Entanglement Measurement
See ARCHIVE.md for full history.

## Next session
**Project objective:** Establish a trustworthy non-MCMC story for multi-well entanglement and ground-state optimization without confusing reference error, measurement error, and ansatz limitations.
**Active plan file:** [plans/2026-04-14_entanglement-measurement.md](plans/2026-04-14_entanglement-measurement.md)
**Recommended starting point:** Validate the Löwdin entanglement measurement on the CI reference wavefunction at d=8. Compute the 2×2 dot-label density matrix for the CI ground state in the Löwdin basis and check if negativity≈0.50. This will determine whether the 0.26–0.30 measured at d≥6 is a measurement artifact (basis truncation) or a genuine ansatz deficit.
**Open questions:**
- Does the CI reference wavefunction in the Löwdin basis give dot-label negativity≈0.50 (pure singlet)?
- If CI also gives neg<0.50, what fraction is explained by the ~5% of norm outside the Löwdin basis?
- Can using Löwdin-orthogonalized HO orbitals in the permanent (instead of raw HO orbitals) fix the d≤4 sector structure?
- Is the d=2 energy below CI (−0.65%) physically meaningful or a Hamiltonian mismatch?
**Unverified assumptions:**
- Löwdin basis with max_ho_shell=2 captures enough of the CI wavefunction to give a reliable negativity (not directly checked for CI reference).
- The permanent ansatz structural limitation at d≤4 is due to raw HO orbital overlap, not a training issue (inferred from theory, not ablated).
**Active workarounds:**
- singlet permanent ansatz uses raw (non-Löwdin) HO orbitals — breaks sector analysis at d≤4
- dot_label_negativity measurement not calibrated against a known reference state
**Foundation status:** Löwdin sector decomposition is verified (Gram error <3e-15). Sector structure at d≥6 correct. Energies within 0.05% of shared-CI. Entanglement quantity itself is unvalidated.
**Context freshness:** fresh
**Contradiction flags:** none — prior Voronoi-based entanglement numbers are superseded by Löwdin results; do not mix.

## Session metrics (latest)
**Steps completed:** 4 of 4 planned
**Material deviations:** 2 (exact_diag CLI bug → Python API workaround; nx scaling fix for d=20)
**Evaluation gates triggered:** 2 (Gram orthonormality; d≥6 sector structure)
**Unresolved uncertainties:** 3 (negativity underestimate, d≤4 permanent limitation, measurement not calibrated on CI)
