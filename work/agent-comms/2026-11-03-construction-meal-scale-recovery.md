# Meal-scale recovery implementation and qualification

3 November 2026. User authorized implementation after review of
`construction-meal-continuation-1`.

## Changes

- Corrected transformed gain units in `prefix_meal_coordinates.py`. Protocol 2
  freezes positive source g=k*q as the characteristic unit. At zero g it uses
  source k times the positive source q unit. Later rate reseeding preserves
  the unit; source values, choice and fallback are audited.
- Added opt-in overall wall allowance to `root_profile_search`, preserving its
  existing default. Invalid/nonfinite/undersized/oversized caps fail before work.
  Explicit caps never increase the numerical allowance or suppress strict
  incumbent writes. Simulated slow-persistence tests check both time limits.
- Added `construction-meal-scale-recovery-1`: five CPU tasks from the returned
  warm and four fixed-positive records. Each fixed task starts from its own
  verified trial; the shared warm 0.423867303091 incumbent remains independently
  retained. Numerical ties keep the old same-rate checkpoint. Both starts are
  independently replayed before fitting.
- Every task gets 1800 numerical seconds, 2000 attempts and 900 seconds explicit
  persistence headroom. Total stage reservations are 5160 seconds/task. Budgets
  remain nonrenewable; historical costs are labeled overlapping, not summed.
- Added isolated Delta launcher/root/config, pinned execution and scheduler
  receipts, cwd/spool-safe status/inspection, and source/report tamper checks.
  No zero-rate rerun, model revision, test data or LLM access.

## Validation

- 255 focused integration tests passed before the final two policy/retention
  regressions; final campaign recheck: 19 passed.
- Scoped Ruff and shell/CLI smoke pass. Full Ruff has the same 37 pre-existing
  unrelated `analysis/claude` findings.
- Production source prepare/verify/report/inspection smoke passes with five
  tasks and the correct per-rate seeds. No scheduler call made locally.
- All six historical terminal records still reconcile exactly.
- All **77** saved-model characteristic probes (five centers plus ± directions
  for every free shared coordinate) agree with independent Radau rollouts.
  All ten gain probes pass; gain units are 0.02543–0.03191. Largest absolute loss
  difference is 2.59e-9, within unchanged comparison tolerances. Root profiling
  occurs inside these bounded checks; there is no full shared-parameter refit.
- Evidence is local under
  `output/construction-meal-scale-recovery-preflight-20261103/`. An initial
  diagnostic script used additive rather than logarithmic timescale probes;
  it was stopped and corrected before the 77-probe report. The final report
  uses the search's actual coordinate rule; the preliminary log is not evidence
  of a production search failure.

## Next result interpretation

This milestone repairs optimizer geometry and exposes explicit I/O accounting.
It does not establish parameter identification, a profile minimum, biological
correctness or interventional prediction. Compare fixed-rate trial errors, not
retained global errors. Continue fitting before requesting equation revisions
based on these unfinished profiles. Joint shared-gain/root least-squares
profiling remains a possible later milestone, not part of this change.

See `docs/PHASE_C_CONSTRUCTION_MEAL_SCALE_RECOVERY.md` for exact upload, run,
status, inspect and download commands. The delivery is code-only; the completed
source campaign must remain on Delta. Keep all old roots and seals unchanged.
