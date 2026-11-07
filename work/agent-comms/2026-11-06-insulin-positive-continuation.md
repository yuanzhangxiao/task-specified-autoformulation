# Orion: warm insulin continuation and an explicit concentration control

The prior candidate-15 revision improved training NMSE and meal-peak timing but
all three searches reached their numerical limits while still improving. Each
retained model also predicts some negative absolute insulin concentrations.
Convergence and physical nonnegativity are different questions: least squares
contains no guarantee that every prediction is nonnegative.

The next milestone pairs each saved endpoint with two new fitting tasks:
unconstrained warm continuation and an explicitly labelled I >= 0 control.
The control implements the physical interpretation of the public target
`I(t): insulin concentration [pmol L^-1]`. It does not claim that the existing
parameter bounds already express this condition. The admitted equations, real
signed feedback coefficient, shared parameter scopes and training-root scopes
are unchanged. No parameter sign is selected by Orion or imposed on Astra.

At fixed outer coefficients, exact affine prediction columns make the new inner
problem a convex QP in three shared forcing coefficients and 32 training roots.
The solver checks full-grid primal feasibility, bounds, dual nonnegativity,
stationarity and complementarity; a successful optimizer flag is not sufficient.
Fitting uses the original training samples. The constraints also include their
midpoints. Separate original-equation Radau replays at eight subdivisions per
interval check for missed negative insulin before retention. This is still a
sampled training-domain check, not an unseen-schedule or continuous-time proof.
Other output minima are reported, without expanding this pilot's constraints.

Each arm receives 1,800 new numerical search seconds / 2,000 attempts, with
separate qualification, replay and local-probe budgets. New roots and durable
receipts preserve the old campaign. Negative lower-error historical fits remain
visible as references but cannot beat an admissible endpoint in the constrained
arm. The earlier positive nested fit is independently verified as a fallback.
Each search is seeded with its own qualified inner solution; this prevents an
infeasible raw source from suppressing feasible checkpoints in the shared search.

Bounded fixed-outer qualification on the real returned models gives:

| Historical label | Original NMSE | Positive conditional NMSE |
| --- | ---: | ---: |
| nested-zero | 0.163647 | 0.208916 |
| signed-negative | 0.167427 | 0.186696 |
| signed-positive | 0.187675 | 0.521631 |

All three pass full inner checks and the finer original-equation replay. These
are starting-point qualifications, not continued outer-search results. The third
positive conditional point is worse than the 0.261011 nested fallback, which is
retained unless a later admissible point improves it.

The endpoint audit tests small feasible outer directions at two scales after
re-solving the inner variables. It records observed descent and unavailable
probes; neither no sampled descent nor Powell termination certifies global
optimality. Useful probe endpoints also require replay and domain qualification.
No new construction revision is automatically triggered. The new local work is
bounded qualification/smoke testing; substantial searches are for Delta.

The runbook is `docs/PHASE_C_INSULIN_POSITIVE_CONTINUATION.md`. The source intake
is pinned to the reviewed three-task archive (plan d187408a2a6b3453810da22b4b0a023b41fda3eb5636da9266f25567c430f851).
The new compact projection has SHA256
34e3ff4a6696775fc16adffa82d8b0bfb4db6ce7aa26af6c229b109547701558.

Verification: 324 focused regression tests passed across the new QP/campaign,
existing signed fitting, meal profiling and charts, root-search termination,
checkpointing and report reconciliation. Ruff is clean for all changed code;
full-repository Ruff reports the same 37 pre-existing findings in analysis/claude.
The campaign uses training only, with no new LLM calls or validation/test access.
Prefix forecasting and the remaining 22-model campaign remain separate.

Bounded end-to-end smoke checks exercised both real primary arms, including
source/fallback verification, feasible profile seeding, search checkpointing,
finer replay, incomplete-audit retention, terminal resume and inspection.
One small outer probe in the positive arm reached 0.172692 with no negative I
on the finer grid; original/affine loss differed by less than 7e-13. This is not
an outer-convergence result and does not change the frozen Delta starts. The
partial audit now preserves observed descent even if later probes time out.
Phase wall/CPU totals include all dense checks and local probes; nested counters
have an explicit narrower scope. The final regression subset (135 tests) passes.
