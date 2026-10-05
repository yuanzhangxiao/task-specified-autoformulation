# Phase C M9: screening limits and assisted-collocation results

Review of `review-20261005-195412.tar.gz`, 2026-10-05 Pacific/Honolulu.
The [M9 design](PHASE_C_FITTING_SCREENING_ASSISTANCE.md) compared bounded RK45
and Radau training screens and tested collocation from previously successful,
training-fitted starts. It changed neither the benchmark nor production defaults.

**Basin remains solved, but the 20-second screening ceiling was too aggressive
for alien-device. All four eligible alien assisted runs stopped at the initial
Radau recheck, before either collocation stage. Their intended diagnostic is
therefore unanswered, not evidence that a good-start collocation solve failed.**

## Outcomes

All 48 entries are terminal: 32 have complete independent train/validation replay,
14 have no selected vector and hence no replay, and two assisted sources were
explicitly ineligible. Here `status: complete` at campaign level means all entries
are recorded, not that every fit succeeded. The 14 unavailable evaluations are
not independent-replay timeouts: fitting supplied no fully training-scored vector.

| Case and start regime | Entries | Complete independent evaluations | Prediction and coefficient recovery |
|---|---:|---:|---|
| Basin, generic | 18 | 18 | 18/18 |
| Basin, assisted | 6 | 6 | 6/6 selected endpoints; all retain the supplied fit |
| Alien, generic RK45 | 9 | 8 | 0/9 |
| Alien, generic Radau | 9 | 0 | Unavailable; 0/9 verified recoveries |
| Alien, assisted, eligible | 4 | 0 | Recheck timed out; collocation never started |
| Alien, assisted, ineligible | 2 | 0 | Explicitly skipped; no fitting |

Alien's eight evaluated endpoints also fail shared-initial recovery. Accuracy
requires both training and validation NMSE <=0.01, plus independent solver
agreement. Dynamic coefficient recovery requires every free coefficient within
1%; shared-initial recovery uses absolute error <=0.01. Basin's prediction gate
is 1e-6. These are the frozen gates, not thresholds selected after this run.

## What the screening experiment actually establishes

Alien has 16 training trajectories of 601 samples each. Every screening point
must finish the **whole training set**, including process startup, model setup
and writing results, within 20 seconds (15 seconds for the ordinary pre-NLP
phase). This deadline was not calibrated against this workload.

- Alien RK45: 35 complete screens and 39 timeouts across nine generic fits.
- Alien Radau: **75/75 generic screens time out**, as do all four assisted
  rechecks: **79/79 timeouts** in total.
- Basin: all 178 screens complete (65 RK45 and 113 Radau, including assistance).
- Among successful screens, median time outside the oracle's recorded residual
  evaluation is 5.00 seconds for basin RK45, 4.89 for basin Radau, and 5.63 for
  alien RK45. This includes startup, setup and other overhead; it is not a pure
  import-time measurement. Median residual evaluation time is respectively
  1.28, 3.35 and 8.70 seconds.

The short allowance can discard useful candidates. The exact M8 retained
reduced-collocation vectors recur in M9's screening pool for all three starts;
both M9 integrators time out on them. For starts 0 and 1, M8 had complete
independent replay, with validation NMSE 0.304323 and 0.877972. M9 RK45 instead
has no endpoint for start 0 and retains a worse point for start 1 (1.021712).
Start 2 already had unavailable independent replay in M8, so its timeout is not
evidence of losing an independently verified result.

Bounded process termination and durable journals worked: stalled points did not
consume an entire remaining fit budget, every attempt has its own directory,
and only complete training scores enter selection. But a technically working
timeout mechanism can make selection worse if its threshold is too restrictive.
Radau has not been shown numerically inferior here: it was not allowed enough
time to score even the previously successful alien solutions.

The killed point workers leave no completed oracle record or per-trajectory
progress. The archive therefore cannot locate their time within compilation,
particular trajectories, or integration. Nor does it prove stiffness caused all
timeouts. The next diagnostic needs that timing breakdown.

## Optimization remains a separate issue

Reduced alien collocation still records `Solve_Succeeded` for all three generic
starts, under both screeners, in 146--313 iterations. Retained nodal losses remain
approximately 0.225, 0.608 and 0.628. Native convergence thus remains distinct
from recovery of an accurate model. Dense solves finish only for start 2;
limited-memory shooting reaches its native-stage time limit in all six fits.
Wall-limited paths differ between arms, so they are not identical-iteration
comparisons even though their initial payloads differ only in screen integrator.

This run does not overturn the earlier M6/M7 rollout-only successes (two of three
alien starts); rollout-only was not rerun in M9. It also does not establish that
the 18-variable alien estimation problem is unidentifiable. The local sensitivity
gate again passes at rank 18, conditional on fixed internal couplings/shapes.
That gate is neither global uniqueness nor a guarantee of successful optimization.

## What the basin assisted runs tell us

All six eligible basin runs complete both stages. Fixing the two coefficients
leaves 1,440 node variables, and the node-only problem converges in one iteration.
Releasing them gives 1,442 variables and converges in 7--9 iterations. The dense
and reduced mesh policies produce the same grid on this small control; these are
not six independent hard-case demonstrations.

At integrated initial nodes, the discrete dynamic defect is about 0.001386.
Enforcing collocation equations reduces it to about 3--5e-16, while nodal NMSE
rises from about 1e-15--9e-12 to 8.24--8.26e-8. Releasing coefficients lowers
nodal NMSE slightly to 7.94e-8. Actual training rollout at that released endpoint
is approximately 2.90e-9, and maximum relative coefficient error is 0.000261
(0.0261%). Both remain excellent, but are worse than the supplied fit.

All six therefore retain the supplied fitted vector, as intended. Their reported
selected-endpoint success must not be credited as new recovery by collocation.
The fixed/released experiment works on this control and shows a small discrepancy
between the discretized objective and the true rollout objective; mesh refinement
has not yet been tested here to isolate its contribution. Across all basin
endpoints the worst validation NMSE is 4.914e-9.

## Recommended next step, not yet implemented

Keep all fitting methods. First repair the diagnostic's timing policy instead
of repeating the complete 48-entry campaign or increasing every NLP budget.

1. **Calibrate and replay screening on saved vectors.** Use the two already
   training-qualified M7 alien fits, all three generic starts, and the saved M9
   pools. Record setup time, each trajectory's progress and integration work.
   Start with a bounded timing pilot, then freeze a case-appropriate allowance
   with headroom before comparing the complete pools. Preserve an overall
   screening ceiling and distinguish timed-out points from poor fits. An
   escalating allowance or RK45-first/Radau-fallback policy can be tested; no
   incomplete rollout should earn an aggregate score. Reuse sealed matching
   training evaluations where applicable, and label all extra screening cost.
   Replaying saved pools needs no repeated NLP optimization. A better recovered
   endpoint would demonstrate a selection improvement, not a better optimizer.
2. **Then rerun the four eligible alien assisted tests.** Require the fitted-start
   recheck to complete under the calibrated policy before scheduling the large
   graphs. Keep the fixed-then-free experiment, independent replay, separate
   supplied-versus-released endpoint reporting, and all source-fitting costs.
   Its outcome can then distinguish discretization/local-solve limitations from
   difficulty reaching a good basin from generic starts.

Do not use true coefficients or validation scores to choose starts, calibrate
selection, or rank the pool. No production default is promoted by this review.

## Provenance and verification

- Archive SHA256: `fa4dbc85481e9a74268ad9d01053e427c29acaa62d85c37ea156c5c780f1490d`.
- Experiment commit: `91a70d2653454fa11a148104c5f330b51f3d982c`.
- Plan SHA256: `79e5e7d759813ee58146fde50096bedb9b26fe9124427c54b6b0be10f720b21e`.
- Input SHA256: `e99529ae95b4345bdb127e0d1dee85b111c6167afe97c29610b741881e156a39`.
- Code SHA256: `2ba87f1acf14c388bbecbe36121631251e9d15e9f5569fc65f37d320962661f5`.
- Delta jobs: prepare `22681259`, fit array `22681260`, report `22681261`.
  One CPU per fit, concurrency two, no GPU, LLM call or test-data access.
- All 177 sealed JSON records verify. All 46 launched backend/payload identities
  reconcile; the two ineligible entries have no backend. The full summary
  reproduces exactly in a separate directory. All 18 RK45/Radau payload pairs
  differ only in screening policy, and ordinary common starts match M8 exactly.
- All 32 replay records and journals agree with reported scores. Maximum complete
  replay solver disagreement is 8.227e-6, below the 1e-4 gate. Coefficient errors
  recompute from selected vectors. Every screen matches its frozen request,
  training data, integrator and parameter digest. Selection always retains the
  lowest complete training score; no timeout was converted to a score.
- Local audit: ignored `artifacts/fitting-m9-review-analysis/audit.py` and
  `analysis.json`; extraction: `artifacts/fitting-m9-review-20261005`.
- Review verification: 10 focused pytest tests pass; the saved eight-arm M9
  smoke passes exact completed-result resume without further fitting. Repository
  Ruff still reports 37 pre-existing errors in untracked `analysis/claude/` files.
  No implementation files changed; the full suite was not repeated for this
  documentation-only review. No new hard-case fitting or integration was run.
