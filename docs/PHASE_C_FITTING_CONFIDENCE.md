# M19: optimization reliability and coefficient uncertainty

Protocol `phase-c-fitting-confidence-1`. This is the next fitting diagnostic,
separate from construction, production defaults and nonlinear benchmark fitting.
It continues **all 24 M18 endpoints**: four known-skeleton linear controls, three
starts, and the original joint/profiled methods. No endpoint is filtered by its
prediction or coefficient error. There are no LLM calls, GPUs or test data.

## Motivation from M18

M18 completed all 24 tasks. All four reference replay/local sensitivity gates
passed; input, source, backend and process artifact digests were verified.

| Control | Joint: all coefficients within 1% | Profiled: within 1% |
|---|---:|---:|
| 3 states | 3/3 | 3/3 |
| 3 states, separated timescales | 2/3 | 3/3 |
| 6 states | 3/3 | 1/3 |
| 6 states, separated timescales | 0/3 | 0/3 |

Prediction passed the 1e-6 evaluation threshold on 11/12 joint and 12/12 profiled
runs. Initial-value recovery passed on 7/12 and 6/12. Profiling consumed 470 versus
703 residual evaluations and 1,273 versus 1,663 aggregate fitting seconds. These
costs are not time-to-equal-coefficient-accuracy comparisons.

Two ordinary six-state profiled fits reached essentially the same incorrect
region: validation NMSE 3.68e-7 but maximum coefficient error 323%. In a read-only
matrix-exponential check, changing just a4 from its reference 1.05 to fitted 4.438
raised validation NMSE to 1.30e-3; changing all parameters/initials to the fitted
vector reduced it to 3.68e-7. Other parameters substantially compensate for a4.

All six fast/slow six-state fits attained validation NMSE about 1e-12 to 1e-11
without recovering every coefficient. They stopped below their time ceilings.
Prediction precision, optimizer convergence and coefficient determination are
separate properties. Full sampled local sensitivity rank does not guarantee
accurate recovery at a particular noise/numerical tolerance.

## Frozen inputs and separation

The exporter requires the exact M18 inputs digest
`4e009057d899bb227303438fc4ba724d1fd25fd7b4b2552bb74d2d6dc29c380f`, checks all
24 selected vectors against their sealed source backends, and retains prior cost
and evaluation fields for comparison. The portable export is self-contained;
no prior campaign directory is needed to run it.

Only the request, training arrays, training-derived coordinates, generic starting
vector and saved incumbent enter the fitting/checking module. Explicit schemas
reject evaluator fields. Validation/reference values are read only after the new
backend is sealed, for retrospective prediction/coefficient/initial scoring.
No bounds, skeletons, observation schedules, or benchmark prompts are changed.

## Computation per endpoint

1. Recheck the incumbent using DOP853 and Radau, and compute the complete output
   sensitivity matrix with respect to **both coefficients and hidden initials**.
   SVD uses frozen parameter units and training output standard deviations.
   Save singular values, rank, column norms, condition number and weakest joint
   direction. This is local diagnostic evidence, not a global identifiability test.
2. Run three joint bounded least-squares searches: from the incumbent, its original
   generic start, and one bounded perturbation along the weakest joint direction.
   Each receives at most 120 seconds/160 residual calls, followed by an independent
   training check. Every nuisance initial value remains free. Report all outcomes.
3. Freeze a profile grid around the best independently checked training vector.
   For **each coefficient and initial parameter**, fix it at four offsets (both
   signs at 1% and 10%) and refit all remaining unknowns. Dynamic-parameter units
   are max(abs(fitted coefficient),0.01); initial-parameter units are the original
   training-derived optimizer units. Thus initial offsets are explicitly scaled
   distances, not claimed relative errors. Out-of-domain points are marked;
   they are never silently clipped or interpreted as data constraints.
4. Each profile point receives 25 seconds/40 residual calls, with 40% of time
   reserved for independent checking. Two nuisance starts (profile center and
   original generic vector) share the optimization allowance. Failed or limited
   searches are unresolved. A successfully verified low-loss alternative is
   evidence of ambiguity even if its optimizer did not converge.
5. A better profile candidate may improve the retained model. If profiles improve
   the reliability endpoint, give the best candidate one more free joint search
   with 120 seconds/160 calls and an independent check. Retain the best complete
   independently checked training vector, including the original incumbent.
   Label this additional-budget diagnostic-assisted recovery. It is not a matched
   algorithm comparison or a zero-cost confidence assessment.

For these noiseless controls only, optimization uses tighter integration
(rtol 1e-10, atol 1e-12), centered coordinates, Jacobian column scaling, and a
constant residual multiplier of 1e6. Multiplication changes numerical stopping
scales, not the least-squares minimizer. The numerical target is NMSE 1e-18;
TRF convergence, evaluation limits and wall limits remain separate stop reasons.
No coefficient-error target is available to optimization.

## What “confidence” means here

With normalized training residual r and loss L=mean(r^2), compute approximate
profiles P_j(v)=min_{theta_except_j,initials} L at the fixed grid points. These
minima are **feasible upper bounds** because nuisance optimization is local and
bounded. A high profile loss does not prove that no better solution exists.

This noiseless experiment uses loss-increase tolerances **1e-12, 1e-10, 1e-8**.
They are declared numerical scenarios, not estimated noise levels, likelihood
ratio cutoffs, or statistical confidence intervals. No “95%” label is reported.
A later noisy-data study would need an explicit likelihood, an appropriate
threshold and coverage calibration before such a label is justified.

For each tolerance, report verified alternative parameter values and a category:

- `weakly_constrained`: at least one materially different tested value gives a
  verified near-equivalent fit. This is constructive evidence of ambiguity.
- `numerically_unresolved`: independent solver disagreement is too large for the
  requested tolerance. Loss disagreement must be <= tolerance/10, and maximum
  normalized prediction disagreement <= sqrt(tolerance)/10.
- `search_incomplete`: optimization/profile coverage or local numerical rank is
  insufficient, or the final vector moved away from the frozen profile center.
- `locally_supported_on_tested_grid`: completed local searches found no such
  alternative on this finite grid. It is explicitly not a global exclusion,
  interval, correctness certificate, or confidence probability.
- `unavailable`: the needed complete training check could not be obtained.

Reliability endpoints also supply alternative-vector witnesses. If profiles
improve the center, this run records that improvement and requests recentering
through its `profile_center_changed` field; it does not pretend that the original
profiles certify the new vector. Conditional assessments can later accompany
candidate skeletons, but an incorrect skeleton can still produce a narrow,
misleading parameter range. That question is outside M19.

## Timing, resume and reports

Persist each operation's identity, allowance, best candidate, actual rollout-call
count, wall time and process CPU time. Completed operations replay verbatim.
One counted rollout call evaluates the complete training split; it is not one
trajectory, integrator step, or right-hand-side evaluation. Started calls that
fail are counted too.
An interrupted operation is marked unresolved and is not restarted with a fresh
budget; unknown cost remains explicit. Later unstarted operations may proceed.
Concurrent writers are excluded by task/operation locks.

Report setup, incumbent check, reliability, profiles and profile-rescue costs
separately. Independent post-selection evaluation has its own measured cost.
Source M18 fitting time is reported separately, never charged again. In-process
clocks exclude initial interpreter/import time; Slurm elapsed time provides the
outer job total. The scalar totals are observed lower bounds when accounting is
incomplete; interrupted operations also retain their full allowance charge.

Three-state endpoints have 24 profile points; six-state endpoints have 48.
The worst six-state allowance is approximately 34 minutes for checks/searches,
plus up to six minutes for two retrospective evaluations and setup. The scheduler
reserves **one hour per CPU task**, 16 GB, six concurrent tasks. Individual stages
stop early when appropriate. No full campaign has been run locally.

The report preserves before, after-reliability and final metrics for every source
method/start; coefficient/initial errors never feed selection. It cross-tabulates
confidence categories with retrospective coefficient recovery and records missing
or failed endpoints. These 24 dependent synthetic endpoints are diagnostic
calibration data, not evidence of a universal probabilistic confidence level.

## Delta commands

Upload `transfers/phase-c-fitting-m19.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. The bundle includes code, tests, configuration and
the frozen M18 endpoint export. Use a new root; do not overwrite M18.

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=/work/hdd/bibo/yxiao2/phase_c/code/fitting-m19
mkdir -p "$AF_CODE"
tar -xzf /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m19.tar.gz -C "$AF_CODE"
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_confidence_delta.sh"
BASH
```

Default output: `/work/hdd/bibo/yxiao2/phase_c/fitting-confidence-v1`.
Existing `AF_PYTHON`/`AF_CASADI_ROOT` fitting dependencies are reused. Optional
`AF_CONCURRENCY` controls only scheduling, default six. No remote sessions or jobs
are launched by the local coding agent.

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m19/scripts/hpc/inspect_phase_c_fitting_confidence_delta.sh
```

The inspector prints before/after metrics and costs and creates a dated review
archive. Missing results remain explicit. Submission intent/receipt handling is
unchanged; an uncertain `sbatch` outcome must be reconciled before retrying.

## Local verification

The 23 targeted M19/M18 tests passed on a frozen source snapshot. They cover
coefficient/initial compensation, fixed-profile nuisance fitting, numerical
derivatives including initials, input separation, failed-rollout handling,
interruption accounting, exact resume, source seals and journaled CPU submission.
The real three-state smoke completed all 24 profile points and resumed without
repeating completed operations. Its deliberately short budgets produced
`search_incomplete`, rather than a false confidence claim. Its observed fitting
and diagnostic overhead was 228 seconds on the local machine; this is a smoke
measurement, not a Delta timing estimate or a recovery result.

The full frozen-snapshot suite returned 4,143 passed, 35 skipped and 65 failures:
54 require external/private benchmark fixtures absent from the snapshot, nine
require checkout metadata or the checkout's `.venv` path, and two older scaled
fitting tests exhausted short wall-time budgets under parallel load. The
checkout-dependent tests passed in a 57-test rerun in the primary checkout; both
timing-sensitive tests passed in a separate sequential rerun. The missing
benchmark fixtures were not copied into this diagnostic. Changed-file Ruff checks
passed; repository-wide `ruff check .` still reports 37 unrelated existing
findings. No production fitter defaults or benchmark data were changed.

## First Delta return: completed fits, incomplete scoring and profiles

Reviewed `review-20261008-034116.tar.gz`, plan
`318cdc03bc281a226a76d4f66f08a44166ab62f54b0d7a642eb9fc5a4fce3738`,
from commit `cced88bb0f3338d2cddc27e970eb8e2ef6f5b529`.
All 24 fitting/checking backends finished. All 2,159 sealed JSON artifacts passed
their content-digest checks; each backend equals its finished fitting checkpoint.
The zero-result summary is a **postfit evaluator failure**: `_evaluation` did not
create its directory before the replay journal's first write. The regression
test now calls the real evaluator rather than mocking its filesystem behavior.
The fix does not alter any optimization objective or fitted vector.

Before resubmission, a read-only matrix-exponential replay of the saved vectors
gave the following retrospective results. This evaluator uses the existing known
linear control generator with piecewise-linear forcing. No optimization or
parameter selection was performed, and reference/validation values did not
enter the fitting stage. These are an independent archive analysis; the original
campaign's DOP853/Radau scoring still needs its recovery jobs.

| Control | All coefficients within 1%, before | After | All hidden initials within 0.001, before | After |
|---|---:|---:|---:|---:|
| 3 states | 6/6 | 6/6 | 6/6 | 6/6 |
| 3 states, separated timescales | 5/6 | 5/6 | 3/6 | 5/6 |
| 6 states | 4/6 | 4/6 | 4/6 | 4/6 |
| 6 states, separated timescales | 0/6 | 1/6 | 0/6 | 0/6 |
| Total | 15/24 | 16/24 | 13/24 | 15/24 |

For the hardest six-state controls, median maximum coefficient error decreased
from 55.57% to 4.832%. The best coefficient recovery has maximum relative error
8.89e-6, but its worst hidden-initial absolute error is still 0.0401. The two
ordinary six-state profiled starts with 323% coefficient error remain in that
region; their validation NMSE is still about 3.68e-7. The bad three-state
fast/slow joint start also remains unsuccessful. Thus the additional restart
searches improve some fits without establishing robust joint recovery.

Of 864 planned profile points, two are outside bounds and **all 862 in-domain
points are unavailable**. They exhausted the 25-second operation allowance
while computing a sensitivity rollout. Each verification currently recomputes
the entire coefficient/initial Jacobian before its independent solver check;
the reserved ten seconds was insufficient on Delta. Saved optimization
checkpoints exist, but a saved candidate is not a completed verified profile.
No profile rescue was triggered.

At each declared loss tolerance, the original backends label six endpoints
`weakly_constrained` and 18 `search_incomplete`. All six weak endpoints are the
six-state fast/slow controls. Their witnesses come from independently checked
**reliability-search alternatives**, not successful profiles. A separate
matrix-exponential training replay of all 862 saved profile candidates also
finds near-equivalent alternatives for those same six endpoints at tolerance
1e-12. This corroborates ambiguity, but does not convert the timed-out searches
into completed profile minima or calibrated confidence intervals. High-loss
profile candidates cannot rule out better nuisance fits.

### Observed additional cost

| Stage | Sum of task wall time | Purpose/outcome |
|---|---:|---|
| Incumbent checks | 287 s | Independent rollouts and joint sensitivity |
| Restart searches and their checks | 5,731 s (1.59 h) | Partial recovery and alternative-vector witnesses |
| Profiles | 21,551 s (5.99 h) | Checkpoints saved; no verified profile points |
| Total, including setup | 27,576 s (7.66 h) | 25,231 process-CPU seconds; 17,736 dataset rollout calls |

This is summed work across tasks, not elapsed queue or campaign time. Median
additional time per endpoint is 18.5 minutes. Profiles consume approximately
78% of the added wall time. Postfit scoring cost is still outstanding. The added
wall time is about 9.4 times the source M18 fit time; these are additional-budget
diagnostics, not evidence of a cost-efficient replacement fitter.

### Scoring-only recovery on Delta

Keep the original `code/fitting-m19` directory and frozen plan unchanged. Its
source/runtime identity is required for deterministic resume. Upload only
`scripts/recover_fitting_confidence_evaluation.py` to
`/work/hdd/bibo/yxiao2/phase_c/recover_fitting_confidence_evaluation.py`.
The helper verifies matching completed backends, creates the missing scoring
directories, and resumes the original controller. It refuses incomplete or
changed fits, preserves operation budgets, and journals separate CPU submission
receipts without replacing the original submission manifest.

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=/work/hdd/bibo/yxiao2/phase_c/code/fitting-m19
export PYTHONPATH="$AF_CODE/src:$AF_CODE:/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python \
  /work/hdd/bibo/yxiao2/phase_c/recover_fitting_confidence_evaluation.py submit \
  --root /work/hdd/bibo/yxiao2/phase_c/fitting-confidence-v1
BASH
```

This requests 24 CPU scoring tasks, six concurrent, one CPU and 15 minutes per
task, followed by a report. It makes no fitting, profile or LLM calls. New job IDs
are in `submission/evaluation-recovery/manifest.json`. After completion, the
existing inspector command above regenerates the report/review archive. Its
original submission listing can still show the old failed jobs; recovery jobs
have separate receipts and logs.

### Next diagnostic, before further scaling

Avoid recomputing a full sensitivity matrix at every profile point: verify the
fixed candidate's outputs first and calculate sensitivities only where needed.
Use exact linear propagation as a qualification control for these linear tests;
it is not a substitute for a nonlinear benchmark integrator. Measure verification
cost before freezing the next budgets, and test targeted profiles of weak joint
directions/parameters. Reuse saved candidates only with explicit provenance and
separate accounting. Retain the conservative distinction between a verified
alternative, an unresolved search, and evidence supporting a local constraint.
None of these changes should retroactively replace the M19 frozen protocol.

### Recovery verification

The full primary-checkout suite passed: **4,277 passed, eight skipped** (Torch
unavailable), with no failures. The final targeted suite passed **28 tests**,
including refusal of incomplete/mismatched backends, saved-result reuse and
idempotent/uncertain scheduler submissions. A real scoring-only smoke using the
original portable M19 source completed both retrospective evaluations in 17.6
seconds, preserved its saved fitting backend exactly, and reused the results on
a second run without refitting. Changed-file Ruff checks and `git diff --check`
passed. Repository-wide Ruff still reports 37 pre-existing unrelated findings.
