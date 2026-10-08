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
