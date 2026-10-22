# Construction fitting: numerical reliability follow-up

This opt-in milestone follows the returned `construction-affine-continuation-1`
campaign. It keeps all eight equation models, both original/reduced parameter
arms, original parameter bounds, training data, scoped initializer semantics and
residual normalization. It does not generate proposals, change the benchmark,
open validation/test data, or change production fitting defaults.

## Why this follow-up is necessary

The first campaign returned 15 of 16 terminal tasks. The user supplied scheduler
evidence that array task `23111879_13` was cancelled (`0:54`, elapsed 96 seconds).
That is model 6's reduced arm. Its numerical cost is unknown; scheduler elapsed
time is not a numerical-fit CPU receipt. Absence from the active queue is not
evidence that it completed.

All 15 retained endpoints passed independent original-equation verification, but
none established local optimization convergence. The initial/recovered training
NMSEs, in frozen input-model order, were:

| Model | Original saved | Continued original arm | Continued reduced arm |
|---|---:|---:|---:|
| 0 | 8.434942 | 2.024695 | 2.329531 |
| 1 | 3.831009 | 2.071627 | 2.380473 |
| 2 | 7.738446 | 2.071624 | 2.071624 |
| 3 | 1.812509 | 1.812509 | 1.812509 |
| 4 | 1.501873 | 1.168429 | 1.168412 |
| 5 | 1.802080 | 1.789900 | 1.790120 |
| 6 | 1.052356 | 1.042719 | unavailable |
| 7 | 1.663064 | 1.188070 | 1.188072 |

There were three actionable numerical issues:

1. Relative derivative disagreement blocked model 3 even though the absolute
   discrepancies were approximately `2.2e-12` and `6.1e-12` in almost-insensitive
   columns. This is different from a materially incorrect derivative.
2. Six saved candidates failed independent solver agreement. Some discrepancies
   were small enough to merit a stricter integration replay; one was materially
   larger. Neither low NMSE nor a tiny loss discrepancy alone justifies bypassing
   the prediction-agreement check.
3. Three searches stopped when a trial produced nonfinite affine states or
   sensitivities. A failed trial need not end a trust-region search whose current
   accepted point is finite.

`xtol` stops did not establish convergence: the corresponding gradients and/or
accepted-step histories still failed the declared local assessment. Budget stops
also remain unresolved. Removing gain redundancy does not resolve every weak
parameter direction or prove identifiability.

## Numerical changes

### Precision-aware derivative assessment

An explicit opt-in audits a fixed ladder of finite-difference scales. It retains
resolved agreement, material disagreement, and numerical inconclusiveness as
different outcomes. Probes respect bounds and preserve kink checks. All calls
and incomplete coverage are recorded; the audit is bound to the exact endpoint,
residual, Jacobian, coordinate units, bounds and policy.

Search qualification is separate from parameter-information qualification.
An unresolved tiny direction may permit search only when its estimated effect
on the objective gradient is negligible under the declared policy. It is not
relabeled as a verified derivative or a well-constrained parameter. Local
assessment includes the gradient-error allowance and withholds stronger parameter
information when precision remains unresolved. These are finite numerical
diagnostics, not rigorous global error bounds or confidence probabilities.

### Independent endpoint verification

Each verification uses the original equations and initializer graph. The
predetermined Radau tolerance ladder is `(rtol, atol) = (1e-9, 1e-11)`,
`(1e-11, 1e-13)`, then `(1e-12, 1e-14)`, compared with exact affine propagation.
It stops at the first passing level. Acceptance thresholds are unchanged:
maximum normalized prediction gap at most `1e-7`, and NMSE discrepancy at most
`max(1e-12, 1e-8 * max(the two NMSEs))`. One finite numerical allowance covers the
entire ladder, with a separate finite inclusive ceiling for checkpoint overhead
and oracle construction. An exhausted ladder remains
unavailable; tolerances are never loosened to accept a checkpoint.

### Invalid trial handling

Only a typed nonfinite numerical trial is eligible for rejection. The initial
point must be valid. A rejected trial cannot update the best point, appear in
accepted-step history, or provide a fabricated finite Jacobian. The trust-region
solver may reduce its step and retry; all attempted evaluations remain charged.
Unrelated exceptions and deadlines still terminate the attempt, preserving the
best finite checkpoint for separate verification. The legacy behavior remains
the default outside this experiment.

## Experiment sequence and accounting

The portable input is a strict training-only projection of the authenticated
old campaign. Its original plan/input identities, operation receipt identities,
selected endpoints, failed verification candidates and known costs are sealed.
Free-form logs and held-out metrics are not imported into fitting.

For each of the 16 arms:

1. Reverify its inherited retained endpoint. If the old task has no retained
   endpoint, use the exact original saved vector and report the missing history.
2. Replay its saved failed verification endpoints in a fixed order. Select a
   replacement only if independently verified and strictly better in training.
3. Qualify the selected point using the opt-in derivative policy.
4. Run one bounded continuation from this point, preserving useful intermediate
   checkpoints. This is not a repeat of the old declared/random restart roster.
5. Independently verify the new endpoint, retain only verified improvements,
   and diagnose its own stopping history.

The frozen per-operation allowances are 600 seconds/1,000 oracle calls for the
one new search; 120 seconds for each qualification/diagnosis; 180 seconds for
each verification ladder or saved-endpoint replay. The inclusive stage ceilings
are respectively 750, 210 and 270 seconds, including oracle construction and
checkpoint overhead; the numerical wrappers also enforce their own smaller
inclusive ceilings. Durations are cooperative limits, not promised runtime.
All attempts, callbacks and interruptions are
accounted for. Historical and new cost remain separate, and an unknown historical
cost is never presented as zero. Source budgets are not renewed.

Operation reservations and content-bound receipts provide deterministic resume.
An interrupted operation cannot silently receive a fresh allowance. Completed
receipts are reused only under the same source, runtime, plan and operation
identity. The report is reconstructed from receipts without fitting. A separate
campaign root preserves the original results.

The report distinguishes execution completion, independent rollout verification,
training fit, local optimization status, and local parameter information.
Neither an available result nor an accurate prediction establishes correct
coefficients, mechanism validity, global optimality or readiness to revise an
equation. Review this follow-up before constructing scientific feedback.

Each arm starts from its own verified history, so the two arms need not share
the same starting point in this follow-up. This tests recovery and continuation;
it is not a fresh, matched-start comparison establishing which coordinate method
is intrinsically faster or more accurate.

## Delta execution

Use the portable bundle's fresh code directory and included sealed inputs:

```bash
export AF_PLATFORM=delta
export AF_RELIABILITY_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-reliability-v1
bash scripts/hpc/start_construction_reliability.sh run
```

The default Python is
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`, with CasADi dependencies
from `/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps`. The array requests one
CPU, 8 GB and 60 minutes per task, concurrency two, and no GPUs. Its report job
uses `afterany`, so failed/cancelled rows remain visible. Preparation validates
schemas on the login node; it does not run the full test suite or fit models.

```bash
bash scripts/hpc/start_construction_reliability.sh status
bash scripts/hpc/start_construction_reliability.sh inspect
```

Download `$AF_RELIABILITY_ROOT/inspection.tar.gz`. Do not change the extracted
source while jobs are active. Ambiguous scheduler replies require reconciliation
with the saved submission intent rather than blind resubmission.

ACES remains optional using `AF_PLATFORM=aces` and a distinct explicit output
root; it is not required for this CPU-only milestone.
