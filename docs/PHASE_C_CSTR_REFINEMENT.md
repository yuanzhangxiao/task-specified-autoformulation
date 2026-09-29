# Phase C: bounded CSTR fitting follow-up

This is the numerical follow-up to [reference qualification](PHASE_C_REFERENCE_QUALIFICATION.md),
owned by Orion. The corrected CSTR input/reference release passed its numerical
audit and known-parameter training/validation replay. Generic fitting is not yet
qualified. Dalla Man and alien-device public-interface qualification remain open;
this experiment neither changes their data nor establishes whole-suite correctness.

**2026-09-28 result:** all fourteen attempts are now recorded. The hard/generic
combined initializer arm reached train/validation NMSE 0.00014821 / 0.00006713.
See [the fitting findings and deferred integration note](PHASE_C_FITTING_FINDINGS_2026-09-28.md)
for all endpoints, checkpoint attribution and hidden-state limitations. Further
fitting development is deferred while the new dataset contracts are qualified.

## Evidence motivating the comparison

The received `cstr-qualification-diagnostics.tar.gz` has SHA-256
`e88be304039f8cef2ace2ffb0be48bff9dc2595988f85e76ea5e4ce443a2f667`.
Its qualification plan is
`f5d68fa869c317ba30e23a8a4f913392f742e854660c9f641105a7bba9e9827d`.
All 18 artifact seals and the four backend-result links were verified locally.

| Original run | Train NMSE | Validation NMSE | Retained parameter source |
|---|---:|---:|---|
| Easy, near reference | 4.22e-10 | 1.54e-10 | Converged collocation, 80.85 seconds; all dynamic coefficient errors below 0.017% |
| Easy, generic | 0.943 | 0.689 | Coarse 0.1-times-start screening point |
| Hard, near reference | 0.167 | 0.180 | Original dynamic guesses; jacket initializer slope changed from 0.6 to 1.6 |
| Hard, generic | 0.820 | 0.413 | Last admissible collocation checkpoint |

All four refinements used derivative-free polling because the equations contain
`max` guards. None reached the end of the initial wide sweep before its
180-second limit. They used 48–75 residual calls, below their 240-call caps.
Three refinements made no improvement over their best screening point; hard/near
improved training cost by 2.31%. The easy/near initializer converged; the other
three initializers timed out. This does not establish that simply continuing
every optimizer would converge to the reference.

Retained dynamic gains respected their nonnegative domains. However, the
hard/generic initializer produced C(0) = -33.19 on one training trajectory. The
hard/near collocation checkpoint also produced negative initial concentrations,
but that checkpoint was not retained. Parameter-domain validity and physical
initial-state validity are different checks.

## Frozen comparison: fourteen independent CPU fits

The original four requests supply equations, public development arrays, and
**original optimizer guesses**, not their fitted winning parameters. Every arm
starts a separate numerical attempt; the source qualification is read-only.

| Arm | Cases | Initialization / refinement seconds | Residual-call cap | Change |
|---|---:|---:|---:|---|
| `wide_standard` | 4 | 120 / 180 | 240 | Current fitter control |
| `local_standard` | 4 | 120 / 180 | 240 | Local polling only |
| `wide_long` | 4 | 300 / 900 | 1200 | Larger budgets only |
| `local_physical_initials` | 2 hard cases | 120 / 180 | 240 | Local polling plus initialization constraints/coordinates |

Compare local versus wide at the standard budget, larger-budget wide versus
standard wide, and physical-initializer versus ordinary-initializer local fits.
Report all starts and endpoints, including failures. The last comparison bundles
positivity with coordinate changes; it cannot separately attribute their effects.
There is no local-long arm in this first follow-up and no best-of-N selection.

Local polling evaluates the exact guarded equations. It begins at radius 0.1,
uses scales `max(abs(best_screened_parameter), 1)`, and alternates coordinate and
seeded orthogonal bases. Each batch contains at most 2p directions for p fitted
parameters. Failed improvement halves the radius; successful improvement expands
it by 1.5, as in the existing policy. It does not perform the initial radius
1/4/16 sweep. Positive parameter bounds, infeasible-point rejection, actual-call
accounting, and independent final replay are unchanged. This is a direct-search
heuristic, not a stationarity certificate or a branch-sensitivity algorithm.

The explicit profiles are `collocation-local-poll-v1` and
`collocation-budget-control-v1`. Older profiles retain the wide search. The
budget control preserves the original five-second node warm-up and ten-second
screening-probe setting; it is not the existing rescue profile, which also changes
those settings. New source identities require fresh fit directories; never patch
old freezes to resume historical runs with changed code.

## Physical initializer intervention

Let L and H be the minimum and maximum initial observed T in **training**, with
D = max(H-L, 1 K). Define w = clip((T(0)-L)/D, 0, 1). The hard concentration map is

```
C(0) = c_lo*(1-w) + c_hi*w,  c_lo >= 0, c_hi >= 0.
```

The two anchor guesses are transformed from the original affine guesses, so
initial values on the present training/validation design are preserved. The
reference witness is transformed similarly. Learned anchor parameters stay
shared across trajectories and frozen for validation. Outside the anchor range,
the concentration map saturates; this is an explicit change from unrestricted
affine extrapolation. No trajectory IDs or hidden trajectory labels enter it.

The jacket initializer remains affine but uses deviation coordinates:

```
Tj(0) = T(0) + offset + change*(T(0)-L)/D.
```

This replaces a roughly 338-K intercept with a much smaller temperature offset.
Both guesses and the reference vector are transformed algebraically, preserving
the same initial values. Its output is not constrained to be positive. The
concentration guarantee concerns C(0), not arbitrary state paths or all physical
domains. These are evaluator-specified coordinates in an assisted diagnostic,
not an automatic claim that the discovery runtime inferred their meaning.

## Execution and retained evidence

The existing qualification CLI handles preparation, task execution, and reporting.
Each task checks known-reference attainability before fitting and independently
replays retained fitted parameters with Radau and DOP853. The existing NMSE and
solver-agreement thresholds remain unchanged. Parameter fitting uses training
only. No test payload, proposer, critic, API key, or GPU is involved.

Preparation and reporting serialize short filesystem operations across workers.
Individual fits keep their existing locks and durable attempt ledgers. Rerunning
a completed task reuses its result. An interrupted fit is recorded as interrupted
and is never silently restarted with a new budget. Successful worker completion
does not imply low error, convergence, or scientific uniqueness.

The launcher submits one CPU array with indices 0–13, at most two concurrent
tasks, one CPU and 8 GB per task, and a one-hour scheduler limit per task. The
one-hour allocation includes bounded reference/fitted replays and overhead; it
does not raise the optimizer budgets above the table. Source records are frozen
before optimization. Each worker refreshes the shared report, including failed
numerical outcomes. A scheduler kill can leave a missing row until read-only
reporting/reconciliation; it must not be called a successful fit.

## ACES commands

Upload the supplied source archive to the group directory, extract to the
supplied new code folder, and verify `SHA256SUMS`. No additional private benchmark
specification files are needed: the original qualification contains the sealed
known-equation requests and parameter witnesses.

```bash
export AF_REFERENCE_MODE=refine
export AF_QUALIFICATION_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-reference-v1/qualification
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-cstr-refinement-v1
bash "$AF_REPO_ROOT/scripts/hpc/submit_continuous_input_audit.sh" aces
```

After submission, in any new shell:

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-cstr-refinement-v1
sacct -j "$(cat "$AF_ROOT/submission/job.id")" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed
jq '{protocol,status,expected,rows}' "$AF_ROOT/qualification/summary.json"
```

The output root and submission receipts are reused only for identical execution.
An ambiguous scheduler reply blocks blind resubmission; inspect accounting and
the saved command before recovery. No automatic cleanup or remote submission
is performed by this development task.

Delta can run the same launcher with `delta`, its existing Python environment,
and a copied original qualification directory. Jetstream2 uses `jetstream2`
and runs the same fourteen tasks sequentially in the existing VM. Run the roster
on one platform first; there is no reason to duplicate it across clusters.

For a compact review archive after completion, include `plan.json`, `tasks.json`,
`summary.json`, and each task's `qualification.json`, `reference_parameters.json`,
`initial_coordinate_audit.json`, `reference_replay.json`, `fitted_replay.json`,
`fit/freeze.json`, and `fit/backend_result.json` where present. Preserve missing
or failed records rather than excluding them from the comparison.

## Local verification before cluster execution

- Full regression suite: 3654 passed, 8 skipped (optional Torch unavailable).
- Changed Python files pass Ruff. Repository-wide Ruff reports 37 existing
  findings in unrelated `analysis/claude` scripts. Shell syntax and diff checks pass.
- The portable source bundle passes all 17 focused launch/follow-up tests;
  its Python source identity matches the project checkout.
- Focused regressions cover matched starts/budgets, source seals, initializer
  values and concentration bounds, interrupted execution, active-fit locking,
  and ambiguous scheduler replies: 64 passed.
- A guarded-ODE smoke test used the real collocation/local-poll path, attained
  train/validation NMSE below 7e-17, and resumed without changing the result.
- Preparation and identical resume succeeded against the received four-case
  ACES qualification archive, producing all fourteen requests.
- With transformed hard initializers, the known reference attained training
  NMSE 1.99e-17 and validation NMSE 1.86e-17 across all development trajectories.
  Radau/DOP853 maximum scaled disagreement was 6.17e-7.

These checks qualified the diagnostic machinery before cluster execution. The
fourteen production-size fits subsequently completed; their results and limits
are recorded in the linked findings note.
