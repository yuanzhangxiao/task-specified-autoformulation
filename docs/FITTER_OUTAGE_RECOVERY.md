# Recovering the interrupted Delta comparisons

The September 10 runs overlapped a reported `/projects` and `/taiga` outage.
The numerical Python, dependencies and source checkouts live on `/projects`, even
though experiment outputs live on `/work`. This makes an infrastructure cause
plausible for missing results and long startup stalls. It does not establish the
cause of every failed fit. This milestone completes unavailable work before
changing any fitting settings or drawing further conclusions from the comparison.

## Original experiments

| Campaign | Original array | Pinned numerical commit | Original output |
| --- | --- | --- | --- |
| Stopping | 21959577 | `78fab68cbe83149218367e95d5d2b2ce62778fd0` | `/work/hdd/bibo/yxiao2/phase_b/fitter-stopping-v1` |
| Piecewise | 21961690 | `4d5f73639e9c0f57bed641f3f510b2e396701672` | `/work/hdd/bibo/yxiao2/phase_b/piecewise-fitter-v1` |

The stopping comparison contains 39 saved starts and two rate-coordinate
refinement arms: ordinary `ftol` versus disabled `ftol`. It does not rerun
collocation. The piecewise comparison contains 28 cases and four arms per case:
two collocation meshes, each followed by branch sensitivity or directional
polling. These counts describe the original full Delta campaigns; local smoke
fixtures may contain fewer cases.

The recovery driver is standalone standard-library code. It imports neither
fitter version in its supervising process. Its numerical child verifies the
original clean commit, runtime, launcher, frozen configuration and data hashes,
then calls that version's original task implementation. Stage wrappers log entry
and exit without changing function arguments, results or budgets. The new branch
does not need to be merged into either original fitter checkout or Sol's branch.

## What is recovered

Preparation inventories checkpoint hashes and writes a separate `snapshot/`.
The original experiment remains unchanged. A frozen `recovery.json` lists every
task, arm, original artifact and artifact omitted from the new attempt. Selection
uses checkpoint availability; it never reads NMSE to decide which arm to retry.

| Saved state | Action |
| --- | --- |
| Terminal result, including numerical failure or unverified replay | Preserve it. Do not rerun it to obtain a better score. |
| Saved `fit.json`, missing final report | Reuse fitted parameters and complete verification/reporting. |
| Saved polling checkpoint and initializer, missing fit | Resume the original point sequence and cumulative time/evaluation budget. |
| Native fit or initializer started, no completed fit/checkpoint | Record an explicit fresh attempt in the recovery directory. Retain a completed initializer when available. |
| No start marker or completed output | Run the original frozen task and budget. |

For an interrupted stopping task with a saved fit, successful replay trajectories
are retained. Cached replay timeout records may be retried in the new attempt;
their prior hashes and omission reasons are recorded. Cached numerical integration
failures are retained. A stopping task that already has a terminal numerical
result is preserved as a whole, including its unverified arms.

Native IPOPT and SciPy optimizer states were not serialized by the original code.
Consequently their interrupted internal iterations cannot be resumed exactly.
The fresh attempt is identified as `restart_interrupted_native`, not presented as
continuation or charged as an uninterrupted old run. An interrupted *recovery*
native fit cannot be silently restarted by repeating `run`; it needs another
explicitly named recovery attempt. Polling resumes its saved budget. A computation
interrupted between numerical evaluation and checkpoint commit may be repeated.

Preparation is resumable after a copy interruption if the source hashes remain
unchanged. Once prepared, repeating preparation never recopies over new results.
An existing partial submission stops with the recorded job IDs instead of
submitting duplicates. Complete submissions simply print those IDs again.

## Startup, supervision and reporting

The submission script copies the supervisor onto `/work` and uses the system
Python (standard library only; Python 3.6 or newer). Each job prints a shell marker
before accessing `/projects`. The numerical child then records interpreter
startup, imports, freeze verification, fitting/refinement, replay and completion
in `stages.jsonl` and `worker.log`.

A short CPU smoke verifies each original environment, solves a tiny IPOPT problem,
and integrates an ODE with an analytic reference using Radau. Its `afterok`
dependency gates that campaign's recovery array; a failed smoke cancels dependent
work instead of spending the full array allocation. Default concurrency is one
task per campaign, so submitting both uses at most two concurrent fitting CPUs.
Each task requests one CPU, 16 GB, no GPU, and one hour of scheduler time.

Original numerical deadlines are retained: 2,460 seconds for a default stopping
pair and 2,700 seconds for a default piecewise case. The larger scheduler envelope
allows startup and cleanup. The supervisor writes timeout status before terminating
the process group and uses bounded waits after both termination signals. A
process stuck in kernel filesystem I/O may remain unkillable until the filesystem
recovers; the report records failure to reap the child rather than waiting forever.
The outer shell watchdog and Slurm remain additional limits.

The summary uses only standard-library JSON and file hashing, with no numerical
imports or ODE solves. It verifies retained result/fit bytes, result identities,
and referenced stopping replay array hashes. All planned arms remain in the
denominator. `verified` means the original numerical task recorded completion,
not scientific correctness, unique latent recovery, or optimizer stationarity.
Stopping recovery retains the original threshold of both clean NMSEs <= 1e-4.
Outage/recovery wall-clock times should not be used for a controlled speed ranking.

## Delta commands

Use a clean checkout of the recovery commit supplied with the implementation.
The two original pinned checkouts should remain at their original commits.
From the recovery checkout:

```bash
bash scripts/hpc/submit_fitter_outage_recovery_delta.sh stopping
bash scripts/hpc/submit_fitter_outage_recovery_delta.sh piecewise
```

Defaults use the existing `autoformalism-v21` Python and
`fitter-methods-v1-deps` directory on `/projects`. Optional environment overrides
are `AF_BOOTSTRAP_PYTHON`, `AF_NUMERICAL_PYTHON`, `AF_CASADI_ROOT`,
`AF_ORIGINAL_OUTPUT`, `AF_LEGACY_REPO`, `AF_RECOVERY_OUTPUT`, and
`AF_ARRAY_CONCURRENCY` (1 or 2). Avoid carrying campaign-specific overrides from
one submission into the other.

Submission records are saved at:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/stopping-outage-recovery-v1/submission.json
cat /work/hdd/bibo/yxiao2/phase_b/piecewise-outage-recovery-v1/submission.json
```

Each contains `smoke_job_id`, `fit_job_id` and `summary_job_id`. After the summary
jobs finish, the compact reports are:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/stopping-outage-recovery-v1/summary-short.md
cat /work/hdd/bibo/yxiao2/phase_b/piecewise-outage-recovery-v1/summary-short.md
```

`summary.md` contains every arm; `summary.json` includes metrics and provenance.
If the summary job itself is interrupted, rerun just the read-only aggregation:

```bash
/usr/bin/python3 /work/hdd/bibo/yxiao2/phase_b/stopping-outage-recovery-v1/recovery-driver.py summarize --output /work/hdd/bibo/yxiao2/phase_b/stopping-outage-recovery-v1
/usr/bin/python3 /work/hdd/bibo/yxiao2/phase_b/piecewise-outage-recovery-v1/recovery-driver.py summarize --output /work/hdd/bibo/yxiao2/phase_b/piecewise-outage-recovery-v1
```

For missing output, inspect `logs/`, `smoke/worker.log`, and
`attempts/task_NNN/{stages.jsonl,worker.log,supervisor.json}`. Recovery task indices
are the original task indices. No original `submission.json`, fit-start marker,
or result needs to be removed.

## Collocation interpretation and next decision

Collocation can begin with node values that do not satisfy the discretized ODE
constraints. It still needs finite, evaluable expressions, a usable numerical
initial point, and enough optimization progress to find a useful solution.
Removing the former successful-rollout requirement permits optimization to start;
it does not guarantee a feasible solution or a good fit. The earlier synthetic
controls demonstrated that removing this requirement can rescue an otherwise
unusable start. The public-candidate runs still require separate diagnosis after
recovering infrastructure-interrupted work.

This milestone does not decide C versus A, alter time-constant/rate coordinates,
change stopping tolerances, redesign candidate equations, or increase numerical
budgets. The completed stopping and piecewise tables will inform the next fitting
milestone, while Sol continues the pre-fitting phases.

## Verification

Tests cover terminal-result preservation, explicit native restarts, saved-fit and
polling reuse, partial preparation, input/path/hash guards, replay-array integrity,
smoke gating, duplicate submission protection, task locks and bounded teardown.
Validation completed: the full repository suite passed with 1,319 tests and three
optional Torch skips; all 18 recovery tests passed after the final report grouping
and freeze-integrity checks. Ruff, shell syntax and whitespace checks passed.

Both original pinned runtimes passed native IPOPT/Radau smokes. On a disposable
copy of the existing piecewise smoke, a saved sensitivity fit remained byte-for-byte
unchanged, polling retained its 44 recorded evaluations and cumulative time, and
both arms completed verification. A further initially unavailable four-arm case
completed verification from scratch. Original smoke artifacts were unchanged.
These are workflow checks, not new evidence for ranking the fitting algorithms.
No Delta jobs were submitted during implementation.
