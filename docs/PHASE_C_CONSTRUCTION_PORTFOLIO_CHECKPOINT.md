# Construction portfolio: compact-checkpoint continuation

3 November 2026. An opt-in CPU continuation of the four verified endpoints from
`construction-portfolio-residual-v1`. It changes progress persistence and
termination reporting, while retaining the equation families, parameter domains,
independent verification requirements and training objective.

## Why this campaign is separate

The previous searches all stopped at their approximately 1,920-second overall
wall-clock ceiling. Models 12/15 spent about 1,173/1,179 seconds in checkpoint
callbacks and only 748/741 seconds in numerical work. Their sealed summaries
incorrectly described this as numerical-budget exhaustion. The endpoints remain
valid: independent original-equation replay verified the retained improvements.
The old results, labels and allowances are preserved as historical records.
Preparation authenticates all four source terminals and projects their runnable
evidence into a compact intake (approximately 1.77 MB in compact JSON), without
copying the nested historical input chain again. Source models, training payload,
numerical settings and retained endpoints remain explicitly bound to hashes.

This campaign starts from those complete retained parameter vectors:

| Model | Training NMSE at the new start | Search |
| --- | ---: | --- |
| 10 | 1.712941252 | Residual-only derivative-free control |
| 12 | 0.436484876 | Profile training initial roots, search shared parameters |
| 14 | 1.688725401 | Residual-only derivative-free control |
| 15 | 0.438860884 | Profile training initial roots, search shared parameters |

Each outer search has a new, explicit allowance of at most **2,000 objective
evaluations and 1,800 numerical seconds**. Models 10/14 retain their stricter
solver precision; models 12/15 retain their exact affine propagation and
conditional bounded least-squares root checks. The new root is a warm
continuation, not restoration of the old internal Powell directions.
The search's overall wall-clock safety ceiling remains 1,920 seconds; this
milestone reduces persistence overhead rather than enlarging that ceiling.

The launcher submits four tasks with **one CPU and 8 GB each, at most two
concurrently**, and a four-hour worker ceiling. A dependent report has 30 minutes.
The computed per-task numerical-stage reservations must fit that envelope.
There are no GPU or LLM calls and no validation/test access.

## Persistence and interpretation

Routine progress saves are compacted and throttled; unchanged counters no longer
require a full durable write before and after each objective. Immutable evidence
is represented by content digests in ordinary progress records; completed
setup/qualification receipts and full terminal records retain its contents.
Improved incumbents, accepted checkpoints and terminal outcomes retain their
durability requirements. An interruption may lose recent counter-only updates,
but not reset the allowance or remove a previously persisted incumbent.
The journal reserves work before execution, so a worker interruption or retry
cannot reset numerical allowances. Progress compaction does not change the
optimizer objective, loosen profile checks or promote an unverified endpoint.
The explicit campaign policy sets the routine checkpoint interval to 30 seconds.
Historical default fitting behavior remains unchanged outside this opt-in run.

Reports distinguish the evaluation cap, the numerical-active allowance and the
overall wall-clock ceiling. Operational `complete` only means a terminal outcome
has been recorded. Read `search_termination` and `optimization_converged` as well.
Checkpoint counts and timings are needed to establish the improvement on Delta;
smaller local records alone do not prove that filesystem waits have disappeared.

All reported NMSEs remain training reconstructions with training-estimated roots.
The independent original-equation replay is a numerical verification, not a
held-out forecast. New retained endpoints require their own causal-prefix
evaluation before making a generalization claim. Optimization adequacy remains
a separate unresolved question when attributing training mismatch to equations.

## Submit on Delta

Use the committed source bundle in a new code directory. Keep that directory
unchanged while jobs run. The interpreter, scientific packages, source tree and
launcher entrypoints are bound to the execution identity.

The source campaign must already be present on Delta. This code bundle does not
include experiment data or historical results.

```bash
export AF_PORTFOLIO_CHECKPOINT_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-residual-v1
export AF_PORTFOLIO_CHECKPOINT_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-checkpoint-v1
bash scripts/hpc/start_construction_portfolio_checkpoint_delta.sh run
```

`run` authenticates the source results, freezes the new plan and submits the array
and dependent report. Progress is flushed before filesystem/Slurm operations.
For separate steps, use `prepare`, `verify`, then `submit`. Existing environment
variables from older campaigns do not choose this campaign's root or source.
Optional `AF_PORTFOLIO_CHECKPOINT_CONCURRENCY=1` permits serial execution; set it
before submission and keep it unchanged for execution verification.

## Status and inspection from any Delta directory

Do not create another empty repository directory to inspect results. Recover
the exact submitted source directory from the execution record:

```bash
bash <<'BASH'
set -euo pipefail
export AF_PORTFOLIO_CHECKPOINT_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-checkpoint-v1
AF_REPO=$(jq -er '.value.repo | select(type == "string" and length > 0)' \
  "$AF_PORTFOLIO_CHECKPOINT_ROOT/launcher_execution.json")
AF_WORKER="$AF_REPO/scripts/hpc/start_construction_portfolio_checkpoint_delta.sh"
test -f "$AF_WORKER" || { echo "Recorded worker is missing: $AF_WORKER" >&2; exit 1; }
bash "$AF_WORKER" status
# Run after completion to package the results:
bash "$AF_WORKER" inspect
BASH
```

`status` reads small sealed receipts; it does not fit or load the full numerical
inputs. `inspect` reconstructs evidence and creates
`$AF_PORTFOLIO_CHECKPOINT_ROOT/inspection.tar.gz`, without running fitting. During
execution, use just the `status` line if an archive is not yet needed.

From the **Mac terminal**, download the archive:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-checkpoint-v1/inspection.tar.gz ~/Downloads/construction-portfolio-checkpoint-inspection.tar.gz
```

## Recovery and retention

Confirmed Slurm submissions are reused; an ambiguous reply blocks automatic
resubmission. Inspect saved submission intents alongside `squeue`/`sacct` before
recovering an uncertain job. A disconnected SSH session alone does not justify
resubmitting, deleting receipts or choosing another root.

If only reporting fails, run `report` or `inspect` using the same recorded
worker path. Interrupted workers retain operation reservations and persisted
incumbents; retrying cannot create a new search allowance. A deliberate
additional allowance needs a separately identified campaign.

Retain the previous source root, new frozen inputs/plan, execution/submission
identities, operation and terminal receipts, logs, and submitted source tree.
No automatic construction revision, scientific-confidence claim or production
policy promotion occurs in this campaign.
