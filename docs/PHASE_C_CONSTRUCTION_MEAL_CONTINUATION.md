# Construction meal continuation and finite-rate controls

3 November 2026. This training-only CPU milestone follows the returned
`construction-portfolio-targeted-1` campaign. It starts from the independently
verified model-15 endpoint with training NMSE **0.4266471542730798**, rather than
the older 0.43846 checkpoint. The source is the saved `candidate-15-saved-meal_flux`
record. Its equations, parameter domains, public training split and original
equation verification remain unchanged.

## Scientific and numerical purpose

The previous best fit used a nearly non-decaying meal reservoir. The positive
coordinate change writes `B=k*A` and `g=k*q` for the exact block
`A'=q*r-k*A`, whose glucose contribution is `+k*A`. It therefore searches
the equivalent shared coordinates suggested by `B'=g*r-k*B` and contribution
`+B`, while propagation and root profiling still use the original ODE and its
finite inverse. The new experiment tests continued fitting and finite positive
decay rates, rather than adding mechanisms or supplying ground-truth parameters.

The frozen roster contains six tasks. `T` is the maximum public training
trajectory duration, equal to 300 for this intake.

| Control | Absorption rate | Coordinates |
| --- | --- | --- |
| Warm continuation | Free, initialized at the verified best endpoint | Positive meal-flux |
| Finite rate 1 | Fixed at `0.001/T` | Positive meal-flux |
| Finite rate 2 | Fixed at `0.01/T` | Positive meal-flux |
| Finite rate 3 | Fixed at `0.1/T` | Positive meal-flux |
| Finite rate 4 | Fixed at `1/T` | Positive meal-flux |
| Exact-zero control | Fixed at `0` | Original model |

Every positive-rate start preserves `g=k*q` and each trajectory's `B(0)=k*A(0)`
when it changes k. Other shared coefficients and training initial roots are
refitted. Fixed-rate trial errors are feasible upper bounds on profile minima;
unfinished nuisance fitting cannot exclude a rate or establish confidence.

Exact zero is a different case. Finite original q and A(0) imply g=0 and B(0)=0
at k=0. A certificate checks that the meal state and its input gain have no
other uses affecting outputs, then chooses q=0 and initial A(0)=0 as an
observed-output-equivalent representative. The shared q coordinate is held at
zero. The existing inner root profiler remains unchanged; this does not claim
that every latent column was removed or that all hidden trajectories are equal.
The zero control cannot import a nonzero meal contribution from a singular limit.

## Termination, retention and interpretation

Root-profile search now records raw optimizer termination separately from a
qualified finite endpoint. A raw success flag without a paired, valid objective
evaluation does not become qualified optimizer success. Callback coordinates
without such evidence cannot be labeled accepted numerical endpoints. This
qualification adds no unbudgeted fitting calls and does not certify stationarity.
Reports distinguish `optimizer_raw_success`, `optimizer_qualified_success`, and
`optimizer_feasible_termination`. A feasible terminal point can coexist with a
raw optimizer failure; neither field alone establishes convergence. Missing
terminal evidence remains unknown.

The original verified incumbent is retained independently of each diagnostic.
New endpoints must pass independent original-equation replay and the numerical
improvement guard. A worse or failed trial cannot erase the source endpoint.
Use `trial_training_nmse` to compare the fixed-rate controls, not the best
retained loss, which may still belong to the inherited endpoint.

The experiment does not call an LLM, revise equations, access validation or test
data, change a benchmark, or establish causal-prefix forecasting performance.
Training initials are fitted from supplied training trajectories. Better
reconstruction does not imply accurate latent physiology or identified parameters.

## Resources and execution identity

Each task has at most **1,800 numerical seconds and 2,000 objective attempts**
for search. Setup, qualification and independent verification have separate
durable reservations. Six Slurm tasks request **one CPU and 8 GB each**, with
**at most two tasks running concurrently**, a four-hour worker ceiling and a
thirty-minute dependent report. The launcher rejects a reservation envelope
over 13,200 seconds per task. Scheduler time is a ceiling, not expected runtime.

The execution seal binds the plan, source identity, actual interpreter,
scientific package versions, launcher files, account and resource limits.
Leave the submitted code directory unchanged until the campaign finishes.
`AF_MEAL_CONTINUATION_CONCURRENCY=1` enables serial operation if set before
submission. Changing it after submission changes the sealed execution identity.

## Upload and submit on Delta

Use the committed code bundle delivered with this milestone. The bundle contains
code only; the completed targeted campaign must already exist on Delta. Its
`SOURCE_COMMIT` and `SHA256SUMS` identify the delivered version. In the **Mac
terminal**, upload the supplied bundle:

```bash
AF_BUNDLE=/Users/yuanzhangxiao/Projects/autoformalism/output/construction-meal-continuation-delivery-20261103/construction-meal-continuation-code.tar.gz
test -f "$AF_BUNDLE" || exit 1
scp "$AF_BUNDLE" yxiao2@login.delta.ncsa.illinois.edu:~/construction-meal-continuation-code.tar.gz
```

On **Delta**, extract into a fresh populated code directory and submit:

```bash
bash <<'BASH'
set -euo pipefail
AF_BUNDLE="$HOME/construction-meal-continuation-code.tar.gz"
test -f "$AF_BUNDLE"
AF_REPO=$(mktemp -d /projects/bibo/yxiao2/repos/construction-meal-continuation.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_REPO"
cd "$AF_REPO"
test -f scripts/hpc/start_construction_meal_continuation_delta.sh
if [ -f SHA256SUMS ]; then sha256sum --quiet -c SHA256SUMS; fi
if [ -f SOURCE_COMMIT ]; then cat SOURCE_COMMIT; fi
export AF_MEAL_CONTINUATION_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-targeted-v1
export AF_MEAL_CONTINUATION_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-meal-continuation-v1
bash scripts/hpc/start_construction_meal_continuation_delta.sh run
BASH
```

The `run` command authenticates the source receipts, freezes the new plan and
submits the array and report. It prints progress before potentially slow
filesystem and scheduler operations. A disconnected terminal alone does not
mean submission failed. `prepare`, `verify` and `submit` are also separate stages.

Campaign overrides are `AF_MEAL_CONTINUATION_SOURCE`, `AF_MEAL_CONTINUATION_ROOT`,
`AF_MEAL_CONTINUATION_CONFIG` and `AF_MEAL_CONTINUATION_CONCURRENCY`. The common
`AF_PYTHON`, `AF_CASADI_ROOT` and `AF_ACCOUNT` settings are preserved. Older
portfolio environment variables do not select this campaign's paths.

## Status and inspection from any directory

The execution record remembers the populated code directory. These commands
work after opening a new SSH terminal; there is no need to make another empty
directory or guess which old checkout contains the worker.

```bash
bash <<'BASH'
set -euo pipefail
export AF_MEAL_CONTINUATION_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-meal-continuation-v1
AF_REPO=$(jq -er '.value.repo | select(type == "string" and length > 0)' \
  "$AF_MEAL_CONTINUATION_ROOT/launcher_execution.json")
AF_WORKER="$AF_REPO/scripts/hpc/start_construction_meal_continuation_delta.sh"
test -f "$AF_WORKER" || { echo "Recorded worker missing: $AF_WORKER" >&2; exit 1; }
bash "$AF_WORKER" status
# Run after the jobs finish to package results:
bash "$AF_WORKER" inspect
BASH
```

`status` reads small plan, terminal and submission receipts, without loading
the numerical input payload or rerunning the report. `inspect` reconstructs the
evidence and creates `$AF_MEAL_CONTINUATION_ROOT/inspection.tar.gz` without
fitting. The report may be incomplete while tasks are still running.

Queue status is independent of reporting:

```bash
squeue -u "$USER" -o "%.22i %.48j %.12T %R" |
  awk 'NR == 1 || /construction-meal-continuation/'
```

Download from the **Mac terminal**:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-meal-continuation-v1/inspection.tar.gz ~/Downloads/construction-meal-continuation-inspection.tar.gz
```

## Recovery and retention

Confirmed scheduler receipts are reused. Ambiguous submission replies block
automatic resubmission: inspect saved intent/reply records and `squeue`/`sacct`
before recovery. Do not delete seals or choose a fresh root merely to bypass an
identity mismatch. If only reporting fails, use the recorded worker's `report`
or `inspect` stage; do not rerun numerical tasks.

Interrupted numerical operations keep their spent reservations and best durable
checkpoints. Resume never silently grants a fresh search allowance. A deliberate
continuation requires a new campaign identity and separately recorded costs.
Retain the source campaign, frozen inputs/plan, submitted code, execution seals,
submission receipts, operation journals, terminal results and logs. All stronger
scientific or predictive conclusions require further evidence.
