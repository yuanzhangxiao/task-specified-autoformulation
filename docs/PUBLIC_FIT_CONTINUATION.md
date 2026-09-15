# Bounded public fitting continuation pilot

The user authorized one additional refinement window after ACES job 2133560.
That run retained a finite training fit when its 180-second refinement budget
ended. Its last four recorded accepted costs decreased from about 4576 to 4301;
the reported training/validation output NMSEs were about 0.895/0.925. These are
reasons to test continuation before interpreting poor fit as model evidence.
They do not establish that a longer fit will succeed.

This is an explicit exception with its own version and output root under the
[operational freeze](FITTER_FREEZE.md). It does not reopen method selection,
change old results, or deploy a general automatic budget-allocation policy.

## Contract and accounting

`ContinuationSelection` has protocol `public-fit-continuation-pilot-1`.
`configs/public_fit_continuation_v1.json` pins the saved repaired model and its
complete backend-result digest. The runtime verifies the historical request,
public train/validation arrays, initialization lowering, profile settings,
result envelope and raw backend. Historical source identity remains recorded;
the continuation gets a new source identity. Old roots are never executed under
the new source hash.

The pilot requires a retained complete training fit from an explicitly
unconverged, budget-stopped sensitivity stage, with at least four coherent
accepted iteration records ending at that selected vector. Their last-window
cost decrease must be at least 1%. This is a guard for this authorized pilot,
specified after seeing its parent history, **not** a validated general stopping
or allocation rule. Validation scores do not enter this gate.

| Allocation | Original | Additional | Total allocated |
| --- | ---: | ---: | ---: |
| Collocation | 120 s | 0 s | 120 s |
| Screening/refinement | 180 s | 180 s | 360 s |
| Residual-call ceiling | 240 | 240 | 480 |

The original 180 seconds included feasibility screening; the additional window
contains only the retained-point check and sensitivity refinement. Actual calls
are counted separately; the parent used 22, not its full 240-call allocation.
Symbolic setup and final scoring are recorded outside the numerical window.
Deadlines remain cooperative and the scheduler imposes an outer 30-minute limit.
This is not a controlled timing comparison.

The continuation uses the same physical parameter coordinates, bounds, Radau
integration tolerances (1e-7/1e-9), normalization and `ftol=None`. It carries all
13 retained parameters for this candidate, including five learned causal
initializer parameters. The scientific initialization plan is unchanged and is
lowered exactly once. Observed initial values remain tied to observations.

No collocation, random starts, start portfolio, polling, input resampling,
new coordinates, or nonlinear-function redesign is introduced. This first pilot
supports a smooth single-target `v01` sensitivity parent. It makes a **new
least-squares invocation from the retained vector**; the previous trust-region
state is not restored. The first full-training evaluation must reproduce the
saved cost within relative 1e-5/absolute 1e-8 tolerance before optimization.
That check consumes the additional budget.

## Incumbent preservation and feedback

The original parameter vector and its scores remain an incumbent. The extension
retains finite full-training evaluations throughout optimization. A new vector
is selected only if its full production-training rollout has lower NMSE than
the parent's; worse or unavailable training results keep the parent. The final
selection is saved before validation is evaluated. If the selected extension
fails validation integration, its score is unavailable and its numerical failure
is reported; validation never selects another parameter vector.

`public-fit-continuation-result-1` separates:

- `status`: continuation execution, including interruption or extension failure;
- `selected`: parent or extension, with the complete parameter vector;
- training and validation output NMSE, with unavailable metrics suppressed;
- additional and cumulative actual residual calls, additional numerical seconds,
  and whether the extra budget was exhausted;
- native optimizer convergence, only when attributable to the selected vector;
- `feedback_status`: `budget_limited_unresolved`,
  `numerical_failure_unresolved`, `local_optimizer_stopped`, or
  `continuation_not_run`.

Every result sets `structural_failure_established=false`,
`independent_replay=not_performed`, and `automatic_followup=false`.
`local_optimizer_stopped` does not imply a global optimum, good prediction or
scientific acceptance. Failed extension execution can coexist with a usable
retained parent. NMSE still scores only observed `v01`, never hidden states.
Future proposer feedback must preserve these distinctions rather than using a
budget stop or integration failure as evidence that a specific term is wrong.
This milestone publishes the handoff; it makes no live LLM calls or autonomous
model revisions.

## Persistence and outputs

The parent experiment is read-only. The new output has a frozen parent snapshot,
selection, progress gate, budgets, source identity and a started marker written
before any numerical work. A separate reservation at
`<parent-experiment-parent>/.public-fit-continuations/<parent-identity>/`
binds that parent to one continuation output. Repeating preparation/execution
returns the existing result; choosing another output directory cannot replenish
its budget. Do not delete this reservation to retry.

An interrupted attempt keeps its checkpoints and original incumbent and returns
`interrupted`; it never receives a fresh window automatically. This does not
claim to resume optimizer state or validate an unscored partial checkpoint.
Recovery after an infrastructure interruption would require a separate decision.

Useful files under `<output>/continuation/`:

- `freeze.json`, `started.json`, `result.json`: lineage and terminal public result;
- `calls/`: residual attempts, accepted iterations, failures and best evaluation;
- `start_check.json`: retained-parameter objective agreement;
- `extension_checkpoint.json`: numerical evidence before final scoring;
- `training_selection.json`: selected vector before validation;
- `backend_result.json`: detailed optimizer, scoring and timing evidence.

The ACES worker also writes `<output>/summary.json`, runtime/preflight logs and
the submission manifest. Submission intent is sealed before `sbatch`, protecting
against duplicate submission and jobs starting before the final manifest exists.
The worker checks commit, selection and parent artifact bytes both before and
after preflight.

## Run on ACES

Use a clean checkout pinned to the implementation commit supplied with this
milestone. No new Delta experiment is required; the saved parent is on ACES.

```bash
export AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-public-fit-continuation-v1
export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/public-fit-continuation-v1
export AF_PARENT_FIT=/scratch/user/u.yx126462/phase_b/prefit-public-fit-v1/fit
export AF_SELECTION="$AF_REPO_ROOT/configs/public_fit_continuation_v1.json"
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
bash "$AF_REPO_ROOT/scripts/hpc/submit_public_fit_continuation_aces.sh"
```

The job runs focused tests and a real numerical smoke, then prepare, inspect,
run and report. The analytical smoke's timeout-history metadata is explicitly a
synthetic fixture; its extension/rollout scores are real. It validates software
behavior, not the efficacy of the progress gate on public models.

After the job:

```bash
cat /scratch/user/u.yx126462/phase_b/public-fit-continuation-v1/summary.json
```

Review the original and extended scores, actual calls, stop reasons and selected
stage together. At the cap, discuss the resulting feedback with the user and
Orion; do not schedule another extension automatically.
