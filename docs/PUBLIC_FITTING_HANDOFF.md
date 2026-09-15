# Public fitting handoff v1

This is the integration interface for the [frozen fitting protocol](FITTER_FREEZE.md).
It does not change either numerical backend or historical experiment callers.
The difficult reference-case comparison is closed; this milestone introduces no
new optimization-method experiment.

## Request and lineage

`autoformalism.schemas.public_fitting.PublicFitRequest` has protocol
`public-fit-1` and these fields:

| Field | Meaning |
| --- | --- |
| `base_candidate` | Canonical `CandidateModel` **before** causal initializer lowering |
| `context` | Original public `ValidationContext`; open rollouts, `fitted_initialization=false` |
| `initialization_plan` | `shared-latent-initialization-1`; exactly one rule for every latent state and none for directly observed states |
| `parameter_guesses` | Optional guesses for base equation parameters only; defaults to `{}` |
| `profile` | Explicit `general-rollout-v1` or `collocation-feasible-v1`; never inferred or replaced by a fallback |
| `random_seed` | Seed for the general fitter; recorded for both profiles. The existing C+S start design is deterministic and has no seed setting |
| `source` | `stage`, `task_id`, `artifact_sha256`; upstream artifact identity |

Source stages are `construction`, `requirement_repair`, `controller_revision`,
and `synthetic_control`. The exporter must verify the referenced upstream
artifact, its public context and its seal convention. This interface binds the
declared source into the request; it does not claim to authenticate an external
artifact that was not supplied. Exporter manifests should retain parent and
repair lineage separately.

The runtime compiles the restricted equations, validates complete latent rule
coverage, then lowers the initializer exactly once. Initial-map parameter guesses
come from the plan, not `parameter_guesses`. Shared latent values/maps are fitted
on training and frozen for validation; directly observed initial states are bound
to their observations. Per-trajectory parameter scopes and initial-value ranges
are rejected so the general backend cannot optimize validation initial values.
Known latent initials require the plan's explicit scientific justification.

`PublicSplit` has `name` (`train` or `val`), `fingerprint`, and `rows`. Each row
has `trajectory_id`, strictly increasing finite `time`, `targets`, `auxiliaries`,
`external_inputs`, and scalar `fixed_covariates`. Time-varying arrays must have
matching lengths. Registered channel roles must match the request exactly.
Unknown fields, test splits and derivative labels are rejected. `pack_split`
exports a `DatasetSplit` without its derivative labels. Upstream loading remains
responsible for verifying these channels against the public manifest.

Preparation serializes the actual public arrays. It hashes their content as
well as the request, initialization plan, lowered candidate, full settings and
Python source tree. Reusing a loader fingerprint cannot conceal changed arrays.
Numerical library/Python versions are recorded when execution begins. Preparing
on one machine and executing on another is allowed only with matching source
and settings; runtime versions remain visible and cross-runtime numerical
equivalence is not assumed.

## Explicit numerical profiles

| Profile | Existing implementation and settings |
| --- | --- |
| `general-rollout-v1` | `fit_candidate`: one bounded nonlinear start, Radau at 1e-7/1e-9, 240 optimizer evaluation ceiling, 300 s cooperative deadline; derivative regression and nonlinear initializer off |
| `collocation-feasible-v1` | `fit_collocation_forward_sensitivity`: 120 s initializer + 180 s refinement, 240 evaluation ceiling, `rollout_or_observed`, 5 s warmup, `ftol=None`, `recovery_policy=feasible`, at most 10 designed starts and 10 s probes, diagnostics enabled; Radau at 1e-7/1e-9; existing automatic piecewise polling policy |

All default settings are materialized in `freeze.json`; partial initializer
guesses are merged with the existing training-only role starts for C+S.
`inspect` checks C+S expression certification without performing optimization.
Its current transfer adapter still supports only target `v01`. A multi-target
request returns `capability_unsupported`; a caller must not reinterpret that as
model invalidity, poor fit, or permission to switch backends. Multi-target C+S is
a separate, still pending integration milestone.

These are cooperative numerical deadlines, not an outer process watchdog.
Symbolic setup, imports and final scoring can add time. Use the scheduler's
ordinary CPU job limit for cluster execution; do not describe 300 seconds as a
guaranteed total process wall time. This interface does not change old timing or
replay policies.

## Execution and results

Python API in `autoformalism.fitting.public_fitting`:

```python
prepare_fit(request, training, validation, directory)
inspect_fit(directory)
result = execute_fit(directory)
```

Preparation accepts `DatasetSplit` or `PublicSplit`. The directory is dedicated
to one exact request. A lock prevents concurrent execution, `started.json`
records consumption before the backend call, and publication is atomic. A
terminal result is returned unchanged on resume. If a previous attempt started
but has no terminal result, resume reports `interrupted`, preserves all files
and grants no fresh numerical budget. Manual recovery of those checkpoints would
need a separately authorized experiment; the wrapper does not resume IPOPT.

`execute_fit` returns a typed `PublicFitResult`. On disk `result.json` contains
`{"result": ..., "sha256": ...}`; `backend_result.json` retains raw diagnostics
and is bound by its digest in the public result. Resume verifies these digests
and exact request/data/lowering/source lineage before returning anything.

Public statuses are `complete`, `fit_failed`, `capability_unsupported`, and
`interrupted`. Invalid request schemas or changed sealed inputs raise an error
before optimization. Runtime exceptions become terminal `fit_failed` results.
CLI exit zero means a terminal status was published, **not** that fitting or
scientific acceptance succeeded; controllers must read `status` and metrics.

`complete` means a parameter vector with complete finite train and validation
rollout scores. Scores concern only declared target channels. Failed trajectories
make the affected metrics unavailable; internal penalty residuals never become
public NMSEs. Poor finite NMSE remains visible without a scientific verdict.
The result also reports native convergence when attributable, budget exhaustion,
call count, and exact lineage hashes. For C+S, native convergence is currently
`null`: its best retained point may come from several stages, and the adapter's
wrapper success placeholder is not selected-point convergence evidence.

Both adapters score production rollouts but do not guarantee a paired BDF/Radau
check. Accordingly `independent_replay` is always `not_performed`. This milestone
does not turn numerical consistency, finite scores or solver success into a
claim of scientific correctness.

## CPU commands

No Delta rerun is needed for the closed comparison. Local analytical handoff
smokes exercise both existing backends, a causal initial map and unchanged resume:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src
.venv/bin/python scripts/smoke_public_fitting.py \
  --profile general-rollout-v1 --output /tmp/public-fit-smoke-general-v1
.venv/bin/python scripts/smoke_public_fitting.py \
  --profile collocation-feasible-v1 --output /tmp/public-fit-smoke-collocation-v1
```

Run these from a checkout of the committed version. Use a new output path when
source changes. The C+S smoke requires the existing CasADi optional dependency.
On Delta, run in an allocated CPU task with the corresponding installed Python;
do not install or fit on a login node. These are small analytical controls, not
evidence of improved recovery on the difficult benchmark.

Once Orion exports a verified public request and development split files, the
CPU handoff uses:

```bash
python scripts/run_public_fitting.py prepare \
  --request REQUEST.json --training TRAIN.json --validation VAL.json \
  --output FIT_DIRECTORY
python scripts/run_public_fitting.py inspect --output FIT_DIRECTORY
python scripts/run_public_fitting.py run --output FIT_DIRECTORY
```

Uppercase paths are exporter outputs/placeholders, not an instruction to rerun
an old root. The next public pilot needs concrete exported artifacts and a pinned
job manifest before submission. Orion owns their construction/repair provenance
and the downstream controller/judge integration; Astra owns this numerical
boundary and its capability tests.
