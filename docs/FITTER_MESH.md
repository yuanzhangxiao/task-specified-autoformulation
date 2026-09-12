# Collocation setup, mesh and handoff comparison

This milestone is opt-in. It does not change the proposer contract or the default
fitter used by an existing experiment. It uses the frozen public candidates from
`fitter-feasibility-v1`; no LLM calls, test data, or latent reference labels enter
optimization. The public candidates are proposed models for a benchmark, not
known-correct generating equations.

## Changes

1. With `recovery_handoff="best_valid"`, subsequent sensitivity attempts and
   directional polling receive the best finite full-training parameters found so
   far, including fitted initial-condition parameters. Earlier screened starts
   remain alternatives. Poll handoffs and their training costs are saved.
2. With `sensitivity_invalid_trials="reject"`, a failed trial integration after a
   valid augmented evaluation returns an infinite residual. SciPy's bounded TRF
   optimizer then shrinks the trial radius. No zero Jacobian is returned. A failed
   first augmented evaluation still triggers the explicit recovery route. All
   stages share the existing residual-call and wall-clock budget.
3. With `collocation_assembly="mapped"`, CasADi `Function.map` applies one Radau
   interval function across the intervals of a trajectory. This is serial graph
   assembly/evaluation on one CPU. With the full mesh, the equations, objective,
   variable order, and derivatives agree with the existing interval loop.
4. `collocation_target_variables` sets a global target for parameters plus state
   values at collocation nodes. Nodes can now be fewer than observations. Every
   observation remains in the objective: the state polynomial is evaluated at its
   time, and then the observation function is applied. This preserves nonlinear
   observation functions and measured observed-state initial boundaries.

Every corner in the supplied piecewise-linear forcing is mandatory. The mesh
allocator deterministically bisects the largest relative-time gaps. If input
corners alone exceed the target, it reports that fact and retains the inputs.
The target is therefore not a guaranteed hard cap. Shared fitted latent initial
values and parameterized initial maps keep their existing semantics.

Collocation state values are temporary optimization variables. Final predictions
always come from an ODE rollout on the original observation grid, followed by
independent Radau/BDF verification. Coarser collocation can introduce approximation
error; final ODE replay alone does not prove that the initializer is mesh-converged.
The two mesh targets and the full mesh provide a resolution comparison.

## Delta experiment

With the four frozen public candidates, there are 47 tasks:

| Arm | Construction | State mesh | Handoffs/trial rejection | C budget |
|---|---|---|---|---|
| previous | Original loop | Full | Previous behavior | 120 s / 150 iterations |
| handoff | Original loop | Full | Best valid / reject failed trials | 120 s / 150 iterations |
| mapped | CasADi map | Full | Best valid / reject failed trials | 120 s / 150 iterations |
| mesh12000 | CasADi map | Target 12,000 variables | Best valid / reject failed trials | 120 s / 150 iterations |
| mesh24000 | CasADi map | Target 24,000 variables | Best valid / reject failed trials | 120 s / 150 iterations |
| more_time | CasADi map | Full | Best valid / reject failed trials | 300 s / 1,000 iterations |

Every arm has the same subsequent refinement budget: 180 seconds and 240 calls,
including feasibility screening. Piecewise cases consistently use the existing
branch-aware initialization and directional polling policy. The first three arms
run on all ten cases. The mesh arms run on the four public and two matched cases.
The larger-budget arm runs on the four public cases and matched quadratic control.
It is a separate budget reference, not an equal-budget speed comparison.

The controls are quadratic and scaled four-state systems, two piecewise cases,
and quadratic/scaled controls sharing the first public candidate's trajectory
counts and time grids. Matched controls use independent stable reference dynamics
and simple constant inputs. They match computational size, not public forcing
complexity or nonlinear difficulty. Comparing them with small controls tests size;
comparing full/mapped construction within each case tests overhead; comparing mesh
resolutions tests the cost/accuracy trade-off. Differences between public and
synthetic cases alone cannot identify a particular equation as the cause.

## Run

Use a clean, pinned checkout on Delta. Set these variables from its root:

```bash
export AF_REPO_ROOT="$PWD"
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-feasibility-v1
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-mesh-v1
export AF_ARRAY_CONCURRENCY=2
bash scripts/hpc/submit_mesh_delta.sh
```

The launcher freezes input hashes/code/runtime, runs a smoke job, then runs the
array only if smoke passes. Each task requests one CPU, 16 GB, 30 minutes, no GPU.
Two concurrent tasks are the default. The summary runs after the array finishes.
Preparation only freezes data and generates independent controls; optimization
runs in compute jobs. ACES is not needed.

After completion:

```bash
cat "$AF_OUTPUT_ROOT/summary.md"
```

If the summary job is interrupted, regenerate it without numerical dependencies:

```bash
"$AF_PYTHON" -S scripts/run_mesh_campaign.py summarize --output "$AF_OUTPUT_ROOT"
```

`diagnostics.json` contains construction time, first/last C iteration, variable
counts, exact mesh boundaries and input-fidelity audit, checkpoint parameters,
screening costs, sensitivity/poll stages, failed-trial counts, and final parameters.
`summary.json` retains all results and independent replay checks. The summary
separates native C success, finite verified final fit, and synthetic recovery
(both clean train and validation NMSE <= 1e-4). Public cases have no recovery label.

## Resume and incomplete jobs

The launcher preserves an existing submission and blocks ambiguous partial
submissions. Reconcile `submission.json` with the queue before retrying. To resume
one inactive array element with the same frozen checkout/environment:

```bash
export AF_CODE_COMMIT="$(git rev-parse HEAD)"
sbatch --array=INDEX --export=ALL \
  --output="$AF_OUTPUT_ROOT/logs/resume-%A_%a.out" \
  --error="$AF_OUTPUT_ROOT/logs/resume-%A_%a.err" scripts/hpc/mesh_delta.slurm run
```

Replace `INDEX` with its task index from `freeze.json`. Completed results are
reused. Saved fits and replays are reused if later replay/reporting was interrupted.
An interrupted native optimization cannot resume its internal solver state:
its checkpoint is retained and the attempt becomes `interrupted`, without a fresh
optimization budget. A fresh scientific attempt requires a new output directory;
do not delete `fit_started.json` or overwrite previous evidence.
