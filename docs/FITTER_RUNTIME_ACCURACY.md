# Fitter runtime and accuracy milestone

The preceding diagnostic reproduced the near-all-one `xtol` stop and showed
substantial improvement after changing the derivative step or ODE tolerance.
The improved fits then exhausted their 300-second budgets. Their traces put
95–98% of fitting time inside residual evaluation; the late RK45 evaluations
reached 22–30 seconds for all 16 training trajectories. Tighter integration
cost approximately 2.8 times as much at the initial vector. The larger-step
runs followed similar early improvements at both integration tolerances.

This milestone keeps the model, parameter domains and sharing, three fixed zero
initial conditions, public data and all-one optimizer start unchanged. It asks
where runtime is spent, which integrators reproduce the same predictions
efficiently, and whether a parameter-scale floor helps numerical derivatives
after parameters become small. It performs no LLM calls or model redesign and
does not open test tables or private reference equations.

## Implementation

`fitting/numerical.py` adds a bounded one-sided numerical Jacobian. The optional
scaled policy requests the absolute perturbation
`step * max(abs(parameter), scale_floor)`. It switches direction or shortens the
probe when a hard bound requires that. The optimizer's previously evaluated
base residual is reused for derivative construction; trial evaluations remain
physical calls. Finite-difference probes can themselves be retained as the
best evaluated trial and are not described as accepted optimizer iterates.

`FitConfig` retains its default relative/SciPy policy. The new options are
`finite_difference_policy`, `finite_difference_step`, and
`finite_difference_scale_floor`. The scaled policy uses `1e-4` if its step is
omitted. These controls apply to the generic bounded least-squares fitting
branch; affine/projection backends keep their existing algorithms. The common
scale floor of one in this experiment reflects its all-one starting scale;
it is not a claim that every scientific parameter has the same natural units.

On a production fitting timeout, the best finite, integration-valid evaluated
vector is now preserved instead of replacing it with the start of that attempt.
The fit remains failed/incomplete, and its unreplayed metrics retain the
existing failure semantics. Diagnostics explicitly mark retention and count
actual residual evaluations. The existing integer `function_evaluations` stays
zero when SciPy was interrupted before returning its count; actual evaluation
counts are available separately. Completed fits still use SciPy's returned
vector, and unsuccessful integrations cannot become a retained best trial.
If an earlier start completed with a finite non-penalty cost, it keeps precedence
over an interrupted start. The interrupted vector remains in that start's
`retained_variables` diagnostic, so preserving progress does not turn an
otherwise usable multi-start result into a failed fit.

`simulate_trajectory` accepts optional `RolloutProfile` instrumentation for
adaptive free rollouts. It measures preparation, integration, actual RHS calls
and their time, post-rollout state checks, observation construction, forcing
construction and observation-expression evaluation. Solver `nfev`, `njev` and
`nlu` are retained separately. Actual RHS calls can exceed solver `nfev`, for
example when an implicit solver estimates its Jacobian. RHS time is nested in
integration time; observation-expression and forcing times are nested in
observation time. These values must not all be added together.

## Frozen input boundary

The source is the existing Delta directory `fitter-stagnation-v1`. Preparation
verifies its original source-code identity and plan hash, its self-consistent
freeze, and the original pinned model/data snapshot underneath it. It checks
the exact reviewed parameter-vector hashes and consistency of the previous
fit and terminal result. Only the public snapshot and provenance are copied
into the new output. Numerical dependency versions must match the earlier run.

Three vectors are fixed before the experiment: all ones, and the penultimate
and latest completed iteration-callback vectors from `current_large_step`.
The latter two are the intermediate and late vectors in the reviewed trace,
not the slightly perturbed best derivative probe. Their full values remain in
external artifacts; only their hashes are committed in the configuration.
No validation score determines which vectors are profiled.

## First array: three profiling tasks

One task per frozen vector uses the first two training trajectory IDs in
sorted order and normalization fitted on the full training split.

Each task compares RK45, DOP853 and Radau at current tolerances `1e-7/1e-9`
and tighter tolerances `1e-9/1e-11`. Each method/tolerance has two independent
physical repeats. A separately stored repeat key prevents cache reuse from
artificially making repeat differences zero. Two additional reference runs
use DOP853 and Radau at `1e-10/1e-12`: 14 timed cases per vector in total.

A reference must agree with its `1e-9/1e-11` run to within normalized residual
RMS `1e-6` and maximum absolute difference `1e-5`. The prespecified preference
is Radau, falling back to DOP853 if that refinement check is unavailable or
fails. This is an empirical numerical consistency check, not a certified
global integration-error bound.

Candidate method/tolerance repeats are compared pointwise against that
reference. Both repeats must have residual RMS difference at most `1e-5` and
maximum absolute difference at most `1e-4`. Only a method passing its current
tolerance at all three vectors is eligible for the longer fits. There is no
automatic choice of a fastest or best-fitting solver: all eligible methods
proceed, and every skipped method remains in the ledger.

At each vector, additional derivative probes compare relative `1e-4`, scaled
`1e-4`, and scaled `1e-5` steps using the reference method at `1e-9/1e-11`.
Actual perturbations, column norms, gradients, conditioning and differences
from the scaled `1e-5` Jacobian are recorded. Agreement between step sizes is
diagnostic and does not prove derivative accuracy or identifiability.

A separate intrusive Python profile uses RK45 at current tolerance on the
first selected training trajectory. Its per-function self/cumulative times
help distinguish expression evaluation, forcing and other Python overhead.
That execution is excluded from the runtime comparison repeats. Optional
profiling adds overhead; two repeats and two trajectories provide a bounded
engineering diagnosis, not a general performance benchmark.

Each profile gets a 600-second shared numerical budget, a 60-second bound on
each residual evaluation, and a 60-second supervisor margin. Cases that fail
or exhaust a budget retain partial timing/call records. Successful numerical
arrays and derivative points are independently checkpointed with digests.

## Second array: six matched fits

The fit array depends on completion of the profile array. Each task checks the
accuracy evidence before executing its numerical fit.

| Method | Existing relative step | Scaled step with floor one |
| --- | --- | --- |
| RK45 | `diff_step=1e-4` | absolute probe `1e-4 * max(1, abs(parameter))` |
| DOP853 | `diff_step=1e-4` | same scaled policy |
| Radau | `diff_step=1e-4` | same scaled policy |

Every arm starts independently at all ones, uses all 16 training trajectories,
current ODE tolerances, one start, at most 100 SciPy `nfev`, and a 900-second
fitting allowance. The method and derivative policy are the only treatments.
The scaled Jacobian reuses the base residual, so it does not add an extra
baseline rollout for each Jacobian. Every physical call and iteration callback
is logged. A callback occurs after Jacobian construction; a missing callback
does not establish that an improving base vector was rejected.

After fitting, DOP853 and Radau independently replay the same retained vector
on all training and validation trajectories at `1e-9/1e-11`, with 120 seconds
per replay. DOP853 supplies the main displayed scores. Both replays must be
finite, and their train and validation NMSE differences must each be at most
`1e-5`, for the task to receive `complete` status. Otherwise the scores remain
visible with `replay_unverified`. This final gate compares aggregate scores;
the earlier fixed-vector gate compares residuals pointwise. Neither status
certifies optimizer convergence or scientific adequacy.

The hard fit-worker cap is 1,200 seconds including both replays and a margin.
The nine worker caps total 153 CPU-minutes, excluding preparation and summary;
actual time can be smaller. There are no GPUs. Profiling and fitting run as
sequential arrays with default concurrency two, so no more than two one-CPU
workers run simultaneously unless the user changes that setting.

## User-run Delta commands

Codex implements, tests, commits and pushes. The user submits the experiment
and returns the report for discussion before further changes. No monitor or
background submission is enabled.

```sh
cd /projects/bibo/yxiao2/repos/autoformalism-v21
git fetch origin codex/fitter-runtime-accuracy
git worktree add --detach ../autoformalism-fitter-runtime-v1 FETCH_HEAD
cd ../autoformalism-fitter-runtime-v1
bash scripts/hpc/submit_fitter_runtime_delta.sh
```

For the exact release, replace `FETCH_HEAD` with the pushed commit in the
handoff. Keep this checkout and the previous diagnostic checkout unchanged.
The submitter verifies the reviewed inputs before dispatch. It requests one
CPU and 8 GB per task, 15 minutes for profiles, 25 minutes for fits, and a
dependent five-minute summary job. `AF_ARRAY_CONCURRENCY` accepts 1–6 and
defaults to two. Other overrides are `AF_PYTHON`, `AF_SOURCE_ROOT`,
`AF_OUTPUT_ROOT`, and `AF_CONFIG`.

The default Python is
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`, source is
`/work/hdd/bibo/yxiao2/phase_b/fitter-stagnation-v1`, and output is
`/work/hdd/bibo/yxiao2/phase_b/fitter-runtime-v1`.

After the summary job finishes, paste the report:

```sh
cat /work/hdd/bibo/yxiao2/phase_b/fitter-runtime-v1/summary.md
```

The submission manifest records each successful scheduler submission before
attempting the next one. Re-running the submitter never duplicates recorded
jobs. If a later submission fails, reconcile the existing jobs before adding
that missing stage. An unresolved `submission.intent` also requires queue
reconciliation. The summary can be regenerated after jobs finish without
repeating numerical work:

```sh
PYTHONPATH=src /projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python \
  scripts/run_fitter_runtime.py summarize \
  --output /work/hdd/bibo/yxiao2/phase_b/fitter-runtime-v1
```

Completed fits, replays, profiling cases and derivative points are reused only
under their frozen identities. Partial optimizer state is not serialized:
an interrupted unfinished fit restarts from the same all-one vector in a new
trace directory. Wall-clock interruption can change how far a restarted fit
gets. Terminal failures remain terminal and visible; changed settings or
deliberate retries require a new frozen run. No baseline data, benchmark
prompts or generated model artifacts are committed with this milestone.
