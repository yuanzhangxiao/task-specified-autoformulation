# Fitter stagnation diagnostic

Delta job 21852272 barely moved the selected anonymous model's seven parameters
from their all-one start: the largest change was 0.4283%. Training replay NMSE
changed from 3.34973 to 3.34478, and validation NMSE from 2.18687 to 2.18323.
The first two starts stopped on `xtol` after 11 reported `nfev` each; the third
exhausted the shared time budget. All final train and validation trajectories
were finite. These observations establish poor progress, but do not establish
its cause or show that the optimizer is generally broken.

This experiment asks whether numerical derivative estimates are unreliable at
the default perturbation size, and whether changing that size or the ODE error
tolerances permits useful progress. It changes no production fitter defaults
and does not redesign the generated equations, parameter domains, parameter
sharing or three fixed zero latent initial values. No LLM is called.

## Frozen comparison

The candidate is `functional_fcdf45e854663fbd`, previously selected by public
scientific review. Preparation copies only the already frozen candidate,
function provenance, public prompt, manifest, training and validation tables
from `staged-fit-v1-2f4d10c`. The configuration pins the original freeze and
terminal-result byte hashes. Every copied asset is verified, and Python and
numerical dependency versions must match the original run. The new source,
launchers and assets are frozen before numerical work. Test tables and private
references are not opened; validation is reported only after fitting.

Four optimizer arms use the same all-one vector, full 16-trajectory training
split, training scaler, parameter bounds, least-squares method and stopping
tolerances. Each gets one start, at most 50 `nfev` and 300 seconds. Using one
start deliberately removes random restarts from this diagnosis; this is not
an exact rerun of the previous three-start budget.

| Array index | Arm | ODE rtol / atol | SciPy `diff_step` |
| --- | --- | --- | --- |
| 0 | Sensitivity profile | Both settings below | Separate forward/central scan |
| 1 | `current_default` | 1e-7 / 1e-9 | Default |
| 2 | `current_large_step` | 1e-7 / 1e-9 | 1e-4 |
| 3 | `tight_default` | 1e-9 / 1e-11 | Default |
| 4 | `tight_large_step` | 1e-9 / 1e-11 | 1e-4 |

All four fitted vectors are independently rescored on the complete training and
four-trajectory validation splits using the same tighter ODE tolerances and a
60-second replay allowance. A split receives no aggregate NMSE if any trajectory
fails. Tighter integration costs more per residual call, so equal wall time need
not produce equal iteration counts. This is a fixed-budget numerical diagnosis,
not a demonstration of asymptotic optimizer convergence.

The sensitivity profile uses the first two training trajectory IDs in sorted
order, selected without inspecting outcomes, and the full training scaler. It
examines both the all-one and previously fitted vectors at both ODE tolerances.
For each parameter, forward and central derivative estimates use step factors
approximately 1.49e-8, 1e-6, 1e-4 and 1e-2. Actual profile perturbations are
`factor * max(1, abs(parameter))`; SciPy's explicit `diff_step` instead scales
relative to the parameter itself. They agree at the all-one anchor. The profile
also evaluates each anchor twice under distinct cache keys to measure exact
repeat differences without reusing the first observation.

## Evidence and interpretation

Every physical optimizer residual call records its parameter vector, residual
norm, cost, duration and integration failures. This includes numerical derivative
probes omitted from SciPy's `nfev` count. Accepted-iteration callbacks record
cost, parameter movement and `nfev`. Normal completion records the stop reason,
gradient, scaled optimality, active bounds and Jacobian diagnostics. The profile
records column norms, singular values, gradients, forward/central disagreement,
adjacent-step disagreement and residual changes between integration tolerances.

Raw-coordinate Jacobian rank and conditioning are descriptive and depend on
parameter scales. They do not establish structural identifiability. Identical
repeat residuals do not rule out deterministic numerical integration error.
The two-trajectory profile may not represent every direction in the full
training objective. Larger perturbations also introduce truncation error.

Evidence for a numerical derivative problem would combine substantial
step/tolerance sensitivity with improved progress under the corresponding
optimizer treatment. A large gradient at an `xtol` stop would argue against
calling that stop a stationary solution. If the estimates stabilize but the
fit still stalls, the traces can distinguish expensive rollouts, poor scaling,
correlated directions or insufficient budget. None of these results alone
proves that the equations can fit the data well. Model redesign remains a
separate decision after this diagnosis.

On a soft fitting timeout, the diagnostic retains the best finite evaluated
trial, explicitly labeled `best_finite_evaluation_after_timeout` with
`optimizer_success: false`. This trial may be a derivative probe. Normal
completion uses the optimizer's returned vector. Neither a finite replay nor
a `complete` task status means optimizer convergence. Trace and checkpoint
writes add some wall-clock overhead, consistently in all four arms.

## Checkpoints and execution bounds

Completed fits and replays are reused after interruption. Profile residual
arrays have individually checked identities and digests and are reused by
point. Distinct anchor-repeat keys preserve the original independent calls.
An interrupted optimizer restarts from the same initial vector in a new
`attempt-*` directory; it does not restore SciPy's internal state. Existing
traces are retained. Terminal failures remain terminal and visible; changed
settings or a deliberate retry require discussion and a separately frozen run.

The profile gets 600 seconds plus a 60-second supervisor margin. Each fit task
gets 300 seconds, a 60-second replay allowance and a 60-second margin. The hard
worker caps total 39 CPU-minutes across the five tasks, excluding preparation
and summary. Each Slurm array task requests one CPU, 8 GB, no GPU and 15 minutes.
The default concurrency is two, adjustable with `AF_ARRAY_CONCURRENCY=1` through
`5`. A dependent five-minute summary job runs after all tasks terminate.

## Commands for the user on Delta

Codex prepares, tests and pushes this implementation. The user runs the cluster
commands and returns the results for discussion before another implementation
or experiment milestone. No background submission or monitoring is enabled.

```sh
cd /projects/bibo/yxiao2/repos/autoformalism-v21
git fetch origin codex/fitter-stagnation-diagnostic
git worktree add --detach ../autoformalism-fitter-stagnation-v1 FETCH_HEAD
cd ../autoformalism-fitter-stagnation-v1
bash scripts/hpc/submit_fitter_stagnation_delta.sh
```

Use the exact pushed commit supplied with the handoff in place of `FETCH_HEAD`
when reproducing that release. Keep this checkout unchanged while jobs run.
The submitter verifies inputs before submitting; it prints array and summary
job IDs in a submission manifest. Defaults use the existing interpreter at
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`, source snapshot at
`/work/hdd/bibo/yxiao2/phase_b/staged-fit-v1-2f4d10c`, and new output directory
`/work/hdd/bibo/yxiao2/phase_b/fitter-stagnation-v1`. Overrides are `AF_PYTHON`,
`AF_SOURCE_ROOT`, `AF_OUTPUT_ROOT`, `AF_CONFIG` and `AF_ARRAY_CONCURRENCY`.

After the summary job finishes, paste this report into the discussion:

```sh
cat /work/hdd/bibo/yxiao2/phase_b/fitter-stagnation-v1/summary.md
```

`summary.json` contains machine-readable rows. Detailed fit diagnostics,
profile results and evaluation traces are under `results/<arm>/`. Missing,
failed and timed-out arms remain in the summary rather than being excluded.
If the dependent summary submission fails, the successful array submission is
already recorded and will not be duplicated by rerunning the submitter. After
the array finishes, the summary can be generated without another fit:

```sh
PYTHONPATH=src /projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python \
  scripts/run_fitter_stagnation.py summarize \
  --output /work/hdd/bibo/yxiao2/phase_b/fitter-stagnation-v1
```

An unresolved `submission.intent` without a manifest requires checking the queue
before retrying; the submitter intentionally will not risk a duplicate array.
There is no ACES workload in this CPU diagnosis. Sol's independently frozen
function-batching comparison supplies the next ACES commands.

## Local verification

The new tests cover exact production-residual parity, real SciPy recovery of a
known synthetic parameter, actual residual-call accounting, derivative-scan
cache reuse, failed-integration exclusion, timeout behavior, all five campaign
tasks, source/runtime/launcher drift rejection, absent test tables, completed
phase reuse, and a real prepare/run/summarize/resume CLI smoke. Run the complete
pytest suite, Ruff and relevant construction/CLI smoke checks before handoff.
