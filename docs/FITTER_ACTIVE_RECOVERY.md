# Active-dynamics fitter recovery

This milestone tests whether the corrected diagnostic fitter can recover a
nonconstant response when its equation family is known to contain the generating
system. It does not change the proposer, production optimizer defaults, benchmark
tables, or finalized prompts. The user submits the CPU jobs on Delta; ACES remains
available for Sol's proposer experiments.

## What the preceding diagnostics established

| Issue | Current evidence and remaining scope |
| --- | --- |
| Adaptive integration missed supplied-input changes | Integration now restarts at changes of slope in the supplied piecewise-linear input. Reference, cross-solver, and tolerance checks validate the corrected forcing treatment in the tested cases. |
| Finite differences were too small compared with rollout error | Steps near `1.49e-8` produced unstable derivative estimates and almost unchanged parameters. Larger steps around `1e-4` gave substantially more consistent derivatives and allowed fitting to move. This diagnostic uses scaled forward differences with `h = 1e-4 * max(1, abs(parameter))`, adjusted at bounds. It does not certify every parameter regime. |
| A nonnegative offset excluded negative constant predictions | The explicit offset role permits signed `c`. The paired experiment changed only this domain and improved training NMSE from the zero-prediction baseline, `2.09055317`, to the training-mean baseline, `1`. This was a parameter-contract restriction, not an optimizer defect. |
| Numerical correctness of the earlier Radau endpoint vectors | Both runtime-v2 Radau vectors now have independent BDF verification. The signed-offset all-one fit also passed BDF and Radau refinement. |
| Interpreting optimizer success as a good model | Reports now distinguish optimizer termination, numerical verification, and trajectory fit. Two signed-offset starts collapsed to a nearly constant output around `-3.99518`; optimizer success did not imply useful dynamics. |

One signed-offset endpoint still timed out under BDF, with `tau` approximately
`2.51e-16`. Agreement between Radau tolerances alone does not close that independent
verification gap. A timeout is not evidence that the two solvers disagree.
The tiny positive numerical parameter floor is not a physical lower bound.

The original candidate has structural restrictions that can prevent matching the
observations. Nevertheless, two starts reaching a constant do not prove that the
constant is the globally best fit. Before making further optimizer changes, this
milestone removes structural mismatch as an explanation for poor recovery.

## Frozen equations and data recipe

Both cases use exactly the reviewed signed-offset equation family:

```text
v01 = c + k*f^2/(1+f^2) + m + k_p*p + k_u*u01
m' = -m/tau + k_u*u01
p' = -p/tau_p + m
f' = k*v01^2/(1+v01^2) - f/tau_f
m(0) = p(0) = f(0) = 0
```

`c` is signed; the three gains are nonnegative and the three time constants
positive. All parameters are global. All latent initial conditions remain fixed.

| Case | c | k | k_p | k_u | tau | tau_p | tau_f |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moderate | -1.4 | 1.1 | 0.4 | 0.65 | 2 | 4 | 1.2 |
| separated | -0.8 | 1 | 1.6 | 0.9 | 0.08 | 5 | 0.6 |

The second case has a substantially faster memory time constant. It is not a
test of the machine-scale time constants encountered in the real-data fits.

The full recipe is `configs/fitter_active_recovery_v1.json`. Every trajectory has
121 observations over `[0, 24]`, sampled at `0.2`. Four training inputs include
positive and negative excursions at different amplitudes, plus a zero-input
control. Two validation inputs use different amplitudes and shifted transitions.
Inputs are continuous piecewise-linear functions; every corner lies on the
observation grid. These fixtures are noiseless and were prescribed before running
the full fitting matrix. No benchmark data or private reference is loaded.

Only time, supplied input, and observed output enter the fitter. Generated hidden
states are retained solely to audit the reference generator; they are not fitting
labels, derivative targets, supplied auxiliaries, or latent-state initializers.
Training statistics define normalization. Validation is evaluated after fitting;
it neither selects an iterate nor changes the recipe. There is no test split.

## Reference and attainability checks

Each case must pass its own guard before any fit can begin:

1. A handwritten RHS and observation equation generate reference trajectories
   independently of the production expression compiler and simulation wrapper.
   The generator integrates each declared forcing segment separately.
2. Radau and DOP853 references at `rtol=1e-11`, `atol=1e-13` must agree. An
   augmented matrix exponential independently checks the linear `m,p` subsystem
   under each linear forcing segment. Maximum discrepancies, divided by training
   target SD, must be at most `1e-7`. The two numerical methods still share SciPy;
   this is not independence from every numerical dependency.
3. The reference training target SD must exceed `0.05`. Each of the direct input,
   `m`, `k_p*p`, and nonlinear feedback contributions must have SD at least 1% of
   target SD. This prevents a nearly constant fixture from counting as an active
   recovery test. It does not establish unique parameter identifiability.
4. The production compiler and simulator, evaluated at the known generating
   parameters, must achieve NMSE at most `1e-10` on both development splits.
   Radau, BDF, and tighter Radau integration must also agree pointwise.

The guard profiles forward and central differences at the generating parameters
and all-one start, using steps `1e-4` and `1e-5` on the first training trajectory.
It logs column sensitivities, singular values, and step disagreement. These are
diagnostics in raw physical parameter coordinates, not an identifiability proof
or a guard acceptance criterion. An unavailable profile is reported explicitly.

## Fitting matrix and success criteria

Every case receives the following three independent starts:

| Start | Purpose |
| --- | --- |
| `all_ones` | Reproduce the ordinary initialization used in the earlier diagnosis. |
| `broad` | A fixed-seed start independent of truth: positive parameters log-uniform on `[0.25, 4]`, and signed offset uniform on `[-2, 2]`. The exact same vector is used for both cases. |
| `near_truth_control` | A deliberately oracle-assisted local control: alternate positive parameters at 85% or 115% of truth and shift `c` by `+0.2`. A pass here is not counted as evidence of ordinary-start robustness. |

Fitting reuses the production rollout residual and bounded least-squares call,
instrumented with one start per task. It uses Radau at `rtol=1e-7`, `atol=1e-9`
and the scaled finite-difference policy described above. Maximum `nfev` is 150;
actual residual calls include Jacobian work and are reported separately.

Each fitted vector receives six full-split replays: train and validation, each
with Radau (`1e-9/1e-11`), BDF (`1e-9/1e-11`), and refined Radau
(`1e-10/1e-12`). Cross-method/refinement differences must be below `1e-6` RMS and
`1e-5` maximum absolute error, in units of training target SD.

Trajectory recovery additionally requires train and validation NMSE at most
`1e-4`, equivalent to RMS error at most 1% of training target SD. Parameter errors
are reported separately and are not a pass condition: distinct parameter vectors
may generate very similar observable trajectories. Prediction SD is also reported
so collapse toward a constant remains visible.

- `complete` means numerical replay verification passed; inspect
  `trajectory_recovered` separately.
- `replay_unverified` means at least one required comparison was unavailable or
  outside tolerance. Inspect the individual replay status and differences.
- `truth_guard_failed` means the case failed its prerequisite or the guard was
  missing; no optimizer ran for a dependent fit.
- `failed`, `worker_failed`, and `timeout` remain visible terminal outcomes.
- Optimizer success is independent of these labels. A timed-out optimizer can
  still leave an accurate, numerically verified best finite evaluated vector.

## Delta execution and checkpoints

Preparation freezes the validated configuration, candidate, parameter starts,
eight-task matrix, numerical package versions, source hash, and launcher hashes.
It does not integrate or fit on the login node. Two guard tasks precede six fitting
tasks; an `afterany` dependency allows the successful case to proceed even if the
other case fails its guard.

Every task requests one CPU, 8 GB, and no GPU, with at most two tasks concurrent.
Each guard has a 300-second work budget plus 60 seconds supervisor grace. Each fit
has 600 seconds plus six independently budgeted 60-second replay checks and 60
seconds grace. The combined worker caps are 114 CPU-minutes. Slurm limits are 10
minutes for guards, 20 for fits, and 5 for summary. These are limits, not measured
runtime predictions.

From a clean checkout pinned to the delivered commit, the user runs:

```bash
AF_REPO_ROOT="$PWD" AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python AF_CONFIG="$PWD/configs/fitter_active_recovery_v1.json" AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-active-recovery-v1 AF_ARRAY_CONCURRENCY=2 bash scripts/hpc/submit_fitter_recovery_delta.sh
```

After completion:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-active-recovery-v1/summary.md
```

`summary.json` contains the full machine-readable record. Per-task directories
hold solver replays, reference checks, optimizer iterations, actual residual-call
logs, parameter values, and prediction arrays. No generated outputs belong in Git.

Reference trajectories and solver replays are individually checkpointed and
hashed. Completed fits and tasks are reused exactly; changed data or runtime
identity is rejected. If interrupted before `fit.json` exists, a fit restarts from
the same frozen start in a new attempt directory, preserving the earlier logs.
This is deterministic restart, not serialization of the optimizer's internal
state; a wall-clock budget can yield a different stopping point across machines.
Terminal failures are not silently retried. A changed protocol or intentional
new run requires a fresh output directory.

The launcher records submission intent before invoking Slurm and records each
returned job ID immediately. An uncertain or partial submission must be reconciled
with the queue manually; rerunning the launcher will not submit duplicates.

## How the results guide the next discussion

| Observed result | Interpretation and next diagnostic |
| --- | --- |
| Truth guard fails | Reference, compilation, input treatment, or numerical verification needs correction; no inference about optimization follows. |
| Near-truth control fails | Local optimization, derivative accuracy, scaling, or budget still needs attention in an attainable active case. |
| Near-truth succeeds, ordinary starts fail | Initialization, local basins, or the allocated search budget is the next issue. This does not demonstrate a structural problem with the fixture. |
| Moderate succeeds, separated case fails | Investigate time-scale separation, sensitivity, stiffness, and solver cost. |
| Both ordinary starts recover both cases | Basic active recovery is demonstrated for these two noiseless fixtures. This strengthens, but does not prove, the structural explanation for the real-data collapse. |
| Trajectories recover but parameters differ | Investigate excitation and practical identifiability before declaring the optimizer broken. |

Noise robustness, much more extreme time constants, endpoint derivative checks,
broader initialization coverage, generalization to other equation families, and
whether a better real-data fit exists remain outside this milestone. No further
fitter policy or model redesign is automatically triggered by the report; results
are reviewed with the user first.
