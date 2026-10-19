# M43: dependent initializers and prefix-conditioned trajectory roots

Implementation milestone, 19 October 2026. The shared construction/fitting contract
is [PHASE_C_OBSERVATION_PREFIX_INITIALIZATION.md](PHASE_C_OBSERVATION_PREFIX_INITIALIZATION.md).
This small CPU qualification enables both agreed initialization options behind a
separate adapter. It does not change benchmark prompts, data or existing campaign
policies, and it makes no LLM calls.

## Two controls, two starts each

Both controls use the known equations

    dx/dt = -theta*x + z + r
    dz/dt = -z
    dr/dt = -2*r
    y = x.

The measured state starts at x(0)=y(0). The initializer graph declares one root
r(0) and the dependent map z(0)=phi*r(0). Theta and phi are shared unknown
coefficients. Fixed coefficients in these synthetic equations define the control;
they are not inferred from hidden benchmark labels.

- **Shared root:** one r(0) is learned across three training trajectories. New
  trajectories reuse it and the learned phi. No new root optimization is needed.
- **Trajectory root:** each training trajectory has its own independent r_j(0).
  All trajectories have the same y(0), so a shared time-zero initializer cannot
  express their different hidden preparations. New roots are inferred from a
  fixed one-unit prefix; theta and phi remain frozen.

Theta has the declared domain [0.1, 0.8], avoiding exchange with the fixed
latent decay rates; the root and phi use nonnegative coefficient domains. The
truth for evaluation is theta=0.35 and phi=0.7. Shared roots are 0.8; trajectory
roots are (0.4, 1.2, 1.6) in training and (0.65, 1.45) in validation.

Each control has two predetermined optimization starts, giving four tasks. The
sampling grid runs from 0 to 4 in steps of 0.1, and the prefix ends at 1. Only
samples strictly later than 1 contribute to forecast scores. Every training row
also receives a fresh prefix inference to distinguish training reconstruction
from prefix-conditioned forecasting. Full-training fitted roots are not used as
prefix starting guesses.

The generative values and analytic trajectories are evaluator-only diagnostics.
The fitting functions receive declared equations, initial guesses and measured
training rows. Two distinct known latent decay rates help separate the root and
map coefficient; this is an implementation qualification, not a claim that
arbitrary latent models are identifiable.

## Numerical path and acceptance evidence

The graph is validated and expanded with the restricted expression grammar.
Forward sensitivities propagate through initializer dependencies. Joint fitting
uses shared columns for theta/phi/shared roots and one coordinate block per row
for trajectory roots. Optimization reuses bounded direct rollout least squares
and guarded coordinate scaling. Prefix inference optimizes roots only, from a
typed data object containing no future readings. Normalization is frozen from
training.

Retained points undergo a second integration-method check. Forecasts use only
initial permitted measurements and declared external schedules. Observations
strictly after the prefix enter scoring only. The report separates numerical
verification, prediction accuracy, parameter/root evaluator errors and local
sensitivity rank. Numerical verification does not mean a globally optimal fit;
full local rank is not a confidence probability or proof of global uniqueness.

The first version does not implement collocation with these graphs, profiling of
trajectory roots, noisy observed-boundary estimation, or implicit initializer
cycles. Production construction integration follows after this qualification.

## Budgets, retention and resume

Each numerical fit/inference has at most 100 search evaluations and 30 search
seconds, followed by a separately recorded ten-second verification allowance.
The frozen prefix and numerical policies are part of the experiment identity.
Each independent stage has durable started/progress/result records. Completed
operations are reused; interrupted operations remain visible and charged without
an automatic budget reset. Native optimizer-state continuation is not claimed.
Reporting may be rerun without repeating fitting. Changed source, runtime,
policy or input identity cannot silently reuse the run.

The four array tasks request one CPU each, 16 GB and ten minutes of scheduler
wall time, with concurrency two by default. No GPU is used. Scheduler wall time
includes Python imports, verification and persistence; it is not a target solver
duration. This run needs no downloaded model archive or prior campaign root.

## Run on Delta

Upload `phase-c-fitting-m43.tar.gz` to the assigned AF_BUNDLE path. Use a fresh
code directory so an older campaign's identity is preserved.

```bash
AF_BUNDLE=/work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m43.tar.gz
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/initialization-m43.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_CODE"
cd "$AF_CODE"
export AF_INITIALIZATION_ROOT=/work/hdd/bibo/yxiao2/phase_c/initialization-qualification-v1
export AF_CONCURRENCY=2
bash scripts/hpc/submit_phase_c_initialization_qualification_delta.sh
```

The wrapper uses the existing `/projects/bibo/yxiao2/venvs/autoformalism-v21`
environment and `/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps` dependencies.
Set `AF_PYTHON` or `AF_CASADI_ROOT` only if those installed paths have changed.

After completion, from the same extracted code directory:

```bash
export AF_INITIALIZATION_ROOT=/work/hdd/bibo/yxiao2/phase_c/initialization-qualification-v1
export PYTHONPATH="$PWD/src:$PWD:/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps"
AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
"$AF_PYTHON" scripts/phase_c_initialization_qualification.py report --root "$AF_INITIALIZATION_ROOT"
jq '{status, expected, recorded, qualification_passed, status_counts, training_reconstruction, prefix_forecasts, cost}' "$AF_INITIALIZATION_ROOT/summary.json"
```

Expected terminal coverage is four tasks. Inspect numerical qualification and
forecast metrics separately from terminal completion. Share `summary.json` plus
the plan, inputs and per-task artifacts for review; none contain real test data.

To download a portable review after reporting:

```bash
AF_REVIEW="$AF_INITIALIZATION_ROOT/review-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_INITIALIZATION_ROOT" \
  plan.json inputs.json summary.json results
printf '%s\n' "$AF_REVIEW"
```

## Local verification

The graph/kernel regression set covers both root scopes, DAG chain derivatives,
cycles and invalid references, native parameter domains, unchanged shared
parameters, prefix-forcing consistency, poisoned suffix observations, a
zero-duration ambiguous prefix, and independent forecast integration failure.
The qualification also checks sealed resume and interruption accounting. Remote
CPU results remain to be collected; local verification is not a new benchmark
performance result.

The four-task local CLI smoke completed all four qualifications. Training
reconstruction NMSE ranged from 5.07e-20 to 9.75e-20; all 20 prefix-conditioned
forecasts had NMSE below 1.98e-20. Maximum independent analytic forecast difference
was 7.92e-10, maximum normalized DOP853/Radau difference 2.33e-9, and all dependent
initializer equalities held. Evaluator-only shared coefficient and root errors
were below 6.50e-10. Local sensitivity ranks were 3/3 for shared-root training
and 5/5 for trajectory-root training. These clean known-equation controls are
intentionally easy; their precision is an interface check, not a realistic
uncertainty claim.

Recorded numerical work was 85 completed search calls and 88 completed
verification oracle calls, with 15.85 CPU / 16.33 summed operation wall seconds.
Oracle calls can contain several trajectory integrations, so these are not RHS
function-evaluation counts. Separate evaluator calls and artifact timing remain
in the saved operation ledger. Local artifacts are in
`output/phase-c-fitting-m43-smoke/` and are not committed.

Verification on the final implementation: 123 focused graph, prefix, campaign,
sensitivity and legacy public-fitting tests passed. The broader parallel suite
reported 6,909 passed, eight optional Torch-dependent skips, and two source-digest
failures because implementation files changed while that suite was running. With
source stable, all 35 tests in the affected `test_measurement_campaign.py` module
passed. No guard was bypassed. All four terminal smoke tasks then replayed without
changing the bytes or timestamps of their 148 JSON artifacts.

Changed files pass Ruff and shell syntax checks; changed Python parses with
Python 3.11 syntax for Delta. Full `ruff check .` reports 37 pre-existing findings
in unrelated, untracked `analysis/claude` scripts; the rest of the repository is
clean under `ruff check . --exclude analysis`.
