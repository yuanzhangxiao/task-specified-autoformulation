# Construction portfolio: targeted parameter controls

3 November 2026. This opt-in CPU campaign starts from the four authenticated
endpoints of `construction-portfolio-checkpoint-v1`. It keeps their equations,
training data, domains and independent original-equation verification. It tests
specific parameter boundaries and numerical coordinates before another broad
continuation or equation-revision recommendation.

## Questions being tested

- **Model 12:** its meal-state decay approached zero. Compare the exact allowed
  zero boundary with positive decay values while refitting nuisance variables.
  A fixed-decay diagnostic is conditional on that value; it is not proof of a
  global optimum or a calibrated confidence interval.
- **Model 15:** compare existing coordinates with an equivalent positive-rate
  parameterization based on the meal state's contribution to glucose. Compare
  matched starts and allowances, with predetermined alternative decay starts.
  The inverse must recover finite original coefficients and all original
  trajectory initial values. Zero absorption is a separate original-model case;
  it must not acquire a nonzero forcing gain or scaled initial state through a
  singular transformation.
- **Models 10/14:** compare saturation scales chosen from public training glucose
  observations. Hold the selected saturation scale while refitting nuisance
  parameters with residual-only search. This does not use a ground-truth
  coefficient or assert that a different equation is needed.

The source endpoints remain independent incumbents. Candidates are retained only
after original-equation verification and the existing improvement checks; ties
or failed trials cannot erase a verified source endpoint. Search stopping,
numerical qualification and scientific model adequacy remain distinct.

This campaign uses training reconstructions with training-estimated initial
roots. It does not produce new causal-prefix forecasts, access validation or
test data, call an LLM, revise a model, or change production-fitting defaults.

## Resources and execution identity

The frozen roster contains **20 tasks**. Here `T` is the maximum public training
trajectory duration and `S` is the median positive absolute public training
glucose observation:

| Model | Tasks | Predetermined controls |
| --- | ---: | --- |
| 12 | 5 | Free warm continuation; fixed decay `0`, `0.1/T`, `1/T`, `3/T` |
| 15 | 7 | Existing versus flux coordinates at the saved decay, `0.1/T`, and `1/T`; exact original zero-absorption control |
| 10 | 4 | Fixed saturation scale at the inherited value, `0.1*S`, `S`, `10*S` |
| 14 | 4 | Fixed saturation scale at the inherited value, `0.1*S`, `S`, `10*S` |

Model 15's matched positive starts preserve `g = k_abs * k_abs_rate` and each
`B(0) = k_abs * A(0)` when changing the starting decay. Every profiled search
has at most **1,800 numerical seconds and 2,000 objective attempts**. The
saturation controls have at most **600 numerical seconds and 250 attempts**.
Preparation, qualification and independent verification have separately
recorded allowances. Unsupported certification must fail explicitly, not
silently substitute a different model or start.

The new coordinates change the shared-parameter outer search. The implementation
continues to propagate the original ODE and solve its original initial-root
least-squares problem; it records the equivalent scaled initials as an audit.
Certification checks the exact reservoir/output expressions, exclusive symbol
uses, independent initial roots and nonnegative unbounded domains. It does not
infer equivalence from parameter names alone. Positive decay is represented
logarithmically without adding an artificial positive lower bound.

For this frozen training split, `T=300` and `S=172.6891611318373`. The original
checkpoint losses are 1.712229230, 0.436354216, 1.688704861 and 0.438459830 for
models 10/12/14/15. These are historical training reconstructions, not expected
outcomes for the new trial starts.

Each array task requests **one CPU and 8 GB**, with **at most two tasks running
concurrently**, a four-hour worker ceiling, and a thirty-minute dependent report.
The task roster and stage allowances are frozen in `plan.json`. The launcher
rejects a per-task reservation ceiling above 13,200 seconds, leaving scheduler
headroom inside the four-hour job. A wall-clock ceiling is not a promise that a
fit will use all of that time.

The execution seal binds the plan, submitted source tree, Python interpreter,
scientific-package identities, worker entrypoints, account and resource limits.
Keep the submitted repository directory unchanged until the run finishes.
`AF_PORTFOLIO_TARGETED_CONCURRENCY=1` enables serial execution if set before
submission; changing it afterward changes the execution identity.

## Submit on Delta

Extract the supplied committed code bundle into a new repository directory and
run the following **from that populated directory**. Do not merely create an
empty directory and change into it. The source campaign must already exist on
Delta; the code bundle does not include experiment data or historical results.

```bash
export AF_PORTFOLIO_TARGETED_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-checkpoint-v1
export AF_PORTFOLIO_TARGETED_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-targeted-v1
bash scripts/hpc/start_construction_portfolio_targeted_delta.sh run
```

`run` authenticates source receipts, freezes the new plan, then submits the array
and report. It prints progress before potentially slow filesystem and scheduler
operations. `prepare`, `verify`, and `submit` are also separate commands. A
disconnected terminal does not imply that Slurm rejected a submission.

Older campaign environment variables do not select this campaign's source or
result root. Explicit overrides use only `AF_PORTFOLIO_TARGETED_SOURCE`,
`AF_PORTFOLIO_TARGETED_ROOT`, `AF_PORTFOLIO_TARGETED_CONFIG`, and
`AF_PORTFOLIO_TARGETED_CONCURRENCY`. The existing `AF_PYTHON`, `AF_CASADI_ROOT`
and `AF_ACCOUNT` overrides remain available.

## Status and inspection from any Delta directory

After submission, recover the exact code directory from its execution record:

```bash
bash <<'BASH'
set -euo pipefail
export AF_PORTFOLIO_TARGETED_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-targeted-v1
AF_REPO=$(jq -er '.value.repo | select(type == "string" and length > 0)' \
  "$AF_PORTFOLIO_TARGETED_ROOT/launcher_execution.json")
AF_WORKER="$AF_REPO/scripts/hpc/start_construction_portfolio_targeted_delta.sh"
test -f "$AF_WORKER" || { echo "Recorded worker is missing: $AF_WORKER" >&2; exit 1; }
bash "$AF_WORKER" status
# After the jobs finish, package the results:
bash "$AF_WORKER" inspect
BASH
```

`status` reads small sealed task/submission receipts. It does not load the
numerical input payload or rerun a report. `inspect` reconstructs result evidence
and creates `$AF_PORTFOLIO_TARGETED_ROOT/inspection.tar.gz` without fitting.

The report distinguishes `trial_training_nmse` from
`training_reconstruction_nmse`. The former belongs to a verified endpoint of
that particular diagnostic (including its held parameter value); the latter
belongs to the best retained model and may still be the original source
incumbent. A missing trial score means no trial endpoint passed verification.
Never substitute the retained source loss into a fixed-parameter profile plot.
Even a verified trial loss is only an upper bound on the profile minimum until
the nuisance optimization is adequate; no failed region is automatically ruled
out and no confidence percentage is assigned.

Read the queue independently:

```bash
squeue -u "$USER" -o "%.22i %.48j %.12T %R" |
  awk 'NR == 1 || /construction-portfolio-targeted/'
```

From the **Mac terminal**, download the archive:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-targeted-v1/inspection.tar.gz ~/Downloads/construction-portfolio-targeted-inspection.tar.gz
```

## Recovery and retention

The submission helper reuses confirmed job receipts. An ambiguous scheduler
reply blocks automatic resubmission; check saved intents and `squeue`/`sacct`
before recovering it. Do not delete seals or choose a new root merely to bypass
an identity mismatch.

If only reporting fails, use `report` or `inspect` with the recorded worker path.
Numerical stages have durable, nonrenewable reservations and retained checkpoints;
retries cannot silently grant a new search budget. A deliberate additional
allowance requires a separate campaign identity.

Retain the source root, frozen plan/inputs, submitted source directory, execution
and submission receipts, operation journals, terminal results, and logs. All
reported improvements require their own causal-prefix evaluation before a
generalization claim. A low conditional objective or successful optimizer stop
alone does not certify parameter confidence or identify a scientific cause of
remaining mismatch.
