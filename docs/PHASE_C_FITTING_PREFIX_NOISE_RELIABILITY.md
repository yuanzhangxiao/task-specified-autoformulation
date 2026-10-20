# M47: reliability of observation-prefix selection under noise

## Motivation and scope

M46 completed all twelve tasks and all 336 diagnostic pairs. Its nearly
confounded noisy control selected duration 0.2: tuning common-horizon NMSE was
0.00668, but fresh-trajectory NMSE was 0.0216 (clean-signal NMSE 0.0207).
All three starts agreed on essentially the same constrained optimum. The
short noisy prefix, rather than unresolved optimization, was the limitation.

M47 compares the unchanged M46 selection rule with an added conditional
prediction-support check. This is a self-contained CPU qualification of known
equations. It changes no benchmark, finalized prompt, production default or
constructor. An actual construction-export fitting qualification remains separate.

## Frozen experiment

The controls remain

    x' = -theta*x + z + r, z' = -z, r' = -lambda*r,
    y=x, x_j(0)=y_j(0)=0.4, z_j(0)>=0, r_j(0)>=0.

The three lambda values are 2, 1.05 and 1. Training jointly estimates one shared
theta and six hidden initials across three trajectories. Only the two hidden
initials are inferred on a new trajectory; theta and normalization stay frozen.

There are 36 tasks: three structures, each with one clean acquisition and five
independent noisy acquisitions, each fitted from two predetermined starts.
Noise is additive Gaussian with known standard deviation 0.01. All trajectories
have 41 readings on [0,4], and the first reading remains exact.

A new root roster is drawn once before execution with seed `[20261020,47,1]`,
uniformly in [0.2,2]. Three training, four tuning and four follow-up root pairs
remain fixed across noise repeats, and differ from M46. Repeat IDs 100–104 use
independent generators `[20261020,47,repeat,2]`. Standardized noise is paired
across structures; both optimizer starts see identical samples. Clean uses ID 99.
There are five independent noise acquisitions, not ten independent experiments.
They do not represent all hidden-state distributions or noise levels.

Both rules share durations (0,0.1,0.2,0.5,1,2), root starts
((0.3,1.3),(1.3,0.3),(1,1)), fitted endpoints and independent rollout checks.
M47 reads no M46 archive. Both choices are sealed together before any follow-up
inference or scoring. Only the union of selected durations is evaluated on the
fresh follow-up acquisitions. Shared work and identical choices are charged once.

## Paired selection rules

**Baseline:** select the shortest duration whose four tuning trajectories have
mean NMSE <=0.01 and maximum normalized prediction spread across starts <=0.01,
both on the common horizon t>2 and immediately after the prefix,
tau<t<=tau+0.5. All durations and all starts must be available.

**Noise-aware:** retain every baseline condition and additionally require the
conditional prediction-support bound below to be <=0.1 normalized RMS on each
horizon for every tuning trajectory. This threshold is the square root of the
existing 0.01 NMSE threshold. The objective-rise multiplier is fixed at four.
Neither setting is a confidence calibration.

Unavailable diagnostics affect the added rule without discarding complete
baseline results. A computed unbounded relaxation is available but inadequate;
it is not a numerical failure. No adequate window is a legitimate outcome.
Individual-root rank does not veto otherwise reliable predictions.

## Conditional predictive support

For fitted theta, write the normalized prefix residual as A xi-d, where
xi=(z(0),r(0)). A and d use only prefix observations and the training scale s.
The independent bounded least-squares reference computes

    S+ = min_{xi>=0} ||A xi-d||_2^2.

The retained endpoint must satisfy its existing conditional optimum tolerance.
Consider the objective-rise support

    C = {xi>=0: ||A xi-d||_2^2 <= S+ + q*(sigma/s)^2}, q=4.

This asks which other initials fit the realized noisy prefix nearly as well.
It is not a confidence region. Let c=A^+d be the unconstrained least-squares
center, S0 its objective, and A=U diag(s_i) V^T have numerical rank r. Define

    rho = sqrt(max(0, S+ + q*(sigma/s)^2 - S0)).

For n future timestamps, B maps roots to normalized future predictions.
Provided B cancels all prefix-null directions, a conservative upper bound on
forecast RMS deviation from retained xi_hat is

    ||B(c-xi_hat)||_2/sqrt(n)
    + rho*||B V_r diag(1/s_i)||_2/sqrt(n).

This encloses the nonnegative support by relaxing its domain. It can be loose;
an unbounded relaxation does not prove the original nonnegative set is unbounded.
The second term, called `noise_support_contribution` in receipts, includes any
gap between constrained and unconstrained minima, not just measurement noise.

A small numerical nullspace leakage alone does not establish cancellation.
Rank-deficient finite forecasts require the declared identical-rate/equal-column
structure. Exactly confounded roots can therefore have finite predictive support
while their individual values remain ambiguous.

Only future timestamps enter this diagnostic, never future measured values or
generating truth. It performs no integration or parameter update. It conditions
on fitted theta and exact y(0), omitting their uncertainty. Passing it does not
bound error against physical truth or guarantee new-trajectory accuracy.

## Outputs, accounting and integrity

Both rules report selected duration and immediate/common follow-up errors.
Retrospective evaluator outputs report shared-theta, individual-root, root-sum
and clean-signal errors. Conditional rank, start variability, reference objective
gaps and predictive support remain distinct. Truth never selects a duration.

Group summaries retain expected, recorded, selected and scored denominators,
independent acquisition counts and individual-root error statistics. Missing
results never become zero error. No best-of-two start selection uses follow-up.

Every operation has a durable reservation and receipt. Interruptions remain
charged and cannot silently restart. Support receives five numerical-active
seconds and an inclusive allowance twenty seconds longer. Checkpoint overhead
stays in total cost, separate from numerical-active time. Deadlines are cooperative.

The M46 portable-report fix permits <=1e-15 absolute drift only in its two
postselection analytic-versus-numerical discrepancy fields. Seals, scores,
choices, parameters, shapes and identities retain exact checks. It returns the
original result without refitting or rewriting receipts. M47 applies the same
narrow comparison to each evaluator arm and requires the exact arm roster.

## Local qualification

The eight focused modules pass 308 tests, covering numerical bounds, missing
evidence, identity validation, exact resume, paired selection and submission.
The returned M46 archive reproduces its original summary without numerical
refitting or changes to its 7,119 sealed JSON files.

Two full-grid noisy smoke tasks (acquisition 100, training start 0) complete:

| Structure | Baseline duration / fresh common NMSE | Added rule duration / fresh common NMSE |
| --- | --- | --- |
| Nearly confounded | 0.2 / 0.0114 | 2.0 / 0.00117 |
| Exactly confounded | 0.2 / 0.00359 | 0.5 / 0.00133 |

Both tasks resume without changing saved JSON bytes or modification times.
These are implementation checks, not an estimate over independent acquisitions.
No policy settings were changed after these checks.

## Run on Delta

Upload `phase-c-fitting-m47.tar.gz` to `AF_BUNDLE`, then run:

```bash
bash <<'BASH'
set -euo pipefail
AF_BUNDLE=/work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m47.tar.gz
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/prefix-noise-m47.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_CODE"
cd "$AF_CODE"
export AF_PREFIX_NOISE_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-noise-reliability-v1
export AF_ACCOUNT=bibo-delta-cpu
export AF_CONCURRENCY=2
bash scripts/hpc/submit_phase_c_prefix_noise_reliability_delta.sh
BASH
```

The launcher prints progress and verifies source checksums and numerical imports.
Each of 36 array tasks requests one CPU, 16 GB and a 45-minute scheduling ceiling;
concurrency defaults to two. No GPU, LLM, prior snapshot or benchmark test data
is needed. Uncertain scheduler replies stop for reconciliation.
The updated repository submitter allows 30 minutes for preparation (verification
plus the eight test modules). The original `6bd2f8e` archive retains its ten-minute
setting; use the recovery below for the affected campaign. This scheduler
allowance is separate from every fitting budget.

The block above is for the first submission. To resume, reuse the original code
directory: submission identity includes its absolute path. Do not extract a new
copy against an existing output root. If submission identity was written, use:

```bash
bash <<'BASH'
set -euo pipefail
export AF_PREFIX_NOISE_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-noise-reliability-v1
AF_CODE=$(jq -er '.value.repo' "$AF_PREFIX_NOISE_ROOT/submission/identity.json")
cd "$AF_CODE"
export AF_ACCOUNT=bibo-delta-cpu
export AF_CONCURRENCY=2
bash scripts/hpc/submit_phase_c_prefix_noise_reliability_delta.sh
BASH
```

If identity does not exist yet, reuse the original extracted directory. An
unconfirmed scheduler reply still requires reconciliation; resume does not
automatically repeat uncertain submissions.

### Recovery of the first preparation timeout

On 2026-10-20, preparation job `23060456` verified all 36 tasks but reached its
original ten-minute scheduler limit during pytest. No test failure was shown.
The dependent array `23060458` was cancelled before starting; report `23060459`
completed with all 36 tasks missing. This is not a numerical fitting result.

Upload `scripts/recover_prefix_noise_preparation.py` as a standalone file to
`/work/hdd/bibo/yxiao2/phase_c/recover_prefix_noise_preparation.py`, **outside**
the original source bundle. Then run:

```bash
bash <<'BASH'
set -euo pipefail
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-noise-reliability-v1
AF_PYTHON=$(jq -er '.value.python' "$AF_ROOT/submission/identity.json")
"$AF_PYTHON" -u /work/hdd/bibo/yxiao2/phase_c/recover_prefix_noise_preparation.py \
  --root "$AF_ROOT"
BASH
```

The helper verifies the original sealed plan, inputs, bundle and job intents,
then requires accounting for the timed-out preparation, completed report and
all 36 cancelled, unstarted array elements. It submits the **original worker
and tests** with a 30-minute preparation limit. Array resources, numerical
budgets and dependencies stay the same. No frozen experiment is regenerated.

Original submission files remain intact. Separate recovery receipts, original
accounting and a copy of the incomplete summary are retained under
`submission/prepare-timeout-recovery/`. Repeating the command reuses confirmed
jobs, even after results start appearing; it never automatically retries an
unconfirmed submission or a terminal recovery job. If accounting is incomplete,
the helper stops without submitting. Use the recovery IDs for status checks:

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-noise-reliability-v1
AF_IDS=$(jq -er '.jobs | [.prepare,.run,.report] | join(",")' \
  "$AF_ROOT/submission/prepare-timeout-recovery/manifest.json")
sacct -j "$AF_IDS" --format=JobID,JobName%48,State,ExitCode,Elapsed,Timelimit
```

Local recovery verification: all 308 original bundled preparation tests pass,
as do 125 current launcher/recovery tests. A standalone smoke check uses the
real original bundle and submission code with simulated Slurm replies; repeat
invocation adds no submissions and preserves original bytes and modification
times. This does not establish completion on Delta. Changed files pass Ruff;
repository-wide Ruff retains 37 unrelated findings in `analysis/claude/`.

After the report completes:

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-noise-reliability-v1
jq '{status, expected, recorded, status_counts, selection_counts, groups, cost}' \
  "$AF_ROOT/summary.json"
AF_REVIEW="$AF_ROOT/review-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_ROOT" \
  plan.json inputs.json summary.json results submission logs
printf '%s\n' "$AF_REVIEW"
```

Keep all starts, failed operations and both policies. `status: complete` means
the experiment finished, not that every forecast passed. Remote M47 outcomes
are pending.
