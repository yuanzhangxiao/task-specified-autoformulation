# Stronger deterministic public-mechanism assessment

Protocol: `fitted-public-mechanism-tests-1`.

This is a **public-mechanism test pass rate**, not a probability that an equation
is scientifically correct. It evaluates the actual saved fitted model, using
public requirements and fixed numerical controls. Graph reachability alone
cannot pass a mechanism. This rubric was developed after inspecting earlier
results: report it as a post-hoc operational assessment, not a preregistered
endpoint. It does not replace independent scientific review.

## What is tested

The versioned rubric is `configs/mechanism_functional_v1.json`. Its nine cases
match the primary table roster, with eleven public mechanism obligations across
the nine cases. Every obligation quotes an actual public task bullet. The
preparation step checks exact bullet coverage and the public prompt SHA-256.

| Cases | Stronger operational test |
| --- | --- |
| Named T1 easy/hard, named perturbed T1 | Effective fitted meal-to-glucose response; positive first resolved local response |
| Anonymous canonical/perturbed T1 | Effective fitted input-to-output response; polarity is unspecified |
| T2 easy | Meal response, plus effective **dynamic** plasma-insulin-to-disposal response, positive in direction |
| T2 hard | Meal response; delayed negative glucose-balance response is a necessary insulin-action **proxy** |
| Named CSTR | Separate additive heat channels; feed and jacket directions; zero balance at equal temperatures with zero reactant; positive reaction heat with reactant present |
| Alien device | Input-driven dynamic memory reaching the output, beyond direct feedthrough |

Signs in the named physiological cases are an explicit scientific
operationalization of the named meal-source/disposal roles. The prose does not
specify a global monotonicity theorem. Anonymous cases do not inherit hidden
physiological signs. No hidden reference equations, fitted reference parameters,
or invented nonlinear-feedback requirement enter this rubric.

The CSTR tests apply to a temperature balance expressed using the public feed,
jacket and concentration coordinates. Additional internal concentration or
energy coordinates need semantic correspondence and remain unresolved. These
checks do not assert correctness of every possible energy-balance formulation.
They test the conditional local heat balance, not a full plant intervention.
Additivity uses mixed finite increments for each pair of public heat drivers,
with perturbations of 5% of their training standard deviations (reactant: 5% of
its current positive value). The mixed residual tolerance is the larger of
`1e-8` and `1e-6 * max(1, absolute evaluated rates)`, in normalized temperature-rate
units. This accepts factored additive equations and detects tested cross-channel
coupling without demanding a particular written equation format.

**T2-hard identification limit:** a delayed fall in glucose can come from greater
disposal or suppressed production. A passing negative-balance proxy therefore
leaves the full disposal obligation unresolved. A wrong-direction counterexample
fails the operational proxy. Neither latent variable names nor model annotations
are accepted as proof of a disposal mechanism.

## Numerical definition

Use the first three training trajectories in lexicographic trajectory-ID order.
For each, evaluate the initial, middle and final sample, on the model's free
rollout. Selection is independent of method, NMSE, and probe outcome. Training
channel standard deviations set input/output scales (floor `1e-6`); the median
sample interval sets the continuous-time unit. State perturbations use
`max(abs(state), 1)` as their scale. No fitted parameter is changed.

At an operating point, hold all other supplied channels fixed and linearize:

\[
 \delta\dot z=A\delta z+B\delta d,\qquad
 \delta q=C\delta z+D\delta d.
\]

Here `d` is the public driver, `q` the specified readout, and `z` the remaining
generated state coordinates. Inspect the direct coefficient `D` and dynamic
coefficients `CB, CAB, ..., CA^{n-1}B`, where `n = dim(z)`. These coefficients
measure the initial local response and cancel opposing parallel paths. A
positive rescaling of `A` for numerical conditioning preserves each
coefficient's sign and zeros; the implementation records the scaling factor.
A causal-response test requires a resolved direct or dynamic coefficient; a
memory/delay test requires a dynamic coefficient. If the clamped dependency
graph has no route through a remaining dynamic coordinate, the memory/delay
requirement fails; a direct effect alone cannot satisfy it. When polarity is specified,
compare the first resolved coefficient with that polarity. Higher-order later
reversal is not tested or excluded. This is a finite local linear-response test,
not proof of global nonlinear reachability, monotonicity, or physical identity.

For T2, the driver is **generated plasma insulin**, not its future measurements.
The runtime clamps the model's public insulin state while differentiating the
other equations. An identity output-to-state map is required; an unmapped
nonlinear output coordinate remains unresolved. In T2-hard it also clamps glucose
storage and uses its balance rate as `q`. Thus simple glucose accumulation cannot
masquerade as insulin-action memory. These clamps define local conditional
responses, not observed physiological counterfactuals.

For D3 the actual recurrence is `x[k+1] = x[k] + increment(x[k], u[k])`.
Use `A = I + Jacobian(increment)` on non-clamped generated coordinates and unit
step time. Never reinterpret D3 as an ODE or multiply its increment by the sample
interval. Supplied auxiliaries are clamped exactly as in its native rollout;
their saved but unexecuted differential equations do not provide pathway evidence.
A sampled delay is dynamic memory in this discrete convention; it does not imply
a particular continuous-time physiological delay.

Derivatives use dimensionless step `2e-5` and half that step. Driver differences
are one-sided when the lower probe would be negative. Accept derivative agreement
within `1e-3 * max(1, largest derivative magnitude)`. Require a response above
`max(1e-8, 10 * coarse/fine response disagreement)`. A locally inactive point is
not itself a scientific counterexample: nonlinear gating can turn a valid path
off. At least one active point is necessary for confirmation. Any tested
wrong-direction counterexample fails; numerical uncertainty prevents confirmation.

Continuous free rollouts use Radau and BDF (`rtol=1e-7`, `atol=1e-9`, 120 seconds
each), with relative state agreement tolerance `1e-4` using per-state
`max(1, maximum absolute state)` scales. Both sets of operating points must yield
the same classifications. Solver disagreement or failure is unresolved evidence,
not a scientific rejection. D3 uses its native recursive rollout. All candidate
expressions use the existing restricted interpreter. At most 128 state
coordinates are supported for local numerical probes; larger models remain
unresolved under the resource budget.

## Score and robust aggregation

For a run with `M` public mechanism obligations, let `P`, `F`, and `U` count
confirmed passes, counterexamples/method failures, and unresolved obligations:

\[
 s_{\rm confirmed}=P/M,\qquad s_{\rm possible}=(P+U)/M.
\]

A mechanism passes only if its operational checks pass. An absent effective
fitted path fails immediately. A known method failure with no saved model is
ranked worst (zero), as in failure-aware NMSE aggregation. Missing provenance,
unsupported coordinates, interrupted computations and inconclusive numerical
evidence are kept unresolved, not silently deleted or called passes.

For compliance, the current reporting convention is **mean (sample SD)**:
average repetitions within each benchmark, then equally weight benchmark means.
The headline SD is the sample SD across benchmark means (`ddof=1`), not across
all individual requirements and not a standard error. Per-case SD is across
repetitions. The possible score is an evidence upper bound, **not a confidence
interval**. Retain pass/fail/unresolved counts and the planned benchmark scope.
Four-case ablations must not be presented as nine-case aggregates. NMSE retains
its separate median/MAD convention.

The immutable numerical v1 protocol and its original report retain their original
median/MAD fields. `scripts/summarize_functional_mechanisms.py` produces a separate
mean/SD report from the sealed plan and worker results, with no numerical rerun:

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/summarize_functional_mechanisms.py" \
  --root "$AF_ASSESSMENT" --output "$AF_ASSESSMENT/mean-sd"
cat "$AF_ASSESSMENT/mean-sd/SUMMARY.md"
```

For ours and component ablations, see
[Component-model mechanism assessment](COMPONENT_MECHANISM_ASSESSMENT.md).

Do not label this endpoint "complete scientific correctness" or equate it with
the older graph endpoint. Suggested paper label: **Public-mechanism tests (%)**,
with the rubric, coverage and unresolved counts in the appendix. Apply the same
rubric to our saved models through the common sealed model-bundle adapter; do
not use a different scoring rule for our method.

## Running on Delta

The v2 archive supplies 108 planned baseline repetitions, including 103 saved
models and five known failures. It does not contain the public trajectory arrays;
the numerical assessment therefore runs where the original public data reside.
The development loader opens training and validation files to verify their
identities; only training arrays define numerical probes. Test files are not
opened. No fitting, LLM calls, model selection or promotion occurs.

The v2 package manifest records the evaluation receipt under
`external-baseline-evaluation-v1/frozen/execution_record.json`.
Use a pinned source archive. Set `AF_REPO_ROOT` to that extracted repository, then:

```bash
(
  set -euo pipefail
  export AF_REPO_ROOT
  : "${AF_REPO_ROOT:?Set AF_REPO_ROOT to the pinned source archive first}"
  export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
  export AF_ASSESSMENT=/work/hdd/bibo/yxiao2/phase_b/functional-mechanism-tests-v1
  export PYTHONPATH="$AF_REPO_ROOT/src"
  export PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  AF_PUBLIC_ROOT=$("$AF_PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path('/work/hdd/bibo/yxiao2/phase_b/external-baseline-evaluation-v1/frozen/execution_record.json')
print(json.loads(p.read_text())['public_data_root'])
PY
)
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/assess_functional_mechanisms.py" prepare \
    --package /work/hdd/bibo/yxiao2/phase_b/results-package-v2.tar.gz \
    --public-root "$AF_PUBLIC_ROOT" --root "$AF_ASSESSMENT"

  AF_COUNT=$(jq '.rows | length' "$AF_ASSESSMENT/plan.json")
  mkdir -p "$AF_ASSESSMENT/logs"
  AF_JOB=$(sbatch --parsable --account=bibo-delta-cpu --partition=cpu \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=03:00:00 \
    --array="0-$((AF_COUNT-1))%8" --job-name=mechanism-tests \
    --output="$AF_ASSESSMENT/logs/%A_%a.out" \
    --error="$AF_ASSESSMENT/logs/%A_%a.err" --export=ALL \
    --wrap='"$AF_PYTHON" "$AF_REPO_ROOT/scripts/assess_functional_mechanisms.py" run --root "$AF_ASSESSMENT" --index "$SLURM_ARRAY_TASK_ID"')
  printf '%s\n' "$AF_JOB" | tee "$AF_ASSESSMENT/last-array-job.txt"
)
```

After the array finishes, regenerate the report (also safe while running):

```bash
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_ASSESSMENT=/work/hdd/bibo/yxiao2/phase_b/functional-mechanism-tests-v1
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/assess_functional_mechanisms.py" report \
  --root "$AF_ASSESSMENT"
cat "$AF_ASSESSMENT/SUMMARY.md"
jq '{status_counts,free_rollouts_started,methods,llm_calls,optimizer_calls,test_data_opened}' \
  "$AF_ASSESSMENT/summary.json"
```

Return `summary.json` for cross-checking and table construction. It contains
per-run evidence and per-case and macro aggregates. Detailed immutable model
inputs are in `plan.json`; rollout checkpoints are below `results/`.

Re-running a completed model reuses its result. Individual rollouts have durable
started/completed records; interrupted rollouts remain unresolved rather than
receiving a reset budget. Source, public data, rubric and numerical environment
identities must match on resume. Use a new root for an explicitly revised
protocol. A second submission can encounter worker locks; it cannot change a
saved decision or run a concurrent duplicate numerical probe.

For our own saved model bundle, replace `--package ...` with `--bundle ...`.
Select a frozen endpoint per lineage upstream: this script rejects duplicate
method/case/repetition entries and never chooses the best round by these scores.


## Verification and present evidence

- The full repository suite completed with 3,520 passing and eight skipped tests
  using six workers and BLAS/OpenMP threads fixed to one. Some optional Torch
  tests are skipped in this environment; the native NumPy D3 recurrence is tested.
- Focused checks cover wrong signs, cancelled parallel paths, generated-insulin
  clamping, false delay from glucose accumulation, latent rescaling, discrete
  increments, thermal boundaries/additivity, solver disagreement, interruption,
  immutable resume, aggregation, public rubric coverage, and the standalone CLI.
- All 103 saved models from the supplied v2 archive build with the runtime and
  new rubric. Five recorded method failures remain in the 108-run denominator.
- New Python files pass Ruff. Repository-wide Ruff reports 37 existing findings
  under `analysis/claude`; those unrelated files are not changed.
- New numerical baseline scores are **not yet computed locally**: the archive
  lacks the original public trajectory arrays. The Delta commands above produce
  them without using held-out data or refitting any model.
