# Three-layer deterministic mechanism assessment

Primary saved-model assessment for the six review benchmarks. CPU replay only:
no generation, refitting, pruning, scientific judge, test access, private
equations or hidden trajectories. Existing campaigns and the fitter stay frozen.
Broader interpretation through `PUBLIC_MECHANISM_AUDIT.md` remains optional.

| Layer | JSON fields | Claim |
| --- | --- | --- |
| Equation requirements | `equation_requirements`, `fitted_equation_requirements` | Formal predicates before/after substituting saved parameters |
| Fitted activity | `fitted_activity` | Resolved conditional public-channel influence; post-pulse persistence for memory |
| Response behavior | `response_behavior` | Observed validation NMSE, direction, centered shape correlation and peak timing |

Every layer has pass/fail/unresolved counts and individual evidence. Completion
means execution completed, not that a model is good. Do not combine these into
an overall scientific correctness score.

## Equation predicates

`configs/mechanism_assessment_v1.json` pins exact public-prompt hashes and
reviewed operational predicates. Each Section A requirement needs a binding.
Tags and candidate explanations never supply evidence. The pre-fitting
restricted parser, zero/cancellation simplifier, graph traversal and nonlinear
source detector are reused on assembled equations from every method.

| Cells | Operational predicate |
| --- | --- |
| Named canonical/perturbed Dalla Man T1 | Meal-event input has a generated causal path to `Gp` |
| Obfuscated canonical/perturbed T1 | Public input has a generated causal path to `v01` |
| CSTR easy | Distinct additive witnesses depending on `Tf`, `C`, and `Tj`, reaching `T` |
| Alien easy | Input drives a dynamic state contributing to `v01` |

The easy alien prompt does **not** require nonlinear feedback. We do not invent
it from tag aliases or the hidden generator. The reusable `nonlinear_feedback`
predicate is tested for future explicit bindings: a nonlinear dependence must
be on a feedback cycle connected to the required driver and target. Nonlinear
input preprocessing and disconnected nonlinear equations do not qualify.

Memory may reside in an observed dynamic state; the public prompt leaves the
internal representation open. This improves on the older graph score's latent
witness restriction without rewriting historical scores. Native identity
carryover alone does not qualify as nonlinear feedback. D3's lack of an
explicit continuous-time equation is reported separately.

CSTR witnesses do **not** prove thermodynamic meaning, signs or balance. A single
product of all three channels does not establish distinct contributions.
Absence is unresolved because an alternative internal model may not directly
use those auxiliaries. Complicated equivalent expressions may need broader
review. General symbolic cancellation, stability and identifiability are not
certified. A 100% formal score must never become "100% scientific compliance."

## Numerical protocol and interpretation

All saved parameters and causal initializers remain unchanged. Continuous-time
models use production free rollouts with Radau and BDF, `rtol=1e-7`, `atol=1e-9`,
120 seconds per rollout, and no target resets after initialization. Disagreement
above `1e-4` training-output standard deviations makes evidence unresolved.
Numerical agreement is not predictive accuracy.

D3 uses native `x[n+1]=x[n]+rhs[n]` without a time-step multiplier, overwriting
supplied auxiliaries each sample. Original auxiliary increment laws are restored
for the native adapter's schema but cannot create fictitious target pathways.
Native execution does not claim BDF/Radau verification.

For each bound driver, activity probes select the training trajectory with the
largest driver range (ties by trajectory ID), independently of fit scores. Add
and subtract 5% of its pooled training range over 25–50% of the horizon, clipped
to that range. Initial public values and other channels are unchanged. Save
normalized output RMS change and post-pulse RMS change; memory uses the latter.
Detection requires an effect above `max(1e-5, 10 * combined solver discrepancy)`.
Flat channels, failed/inconsistent rollouts and unresolved effects stay
unresolved, rather than proving global inactivity. Both probe directions are
saved and no outcome-dependent extra probes are scheduled.

These are conditional model sensitivities. Holding supplied auxiliaries fixed
while perturbing another channel is **not** claimed to be a physically
realizable whole-system intervention. Input influence alone cannot isolate a
nonlinear feedback loop: a future loop-specific activity request remains
unresolved; fitted zero/cancelling feedback is detected by the equation layer.
The default six cells contain no nonlinear-feedback obligation.

Response metrics reuse `qualitative_response_metrics`: direction at the
observed response's greatest displacement from initialization, centered shape
correlation, and peak-time error divided by horizon length. A descriptive band
requires NMSE <= 0.1, correct direction, correlation >= 0.9 and timing error <=
0.1. Raw values remain available; these thresholds are not scientific laws.
Flat observed responses have unresolved shape/direction/timing but retain NMSE.
Whole-split NMSE requires every trajectory to replay and uses sample-weighted
errors per target followed by an equal mean across targets.

Validation is reused development data and may already have influenced selection;
source provenance determines a baseline's earlier fitting use. These are **not
pristine test results**. Unpaired observations cannot establish pathway-specific
causal counterfactuals. Existing authorized post-freeze test evaluations remain
separate. No result is automatically routed to proposal generation.

## Export saved models

Use an isolated checkout at the supplied commit as `AF_REPO_ROOT`. The following
are documented source roots; substitute the actual completed campaign if needed.

On **Delta** (no Julia, Torch, refitting or LLM calls):

```bash
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export PYTHONPATH="$AF_REPO_ROOT/src"
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/mechanism_audit.py" export \
  --baseline-root /work/hdd/bibo/yxiao2/phase_b/baseline-validation-full-v1 \
  --d3-root /work/hdd/bibo/yxiao2/phase_b/d3-native-pilot-v1 \
  --output /work/hdd/bibo/yxiao2/phase_b/mechanism-assessment-inputs-v1/external.json
```

For the six-cell baseline-only replay, use `baseline-validation-v1` instead.
If native D3 is on another host, omit `--d3-root`, export it there separately,
then include that file as another `--bundle`. Do not substitute older benchmarks.
Transfer `external.json` through the file managers to
`/scratch/user/u.yx126462/phase_b/mechanism-assessment-inputs-v1/external.json`
on ACES. Previous equation-audit bundles also work; fresh exports additionally
preserve source train/validation fingerprints when available. Missing source
fingerprints are explicitly reported rather than certified.

## ACES: prepare and submit

```bash
module load GCCcore/13.2.0 Python/3.11.5
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export PYTHONPATH="$AF_REPO_ROOT/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export AF_INPUT_ROOT=/scratch/user/u.yx126462/phase_b/mechanism-assessment-inputs-v1
export AF_REVIEW_ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v4-signfix
export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/mechanism-assessment-v1

"$AF_PYTHON" "$AF_REPO_ROOT/scripts/mechanism_audit.py" export \
  --review-root "$AF_REVIEW_ROOT" --round 12 --arms full \
  --output "$AF_INPUT_ROOT/ours-round12.json"

"$AF_PYTHON" "$AF_REPO_ROOT/scripts/assess_mechanisms.py" prepare \
  --bundle "$AF_INPUT_ROOT/external.json" \
  --bundle "$AF_INPUT_ROOT/ours-round12.json" \
  --public-root "$AF_REVIEW_ROOT/public" --root "$AF_OUTPUT_ROOT"

cat "$AF_OUTPUT_ROOT/SUMMARY.md"
jq '.rows[] | select(.status!="ready") | {method,benchmark_id,error}' \
  "$AF_OUTPUT_ROOT/plan.json"
bash "$AF_REPO_ROOT/scripts/hpc/submit_mechanism_assessment_aces.sh"
```

Preparation opens only public train/validation and computes equation predicates.
The array requests one CPU, 8 GB, three hours per model, at most eight concurrent
tasks, and zero GPUs. A dependent CPU job writes the summary. No model weights
or judge revision are needed. Missing sources remain in the roster. To include
our ablations, export the explicitly desired `--arms` into a new bundle/root.

## Results and resume

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/assess_mechanisms.py" report --root "$AF_OUTPUT_ROOT"
cat "$AF_OUTPUT_ROOT/SUMMARY.md"
jq '{status_counts,groups}' "$AF_OUTPUT_ROOT/summary.json"
jq '.rows[] | {method,benchmark_id,repetition,status,
  equations:.equation_requirements.requirements,
  fitted:.fitted_equation_requirements.requirements,
  activity:.fitted_activity.requirements,
  responses:.response_behavior}' "$AF_OUTPUT_ROOT/summary.json"
```

Each model and numerical call has an immutable checkpoint. Repeating a completed
row reuses its work. Calls interrupted after starting become explicit unresolved
evidence; terminal failures do not gain a new budget. Code, settings, data or
model changes require a new plan. After an earlier allocation has ended, resume
an unfinished row by submitting the worker with `--array=INDEX`, the same CPU
limits and `AF_COMMIT=$(git -C "$AF_REPO_ROOT" rev-parse HEAD)`. Do not delete
results or bypass a submission intent. Per-model locks prevent concurrent runs.

Use `summary.json` for tables/figures. Unknown response denominators remain
explicit; unresolved units are never dropped to inflate percentages. Source
validation NMSE and newly replayed response NMSE remain distinct. Optional
scientific reviews do not change these deterministic metrics.

Verification: focused success/failure tests, full `pytest`, `ruff check .`, shell
syntax checks and `scripts/smoke_mechanism_assessment.py`. The offline numerical
smoke uses a known analytic response and exercises all three layers and resume.
