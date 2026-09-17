# Saved-model public mechanism audit

This milestone computes structural graph scores for saved SINDy, PySR,
GPT-5.6 Sol, native Phase-B D3, and an explicitly chosen Autoformalism round.
It separately runs a common public equation rubric. It performs **no discovery,
fitting, numerical rollout, behavioral probe, or test/private-reference access**.
Historical campaigns, graph gates, and metric aliases remain unchanged.

## Interpretation

| Output | Meaning |
| --- | --- |
| Graph compliance | Legacy syntactic driver/path/memory predicates, independently of annotations. No fitted activity or scientific certificate. |
| Equation review | Pass/fail/unresolved for each quoted Section A mechanism, plus separate Section C/D modeling criteria. |
| Behavioral evidence | `not_assessed` in this milestone. Existing validation NMSE is contextual fit evidence, not a mechanism certificate. |

The rubric is extracted from the **actual, hash-matched public prompt**. It does
not add nonlinear feedback, hidden dimensions, exact reference equations, or
domain vocabulary that the prompt did not supply. Compound task bullets are
conjunctive: the reviewer must discuss every clause. For CSTR this includes feed
transport, reaction heat, and jacket exchange. General modeling requirements do
not inflate the task-mechanism denominator.

Two seeds of GPT-OSS-120B independently receive the same anonymous equation
packet. Method, candidate ID, rationale, mechanism tags, NMSE, prior graph
verdicts, and other reviewers' answers are absent. The packet contains equations,
public channel roles, units, initializers, constraints and fitted coefficients.
Thus a zero fitted weight is visible, but appreciable dynamic activity is not
assumed. Missing annotations or narratives alone cannot fail a mechanism.

Each review must cover every rubric ID exactly once and cite actual equation,
parameter, initializer, constraint, or inventory IDs. Missing units and invented
citations trigger bounded format repair. Two matching verdicts yield that advisory
verdict; missing reviews and disagreements yield unresolved. Unresolved units
remain in the denominator. Missing model counts are always shown separately;
the available-model supported fraction must not be represented as an all-run
success rate. Modeling compliance is reported separately from mechanisms.

This is a **new, uncalibrated equation-review rubric**, not a reuse claim for the
earlier paired judge's calibration. Repeated seeds of one model are not independent
expert validation. Inspect the equation evidence and have a human assess a balanced
sample before using its numbers as a primary paper claim. All reports explicitly
set `scientific_correctness_certified=false`. No overall scientific-compliance
percentage combines graph, equation, and behavioral results.

## Source adapters and provenance

- SINDy/PySR/GPT: portable snapshots of the exact embedded subjects in an existing
  `baseline_validation.py` plan. No train-plus-validation refit. Saved validation
  replay scores are joined only after their plan/index identities match. Legacy
  training-release provenance limitations remain in the report.
- D3: only the exact Phase-B native campaign's `native-selection.json`. Equations
  remain sample increments, `x[n+1]=x[n]+f`, without a dt multiplier. Supplied
  auxiliary increment equations are excluded from the *target-rollout* dependency
  graph because those auxiliaries are supplied at every sample. Target mappings
  and initial values follow native rollout identity semantics. Original auxiliary
  equations are retained in source provenance. Continuous-time compliance is a
  separate representation question. Historical D3 is never substituted for an
  unavailable Phase-B model.
- Autoformalism: snapshot the retained model at an explicit global round; no
  search over rounds by score, no historical proposal resurrection, no closure of
  the ongoing search campaign. Default `--arms full`; specify the other arms if
  desired. `no_spec` is assessed post hoc against the full public task and marked
  as not having seen that task during search.

Model bundles contain public equations and saved validation scores, never
trajectory tables, credentials, cached API responses, test scores, or private
mechanism labels. They can be copied between Delta and ACES. Every source and
audit plan is content-addressed. A changed model, rubric, code, prompt or judge
revision requires a new audit root. Duplicate method/cell/repetition/round sources
are rejected instead of selecting one by its scores.

## Commands

Use a separate checkout at the commit supplied in the completion message. Do not
update a running campaign's checkout. `AF_REPO_ROOT` below is that checkout.

### 1. Delta: export existing external models (CPU only)

Set source roots to the completed campaigns you used for the reported numbers.
The paths below are the documented defaults; if yours differ, substitute their
actual directories. Both plans must exist. Export does not import Julia, PySR,
Torch, or start a provider.

```bash
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export PYTHONPATH="$AF_REPO_ROOT/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export AF_BASELINE_ROOT=/work/hdd/bibo/yxiao2/phase_b/baseline-validation-full-v1
export AF_D3_ROOT=/work/hdd/bibo/yxiao2/phase_b/d3-native-pilot-v1
export AF_BUNDLE=/work/hdd/bibo/yxiao2/phase_b/mechanism-audit-inputs-v1/external.json

test -f "$AF_BASELINE_ROOT/plan.json"
test -f "$AF_D3_ROOT/plan.json"
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/mechanism_audit.py" export \
  --baseline-root "$AF_BASELINE_ROOT" --d3-root "$AF_D3_ROOT" \
  --output "$AF_BUNDLE"
```

If only the six-cell baseline replay was run, use `baseline-validation-v1` instead.
If D3 is on another machine, export it separately with `--d3-root` and add the
result as another `--bundle` during preparation. An unavailable new D3 campaign
must not be replaced by historical benchmark models.

Copy `external.json` to
`/scratch/user/u.yx126462/phase_b/mechanism-audit-inputs-v1/external.json` on ACES.
The Open OnDemand file managers can download/upload this small public-model file;
no new cross-cluster SSH authentication is required. If all source campaigns
are already on ACES, run the export there against those paths instead.

### 2. ACES: export our round-12 models and compute graph scores

```bash
module load GCCcore/13.2.0 Python/3.11.5
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export PYTHONPATH="$AF_REPO_ROOT/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export AF_INPUT_ROOT=/scratch/user/u.yx126462/phase_b/mechanism-audit-inputs-v1
export AF_REVIEW_ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v4-signfix
export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/mechanism-audit-v1
export AF_HF_HOME=/scratch/user/u.yx126462/huggingface-cache

"$AF_PYTHON" "$AF_REPO_ROOT/scripts/mechanism_audit.py" export \
  --review-root "$AF_REVIEW_ROOT" --round 12 --arms full \
  --output "$AF_INPUT_ROOT/ours-round12.json"

export AF_JUDGE_REVISION="$(cat "$AF_HF_HOME/hub/models--openai--gpt-oss-120b/refs/main")"
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/mechanism_audit.py" prepare \
  --bundle "$AF_INPUT_ROOT/external.json" \
  --bundle "$AF_INPUT_ROOT/ours-round12.json" \
  --public-root "$AF_REVIEW_ROOT/public" \
  --model openai/gpt-oss-120b --model-revision "$AF_JUDGE_REVISION" \
  --root "$AF_OUTPUT_ROOT"

cat "$AF_OUTPUT_ROOT/SUMMARY.md"
jq '.rows[] | select(.status!="ready") | {method,benchmark_id,repetition,error}' \
  "$AF_OUTPUT_ROOT/summary.json"
```

This step already computes graph numbers. Inspect missing models and prompt
mismatches before spending GPU time. No GPU or LLM call occurs during preparation.
An unavailable graph specification does not become zero scientific compliance.
It is recorded as `spec_prompt_mismatch`; a correctly matched source prompt may
still receive its own equation review.

The older loose prompt copies in the repository root may differ from the
campaign prompts. Use the campaign's frozen `public` directory, not those copies.
The default rubric/model revision is pinned before judging. Cached model files
must exist; the GPU job uses offline mode and will not download model weights.

### 3. ACES: scientific equation audit

The established 120B serving configuration uses **two H100s**. This is a separate
evaluation job; it does not rerun proposal or fitting. The smaller 20B model is
supported only as a separately labeled pilot: prepare a new root with its exact
revision and request one GPU. Do not mix its verdicts with 120B results.

```bash
export AF_COMMIT="$(git -C "$AF_REPO_ROOT" rev-parse HEAD)"
export AF_VLLM_IMAGE=/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif
mkdir -p "$AF_OUTPUT_ROOT/logs"

# Submit once. Keep the receipt; inspect squeue before any retry.
sbatch --parsable --account=156264627414 --partition=gpu \
  --nodes=1 --ntasks=1 --gres=gpu:h100:2 --cpus-per-task=8 --mem=128G \
  --time=06:00:00 --signal=B:TERM@300 --export=ALL \
  --job-name=mechanism-audit \
  --output="$AF_OUTPUT_ROOT/logs/audit-%j.out" \
  --error="$AF_OUTPUT_ROOT/logs/audit-%j.err" \
  "$AF_REPO_ROOT/scripts/hpc/run_mechanism_audit_aces.sh"
```

Each model gets two seeded reviews, each capped at three physical requests,
8,192 output tokens and 300 seconds per request. Format repair reuses the same
model and rubric. Requests/responses are cached and usage logged. An interrupted
request consumes its attempt and is not silently resent. The allocation stops
starting calls with insufficient time; rerunning the same completed/incomplete
plan reuses saved work. After the prior allocation has ended, the same submission
command resumes pending reviews without granting terminal failures a fresh budget.

### 4. Inspect the separate metrics and evidence

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/mechanism_audit.py" report --root "$AF_OUTPUT_ROOT"
cat "$AF_OUTPUT_ROOT/SUMMARY.md"
jq '{input_status_counts,review_status_counts,accounting,groups}' \
  "$AF_OUTPUT_ROOT/summary.json"
jq '.rows[] | {method,benchmark_id,repetition,validation_nmse,
  graph:.graph.graph_mechanism_compliance,
  science:.equation_review.counts,
  requirements:.equation_review.requirements}' "$AF_OUTPUT_ROOT/summary.json"
```

`packets/*.json` are anonymous human-review inputs. `reviews/INDEX/SEED_SLOT/`
contains exact saved requests, responses, usage, validation failures and the
review result. `summary.json` contains model identity and both reviewers' evidence
for every requirement. Use these files for figures and tables; do not relabel
the graph column as full mechanism compliance.

## Verification and next boundary

Run `pytest`, `ruff check .`, and `scripts/smoke_mechanism_audit.py`. Regression
cases cover zero fitted coefficients despite graph success, missing annotations,
wrong prompt hashes, omitted rubric units, fabricated evidence references,
disagreement/missing-review denominators, D3 supplied-auxiliary semantics,
bounded failures, deterministic resume, and preserving unavailable sources.

Actual fitted behavioral mechanism tests remain a separate milestone. Any such
test must prespecify public input probes and meaningful effects per requirement,
keep parameter values fixed, and distinguish an uninformative observation from
a falsified mechanism. Existing held-out metrics remain behind their original
post-freeze boundary and are not fed to this judge or ongoing proposal generation.
