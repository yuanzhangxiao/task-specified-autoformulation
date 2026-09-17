# Continue from round 7 with one declaration per new parameter

This milestone imports each lineage's own retained round-7 checkpoint from
`review-continuation-v3` and runs **five additional visits, rounds 8–12**, in
`review-continuation-v4`. It preserves the 44-lineage roster and earlier results.
The two lineages without a retained model remain unavailable. The source cannot
be imported after it has been sealed for held-out evaluation.

The model settings, scientific content revision policy, ablation assignments,
fitter profile, numerical budgets, initial-condition rules, and selection rule
are inherited. For the existing campaign, that means GPT-OSS-20B with low
reasoning, one H100 for proposal work, and CPU fitting. This is another explicitly
versioned development phase, not a new unchanged-protocol ablation experiment.
Mark the changes after rounds 2 and 7 in convergence plots.

## Revision contract

The proposer supplies scientific changes without an action-type label. Existing
equation and parameter names come from the request. Equations contain their
complete replacement RHS. **Only genuinely new equation/output parameters are
declared, once, at the top level:**

```json
{
  "hypothesis": "A saturating response may explain the late residual.",
  "evidence_refs": ["E001"],
  "equations": [
    {"component": "m", "expression": "-rate*m+gain*u01"},
    {"component": "f", "expression": "amp*tanh(shape*m)"}
  ],
  "new_parameters": [
    {"name": "amp"},
    {"name": "shape", "role": "shape"}
  ],
  "remove_variables": [],
  "output_expression": null,
  "initializers": []
}
```

This illustrates the interface, not a recommended model. It assumes `rate`,
`gain`, `m`, and `f` already exist. New variables still require their type and,
when latent, an initializer. Parameters inside a newly supplied initial causal
map retain that map's existing local declaration format.

- Existing parameters inherit their original role and domain. Redundant
  declarations cannot change either, even if they contradict one another; the
  ignored requests are recorded. The parent model stays unchanged until the
  whole revision is accepted.
- A new direct gain or additive constant receives its role from the existing
  narrow expression checks. Internal nonlinear parameters require an explicit
  scientific role: `shape`, `positive_shape`, `rate`, `scale`, or `time_constant`.
  Signed shapes remain possible. No general rule forces all parameters positive.
- New shared parameters are resolved across every edited equation and output
  expression before compilation. Incompatible new declarations or uses receive
  a diagnostic identifying the parameter and uses. The runtime does not guess
  scientific intent, split a shared parameter, or silently tie separate ones.
- Grammar, dependencies, target generation, initialization, model size, public
  requirements, and ablation checks remain active. Citation warnings follow v3.

The previous implementation could reject repeated declarations before checking
that the name was inherited. The new interface removes this source of failure.
It does not promise to resolve every historical shared-declaration error: some
may concern genuinely new incompatible definitions.

Each visit still allows at most three physical proposer requests and one fitting
allocation. If revision fails or returns no change, that allocation refits the
retained model from its complete fitted parameter vector. A failed challenger
cannot erase the incumbent or close an otherwise valid lineage. Parameter
fitting and residual evidence use training data; retained-model selection uses
validation. This batch ends after round 12. No test evaluation is submitted.

## Interpretation and preflight artifacts

`no_spec` retains its internal task identifier but is displayed as
**prediction-only**. It removes scientific requirements and their enforcement,
while retaining the prediction task and executable-model requirements. It tests
the consequence of withholding scientific specifications; better NMSE alone
does not make it a better scientific model. Compare predictive quality with
task satisfaction, and distinguish this control from the other ablations.

Reports add `arm_label` and `graph_check_status`. An ambiguous graph inference is
`unresolved`, distinct from a failed predicate. Existing certificate values and
eligibility are not rewritten. For CSTR's `controlled_balance`, the public spec
has no required driver. A generic driver-to-target graph check therefore does
not establish whether the temperature equation distinguishes feed transport,
reaction heat, and jacket exchange. That requires examining the actual equations;
inventing a required driver would change the task.

The first CPU preparation job runs tests, an offline smoke with real fitting,
and two diagnostics before GPU work:

| Artifact | Contents |
| --- | --- |
| `parameter_response_audit.json` | Replays the source phase's saved responses under the new contract, classifies declarations as inherited/new using the actual parent, and retains remaining failures and requirement findings. No LLM or fitting calls. |
| `requirement_review.json` | Unresolved predicates, public requirements, actual retained equations, fitted parameters, and initial-condition plans. |
| `REQUIREMENT_REVIEW.md` | Readable equations and unresolved requirements, including the CSTR balance review prompt. No new scientific verdict. |
| `summary.json`, `rounds.csv`, `SUMMARY.md` | New-phase outcomes, imported anchor, proposal failures, fallback fits, trial/retained NMSEs, and incremental costs. |

Offline replay never promotes a rejected historical proposal into the retained
model. It diagnoses the interface; fresh visits construct and fit actual new
challengers. Source responses can remain blocked without failing preflight.
Provenance or execution errors do fail preflight.

## Run on ACES

Use the exact commit from the implementation response. These commands reuse an
existing authenticated clone; they do not initiate another SSH clone. The helper
creates a separate pinned checkout and a separate campaign directory.

```bash
(
  set -euo pipefail
  REV=REPLACE_WITH_REPORTED_COMMIT
  BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  git -C "$BASE" fetch origin codex/prefit-aces-v1
  export AF_REPO_ROOT="/scratch/user/u.yx126462/repos/autoformalism-review-v4-${REV:0:7}"
  if [ ! -e "$AF_REPO_ROOT/.git" ]; then
    git -C "$BASE" worktree add --detach "$AF_REPO_ROOT" "$REV"
  fi
  [ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" = "$REV" ]
  export AF_PYTHON="$BASE/.venv/bin/python"
  export AF_SOURCE_ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v3
  export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v4
  export AF_SOURCE_ROUND=7 AF_ADDITIONAL_VISITS=5
  export AF_CONTINUATION_PROTOCOL=review-deadline-4
  bash "$AF_REPO_ROOT/scripts/hpc/submit_review_continuation_aces.sh"
)
```

The first submission queues preparation, the first proposer visit, its CPU fit
array, reporting, and a dispatcher. Subsequent visits are queued only after all
previous result records exist. `all_rounds_submitted=false` is expected initially.
There is no dependency on retired source job IDs. The scheduler records every
submission intent and reply; an uncertain reply stops for inspection instead of
submitting duplicate jobs. Repeating a completed submission returns its manifest.

The default GPU request is one H100, 8 CPUs, and 64 GB; each fitting task requests
one CPU and 16 GB, with up to 16 fitting tasks running concurrently. Existing
timeouts and per-fit numerical budgets are unchanged. `AF_ADDITIONAL_VISITS` can
be 1–5 when creating a phase and cannot change on resume. A later continuation
can import a completed v4 phase into another new directory, but this submission
does not automatically do so.

## Inspect results

After preparation, inspect the audit and equations:

```bash
ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v4
jq '{status_counts, source_terminal_errors, rejected_then_valid}' \
  "$ROOT/parameter_response_audit.json"
cat "$ROOT/REQUIREMENT_REVIEW.md"
```

During or after the run:

```bash
cat "$ROOT/submission_manifest.json"
sacct -j "$(jq -r '[.jobs[]] | unique | join(",")' "$ROOT/submission_manifest.json")" \
  --format=JobID,JobName%30,State,ExitCode
cat "$ROOT/SUMMARY.md"
jq '{status_counts, proposal_status_counts, fallback_fits}' "$ROOT/summary.json"
```

Reported global rounds are 7–12; local `phase_round` 0 is the imported round-7
anchor. On disk, `results/<task>/round_01` is global round 8. Imported checkpoints
are charged zero new calls or fitting attempts; earlier spending remains in the
source reports. Keep phase boundaries and budgets visible in tables and plots.

## Local verification

```bash
.venv/bin/pytest -q tests/test_review_parameters.py tests/test_review_continuation.py \
  tests/test_review_deadline_v2.py tests/test_review_deadline.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/smoke_review_parameters.py
.venv/bin/pytest
.venv/bin/ruff check .
```

The smoke uses a deterministic mock LLM transport and real numerical fitting. It
checks global parameter compilation, inherited-role preservation, fallback fits,
chained checkpoint import, source immutability, and deterministic resume.

Changed components are the v4 revision adapter and shared compiler entry point;
continuation import, dispatch, CLI, and reporting; the parameter audit and smoke;
and their tests. Earlier protocols remain available. The numerical fitter,
benchmark data, finalized prompts, and scientific judge are outside this change.
