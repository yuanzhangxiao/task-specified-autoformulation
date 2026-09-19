# Milestone 2: shared-process guidance pilot

This is an opt-in development comparison, `shared-process-pilot-1`. It uses the
existing algebraic-variable and hyperedge representation, the existing equation
compiler, and the frozen `collocation-single-target-v2` fitter. No new judge,
pruning method, automatic parameter tying, or held-out test access is added.

## What changes

The guidance arm distinguishes memory states from reusable algebraic laws:

- Variables: a scientifically useful algebraic law may be introduced even when
  it could be written inline. No independent initial condition is invented.
- Topology: consumers refer to that law; roles distinguish assembly, transfer,
  conversion and response. Shape descriptions belong to the defining law, not a
  second nonlinearity applied by every consumer.
- Functions: define the law once, retain its parameter identity through reuse,
  and introduce consumer conversions/gains only when scientifically needed.
- Revision: reconsider the complete affected equations using
  `scientific-content-revision-6`. The six-equation/two-new-variable patch quotas
  are removed in **both** guidance and control arms.

Guidance is optional modeling advice, never an added benchmark requirement. It
allows signed laws, necessary independent response gains and reduced models with
no shared process. It does not infer conservation from opposite signs. It does
not automatically transform fitted candidates or certify unit conversions.
The historical control prompt is unchanged; old campaign protocols still dispatch
their original revision schemas.

Both arms use initial construction capacities of 64 generated variables, 32 terms
per equation and 512 terms total (the lowering representation has a 512-interaction
capacity). These are explicit resource capacities, not desired model sizes.
The model serializer retains its 64-state/256-process/256-parameter capacities,
and the expression grammar and provider limits remain unchanged. A larger model
receives no extra requests or fitting time. Broader initial capacities prevent
optional algebraic laws from competing under the former 12-variable agenda cap.
This control is therefore a fresh matched control, not a replay of historical
round-15 results.

Local verification (2026-09-19): full `pytest` completed with 2,416 passed and
four optional Torch-dependent tests skipped. The prescribed-provider smoke
completed two real fitting rounds and resumed without repeating fitting or
provider calls. Changed-file lint, shell syntax and whitespace checks passed.
Repository-wide `ruff check .` still reports 37 pre-existing findings in the
unrelated `analysis/claude` scripts. Live proposer quality remains to be tested.

## Frozen comparison

| Factor | Values |
| --- | --- |
| Development cells | Canonical obfuscated Dalla Man, named CSTR, functional alien device; all easy |
| Seeds | 0, 1 |
| Initial information | `full`, `brief_only` |
| Shared-process guidance | off, on |
| Rounds | 0: fresh construction and fit; 1: one coherent revision and fit |
| Planned work | 24 lineages, at most 48 numerical fitting attempts |
| Proposer | The historical deadline campaign's pinned GPT-OSS-20B, one H100 |
| Scientific LLM judge | Off, unchanged for this comparison |
| Numerical method | Frozen `collocation-single-target-v2`, one CPU per fit |

Guidance order alternates within cell/seed/prompt pairs. Seeds, model revision,
provider settings, construction capacities, deterministic checks, revision schema,
fitting allowances and validation/complexity selection are otherwise identical.
Each pair has separate content-addressed request caches; matching seeds do not
imply identical provider responses to different prompts.

`full` supplies descriptive training observations during initial construction;
`brief_only` omits them. Both receive qualified training residuals for revision.
Each revision has three physical attempts and the same per-call output/context
limits, with no cumulative revision-token gate. Initial construction retains its
128-request / 524,288-token budget and 8,192-token per-response cap. These limits
match across arms and are recorded; usage and unknown-usage charges remain visible.

The established controller retains a valid incumbent and can spend the round's
existing fit allocation on an unchanged refit if revision fails or requests no
change. This is reported as fallback, not successful revision. No additional
fitting budget is created. Missing construction, missing residual evidence,
failed fitting, numerical stopping and scientific checks remain separate.

The pilot deliberately does not multiply by all six cells or fifteen rounds.
It is a feasibility/effect-size screen over three families and two seeds, not a
final method-selection test. Broader Dalla Man contrasts follow only if useful.

## What to read

`SHARED_PROCESS_SUMMARY.md` and `shared_process_summary.json` contain every planned
row. `summary.json`, `rounds.csv` and `revision_diagnostics.json` retain ordinary
campaign reporting. Per-call requests and responses remain under each round's
`calls/`; full equation bundles remain in sealed proposal/result files.

Compare, by cell/seed/prompt pair and round:

1. Construction success and deterministic target/mechanism predicates; do not
   combine these into an overall scientific correctness percentage.
2. Output train/validation NMSE, trial versus incumbent selection, and fallback.
3. Actual parameter, state and algebraic-process counts, plus repeated-law
   witnesses. The report separates sharing between governing definitions from
   sharing that merely includes an observation mapping.
4. Cumulative token/request and fit-attempt counts, with unknown token usage
   separate. Token totals are not silently treated as exact when usage is absent.
5. Human review of whether each reused law has a sensible scientific meaning.
   More process names is not a success criterion. Factoring an already identical
   expression does not reduce parameter count or improve its predictions.

No automatic winner or follow-up is selected. Scientific claims and physical
balance relationships still require review beyond syntax. The existing local
function obligations, inventory-revision routing limitations, nonlinear grammar
and causal-initialization rules remain; this milestone does not add a general
symbolic unit/conservation verifier or change the calibrated judge.

For the two round-0 empty-parameter-vector execution failures, use the separate
[parameter-free recovery runbook](SHARED_PROCESS_FIXED_MODEL_RECOVERY.md). It
preserves this campaign and imports its results into a corrected execution root
before any revision round is submitted.

## ACES commands

Use group scratch for source, outputs and new runtime caches. This avoids the
previous user-scratch file-count exhaustion. No new virtual environment or model
copy is needed. No directory or result is deleted by these commands.

First set `AF_COMMIT` to the full pushed commit supplied with this milestone.
Then run the following **on ACES** (in a subshell so an error does not close the
login session):

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?set the full milestone commit first}"
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/shared-process-${AF_COMMIT:0:7}"
  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  export AF_PUBLIC_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1
  export AF_OUTPUT_ROOT="$AF_GROUP/shared-process-pilot-v1"
  bash "$AF_REPO_ROOT/scripts/hpc/submit_shared_process_pilot_aces.sh" 0
)
```

If the development files have moved, set `AF_PUBLIC_ROOT` to the directory that
contains `phase_b_v1/<cell>/manifest.json`, `proposer_prompt.txt`, `train.csv`, and
`validation.csv`. The launcher checks all required files before copying or
submitting anything. Use the same public dataset version for both arms.

Round 0 queues a CPU preflight (tests, real small-fit smoke and serving-image
hash), one H100 proposer job, up to eight concurrent one-CPU fit tasks, and a
report job. The next round is **not automatically submitted**. Submission intent
and every scheduler reply are saved; an ambiguous reply is never retried blindly.
An identical completed submission command returns its existing receipt.

After those jobs finish:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/shared-process-pilot-v1
cat "$ROOT/submission_manifest.json"
cat "$ROOT/SHARED_PROCESS_SUMMARY.md"
jq '{status_counts, proposal_status_counts}' "$ROOT/summary.json"
jq '{terminal_errors}' "$ROOT/revision_diagnostics.json"
```

The round-1 rows will still be `missing` at this point. Once the first-round
handoff is satisfactory, submit the revision round with the **same commit/source**:

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?use the same milestone commit}"
  export AF_REPO_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/repos/shared-process-${AF_COMMIT:0:7}
  export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/shared-process-pilot-v1
  bash "$AF_REPO_ROOT/scripts/hpc/submit_shared_process_pilot_aces.sh" 1
)
```

Every round-0 task must have a sealed terminal result before round 1 is queued.
A numerical/construction failure can be a terminal result; a missing file cannot.
Do not delete consumed markers or restart fitting to cure an infrastructure
failure. Preserve the receipt/logs for explicit recovery instead.

After round 1, read the same reports and share `shared_process_summary.json` and
`revision_diagnostics.json`. This pilot has no test-set command.
