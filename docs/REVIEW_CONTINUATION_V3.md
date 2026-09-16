# Continue development from the retained round-two models

This is a new, explicitly versioned development phase. It imports each lineage's
retained result at v2 round 2 and performs **five additional visits**, numbered
3 through 7 in the reports. The original campaign is never overwritten. An
imported checkpoint is not a new construction, provider call, or fitting attempt.
The two lineages without a retained model remain unavailable; they are not
silently reconstructed. Closed lineages with a valid incumbent reopen.

Each lineage retains its own model and learned parameters, including all causal
initial-map coefficients. Full and refit-only do not exchange incumbents. The
44-lineage roster and all ablations are preserved. The source model, revision,
reasoning effort, context/output limits, fitter profile, solver settings and
per-fit budget are inherited from the original plan. For this particular source,
the proposer is GPT-OSS-20B, low reasoning, one H100. No model upgrade is implicit.

The fitter remains `collocation-single-target-v2`: 120 seconds of collocation
and 180 seconds of refinement per numerical attempt, with the existing separate
setup/scoring supervision. This phase does not reopen fitting-method research.
The scientific judge remains off. Parameter fitting and proposer evidence use
training only; incumbent selection uses validation. No test or private-reference
evaluation is submitted or performed by these commands.

## Revision interface

The proposer returns scientific content, not a controller action category:

```json
{
  "hypothesis": "A slower memory response may explain the late residual.",
  "evidence_refs": ["E001"],
  "equations": [{"component": "m", "expression": "-rate*m+gain*u01"}],
  "remove_variables": [],
  "output_expression": null,
  "initializers": []
}
```

This example's existing variable/parameter names must come from the actual
request; it is not a required scientific model. New states/processes still need
explicit dynamic/algebraic type, and new latent states require initialization.
The runtime owns the public target identifier. `output_expression` changes its
generating expression; it cannot mistake an internal process for an output
channel. `remove_variables` is for modeled variables, not coefficients or JSON
field names. Complete equations replace earlier definitions; unused parameters
are already removed by the model compiler.

Evidence is rendered using a packet-local short catalog. Summaries, windows,
and actually available detailed samples have distinct references. Unknown or
missing citations become explicit warnings rather than discarding an otherwise
executable proposal. They never become verified references and never certify a
scientific assertion. A warning does not bypass syntax, closure, channel access,
parameter-role, initial-condition, public-requirement, or ablation checks.

Identical duplicate definitions are collapsed. Removing and redefining the same
variable means replacement, with an audit entry. A request to remove a known
parameter is redundant only if the rebuilt model really no longer uses it.
Conflicting definitions, unresolved symbols, removal of unknown variables, and
scientifically inadmissible changes still require correction. Internal nonlinear
parameter roles remain scientific declarations; runtime does not indiscriminately
force them positive. All normalization and warning details are saved.

The old v1/v2 citation policy and response schema retain their original behavior.

## A rejected proposal no longer ends a valid lineage

Every new visit allows at most three physical proposer requests, with the same
existing per-request limits. After those requests:

1. An admissible changed model gets this visit's one fitting allocation.
2. An empty/no-change reply, exhausted revision repair, or unavailable residual
   packet uses that same allocation to refit the incumbent from all retained
   parameters.
3. A worse or numerically failed challenger cannot erase the incumbent. The
   next planned visit remains available whenever an incumbent exists.

There is no second numerical attempt after a changed model consumes its fit
budget. Resuming an interrupted attempt does not grant another allowance.
Five new visits is the complete authorized batch; nothing automatically extends
it. Refit-only uses the same per-visit fit profile without proposer calls.

`status`, `proposal_status`, `fit_trigger`, and `citation_status` are separate
report columns. A successful fallback fit cannot hide a failed proposal. NMSEs
are reported for both trial and retained candidates. Retained curves are
monotone in validation by selection and do not prove convergence.

## Run on ACES

Use the exact commit supplied in the implementation response, and the existing
authenticated clone. The helper preserves the source directory, verifies hashes,
and prepares a separate destination. `AF_ADDITIONAL_VISITS` can be 1–5 at initial
preparation; it cannot be changed after the new phase is frozen.

```bash
(
  set -euo pipefail
  REV=REPLACE_WITH_REPORTED_COMMIT
  BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  git -C "$BASE" fetch origin codex/prefit-aces-v1
  export AF_REPO_ROOT="/scratch/user/u.yx126462/repos/autoformalism-review-v3-${REV:0:7}"
  if [ ! -e "$AF_REPO_ROOT/.git" ]; then
    git -C "$BASE" worktree add --detach "$AF_REPO_ROOT" "$REV"
  fi
  [ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" = "$REV" ]
  export AF_PYTHON="$BASE/.venv/bin/python"
  export AF_SOURCE_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-v2
  export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v3
  export AF_SOURCE_ROUND=2 AF_ADDITIONAL_VISITS=5
  bash "$AF_REPO_ROOT/scripts/hpc/submit_review_continuation_aces.sh"
)
```

The first CPU preparation job runs local contract tests, a real-fitting offline
smoke, and an offline replay of the original saved revision responses under the
new checks. The replay changes interface field names without guessing scientific
intent, preserves blocked outcomes, and calls neither an LLM nor the fitter.
Its result is `source_response_audit.json`; passing content checks is not a claim
that the replayed equations fit well. Failed audited proposals do not fail the
preparation job; execution/provenance errors do.

One H100 proposer job and a CPU fitting array are queued per visit. A CPU finish
job reports the outcomes; its dispatcher queues the next visit only after all
prior result records exist. Submission uses real script files through the ACES
wrapper, not `sbatch --wrap`. It needs no dependencies on old v2 scheduler IDs.
Every raw scheduler reply and accepted ID is retained. Re-running a successfully
submitted visit returns its manifest; uncertain partial submission stops for
inspection instead of duplicating jobs. The first command queues only the first
visit and its dispatcher, so `all_rounds_submitted=false` is initially expected.

The preflight runs before GPU use. A source campaign already sealed for held-out
evaluation cannot be reopened by this importer. A missing source checkpoint is a
preparation error, not permission to reconstruct or grant a new source fit budget.

## Inspect progress and results

```bash
ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v3
cat "$ROOT/submission_manifest.json"
sacct -j "$(jq -r '[.jobs[]] | unique | join(",")' "$ROOT/submission_manifest.json")" \
  --format=JobID,JobName%30,State,ExitCode
cat "$ROOT/SUMMARY.md"
jq '{status_counts, llm_calls, fitting_calls}' "$ROOT/source_response_audit.json"
```

`summary.json` and `rounds.csv` contain global `round` (2–7), local `phase_round`
(0–5), trial and retained NMSE, selection origin, proposal failures, fallback
fits, citation warnings, and incremental costs. Local zero is the imported
checkpoint. When inspecting files or using the CLI, phase rounds name directories:
`results/<task>/round_01` is global round 3. Historical costs are excluded from
incremental counters; use the original v2 report for earlier spending.

Because the revision and continuation policies change after round 2, label plots
with that transition. This is an exploratory continuation, not an unchanged-
protocol convergence experiment or a clean new end-to-end ablation. No improvement
is guaranteed; equations, fit quality, public requirements, and later intervention
results remain distinct outcomes.
