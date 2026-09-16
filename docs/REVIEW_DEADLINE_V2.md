# Corrected review campaign (v2)

This campaign tests whether allowing coherent scientific model revisions makes
the public-feedback visits useful. It preserves the six-cell benchmark matrix,
ablations, construction method, and frozen numerical fitting settings. It is a
new experiment, not a repair of the recorded v1 results.

## What changes

- The proposer supplies a scientific hypothesis, references to supplied evidence,
  and model content: revised equations, any new variables, observation mappings,
  and required latent initializations. It does not select an action or routing
  enum. A single revision may change several related equations.
- The runtime derives changed sources, pathways, functions, and states from the
  revised model. It rebuilds the fit request and validates the whole result
  using the restricted expression grammar and public channel contract. Undefined
  symbols, forbidden observation access, missing state initialization, and
  scientifically specified ablation violations remain errors with repair feedback.
- Revision visits allow three physical provider requests. They no longer have
  the additional cumulative 98,304 gate that mixed request bytes with reported
  tokens. Each request still has the serving context limit, output limit, and
  timeout. Actual usage is logged; interrupted requests consume their attempt.
- Existing compatible fitted parameters seed the revised fit. Newly introduced
  parameters and changed initializations are validated and initialized by the
  runtime. The optimizer, numerical tolerances, and fit budgets are unchanged.

Initial construction retains its existing budget. Removing the revision gate
does not enlarge the 32,768-token serving context or the 8,192-token requested
output limit. Unsupported or incomplete scientific content is not silently
invented by the runtime.

## Matrix and controls

The matrix has 44 lineages and three visits per lineage (132 planned rows):
initial construction and two subsequent revision/refit visits, with seeds 0 and 1.

| Family | Public presentation | Variant |
| --- | --- | --- |
| Dalla Man | Named | Canonical |
| Dalla Man | Obfuscated | Canonical |
| Dalla Man | Named | Perturbed |
| Dalla Man | Obfuscated | Perturbed |
| CSTR | Named | Canonical |
| Alien device | Functional description | Canonical |

All six cells have full, brief-only, and refit-only arms. No-latent is tested on
named canonical Dalla Man and CSTR; no-specification is tested on obfuscated
canonical Dalla Man and the alien device. Refit-only shares its full arm's
visit-zero model, then spends fitting budgets without a revision call.

The controlled intervention demonstration is also run. It illustrates how a
scientific requirement can distinguish two models that fit the observational
data equally well. It is separate from the six-cell search results.

## Fresh run and provenance

Use a fresh complete matrix in `review-deadline-v2`, including fresh visit-zero
construction. Imported v1 checkpoints would require a separate cross-version
provenance migration because the execution identity binds source and runtime.
That migration is outside this deadline milestone. The method and budgets of
visit-zero construction remain unchanged, but v1 and v2 are separate stochastic
campaigns, not an exactly paired later-visit ablation.

Preserve `review-deadline-v1` as diagnostic evidence. Its narrow repair contract
and token-accounting limitation confound its revision failure rates. Do not pool
v1 and v2 rows or silently replace v1 results. A v2 launcher refuses a v1 output
directory. A fully submitted v2 directory returns its existing submission
manifest; partial submission stops for inspection instead of creating duplicates.

The plan freezes public assets, code, runtime, and launchers. Cached LLM calls
and completed fits are reused only under their bound identity. Resume does not
grant a fresh fitting allowance to an interrupted numerical worker.

## Run on ACES

Use the exact final commit supplied with the implementation. These commands use
the public development files already staged for v1; no new Mac transfer is
needed. Run them in a subshell so a failure does not close the login shell.

```bash
(
  set -euo pipefail
  REV=REPLACE_WITH_COMMIT_FROM_IMPLEMENTATION_MESSAGE
  BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  git -C "$BASE" fetch origin codex/prefit-aces-v1
  export AF_REPO_ROOT="/scratch/user/u.yx126462/repos/autoformalism-review-v2-${REV:0:7}"
  if [ ! -e "$AF_REPO_ROOT/.git" ]; then
    git -C "$BASE" worktree add --detach "$AF_REPO_ROOT" "$REV"
  fi
  [ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" = "$REV" ]
  export AF_PYTHON="$BASE/.venv/bin/python"
  export AF_CONFIG="$AF_REPO_ROOT/configs/review_deadline_v2.json"
  export AF_PUBLIC_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1
  export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-v2
  bash "$AF_REPO_ROOT/scripts/hpc/submit_review_deadline_v2_aces.sh"
)
```

Submission first creates one CPU preparation job, one CPU controlled-demo job,
and the first proposal/fit/finish group. A small CPU dispatcher runs after the
finish job and submits the next visit; the third visit follows in the same way.
This avoids queuing all 120 fitting tasks at once under ACES's 90-job user limit.
Each queued visit has at most 44 fitting tasks. Other simultaneous campaigns can
still consume the remaining scheduler allowance.

Each proposal job requests one H100; fitting arrays request one CPU per task.
The preparation job runs contract tests and offline smoke tests before dependent
GPU work. `submission_manifest.json` is updated after each successful round
submission. `submitted_through_round` and `all_rounds_submitted` distinguish a
queued first visit from a fully submitted campaign; future job IDs do not exist
yet. Initially `round_submission_complete` is true and `submission_complete` is
false; the latter becomes true after the last visit is submitted. This launch
does not submit Delta jobs or external baseline runs.

A scheduler rejection leaves the round's intent and any known job IDs under
`submission-intent`. Repeating the launcher stops rather than duplicating an
uncertain partial submission. Inspect those IDs and the dispatcher log before
recovering a rejected submission. No new scientific or numerical budget is
created by a dispatcher.

Do not estimate the new campaign's duration from v1's fast rejected visits:
accepted revisions can consume a full fit budget. At a deadline, report planned,
completed, failed, and missing rows separately. Queue delays and GPU startup
also count against elapsed time.

## Read results

The finish jobs refresh the report. After completion:

```bash
ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-v2
cat "$ROOT/submission_manifest.json"
cat "$ROOT/SUMMARY.md"
cat "$ROOT/controlled-demo/SUMMARY.md"
```

For tables and figures use `summary.json` and `rounds.csv` in this v2 directory.
The table/figure instructions in `docs/REVIEW_DEADLINE_ANALYSIS_HANDOFF.md`
apply, substituting the v2 root and recording its plan hash and commit. Keep the
historical v1 snapshot in that document labeled as historical.

Track accepted revisions, inferred changes, actual provider calls, fit outcomes,
graph certification, and retained versus trial NMSE. Report individual seeds
and paired ablations. Three visits do not establish convergence. A poor fit
with exhausted numerical budget is qualified mismatch evidence, not proof that
the proposed equations cannot fit the data.

For revision details, read `results/<task_id>/round_XX/proposal.json`:
`decision.provenance.routes` records inferred changes, and `attempts` records
accepted or rejected replies and concrete retry feedback. These details are not
separate columns in `rounds.csv`; a figure-generation agent should join them by
task and round rather than infer a revision from a changed NMSE.

## External baselines and held-out evaluation

`docs/REVIEW_BASELINE_COMPATIBILITY.md` records the baseline inventory and common
evaluation path. The SINDy/PySR development adapter preserves the selected
train-only equations. This adapter alone does not establish matched baseline
results: exact cell coverage, frozen selections, and the common rollout
evaluation are still required. Historical conditional one-step metrics must not
be placed beside v2 open-rollout NMSE as a matched comparison.

This submission uses public training and validation only. Benchmark held-out
evaluation requires sealing the campaign selections first and remains a
separate stage. Neither test trajectories nor held-intervention outcomes enter
proposal generation or model selection.
