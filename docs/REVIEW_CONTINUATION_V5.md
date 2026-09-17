# Continue from round 12 with advisory revision size

The completed sign-corrected v4 phase ran rounds 10–12. Of 90 revision visits,
60 returned accepted revisions, 29 failed, and one returned no change. Sixteen
accepted-revision trials replaced the incumbent; retention can include tiny
score improvements or complexity tie-breaks. The 29 terminal failures were:

| Cause | Visits |
| --- | ---: |
| Accumulated model exceeds construction limits | 22 |
| Public requirement or ablation check | 5 |
| Missing explicit initializer for a new latent state | 1 |
| Unused new parameter declaration | 1 |

The two shown requirement failures introduced hidden dynamic states into the
no-latent control. That restriction remains valid. The data do not establish
that every other requirement failure is an ablation violation.

## Revision policy

`review-deadline-5` imports each lineage's retained round-12 model into a fresh
phase. The default experiment below runs **three visits, rounds 13–15**. The
44-lineage matrix, two closed lineages, model settings, numerical budgets,
initialization protocol, and validation selection rule are inherited. This is
continued development with another explicit protocol boundary, not a single
unchanged-protocol convergence or ablation experiment.

The initial-construction values (12 generated variables, eight outer additive
terms per equation, 32 total terms) are **advisory during revision**. A valid
revision is no longer rejected solely for exceeding them. They are historical
reference values, not scientific bounds. The runtime reports dynamic/algebraic
variable counts, terms per equation, total terms, equation parameters, and fitted
parameters, including initialization parameters. Counting uses the existing
outer additive shell; these counts are descriptive and depend on expression
representation, not a representation-invariant measure of model complexity.

The proposer sees these counts before revision. Each accepted revision records
before/after counts and every exceeded reference. The prompt still asks for
parsimony. Scientific requirements, expression grammar, parameter domains,
target generation, and ablation checks remain active. Per-reply limits of six
equation definitions and two new variables, response sizes, three physical LLM
attempts, and one fitting allocation per visit remain bounded. A larger model
receives no extra numerical budget and can still time out or fit poorly.

The no-latent control explicitly advertises that only states directly mapped to
public targets may have ODEs. New algebraic processes remain allowed. The runtime
does not silently reinterpret a hidden ODE as an algebraic law. Public requirements
and post-compilation control checks still determine eligibility.

New declarations that are genuinely unused are removed with an audit record.
Expressions are parsed through the restricted grammar; cleanup never invents
missing symbols or changes equations. References in output mappings and initial
causal maps prevent an erroneous unused-parameter deletion. Initial-map parameters
still require local declarations; a misplaced global declaration receives an
explicit scope diagnostic rather than an inferred sharing relationship.

A new latent dynamic state still requires an explicit initializer. Error feedback
now supplies the concrete `causal_map: null` option for a shared train-fitted
initial value, or asks for a causal map. The runtime does not choose one silently.
Observed-state initials remain observed; no validation initial values are fitted.

## Preflight and diagnostics

The preparation job runs focused tests and a mock-LLM smoke with real fitting.
It also replays saved replies from the source phase with the new contract, using
their original parents. This opens no test data, calls no LLM, performs no fitting,
and never promotes a historically rejected proposal into a retained model.

`parameter_response_audit.json` reports response-level outcomes separately from
failed visits with at least one newly valid reply. It exposes any previously
accepted reply now blocked, exact terminal messages, size audits, and declaration
cleanup. An interface recovery is not evidence of improved fit or scientific
correctness. Historical requirement failures and missing initializers may remain.

`summary.json` and `rounds.csv` include trial and retained complexity, terminal
revision diagnostics, and trial-retention flags. `revision_diagnostics.json`
summarizes exact error messages and counts, accepted/retained revision visits,
and fitting visits above original size references. `REQUIREMENT_REVIEW.md`
exports unresolved scientific checks for the imported models, without certifying
them. The inherited CSTR ambiguity still needs an equation-based review.

## ACES commands

Use the exact commit from the implementation response. This reuses an existing
authenticated clone and keeps all completed campaigns and pinned checkouts intact.

```bash
(
  set -euo pipefail
  REV=REPLACE_WITH_REPORTED_COMMIT
  BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  git -C "$BASE" fetch origin codex/prefit-aces-v1
  export AF_REPO_ROOT="/scratch/user/u.yx126462/repos/autoformalism-review-v5-${REV:0:7}"
  if [ ! -e "$AF_REPO_ROOT/.git" ]; then
    git -C "$BASE" worktree add --detach "$AF_REPO_ROOT" "$REV"
  fi
  [ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" = "$REV" ]
  export AF_PYTHON="$BASE/.venv/bin/python"
  export AF_SOURCE_ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v4-signfix
  export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v5
  export AF_SOURCE_ROUND=12 AF_ADDITIONAL_VISITS=3
  export AF_CONTINUATION_PROTOCOL=review-deadline-5
  bash "$AF_REPO_ROOT/scripts/hpc/submit_review_continuation_aces.sh"
)
```

One H100 serves proposal work; fits use the same CPU array and existing per-task
resources. There is no Delta work or automatic held-out evaluation. The helper
queues subsequent visits only after prior results exist, retains uncertain
scheduler receipts for inspection, and stops after round 15.

After preparation:

```bash
ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v5
jq '{status_counts, failed_visits_with_saved_attempts,
     failed_visits_with_newly_valid_reply, previously_accepted_now_blocked,
     source_terminal_messages}' "$ROOT/parameter_response_audit.json"
```

During or after the run:

```bash
cat "$ROOT/submission_manifest.json"
sacct -j "$(jq -r '[.jobs[]] | unique | join(",")' "$ROOT/submission_manifest.json")" \
  --format=JobID,JobName%30,State,ExitCode
cat "$ROOT/SUMMARY.md"
cat "$ROOT/revision_diagnostics.json"
```

For a larger-model failure, inspect the saved trial size and numerical status
before changing the protocol. Neither successful execution nor more accepted
revisions establishes better scientific models. Future choices about complexity
penalties or additional numerical resources are separate experiments.

## Verification

```bash
.venv/bin/pytest -q tests/test_review_revision_v5.py tests/test_review_parameters.py \
  tests/test_review_continuation.py tests/test_review_deadline_v2.py tests/test_review_deadline.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/smoke_review_revision.py
.venv/bin/pytest
.venv/bin/ruff check .
```
