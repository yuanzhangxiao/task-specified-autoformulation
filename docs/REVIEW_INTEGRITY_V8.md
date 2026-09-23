# Explicit parameter repair and stable selection

`review-deadline-8` continues an immutable fitted checkpoint with two corrections
to protocol 7. It retains the response summaries, public requirements, fitter,
three-attempt proposal budget, and incumbent fallback policy. Earlier protocols
retain their historical declaration and selection semantics. Existing checkpoints
are imported exactly; this is not a retrospective rewrite of the earlier search.

## What the public T2 task requires

The frozen canonical named T2 easy and hard prompts require:

1. Meal timing and amount to contribute causally to plasma glucose.
2. A delayed insulin-action pathway regulating insulin-dependent disposal.
3. An explicit continuous-time model generating every target, including insulin.

They do **not** explicitly require meal- or glucose-dependent insulin production.
The delayed-action requirement describes insulin's downstream effect; it does not
specify the upstream production law for insulin. No new dependency check or hint
has been added. The proposer must infer additional dependencies from the public
training observations and scientific context. Training response summaries continue
to show observed versus predicted responses for all targets without inserting a
reference equation or a required cause.

This interpretation was checked against `brief.scientific_context` and the
public contracts in the downloaded final-v7 plan
`ae50b5e87ded15a1e0752e8ebeaeb5ca0e9e53eb4e7435b49eda4aa8b38d0f98`.
Neither benchmark assets nor prompt files change in this milestone.

## Parameter declaration correction

Policy: `reject-conflicting-existing-role-1`.

Both an existing canonical name and its displayed alias identify the same
parameter. Referencing either still inherits the exact declaration. A redundant
declaration with the same role, or no explicit role, remains admissible. An
explicit conflicting role now produces `EXISTING_PARAMETER_ROLE_CONFLICT` before
the transaction commits or any child fit starts. This also applies when a
canonical name happens to look like a displayed alias, such as `par_004`.

The repair feedback identifies the supplied name, canonical name, displayed
alias, existing role and requested role. It instructs the proposer to either:

- omit the redundant declaration to reuse the existing quantity; or
- choose a fresh name, declare its role, and change the intended references to
  represent an independent quantity.

The runtime never guesses a new name or silently changes the scientific meaning.
The incumbent remains unchanged after rejection. Existing parameters are still
eligible for training refitting; inheriting a declaration does not fix its value.
This correction does not infer dimensional incompatibility from arbitrary uses
of an undeclared existing reference.

## Validation selection correction

Policy: `validation-tolerance-then-terms-1`. For incumbent and trial validation
NMSEs `a` and `b`, use the symmetric fixed band

```text
delta = 1e-8 + 1e-6 * max(abs(a), abs(b))
```

Outside the band, the lower validation NMSE wins. Inside the band, fewer outer
additive terms win; equal term counts retain the incumbent, including its fitted
parameters. Complexity uses the same state-equation and process term count as the
previous controller. No new complexity metric is introduced. Missing,
ineligible or nonfinite trial scores cannot replace an eligible incumbent.
Training NMSE does not rank candidates; both development rollouts must be
available. Test observations and test metrics are not used.

The band is a numerical decision convention, not an estimated uncertainty or
scientific-equivalence certificate. It is deliberately small and fixed across
tasks. A simpler model can be selected with a slightly higher validation error
inside the band. It cannot prevent repeated validation selection from overfitting,
and it does not establish that an extra state is scientifically unnecessary.

The immutable plan records the policy and constants. Every new round and summary
row records `selection_audit`, including scores, term counts, comparison band,
decision and reason. Recovery publication uses the same selection function.

## Verification on the saved replies

Without new LLM calls or fitting, the strict compiler rejected all four saved
conflicts in the final v7 archive: T1-hard seed 0 rounds 15–17 and T2-easy seed 1
round 16. These are admissibility replays, not counterfactual fresh repair results.

For T2-hard seed 0 round 16, the archived trial changed validation NMSE from
`0.9778491472623487` to `0.9778491472283518`, while increasing additive terms
from 9 to 11. The new comparison keeps the incumbent: the `3.40e-11` improvement
is within the `9.88e-7` band. Historical result files were left unchanged.

## Local checks

Run from the checkout containing this change:

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_review_integrity.py tests/test_response_revision.py \
  tests/test_review_parameters.py tests/test_review_multi.py

PYTHONPATH=src:. .venv/bin/python scripts/smoke_review_response.py \
  --protocol review-deadline-8 --output artifacts/review-integrity-smoke
```

The smoke uses real synthetic three-target CPU fits, a mocked proposer and
tokenizer, a conflicting declaration followed by explicit correction, delivery
failure preservation, source immutability and exact resume. It makes no live LLM
calls and opens no test data.

## Optional ACES continuation

Use a fresh clean checkout pinned to the commit reported with this change.
Do not update a checkout used by submitted jobs. The source is the completed v7
campaign, global round 17; the destination below is new. This imports all six
incumbents and runs three new visits (global rounds 18–20). It does not recover
previously rejected candidates or remove a state already retained by v7.

```bash
module load GCCcore/13.2.0 Python/3.11.5
export AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-review-integrity-v8
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export AF_SOURCE_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-response-feedback-r14-v1
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-review-integrity-r17-v1
export AF_SOURCE_ROUND=17 AF_ADDITIONAL_VISITS=3
bash "$AF_REPO_ROOT/scripts/hpc/submit_review_integrity_aces.sh"
```

The prepare CPU job runs the targeted regressions, the v8 synthetic smoke and
training-response preview before any H100 proposer job. Existing caches, serving
image and Python environment follow the established continuation launcher.
The submission receipt is the authority for job IDs. To inspect progress:

```bash
jq '{protocol,commit,global_round,additional_visits,all_rounds_submitted,jobs}' \
  "$AF_OUTPUT_ROOT/submission_manifest.json"
PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" \
  "$AF_REPO_ROOT/scripts/review_deadline.py" report --root "$AF_OUTPUT_ROOT"
jq '{status_counts,proposal_status_counts,missing:(.status_counts.missing // 0),delivery_failure_visits}' \
  "$AF_OUTPUT_ROOT/summary.json"
jq '[.rows[] | select(.selection_audit != null) | \
  {task_id,round,selection_audit}]' "$AF_OUTPUT_ROOT/summary.json"
```
