# Typed action feedback: 20B confirmation

The pending 120B focused retest is useful for comparing model sizes, but is not
a prerequisite for correcting an independently demonstrated feedback defect.
This milestone leaves that experiment and its frozen bundle unchanged. It does
not draw a 20B-versus-120B conclusion.

## Evidence and scope

The 27 October 20B focused retest completed all 15 calls. Nine of twelve
follow-ups repeated unchanged failures. Two attempted to delete invalid pending
parameter-as-equation actions merely by omitting them; explicit delta semantics
correctly retained them. Several other errors received generic full-patch advice
that was poorly aligned with the selected-branch response contract.

The opt-in `typed-branch-action-feedback-1` policy changes feedback only:

- A parameter accidentally listed under equation `edits` receives its exact
  `drop_actions` field/key. The advice is conditional on intending a trainable
  parameter, rather than a fixed-value condition or a generated variable.
- A new definition placed under existing-component edits receives the appropriate
  declaration location and explicit cleanup. A genuinely unknown symbol requires
  the proposer to choose its role; names such as `k_absorption` do not decide it.
- An invalid initializer edit shows eligible targets and distinguishes existing
  latent initialization from a new state's `initial` field. Other permitted RHS
  edits remain available; fixed observed initials remain protected.
- A qualified model-data conflict explicitly says the pending mathematical patch
  has already been tested and rejected. Reapplying it or changing its prose does
  not repair the witness. The proposer chooses which mathematical premise to change.

Only designated instruction fields change. Scientific formulas, quoted evidence,
schemas, system text, candidate composition, selection, repetition detection and
acceptance checks remain unchanged. Initial/fresh requests are unchanged. Neither
deleting an invalid record nor fixing its serialization guarantees eligibility.

The zero-meal glucose-balance concerns in the equation audit are not converted
into a new hard gate here. Eligible models can still have deferred scientific
checks, and this experiment does not evaluate NMSE or establish scientific validity.

## Saved-response verification

Replay uses all 15 saved provider replies and 27 candidate records from archive
`22bff021a93ba3f518eae383cbdb8cb58b34d0d9fd622d0618be56741e503396`.
Historical responses are deliberately substituted into the revised requests to
check deterministic software behavior. They are **not** responses obtained from
the new prompts and do not measure whether the new advice helps a model.
Candidate composition, outcomes and checks must remain unchanged. Prospective
requests and their differences are retained for human review. No new generation,
fitting, rollout or test access is performed by this replay.

Completed offline verification: all 27 complete candidate records and all episode
outcomes/routing are identical, excluding the new policy label. Three initial
requests are byte-identical; twelve follow-up requests change. All 13 displayed
drop fragments validate against their typed schema and corresponding pending
actions. CPU Harmony sizing exactly matches 15 saved server tokenizations; the
revised requests peak at 20,844 input tokens (24,000 limit), adding at most 356
tokens per request. Live server preflight remains authoritative. The integrated
construction/launcher regression selection passes 306 tests; changed-file Ruff
and shell syntax checks pass. Whole-tree Ruff retains unrelated existing findings.

## ACES confirmation

The separate `action-feedback` profile is 20B-only. It uses the same frozen
P00-r1 parent and logical seeds 0, 1 and 2 as the focused retest, with five initial
alternatives and at most four shallow follow-ups per episode: at most 15 calls.
The 8,192-token output and 32,768-token context budgets are unchanged. Model-call
seeds retain the existing derivation from logical seed, stage and attempt; a new
cache namespace does not alter that derivation. Serving can remain nondeterministic.

The profile requests one H100 and uses the fresh root:

```
/scratch/group/p.nairr260351.000/u.yx126462/phase-c-search-action-feedback-20b-v5
```

Upload the delivered source archive to ACES group scratch and extract it there.
Set `AF_CODE` to the exact extracted bundle directory. Run each command separately:

```bash
bash "$AF_CODE/run_search.sh" run 20b full wave1 action-feedback
bash "$AF_CODE/run_search.sh" status 20b full wave1 action-feedback
bash "$AF_CODE/run_search.sh" inspect 20b full wave1 action-feedback
```

`full` here means only the three-episode confirmation. An optional `smoke` runs
the first episode within that same budget. Inspection creates `inspection.tar.gz`
inside the new result root. Submission identity, checkpoints, caches and charges
are retained on resume. Reconcile ambiguous scheduler replies before using a new
wave name. The profile rejects 120B; the pending v4 120B job continues separately.
No automatic follow-up or fitting is launched.

Assess the returned initial alternatives separately from follow-up repairs,
including repeated actions, successful metadata repairs, newly exposed scientific
conflicts and deferred checks. Three seeds on one parent are a targeted diagnosis,
not a general benchmark or proof that breadth/depth settings are optimal.
