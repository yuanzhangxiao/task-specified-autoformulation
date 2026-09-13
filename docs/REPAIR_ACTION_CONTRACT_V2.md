# Repair action contract v2: typed pending work and evidence-based reporting

This is the next opt-in milestone after the ACES `af465d9` runtime-only run.
That run recorded 21 edit/keep conflicts and four unnamed initializer restrictions.
Version 2 changes the action interface, draft handling, and numerical reporting;
it does **not** change the optimizer, budgets, source candidates, public data,
scientific judge, routing priority, or production defaults. The existing judge
job must continue in its original pinned checkout/output root.

## Provider contract

`RepairActionV2` generates the actual strict provider schema. Existing fields
`scope`, `hypothesis`, `equations`, `remove`, `mappings`, and `initializers` retain
their meanings and bounds. The redundant `keep` inventory is absent. Unmentioned
components of the committed model remain unchanged. A new optional list is:

```json
{"withdraw": [{"kind": "initializer", "target": "z"}]}
```

`kind` is `equation`, `mapping`, `initializer`, or `remove`. A withdrawal cancels
an existing **draft operation**; it does not delete the model variable. It must
name a draft operation and cannot coexist with replacing the same operation in
that response. Removing a committed variable still requires `remove` and model
scope. Editing and removing one variable together remains contradictory.

Omission on retry retains pending work. Thus a response with `initializers: []`
does not silently cancel a previous initializer. Every pending entry identifies
its action kind, target, and named diagnostic. A provider either replaces that
entry, explicitly withdraws it, or returns `scope: no_change` to abandon the whole
transaction. An invalid entry is never silently removed just to pass validation.

Valid equation edits survive failures in other actions. Initializer failures
name the state and eligible latent states, and never blame an otherwise valid
equation. A later mapping/type change may make a pending initializer valid;
eligibility is checked against the **resulting** model, not just the parent.
`causal_map: null` requests a shared training-fitted initial value, not cancellation.
Observed initial boundaries, training/validation separation, and causal-map rules
are unchanged. Numeric initializer guesses remain runtime-owned.

Function scope remains strict even if an older provisional edit used model scope.
Each provisional equation retains the scope that authorized it. Whole-model
closure, target availability, domains, shared parameter roles, input paths and
public structural obligations are checked before one atomic commit. Global
dependency failures remain transaction-level; the runtime does not fabricate an
equation-level scientific cause. Pending checkpoints bind `repair-action-2`;
version-1 checkpoints cannot silently resume under the new code.

## Offline historical replay

`scripts/replay_repair_actions_v2.py` reads the old comparison-1 plan, input-plan
and frozen candidate digests, cached request identities and transactions. It
verifies the public asset ledger and reconstructs the original validation context
from public training/validation data (including supplied-channel bounds). It does
not fit, call a proposer/judge, or open private/test data. It records source file
hashes and refuses changed output on replay.

Each reply is checked against its historical parent and captured provisional
context. Only **actual historical commits** advance the parent; admissible replay
alternatives never become invented future search states. Short parameter aliases
are reversed before checking, without changing the mathematical expressions.

Two result views are mandatory:

1. `conservative`: harmless nonoverlapping legacy keep entries are removed with
   an audit; an edit/keep overlap remains ambiguous and is rejected.
2. `keep_inventory_assumption`: an overlap is interpreted as retaining the
   variable while editing its equation. This assumption is logged explicitly.
   Unknown kept names and keep/removal conflicts still fail.

`committed` here means statically admissible under that view, not a newly fitted
candidate, scientific improvement, or avoided live call. `pending` retains named
unresolved operations. A historical reply omitting an invalid initializer does
not retroactively count as an explicit withdrawal. Old provisional scope was not
fully recorded; replay uses the previous captured response scope (function when
unavailable) and does not claim exact counterfactual search.

Run replay to a new file **outside** the source root. No source artifacts are
rewritten. This is a CPU/local operation; it needs no GPU job or model download.

## Fitter reporting, not fitter changes

The feasibility wrapper's top-level `optimizer_*success` flags are placeholders.
Version 2 reads the nested sensitivity/poll stages and reports separately:

- finite candidate / production training replay;
- any stage's native termination and verified optimizer success;
- whether the selected parameter vector matches a converged, verified stage;
- initializer status/message, selection provenance, finite evaluation count;
- per-stage messages including sensitivity unavailability and timeout.

Missing stage evidence remains unknown. A finite point retained after timeout
does not count as convergence. A converged stage at a different parameter vector
does not certify the selected vector. `verified_optimizer_success` additionally
requires production training replay. `fit_status: complete` retains its original
meaning: finite production training and validation rollouts. Scientific quality
and good predictive accuracy are not implied. The same compact stage evidence
is supplied to numerical feedback, never to the numerically blind judge.

## Verification and next live experiment

Run `tests/test_repair_comparison.py`, `tests/test_repair_drafts.py`, and
`tests/test_repair_action_replay.py`, then the real CPU fitter smoke
`scripts/smoke_repair_feedback_comparison.py`. The ACES preparation job runs
these tests automatically before the GPU dependency is released.

Local verification: 74 focused tests plus the fitter CLI deterministic-resume
regression passed (75 total); Ruff, shell syntax, diff checks, and the real CPU
smoke passed. The smoke recovered training/validation NMSE below `4e-16` with
identical resume. Full pytest: 1,404 passed, three optional skips, and 39 failures
from absent benchmark fixtures in this checkout. No fixtures were synthesized or
benchmark files changed to hide those failures. Historical ACES replies still
require the read-only replay on their actual frozen source root.

Use the existing split launcher in a **new clean checkout and new output root**.
It freezes `repair-feedback-comparison-2`, action/reporting protocol identities,
the same 20B model/SIF and fitter configuration. Start only `AF_ARM=redesigned_runtime`
(one H100, two seeds, four rounds). A future matched judge arm can be submitted
separately with the same new plan; the old judge arm is not its matched control.
No historical cache is reused as if it answered the changed provider request.

The earlier image-discovery, module-loading, explicit node/task allocation,
independent-arm manifests, scheduler ambiguity guards and resume behavior remain.
There is no automatic image rebuild or hash replacement.

Limits: general cross-model scientific ambiguity and global dependency failures
can still require retries; domain certificates remain incomplete; numerical
budgets may still expire. This milestone addresses interface failures, not the
attainability of an arbitrary proposed model or the optimal repair policy.
