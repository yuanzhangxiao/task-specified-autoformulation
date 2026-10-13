# Concrete topology edit feedback

Policy: `minimal-edit-feedback-1`. Study: `edit_feedback_confirmation`.
This milestone changes conditional editing feedback only. It is implemented for
review; no new live experiment has been launched.

## Evidence and scope

The decisions-policy run finished all 40 episodes: 37 structurally complete,
three incomplete, with 162 measured requests and 1,769,985 observed tokens.
All variable stages completed. These are structural outcomes, not scientific
recovery or fitted mechanism-compliance scores. The previous run completed 38/40;
this difference alone does not establish a prompt effect.

The three remaining blocks expose specific editing problems:

- T2-hard seed 1 put positive and negative uses of `X_delayed_update` into the
  same `X_delayed` equation. It then tried to remove the process using both
  `remove_processes` and `remove_equations`; the latter key rejected the entire
  transaction. Changing its already-`influence` classification cannot resolve
  repeated uses of the same receiver.
- T2-hard seed 4 retained ordinary equation records for supplied `Uii`, meal,
  insulin input, and time. The compiler surfaced only the first missing
  generated-variable declaration. Later replies attempted to add and remove an
  uncommitted `Uii` process simultaneously.
- Alien-device seed 4 retained equation records for its external input and
  time. Repeating valid generated equations did not remove those records.

## Implemented feedback

1. Report every retained undeclared LHS together, with its exact name and public
   role. Inputs, covariates and time must remain supplied: show `remove_equations`
   while preserving intended RHS references. An observation such as `Uii` may
   instead be generated through an explicit permitted type declaration. Public
   targets must be generated. Unknown names require a proposer declaration or
   explicit correction of a mistaken label. Process-owned definitions are excluded.
2. Tailor duplicate-receiver advice to the actual violation. Each process has
   one signed use per receiving variable; neither a kind change nor a sign flip
   alone resolves repeated uses. Distinct intended laws may be represented as
   distinct local terms or processes. The runtime does not cancel, merge, split,
   delete, or choose signs for them.
3. Identify the owner of a rejected deletion. A retained process uses
   `remove_processes`, and its name must be omitted from `remove_equations`.
   Preserve unrelated valid equation removals and coordinate affected consumers.
   Removing a process removes its definition and inserted uses, not a separately
   retained variable declaration.
4. Distinguish retained processes from rejected-only additions in conflicting
   replace/remove replies. Withdraw an uncommitted addition by omitting both its
   proposed entry and removal key. No part of a rejected transaction was applied.

The feedback references the existing equation-based model view. Generic compiler
error evidence remains visible; repeated repair options are replaced with a
pointer to the specific issue. System/stage instructions, public briefs, schemas,
scientific questions, predicates, model view and budgets remain unchanged.
Historical prompt policies remain selectable and unchanged. No new diagnostic is
an acceptance gate, and no draft is changed by preparing a prompt.

## Public-requirement gaps for the next milestone

The following inventory is a review agenda, **not new runtime enforcement**.
It separates requirements from what the current graph actually demonstrates.

| Public intent | Existing check | Gap and next assessment |
| --- | --- | --- |
| Source/sink and exchange meanings agree with balances | Dependencies and process-use/sign format | Some declared nonnegative rates have inconsistent uses. Clarify positive-rate conventions, allowing signed net rates; do not infer signs from names. |
| CSTR distinguishes feed, reaction heat and jacket exchange | Thermal-state cycle; broader controlled-balance inference unresolved | A feed-only temperature balance can pass. Locate each required mechanism in the assembled model, allowing inline/shared laws and alternative thermal coordinates. |
| T2 utilization has a coherent meaning and effect in glucose balance | T2-easy: input-to-memory-to-utilization and supplied-utilization composition paths; T2-hard instead checks the insulin-memory path to glucose and does not require generated `U` | A disposal term can enter the derivative of a disposal quantity while disconnected from glucose. Inspect rate/derivative/readout meanings and equivalent removal paths; do not impose algebraic `U` or one literal edge. |
| Both coupled basins accumulate water | Upstream memory and assigned downstream feedback coordinates | An upstream-only state can be assigned as downstream storage. Ask how each basin's water storage and local runoff are represented; allow transformed or mixed coordinates. |
| Internal transfer conserves volume; outlets satisfy the public regime | Receiver/sign format and storage feedback | Functions, conversions, thresholds and cancellation remain deferred. Check complete physical balances after functions; repeated arguments alone do not prove duplication or tailwater dependence. Explicit forbidden-path requirements still apply. |
| Appropriate transient response and return behavior | Differential declarations and memory paths | Paths do not establish stability or restoration. Evaluate functions and initialization, permitting accumulators and alternative feedback routes; never require decay universally. |

These observations use public specifications and saved models. They are not a
scientific correctness score or a reason to pick a best seed. Some are explicit
meaning conflicts; others remain unresolved until function construction and
behavioral assessment. A stronger generic requirement representation needs
review before it changes acceptance. Benchmark-specific equations are not added
to the constructor.

## Updated prompt document

The review is generated from the actual saved predecessor of every request,
including its previous rejected reply. Red/green changes are marked, with links
to the three failure patterns. Historical replies are not replies to the revised
prompts, and no hypothetical repaired chain is claimed.

```bash
cd /Users/yuanzhangxiao/Projects/autoformalism
PYTHONPATH=src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/jetstream-decisions-inspection-20261013 \
  --output output/edit-feedback-prompts-review-local \
  --policy minimal-edit-feedback-1
open output/edit-feedback-prompts-review-local/PROMPTS.html
```

The output directory must be new/empty. The generated report and exact requests
stay outside Git. The maintained review for this revision is
`output/edit-feedback-prompts-review-20261013-final/PROMPTS.html`.

## Offline verification

All 162 cached requests from 40 episodes were checked against their actual
predecessors. Only 13 user packets change; all variable-stage packets remain
identical. Eight requests gain grouped LHS guidance, one gains process-owner
removal guidance, and one gains replace/remove-conflict guidance; these counts
can overlap. Duplicate-use advice changes four requests. Total user text grows
by 0.37% in UTF-8 bytes, not a measured token estimate. All 163 HTML pages have
valid local links. The public brief, model view, requirement checklist, schema,
stage instructions, generation settings and non-editing feedback are preserved.
No saved model was repaired or newly accepted in this offline exercise.

## Next live confirmation, after prompt review

The registered Jetstream wrapper is
`scripts/jetstream/run_edit_feedback_construction.sh`. It uses a fresh
`edit-feedback-seeds5-v1` root, all eight Full cells and seeds 0–4 (40 episodes),
with the existing verified local server, five global repairs and unchanged
request/token limits. It stops after topology: no functions, fitting, test access,
or automatic follow-up. Old campaigns must resume with their original pinned
source. Registration does not submit an experiment.

A pinned source bundle and explicit VM launch command should be supplied after
review; do not run a new wrapper against an old campaign root or alter an active
checkout. Compare stage outcomes, failure patterns and median/MAD costs across
all seeds, while auditing scientific consistency separately.
