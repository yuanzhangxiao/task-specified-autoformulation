# Phase C: repair demonstrated conditional conflicts

This isolated saved-draft experiment tests repair feedback without repeating
variable and topology construction. It does not change historical construction
policies, public specifications, benchmark data, fitting, or model selection.

## Motivation

The last evidence-repair campaign produced 37 structurally eligible drafts out
of 40. Twelve final drafts still had 28 conditional training conflicts. All 28
were answered by retaining an approximation. The old policy correctly retained
the conflicts in the evidence, but allowed those answers to complete the repair
stage. Completion therefore did not demonstrate that the model could explain
the observations.

The three remaining structural failures include two coupled-basin drafts that
repeatedly selected ordinary term index 2 for an outlet stored as a process use.
The equations already contained the intended outlet. The previous feedback
listed valid selectors but repeated the same substantive instructions; only
budgets, counters and edit receipts changed.

## Revised repair contract

- **Request an edit for a demonstrated conditional conflict.** The runtime
  displays the actual matched observations, allowed conditioning, uncertainty
  bounds and assumptions. A witness outside known error bounds requires repair
  under those assumptions. The proposer chooses the scientific distinction and
  returns the model edit; the runtime does not prescribe a glucose-specific edge
  or infer a missing mechanism.
- **Answers are not repairs.** The actionable response schema has no
  `retain_approximation` choice. `revise` without an edit cannot clear a conflict.
  `dispute_assumption` requires supporting evidence but remains an unverified
  claim. It does not change frozen evidence or discharge a diagnostic.
- **Preserve uncertainty.** A comparison not decisive under its error bounds is
  not presented as a proven error. If an edit introduces unknown initialization,
  invalid structure, or another unsupported condition, a disappeared diagnostic
  remains deferred. Verified removal of a restrictive dependency assumption is
  reported separately from fitted accuracy. No topology pass certifies recovery.
- **Make repeated failures explicit.** A reminder shows the previous reply,
  retained error and repetition count. For boundary references it distinguishes
  ordinary-term indices from separately inserted process uses. Three repeated
  unsuccessful semantic attempts stop the episode. Different rejected
  alternatives can still use the remaining budget; prose changes do not count
  as scientific repair.
- **Let the runtime translate unambiguous bookkeeping.** External-effect feedback
  gives a content-bound handle for each eligible existing contribution, its
  equation, arguments, declared meaning and exact stored reference. The proposer
  selects the physical effect; the runtime maps that handle to the existing
  schema. Stale handles and simultaneous changes to the addressed effect are
  rejected. Newly introduced effects receive a handle on the next turn.

## Paired experiment

The input freezer checks every saved call chain and recomputes final assessments
from the prior 40-task public-only campaign. It selects all 15 final drafts with
a structural failure or conditional training conflict, without scoring or ranking
models. Each draft receives two independent repair episodes:

| Arm | Feedback and completion |
| --- | --- |
| `control` | Existing `minimal-evidence-feedback-2` prompts, selectors, advisory-answer completion and repetition behavior |
| `actionable` | Direct repair requests, unresolved assumption challenges, contribution handles and specific repetition feedback |

The 30 episodes share paired starting drafts, public evidence, sampling settings,
seeds and equal maximum budgets: **10 calls and 262,144 charged tokens per arm**.
Historical answers are not reused. Arm order alternates by seed. This is an
additional bounded repair experiment, not a replay of the historical sampling
sequence or an equal-total-budget comparison with a fresh construction campaign.
The existing local H100 model server is reused after compatibility verification.

Every request/response is cached. Each accepted or rejected transaction is sealed
with its predecessor, exact request, response hash, structural assessment and
conditional diagnostic history. Resume reconstructs that chain and spends only
the remaining budget. Reports distinguish structural eligibility,
`repair_required`, `review_deferred`, and `checks_clear`; the control may finish
while its independent scientific status is still `repair_required`. All final
models and feedback remain inspectable. Token summaries use median and unscaled
MAD. No fitting, solver rollout, test access, model promotion or automatic
follow-up occurs.

## Running the delivered bundle

Extract the pinned `phase-c-actionable-repair-<commit>.tar.gz` on the Jetstream2
VM, enter that directory and verify `MANIFEST.sha256`. The bundle contains its
sealed starting drafts and public training evidence, so the old inspection
archive need not be uploaded.

```bash
sha256sum -c MANIFEST.sha256
export AF_JS2_ROOT="$HOME/af-construction"
export AF_JS2_CAMPAIGN_ROOT="$AF_JS2_ROOT/saved-actionable-repair-v1"
sudo -v
bash scripts/jetstream/run_saved_topology_repair.sh run
```

Use the same environment and folder with `status` to inspect progress. After the
worker stops, use `inspect` to render reports and create
`$AF_JS2_CAMPAIGN_ROOT/inspection.tar.gz`. Repeating `run` resumes the same sealed
experiment. The worker has a three-hour draining boundary; a partial report is
not completion. The server must match the existing saved compatibility probe.

For offline preparation from a prior inspection archive:

```bash
export PYTHONPATH="$PWD:$PWD/src"
python scripts/saved_topology_repair.py freeze --source OLD_INSPECTION --root INPUTS
python scripts/saved_topology_repair.py preview --root INPUTS --output PROMPT_REVIEW
```

The HTML preview contains every exact first request, including feedback, model
view and response schema. Qualification covers successful and failed edits,
uncertainty/deferred transitions, repeated versus different attempts, cached
resume, interrupted budgets, stale handles and foreign/tampered artifacts.
Live results are pending; this milestone makes no claim that the revised feedback
improves scientific recovery.
