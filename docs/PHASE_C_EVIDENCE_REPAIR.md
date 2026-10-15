# Phase C: repair bookkeeping without scientific rollback

The opt-in `minimal-evidence-feedback-2` policy corrects repair presentation and
advisory identity. It preserves the v1 variable, shared-process and ordinary
initial-topology requests, schemas, public evidence, structural acceptance rules,
model settings and budgets. Historical policies and saved campaigns are unchanged.
No benchmark data or finalized public specification is changed.

## Why the distinction matters

A structurally accepted draft need not be scientifically adequate. In the saved
T2-easy seed-2 episode, glucose already depended on meal, utilization and EGP;
this was not the meal-only glucose case. Its restricted insulin subsystem
`dI/dt = f(insulin_input) - g(I)` received conditional training conflicts. Later
repair introduced algebraic cycles without resolving the insulin restriction.
That is an unsuccessful scientific repair and a loss of structural validity,
not evidence of losing a scientifically correct model.

The new policy keeps scientific revisions available even when all structural
checks pass. It does not restore earlier models automatically or prefer them
because they passed weaker checks.

## Changes

1. **Separate work types.** Repair feedback distinguishes model/structural errors,
   record corrections, pending scientific advisory questions and completion.
   Unknown requirement records get an exact record-removal example; no deletion
   of their referenced variables or processes is requested. The proposer still
   chooses which effects and dependencies are scientifically appropriate.
2. **Explicit completion.** A reply can contain only advisory answers or an
   explicit completion request. `stage_complete=false` continues to mean that
   the proposer requests more work. A `revise` answer alone does not edit the
   model or resolve the conflict. Retaining an approximation or disputing an
   assumption requires an explanation and does not certify accuracy. No answer
   disposition is preselected in the repair examples. Accepted answers and their
   explanations remain visible at completion, with an explicit unverified status.
3. **Exact boundary selectors.** Feedback enumerates existing, structurally
   eligible contribution references and their declared meanings. A process
   stored in `processes` uses the historical `shared_process` reference field,
   including one-receiver processes; ordinary-term indices exclude those uses.
   The runtime does not decide which contribution is an external loss. Missing
   effects still require scientific edits. Stale references remain invalid.
4. **Stable conditional questions.** Versioned question identities bind relevant
   diagnostic topology, evidence and assumptions. Prose or unrelated-subsystem
   edits do not unnecessarily request another answer. Relevant topology/evidence
   changes are reassessed. Full draft hashes remain in provenance. A recorded
   answer never deletes the diagnosed conflict or verifies its explanation.
5. **Visible lost checks.** Accepted edits record which previously passed public
   predicates no longer pass. A compiler-invalid draft may make a check
   unavailable; this is distinguished from a failed path. The report requests
   the intended corrected implementation, not automatic rollback.

The three training diagnostics remain conditional exact-record reproduction
questions, not NMSE tests or additional structural gates. Unknown assumptions
remain deferred. Function signs, overlap, conservation and fitting are later work.

## Verification and comparison

`audit_evidence_repair_feedback.py` evaluates both policy facades on each actual
saved predecessor and successor. It reconstructs advisory answers only from
accepted historical replies to displayed questions. It does not apply those
replies to a hypothetical new prompt chain, fabricate repaired models, or make
LLM/optimizer calls. Its HTML includes exact new feedback and marked differences.

Unit/integration tests cover unchanged initial requests, semantic identity,
correct reference choices, record-only corrections, lost required paths, explicit
completion, unresolved revise decisions, the eight-case campaign and cached
resume. The next live study is 40 fresh constructions: eight cases, Full
specification, seeds 0–4. The budget remains 133 requests / 655,360 tokens per
episode, including ten global repair calls / 262,144 reserved tokens. No fitting
or test-data access is involved.

The same seed roster does not guarantee identical later sampling: seeds also
include the stage's global event index, so extra calls shift later seeds. This
milestone deliberately preserves that historical schedule. Compare stage-specific
structural status, scientific conflicts/deferred hypotheses, equations and cost;
do not interpret a higher completion count alone as scientific improvement.

## Jetstream2

Use the delivered bundle on the existing verified H100 VM. Its source and public
inputs are pinned and checked by `MANIFEST.sha256`. Run from the extracted folder:

```bash
export AF_JS2_ROOT="$HOME/af-construction"
export AF_JS2_CAMPAIGN_ROOT="$AF_JS2_ROOT/evidence-repair-seeds5-v2"
sudo -v
bash scripts/jetstream/run_evidence_repair_construction.sh run
```

The new wrapper defaults to the bundle's sealed training-evidence file. Use the
same folder/environment for `status` and, after completion, `inspect`. Rerun
`run` to resume this exact campaign; do not point it at an earlier campaign root.
Existing compatibility provenance must match the current server. The launcher
does not start another GPU server or change the installed model.
