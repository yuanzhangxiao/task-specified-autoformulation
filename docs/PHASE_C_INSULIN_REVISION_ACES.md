# Candidate 15 proposer revision on ACES

## Repair using the deterministic feedback

ACES job 2315743 completed this request. The returned basal-secretion edit
referenced `direct_ingestion`, which existed only in the previous draft,
and did not declare it. No child was admitted for fitting. The result
review includes the exact rejection, the signed boundary equation and
training replays under two explicit interpretations.
[Basal repair results](../work/agent-comms/2026-11-08-insulin-basal-repair-results.md)

One GPT-OSS-120B call receives the primary constrained fit, the earlier
unfitted patch, its resolved parameter spellings and the deterministic feedback
from the public training evidence. The request includes the three 90 g meal
windows and the legal 150 g example in which insulin fell from +26.10 at
minute 31 to −1.53 pmol/L at minute 32. The reply is a full replacement patch
against the fitted parent. Its new-state and process allowances are available
even if it discards the entire clock proposal.

Upload `phase-c-insulin-feedback-repair-v1.tar.gz` from
`output/insulin-feedback-repair-delivery-20261107/` to
`/scratch/group/p.nairr260351.000/u.yx126462`. Submit from the ACES login node.

```bash
cd /scratch/group/p.nairr260351.000/u.yx126462 && tar -xzf phase-c-insulin-feedback-repair-v1.tar.gz && cd phase-c-insulin-feedback-repair-v1 && sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_insulin_feedback_repair_aces.sh run
```

Check the job or package the results from any ACES directory.

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/phase-c-insulin-feedback-repair-v1/scripts/hpc/start_insulin_feedback_repair_aces.sh status
```

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/phase-c-insulin-feedback-repair-v1/scripts/hpc/start_insulin_feedback_repair_aces.sh inspect
```

Download `phase-c-insulin-feedback-120b-v1/inspection.tar.gz` from group scratch.
After extraction, the same wrapper's `preview` action can prepare and display
the request before submission. The job uses two H100s, one hour, one generation
reservation and a 4,096-token output limit. Repeating a confirmed submission
returns its original job ID. The cached response can be assessed again offline.

The verifier matches Greek letters and their case-sensitive spelled-out names
in whole identifiers against scientific declarations. Initializer formulas use
their own local aliases and parameters. Conflicting declarations and unknown
symbols surface as errors. The raw response is cached and each substitution
is logged by field. A replay of the same equation family, including a change
of parameter roles alone that still admits the known counterexample, is
reported as a repeated repair. A different model receives the existing public
path, training-consistency and boundary checks. See `summary.json` for
`insulin_boundary_check.boundary_sign` and `identifier_spellings`.
An eligible patch with a deferred sign still needs further assessment. Orion
can refit an admitted patch separately. [Reference fit and equations](PHASE_C_INSULIN_MODEL_COMPARISON.md)

## Revising the insulin balance at depletion

ACES job 2314569 completed. The reply used Greek τ in two formula symbols
whose declarations used the ASCII name tau. A spelling-only replay passed
static checks and deferred the boundary assessment. A permitted parameter
choice then produced negative insulin on the 150 g training schedule. The
scientific repair and its numerical witnesses are in the
[returned proposal audit](../work/agent-comms/2026-11-07-insulin-boundary-proposal-results.md).

The 7 November run uses Orion’s primary concentration-constrained endpoint
with training NMSE 0.165562. On the 150 g training trajectory, observed insulin
is 22.62 pmol/L at minute 300. The matched unconstrained fit gives −31.75.
The constrained fit reaches zero within numerical tolerance and passes all
38,416 fine-grid samples. At that terminal state, setting I exactly to zero
gives a glucose contribution of +0.123 and an action-state contribution of
−0.925 pmol/L/min. With ingestion and external insulin both zero, the net rate
is −0.802 pmol/L/min. A one-minute diagnostic extension gives −0.794 pmol/L.
The final observation is at minute 300.

The proposer receives this boundary and the three shifted 90 g meal windows.
Observed insulin rises by 1.97 pmol/L in each window, compared with fitted
rises of 162.19, 165.86 and 167.27. The direct meal term alone contributes
+154.27. The proposer chooses the function form, gives the assembled rate
at I=0 and explains how the early meal response changes. The required delayed
I-to-disposal pathway remains part of the public task. Orion’s result record
contains the matched fits and diagnostic provenance.
[Fitting handoff](../work/agent-comms/2026-11-07-insulin-positive-continuation-results.md)

The runtime expands the proposed insulin balance after the existing structural
and model-data checks. Bounded polynomial arithmetic rejects a zero-input
boundary law that exactly preserves the known outward rate, or a law proven
strictly outward at the displayed state across the declared parameter
domains. A sufficient polynomial sign check can establish a nonnegative
boundary rate under the displayed assumptions. Other forms receive a deferred
sign assessment and a numerical point diagnostic when their inputs are
available. That diagnostic uses inherited values and new parameter guesses.
The subsequent fit must check trajectories, meal-window changes, and the
depleted-state behavior independently.

Upload `phase-c-insulin-boundary-revision-v1.tar.gz` to ACES group scratch.
Paste this line on ACES to prepare the request and submit the one-call 120B job.

```bash
cd /scratch/group/p.nairr260351.000/u.yx126462 && tar -xzf phase-c-insulin-boundary-revision-v1.tar.gz && cd phase-c-insulin-boundary-revision-v1 && sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_insulin_boundary_revision_aces.sh run
```

For status and inspection, use the same wrapper from any directory.

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/phase-c-insulin-boundary-revision-v1/scripts/hpc/start_insulin_boundary_revision_aces.sh status
```

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/phase-c-insulin-boundary-revision-v1/scripts/hpc/start_insulin_boundary_revision_aces.sh inspect
```

Download `phase-c-insulin-boundary-120b-v1/inspection.tar.gz` from group
scratch. The new run has its own sealed plan, output directory and one
generation reservation. It uses the same 2-H100 configuration and 4,096-token
generation limit as the earlier revision. The prepared prompt is in
`output/insulin-boundary-prompts-review-20261107/PROMPTS.html`.

## Earlier revision after the direct meal control

The verified nested child has training NMSE 0.261011 but still peaks 91 minutes
early on all three shifted 90 g meals. This job asks GPT-OSS-120B for one insulin
law revision and applies deterministic admission. It runs no optimization,
validation/test evaluation, automatic follow-up, or model promotion.

The returned ACES job 2310039 admitted a revision adding the existing insulin
memory state to production. Its new coefficient remains unfitted. A fixed
zero-coupling replay recovers the prior NMSE to 1.23e-12; the default guess of
one creates a growing mode with the other inherited coefficients fixed.
See [the equation audit and Orion fitting handoff](../work/agent-comms/2026-11-06-insulin-revision-results-and-refit.md).

The numerical evidence and Orion work order are in
[the returned-fit handoff](../work/agent-comms/2026-11-05-insulin-meal-control-results.md).

## Request and admission

The proposer sees the public task, fitted child equations and shared coefficients,
measured peak times/heights, and the conditional observed-glucose diagnostic.
The user message also gives the fixed eight-unit observation-prefix policy.
It may replace the entire RHS of I or `insulin_production`, with at most one
connected new differential state, one connected algebraic process, and eight
new dynamics parameters. New states require explicit causal initialization.
All other existing equations and public data access remain fixed.

The prompt clarifies complete replacement, expression grammar, variable versus
parameter declarations, and initializer-local parameter scope. Initial maps can
use permitted time-zero readings or state initials. Trajectory-specific roots
remain eligible for training fitting, with inference from only the fixed prefix
on held-out trajectories. No synthetic example's function is prescribed in the
live prompt.

The runtime applies the existing atomic structural normalization, compilation,
public-path checks, new-component connectivity, repeated-family detection and
qualified model/data tests. Connectivity uses the actual additions after
normalization, so a harmless restatement of an unchanged parent process does
not create a false disconnection error. Rejected replies and unresolved checks
remain visible. `eligible_for_refit` does not establish improved peaks or full
scientific adequacy.

There is exactly one generation attempt, with a 4,096-token output limit and
32,768-token total allowance. Measured context admission precedes generation.
An interrupted uncertain attempt is not resent. A cached completed response can
be revalidated without another generation. Source inputs and call results are
immutable. This milestone does not expand the search breadth or repair depth.

## Provenance boundary

The compact source packet is a projection of the reviewed fit receipts from
commit `9d7f5e3`, plan
`c2ac4c6e173950ae587f1bd36e4123ba5684ba3b7ed849f1b970ee174929e8a1`.
Its content SHA-256 is
`fc5372e439b37f4ff7f0a93dce071f0480013e3b612fa00c4045d93c95bddbfe`.
Only this reviewed intake may enter the bundled launch path. A new construction
plan binds it to the updated proposer code. Historical fitting receipts are
preserved; their source identity is not replaced with the new construction hash.

The ACES execution record separately pins the 120B model revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, vLLM 0.27.1 image bytes,
offline model metadata/shard inventory, Harmony vocabulary, launcher code,
context length and two-H100 runtime probe. Weight inventory uses immutable cache
blob names and file sizes, not a claim of independently hashing all weight bytes.
The generic provider alias check alone does not verify a model revision.

## Upload and launch

Upload `phase-c-insulin-revision-v1.tar.gz` from
`output/insulin-revision-delivery-20261105` to
`/scratch/group/p.nairr260351.000/u.yx126462` through the ACES file manager.
The bundle includes the approved public intake and offline Harmony vocabulary.
It uses the existing ACES model cache and vLLM image without downloading weights.

On ACES, paste this single line:

```bash
cd /scratch/group/p.nairr260351.000/u.yx126462 && tar -xzf phase-c-insulin-revision-v1.tar.gz && cd phase-c-insulin-revision-v1 && sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_meal_insulin_revision_aces.sh run
```

One job requests two H100 GPUs, 16 CPUs, 128 GiB RAM and one hour. Server and
compiler caches use the allocated node's temporary directory and are cleaned
at exit. The worker uses the submitted code directory even when Slurm spools
its shell script. Loading Python modules cannot change which project checkout
is imported. The launcher uses a task-specific output root, ignoring generic
roots left by prior campaigns.

The default output root is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-insulin-revision-120b-v1`.
Repeating the launch command returns a confirmed existing job rather than
submitting another. If the scheduler response is uncertain, inspect the saved
intent and queue before recovery. Do not remove submission or generation
receipts to bypass that protection.

Status from any ACES directory:

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/phase-c-insulin-revision-v1/scripts/hpc/start_meal_insulin_revision_aces.sh status
```

After completion, refresh the inspection package:

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/phase-c-insulin-revision-v1/scripts/hpc/start_meal_insulin_revision_aces.sh inspect
```

Download `phase-c-insulin-revision-120b-v1/inspection.tar.gz` from the same group
scratch directory. It contains the exact prompt/response, runtime and submission
receipts, reports, and any admitted typed model with its Orion refit handoff.
Status and inspection need no GPU allocation or running model server.

Optional task-specific overrides are `AF_MEAL_INSULIN_REVISION_ROOT` and
`AF_MEAL_INSULIN_REVISION_SOURCE`. Existing serving asset locations can be set
with `AF_HF_HOME`, `AF_VLLM_IMAGE`, `AF_HARMONY_ASSETS` and `AF_PYTHON`.

## Review and fitting handoff

The live prompt preview is generated at
`output/insulin-revision-prompts-review-20261105/PROMPTS.html`.
The returned `REPORT.html` shows the complete reply, pre/post equations and
admission evidence. `result.json` includes the typed model and a `refit_handoff`
only when admission succeeds. Orion then qualifies a backend for that model's
actual dynamics and initialization scope, independently replays fitted
parameters, and compares against the retained nested child. The reference
training identity, prefix policy and numerical allowances travel with the model.
A nonlinear proposal must not be forced through an affine-only profiler.
