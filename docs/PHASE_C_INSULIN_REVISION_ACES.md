# Candidate 15 proposer revision on ACES

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
