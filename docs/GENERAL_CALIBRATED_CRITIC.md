# Milestone 4: general calibrated judge as advisory critic

This opt-in integration pilot starts from the six **selected** Milestone 3 pruning
models: both prompt variants for obfuscated Dalla Man, CSTR and the alien device.
It does not replace historical parents, rerun construction/pruning, or change the
frozen fitter. It contains no basin-specific probes. It is an engineering pilot,
not an equal-budget experiment establishing that a critic improves discovery.

## What is reused, and what is new

The scientific assessment reuses `repair_scientific_judge` and `PairedHybridJudge`:
GPT-OSS-120B, low reasoning, temperature 0.2, 6144 output tokens, the existing
atomic-evidence and hybrid prompts, blinded forward/reverse orientation, question
consensus, and the existing neutral-fixed-denominator scoring. The second seed is
fallback after terminal failure, not a second independent expert. A successful
paired review normally has two atomic calls and two hybrid calls; retries can
increase physical calls. Model revision, prompts, inference settings, code and
serving-image SHA are frozen before execution.

The initial review uses the existing self-pair convention. The changed-child
review compares parent and child. Both receive only public task/context and
**symbolic** candidate artifacts, including symbolic causal initialization. They
receive no fitted parameter values, trajectories, NMSE, external certificates,
repair history, or routing request. This preserves the established judge contract.
Its calibration does not establish that its findings are correct on every model,
or that following them improves a fit.

The new part is the **advisory feedback adapter** and its scheduling. It joins:

1. existing deterministic public predicates, with unresolved entries preserved;
2. fixed-parameter training residual evidence and optimizer-status qualification;
3. separately labelled consensus absolute FAILs and comparative parent-preferred
   concerns extracted by the existing judge adapter.

The existing whole-model revision proposer decides what deserves investigation.
It may change variables, topology, laws, shared processes and causal initializers
coherently, or choose no change. The runtime displays the complete current model
and inferred shared-law consumers, performs the usual atomic compiler repairs,
and checks the public requirements. There is no separate LLM router, new critic
rubric, extra confirmation call, or benchmark-specific repair action. This use of
advice for proposal/routing is an **uncalibrated extension**, not part of scientific
certification. Proposer evidence references retain their existing advisory policy.

Critic preference cannot select or veto a candidate. An indeterminate, interrupted,
or unavailable review provides no reliable advice; it is neither a pass nor a
scientific defect. Ordinary revision and fitting can still proceed. A runtime
hard failure cannot be overridden by persuasive judge feedback. In particular,
CSTR's unanchored `controlled_balance` remains ambiguous and development-eligible
under the existing public-check contract; the critic cannot turn it into a
machine-certified pass. No missing benchmark anchors are invented.

## One bounded episode

The six scheduled phases are:

1. **CPU preparation:** verify completed pruning and original development receipts,
   reconstruct canonical models and boundary rules, pin model/image revisions,
   download the pinned model snapshots, and replay the retained parameters on
   training data only. The replay has its existing separate 300-second cap.
2. **Parent reviews:** review each actual retained model after pruning, using two
   H100s and the established 32768-token serving context.
3. **One revision episode:** GPT-OSS-20B, low reasoning, the existing three cached
   physical attempts and 8192-token per-response cap, on one H100. There is no new
   cumulative token cap. The request includes complete equations and separately
   labelled feedback. A missing training replay preserves the parent rather than
   fabricating residuals.
4. **Changed-child reviews:** use the same judge protocol before fitting. An
   unchanged model reuses its review. Exact repeated review requests share a cache.
   Child findings are stored for analysis and a possible later round; they do not
   trigger a second repair episode in this pilot.
5. **CPU fits:** at most one legal changed child per parent, with compatible learned
   parameter/initializer values and the frozen `collocation-single-target-v2`
   caps: 120 seconds initialization, 180 seconds refinement, 240 residual calls.
   A failed/no-change proposal does not receive an extra unchanged refit.
6. **Report:** compare child and retained parent by finite validation NMSE, then
   additive-term count; retain the parent on an exact tie. There is no 1% pruning
   tolerance here: that belongs to the completed pruning decision. Judge scores
   are absent from the selection interface. Preserve trial and retained results.

Preparation also runs one small synthetic smoke fit; this is not a benchmark
refit. The benchmark maximum is six new fits and twelve paired review requests
before unchanged-model reuse. No next round, production promotion, test access,
or independent BDF/Radau replay is performed automatically.

## Provenance and accounting

`plan.json` copies only public development assets and chosen parent models, and
retains original parents separately. Historical sibling inputs, result/backend
receipts and original lineage are verified without demanding that old package
hashes equal the new executable. All new execution is pinned to this package,
launcher, Python runtime, image and model revisions.

Per task, `evidence.json`, `parent_review.json`, `proposal.json`, `child_review.json`
and `result.json` keep each handoff separate. `calls/` retains proposer requests,
responses, compiler retry feedback and observed usage. `judge_cache/<hash>/`
contains the exact unchanged judge request, cached responses/events, and a sealed
review receipt. An interrupted review or fit consumes its attempt; rerunning a
stage never silently grants a fresh numerical or review budget. Partial scheduler
submissions retain intent/reply records and are not blindly resubmitted.

`summary.json` reports review availability, proposal status, parent/child public
predicates, complexity, fit status, convergence qualification, training/validation
NMSE, and selected origin. It counts judge costs once per unique request and
proposer costs once per physical call. Unknown usage stays unknown. Recorded
calls/tokens can be lower bounds after an interrupted provider call. Persisted
partial calls remain counted even without a final proposal/review receipt;
unpublished judge reviews explicitly report incomplete usage. Historical
construction/pruning costs are not included in these incremental costs.

The legacy review adapter's `prefit_unpruned` metadata is retained in its raw
response for provenance. It does not describe these imported, potentially pruned
parents: each new stage receipt explicitly records its actual placement and that
the legacy flag is inapplicable. The calibrated prompts and responses are unchanged.

## ACES commands

Local verification (2026-09-22): the full test wrapper passed 3,022 tests with
eight optional-dependency skips. After the final accounting hardening, 99 focused
tests passed, including the shared-process pilot file excluded by that wrapper.
The synthetic end-to-end smoke exercises one real fit, prescribed judge/proposer
replies, and identical resume with no extra work. Changed-file lint and shell
syntax checks pass. Repository-wide Ruff still reports 37 pre-existing findings
in unrelated `analysis/claude` scripts; those files were left unchanged. Live
GPT-OSS serving and scientific usefulness remain for this ACES pilot to establish.

Use the full commit supplied with this milestone. These commands only read the
old personal-scratch checkout; fetches, archives, outputs and new caches go to
group scratch, avoiding the previously exhausted personal inode quota.

```bash
(
  set -euo pipefail
  export AF_COMMIT=REPLACE_WITH_FULL_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/general-critic-${AF_COMMIT:0:7}"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"
  export AF_SOURCE_ROOT="$AF_GROUP/process-pruning-v1"
  export AF_OUTPUT_ROOT="$AF_GROUP/general-critic-v1"
  export AF_HF_HOME="$AF_GROUP/huggingface-cache"

  AF_OBJECTS="$AF_BASE"
  if ! git -C "$AF_OBJECTS" cat-file -e "$AF_COMMIT^{commit}" 2>/dev/null; then
    AF_OBJECTS="$AF_GROUP/git-cache/general-critic.git"
    mkdir -p "$AF_GROUP/git-cache"
    if [[ ! -f "$AF_OBJECTS/HEAD" ]]; then git init --bare "$AF_OBJECTS"; fi
    git -C "$AF_OBJECTS" fetch --depth=1 --no-tags \
      "$(git -C "$AF_BASE" remote get-url origin)" "$AF_COMMIT"
  fi
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_OBJECTS" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  bash "$AF_REPO_ROOT/scripts/hpc/submit_general_critic_aces.sh"
)
```

The image defaults to the previously verified personal-scratch SIF (read only).
Set `AF_VLLM_IMAGE` if it was moved. Preparation downloads model files into the
specified group cache if absent; allow time/storage for the 120B snapshot.
`AF_JUDGE_REVISION` may be supplied explicitly; otherwise the launcher takes the
previously cached 120B revision when available, or resolves it once from the model
registry. Repeated submission uses the recorded revision and returns existing job
IDs. It never follows a moving model revision on resume.

GPU review jobs request two H100s each, sequentially, with four-hour allocation
limits; the proposer requests one H100 for one hour. These allocation limits
include serving overhead and do not change per-call/fit budgets. Six CPU fit tasks
use one CPU/16 GB each, at most four concurrently. There is no need to start a
manual model server.

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/general-critic-v1
cat "$ROOT/SUMMARY.md"
jq '{status_counts, selection_counts, review_status_counts,
     judge_cost, proposer_cost}' "$ROOT/summary.json"
jq '[.rows[] | {task, parent_review: .parent_review.review.status,
  proposal: .proposal.status, child_review: .child_review.review.status,
  selected: .result.selected, parent_complexity, child_complexity,
  parent_val: .parent.validation.normalized_mse,
  child_val: .result.child_fit.validation.normalized_mse,
  parent_findings: .parent_review.review.findings,
  child_findings: .child_review.review.findings}]' "$ROOT/summary.json"
AF_JOBS=$(jq -r '.jobs | [.[]] | join(",")' "$ROOT/submission_manifest.json")
sacct -j "$AF_JOBS" --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS
```

If an allocation fails, inspect its logs/receipts first. The CLI supports replaying
individual stages with `--index` without repeating completed work, but the
submission wrapper deliberately does not duplicate an existing or uncertain job
chain. Review this pilot before Milestone 5's controlled component comparisons.
