# Bounded initialization conflict repair

This opt-in milestone makes qualified deterministic model–data conflicts block
initializer acceptance. Schema and dependency validity remain separately recorded
for diagnosis; they do not constitute acceptance while an active conflict remains.
The original dependency-only diagnostic and all historical checkpoints are retained.

## Scientific and computational scope

The returned dependency diagnostic had two valid declarations, but neither removed
its original matched-trajectory conflict. Both choices were shared maps of the
same measured initial insulin. The old loop stopped on dependency validity and
only assessed the evidence afterward, leaving its second-call allowances unused.
The new loop evaluates each valid declaration before acceptance and presents a
surviving conflict with the current relation, measured conditioning values,
assumptions and explicit repair choices. Historical witnesses are rechecked, not
copied into the prompt as conclusions about the new initializer.

A qualified contradiction is an acceptance failure under its stated assumptions.
Unavailable assumptions or prefix-inferred roots remain deferred evidence, never
a demonstrated scientific repair. Independent trajectory roots require the existing
observation-prefix authorization. Their identifiability and forecast accuracy still
require fitting. The prefix duration remains Orion's fitting hyperparameter and
must be frozen before held-out forecast evaluation.

The proposer can return a complete initializer replacement or explicitly report
that resolving the conflict requires a topology revision. The latter ends the
bounded episode without acceptance or automatic topology edits. No new variables,
ODEs, readouts, function formulas, fitting, model promotion or test access occur.

## Five-seed diagnostic

`scripts/construction_initialization_repair.py` reads the complete returned V1
initialization-graph inspection archive. It verifies sealed plans, input packets,
accepted dependency records, cached source responses and source cost totals.
It keeps the full archive and starts each new episode from the same latest saved
initializer choices and frozen model/data packet.

- Two saved models, each with proposal seeds **0, 1, 2, 3, 4**: ten episodes.
- At most **three calls per episode**, including initial repair and two further
  repairs: thirty total, with no hidden retry or automatic follow-up.
- At most 393,216 charged tokens per episode and 8,192 output tokens per call.
- All seeds are reported. There is no best-of-five or score-based selection.
- These are repeated proposal seeds on two fixed models, not ten independently
  constructed scientific models or a benchmark-wide performance estimate.

New calls use distinct namespaces, immutable source/runtime/settings identities,
cached responses and deterministic resume. A completed or unavailable episode is
not silently repeated. Changing the plan requires a fresh output root. Historical
costs are stored once per original source model and once for the original diagnostic;
the source-model breakdown is not added again to that diagnostic total. None of
these costs is multiplied by the five new proposal seeds.

The summary distinguishes `structural_valid`, `accepted`, active conflict counts,
deferred checks and provider availability. Acceptance establishes only passage of
the implemented deterministic checks. It does not establish correct initialization,
scientific recovery or predictive accuracy.

## ACES execution

The bundle contains the frozen source inspection under `inputs/source-inspection.tar.gz`
and the existing Harmony assets. Upload and unpack it under writable group scratch,
then run from the unpacked checkout:

```bash
bash scripts/hpc/start_initialization_repair_aces.sh run
bash scripts/hpc/start_initialization_repair_aces.sh status
bash scripts/hpc/start_initialization_repair_aces.sh inspect
```

The default output root is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-initialization-repair-seeds5-v1`.
`AF_GROUP`, `AF_OUTPUT_ROOT`, `AF_GRAPH_REPAIR_SOURCE`, `AF_PYTHON`, `AF_VLLM_IMAGE`
and `AF_HF_HOME` can explicitly override site paths. The wrapper uses one H100,
one hour, the pinned gpt-oss-20b revision and the existing serving image. Compiler
caches stay in node-local `/tmp`, preserving scratch inode headroom. The launcher
retains runtime/image verification and idempotent scheduler submission receipts.
`inspect` is offline and writes `inspection.tar.gz`; it does not call the proposer.
