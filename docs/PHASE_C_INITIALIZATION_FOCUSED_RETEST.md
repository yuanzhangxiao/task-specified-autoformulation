# Focused initialization repair: ACES confirmation

The initialization-v2 audit found two oversized next requests and one incorrect
initial-time witness clearance. The new opt-in
`saved-topology-initialization-3` protocol fixes these before another full run.
Historical prompt policies, saved experiments, benchmark data and public
specifications are unchanged.

## Scope

Three explicitly selected T2-easy saved final drafts (seeds 0, 1 and 2) retain
their initializers and all recorded trajectory witnesses. Seeds 0 and 2 also
retain the rejected reply that caused feedback growth. This is a separately
budgeted diagnostic retest, with at most ten calls and 262,144 charged tokens per
case, not an equal-budget comparison or a silent reset of a historical episode.
Historical costs are saved separately. There is no fitting, model promotion,
automatic follow-up or test-data access.

The earlier response with 33 process declarations contained 33 exactly identical
records for `basal_insulin_production`. The provider returned `finish_reason=stop`;
the response was not truncated. The saved response does not establish the cause
of that repetition. The runtime correctly rejected the transaction, but then
echoed the entire list in feedback. V3 displays consecutive identical prior
records once with an occurrence count, preserving order and exact expansion.
It does not merge proposed scientific effects or partially accept edits.

## Initial-time check

For an algebraic readout at the common initial time, expand algebraic dependencies
and stop at differential states. Compare those states' declared initializers and
the instantaneous supplied readings. Do not follow the differential RHS to infer
differences in initial values. At positive times, retain the existing causal
prefix comparison. Unavailable assumptions remain unavailable.

In saved seed 1, `U = f(Uii) + h(X)` with a shared `X(0)` cannot distinguish two
initial `U` measurements when `Uii(0)` agrees. Different glucose initial readings
in an ancestor of `dX/dt` do not resolve that initial conflict. The runtime asks
for a justified readout/initializer revision or an evidenced assumption challenge;
it does not prescribe which scientific dependency to add.

This milestone corrects retained witness checks. The optional all-pair initial
readout discovery helper is tested but not enabled in live prompts, avoiding an
unreviewed expansion of feedback volume.

## Prompt changes

`mechanism_realizations` explicitly means the proposer's requirement-to-component
mapping, not satisfied requirements. The dictionary uses “algebraic equations”
and explains that a declared public target output is an eligible generated
receiver. Each repair call is stateless, so complete public context remains
available. Current feedback and equations appear first; immutable public context
appears last. Only exact duplication is condensed. Ordering is preserved in the
actual HTTP request, rather than erased by sorted JSON.

The stage instruction gives the current task. Issue-specific guidance is attached
to corresponding findings. Core edit semantics remain available because they
define how a reply changes the retained draft. See
`PHASE_C_INITIALIZATION_PROMPT_AUDIT.md` for the rule-by-rule provenance audit.

## Freeze and preview locally

```bash
PYTHONPATH=src:. .venv/bin/python scripts/saved_topology_repair.py focus-initialization \
  --source output/initialization-v2-analysis-20261016/inspection \
  --root output/initialization-focused-inputs \
  --task cell02_seed0_full_actionable_initialization \
  --task cell02_seed1_full_actionable_initialization \
  --task cell02_seed2_full_actionable_initialization
PYTHONPATH=src:. .venv/bin/python scripts/saved_topology_repair.py preview \
  --root output/initialization-focused-inputs \
  --output output/initialization-focused-prompts-review-20261017
```

The seal binds current source. Regenerate into a fresh output directory after any
source change. The delivered bundle contains the frozen plan under
`inputs/saved-initialization-v3` and includes the complete source needed for its
identity checks.

## Run on ACES

Upload the delivered tarball into
`/scratch/group/p.nairr260351.000/u.yx126462`, extract it, then run:

```bash
bash /path/to/extracted-bundle/scripts/hpc/start_saved_initialization_aces.sh run
bash /path/to/extracted-bundle/scripts/hpc/start_saved_initialization_aces.sh status
bash /path/to/extracted-bundle/scripts/hpc/start_saved_initialization_aces.sh inspect
```

The wrapper uses one H100, the existing pinned GPT-OSS-20B cache and vLLM 0.27.1
image, and a three-hour allocation. It checks storage/file creation before
submission, tests a CUDA kernel, records serving provenance, and places compiler
caches in node-local temporary storage. Extraction itself still needs free file
quota. Paths can be overridden through `AF_PYTHON`, `AF_VLLM_IMAGE`, `AF_HF_HOME`
and `AF_OUTPUT_ROOT`. Submission receipts make repeating `run` idempotent.
After an interrupted allocation, inspect its logs and use a new `AF_WAVE` to
request another allocation against the same checkpoint and remaining budgets.

Results default to `phase-c-initialization-focused-v3` under the group directory.
`inspect` writes `inspection.tar.gz` there. No serving equivalence to the former
Jetstream VM is assumed; this retest records the ACES environment independently.

## Acceptance and limitations

Offline checks must preserve legacy replay, exact previous-reply expansion,
public evidence, transaction atomicity and resumed call identities. The saved
initial-time witness must remain actionable. Live confirmation must establish
that the smaller requests fit the server context and inspect the returned model
changes. Passing schema/structural checks alone does not establish scientific
correctness, and a successful three-case retest does not qualify all benchmarks.
