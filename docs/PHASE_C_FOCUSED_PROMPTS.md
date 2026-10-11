# Focused construction requests — review before testing

Policy: `minimal-focused-construction-1`. Study: `focused_prompt_confirmation`.
This is an opt-in implementation and an offline review, not a new live result.
Historical plans must continue with their pinned code. Benchmark prompts, data,
fitter, scientific acceptance rules and test-data separation are unchanged.

## Proposer view

The full draft remains in sealed runtime events. The LLM receives:

- The short system message, current-stage instructions, and the unchanged public
  brief including training evidence.
- One `model_view`: variable types/meanings, symbolic process definitions and
  assembled balances, with joint arguments, signs and fixed conversions. Symbols
  are topology placeholders; function forms and fitted parameters remain later
  work. Unknown conversions are explicitly not claims of unity.
- One `requirements_checklist`, separating declaration/path observations from
  deferred/unavailable scientific checks. `memory_assignment_required` refers
  only to an explicit state assignment, not whether the public mechanism matters.
- `receiver_choices`, a small `previous_edit` receipt, and specific `feedback`.
- Editing rules and the existing stage-specific JSON response schema.

`bookkeeping_policy`, `interaction_review_handoff`, duplicate draft declarations,
and nested copies of diagnostics are absent from the LLM packet. The runtime still
retains the policy, original before/after declarations, calls, receipts, and final
interaction handoff. No full rejected model is repeated. Naming errors retain the
conflicting identifiers, drivers, receivers and conditional repair choices.

The model view is accepted state. The previous-edit receipt says what changed or
why a transaction was rejected. A rejected transaction still commits no partial
edits. Generated algebraic observations and external supplied inputs remain
distinct: observations may be explicitly modeled when permitted; external inputs,
time and covariates cannot become generated receivers.

## Feedback and continuation

1. Missing memory assignments show structured syntax examples for each
   type-eligible state. No example is a scientific recommendation. The proposer
   chooses the appropriate state(s), including new declarations if needed.
2. Invalid supplied receivers show a conditional process edit retaining its
   drivers and valid uses. Moving the intended effect to a local term is an
   alternative. Recently removed process definitions appear in the next request;
   the runtime neither restores nor reinterprets them automatically.
3. Missing RHS entries no longer hide all other dependency observations.
   Provisional driver/memory/target paths and reviewed feedback/forbidden-path
   observations use the declared edges directly. They remain advisory; no fake
   equation is inserted, no binding is inferred and no final check is weakened.
4. A variable-stage delivery/completion repair sequence receives the original
   local attempts plus two recovery calls (five consecutive attempts with the
   current three-attempt setting). Valid incremental declaration turns can still
   continue under the overall request/token cap. Successful completion resumes
   shared processes, then remaining topology. Exhausting a repair sequence stays
   explicitly incomplete rather than jumping to whole-model reconstruction.
5. The new study reserves at most five global repair calls, under the existing
   total request/token ceilings and repair-token budget. Success stops early.
   Two consecutive no-change replies with identical remaining diagnostics stop
   with an explicit no-progress reason. This is a different allocation of the
   existing overall budget; it is not a matched-budget result against earlier runs.

These changes do not infer function-level duplication, impose textbook names or
certify scientific correctness. Multiple drivers in one term remain one joint
function. The variable and shared-stage reports retain their existing first,
post-local and post-global checkpoints.

## Offline review command

From the project checkout, with a saved inspection unpacked outside the new output
directory (no LLM, optimizer or scheduler calls):

```bash
PYTHONPATH=src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/minimal-deferred-v2-inspection-d04b08f9 \
  --output output/focused-prompts-review-20261010
```

Use a fresh output directory. `PROMPTS.html` lists every stage and request, with
direct feedback examples, the new model view, complete request/schema/settings,
and marked differences. `PROMPTS.json` and `SUMMARY.json` support inspection.

The renderer verifies sealed historical event/call chains and projects each
actual displayed predecessor using the same function as the live runner. It does
not chain old replies through hypothetical new feedback. Byte counts are not
token counts. Existing focused-policy calls are shown as recorded. A fresh live
confirmation is needed to measure actual prompt tokens and proposer behavior.

No live test is submitted by the renderer. Review the prompts before using the
new opt-in campaign study.

## Repeated construction seeds

The ACES launcher now defaults this study to five sampling seeds (0–4): eight
public cells, Full specification, 40 fresh variable-to-topology episodes. This is
repetition of each cell, not five new datasets. Every episode keeps the reviewed
prompt policy, public context, model revision, temperature (0.2), checks, and
request/token budgets. Seeds propagate to every physical request through the
existing stage/attempt seed derivation. Each episode has its own namespace and
cache. Cases rotate within each seed's block to reduce fixed scheduling order.
No functions, fitting or test data are involved.

Use all five outcomes, not the best seed. Compare future prompt versions on the
same seed roster and serving configuration. Report per-case first reply,
post-local-repair and post-global-repair checks from `VARIABLES.md`,
`SHARED_PROCESSES.md`, and the topology report. `summary.json` adds
`per_case_seed_results`: planned/finished counts, statuses and every seed's
outcome, plus terminal-run cost median and unscaled MAD. Pending is not failure
or a zero-cost completed episode. Targeted error counts and equation inspection
remain necessary: structural completion alone does not establish a scientific
fix. Five seeds support diagnosis, not a precise estimate of rare-error rates.
Same-seed runs on different hardware are not guaranteed to be identical.

From a pinned new checkout on ACES:

```bash
export AF_COMPARISON_STUDY=focused_prompt_confirmation
export AF_COMPARISON_SEEDS='0 1 2 3 4'
export AF_COMPARISON_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-focused-prompts-seeds5-v1
bash scripts/hpc/start_phase_c_construction_comparison.sh run
```

After the jobs finish, use the same exports and checkout:

```bash
bash scripts/hpc/start_phase_c_construction_comparison.sh inspect
```

Workers retain the bounded three-hour window. A 40-task campaign may need more
than one allocation. If the report still contains pending tasks and the previous
wave is terminal, resume the identical plan with a new wave name:

```bash
export AF_COMPARISON_WAVE=seeds5-continuation-2
bash scripts/hpc/start_phase_c_construction_comparison.sh run
```

Completed outcomes and cached calls are reused, without resetting any episode's
budget. There is no automatic resubmission. Do not edit a saved plan or switch
its code/platform. Earlier eight-task plans must use their original source.

For direct plan preparation, `phase_c_construction_comparison.py prepare` accepts
`--study focused_prompt_confirmation --seeds 0 1 2 3 4`. The explicit roster is
sealed in the plan and checked on resume. Other studies retain their historical
rosters; the Python API's omitted-seeds default remains seed 0 for compatibility.

## Jetstream2 GPU VM deployment decision

Update: the user has now created an H100 80GB HBM3 VM (driver 580.159.03)
with a 200GB root disk. The earlier handoff's A100-only access assumption no
longer applies to this allocation. A separate bounded serving check is in
`JETSTREAM_CONSTRUCTION_COMPATIBILITY.md`; a full VM campaign is not yet launched.

A GPU VM can keep the model loaded across experiment batches without a Slurm
submission for each batch. VM creation/unshelving remains subject to available
capacity and project-wide quotas, including other agents' running instances.

As of 2026-10-10, Jetstream2 documents `g5.xl` (one H100, 80 GB GPU RAM) at
128 GPU SUs/hour, and `g3.xl` (one A100, 40 GB) at 64 GPU SUs/hour. A 40,000-SU
GPU allocation therefore represents 312.5 H100 hours or 625 A100 hours, excluding
any other consumption. H100 access may require enablement for the project.
See [flavors](https://docs.jetstream-cloud.org/general/instance-flavors/) and
[quotas](https://docs.jetstream-cloud.org/general/quotas/).

Claude's project-specific handoff at
`work/agent-comms/2026-10-10-claude-jetstream2-guide.md` reports A100-only access
and no GPU VM yet. Plan around one **g3.xl A100** unless the project UI confirms
otherwise; the general documentation's H100 listing does not establish this
allocation's access. Use the user's newer **40,000 GPU SUs**, not the handoff's
older 20,000. A100 support appears in vLLM's
[GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html),
subject to an actual pinned-container/32K-context preflight. Do not silently
change model size, quantization, context, sampling or repair budget while
comparing prompt versions.

The existing `js2` is a CPU VM with active baseline work. The baseline branch's
`gpu_server.sh` pins the same GPT-OSS-20B revision and vLLM 0.27.1/32K context,
but its setup installs baseline tools and its sequence concurrency is four times
its task count. Reuse the serving pins, not that entire baseline setup or its
sampling settings. Construction currently serves at most one sequence at a time.
Use a separate construction environment and result root, without changing an
active baseline checkout/server. GPU hardware identity and the actual driver
must be recorded; the CUDA 13 image requires a successful kernel preflight.

In Exosphere, select the IU allocation CIS261457, Create → Instance, an official
Ubuntu image, quantity one, and the available GPU flavor. A 200-GB volume-backed
root is a suggested starting size for the environment, one model and results,
subject to remaining storage quota. Add an SSH key and wait for setup to finish.
Use [Exosphere's creation guide](https://docs.jetstream-cloud.org/ui/exo/create_instance/).
After connecting, inspect `nvidia-smi`, `df -h`, and `df -i` before installation.

With an already configured project-scoped OpenStack CLI, read-only quota and
inventory checks are:

```bash
openstack quota show --usage
openstack flavor list
openstack server list
```

The current production launcher is still ACES-specific and pins
`platform=aces-h100x1`. This change adds repetition, not a tested Jetstream VM
launcher. A VM deployment needs its own recorded platform, pinned model/image,
GPU/driver record and startup smoke test before using a fresh plan. The eventual
inference server should bind to localhost and be accessed from the same VM or
through SSH, with complete request caches retained on persistent storage. It is
separate from the managed Jetstream LLM API used in earlier critic experiments.

Shelve an idle VM to stop compute charging; stopped/suspended instances still
consume SUs. Unshelving may encounter capacity limits. See
[Jetstream accounting](https://docs.jetstream-cloud.org/general/access/).
Jetstream staff may also limit GPU runtime to two weeks when resources are in
contention: [resource policy](https://docs.jetstream-cloud.org/general/resources/).
