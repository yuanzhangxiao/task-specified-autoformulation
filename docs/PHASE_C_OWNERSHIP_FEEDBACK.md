# State/process ownership feedback and training evidence

Policy: `minimal-edit-feedback-2`. Study: `ownership_feedback_confirmation`.
The implemented change concerns editing feedback. The training-evidence design
below is a proposal for review, not a new prompt feature or acceptance gate.

## Observed problem and narrow correction

The five-seed edit-feedback run completed all 40 episodes: 39 structurally
complete, one incomplete, 163 measured requests and 1,786,222 observed tokens.
Structural completion is not scientific recovery. The previous decisions run
completed 37/40; sampling variability prevents treating that difference alone as
a measured effect of the feedback change.

In T2-hard seed 0, an attempted process definition reused the name of a retained
differential state. The reply was rejected. The next prompt simultaneously said
to preserve the state and to remove its ordinary equation in favor of the
process. The latter advice was meant for an algebraic quantity, not this state.
The proposer subsequently tried to remove a process that had never been
committed.

The revised presentation resolves the two instructions using the actual retained
declarations:

- A differential state retains its ODE. To introduce an instantaneous function
  of that state, use a different process name, or an ordinary contribution.
- A rejected process addition does not exist in the retained model. Withdrawing
  it means omitting the attempted addition, not deleting a retained definition.
- Removing a process requires a retained process key. A state name is not such
  a key merely because a rejected reply tried to define a process with it.
- A deliberate change in scientific representation remains a coordinated,
  explicit proposer edit; a naming collision does not justify an inferred
  runtime type change.

The acceptance predicates, transaction semantics, scientific requirements,
training evidence, schemas, stage instructions and budgets are unchanged.
Earlier prompt policies remain available for deterministic historical resume.
The runtime does not reinterpret a rejected reply as an accepted model.

## Initial training highlights: proposed design

The benchmark-general follow-up is
[`PHASE_C_TRAJECTORY_COMPATIBILITY.md`](PHASE_C_TRAJECTORY_COMPATIBILITY.md),
which assigns proposed checks to stages and makes their assumptions explicit.

The existing `training_observations` packet already describes sampled extrema,
bounded trajectory paths, initial ranges and input changes. It gives individual
trajectory statistics, without a concise cross-trajectory observation list.

A short, deterministic list at the start of the packet could foreground:

1. Per-channel sampled extrema and locations, without selecting causal peak
   pairings or a delay interpretation.
2. Changes that continue after an input pulse ends.
3. Contrasting responses across trajectories with explicitly identified matching
   inputs or initial observations.

Each statement should cite its training trajectory, channel and sample/event
indices. Use measured units and say when a maximum is tied or evidence is
truncated. The same frozen data-only summary should accompany initial
construction and repair. Preserve the detailed packet for inspection; avoid
duplicating it in feedback. No second LLM is needed to extract these facts.

A later sampled peak is evidence of timing, not a causal delay estimate. Different
responses at the same instantaneous input do not identify a missing variable:
history, initial state, measurement noise or allowed parameter heterogeneity may
explain them. The list should therefore separate observations from conditional
interpretations and leave the scientific choice to the proposer.

## Detecting a missing dependence after topology

The useful question is whether the proposed model can distinguish two observed
training trajectories. This is more specific than checking whether one input
takes the same value.

For an algebraic output, compare every proposed argument, covariate and parameter.
For an ODE, expand the complete generated ancestor subsystem, including shared
processes, coupled states, initialization maps and any permitted history inputs.
Under common parameters and a unique deterministic forward solution, matching
initial conditions and complete input histories imply matching predictions.
Different observed responses then contradict that candidate under those
assumptions. The contradiction does not identify which new scientific dependence
should be introduced.

For example, a candidate `dI/dt = f(I, external_insulin)` with the same initial
`I`, zero external insulin in both trajectories, and shared parameters must
predict the same insulin history in both. The saved public training packet has
two trajectories with matching initial insulin but markedly different later
insulin readings. A meal/glucose-driven mechanism or another permitted source
of distinguishing information is missing from this candidate. The runtime can
display the witness and ask the proposer how to represent it; it cannot conclude
that a specific textbook edge or equation is mandatory.

Important scope limits:

- Equal instantaneous inputs do not establish equal input histories or equal
  state initialization. Sparse displayed samples do not establish equality
  between samples unless the public interpolation contract provides it.
- Unknown latent initialization, differing permitted parameters, unavailable
  observed histories or covariates make the comparison unresolved.
- Near-equality and noisy observations require quantitative tolerances and
  behavioral assessment, not an unconditional topological rejection.
- The current evidence packet hashes target and external-input arrays. It does
  not contain full auxiliary histories or fixed-covariate values. A general
  comparison must obtain and freeze all relevant training information before
  claiming that candidate inputs are indistinguishable.

This suggests two distinct components: data-only highlights in the initial
prompt, then candidate-specific sufficiency diagnostics after topology. Neither
component should choose a new scientific dependence automatically. They are not
implemented by the ownership-feedback patch.

## Review and optional fresh confirmation

Generate an offline preview from the actual saved predecessors:

```bash
cd /Users/yuanzhangxiao/Projects/autoformalism
PYTHONPATH=.:src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/jetstream-edit-feedback-inspection-20261013 \
  --output output/ownership-prompts-review-local \
  --policy minimal-edit-feedback-2
```

Use a new/empty output directory. Historical replies are not replies to the new
prompts. The maintained marked preview is
`output/ownership-prompts-review-20261013/PROMPTS.html`; the separate factual
insulin example is
`output/ownership-feedback-review-20261013/training-observation-example.json`.

The registered `scripts/jetstream/run_ownership_feedback_construction.sh` wrapper
uses a fresh `ownership-feedback-seeds5-v1` root, all eight Full cells and seeds
0–4. It reuses the verified local server and unchanged limits, stopping after
topology. Registration and offline preview do not launch a live experiment. A
new pinned source checkout is required; historical campaigns keep their original
source and plan.

## Offline verification

The preview checks all 163 saved requests from the actual retained predecessors.
Only the two affected T2-hard seed-0 repair packets change. All 42 variable-stage
requests remain identical, as do the public brief, model view, requirement
checklist, output schema, generation settings and other non-feedback fields.
The revised user packets grow by 0.052% overall in UTF-8 bytes; no serving token
measurement is claimed. All 165 HTML pages and 341 local links were checked.

Focused tests cover existing feedback, malformed replies, a valid deliberate
state-to-process redesign, retained/rejected ownership, policy registration,
fixed budgets, cached resume and offline/live prompt agreement. Preparing a
prompt does not change acceptance or mutate the draft. The tests and previews
do not establish that the proposer will repair the remaining live episode.

The full regression run completed with 4,783 passes, eight skips and five
failures while source edits were still occurring. All five failed tests passed
when rerun after the source was fixed; source/runtime fingerprint checks explain
the directly reported failures. Focused ownership and routing tests also passed.
Ruff passes on every changed Python file; repository-wide Ruff still reports 37
pre-existing findings in untracked `analysis/claude` scripts. No unrelated files
were changed to silence those findings.

## Live confirmation: 2026-10-13

The `phase-c-ownership-feedback-e867372` inspection contains all 40 terminal
episodes: eight Full cases, seeds 0–4, ending after topology. Its plan identity
is `dd2ed5be34a26b5988c77e51c1ff1bd63646643bf97fed5d1e064de7a82c0c1a`.
The supplied archive SHA-256 is
`707fe7937ecd265b17758e93c929c3a5d1d4f65fb5676b78d20eaee44777b937`.

| Checkpoint or cost | Previous edit feedback | Ownership feedback |
| --- | ---: | ---: |
| Variable stages complete | 40/40 | 40/40 |
| Shared-process stages complete | 35/40 | 37/40 |
| Whole topology ready before global repair | 27/40 | 25/40 |
| Whole topology ready after global repair | 39/40 | 39/40 |
| Physical requests | 163 | 160 |
| Observed total tokens | 1,786,222 | 1,764,729 |
| Episode tokens: median [unscaled MAD] | 35,212 [10,698.5] | 36,305 [10,472] |
| Requests with unmeasured usage | 0 | 0 |

Stage completion certifies the implemented checks, not scientific correctness.
An empty shared-process decision is permitted. The pre-global checkpoint includes
local stage retries. The prior source is `401a995`; the new source is `e867372`.
Scientific rules, public cases, schemas and request budgets are unchanged.

The corrected ownership feedback was exercised **zero times**. The previously
blocked T2-hard seed 0 completed without its earlier collision, while Alien-device
seed 1 became incomplete. All 40 first-variable requests, including seeds, are
identical across runs, but 17 responses differ. T2-hard seed 0 diverges at an
identical shared-stage request before the correction could be shown. These
outcomes do not isolate an effect of the patch.

### Reply metadata entering the model

Alien-device seed 1 already contains the input-to-memory-to-output path, but its
ordinary-topology reply adds an empty equation named `stage_complete`. Both
repair requests display the correct `remove_equations` option. The proposer
instead requests removal of a nonexistent variable; its next reply includes the
correct equation removal but repeats the invalid variable removal. Atomic
validation rejects both transactions, and the two-unchanged-repairs stop applies.
It does not exhaust the five-repair limit. Applying only the equation removal
passes the existing structural checks in an offline hypothetical edit; the saved
campaign and its 39/40 result remain unchanged.

T1-hard seed 1 exposes the opposite failure: a process named `stage_complete`,
explicitly described as a completion placeholder, is accepted and contributes to
the glucose balance. This is a protocol/model separation error that does not
require fitting to identify. A narrow next correction should recognize reply
control fields and give removal feedback in the correct retained namespace.
The valid deletion hint was already present in the blocked example; its absence
must not be claimed as the cause.

### Scientific findings and next milestone

All ten T2 final models leave insulin dependent only on insulin itself and/or
external insulin. The public training packet includes two trajectories with
equal initial insulin and zero external insulin throughout, but different later
insulin readings. Under shared deterministic, well-posed laws and parameters,
and the prescribed observed-state initialization, these isolated subsystems
cannot reproduce both histories exactly. This conditional witness neither
identifies a unique missing edge nor measures achievable NMSE. None of the five
seeds in either T2 case resolves this particular gap.

All five coupled-basin topologies retain two dynamic storage coordinates and
transfer uses with negative upstream and positive downstream signs. This is
useful structural evidence; functions, initialization and conversions remain
unassessed. A null conversion is unknown, not a verified unit conversion.
Similarly, matching arguments do not establish duplicate physical effects, and
the differential type of a rate-valued quantity is not inherently invalid.

After the narrow metadata correction, proceed with the benchmark-general
training highlights and conditional model/data diagnostics in
`PHASE_C_TRAJECTORY_COMPATIBILITY.md`. Increasing the structural completion rate
alone would miss the principal remaining limitations.

### Inspection verification

The audit validates all 243 current and 246 previous sealed JSON artifacts,
reconciles the raw usage, and recomputes all 160 combined pre-global/final
assessments across the two runs. All match. The 160 current edit transactions
also replay exactly (153 accepted, seven rejected). One output-length-limited
response recovered through its local retry. No token-admission stop was recorded.

The local review is `output/ownership-analysis-20261013/REPORT.html`, with all
40 equation histories, complete requests, feedback, replies and audit JSON files.
Sixty relevant tests passed; repository-wide Ruff reports the same 37 unrelated
findings in untracked `analysis/claude` scripts. Inspection made no runtime or
benchmark edits, LLM calls, optimizer calls, rollouts or test-data access.
