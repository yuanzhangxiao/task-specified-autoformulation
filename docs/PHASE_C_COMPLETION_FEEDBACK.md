# Targeted construction completion feedback

Policy: `minimal-completion-feedback-1`.
Study: `completion_feedback_confirmation`.

This is a fresh eight-case, Full-specification, five-seed confirmation on the
verified Jetstream H100 server. It stops after topology. No interaction functions,
fitting, critic, test access or automatic follow-up are included. Use a fresh root;
historical campaigns remain tied to their original code.

## Returned five-seed run and ownership correction

The run at `0309cac` finished all 40 tasks: 24 complete before global repair and
35 afterward, versus 25 and 35 in readable-v1. There were 183 measured requests,
1,914,783 tokens, no budget stops, and one length-limited response. Episode token
median/MAD was 43,255/13,515.5. Seven post-first-reply memory-assignment questions
were resolved in the next reply. Variable stages completed in 39/40 episodes;
shared-stage local checks completed in 32/39 reached episodes. These counts are
declaration/structural checks, not scientific correctness.

| Case | Readable-v1 final / 5 | Completion-v1 final / 5 |
| --- | ---: | ---: |
| T1-easy | 5 | 5 |
| T1-hard | 5 | 5 |
| T2-easy | 4 | 5 |
| T2-hard | 4 | 4 |
| CSTR | 4 | 2 |
| Alien device | 5 | 5 |
| Coupled basin | 3 | 4 |
| Independent basin | 5 | 5 |

The same cells, seeds, serving provenance and model/budget settings were verified.
At matching case/seed positions, 32 passed both runs, two failed both, three
improved and three worsened. These fresh samples do not identify prompt changes
as the cause of each outcome. Every final assessment and all 183 request/event
records were reproduced or verified offline without generation or fitting.

Inspection found a feedback regression: `_empty_equations` treated a retained
variable-stage declaration as requiring an ordinary equation even after a process
declaration already owned its definition. Seven requests in T2-hard/seed 0 and
CSTR/seed 0 consequently asked for `equations` entries that the duplicate-definition
validator prohibited. The correction excludes process-owned names from this
feedback. It does not relax the validator or normalize incompatible definitions.
Deleting a process while retaining its variable declaration still requires an
equation. Regression tests cover local and shared process definitions, both
topology and repair stages, and removal transitions.

Other failures remain: coupled-basin/seed 0 omitted its algebraic-readout feedback
assignment; CSTR/seed 2 proposed a valid flow-classification change inside three
transactions that also both replaced and removed `C`; CSTR/seed 4 changed process
meanings/consumers and left an undefined jacket variable and composite balance.
No partial transaction is salvaged or scientific edit inferred by this correction.

The corrected launcher uses `completion-seeds5-v2`. Preserve v1 and inspect/resume
it only with its original `0309cac` bundle. The policy label and scientific checks
remain unchanged; source and execution digests distinguish the corrective commit.
The correction removes exactly seven erroneous feedback items on saved inputs;
it does not recover historical models or establish improved live completion.

Scientific inspection remains necessary: CSTR/seed 1 passed the implemented graph
checks with only jacket heat exchange, omitting feed and reaction contributions.
Coupled-basin/seed 3 returned downstream outlet flow to upstream storage. Those
public-science concerns are not resolved by this feedback correction, and changes
to the scientific coverage contract need a separate, reviewed milestone.

## Evidence and scope

The readable-prompt run reached 35/40 final structural completions (25/40 before
global repair). Every seed finished 7/8, with different failing cells. Variables
completed in 38/40; shared-stage checks completed in 32/38 reached episodes.
The 189 calls used 1,936,940 observed tokens with complete usage; episode median
39,515, unscaled MAD 13,647. All call/event chains and final assessments were
verified offline. These are structural checks, not scientific recovery or NMSE.

Five incomplete episodes motivated four changes:

1. Missing required memory assignments become direct structured confirmation
   questions. Each eligible state is a conditional choice; a matching variable
   name is not an assignment. The proposer can choose multiple states or declare
   another one. No automatic binding or extra call is introduced.
2. Equation feedback identifies the exact generated quantity, its current
   entry, and already inserted shared uses. Defining a quantity used on its RHS
   is not the same as defining the missing quantity. A generated supplied
   observation may be explicitly returned to supplied use if scientifically
   intended; public targets cannot. Incorrect optional memory bindings show an
   explicit `remove_bindings` edit, which removes no state or public requirement.
3. Transfer errors show the boundary-flow versus internal-transfer decision
   together. The conditional classification-only edit keeps all signs,
   dependencies, conversions and uses. It does not fix duplicate or supplied
   receivers. Runtime does not invent another balance or silently choose the
   scientific classification.
4. New-request budget diagnostics save exact charged tokens, active cap, UTF-8
   request bytes, output allowance, reservation, shortfall and the separate
   serving-token context result. Blocked requests initiate no generation and
   consume no reservation. `summary.json` includes `budget_stops`; full diagnostics
   are under each task's `calls/budget_preflight/`. Global-repair prompts show the
   active allowance, reconstructed at the historical call position on resume.

The system/stage instructions, model view, response schemas, public specifications,
scientific predicates and model settings are unchanged from readable-v1. Only
conditional feedback and operational budget visibility change. Equal-dependency
contributions remain deferred to interaction review.

## Budget boundary

Each construction retains at most 128 requests and 524,288 charged tokens.
131,072 tokens and at most five calls are reserved for global repair. Unspent
construction allowance does not become extra repair allowance. The variable
stage retains its three initial attempts plus two bounded recovery attempts;
global repair can stop on success or two identical no-progress replies.

Admission still reserves `len(json.dumps(full_request).encode()) + max_tokens`.
The full request includes the response schema. This byte-based bound differs from
the tokenizer's context check and can stop a request that fits the context.
Reported provider token usage replaces the reservation after a response; unknown
usage remains charged conservatively. This milestone makes that distinction
auditable without silently increasing the budget or changing the accounting rule.

## Preview and offline verification

```bash
PYTHONPATH=src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/jetstream-readable-inspection-20261012 \
  --output output/completion-prompts-review \
  --policy minimal-completion-feedback-1
```

The HTML marks request differences and shows exact model/feedback payloads on
each actual historical predecessor. It does not claim that old replies answered
the new questions or establish improved live outcomes. Historical repair-budget
values use recorded charges; prospective costs remain unknown.

Relevant tests: `test_construction_completion.py`, `test_construction_focus_review.py`,
`test_jetstream_construction_campaign.py`, and existing constructor/ledger tests.
They cover both valid choices and unresolved/invalid ones, unchanged acceptance,
cached resume without new requests, and budget rejection before generation.

## Jetstream execution

Use the delivered, checksummed source/input bundle on the existing VM. It reuses
the running localhost-only server, virtual environment and compatibility record;
it installs no packages or weights. From the extracted bundle directory:

```bash
sha256sum -c MANIFEST.sha256
bash scripts/jetstream/run_completion_construction.sh run
bash scripts/jetstream/run_completion_construction.sh status
```

Default root: `/home/exouser/af-construction/completion-seeds5-v2`.
Default roster: eight cells × seeds 0–4, including both basins. A background worker
has a three-hour window. If it drains with pending work, repeat `run` from the
same bundle and root; per-episode budgets do not reset.

After the worker stops:

```bash
bash scripts/jetstream/run_completion_construction.sh inspect
```

Download from the local Mac terminal:

```bash
scp exouser@149.165.152.222:/home/exouser/af-construction/completion-seeds5-v2/inspection.tar.gz \
  ~/Downloads/jetstream-completion-v2-inspection.tar.gz
```

Optional ACES launch uses a separate root after quota headroom is restored:

```bash
export AF_COMPARISON_STUDY=completion_feedback_confirmation
export AF_COMPARISON_SEEDS='0 1 2 3 4'
export AF_COMPARISON_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-completion-feedback-seeds5-v2
export AF_COMPARISON_SOURCE="$PWD/inputs/topology-source"
bash scripts/hpc/start_phase_c_construction_comparison.sh run
```

Do not pool ACES and Jetstream results as one matched prompt comparison.

## Remaining limitations

Structured choices can still be misunderstood or ignored. The runtime cannot
choose scientific mediators, signs, or dependencies. An error introduced by the
last permitted reply remains unresolved. Structural completion does not certify
rate interpretation, conservation, function overlap, mechanism compliance, or
predictive accuracy. New live results are needed to evaluate this feedback.
