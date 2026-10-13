# Public requirement coverage in minimal construction prompts

Policy: `minimal-public-coverage-1`. Study: `public_coverage_confirmation`.
This is an opt-in prompt and feedback milestone, stopping after topology.

## Motivation and scope

Completion-v2 finished all 40 episodes, with 38 structurally complete topologies
(26 before global repair), 175 measured requests and 1,795,243 observed tokens.
All variable stages passed their declaration checks. Scientific inspection still
found inconsistent interpretations, including a generated disposal quantity
disconnected from the glucose balance, an exchange removing material from both
compartments, and a downstream outlet returning water upstream. These findings
are not equivalent to a comprehensive scientific correctness score.

The new policy changes three stage instructions and conditional feedback:

- Shared processes: identify the physical loss/gain before choosing signed uses;
  distinguish an internal transfer from a flow leaving the modeled system.
- Remaining topology: review each public mechanism against the assembled model,
  including consistency between related targets and balances. A declaration or
  passing path predicate alone does not establish this interpretation.
- Repair: preserve intended mechanisms across coordinated edits, and correct a
  mistaken sign instead of relabeling an internal transfer to bypass its check.
- Existing failed path checks now name exact endpoints and the fields that can
  repair a pathway, without prescribing a direct edge or a scientific sign.
  Multiple assigned memory states retain the check's “at least one” semantics.
- Duplicate `equations` keys show how to return one complete local-process list
  for each quantity. The runtime does not merge conflicting lists or salvage
  part of an invalid transaction.
- Flow-classification feedback no longer offers a ready-made classification-only
  patch. The proposer chooses the meaning, uses and signs explicitly.
- Supplied-receiver feedback presents both permissible modeling and continued
  supplied use, instead of a ready-made patch dropping the receiver. A modeled
  exchange remains an internal transfer when both balances are generated.

The system instruction, variable-stage instruction, model view, response schema,
public science/training evidence, scientific predicates and budgets are unchanged.
Historical prompt policies retain their own wording. Source hashes also pin each
run; resume historical runs with their original bundle, not this one.

No mandatory `U -> Gp` edge, textbook state inventory, universal decay,
natural-language scientific verifier, automatic sign flip, or extra self-review
call is added. Equivalent shared and local representations remain allowed.
Matching dependencies do not trigger a topology overlap rejection. Function
forms, signed-rate interpretation, conversions, conservation, thresholds and
predictive behavior still need later assessment where not decidable here.

## Confirmation and inspection

Use the existing verified Jetstream H100 server and a fresh root. The roster is
the same eight Full-specification cases (including both basins), seeds 0–4:
40 fresh constructions. Settings remain GPT-OSS-20B, temperature 0.2, low
reasoning, 32,768 context tokens, 8,192 maximum output tokens, 128 requests and
524,288 charged tokens per construction. Up to five global repairs share the
reserved 131,072 tokens. Existing local retries and early stops are unchanged.
No fitting, critic, test access or automatic follow-up is included.

From the delivered, extracted, checksummed bundle on the VM:

```bash
sha256sum -c MANIFEST.sha256
bash scripts/jetstream/run_coverage_construction.sh run
bash scripts/jetstream/run_coverage_construction.sh status
```

Default root: `/home/exouser/af-construction/coverage-seeds5-v1`. The launcher
starts a background worker and reuses the local server without installing
packages or downloading weights. If its three-hour window ends with pending
episodes, repeat `run` from the same bundle/root; budgets do not reset.

After the worker stops:

```bash
bash scripts/jetstream/run_coverage_construction.sh inspect
```

From the Mac terminal, download:

```bash
scp exouser@149.165.152.222:/home/exouser/af-construction/coverage-seeds5-v1/inspection.tar.gz \
  ~/Downloads/jetstream-coverage-inspection.tar.gz
```

Inspect variable and shared-process reports, pre/post-global outcomes, all
equations and exact request/response traces. Compare per-case/per-seed results
with completion-v2; report median/unscaled MAD costs. Assess scientific coverage
separately from runtime completion, retaining ambiguous cases. Do not select a
“scientifically correct best-of-five” from structural status alone or pool
repeated seeds across prompt versions as independent repetitions.

## Offline prompt review and verification

```bash
PYTHONPATH=src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/jetstream-completion-v2-inspection-20261012 \
  --output output/coverage-prompts-review \
  --policy minimal-public-coverage-1
```

This preview verifies saved call/event chains and applies the new presentation
to each actual historical predecessor. It marks differences but does not treat
historical replies as responses to the new prompts or create a hypothetical
repaired chain. It preserves public evidence and response schemas verbatim.

Tests cover exact path endpoints and alternative memory states, immutable
drafts, untrusted rejected payloads, duplicate-entry repair, sign/classification
choices, unchanged variable requests, live/preview agreement, bounded cached
resume and the frozen five-seed roster. Scientific improvements require the
new live results; these instructions do not certify them in advance.
