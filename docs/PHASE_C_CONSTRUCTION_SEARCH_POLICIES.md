# Construction search policies: design and ACES runbook

25 October 2026. `construction-search-policy-1` compares construction search
policies using six frozen T2-easy models selected for defect coverage. It is not
a fresh six-benchmark evaluation. This delivery implements the experiment and
offline checks; it does not claim a live GPU result, fitted accuracy, or model
promotion.

## Question and fixed inputs

Does independent breadth or several alternatives followed by shallow repair
produce more distinct eligible models than sequential repair with restart?
Compare policies within each model size before interpreting 20B/120B differences.

Both arms use the same saved parents, training evidence, public requirements,
editing scope and deterministic gates. They retain declaration normalization and
the qualified shared-symbolic-initial-value check from the preceding milestone.
Benchmark data, finalized public prompts and observation-window policy do not
change. No validation/test data, numerical fitting or NMSE-based selection enters
proposal generation.

| Policy | Up to three calls in one episode | Maximum candidate patches |
| --- | --- | ---: |
| `sequential_restart` | One proposal, then repair; after a repeated unchanged failure, use the remaining call for a fresh alternative | 3 |
| `independent` | Three fresh proposals from the same parent, without previous failed replies | 3 |
| `options_shallow` | Up to three complete alternatives in call 1; repair two distinct failed branches at most once each | 5 |

The sequential policy includes restart, so this is not a pure breadth-versus-depth
ablation. All three policies use the corrected interface; this experiment does
not separately estimate the effect of each bookkeeping improvement.

## Bookkeeping and diversity

Every trial is a complete patch against the unchanged parent. Failed trials are
not partially installed. Repair requests show the complete intended patch and,
when statically available, its equations and initialization rules. A compact
ledger retains attempted families, affected targets, edits and unresolved errors.
Feedback also identifies earlier trial actions omitted from the latest patch;
their usefulness is not assumed. The proposer must retain, replace or deliberately
revert them. The runtime never merges alternatives automatically.

Repeated failure means the same identified complete model family and unchanged
diagnostic after normalization and validation. Correcting a serialization defect
can be mechanical progress even when the intended equations are unchanged. Once
sequential repair repeats an unchanged failure, its fresh request keeps the
compact ledger but omits the previous full failed patch. Independent proposals
receive no failed-trial history.

Family identity ignores consistent renaming of new private states, processes and
parameters while preserving equations, initializer scope/maps, parameter sharing,
roles, domains and public/parent identities. Replacing one filter with an otherwise
identical filter whose freely fitted time constant has a new name is the same
family. Retaining the first filter and adding a second is different. Same topology
alone is not sufficient for equality.

This is bounded structural canonicalization, not general algebraic equivalence.
Expression ordering, literal constants and nonlinear forms remain significant.
Existing parent symbols remain anchored. The bounds are 128 private symbols,
4,096 residual canonical labelings and 32,768 expression nodes. Unsupported or
over-budget identities defer; they are not asserted to match. Exact patch
fingerprints provide a narrower fallback for bookkeeping, not proof of diversity.

Each option identifies its affected target and changed scientific premise. Fewer
than three coherent options are allowed with an explanation. Valid sibling
options are assessed independently. An invalid whole batch or truncated response
ends that option episode without inventing alternatives; it remains a reported
format/provider outcome. Failed branches are chosen by fixed seeded ordering,
with recognized duplicate families skipped. No numerical ranking is used.

## Matrix and budget

| Setting | Per model arm |
| --- | ---: |
| Parents × seed bundles × policies | 6 × 3 × 3 |
| Episodes | 54 |
| Maximum generation calls | 162 |
| Calls per episode | 3 |
| Charged-token ceiling per episode | 98,304 |
| Context / output ceiling per call | 32,768 / 8,192 |
| Temperature / reasoning | 0.2 / low |
| Included operational smoke subset | First sorted parent, seed 0, all three policies |
| Smoke episodes / maximum calls | 3 / 9 |

Both model arms together have at most 324 calls. Smoke calls are included in the
full allowance, not added to it. The three initial options share one 8,192-token
completion. Different candidate counts, packing, correlated options and early
termination are part of the policy tradeoff. Equal ceilings do not imply equal
realized token use or independent samples. Independent breadth collects all three
proposals even after an eligible one; sequential repair can stop early.

20B uses revision `6cee5e81ee83917806bbde320786a8fb61efebee` on one H100,
8 CPUs and 64 GiB RAM. 120B uses revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a` on two H100s, 16 CPUs and 128 GiB RAM.
Each allocation requests four hours with a 12,600-second worker window. This is a
bounded allowance, not a guarantee all episodes finish. Local vLLM 0.27.1,
tokenizer assets, model revisions and serving identities are verified. Compiler
caches use node-local temporary storage.

## Reports and interpretation

Reports retain every episode, request, response, candidate, diagnostic and cost.
They summarize eligible episodes and parents, recognized distinct families,
candidate outcomes, deferred checks, and calls/tokens to the first eligible model.
Token spread uses median and unscaled MAD; success-only time-to-eligibility
summaries are explicitly labeled. Unavailable episodes remain in the planned
roster. Equation audits remain necessary to assess scientific plausibility.

An eligible candidate has no currently certified conflict under the finite gates.
Deferred evidence is unresolved; eligibility does not establish scientific
validity or fitting adequacy. Eligible families receive separate handoff packets
for independent fitting, without launching fitting or choosing a winner. Duplicate
recognized eligible families within an episode receive one handoff; all raw trials
remain available.

## Offline saved-reply check

All 108 historical replies were rechecked without generation: 54 per model arm.
The 20B replies contain four eligible replies across two episodes, and 120B
contains one across one episode, matching the preceding corrected gate replay.
All five retain deferred scientific checks. The new family/failure comparison
recognizes six repeated unchanged failures for 20B and 21 for 120B. These are
historical-reply diagnostics, not results from the new search policies.

The original Downloads archives were no longer present. Before deterministic
repacking, all 108 preserved call records were matched to the prior authenticated
replay hashes. The loader still validates inner plans, parent models, tokenization
and completion receipts. Both the original archive identities and new literal
repack pins are recorded; no archive code is executed. Exact replay/resume and
clean-bundle preparation were checked offline.

## Running on ACES

Upload and extract the delivered bundle in group scratch. Use its absolute
directory for `AF_CODE`; do not reuse a partial extraction or a previous campaign's
source directory. The bundle includes its pinned saved inputs and Harmony asset.
Its `run_search.sh` wrapper verifies every manifest entry before any launcher
action, so a failed or partial extraction cannot proceed to submission.

```bash
AF_CODE=/scratch/group/p.nairr260351.000/u.yx126462/EXTRACTED_BUNDLE_DIRECTORY
bash "$AF_CODE/run_search.sh" run 20b smoke
bash "$AF_CODE/run_search.sh" run 120b smoke
```

Default result roots are
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-search-policy-20b-v1`
and the analogous `120b-v1` directory. Generic old `AF_OUTPUT_ROOT` values are
ignored. Use `AF_CONSTRUCTION_SEARCH_ROOT` only for an intentional per-arm override.

After smoke jobs end, refresh reports and inspect their operational outcomes:

```bash
bash "$AF_CODE/run_search.sh" report 20b
bash "$AF_CODE/run_search.sh" report 120b
bash "$AF_CODE/run_search.sh" inspect 20b
bash "$AF_CODE/run_search.sh" inspect 120b
```

`smoke_complete: true` means all three smoke episodes have terminal receipts,
not that they found scientifically valid models. Pending/preflight-blocked smoke
episodes prevent full submission. After reviewing smoke, full phases require
explicit commands and reuse all saved smoke work:

```bash
bash "$AF_CODE/run_search.sh" run 20b full
bash "$AF_CODE/run_search.sh" run 120b full
```

`status SIZE` shows jobs and saved summaries; `report SIZE` refreshes them.
`preview SIZE` rebuilds `PROMPTS.html`. `inspect SIZE` creates
`inspection.tar.gz` inside that arm's result root for download.

If a confirmed allocation ends before completing its phase, use a new explicit
wave while keeping the same source, output root and phase:

```bash
bash "$AF_CODE/run_search.sh" run 20b full wave2
```

Repeating the same wave returns its existing receipt. A new wave preserves all
call/token charges and refuses to overlap a prior active allocation. Ambiguous
submission responses remain blocked for scheduler reconciliation; changing the
wave name cannot bypass them. Source/runtime drift blocks resume. There is no
automatic full-phase submission, repeated allocation, fitting or follow-up.
