# Structural repair: offline corrections and diversity design

25 October 2026. This milestone corrects declaration handling and one qualified
initialization proof, then replays saved replies. It does not launch a new model
comparison, search-policy experiment, or numerical fitting campaign.

## What the returned experiment tested

The previous run compared 20B and 120B under the same broad editing contract:
six saved T2-easy parents, three proposal seeds each, and three calls per episode.
Each call produced one complete parent-relative patch. It included multiple
independent seeds and short repair sequences, but no explicit diversity selection,
maintained candidate population, fitting, or adaptive fitting budgets. The broader
population architecture in the earlier design was a proposal, not an implemented
feature of that run.

## Corrected runtime

`construction-structural-revision-2` uses a new namespace and output roots. Old
plans and replies remain immutable and must be read with their original producer
or the explicit offline replay tool, never resumed under changed code.

The raw reply still passes field types, restricted syntax and original list-size
limits. The runtime can then consolidate identical declarations and matching
new-state/new-process equations repeated in `edits`, as well as matching new-state
initializers repeated in `initialization_edits`. It records the raw reply and
normalization diff. It does not choose between conflicting definitions, infer a
missing scientific expression, change parameter roles, or expand editing scope.
The normalized complete candidate still undergoes all static and scientific gates.
Evidence-property aliases resolve only when present in the actual displayed
candidate evidence; unknown citations remain invalid.

The opt-in `shared-symbolic-initial-readout-1` proof retains a shared symbolic
initial expression when a single unknown state can be isolated with a nonzero
rational coefficient. For example, exact observed `U(0) = A(0) + X(0) + c*B(0)`
with common U/A/B initials implies a common X initial for every shared c. Knowing
that equality does not require identifying c or X numerically. The affine-family
check can preserve the resulting trajectory identity. Nonzero observation
uncertainty, unconstrained remaining roots, unknown-parameter division, products
of unresolved coordinates and unsupported symbolic elimination remain deferred.
Historical affine callers retain their old behavior unless explicitly opted in.

## Saved-reply replay result

The corrected implementation rechecked all 108 replies and verified an identical
checkpointed resume. This is not a simulated live continuation: every reply uses
the unchanged parent that actually prompted it.

| Result | 20B | 120B |
| --- | ---: | ---: |
| Saved replies | 54 | 54 |
| Replies passing the current finite gates | 4 | 1 |
| Episodes containing those replies | 2 | 1 |
| Qualified model-data conflicts reached | 15 | 24 |
| New adopted models / fitting exports | 0 | 0 |

All five eligible reply assessments remain deferred. The four 20B replies are
the same direct-to-insulin independent-memory repair idea (P06-r1 seeds 1/2),
with a time-constant naming difference between seeds. The 120B reply is P06-r2
seed 2's second independent memory feeding I and U. These are promising changes
to the diagnosed premise, not established scientific adequacy or fitted accuracy.
All three P00-r1 seed-0 120B replies that escaped the old symbolic check now fail
the six affine-family conditions rather than being treated as unresolved evidence.

Local report: `output/structural-replay-analysis-20261025/REPORT.html`; full row
evidence and checkpoint receipts are in its `replay/` subdirectory. The exact
updated requests are in `prompt-preview/PROMPTS.html`. Source inspection archives,
old reports and historical token budgets remain unchanged.

Verification: 235 relevant tests passed, including the fitting-resume module;
shell syntax, CLI smoke and identical saved-replay resume passed. A broad pytest
run was interrupted after 1,557 passes and one source-identity mismatch while
files were changing; the affected module passed after source stabilization. The
full repository suite is not claimed complete. Changed-file Ruff passes; global
Ruff reports 37 existing findings in `analysis/claude/make_tables.py`.

## Repetition audit

The two arms have 36 continuation calls each. Twenty-four calls repeat the
recorded scientific patch fingerprint within their episode: six in 20B and
eighteen in 120B. Each also retains the same unresolved first diagnostic and
citation IDs. Twenty-two diagnostics are identical; two differ only in the child
provenance hash, with identical scientific witnesses.

Sixteen repeats occur on the final call. The explicit repeat warning reaches
only two 20B requests and six 120B requests, all on the final attempt. Five of
those six warned 120B replies repeat again. This identifies limited opportunity
for the existing warning, not a causal estimate of its effectiveness.

Three further equal-mechanism transitions become visible after removing duplicate
serialization, but they progress from schema errors to later validation. Those
are useful mechanical repairs, not stagnation. Repetition handling must therefore
normalize and validate first, and then compare both the resulting candidate and
its unresolved conflict. Different prose, renamed symbols or coefficients within
the same contradicted family do not establish useful diversity.

## Recommended search-policy experiment, not yet implemented

Asking for three options is promising, but does not guarantee three distinct
scientific alternatives. Each option should identify the affected target/witness,
the premise it changes, and a self-contained complete patch. Let the proposer
state that fewer coherent alternatives are available instead of requiring filler.
Validate alternatives separately; do not merge their edits.

A bounded diagnostic can compare:

| Policy | Calls within one episode | Maximum complete patches |
| --- | --- | ---: |
| Serial repair | One proposal plus up to two continuations | 3 |
| Independent breadth | Three fresh one-option calls from the same parent | 3 |
| Three options plus shallow repair | Up to three options in call 1; repair two different options at most once each | 5 |

Use the same three-call, 98,304 charged-token and 8,192-output-token-per-call
ceilings. Three options in one call share its output allowance; it is not tripled.
This explicitly compares packing/breadth/depth policies, not equal numbers of
candidates or independent samples. Report actual spend, incomplete options and
truncation. Match evaluation by parent-level success and cumulative tokens, rather
than rewarding a policy for submitting more individual candidates.

A six-parent, one-seed-bundle diagnostic costs at most 54 calls per model across
all three policies. Any later seed replication is separately budgeted. Keep the
runtime, witnesses and model fixed when isolating search-policy effects; model-size
replication can follow. No numerical rank is needed yet. Candidate eligibility
means no currently certified conflict, with deferred checks shown separately.
Scientific validity and eventual predictive quality require further evaluation.

Other changes worth isolating are:

- Keep a compact cumulative ledger of attempted mechanisms and unresolved target
  conflicts, rather than only the last reply. Show the complete current proposed
  patch so an earlier useful correction is not accidentally omitted.
- After normalization and validation, stop or branch on the first repeated
  unchanged failure; preserve the incumbent and spend remaining allowance on a
  fresh alternative. Do not block a successful metadata correction merely
  because its equations are unchanged.
- Distinguish cache identity, structural/scientific identity, and unresolved
  diagnostic identity. An exact syntactic fingerprint is useful bookkeeping, not
  a proof of algebraic equivalence or diversity.
- Later, allocate fitting budgets adaptively only after numerical score/recovery
  controls support the intended ranking decision. Do not prune scientifically
  plausible models because an unqualified optimizer failed to find parameters.

## Offline replay command

The CLI authenticates the two pinned uploaded archives before reading members,
checks sealed parent/call/completion receipts, and checkpoints each recheck under
a new implementation identity. It reads no archived code and makes no provider,
optimizer or rollout calls. Each saved reply is independently evaluated against
its original parent; later saved replies are not counterfactual responses to new
feedback. Passing current gates is not recorded as a new live success or fitting
export. Source records and historical budgets remain unchanged.

From the project checkout on the local Mac:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/replay_structural_revision.py \
  --archive-20b "$HOME/Downloads/inspection (1).tar.gz" \
  --archive-120b "$HOME/Downloads/inspection (2).tar.gz" \
  --root output/structural-replay-analysis-20261025/replay
```

An exact rerun resumes/verifies saved records. Different inputs or implementation
require a fresh output directory. No ACES/Delta/Jetstream submission is required
for this milestone. The prompt preview is retained with the local analysis report.
