# Construction search: declaration and initializer requalification

The subsequent v2 smoke and the prospective v3 five-option pilot are documented
in [the breadth pilot and combined initialization checks](PHASE_C_BREADTH_PILOT.md).

26 October 2026. This is an offline correction and saved-reply audit following
the completed `construction-search-policy-1` experiment. No new generation,
fitting, rollout, held-out access, model adoption or fitting export is authorized
by replay. Historical outcomes and costs remain unchanged.

## What the completed experiment established

The six related T2-easy parents completed all 108 episodes across both model sizes.
All 281 calls returned, producing 289 candidate patches. The old finite gates
found 13 eligible 20B candidates and 11 eligible 120B candidates, all with deferred
checks. Eligibility was not scientific validation or fitting adequacy.

Three options were optional: 32 of 36 initial option replies returned one option.
Only four returned three. The completed run therefore does not establish the
value of enforced three-option breadth. In addition, 65 candidates hit declaration
identity errors; the subsequent audit identified safe normalization opportunities
and a misleading reserved-prefix diagnostic. One previously eligible 120B model
retained a shared affine initializer family that contradicts training observations.

## Prospective contract

`construction-search-policy-2` requires exactly three initial options in the
`options_shallow` policy. The JSON schema and runtime envelope both enforce the
count; there is no `fewer_options_reason` escape. Each option must propose a
revision and identify its affected target and changed premise. A short batch is
an explicit protocol failure, not an eligible one-option breadth experiment.
Malformed siblings are recorded independently when the envelope has three entries.
There are no extra free retries or synthesized alternatives.

The three options still share one 8,192-token completion and the episode retains
its three-call ceiling. Subsequent calls repair at most two distinct failed
branches once each. Three entries alone do not prove scientific diversity:
complete-model family identities and equation inspection remain necessary.
Private-symbol renaming does not establish a new repair family.

A reply contains **changes only**, not the complete parent model. The runtime
preserves untouched definitions and derives dependencies from expressions. An
edited equation supplies its complete replacement RHS; a new state supplies its
RHS and initializer. For example, changing `x' = A+B` into `A+C` requires an `x`
edit with RHS `A+C`, plus any genuinely new definitions needed by `C`; unchanged
`A` is not redeclared. Because rejected patches are not installed, a retry must
include all earlier changes still intended relative to the parent. Runtime never
guesses a scientific dependence or merges competing alternatives.

The opt-in `parent-relative-declaration-normalization-1` boundary removes only:

- Repeated existing global parameters with identical roles and unconstrained
  domain metadata; the original identity, sharing and metadata remain intact.
- A repeated top-level parameter already declared in exactly one initializer
  scope with the same role, absent from the complete dynamics and output laws.
- Unchanged existing process RHSs under restricted AST identity, with no competing
  edit or removal; parent scientific metadata is preserved.

Normalization neither turns a new declaration into a scientific edit nor changes
state/process types, parameter domains, supplied-channel identities or initializer
scope. A patch containing only unchanged restatements becomes `retain` and still
undergoes every data gate. Reserved-prefix, actual identity and initializer-scope
errors receive separate per-name explanations and repair instructions. Raw replies
and all normalization actions are retained.

## Weighted shared initializer certificate

`weighted-shared-symbolic-initials-1` expands actual lowered initializer maps into
expressions affine in shared unknown coordinates. For each coordinate, it seeks
exact parameter-independent weights that cancel its coefficient across the
conditioned trajectories, as well as the constant part and the sum of weights.
For a shared closed affine ODE with identical complete forcing, this establishes
the corresponding zero weighted output identity for all common times.

This covers the missed example with `I_mem(0)=a*I(0)+b*Gp(0)` and
`S(0)=c*I(0)+d*Gp(0)`: the initial values may differ between trajectories yet
still obey the same weighted identity for every choice of shared coefficients.
No fitting or parameter guessing is needed.

The certificate requires authenticated exact conditioning and observations,
shared parameters, complete matching forcing, unique open-loop solutions and no
resets/impulses. Relevant free trajectory roots, nonlinear shared-coordinate
products/divisions, unknown uncertainty or exhausted proof bounds defer. It is an
exact-reproduction necessary condition, **not a lower bound on achievable NMSE**.
The previous symbolic-readout behavior remains unchanged when the new flag is off.

## Reproducibility and interpretation

The saved-reply replay authenticates the old plans, parent contents, request/call
ledgers, tokenization and completion receipts. Each original patch is checked
independently against its original parent. Historical option-count shortfalls
are reported separately and are not retroactively used to discard old candidates.
Later historical replies did not receive the corrected feedback: rechecked counts
are not a live policy result and do not estimate the new prompts' effectiveness.

New live plans, if run, use fresh `phase-c-search-policy-{20b,120b}-v2` roots and
bind both new gate policies, prompt/schema hashes and executable code. Old roots
must be inspected with their original frozen bundle. No old campaign is resumed
using the changed source.

## Completed offline replay

All 289 patches and their 281 original call records were authenticated and
rechecked. The original Downloads archives were no longer available; the replay
uses separately pinned preserved file trees and verifies their plans, parents,
requests, tokenization and completion receipts. Archive hashes are retained as
historical provenance, not claimed as freshly reverified archive bytes.

| Model | Saved patches | Historical eligible | Rechecked eligible | Newly eligible |
| --- | ---: | ---: | ---: | ---: |
| 20B | 138 | 13 | 12 | 0 |
| 120B | 151 | 11 | 10 | 0 |

Both newly blocked candidates violate the weighted affine-family condition:
120B `P05-r2__seed1__options_shallow/call1-option0` and
20B `P01-r0__seed0__sequential_restart/call1-option0`. The latter's constant
offset repaired the initial `U` identity but left an incompatible trajectory
family. Remaining eligible candidates retain deferred checks.

Of the 65 historical `DECLARATION_IDENTITY` outcomes, 21 retain that diagnosis,
eight receive specific reserved-name feedback, and 36 reach a deeper outcome.
Twenty of those 36 now expose a qualified model-data conflict. Across the complete
replay, 52 outcome labels change and model-data conflicts increase from 103 to 130.
No additional candidate becomes eligible merely by clearing an interface obstacle.
These are gate corrections on saved replies, not results of new proposer behavior.

Reproduce locally from the preserved inspection trees:

```bash
.venv/bin/python -m scripts.replay_search_policy_results \
  --source-20b output/construction-search-results-20261025/input/20b \
  --source-120b output/construction-search-results-20261025/input/120b \
  --root output/construction-search-v2-replay-20261026
```

Completed rows are reused only under matching code and input identities. The
report includes every original patch, available equations/initializers, current
diagnostics and prospective feedback. `PROMPTS.html` marks the changed contract.
