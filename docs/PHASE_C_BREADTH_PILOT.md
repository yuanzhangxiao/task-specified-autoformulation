# Five-option construction pilot and combined initialization checks

26 October 2026. This milestone extends construction search to five mandatory
alternatives and closes the specific initializer/readout proof gap found in the
v2 smoke. It preserves saved experiments and does not launch fitting or open
held-out data. The purpose is to find useful construction strategies; the new
budget is explicitly larger than the earlier three-call budget.

## Why the checker changes

Both v2 smoke arms completed operationally, and both returned exactly three
initial options. The finite gates recorded one eligible 20B candidate and six
120B candidates. All retained deferred checks. An independent equation audit
identified a missed exact-reproduction conflict in the 120B options episode:

```
B' = 0; B(0) = beta * I(0)
U = A + X + B; A(0) = recorded Uii(0)
```

The trajectory-specific unknown X(0) is constrained by the observed output:
`X(0) = recorded U(0) - recorded Uii(0) - beta * recorded I(0)`.
Together with the shared affine dynamics, this preserves a weighted initial
relation that the training trajectories violate. Treating the initializer map
and readout separately missed this implication. This proof concerns simultaneous
exact reproduction under the authenticated noiseless contract. It does not
bound achievable approximate NMSE.

`combined-symbolic-map-readout-initials-1` expands the actual lowered initializer
maps, retains explicitly distinct trajectory unknowns, and eliminates those
unknowns only through exact observed initial readouts with rational coefficients.
It supports dependent initializer chains and simultaneous readout constraints.
Only resulting initial expressions free of trajectory unknowns can participate
in shared-coordinate cancellation. A genuinely free split, such as free X(0)
and B(0) with only their sum observed, stays unresolved.

Unknown-parameter pivots, products of unknown coordinates, uncertain observations,
unsupported expressions and exhausted proof bounds defer. The new calculation
preserves already established initializer-map evidence when an additional
readout is unsupported. Historical/default gates are unchanged unless the new
flag is explicitly enabled. V3 requests, child checks, citation normalization,
plans and cache identities consistently enable it.

## Prospective breadth profile

The default `breadth-five` profile uses the same six saved T2-easy parents and
three seed bundles on each proposer size. These are six related models, not six
benchmarks. The first request must return exactly five independent repairs of
the same parent. Each names its affected target and changed scientific premise.
Cosmetic renaming is not a new model family.

After the first batch, up to four distinct failed families receive one repair
each. Eligible alternatives are retained independently. Invalid siblings do not
invalidate valid ones when the envelope has the required count. An invalid
envelope does not receive an uncharged retry or invented replacement options.
No candidate is adopted as a new parent or merged with another alternative.

| Setting | Per model arm |
| --- | ---: |
| Saved parents × seed bundles | 6 × 3 |
| Episodes | 18 |
| Mandatory initial options per episode | 5 |
| Maximum calls per episode | 5 |
| Maximum candidate patches per episode | 9 |
| Maximum generation calls | 90 |
| Maximum charged tokens per episode | 163,840 |
| Context / output allowance per call | 32,768 / 8,192 |
| Temperature / reasoning | 0.2 / low |
| Included smoke subset | First sorted parent, seed 0 |
| Smoke episodes / maximum calls | 1 / 5 |

Both model arms therefore have at most 180 calls. Smoke work is part of that
allowance. The five initial options share one 8,192-token output allowance. A calibrated
CPU preflight found that reserving 12,288 output tokens would leave too little
room for some late repair packets; the larger episode budget comes from five
calls, not a larger per-call completion. The server
counts each actual prompt before generation; context or budget exhaustion is
recorded explicitly without truncating evidence or renewing the budget.

20B uses one H100 and 120B uses two H100s, with the existing pinned model revisions
and serving image. The four-hour allocation and 12,600-second worker window
remain bounded scheduling allowances. There is no automatic continuation.

The corrected `three-policy` profile remains explicitly selectable, with the
earlier three-call/three-option allowances. It is not automatically launched.
Comparison with historical three-option results must account for the changed
gate, larger budget, and altered feedback; it is not an isolated causal estimate
of option count. Reports retain all costs, deferred checks and candidate equations.

## Replies contain changes, not complete restated models

The existing edit contract is unchanged. The proposer supplies each changed
equation's full replacement RHS and any new scientific declarations it requires.
Untouched definitions are retained. The runtime derives source lists and accepts
unambiguous serialization normalization, but does not invent an equation,
initializer, parameter-sharing choice or mechanistic dependence. Since rejected
patches were never installed, each retry includes all changes still intended
relative to the frozen parent.

## Reproducible offline replay

Recheck the v2 smoke's 20 saved patches without generation:

```bash
.venv/bin/python -m scripts.replay_search_policy_results \
  --source-version v2-smoke \
  --source-20b output/construction-search-v2-smoke-20261026/input/20b \
  --source-120b output/construction-search-v2-smoke-20261026/input/120b \
  --root output/construction-search-v3-smoke-replay-20261026
```

The same tool with its default `--source-version v1` rechecks all 289 patches in
the pinned older full experiment. Source manifests, parent controls, original
requests/replies, tokenizations and completion receipts are checked before reuse.
V2's unstarted full-roster cases are preserved as unstarted, not inferred failures.
Rechecks are independent patches against their original parents; later historical
replies did not receive the new feedback. No historical output is overwritten.

Completed offline verification rechecked 309 saved patches. In the v2 smoke,
20B remains 1/11 eligible and 120B changes from 6/9 to 5/9: the combined proof
catches the specific independently audited contradiction, with the other 19
outcomes unchanged. In the older 289-patch experiment, eligibility remains
12/138 for 20B and 10/151 for 120B under the corrected gate, unchanged from the
previous corrected replay. These are candidate counts, not independent benchmark
success rates; unresolved scientific checks remain visible.

The CPU token preflight reproduces all 16 saved request counts and measures six
new initial packets plus 25 stress repair packets. The largest is 22,599 input
tokens, below the 24,000-token allowance after reserving 8,192 output and 512
safety tokens. Stress packets include eight prior candidate records and the newly
blocked 120B case. They are estimates for constructed histories, not a guarantee
about future live replies; the server's per-request measurement remains required.

## ACES commands

Upload the delivered bundle to group scratch, extract it, and set `AF_CODE` to
that exact absolute directory. Its wrapper checks the bundle manifest before
any launcher action. Run the two smoke commands:

```bash
bash "$AF_CODE/run_search.sh" run 20b smoke
bash "$AF_CODE/run_search.sh" run 120b smoke
```

The fresh result roots are
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-search-breadth-five-20b-v3`
and the corresponding `120b-v3` directory. Old source/result directories are
not reused. After the jobs finish:

```bash
bash "$AF_CODE/run_search.sh" inspect 20b
bash "$AF_CODE/run_search.sh" inspect 120b
```

Each creates `inspection.tar.gz` in its result root. Inspect smoke results before
the separate full commands; these reuse the smoke calls and their charges:

```bash
bash "$AF_CODE/run_search.sh" run 20b full
bash "$AF_CODE/run_search.sh" run 120b full
```

`status SIZE` shows scheduler state and saved summaries; `preview SIZE` regenerates
the exact prompt HTML. A confirmed interrupted allocation can be continued with
an explicit new wave, e.g. `run 20b full wave2`, retaining the same immutable plan.
Ambiguous scheduler replies must be reconciled, not bypassed by renaming a wave.

The optional corrected reference profile requires an explicit fifth argument,
e.g. `run 20b smoke wave1 three-policy`, and has its own root. Subsequent actions
must use the same explicit profile. No reference sweep is part of the default
five-option pilot.
