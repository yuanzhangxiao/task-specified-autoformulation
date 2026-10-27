# Common-input proof and explicit branch repair retest

26 October 2026. This milestone addresses the v3 five-option smoke's missed
zero-input contradiction, lost branch declarations and ambiguous citation-only
feedback. The experiment remains construction-only: no fitting, candidate
adoption, benchmark changes or test-data access.

## Qualified common-input proof

The opt-in `certified-common-zero-forcing-affine-1` check specializes the candidate
only on a witness interval whose authenticated complete input schedules certify
exact zero. Restricted expression parsing and parameter domains must establish
that the specialization is defined; `0 / unknown_denominator` is not silently
simplified. Original equations, restricted equations, forcing schedules, interval
and domain assumptions are recorded. Generated memory states are not assumed zero.
When several target checks use the same certificate and equations, the prompt
displays that context once with explicit references from each target's witness.
This removes duplicated payload without removing evidence or changing a check.

For example, `V*m/(K+m)` with `K>0` becomes zero on a certified zero-meal interval.
It therefore cannot invalidate a state-affine trajectory identity on that
interval. Shared affine dynamics preserve qualified initial weighted relations;
recorded targets can contradict those relations without numerical fitting.
This concerns exact reproduction under the authenticated noiseless contract,
not a lower bound on achievable approximate NMSE.

The new check is deliberately bounded. Unsupported nonlinear state functions,
unknown or singular domains, uncertain/incomplete schedules and exhausted proof
bounds defer. It is not a general theorem prover for arbitrary common nonzero
forcing. Previously established checks remain available; specialization never
changes the model, training data or historical outcome.

## Explicit edits to an unaccepted branch

Initial options remain complete sets of changes to the frozen parent, not full
restated models. In a selected-branch follow-up, `BranchRepairReply` pins the
pending candidate ID and exact patch hash and chooses one of three modes:

- `delta`: retain omitted pending actions exactly, including earlier parameter
  declarations. Supplied entries replace whole entries with the same action key.
  A changed equation supplies its complete RHS. Empty lists do not clear actions.
- `replace`: discard the previous pending patch and supply a complete replacement
  relative to the original parent.
- `withdraw`: stop pursuing the branch, without accepting the parent or a model.

`drop_actions` explicitly removes a pending intention in delta mode. For example,
`{"field":"new_processes","key":"unused_link"}` drops that proposed addition;
it does not delete an existing parent process. Parent removal still requires the
existing explicit removal action. Conflicting/duplicate keys, stale or sibling
references, unsupported fields and size violations fail closed. Nothing is inferred
from a parameter's name or scientific description.

Composition records retained, replaced and dropped actions and the complete
resulting patch. The composed patch then undergoes normalization, static checks,
public-path checks and qualified candidate-specific data checks. Sibling options
never merge; eligibility does not adopt a model or certify scientific correctness.
An invalid composition leaves the prior pending patch available for correction.
An unreadable/provider-error follow-up likewise preserves the pending patch;
neither transport failure nor a failed envelope becomes a new scientific edit.

History shows literal attempted RHSs, initial rules, parameter roles and removals,
without repeating descriptive prose or source lists. Repetition is assessed after
composition and validation. Citation-only failures now say that the trial is
scientifically unassessed: correcting its evidence label can preserve every
scientific declaration. Parent conflicts are labeled as parent evidence, not as
proof that an unassessed trial still has them.

Historical system text and reply semantics remain the default. The new profile
explicitly pins both opt-in policies, system text and branch-response schema.
Saved historical replies remain full parent-relative patches during replay; they
are not silently interpreted as deltas. Prospective composition demonstrations
are reported separately from raw replay and live outcomes.

## Targeted ACES roster

`focused-retest` selects P00-r1, the first sorted frozen parent in the authenticated
source, with seeds 0, 1 and 2 on each model size. This is one related T2-easy model,
not a benchmark-wide comparison or a test of five versus three options.

| Setting | Per model arm |
| --- | ---: |
| Episodes | 3 |
| Initial alternatives per episode | Exactly 5 |
| Maximum distinct shallow repairs per episode | 4 |
| Maximum calls | 15 |
| Output / context tokens per call | 8,192 / 32,768 |
| Maximum charged tokens per episode | 163,840 |
| 20B / 120B GPU request | 1 H100 / 2 H100 |

Both arms together allow at most 30 generation calls. Complete usage and any
unmeasured requests remain reported. Server-side token preflight is authoritative;
no evidence truncation or budget renewal occurs. This focused profile can launch
its three-episode roster directly. Its optional one-episode smoke is included in
the same budget. Historical profiles retain their separate smoke gate.

Fresh result roots:

```
/scratch/group/p.nairr260351.000/u.yx126462/phase-c-search-focused-retest-20b-v4
/scratch/group/p.nairr260351.000/u.yx126462/phase-c-search-focused-retest-120b-v4
```

After uploading and extracting the delivered bundle in group scratch, set
`AF_CODE` to its exact absolute directory and run:

```bash
bash "$AF_CODE/run_search.sh" run 20b full wave1 focused-retest
bash "$AF_CODE/run_search.sh" run 120b full wave1 focused-retest
```

Here `full` means the three-episode focused roster, not the older 18-episode
campaign. No separate fitting or follow-up job is submitted. Status and packaging:

```bash
bash "$AF_CODE/run_search.sh" status 20b full wave1 focused-retest
bash "$AF_CODE/run_search.sh" status 120b full wave1 focused-retest
bash "$AF_CODE/run_search.sh" inspect 20b full wave1 focused-retest
bash "$AF_CODE/run_search.sh" inspect 120b full wave1 focused-retest
```

Each result root receives `inspection.tar.gz`. A confirmed interrupted allocation
can resume with an explicit new wave such as `wave2`, retaining the same root,
plan, cache and charges. Reconcile ambiguous scheduler replies before resubmission.
Do not change a profile or repair an incomplete extraction in place.

## Offline evidence and limitations

The v3 smoke's 17 saved patches are rechecked independently with the common-input
flag. The corrected proof blocks one formerly eligible 120B candidate (2 to 1
eligible among its eight patches); the other 16 outcomes are unchanged, including
zero eligible among the nine 20B patches. A separate diagnostic uses explicit
prospective delta envelopes to show
citation-only correction and preservation of the two omitted 120B parameter
records. Those authored demonstrations are not new model responses or fitting
results. Source receipts, exact patches, deferred checks and old costs are retained.
Both composed examples pass the current gates with scientific checks still deferred.

CPU Harmony token sizing exactly reproduced nine saved server counts and admitted
all 21 current/stress packets. The largest used 21,086 input tokens, below the
24,000-token preflight limit. This calibration does not replace the live server's
token check or guarantee that arbitrary future histories will fit.

The previous 120B internal-exchange sign concern still needs scientific review.
This milestone does not infer transfer intent from names, introduce a generic
same-sign prohibition or automatically flip signs. Passing the finite checks
remains insufficient evidence of scientific validity or favorable NMSE.

## Returned 20B results, 27 October 2026

The 20B archive SHA-256 is
`22bff021a93ba3f518eae383cbdb8cb58b34d0d9fd622d0618be56741e503396`.
It matches source commit `f8c9e23` and the delivered focused plan. All three seeds
completed; all 15 responses returned HTTP 200, `stop`, and schema-valid JSON.
An independent 194-check audit verified the source, plans, receipts and costs.
Observed usage was 267,532 tokens; episode median 87,731, unscaled MAD 1,841.
Maximum input was 20,568 tokens, below the 24,000-token admission limit.
The user reports 120B still pending; no model-size conclusion is available.

| Seed | Initial eligible / 5 | Follow-up eligible / 4 | Repeated failures / 4 |
| --- | ---: | ---: | ---: |
| 0 | 1 | 0 | 4 |
| 1 | 1 | 1 | 2 |
| 2 | 0 | 0 | 3 |

There are three eligible records among 27, spanning two episodes. Every eligible
record retains deferred scientific checks. No fitting, rollout, adoption or
test-data access occurred. The three saved fitting-interface packets are not
numerical results.

All ten delta follow-ups composed as specified. One replacement referred to the
wrong branch and digest and was correctly rejected; one branch was withdrawn.
Seven follow-ups repeated the same mathematical actions. Two more attempted to
remove invalid parameter-as-equation entries by omission, without `drop_actions`;
delta retention therefore correctly preserved them and their errors. Parameters
were generally restated, so omitted-parameter retention was not exercised live.
The generic feedback still contains a full-patch instruction in one fallback and
does not show the exact deletion needed. Separate authored in-memory corrections
of the two missing drops advance to qualified affine conflicts, not eligibility.

The one successful follow-up declares the previously missing `k_absorption`
parameter while preserving its intended absorption equation. It is legitimate
interface progress without a new scientific equation. Unknown-process errors
and parameter-as-equation errors need different action-specific feedback.

The three eligible constructions are:

- Seed 0 option 4: `A' = k_d*(D-A), D' = k_A`, with a new trajectory-specific
  initial `D(0)`. The family remains affine but its initial-state space expands.
- Seed 1 option 3: saturating insulin-action decay `k*X/(K_M+X)`; the affine
  proof becomes inapplicable. Positivity and pole avoidance remain unverified.
- Seed 1 repaired option 2: `M' = k_abs*(meal-M)` and an added `+M` in the
  glucose balance, with trajectory-specific `M(0)`. This supplies a delayed
  absorption contribution; numerical and scientific adequacy remain untested.

The first two still have `Gp' = C_m*meal-U`. On `train_011`, the authenticated
complete meal schedule is zero on `[0,1]`, while glucose increases from
155.325006223101 to 157.596237922436. Exact reproduction would require
`integral_0^1 U(t) dt = -2.271231699335`. They must either miss this glucose
increase or predict negative disposal between samples. This is a scientific
concern under the nonnegative-disposal interpretation. It is not a newly
implemented finite-sample rejection: target interpolation is not authenticated,
and signed coefficients do not establish positivity of the predicted rate.
The absorption candidate does not imply the same identity.
This concern also occurs after the eight-unit observation prefix: on `[9,10]`
of the same trajectory, zero meal accompanies glucose increasing from
166.7836252405335 to 167.35174265652742. The unchanged balance would require
`integral_9^10 U(t) dt = -0.568117415994`, with the same qualification. Neither
witness is a bound on achievable approximate NMSE.

The common-zero specialization flag was enabled but no returned candidate
exercised that new proof branch. The eight qualified rejections used existing
affine-family witnesses. Historical saturation replay remains the direct check
of the specialization fix. Public-path passes also remain limited: the stored
insulin-memory witness is unnecessarily cyclic although a valid simple path
exists, and an initialization-only modeled auxiliary does not certify agreement
with that auxiliary's full supplied trajectory.

The local report in `output/construction-search-v4-20b-analysis-20261027/REPORT.html`
contains all 27 candidate pages, all 15 complete requests/responses, equations,
branch and operational audits, and an outcome plot. Keep the pending 120B run
under the frozen contract. Proposed next corrections are specific edit/delete
feedback, removal of the remaining full-patch fallback wording, and a qualified
balance/sign audit before any additional rejection rule. No production policy
was changed during this analysis.
