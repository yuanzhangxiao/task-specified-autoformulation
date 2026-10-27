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
