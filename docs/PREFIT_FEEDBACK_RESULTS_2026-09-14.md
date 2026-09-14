# Matched pre-fitting feedback: ACES results, 2026-09-14

The structured-feedback arm did not improve local repair in this pilot. Ordinary
error text repaired every selected case on its first attempt. Structured feedback
needed more attempts and left one initializer unresolved. The difference was
entirely in initializer assignment notation, not the function cases.

This is a comparison of two renderings of the same current runtime diagnostics.
The error-text arm already receives actionable messages explaining the permitted
correction. It is not the old, less informative constructor or a pre-redesign
baseline. Both arms share the modern validators, normalization and local routing.

## Provenance and scope

- Protocol: `prefit-matched-feedback-1`.
- Code: `ca43a43fdecec6bc99e935eadf7712b0cf6283ef`.
- Plan: `49a755f554ebc2bfd95b8d6820e553aa105fdba3593947d9f3c7fa8f9972d8c8`.
- ACES root: `/scratch/user/u.yx126462/phase_b/prefit-feedback-v1-fix1`.
- Evidence: user-provided completed summary, plus read-only inspection of the
  saved plan, per-episode states and canonical replies through the ACES shell.
- Eight repair cases: five function slots and three initializers. Four valid
  controls, all function slots. Each case has three seeds in each arm.
- No fitting, scientific judge, test data or private reference access.

The preceding replay examined 163 historical local responses: 137 valid before
assignment normalization and 140 afterward, with three recoveries. It retained
70 requests outside its supported scope. Historical construction completion was
11/12, which is not changed retrospectively by this local replay.

## Aggregate comparison

| Measure | Error text | Structured feedback |
| --- | ---: | ---: |
| First-attempt repair validity | 24/24 | 20/24 |
| Final repair validity | 24/24 | 23/24 |
| Physical repair requests | 24 | 29 |
| Repair tokens, provider-reported total | 178,814 | 212,862 |
| Repeated baseline-error evaluations | 0 | 6 |
| Valid controls after review | 12/12 | 12/12 |
| Controls with changed canonical structure | 4/12 | 3/12 |
| Newly observed diagnostic-code evaluations | 0 | 0 |
| Total physical requests including controls | 36 | 41 |
| Total tokens including controls | 284,217 | 321,081 |

Structured feedback used 20.8% more repair calls and 19.0% more repair tokens;
including controls, token use increased by 13.0%. Provider-reported cumulative
latency was approximately 30.33 versus 30.08 seconds including controls. That
measurement excludes allocation and server startup and does not establish a
speed advantage. All requests had reported usage. Paired final outcomes were
23 successful in both arms and one successful only with error text.

## Trace findings

### Initializer notation accounts for every unsuccessful attempt

For case `8f4751cc65f7` (initializer `x`), the saved expression uses the initial
label `x0`. The current normalizer accepts `x` or an unoccupied `x_0` alias but
rejects `x0`. All three error-text seeds changed the label to `x` immediately while
retaining the affine RHS. All three structured-feedback seeds initially repeated
the rejected label. Seed 0 repeated it on all three attempts and exhausted its
budget; seeds 1 and 2 corrected it on their second attempts.

For case `c15bf3fa460c` (initializer `M`), structured-feedback seed 1 similarly
retained the rejected `M0` label once, then corrected it. Its paired error-text
episode corrected the label immediately. These two cases account for all six
repeated-error evaluations and all five additional requests.

All five function repair cases passed on the first attempt in both arms: 15/15
episodes per arm. Initializer first-attempt validity was 9/9 versus 5/9; eventual
validity was 9/9 versus 8/9. The observed failure concerns notation handling, not
evidence that the proposed initializer's scientific RHS is infeasible.

The traces support making harmless initializer notation a runtime responsibility.
They do not establish why the model attended differently to the two payloads.
One hypothesis is that the correction instruction was less salient among the
structured metadata. That remains a hypothesis requiring a separate comparison.

### Changed controls include both renaming and parameter removal

Four changed episodes came from case `032eb80c63d1`: two per arm. They retained
the same reciprocal-time-constant function and parameter role but renamed the
local time constant. At this isolated slot's scope these are parameter renamings;
exact canonical hashes count them as changes nonetheless.

Three episodes made a more substantive edit. Both arms at seed 2 removed the
time-constant divisor in case `0a47f73ac34c`; error text at seed 2 also removed it
in case `0ed17c246418`. The responses dropped the corresponding parameter and
retained only the source variable. They still satisfy the local dependency/sign
checks, but reduce the parameterized model family. Their predictive or scientific
effect has not been evaluated.

Thus the aggregate changed-control counts cannot directly be labeled damage
rates. After manual classification, the counts are two renamings plus two
parameter removals for error text, and two renamings plus one removal for
structured feedback. Zero new deterministic codes is not preservation of the
original scientific proposal.

## Recommended next steps

1. Keep concise, actionable error text as the reference proposer feedback. Retain
   structured diagnostics internally for auditing and routing; this result does
   not justify exposing the entire metadata bundle to the proposer by default.
2. Extend initializer normalization to conventional `state0` notation only when
   its target is unambiguous and collision checks pass. Preserve the RHS and
   continue to reject derivative/topology edits in an initialization call.
3. For deterministic local repair, keep an already valid slot unchanged unless
   a separate scientific concern authorizes revision. In evaluation, distinguish
   local parameter renaming from changes to parameterization or functional form.
4. Re-evaluate any revised feedback on a fresh, frozen case set that includes
   harder function and scientific-content failures. Keep the fitting experiment
   separate and do not silently change this completed campaign's evidence.

There are eight repair cases, not 24 independent scientific tasks. Controls cover
four simple function slots and no initializer controls. The selected cases are a
small historical subset, and the strong error-text performance leaves little
room for improvement on this subset. This result does not establish scientific
quality, whole-model construction gains, or a benefit from redesigned routing.
No new experiment or production-policy change accompanies these result notes.

Subsequent user clarification: equation type belongs to the topology schema and
expression fields should contain only the RHS. The follow-up therefore removes
assignment suggestions from initializer prompts and feedback, retains the existing
normalizer without expanding aliases, and enforces preservation of valid inputs
without proposer calls. See [the updated contract](PREFIT_FEEDBACK_PILOT.md#rhs-only-proposer-contract).
Those changes do not revise any measured result above.
