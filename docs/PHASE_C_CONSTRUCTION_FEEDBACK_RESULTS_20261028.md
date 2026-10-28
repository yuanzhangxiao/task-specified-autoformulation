# Returned construction feedback and 120B results — 28 October 2026

Both returned jobs completed without operational or outer-provider-schema
failures. Independent audits passed 237 checks for 20B and 163 for 120B.
There was no fitting, model adoption, new generation during analysis, or test
data access. All runs use one frozen T2-easy parent, P00-r1, and three seeds.

| Run | Eligible initial records | Eligible follow-ups | Repeated failed follow-ups | Eligible episodes | Calls | Episode tokens: median [unscaled MAD] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier 20B focused | 2/15 | 1/12 | 9/12 | 2/3 | 15 | 87,731 [1,841] |
| New 20B action feedback | 2/15 | 1/12 | 0/12 | 2/3 | 15 | 93,095 [3,040] |
| 120B focused | 4/15 | 2/6 | 2/6 | 2/3 | 9 | 70,535 [406] |

These are finite-gate eligibility counts, not scientific approvals. The earlier
20B and returned 120B share the same runtime, prompts, source, gates and budget;
their initial request bodies differ only in model. The new 20B uses different
follow-up feedback. Its three initial request bodies, including sampling seeds,
are identical to the earlier 20B run, but all fifteen initial raw patches differ.
The repetition change therefore does not isolate the causal effect of feedback.

## What the new 20B feedback achieved

Four of six follow-ups receiving typed action choices clear interface validation
and expose qualified mathematical conflicts. The other two introduce new
parameter/process name collisions. Explicit cleanup is used: five deltas contain
six drops, and one whole-patch replacement drops two earlier actions. These eight
reported omissions are intentional edits, not demonstrated runtime loss.

Three eligible records represent only two complete model identities. One adds a
second insulin memory; the other adds a meal reservoir. The eligible follow-up
reproduces the reservoir already proposed by a sibling. Thus twelve follow-ups
add no new distinct eligible family. The previous 20B run had three identities.
Reduced repeated-failure flags partly reflect transitions to different interface
failures, rather than demonstrated scientific progress.

With G=Gp, m=meal rate and U=disposal, the memory-only variant retains
`G' = C_m*m - U`. The previously archived zero-meal interval with increasing G
therefore requires a negative integral of U to match exactly. This is inconsistent
with an everywhere-nonnegative disposal interpretation, but no interior target
interpolation or positive-invariance proof is added here. It is not an NMSE bound.

The reservoir variant has `Z' = m-k*Z` and
`G' = k*Z-U+(C_m-1)*m`. It can supply glucose after m becomes zero, while still
retaining parallel instantaneous meal terms. Its free trajectory initial Z(0)
is not evidence of a real physiological state. Both eligible families are affine;
linearity itself is not a reason to reject them.

## What 120B adds, and why it stopped at nine calls

Six eligible records have six distinct complete-model/domain identities, but
fewer broad scientific mechanisms: meal reservoirs, a two-stage insulin-memory
chain, saturating glucose-dependent insulin production, and a clearance feedback
signal. Four add free latent initial roots; two nonlinear production laws make
the affine-family checks defer. All six remain scientifically unestablished and
unfitted. This small single-parent result does not establish 120B superiority.

The two saturation records differ in parameter domains. Allowing K=0 in
`V*G/(K+G)` leaves a possible 0/0 at G=0; strictly positive K avoids that point for
nonnegative G, but whole-system positivity and stability remain unproved.
One claimed clearance compartment uses `I' = parent_I'-a*Z` and
`Z' = b*I-c*Z`: these are different removal/entry fluxes if interpreted as
transferred insulin. This is a physical-interpretation concern, not an observed
negative rollout or a new blanket rejection rule.

Seeds 0 and 2 each repair their three failed initial branches and finish after
four calls. Seed 1 returns five `requires_wider_revision` patches with actions.
The provider schema permits that decision, but the initial-option parser requires
`revise`; all five wrappers are rejected, their candidate payloads become null,
and routing offers no follow-up. Four call slots remain unused. Raw responses
are retained in the provider cache. The system's wider-revision instruction also
requires no actions, which these responses violate. This is a schema/recovery
alignment gap, not evidence that the five proposals are otherwise valid.

A clearly labeled offline counterfactual changes only that decision and uses
the frozen f8c9e23 assessor. All five still reject: one unused unattached gain,
two existing-state redeclarations, and two edit/removal conflicts. No live
response is changed and no eligible model is invented by this exercise.

The proposer also makes incorrect explanatory claims: an existing first-order
insulin state already stores history, and instantaneous dependence on a current
input is causal. Added delay states may be useful choices, not consequences of
those claims. Scientific interpretation must follow the actual equations.

## Recommended next milestone

1. Align the initial-option decision schema, instructions and recovery routing;
   preserve rejected payloads and clarify ambiguous scope within the remaining
   fixed allowance. Never silently convert unsupported changes into acceptance.
2. Extend explicit declaration-movement feedback to existing components wrongly
   listed as new. Normalize identical duplicate action entries only when
   unambiguous; conflicting duplicates remain errors. Show valid initializer
   alias usage in local feedback.
3. Show whether newly added components influence the affected target, including
   initializer dependencies. Disconnected additions cannot repair an unchanged
   target subsystem; this should be explained alongside its actual witness.
4. Replay the saved failures offline before another live test. Retain eligible
   candidates for equation review and later qualified fitting; do not select
   models solely because nonlinear terms or free initials evade a bounded check.

This analysis changes no production code, prompt, acceptance gate or budget.
Neither returned run demonstrates a newly used common-zero specialization
certificate; enabling that policy is distinct from exercising its new proof path.

## Evidence

- New 20B archive: `dfaaa0056d86c8c8ee4762968bbb254bcfaee77cdebd045c75d29c026c2e71de`;
  source `82424b5`, job 2262249, profile `action-feedback`.
- Returned 120B archive: `40c84c9002c58d7446b783ecd666a78323cdfa81f70c0a6a60e6af4d4397f4c3`;
  source `f8c9e23`, job 2260281, profile `focused-retest`.
- Previous 20B archive: `22bff021a93ba3f518eae383cbdb8cb58b34d0d9fd622d0618be56741e503396`.
- Local report: `output/construction-feedback-results-20261028/REPORT.html`.
  It includes plots, all 48 new candidate records, all 24 exact requests/replies,
  expanded equations, initial rules, independent audits, counterfactual labels,
  reconstruction scripts and archive hashes. Generated evidence is not committed.
