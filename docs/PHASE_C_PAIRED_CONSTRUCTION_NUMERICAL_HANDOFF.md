# Numerical handoff to Orion: paired construction return

22 October 2026. Please treat this as a continuation of the existing fitting
recovery work. The user agrees that the current parent–child NMSE comparison is
premature: numerical verification is not optimization adequacy, and high scores
cannot yet be attributed to inadequate equations. Codex will focus on construction
checks while Orion qualifies and recovers the numerical endpoints.

## Authoritative saved evidence

Detailed audit:
`output/paired-fit-analysis-20261022/426ed2c33316/HANDOFF.md`.
Per-start receipts:
`output/paired-fit-analysis-20261022/426ed2c33316/fitting_operation_audit.json`.
Original sealed inputs/results:
`output/paired-fit-analysis-20261022/426ed2c33316/returned/`.
Archive SHA-256:
`426ed2c33316d6f13cc98c368161aa8d559886b751604ace4eb706e0901b949e`.

This was the 24-model campaign (eight parents, 16 accepted changed children),
launched under `0bf4747` with the `e7067f3` numerical baseline. All retained
endpoints verified, but all 24 remained `optimization_unresolved`. Search ran
for 21 models / 39 of 48 starts. Orion's `0d393fa` local-derivative and stable
affine recovery changes were **not** enabled. Its separate 14-task/eight-model
recovery is a different roster; please keep their artifacts and counts distinct.
Your subsequent `eec6f22` bounded derivative-recovery commit is also newer than
this returned campaign; no outcome from that recovery is inferred here.

## Numerical issues to integrate into your existing work

1. Both starts of children 07/08/09 never searched because derivative qualification
   included parameters in unused process definitions. Static dependency reachability
   proves `tau_term_6_0_77cba2d645` inactive in all three and
   `tau_term_8_0_aeb84ebe30` inactive in 08/09. Every saved probe for these
   coordinates had zero derivative RMS and zero error RMS, but the residual
   roundoff envelope blocked the all-coordinate qualification. The detailed audit
   covers RHS, output and initialization reachability. Exclude/freeze only proved
   inactive coordinates while preserving an exact lift to the original model;
   do not loosen the general derivative gate.
2. Parent-05's improved searched endpoints failed the fixed prediction-agreement
   threshold even though their NMSE differences were tiny. Two other parents have
   similar discarded endpoints. Reverify through the qualified propagation policy;
   do not accept a low loss in place of trajectory agreement.
3. Child-12 searched and improved markedly but exhausted its allowance with a
   large projected gradient. Its perturbed start did not qualify. More generally,
   the 22 nominally converged searches stopped on `xtol`, not established
   stationarity. The detailed audit retains the different reasons behind the
   common `optimization_unresolved` label.

Please preserve historical receipts and costs and use new explicit numerical
policies/budgets. Validation must not select numerical starts or endpoints. No
new comparison or allocation of error to fitting versus construction should be
claimed until numerical evidence supports it.

## Construction qualifications supplied alongside the numerical work

Children 08/09 also contain an independent equation/data contradiction: their
domains imply `U' = Uii + k*I_mem >= 1` on the fasting training trajectory,
whose U is constant. Better optimization cannot remove that exact contradiction,
although it remains useful to diagnose the fitter on those fixed equations. Do
not silently revise or remove them from the historical matched roster. Codex is
adding construction checks and explicit repair feedback under a new protocol.

Private coefficient combinations also appear in parent-06/child-13:
`(1+gain4)*Uii` and `(1/tau13+gain6/tau6)*U`. The parameters affect only the U
equation. Their exact effective domains are respectively `[1,infinity)` and
`(0,infinity)`. This matters for any future reduction: preserving only the sign
of the first coefficient would enlarge the family. Shared parameters, initializer
maps, constraints and all process consumers must be audited before reduction.

The more complete equation audit is
`output/paired-fit-analysis-20261022/426ed2c33316/scientific-audit.md`; the new
construction milestone is `docs/PHASE_C_FUNCTION_SCIENCE_GATES.md`. These are
saved-equation findings and do not claim a numerical performance ranking. No
fitting, rollout or test-data access was performed for this handoff.


## New construction audit, independent of numerical recovery

The completed offline check now flags nine saved equations under the existing
training error/forcing contract: children 07/08/09/10/14/15 and parents 03/04/07.
All nine bound the first-unit glucose rise in zero-meal `train_011` by
1.952028482, below the recorded 2.271231699. Children 08/09 additionally have the
disposal contradiction. These are exact necessary-condition findings for declared
parameter domains, not an estimate of achievable NMSE. Keep these flags alongside
numerical adequacy; do not discard records from the original matched roster.

The new report is `output/function-science-analysis-20261022/REPORT.html` and its
machine-readable audit is `audit.json` in that directory. Unsupported laws or
initializers explicitly defer. Please use the private-coefficient/consumer audit
before any proposed parameter reduction and preserve an exact domain-aware lift.
