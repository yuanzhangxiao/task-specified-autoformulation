# Does GPT-OSS improve feedback on fitted equations?

The deterministic critic located an error that propagated between outputs. The added
GPT-OSS review respected the broad decision to seek more evidence on all three fresh
parents. Its explanations sometimes went beyond the measurements, including a
misplaced term and a proposal to fit on validation. I would supply the deterministic
measurements directly to the proposer and keep the LLM's additions advisory.

## What was tested

Commit `0aace73` froze the rules on c15-2 and was pushed before the fresh-parent
replays. The primary roster was all three other original parents from the November 3
targeted fitting campaign. They span two equation forms in one T2-easy cell and all
share the same 16 public training schedules. Parents 10 and 14 use identical equations
with different coefficient domains. Their endpoints held the saturation coefficient
fixed at two levels. Parent 12 supplied unrestricted fits from two starts. Previously
studied c15-3 supplied a separate control with two completed meal-memory fits.

The baseline feedback contains the deterministic advice and its underlying packet.
The assisted version adds one hosted GPT-OSS-120B review of that same packet.
Each packet includes the public task, validated equations, measured discrepancies,
numerical availability, equation paths and the fitting decision. The assessment
below is a retrospective reading of the replies against those records. No proposed
revision was executed or selected in this feedback comparison.

## Paired results

| Parent | Deterministic feedback | What GPT-OSS added and how it holds up |
| --- | --- | --- |
| 10 | Neither endpoint cleared independent free-rollout verification. One had solver disagreement. DOP853 reached the work limit on the other. Obtain a verified replay. | Requested another numerical replay. It conflated solver trouble with insufficient experimental data and said persistent output error could disconfirm the delayed-action pathway. The numerical records cannot establish that. |
| 12 | One endpoint cleared replay. On that fit, insulin fell by 84.09 during minutes 100–199 of the 150 g trajectory, compared with an observed fall of 537.42 pmol/L. The second endpoint exhausted the replay allowance. Withhold a replicated equation diagnosis. | Requested a second replay and suggested comparing balances across windows. The numbers support the first-endpoint discrepancy. Attributing it to insulin production or decay and requiring revision of the action pathway goes beyond this evidence. |
| 14 | Both endpoints predict declining disposal on minutes 0–99 of the 150 g meal. The observed change was +9.38, against fitted changes of −10.66 and −9.75 mg/kg/min. Driving the model with the recorded insulin makes disposal RMSE worse at both endpoints. Release the fixed coefficient and continue fitting. | Suggested distinguishing action strength from action dynamics by measuring the hidden action state. That would require a new experiment. It also put an insulin gain directly in the disposal law. The fitted equation for disposal is `U = af_internal_aux_1 + X_insulin_action`; the named gain multiplies insulin in `insulin_action_on_X`, a process entering the derivative of the action state. |
| c15-3, previously studied control | The same window has observed disposal change +9.38, against fitted +3.62. Substituting recorded insulin reduces disposal RMSE from about 2.16 to 1.88 mg/kg/min at both starts. Conditional review of the insulin-to-action-to-disposal path is warranted. | Suggested comparing an action-gain change with a change in delay. Those alternatives are testable on training. The review also proposed fitting on validation, and its disconfirmation criterion asks for +9.38 to fall within a range near +3.62. Both parts are unusable. |

The opposite directions on 14 and c15-3 are informative. In c15-3 the predicted
insulin trace explains part of the disposal error within the fitted model. In 14,
correcting that trace makes disposal prediction worse. In either model a substitution
can affect multiple downstream routes. Its result does not uniquely identify a
physiological fault or the appropriate new equation.

One follow-up suggested by the review on 14 is feasible with existing readings.
The public brief describes total disposal `U` and supplies its insulin-independent
component `Uii`. Their difference can be compared with the modeled
insulin-dependent contribution on aligned training samples. At minutes 0 and 99
of this trajectory `Uii` equals 1 at both endpoints and `U - Uii` rises from
0.916 to 10.300 mg/kg/min. This contrast tests the output decomposition. It
does not measure the latent action state directly.

## What to pass to construction

1. Expose the measured discrepancy, signed balances where available, and the effects
   on every reachable output. Include the independent-start and solver coverage in
   the human-readable message, especially when only one start cleared verification.
2. Let the runtime enforce the stage and the training-data boundary. Record which
   proposed checks are executable on the public training data and which require
   additional observations. Treat the latter as research suggestions.
3. For a proposed equation edit, require an exact equation and term reference, a
   testable alternative and a measurable change in a future free rollout. An
   existing equation ID alone did not catch the incorrect term placement on 14.
4. For another blinded comparison, include independent parents with unrestricted,
   replay-verified fits. None of the three fresh parents qualified for an equation
   revision. Measuring whether critic-assisted edits actually improve held-out
   prediction still requires matched proposals, refits and reserved schedules.

All four complete replies reported the requested model and used recognized
reference IDs. Each fresh review asked for more evidence. The control review
requested an equation investigation. The three fresh reviews used 9,849 reported
tokens in total. The control used 3,854. Repeating the campaign reused the sealed
endpoints and all four cached generations.

[Frozen protocol and reproduction commands](../../docs/PHASE_C_GRAPH_CRITIC.md)
· [Deterministic packets](../../output/graph-critic-paired-v1/feedback/)
· [Paired reviews and usage](../../output/graph-critic-paired-v1/comparison/result.json)
