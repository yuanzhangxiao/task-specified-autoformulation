# V7 Dalla Man model inspection: downloaded results through round 15

## Conclusion and campaign coverage

The completed round makes two retained improvements, but supplies no model with
both a good training fit and a convincing mechanism for the intended intervention
figure. The T2 insulin source remains disconnected from meals and glucose.

Source archive: `dalla-v7-review-20260923-082613.tar.gz`.
Verified plan: `ae50b5e87ded15a1e0752e8ebeaeb5ca0e9e53eb4e7435b49eda4aa8b38d0f98`.
Actual ACES root:
`/scratch/group/p.nairr260351.000/u.yx126462/dalla-response-feedback-r14-v1`.

The archive contains 28 JSON files, including 20 with verified content seals.
It contains six imported round-14 results and six completed round-15 results.
All twelve round-16/17 results are absent. Its submission manifest records
global round 16 and `all_rounds_submitted: false`. That establishes submission,
not the current scheduler state or why later results are absent.

The round-16 jobs recorded in this snapshot are proposer 2157229, fit array
2157230, finish 2157231, and next-round dispatcher 2157232. Check those jobs on
ACES before treating this as the final three-round outcome. No remote session
was opened for this inspection.

## Numerical outcomes

All six fitted trials were inspected, including the four that were not retained.
Five structural patches passed the construction checks. One revision failed
three attempts and used the incumbent-refit fallback. Two trials were retained
by the existing validation-based selection rule.

| Task | Parent training / validation | Trial training / validation | Selected outcome |
| --- | ---: | ---: | --- |
| T1-hard seed 0 | 0.232584 / 0.303552 | 0.235505 / 0.349404 | Parent retained |
| T1-hard seed 1 | 0.400073 / 0.490862 | 0.353788 / 0.398207 | Trial retained |
| T2-easy seed 0 | 0.898731 / 0.870820 | 0.894259 / 0.872149 | Parent retained |
| T2-easy seed 1 | 1.004513 / 0.969037 | 0.938570 / 0.925741 | Trial retained |
| T2-hard seed 0 | 0.921255 / 0.977849 | 0.921242 / 0.978190 | Parent retained; fallback refit |
| T2-hard seed 1 | 0.949987 / 0.973726 | 1.282798 / 1.297479 | Parent retained |

T2 aggregate scores average multiple targets. Their improved aggregate scores
must not be described as recovery of the insulin trajectory.

Five trial fits exhausted their fitting budget; the T2-hard seed-1 trial did not.
Native optimizer convergence is not established by these results. Better fitting
could change numerical quality, but parameter changes cannot introduce a missing
causal driver into the insulin equation.

## Fitted equations and trajectory interpretation

### T1-hard seed 1: useful meal term, still poor timing

The retained revision adds

```text
dM/dt = 0.1 * meal_event_g - M / 30
additional contribution to dGp/dt = 0.1 * M
M(0) = 0
```

This is a positive meal-memory pathway with a 30-minute timescale. Validation
NMSE falls by about 18.9%. However, the fitted direct meal coefficient remains
about 1.92074. Rollouts still respond abruptly during the meal pulse instead of
reproducing the delayed glucose rise. Large, slowly varying latent components
also remain. Training NMSE 0.354 is not the desired accurate training match.
The state called I in this T1 model is latent, not the observed insulin target.

T1-hard seed 0 also proposes a meal-memory state, but its fitted denominator is
2.06e-19. Production's 1e-12 denominator floor makes that pathway effectively
instantaneous at the plotted timescale. It does not provide the intended delay,
and its trial is correctly not retained by the validation rule.

### T2-easy seed 1: an improved disposal curve without recovered insulin action

The retained fitted equations include approximately

```text
dI/dt = 3.82104 * insulin_pmol_per_kg_min + I / 232.132
dX/dt = 4.94e-324 * I - X / 113.734
dY/dt = 3.82104 * insulin_pmol_per_kg_min - Y / 90.4591
U = Uii + 622.261 * X + 0.152315 * Y
X(0) = 0.0623713 + 1.04339e-6 * I(0)
Y(0) = -256.333
```

Disposal U improves from training/validation NMSE 0.991136/0.992119 to
0.808263/0.863591. Insulin I barely changes: 1.326492/1.247285 becomes
1.326508/1.246276. The current numerical I-to-X coupling is negligible.

With zero external insulin infusion, the added Y state simply relaxes from a
negative fitted initial value. The disposal contribution is approximately a
difference of exponentials:

```text
U(t) - Uii(t)
  = 622.261 * X(0) * exp(-t / 113.734)
    - 39.04 * exp(-t / 90.4591)
```

This expression neglects only the numerically negligible I-to-X forcing. It
produces a broad hump anchored to the initial time, rather than a response to
meal timing. In the plotted validation schedule, predicted U starts rising
before the meal at minute 30. The fitted hump improves some scores but does not
establish the desired intervention mechanism. The proposer describes a
"saturable" process, but the added equation is linear and has no saturation.

This insensitivity is visible directly in the replay: between training_002
(60 g meal at minute 0) and validation_000 (75 g meal at minute 30), predicted U
differs by at most 4.8e-7 and peaks at minute 104 in both. Reference U peaks at
minute 111 versus 137, with amplitudes 4.91 versus 5.86. These schedules change
both timing and amount, so this is not an isolated estimate of either effect.

T2-easy seed 0 similarly adds an infusion-driven state to U, without revising I.
Its original reversed infusion and insulin-to-disposal effects remain in the
fitted trial, and its validation score worsens slightly.

### All T2 revisions leave the missing insulin source unresolved

None proposes a change to the I equation. All four fitted T2 trials still have
`dI/dt = a * insulin_pmol_per_kg_min + b * I`, including the fallback refit.
For a fixed initial I and zero infusion, changing a meal schedule cannot change
predicted I. Both T2-hard retained models therefore remain the old, nearly
constant-insulin models.

The T2-hard seed-1 trial adds another insulin-action state downstream of I, but
its training and validation scores both worsen substantially. Adding downstream
memory does not repair a missing source for the observed insulin target.

## Feedback delivery and repeated rejection

The completed round records eight physical proposer requests and 56,941 observed
tokens, with no unknown-usage calls or delivery failures. Reported prompt input
counts range from 4,892 to 7,999. Compact response feedback addresses the earlier
oversized-request problem for this round; it does not yet establish better
mechanistic revision quality or performance across the unfinished campaign.

The imported T2-easy seed-1 response preview already exposes the insulin error:
on training_005, observed I peaks at 582.53 pmol/L at minute 90, whereas the
prediction has no interior peak and rises to 92.70 at minute 300. The chosen
patch cites R004, the insulin example, but modifies disposal instead of I.
Verifiable reference identifiers do not verify the proposer's causal explanation.

T2-hard seed 0 repeats the same rejected patch three times. It puts
`par_001 * A - Uii - J` in the Gp output mapping, where the parent uses `Gp`.
That expression is a rate contribution, not the glucose state. Under the runtime's
initialization rules, the remapped Gp state then needs a latent initializer; the
reported error is `new latent states require explicit initializer definitions:
['Gp']`. The message obscures the output-mapping problem. A more actionable
correction would point out the changed Gp mapping and ask the proposer to preserve
`Gp -> Gp` when only changing downstream action dynamics. The incumbent was
preserved after rejection and after the unsuccessful fallback fit.

For a later feedback improvement, use the public training evidence to call out
the disconnected response explicitly: the insulin target changes after meals
without external infusion, while the candidate's insulin equation has no route
for that response. Ask for an evidence-supported revision of that target pathway,
without prescribing the private reference equations. This is a suggested next
step, not an implemented change or a reason to alter the running protocol.

## Verification and artifacts

All 12 distinct imported/new request-and-parameter records compile and reproduce
their saved lowered-candidate identity and parameter names. The six new trial
records were replayed across all 16 training and four validation trajectories:
120 complete fixed-parameter rollouts. All per-target aggregate scores match the
saved values within 1.9e-15. Checkpoint resume performs zero new integrations.
This is a production-interpreter replay, not independent-solver certification.

No new fitting, live LLM call, test-data access, or benchmark change was performed.
No new intervention was designed. The plot compares the same illustrative public
training_002 and validation_000 schedules used in the preceding audit. Both
retained improvements are shown; all six trial models and all target trajectories
are exported, including the unsuccessful trials.

Artifacts: `artifacts/dalla-v7-model-review-2026-09-23/`, including
`verified-models.json`, `fitted-equations.md`, `replay-summary.json`,
`trial-curves.csv`, and `retained_improvements.png` / `.svg`.

Relevant collector, multi-target profile, and trajectory replay tests: 28 passed.
Repository-wide Ruff still reports 37 existing issues in unrelated untracked
`analysis/claude/` files. Those files were left unchanged. The only tracked change
is this report; generated model records and plots remain local artifacts.
