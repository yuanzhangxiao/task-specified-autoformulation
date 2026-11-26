# Choosing T2 functions on events where forecasts disagree

The experiment compares two ways to choose an equation from training trajectories. One uses all available selection events. The other first finds the half on which the candidate models predict the most different trajectories, then chooses the model with lower error on that half. A fixed affine model and a uniformly random half of the events provide references. The comparison is conditional on four source–equation pairs identified in the previous [partial-observation audit](PHASE_C_PARTIAL_OBSERVATION_SHAPES.md).

## What changed between models

One contribution to a differential equation used either \(\beta (x-\mu)/s\) or \(\beta (x^2-\mu^2)/(2\mu s)\), alongside an adjustable intercept. Center \(\mu\) and scale \(s\) came from the fitting events. Both forms have the same value and derivative at \(\mu\) for a shared coefficient. The square form constrains the shape to an offset plus a square. A general quadratic would have another independent coefficient. All other equation expressions, hidden initializers and parameter counts were the same within each comparison; each candidate could fit all of its coefficients.

The four comparisons were EGP in \(\dot U\) on graph00, EGP in \(\dot I\) on graph04 and graph05, and \(I\) in \(\dot U\) on graph05. Each form used two optimization starts with a limit of 500 evaluations and 90 seconds each. All 32 starts used 500 evaluations. The better feasible vector on its eight fitting events was retained. The scoring simulation propagated all generated states from their allowed initials; measured future targets were used only to calculate errors. Sampled negative \(G_p,I,U\), or a failed numerical replay, disqualifies a forecast.

## Choosing an equation and measuring NMSE

Eight official training events fit the parameters. Of the remaining eight, four selected the form and four scored the frozen choice. Exchanging selection and scoring events and repeating with the complementary fit produced 16 comparisons from the same 16 trajectories. Repeated use of trajectories and graphs limits the number of independent comparisons.

1. On the four selection events, compare the candidates' predictions for all three targets after time zero. The gap is the mean squared difference after division by each target's fitting-set standard deviation.
2. If both candidates have valid forecasts on the four events, select the lower mean NMSE on the two events with the largest gap. Compare with choosing on all four, and with the exact expectation across all six possible pairs. When only one candidate qualifies, use it. When neither qualifies, record an abstention.
3. Freeze the choices. On each held-out set, score the same post-initial times with equal weight per target, event and comparison. Report common-set scores and each policy's coverage. The primary numerical comparison requires every held-out event to yield a valid prediction for every policy. The random comparator averages model losses over choices; it does not average the models' predictions.

On **omitted training events**, all policies could be compared in 12 of 16 rotations. Mean NMSE was 0.948793 for fixed affine, 0.935442 for the highest-disagreement pair, 0.936590 for a random pair and 0.935015 using all four events. The largest change was graph00, where replacing EGP by its squared form in \(\dot U\) lowered NMSE from 0.7383 to 0.6547 across its two comparable rotations. Excluding graph00, the highest-disagreement and fixed-affine policies averaged 0.9916 and 0.9909. Seven of 128 individual forecasts had negative insulin. Three distinct events accounted for all seven. Every optimization start reached its evaluation cap, so these are comparisons of attained fits.

## Independent replay on the official validation split

The four official validation trajectories entered only after the fitted vectors and all sixteen training-based choices had been frozen. Each fitted candidate generated its own validation trajectories from the public initializers and allowed forcing. Each policy was evaluated on the same four events for each comparable rotation. The test split stayed sealed.

| Training-based choice | Mean official validation NMSE on 13 shared rotations | Rotations with four eligible validation forecasts, out of 16 |
|---|---:|---:|
| Fixed affine | 0.919741 | 14 |
| Two events with the largest prediction differences | 0.896979 | 13 |
| Two uniformly random events, exact expectation | 0.892821 | 13 |
| All four selection events | **0.885796** | 13 |

Choosing from the highest-disagreement events lowered mean NMSE by 2.47% relative to fixed affine. Across the 13 comparable rotations it beat all-event selection zero times, tied nine and lost four. All-event selection reduced NMSE by 3.69% relative to fixed affine. Each adaptive approach abstained on three rotations because neither candidate qualified on the four training selection events. On validation, three of the 64 distinct candidate–event forecasts generated negative insulin, all on val_003. These coverage counts accompany the common-set NMSE so that abstentions remain visible. The same four validation events and some fitted models recur across rotations.

Use all available selection events when their outcomes are already available. Disagreement can highlight which trajectory is worth inspecting. This experiment gives no empirical advantage for discarding the other selection outcomes and measures only two equally sized families.

## What the observations identify in T2

Under \(U=U_{ii}+D\), effective insulin-dependent disposal \(D\) can be reconstructed at all 4,816 time points in the 16 training events. At the same initial insulin concentration (25.58781 pmol/L), events train_012 and train_011 give \(D(0)=0.83507\) and \(0.98904\) mg/kg/min. Any common memoryless map \(D=h(I)\) must miss at least one recorded initial value by 0.07698 mg/kg/min. This follows directly from half the difference, without estimating any derivatives or model parameters.

This example identifies a training signal for a latent-dependent contribution and rules out one set of arguments for it. A model can still explain that signal through different latent action states, initializations, or dependencies. In particular, graph02 uses a dynamic insulin-action state with an initializer that also receives EGP. The repeated insulin value alone imposes no equality on those two initial hidden states. A deployable candidate has to generate \(U\); feeding it its own measured future values would invalidate the forecast. [Conditions for identifying individual latent equations](PHASE_C_LATENT_FUNCTION_IDENTIFICATION.md)

## Reproduction

From the repository root, the commands use the frozen T2 packets of the matched equation-construction study. Each fit, curve, choice and score has a checked artifact identity for resume. The official validation commands replay this study's specific saved fitting and choice receipts in the default output directory. A separate fresh optimization run can produce different receipts and requires its own frozen validation plan.

~~~sh
.venv/bin/python -m scripts.run_hidden_form_forecasts --workers 4
.venv/bin/python -m scripts.report_hidden_form_forecasts all
.venv/bin/python -m scripts.validate_hidden_form_choices prepare
.venv/bin/python -m scripts.validate_hidden_form_choices evaluate
.venv/bin/python -m scripts.report_hidden_form_forecasts report
.venv/bin/python -m scripts.audit_t2_disposal_readout
.venv/bin/python -m scripts.check_anchored_latent_functions
~~~
