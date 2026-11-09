# Latent dynamics under meals and imposed insulin

The historical T2-hard models predicted the glucose response to a meal much more accurately than the response to an isolated insulin pulse. Across the 16 saved fits, all 16 beat a zero-change prediction for glucose after a 48 g meal (relative squared response error 0.053–0.493). On the high fasting insulin pulse, zero of 16 did so (1.078–305,301). These 16 fits span two prompt variants, four critic/verifier configurations and two algorithm seeds. They share one benchmark.

For the Full prompt with critic and verifier enabled, the two algorithm seeds gave the following development results. The error compares the predicted intervention-minus-control time series with the simulator's paired difference. A zero-change prediction scores one. Insulin is a generated target in this task.

| Intervention and matched control | Glucose seed 0 | Glucose seed 1 | Insulin seed 0 | Insulin seed 1 |
|---|---:|---:|---:|---:|
| 48 g meal vs fasting | 0.481 | 0.053 | 1.000 | 0.676 |
| Split meal vs one equal-total-mass meal | 0.625 | 0.183 | 1.000 | 1.208 |
| Short low insulin pulse vs fasting | 198.789 | 1.108 | 3.489 | 1.821 |
| Short high insulin pulse vs fasting | 199.889 | 1.105 | 3.488 | 1.819 |
| Late long pulse vs fasting | 35.359 | 0.566 | 3.245 | 1.950 |
| Long fed pulse vs the same meal | 157.889 | 1.682 | 3.961 | 2.160 |

The two input archives bind the development roster, public release, 208 paired-solver rollout records and 1,068 historical score rows. The review retains all 212 generated-target records across the 16 historical T2 fits and four privileged reference fits. The reference fits have distinct release IDs. T2 conclusions above use only the historical rows. The accompanying figure plots the simulator and both Full seeds on the meal and short high-pulse interventions.

## Why seed 0 raises glucose after insulin

With 0.64 pmol/kg/min imposed for 17 minutes, the simulator's glucose change reached −2.839 mg/kg at minute 148. The seed-0 prediction reached +35.585 mg/kg at minute 300. Expanding the process definitions and substituting the fitted gains gives **+0.0000875393 × I × DelayedInsulinAction** in its glucose derivative. The candidate negates the process internally, then subtracts the process in the glucose balance. Its state named X has no path into glucose after expansion.

Integrating the intervention-minus-fasting change in that product to the model's glucose peak gives +35.585034 mg/kg. The discrepancy from the actual saved glucose response is 0.000041 mg/kg on the one-minute integration grid. Expanding the product about the paired control isolates each factor. The change in generated insulin multiplied by the control value of DelayedInsulinAction accounts for +35.585033 mg/kg with either solver. The terms involving a change in DelayedInsulinAction each contribute less than 0.000002 mg/kg in absolute value, and their signs vary between solvers at that scale. Its fitted input gain from I is only 3.05 × 10⁻⁹. Most of the response comes from current insulin multiplying the pre-existing mediator. This decomposition describes the fitted equations under the stored input perturbation.

Seed 1 subtracts **0.0000483850 × X_delayed** from its glucose derivative. The paired change in that term integrates to −3.863270 mg/kg at minute 300, within 0.000007 mg/kg of the stored prediction. It has the right signed peak, but the simulator has largely recovered by minute 300 (−0.198 mg/kg). Its full-trajectory glucose response error is 1.105.

## What the latent coordinates measure

All 64 candidate/quantity decisions are recorded for the four hidden simulator quantities Ra, Uid, EGP and X across 16 learned fits. Each review identifies the exact candidate and fitted parameter receipt, the declared and glucose-reachable states, the signed terms of the fitted glucose equation and a model-specific finding. None has a native hidden readout with independently supported physiological role and scale. Learned states and processes declare unspecified units. The ten quantity decisions for privileged reference structures identify their native physical outputs separately.

We also evaluated two model-defined fluxes that require no fitted decoder. One sums the terms in the glucose derivative driven exclusively by meal input; the other negates the sum of all insulin-linked terms so that a positive value denotes a plasma sink. Their units follow from the public plasma-mass balance in mg/kg/min. For the short high pulse, the two Full models' plasma-sink changes reach signed peaks of −0.290 and +0.030 mg/kg/min. The simulator tissue-disposal change reaches +0.170 mg/kg/min. Waveform errors are 11.728 and 1.288. In the simulator Uid enters the *tissue* balance, while glucose exchange and hepatic production also affect plasma glucose. This comparison measures the behavior of a candidate's own signed glucose-balance terms. It does not identify a unique tissue flux. An executable, fit-bound operational expression map accompanies all 192 flux comparisons.

Four decoders were compared using exactly the same saved intervention pairs. Each of the scalar fits chooses one varying coordinate and estimates an affine map from four training schedules. The glucose-reachable restriction follows the fitted, expanded equation graph with exact zero terms and algebraic cancellations removed. The generated-insulin control uses I alone. The supplementary multivariate ridge decoder uses all learned latent coordinates, fits coordinate centers and scales within each training fold, chooses among five positive penalties by leave-one-training-schedule-out MSE, then refits on all four schedules. It uses the held-out development contrasts only for scoring. This is a retrospective sensitivity analysis. The originally reported scalar endpoint and frozen discovery models are unchanged.

| Disposal decoder on short high pulse | Seed 0 RSE | Seed 1 RSE |
|---|---:|---:|
| Original single latent coordinate | 0.9992 | 1.6366 |
| Single glucose-reachable coordinate | 1.0000 | 1.6366 |
| Generated insulin only | 2,598.15 | 4.4221 |
| All latent coordinates, ridge | 0.9997 | 1.2137 |

The original seed-0 scalar decoder selected X. Restricting the probe to a glucose-reachable latent selects DelayedInsulinAction and gives essentially a zero-change response. Seed 1's scalar probe already uses a glucose-reachable coordinate. Ridge improved its numerical score in this contrast, while its error remained above one. Seven of the 64 original quantity-specific scalar probes used a coordinate outside the fitted glucose graph. Graph reachability establishes a possible dependency. The numerical contribution can still be tiny.

The two development-matched pairs also yield complete comparisons. In the Brief-only seed-0 pair the validation NMSEs were 1.015 and 0.975 and the long fed insulin-pulse disposal-probe errors were 25.971 and 4.481 (critic/verifier off and on, respectively). In the Full-prompt seed-0 pair the validation NMSEs were 0.976 and 0.989 and the corresponding disposal-probe errors were 2.781 and 1.001. Both pairs have equal state and fitted-parameter counts within the pair. The roster retains all 24 prespecified candidate pairs, including the 22 pairs outside the prespecified calipers. The 120 quantity/condition/readout rows for the two eligible pairs do not supply independent repetitions.

## Reproduction

From the project checkout, point the arguments to the original analyst evidence and its separate candidate-trajectory supplement. The command checks both archive digests. It writes the same results when resumed against the same scientific runtime and source. If changing the analysis or runtime, supply a fresh output directory.

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.iclr_latent_review \
  --evidence output/iclr-nonbaseline-evidence.zip \
  --traces output/iclr-nonbaseline-fidelity-traces.zip \
  --output output/iclr-local-completion/latent
~~~

The key tables are `target_effects.csv`, `learned_target_summary.csv`, `physical_readout_review.csv`, `physical_readout_map.json`, `signed_glucose_terms.csv`, `paired_product_factors.csv`, `glucose_balance_closure.csv`, `model_implied_flux_effects.csv`, `probe_calibration.csv`, `supplementary_probe_effects.csv`, `matched_pairs.csv` and `matched_pair_effects.csv`. The summary binds each table's digest. `historical_glucose_insulin_and_disposal.pdf` shows the paired trajectories. Negligible reference effects and solver disagreements retain explicit statuses and empty relative-error fields. The grid-quadrature residual is reported for every glucose term decomposition, including meal-event schedules where sampled forcing limits the accuracy of rate integration.
