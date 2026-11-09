# Latent dynamics in the glucose models

The named T1 models use internal dynamics to turn a meal input into a delayed glucose response. Their timing improves over the one available matched no-latent model, but their meal-appearance flux is too small. On T2-hard, the fitted models give little evidence of recovering remote insulin action under two additional fasting insulin pulses. These are exploratory evaluations of frozen models on specified synthetic preparations. They measure response behavior in physical time without using the paper's compliance score.

## Meal timing in free rollout

On canonical T1-easy, we compare a 60 g meal at minute 60 with fasting, then split the same meal between minutes 60 and 90. Each model predicts glucose freely after initialization. The simulator regenerates all permitted physiological auxiliaries for each schedule. Each response subtracts the same model's matched control. Relative squared error (RSE) divides the squared response-curve error by the energy of the reference response. Exact agreement scores zero and predicting zero change scores one.

| Named T1-easy, seed 0 | Single-meal response peak, min | Single-meal peak, mg/kg | Split-minus-single trough, min | Split response RSE |
|---|---:|---:|---:|---:|
| Simulator | 129 | 97.50 | 92 | 0 |
| Full | 122 | 65.77 | 90 | 0.277 |
| No latent | 61 | 87.89 | 61 | 1.735 |

The no-latent model was closer on single-meal peak height. Its response came almost immediately after eating. Only one named no-latent seed produced a retained model. On the obfuscated version, Full had lower split-response error than prediction-only in both seed pairs (0.677 versus 1.135 and 0.625 versus 0.904). All three obfuscated GPT-5.6 Sol repetitions did better than both Full seeds on both meal contrasts. All four Full models beat the corresponding SINDy, PySR and native D3 repetitions on those two contrasts. The named and obfuscated cases use the same physiology.

The comparison covers 31 of 32 planned endpoints and 248 of 256 contrasts. It includes tissue-glucose shifts, where both obfuscated Full models predict zero effect. Full, prediction-only and no-latent use historical round-12 retained models. Prediction-only also omits scientific background. The separate perturbed-dynamics sensitivity table retains all five endpoints and the original solver qualification of each score. [All methods and repetitions](comparisons/README.md) · [Per-contrast values](comparisons/paired_responses.csv) · [Figure](comparisons/paired_meal_responses.png)

## Direct measurement of the meal pathway

A good glucose prediction can use the supplied physiological auxiliaries. We therefore also measure how much glucose the model's own meal pathway contributes per minute. We fix measured plasma glucose and every supplied auxiliary at their public basal values, evolve each candidate's remaining states, and read its original plasma-glucose equation. Subtracting its fasting rate gives the meal-driven flux in mg/kg/min. In the canonical reference balance the only remaining meal-dependent term under those clamps is the hidden rate of gut glucose appearance, (R_a=f k_{abs}Q_{gut}/BW). The reference gut subsystem is driven solely by the meal schedule. This lets us compare the signed flux directly without choosing a latent name, calibrating a gain, or aligning a time shift.

A second calculation holds each candidate's internal states to its own time-varying fasting trajectory and reapplies the meal input. This keeps fitted parameters, glucose, auxiliaries and direct input effects fixed across the comparison. The change in RSE measures what the fitted internal dynamics contribute **within that model**. It does not estimate how well a newly trained fixed-state model would do.

We used all 25 available continuous-time endpoints in the source inventory. All 75 endpoint/schedule combinations passed the Radau/BDF response agreement check. Reference curves independently passed Radau/DOP853 comparison. Each schedule spans 360 minutes. The inputs are 45 g at minute 37, 75 g at minute 83, and 30 g plus 45 g at minutes 43 and 128. We preserve the historical event representation: the simulator receives exact meal jumps and each fitted model receives its original minute-grid, linearly interpolated meal channel. The isolated flux is a conditional equation diagnostic. The free physical rollout above answers a different question.

| Endpoint | Median flux RSE | Median flux RSE with fasting internal states | Median predicted/reference area |
|---|---:|---:|---:|
| Named Full, seed 0 | 0.758 | 1.000 | 0.142 |
| Named Full, seed 1 | 0.821 | 1.000 | 0.155 |
| Named no-latent, seed 0 | 5.230 | 5.230 | 0.145 |
| Obfuscated Full, seed 0 / 1 | 1.024 / 0.393 | 1.001 / 1.000 | −0.008 / 0.491 |
| Obfuscated prediction-only, seed 0 / 1 | 0.152 / 2.220 | 1.105 / 2.210 | 1.078 / 0.069 |
| Named GPT-5.6 Sol, reps 0 / 1 / 2 | 0.686 / 0.013 / 0.811 | 1.000 / 1.000 / 1.000 | 1.097 / 1.114 / 1.112 |

Each entry summarizes the three schedules for the indicated individual endpoint. The named Full models capture a delayed component, but account for only about 14–15% of reference appearance by integrated area, and their median peaks are 19 minutes early. No-latent puts almost all of its contribution at the meal. The named Sol repetition with RSE 0.013 is a counterexample to any claim of superiority over the external agent. All six PySR and six SINDy endpoints have zero meal-flux response in this clamp. The source inventory lists six native D3 rows as unavailable for this continuous-rate assay; D3 participates in the free-rollout comparison above. [Scores by schedule](meal-appearance/scores.csv) · [All endpoints](meal-appearance/endpoints.csv) · [Flux figure](meal-appearance/meal-appearance.pdf)

## Remote insulin action on T2-hard

The T2-hard candidates must generate both insulin and plasma glucose. Only the prescribed forcing, permitted auxiliary and initial target values enter rollout. For each frozen model we use three registered training designs to fit an evaluation-only scalar affine readout of the simulator's hidden remote action (X). We choose among candidate latent coordinates using those calibration trajectories alone. We also fit a separate readout using only the model's generated insulin. Neither readout changes the model or its parameters. A model with no varying latent coordinate receives an explicit unavailable latent score and remains eligible for target-response scoring.

We then apply two new fasting infusions, 0.5 pmol/kg/min over minutes 50–65 and 0.35 over minutes 120–150, each against fasting without infusion. Both reference and candidate receive the same sampled insulin input on these fresh contrasts. The archive contains 16 six-visit component endpoints (two prompts, two seeds and four critic/verifier arms), three Sol, three SINDy, and one executable PySR model. All 23 available subjects completed their rollouts. Two PySR and three D3 identities remain in the 28-subject denominator. No official test trajectory, new model fit or LLM call was used.

| Full prompt, critic and verifier on | Early pulse action RSE | Late pulse action RSE | Early / late predicted peak change |
|---|---:|---:|---:|
| Reference | 0 | 0 | 7.199 / 8.580 |
| Seed 0 | 0.999 | 0.998 | 0.008 / 0.012 |
| Seed 1 | 1.664 | 1.296 | 2.319 / 2.580 |

Both Full seeds got the response direction right. Seed 0 produced less than 0.2% of the reference peak on either pulse. Seed 1 reached its largest response at the 300-minute observation boundary. The affine probes measure how well one learned coordinate tracks physical action. Physical identity, disposal fidelity and recovery through a distributed nonlinear representation remain separate questions. [Pulse scores and absolute errors](t2-action/metrics.csv) · [Insulin-action trajectories](t2-action/action-response.pdf)

**Similar validation fit can conceal different responses.** Two of 24 within-prompt, within-seed arm pairs met the development-only matching criteria: validation errors within 20%, equal state count, similar parameter count and distinct equation fingerprints. In the Full pair, both components on versus both off gave early-pulse action RSE **0.999 versus 3.657** at a validation-error ratio of 1.014. In the Brief-only pair the ordering reversed, **7.045 versus 2.416**, at a ratio of 1.042. Late-pulse results agree in direction within each pair. These two pairs support a distinction between fit and behavior. They do not establish an average component advantage. The manifest retains all eligible and ineligible pairs. [Development matching](subjects/matched-pairs.json)

**What changed during search.** In the first planned named T2-hard Full lineage, visit 1 changed delayed insulin action from an algebraic expression into an evolving state, increasing state count from three to four. Validation NMSE fell from 1.235 at visit 0 to 1.132 at visit 1 and 0.989 by visit 3. Trials at visits 4 and 5 scored 1.095 and 1.096 and the incumbent persisted. Its small subsequent intervention response illustrates why admitting a state and improving fit do not establish correct latent dynamics. [Selected and trial equations, parameters and visit history](subjects/lineage.json)

## Where the submitted tables came from

The external NMSE values in Table 2 are test results with three repetitions per case. The four method/ablation NMSE values are validation results at six visits, pooled over two prompt variants and two seeds before aggregating cases. The snapshot had no pruning endpoints. These columns cannot support a common held-out accuracy ranking. On the consistent fitted-equation compliance endpoint, Full-prompt both-on scored 58.3%, while Brief-only both-on scored 72.2% before final pruning. The submitted 88.9% matches Brief-only graph certification. The other three submitted ablation compliance rows use a mixture of prompt variants. We could not trace the 85.0% prose value. The new intervention analyses use frozen executable models and do not use these mixed table values as inputs. [Original validation table](provenance/validation_ablation.tex) · [Original external test values](provenance/external_test_nmse.tex)

## Reproduction and cluster use

The archive contains the compact frozen model bundles, per-model scores, trajectories for the new mechanistic assays, analysis scripts and a SHA-256 manifest. Run the modules from the project checkout with its Python environment. Set `EVIDENCE` to the directory where you extracted the result ZIP and `REPLAY` to a new output directory. The initial reconstruction of the physical T1 comparisons also requires the original historical replay artifacts in the checkout.

~~~bash
export EVIDENCE=/path/to/extracted-evidence
export REPLAY=/path/to/new-replay
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m scripts.evaluate_t1_meal_appearance \
  --models "$EVIDENCE/inputs/t1-models.json" --output "$REPLAY/meal-appearance"
.venv/bin/python -m scripts.report_t1_meal_appearance --root "$REPLAY/meal-appearance"
.venv/bin/python -m scripts.evaluate_t2_latent_dynamics prepare \
  --models "$EVIDENCE/subjects/models.json" --root "$REPLAY/t2-action"
.venv/bin/python -m scripts.evaluate_t2_latent_dynamics run \
  --root "$REPLAY/t2-action" --workers 4
.venv/bin/python -m scripts.plot_t2_latent_dynamics \
  --root "$REPLAY/t2-action" --pairs "$EVIDENCE/subjects/matched-pairs.json"
.venv/bin/python -m scripts.compare_t1_rebuttal_effects --artifacts artifacts \
  --cache "$REPLAY/t1-effect-cache" --output "$REPLAY/comparisons"
.venv/bin/python -m scripts.plot_t1_rebuttal_effects "$REPLAY/comparisons"
~~~

All reported numerical results ran locally. To add the three missing native D3 rows, use a Delta CPU allocation with the original Phase B D3 full campaign. From the checkout containing the bundled scripts, set `AF_PYTHON` to the project's interpreter, `EVIDENCE` as above, and `REPLAY` to a fresh writable directory. This reads the native fitted selection, preserves its discrete increments and performs no proposal or refit.

~~~bash
export D3_ROOT="/work/hdd/bibo/$USER/phase_b/d3-native-vllm-120b-v1"
"$AF_PYTHON" -m scripts.prepare_t2_hard_subjects \
  --retained-bundle "$EVIDENCE/subjects/models.json" \
  --native-d3-root "$D3_ROOT" --output "$REPLAY/subjects-with-d3.json"
"$AF_PYTHON" -m scripts.evaluate_t2_latent_dynamics prepare \
  --models "$REPLAY/subjects-with-d3.json" --root "$REPLAY/t2-with-d3"
"$AF_PYTHON" -m scripts.evaluate_t2_latent_dynamics run \
  --root "$REPLAY/t2-with-d3" --workers 4
~~~

The already implemented LLM-SR and LLM-ODE campaigns use a different Phase C release. A comparison against those models needs a common release, input convention and frozen cohort. A matched fixed-representation search would additionally require equation revisions from a shared initial candidate under a state freeze. The retrospective no-latent arm and the within-model fasting-state intervention answer narrower questions.
