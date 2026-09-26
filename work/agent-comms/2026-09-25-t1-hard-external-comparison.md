# T1-hard: new component candidates versus frozen external baselines

The matching external-baseline comparison is complete. The new Autoformalism candidates outperform the saved SINDy, PySR and D3 endpoints on the displayed absolute trajectories and on pooled training/validation error. **GPT-5.6 Sol is stronger on validation and the absolute intervention trajectories.** This case does not establish an intervention advantage over Sol.

## Scope and provenance

Exact cell: `phase_b_dalla_man_t1_canonical_named_hard`.

Replayed all 12 available external endpoints: SINDy, PySR, D3 and Sol, repetitions 0–2 of each. Selection used only the cell, method and repetition. No external endpoint was chosen based on its test or intervention score. Existing endpoint/test-score fields in the adapted archive were dropped by `sanitize_subject` before validation and replay; no test trajectory was opened. The source archive itself contains previously recorded endpoint fields, so this is not a claim that its raw bytes contain no test metadata.

Reused the six Autoformalism candidates frozen in the preceding inventory review. The figures retain its same four displayed candidates; the complete data include all six. These are new component-campaign R4/R9/R10/R14 models, not the historical assisted R4 rescue. Autoformalism candidate identities, source occurrences, task component flags, equations, fitted parameters and initialization policies are included in `autoformalism-models.json`. Baseline identities, execution semantics and frozen parameterizations are in `baseline-models.json`.

All models have the same public channel contract: predicted target Gp, supplied auxiliary Gt, and meal_event_g input. Initial observed Gp is provided; subsequent observed Gp is not used to reset the rollout. Gt is supplied throughout under the T1-hard contract, including the intervention-specific regenerated Gt trajectory. This is conditional prediction with permitted auxiliary observations, not recovery of the complete autonomous physiological system.

Parameters, equations and initializers were unchanged. Continuous models used the existing Radau evaluator (rtol 1e-7, atol 1e-9); D3 retained its native recursive discrete increment on the one-minute grid. No ODE conversion of D3, refitting, new LLM calls or remote sessions occurred.

Evaluated 16 training trajectories, four validation trajectories and the same seven previously defined exploratory probes for each model. The 324 new baseline rollouts and 162 reused Autoformalism rollouts all succeeded. Exact time grids, reference arrays and training normalization scales match across all 18 models. The adapted baseline subjects omit original training-file hashes; exact cell/channel matching and frozen subject provenance are checked, but original training-data identity is not independently reconstructed.

Baseline archive SHA-256: `f352000cb518161b573f9f981be685fdee45a8e85b97c2e58cedf0a502b68f58`.
Baseline replay summary seal: `98634cb100c31f897c30d82c9e1e1616feae311dcd5fc07a5e4aa42afaeb225c`.
Autoformalism replay summary seal: `cefe23ff3e8eb194334954a0769a3cf4451055e44161a8434488c8f8fc22b071`.
Full source, runtime and plan hashes are in the accompanying sealed `plan.json` and `verification.json`.

## Results

Numbers below are absolute-trajectory NMSE using the common training standard deviation of Gp, 48.706498330673135. Training and validation pool every sample from every series. Baseline ranges cover all three repetitions; endpoints of different columns need not belong to the same repetition.

| Model | Training | Validation | Split meal | Meal + Gt(0) −20% | Meal + Gt(0) +20% |
|---|---:|---:|---:|---:|---:|
| Brief-only R10, retained | .01484 | .01546 | .02077 | .16657 | .19097 |
| Brief-only R14, trial | .01216 | .02220 | .01274 | .03445 | .05973 |
| Brief-only R4, one latent | .05182 | .07377 | .05077 | .05593 | .07538 |
| Full R4, trial, one latent | .04676 | .06701 | .04730 | .05634 | .06985 |
| Sol, reps 0–2 | .08615–.09451 | .00688–.00883 | .00654–.00779 | .00036–.00059 | .00030–.00115 |
| PySR, reps 0–2 | .35511–.38058 | .35979–.37111 | .32086–.32558 | .38061–.40994 | .25790–.27170 |
| D3, reps 0–2 | .42718–1.43962 | .41785–1.66216 | .28181–1.17288 | .30122–1.15569 | .23657–1.12550 |
| SINDy, reps 0–2 | 1.48248 | 1.71379 | 1.21432 | 1.19018 | 1.17440 |

The three SINDy trajectories coincide; no repetition was dropped. D3/PySR display delayed, attenuated meal responses; these are descriptive curve observations, not a scientific-compliance verdict. Their training errors are already much larger, so this is not the desired clean example of equally good training fit followed by sharply different intervention generalization.

Absolute error and response error should remain distinct. For a probe and its matched control, response RSE is

`sum_t [(pred_probe - pred_control) - (ref_probe - ref_control)]^2 / sum_t [ref_probe - ref_control]^2`.

For the split-minus-single response, Brief-only R10 has RSE .17876 versus Sol .20108/.21643/.25403. This modest response-shape advantage does not overturn Sol's lower absolute split-meal errors. For the initial tissue-glucose responses, Sol is also stronger: RSE .00482–.00593 for −20% and .00573–.01412 for +20%. The two simpler Autoformalism R4 models are much better on those responses than Brief-only R10, but are still less accurate than Sol here. `metrics.csv` contains all absolute and paired-response errors, including fasting controls.

## Figure and interpretation rules

Each `comparison-<method>.png/.svg` contains the same six panels: training 001, validation 001, split meal, meal with initial tissue glucose −20%, meal with initial tissue glucose +20%, and split-minus-single response. The two displayed development series are inherited from the previous plot, not selected after viewing baseline results. All 20 development series for all 18 models are in the data.

Split meal means 30 g at minute 60 plus 30 g at minute 90. Its control is 60 g at minute 60. The Gt probes change initial tissue glucose, not initial plasma glucose, while holding other physical initial states fixed. They are not mass-conserving redistribution experiments. The remaining fasting probes and matched controls are preserved in the data; no schedule was changed after viewing these results.

`comparison-sol-responses.png/.svg` shows the three response differences separately. These plots keep full curve ranges; no unfavorable external repetition or early transient is clipped.

Prior candidate caveats still apply: R14 is an unretained trial, the Full R4 trial has shared processes disabled, and low-error Brief-only models have questionable latent initial values. These results do not demonstrate pruning, complete physiological recovery, or a win by the fully enabled pipeline. Probes have been inspected previously and remain exploratory rather than independent confirmatory tests.

The existing generator uses meal jumps in the physical gut model, whereas continuous learned models consume the public pulse channel. In particular, pulses at the initial time can have different integrated forcing from interior pulses. The present new probes place meals at minutes 60/90, avoiding the initial endpoint, but common forcing representation remains part of the evaluation. Do not infer mechanism quality solely from the pooled training-versus-validation gap.

## Deliverables and checks

Generated files are uncommitted under `artifacts/t1-hard-external-comparison-2026-09-25/`:

- Five comparison figures in PNG and SVG, including the separate Sol response plot.
- `all-curves.csv.gz`: 146,286 prediction rows across 486 rollouts; columns identify model, split, case, trajectory, time, reference Gp and predicted Gp.
- `metrics.csv` / `metrics.json`: all 18 models and all seven probes.
- `baseline-models.json` / `autoformalism-models.json`: exact frozen models and provenance.
- `plan.json`, `summary.json`, `verification.json`, `files.sha256.json`: input/result identities, numerical verification and file digests.
- `replay.py`, `report.py`: analysis orchestration; production runtime unchanged.

Independent checks recomputed every NMSE and paired-response RSE and matched reference arrays across all models: maximum metric discrepancy 5.33e-15. Cache replay reproduced the identical sealed summary with no new simulations. All five figures were visually inspected. Focused regression tests: **20 passed in 1.03 s** (`tests/test_collect_component_models.py` and `tests/test_t1_external_interventions.py`). Full-repository `ruff check .` still reports the same 37 preexisting issues in `analysis/claude`; those files were not modified.

The plotting package is `transfers/t1-hard-external-comparison-20260925.tar.gz`. It includes the files above and this note as README. It excludes the full raw baseline archive and its original endpoint/test fields, unrelated inventory records, and superseded plots.
