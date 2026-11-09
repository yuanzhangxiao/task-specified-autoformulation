# Privileged reference fits and independent intervention evaluation

## Fitted comparison families

These internal controls quantify the benefit of being supplied the generating structure and physically informed parameter scales. The fitter reads each public Phase C development cell's training and validation projections. It fits the training trajectories using shared coefficients, selects one of three starts on validation, then freezes that endpoint before the paired intervention assay. Seed numbers index optimizer starts. Each start is reported. Runtime data consist of permitted input and auxiliary schedules and initial target observations. New subjects receive no private trajectories or individually fitted initial conditions.

| Task | Supplied structure and information | Fitted quantities |
|---|---|---|
| Named CSTR hard with common hidden preparation | Coupled source, feed and jacket balances with the generating Arrhenius law and concentration/temperature guards. Only reactor temperature is observed. | Seven positive dynamic coefficients and shared initial concentration and jacket temperature, with measured reactor temperature at time zero. |
| Canonical and perturbed named T1 easy | The same canonical plasma glucose mass balance with the three-compartment gut subsystem in both cells. Measured tissue glucose, production, insulin-independent use and excretion enter as permitted auxiliaries. The perturbed generator changes exchange dynamics. | Plasma/tissue exchange rates, gastric/intestinal rates, gastric-shape parameters and the effective appearance gain. Initial gut is empty by preparation and initial plasma glucose is observed. Each cell gets a separate fit. |
| Named T2 hard | Coupled canonical glucose, gut and insulin system with 12 physical states, the generator's meal-onset bookkeeping and a public observation map for glucose mass and insulin concentration. The source of insulin forcing remains separate from generated plasma insulin. | Free physical rate/scale combinations, shared basal quantities, hidden tissue/liver initial offsets and gains on initial target readings. Insulin action, gastric contents and secretion dynamics use their specified basal preparation. |

The reference code discloses each fitted coordinate and its scale. Every starting log-multiplier is independently sampled from [−0.10, 0.10]. Every listed parameter can move by a factor between exp(−0.5) and exp(0.5) of its scale, with the gastric upper shape parameter additionally bounded below one. These bounded, physically informed parameterizations give the reference a substantial information advantage. They do not estimate unrestricted structural performance or establish identifiability. The initializers cannot condition on future target observations.

Each objective weights target errors by the public training standard deviation. The trusted CasADi transcription uses bounded nonlinear least squares with exact automatic derivatives and public piecewise-linear forcing. Public ten-minute triangular meal rate schedules determine commanded mass. At the start of each meal a hybrid memory stores new mass plus current stomach content, reproducing the reference normalization. This bookkeeping is privileged structure outside the autonomous discovery grammar. The fitted endpoint is replayed at twice the integration resolution. A third resolution is used on independent intervention schedules, and private reference curves use two independent adaptive solvers. The independent numerical test also reproduces the simulator's hidden channels with the true coefficients and checks that later observed targets cannot enter the fit adapter's forward map.

The three-start campaign (50 optimizer evaluations maximum per start) produced 12 qualified fits. Selected validation NMSE was 9.38e-15 on CSTR-hard, approximately 5.80e-12 on canonical T2-hard and 3.68e-10 on canonical T1-easy. Fitting exactly the same canonical conditional balance to perturbed T1-easy gave 0.8651. This quantifies the effect of the exchange-law change for this parameterization, information interface and bounded optimizer. It does not establish that a human expert would perform poorly. `reference_structure_results.csv` reports all starts and `literature_transfer_results.csv` contains the six canonical-structure T1 fits. No third-party discovery method is implemented here.

## Frozen paired assays

`scripts/iclr_anchor_fidelity.py prepare` consumes the exhaustive internal search export or the frozen fitted-reference export. It commits model identities, applicability, all schedules, the calibration/contrast split, source versions, public training scales, and matched pairs. T2 receives low and high insulin amplitudes with a common onset, a later, longer infusion, and a longer infusion in a fed preparation; meal versus fasting and split versus single meal are also paired. Only imposed insulin forcing is manipulated. The reference generates plasma insulin, remote action, disposal, meal appearance, endogenous production and glucose jointly. All target measurements after time zero are excluded from model inputs. Named CSTR-hard receives low/high feed concentration and cool/warm jacket changes with different onsets and durations, paired against the same reset. Concentration and jacket temperature remain hidden. The reactor clock uses normalized model time. Signed jacket heat is positive into the reactor, measured in K per normalized time unit. Dalla Man times are in minutes.

Fresh future-search confirmation uses different doses/timings from the local development assay. The confirmation run requires a new sealed internal search cohort. Public status, failed or unavailable models, unresolved numerical effects and physically negligible reference contrasts remain in the output. Per-target absolute NMSE uses its complete public training scale. Hidden quantity NMSE uses the explicitly separate training-calibration scale. Effect RSE uses each model's own paired control and the entire reference difference, with a near-zero reference flagged as nondiagnostic. Magnitudes, absolute RMSE, signed peak and timing, total integrated effect, and the direction/integral in the frozen response window are retained. A tiny predicted change may have no resolvable peak time.

Direct hidden metrics require a physically reviewed readout. The internal reference fits supply the direct equations. For discovered models, an optional evaluator-only JSON map binds the exact candidate content hash to a mapping such as `{"Uid":{"expression":"...","unit":"mg/kg/min","evidence":"documented unit and role correspondence"}}`. Every expression uses the project's restricted evaluator. Unknown candidate hashes and wrong units fail before scoring. An absent verified readout remains unavailable. A separate affine probe selects a single varying latent coordinate and calibrates its slope and intercept only on designated public training schedules with evaluator-private reference labels after model selection. Its performance measures accessible information about each hidden waveform. Model parameters and the probe are fixed for all contrasts. The native physical readout and calibrated measurement are reported separately.

Pairs are fixed before intervention scoring within the same benchmark, prompt variant, correct task and seed. The same state count, distinct equation fingerprint, validation NMSE ratio at most 1.20 and parameter-count gap at most max(2, 20% of the smaller count) are required. `matched_fit_pairs.csv` includes reasons for unmatched candidates and paired outcomes for every accepted pair. Mechanistic misspecification and public-specification degradation are separate comparisons.

## Development results on fresh schedules

On the timing comparison of two 24 g meals against one 48 g meal, canonical T1-easy split-minus-single glucose response RSE was approximately 9.33e−10. Refitting the same plasma exchange law on perturbed T1 data gave 0.596 on this response, and 0.295 on the direct meal-appearance response. Altering exchange dynamics in the simulator degraded the transferred fit on the hidden meal readout even though the simulator's gastric law stayed fixed.

For the 0.64 pmol/kg/min, 17-minute fasting insulin pulse on historical T2-hard, the two Full-prompt, critic-and-verifier-enabled seeds gave affine-decoded insulin-dependent disposal response RSE of 0.999 and 1.637. Their endogenous production response RSE was 1.005 and 17.58. An RSE of one corresponds to predicting zero change. Meal appearance has zero physiological response to insulin in this fasting preparation. The separately fitted privileged model on the corrected Phase C release gives direct disposal response RSE below 1e−8. Those cohorts have different training data versions; use the future frozen controlled search for a same-release comparison.

For corrected CSTR-hard, the fitted reference achieved temperature response RSE below 1e−11 under the high feed-concentration pulse. Its directly computed reaction heat and jacket heat changes also scored below 1e−11. This is a check on numerical attainability with the coupled balances supplied. CSTR hard discovery performance awaits the planned runs.

## Commands

From the repository root, set `AF_PYTHON` to the project environment with CasADi installed and `AF_RELEASE` to the Phase C development-2 public directory. The reference fits require CPU only. Per-parameter-evaluation residuals and exact Jacobians are cached by parameter-vector digest for deterministic restart. Retain `fits/*/<start>/` on the computing host for restart; the sealed fit results, plan and compact export suffice for transfer. Generate a fresh root if source, runtime, public arrays or budgets change.

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$AF_PYTHON" -m scripts.iclr_reference_fits prepare --root "$AF_FITS" \
  --public-root "$AF_RELEASE" --starts 3 --max-evals 50
for CELL in $("$AF_PYTHON" -c 'from scripts.iclr_reference_fits import CELLS;print(*CELLS)'); do
  for START in 0 1 2; do
    "$AF_PYTHON" -m scripts.iclr_reference_fits fit --root "$AF_FITS" \
      --public-root "$AF_RELEASE" --cell "$CELL" --start "$START"
  done
done
"$AF_PYTHON" -m scripts.iclr_reference_fits freeze --root "$AF_FITS"
~~~

For the new main search, complete and export its planned endpoints first. `AF_STRESS_PUBLIC` contains the exact generated public files if using `acquisition_00` or `sensor_00` aliases. The adapter verifies the original specification digest and equality of the two public splits to the corresponding clean anchors. For direct Phase C cell IDs this option may be omitted. Keep the private evaluator data root away from the proposer.

~~~bash
"$AF_PYTHON" -m scripts.iclr_anchor_fidelity prepare --mode confirmation \
  --models "$AF_SEARCH/frozen_models.json" --models "$AF_FITS/frozen_models.json" \
  --public-root "$AF_RELEASE" --stress-public-root "$AF_STRESS_PUBLIC" \
  --data-root "$AF_PRIVATE_DATA" --root "$AF_FIDELITY"
"$AF_PYTHON" -m scripts.iclr_anchor_fidelity run --data-root "$AF_PRIVATE_DATA" \
  --root "$AF_FIDELITY" --workers 4
"$AF_PYTHON" -m scripts.plot_iclr_anchor_fidelity --root "$AF_FIDELITY"
~~~

For historical six-visit T2 comparison, pass the prior dynamics ZIP's `subjects/models.json` plus the fitted reference export, omit the new search bundle, and use `--mode development`. The earlier archive uses the Phase B T2 event representation, distinct from the Phase C ingestion-rate release. Rows carry release-specific cell IDs. Output includes `frozen_selection_manifest.json`, `intervention_effects.csv`, `hidden_process_metrics.csv`, `matched_fit_pairs.csv`, fitted/calibrated readout records, versioned reference traces, checkpointed candidate traces and `absolute_and_difference_trajectories.pdf`. Future-search results depend on executing the cluster jobs.

## Historical corrected CSTR easy held-out evaluation

The original nine-case protocol uses a different reactor task, release and sampling interface from the Phase C hard anchor. The new four-block rerun has two seeds, two prompt variants and four critic/verifier arms. When all 16 lineages have reached their final pruning/critic disposition, run the native campaign export and score on a CPU allocation. `AF_CSTR_STAGE` is the development staging directory containing `correction.json`. `AF_PRIVATE_DATA` holds the pinned private simulator specification. Use a fresh evaluator-only `AF_CSTR_EVAL` directory outside the search, stage and data roots.

~~~bash
"$AF_PYTHON" scripts/component_campaign.py export --root "$AF_CSTR_RUN"
"$AF_PYTHON" -m scripts.iclr_corrected_cstr_test evaluate \
  --root "$AF_CSTR_RUN" --staging "$AF_CSTR_STAGE" \
  --data-root "$AF_PRIVATE_DATA" --output "$AF_CSTR_EVAL"
~~~

The evaluator checks the complete final roster, immutable fitted subjects, both corrected training/validation fingerprints and submitted public prompt against the staged historical plan. It then regenerates the original operating schedules under `reference-events-2` while preserving the original measurement roles and event-input convention. Every train, validation and test schedule must pass the independent solver and grid refinement checks. The checked test CSV is generated inside the new evaluation directory. The command accepts no historical test CSV argument. Predictions reuse the project's frozen free-rollout scorer and the original corrected training scale, with measured concentration and jacket temperature supplied. An interrupted or unavailable subject remains in the 16-row table. This script produces a corrected held-out NMSE table, with no new functional-compliance claim. The development-only command successfully reproduced all 16 training and four validation projections on the local host. The real held-out evaluation awaits the corrected search freeze.

~~~bash
"$AF_PYTHON" -m scripts.iclr_corrected_cstr_test development-check \
  --corrected "$AF_CORRECTED_CSTR" --data-root "$AF_PRIVATE_DATA" \
  --output "$AF_CSTR_DEV_CHECK"
~~~

The release receipt and model results are `release.json`, `corrected_cstr_test.json`, and `corrected_cstr_test.csv`.
