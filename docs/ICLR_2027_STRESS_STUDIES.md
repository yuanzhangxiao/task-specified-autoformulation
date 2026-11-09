# Data quality and dynamical dimension experiments

The opt-in release contains 10 one-factor acquisition conditions, six procedural dimension conditions, four optional coupling and decay-spread controls, and one historical CSTR correction. At three algorithm seeds with Full and fixed-Γ this is 126 discovery lineages. On the development build all 21 cells were prepared. No prospective prediction or discovery-cost estimate is available until models are trained, selected, frozen and evaluated.

## Observation quality

| Case | Conditions | Fixed quantities |
|---|---|---|
| T2-hard | Gaussian measurement SD of 0%, 1%, 5%, 10% of each clean training-channel SD | Targets Gp and I, Uii auxiliary, original physical meal and insulin forcing, grid and horizons |
| T2-hard | Observation stride 1, 2, 4 plus all necessary forcing corners | No measurement noise, same underlying trajectories and external forcing interpolants |
| CSTR | T; T+C; T+Tj; T+C+Tj | Hard public task, temperature target, training/validation preparations, external input paths |
| CSTR historical easy | Segmented-integration correction of original named easy data | Historical easy public brief, original preparation and conditional input convention |

Each T2 condition uses 16 training and four validation trajectories of 300 minutes. Corner-preserving stride 1/2/4 produces 4,816 / 2,432 / 1,247 training samples. Independent Gaussian draws use a fixed hash of the data seed, trajectory identity, role and channel. The same draws are scaled to each requested noise level. The initial target measurement receives the condition's noise. Constant measurement channels remain exact. No external input receives noise. The two methods see identical public arrays including the original training and validation split membership. No clean derivatives or derivative labels enter discovery. Target normalizations are recomputed from each conditioned training split.

The CSTR masks have exactly the same Section A scientific obligations and preparation. The two one-auxiliary conditions compare sensor identities at fixed count. The corrected historical easy cell uses the exact byte-pinned `reference-events-2` public export. Its proposer-prompt hash is `0a3b4c055402ed26352363c4d367b791c41460ea21e8ca42043c495ea68eb99c`. Keep this separately labeled from the reset Phase C masks.

## Dimension, coupling and cost

Every procedural draw uses the original notebook's damped skew-coupled, bounded nonlinear alien-device grammar and the existing private ODE simulator. The sampled systems have 4, 8 or 12 total dynamic states, including the observed output. Generator seeds 11639 and 27181 are used at every dimension. The learner sees v01 and u01, a common scientific brief and reset protocol, and an opaque cell ID. It receives neither equations nor dimension. Six training and two validation schedules per system share a 24-unit horizon and 0.25-unit grid.

Draw admission uses training designs only. A unit pulse must change peak output by 0.05 to 8 units relative to its fasting/zero-input counterpart. For each latent coordinate the largest absolute central-difference output response to ±0.001 in its initial value, divided by 0.002, must reach 0.005. This measures local influence, not the minimal input/output realization order. For the six original independent systems there were eight attempted draws; one eight-state generator seed needed three attempts. The four secondary systems double the recurrent skew coupling or spread log decay about its mean on the same eight-state draws. They retain their denominators if a perturbation fails qualification. Accepted systems had maximum output sensitivities of at least 0.088 for every hidden coordinate. A second training-only audit in `scaling_excitation.json` reports each hidden coordinate’s excursion under the unit pulse relative to a zero-input control. All coefficients, input-effect amplitudes, rejected systems, acceptance diagnostics and private masks are in `scaling_system_manifest.json`. Keep that file evaluator-side.

Report discovered state and parameter counts, target and intervention errors, completeness, recorded optimizer and model service costs, and unmeasured costs. Pair by generator draw and algorithm seed. A few systems of increasing dimension are descriptive evidence for this family. Independent draws from the same grammar are distinct from independent scientific domains.

## Run and resume

From the project checkout set `AF_DATA` to a separate experiment directory, `AF_SEARCH` to a fresh controlled-search root, and `AF_PYTHON` to the project interpreter. The preparation reads the corrected Phase C release and the corrected-only CSTR export. Override their locations with `--release` and `--corrected` when needed.

~~~bash
"$AF_PYTHON" -m scripts.iclr_stress prepare --root "$AF_DATA" --extended-scaling
mapfile -t stress_cells < <(jq -r '.cells[] | select(.status != "generation_failed") | .cell | select(. != "acquisition_00" and . != "sensor_00")' "$AF_DATA/manifest.json")
"$AF_PYTHON" -m scripts.iclr_controlled_search prepare \
  --public "$AF_DATA/public" --cells "${stress_cells[@]}" \
  --arms full fixed_gamma --seeds 0 1 2 --visits 15 --root "$AF_SEARCH"
~~~

The two anchors (`acquisition_00` and `sensor_00`) take their Full/fixed-Γ repetitions from the main matched ablation campaign. The other 19 cells give 57 paired blocks. Use the controlled-search ACES launcher and the same proposer/critic calibration and fit budgets as the main ablations. Prepare on the target environment so that captured numerical-library identities match. Each source cell and generated system has a digest and a checkpoint. Each group of paired arms is charged for its search work with the common initial draft identified by the controlled-search ledger.

Once each campaign has an immutable endpoint export, run its evaluation in its own directory. These evaluators use CPUs. Keep the discovery workers on the public subtree only. The two input bundles must both include Full and fixed-Γ for seeds 0, 1 and 2 for every cell they contribute. Full-model failure, construction failure, missing native evidence and numerical failure remain distinguishable from completed scores. Optionally pass `--reference-root /path/to/frozen-B1-fit` to evaluate a privileged fitted clean-anchor reference on the T2 acquisitions and CSTR masks. Parameters remain fixed across acquisition conditions, and the training/validation input digests must match the original clean anchor.

~~~bash
"$AF_PYTHON" -m scripts.iclr_stress evaluate --root "$AF_DATA" \
  --models "$AF_SEARCH/frozen_models.json" --output "$AF_DATA/score-stress" --data-root data_raw
"$AF_PYTHON" -m scripts.iclr_stress evaluate --root "$AF_DATA" \
  --models "$AF_ANCHOR_SEARCH/frozen_models.json" --output "$AF_DATA/score-anchors" --data-root data_raw
"$AF_PYTHON" -m scripts.iclr_stress report --root "$AF_DATA" \
  --evaluations "$AF_DATA/score-stress/evaluation.json" "$AF_DATA/score-anchors/evaluation.json" \
  --output "$AF_DATA/report"
~~~

For matched-data fitted reference diagnostics, feed the same impaired training and validation rows to the privileged B1 fitter. The bounded roster contains the 10% noise T2 condition and the CSTR condition with both auxiliary sensors. The coupled CSTR reference integrates its own compartment states and uses the permitted measured initial temperature. Its reproducible shared hidden initialization is train fitted. No hidden initial values are read per test trajectory. The fitted parameters and starts remain separately identifiable from the true-parameter simulator.

~~~bash
"$AF_PYTHON" -m scripts.iclr_stress_reference --stress "$AF_DATA" --public "$AF_DATA/reference-fit-inputs"
T2=phase_c_dalla_man_t2_canonical_named_hard_rates_v1
CSTR=phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1
"$AF_PYTHON" -m scripts.iclr_reference_fits prepare --public-root "$AF_DATA/reference-fit-inputs" \
  --cells "$T2" "$CSTR" --root "$AF_D1_REF"
cp "$AF_DATA/reference-fit-inputs/conditions.json" "$AF_D1_REF/conditions.json"
for cell in "$T2" "$CSTR"; do
  for ((start=0; start<$(jq '.starts' "$AF_D1_REF/plan.json"); start++)); do
    "$AF_PYTHON" -m scripts.iclr_reference_fits fit --public-root "$AF_DATA/reference-fit-inputs" \
      --root "$AF_D1_REF" --cell "$cell" --start "$start"
  done
done
"$AF_PYTHON" -m scripts.iclr_reference_fits freeze --root "$AF_D1_REF"
"$AF_PYTHON" -m scripts.iclr_stress_reference --stress "$AF_DATA" \
  --fit-root "$AF_D1_REF" --report "$AF_DATA/report/privileged-development.json"
~~~

All six prespecified conditioned starts were fitted locally. At 10% noise, the selected T2 start reached clean validation NMSE 0.0000566 across Gp and I (0.0000429 and 0.0000704), compared with 0.0101 against the noisy validation measurements. All three starts reached the 50-evaluation cap. On the CSTR all-sensor condition, the selected coupled-balance fit reached validation NMSE 9.38×10⁻¹⁵ and its optimizer converged. Each score averages four validation trajectories. These results quantify behavior with supplied equations and the stated initialization assumptions; robustness of end-to-end discovery awaits the paired searches.

Pass `--reference-root "$AF_D1_REF"` when evaluating the stress sweep. Each fit must match the exact altered development-file hashes. Its measurements are written to `privileged-references/<cell>.json`. These are separately labeled reference diagnostics and do not increase the Full/fixed-Γ denominator.

The reference generator and the predictor each use two solvers. For each held-out schedule a paired control preserves the preparation with baseline inputs. Predictions receive only the acquired initial target values, public forcings and permitted auxiliary sensors; the independent evaluator scores against the clean dense target grid. Both members of a pair use the same noise realization. Report per-target absolute clean MSE, sample-mean NMSE on the common dense grid at the condition's training scale, common clean-training scale NMSE, and intervention-response RSE. The latter compares predicted and true matched differences with the zero-effect prediction at one; negligible true effects have no relative score. Numerical disagreement is unresolved. Keep all seeds and every failed row in `completion.csv`. A report generated before any endpoint export describes planned counts only.

For scheduler resource usage, collect array-task rows with their array IDs, including any recovery arrays, for one search plan per invocation.

~~~bash
sacct -X --array -j "$AF_ARRAY_IDS" -P --format=JobID%45,ElapsedRaw,TotalCPU,AllocCPUS,AllocTRES%120,State > "$AF_DATA/stress-sacct.psv"
"$AF_PYTHON" -m scripts.iclr_stress_resources --plan "$AF_SEARCH/plan.json" \
  --sacct "$AF_DATA/stress-sacct.psv" --output "$AF_DATA/report/scaling_scheduler_cost.csv"
~~~

The scheduler ledger deduplicates identical records, excludes array parent and step rows, and keeps retries as separately billed job IDs. It reports actual CPU seconds and allocated GPU-hours for the complete paired block. Allocation is separate from measured accelerator utilization and from the per-arm proposer tokens, numerical fitting and wall times in `scaling_costs.csv`. Competing jobs and hosted service overhead limit comparison of wall times.
