# Fitted equations, historical cohorts and canonical T1 transfer

The historical six-visit snapshot contains 160 lineages. We recovered the executable selected model and six-visit functional assessment for 78, covering the four named glucose cases. Eighty other lineages retained a validation result with no matching local model export. Two had no retained fit. The per-seed table preserves all rows, prompts, selected visits, data and result digests, component arms and known model/fit digests. The graph and functional checks report separate outcomes. The cohort registry distinguishes the earlier round-12 T1 comparisons, the six-visit Phase B cohort, the evaluation-only calibration and retrospective interventions, the privileged fits, and each prospective campaign.

## What changed in the T2-hard example?

The Full-prompt seed-0 search already had a memory state at its first retained visit. The fitted glucose derivative contained a contribution proportional to insulin times that state. We compared both the trial and the retained compiled right-hand side at each visit, holding the other coordinates and all forcing fixed. The signed coefficient is the unit-square mixed difference in the glucose derivative when varying insulin and the driving state. These fitted expressions are bilinear in those two arguments.

| Retained visit | State carrying the interaction | Fitted coefficient in the glucose derivative | Validation NMSE |
|---|---|---:|---:|
| 1 | X | −0.29672 | 1.235 |
| 2 | DelayedInsulinAction | −0.09535 | 1.132 |
| 3 | DelayedInsulinAction | +0.00007733 | 1.011 |
| 4–6 | DelayedInsulinAction | +0.00008754 | 0.989 |

Visit 2 adds a fourth declared state and reroutes the disposal expression through it. X still evolves but has no path to either generated target. Visit 3 reverses the sign of the bilinear glucose contribution. Positive insulin and positive action now add to the glucose derivative through the process named disposal. The last two trial models scored worse than the visit-4 incumbent and remained unselected. The local derivative calculation isolates this term. The separate paired fasting pulse evaluates the coupled model. Historical metadata records bounded runtime edits, fit origins and selection receipts. Verbatim critic text and causal attribution of each change to particular advice are unavailable.

## Variation across seeds

Both equations were available for 38 of 80 planned pairs. Fifteen pairs had equal target-reachable state dimension. Eight of the 78 recovered retained models contained at least one state without a dynamic path to a target. Seventeen pairs shared the same set of public-input-to-target dependencies, although none had an identical full signed topology under the original canonical fingerprints. The active-graph calculation uses the project's restricted equation simplifier after substituting fitted coefficients. It follows processes and differential dependencies from every output. Reachability gives an upper bound on how many states could affect these outputs. It does not test minimal dynamic order, excitation on the study's schedules, or all possible algebraic cancellations.

The public functional tests yielded exactly the same per-requirement status in 23 of 38 pairs. Fifteen pairs disagreed on at least one. Fitted coefficients and named roles alone do not establish that an active state is a physical glucose or insulin compartment.

The historical T2-hard panel supports behavioral comparison of eight paired arms across six contrasts and both generated targets. All 96 comparisons had two numerically scorable target curves. For each seed, we subtract its own matched control. The pair distance is the squared difference between these two response curves divided by the reference response energy. For the Full-prompt model with critic and verifier, this distance for glucose was 0.367 on the 48 g meal and 206.45 on the short high fasting pulse. This distance answers a different question from the two separate seed-to-reference RSEs.

The fixed-cohort learning plot tracks exactly the same models throughout each window. On the six-visit window the Full-prompt method had both fit and functional scores for seven of 18 runs at every visit. Their median validation NMSE declined from 0.762 to 0.397. Their mean functional pass fraction went from 0.786 to 0.643. At 15 visits, neither prompt variant had any fully assessed run over the entire window. Available-case late-visit functional scores cannot support a conclusion about improvement of the originally planned cohort.

## Canonical structure transfer under bounded fitting

The original four structural problems had three optimizer starts each. All 12 fitted-record Jacobians could be recovered at the exact saved parameter vector. The parameter table reports each named coefficient, its physical scale, fitted value, bounds and bound-hit flag. The conditioning table reports singular values, numerical rank and condition number for the training residual Jacobian in log-parameter units. They describe local sensitivity at a fitted point and depend on the parameter and residual scaling.

In the original perturbed T1 fit, every start had seven of nine coefficients on a bound. We sealed a separate development sensitivity before refitting. The canonical equations, causal initializer, public training and validation schedules, exact Jacobians, two/four-substep integration check, target normalization and 50-evaluation limit were preserved. Eligible scale multipliers spanned one quarter to four times the nominal physical scale. The gastric maximum and minimum rate boxes meet at their geometric midpoint to preserve their ordering. The upper shape limit remains 0.98. The effective absorption gain keeps its original upper limit. We tested the three original initializations and three seeded points distributed within the wider box. Both groups are reported.

| Perturbed T1-easy | Validation-selected NMSE | Bound hits at selected fit | Converged starts |
|---|---:|---:|---:|
| Original log bounds of ±0.5, subject to the shape cap | 0.865106 | 7 / 9 | 3 / 3 |
| Expanded bounds, original starts | 0.295641 | 3 / 9 | 3 / 3 |
| Expanded bounds, distributed starts | 0.295642 | 3 / 9 | 3 / 3 |

The model family fits substantially better when several bounds are relaxed. Each wider fit still reached the lower bounds of the tissue-to-plasma exchange rate, minimum gastric emptying rate and effective appearance gain. The difference between 0.865 and 0.296 quantifies sensitivity to this parameterization. An unconstrained structural error floor remains unknown. This retrospective sensitivity used training data for fitting and validation for within-profile selection. No confirmation schedule entered it.

## Reproduction

Run from the project checkout with the original validation archive and four-case model inventory, the prior local analyst results, the Phase C public development release and the saved fidelity subject trajectories. Missing fit Jacobian caches produce explicitly unavailable conditioning diagnostics. Rebuilding all legacy conditioning diagnostics requires the original objective caches from `output/iclr-2027-new-controls/reference-final/fits/`. The checksum-identified outputs are under `output/iclr-local-completion/historical/`.

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m scripts.iclr_transfer_sensitivity
.venv/bin/python -m scripts.iclr_historical_followup
.venv/bin/python -m pytest -q tests/test_iclr_historical_followup.py
~~~

Use `historical_per_seed.csv` with `cohort_registry.csv`, `manuscript_corrections.csv` and `claims_status.csv` for the reporting audit. `visit_structure_edits.csv` records all 2,400 planned visits, including unavailable fields. `worked_lineage.csv` retains both the trial and the incumbent. `paired_active_stability.csv`, `seed_behavior.csv`, and `fixed_cohort_learning.csv` contain the scientific comparisons. `reference_parameters.csv`, `canonical_structure_transfer_and_conditioning.csv`, `transfer_plan.json` and the individual fit receipts identify the reference experiments.
