# New Dalla component models: equation review and frozen replay

The new archive contains worthwhile T1-hard candidates. It does not yet establish a clean, fully automatic, pruned model that is best across interventions. Two different families are worth retaining: very accurate meal-response models with questionable latent initial conditions, and simpler positive meal-memory models with more credible tissue-response coefficients.

This report concerns `final-components-v1`, not the earlier assisted R4/R9/R2 rescue experiments. All rounds below are **zero-based**, exactly as in the archive. In particular, “Full R4” here is a different model from the previous assisted R4.

## Coverage and what was inspected

Source archive SHA-256: `143cb9f9cac50bd43d8cc9231812846fa44c120b56ba1ce82a96d2bbc33e4937`.
Inventory seal: `ba5a433bbd32ff23e43faf07ff2750b1616c17af3be1bac93df4d9f7a292b97b`.
Campaign plan: `cdfbd68d07a88f510a869c595c457dfcdf6595d42cf6fb8983bf10a901d5b950`.
Source root: `/scratch/group/p.nairr260351.000/u.yx126462/final-components-v1`.

Read INDEX.tsv, all four benchmark document structures, and the complete machine-readable inventory. Expanded algebraic processes and substituted saved parameters using the restricted expression grammar for all 1,334 records. Screened mechanism paths, constant derivative coefficients, parameter domains, target mappings and initialization. Manually examined the best-training candidate from each fitted lineage, the leading candidates in each cell, sign/coverage alternatives regardless of NMSE, and all 37 unfitted records without an exactly matching fitted candidate object. This is an equation screening exercise, not 1,334 independent scientific certifications.

| Cell | Fitted records | Unfitted records | Best saved train NMSE | Best saved validation NMSE |
|---|---:|---:|---:|---:|
| T1 easy | 177 | 169 | 0.01960 | 0.01289 |
| T1 hard | 178 | 165 | 0.01216 | 0.01546 |
| T2 easy | 154 | 153 | 0.68463 | 0.73509 |
| T2 hard | 179 | 159 | 0.36828 | 0.39756 |

The two minima in a row need not belong to the same model. Fitted records are not independent restarts; repeated incumbents and closely related vectors remain represented.

Of 1,200 planned visits, 747 are complete, 12 construction_failed, 21 worker_interrupted, and 420 missing. Among lineages with completed fits, 47 critic-enabled lineages stop at round 5 or 6; 31 critic-disabled lineages reach round 14. Two additional lineages have no completed fit. No exported occurrence comes from a pruning result. Thus this snapshot neither supports a fair critic comparison nor demonstrates completed pruning. The 37 unmatched unfitted records mostly concern the next round at the snapshot boundary; absence here does not establish permanent failure.

Flags in task names: c = critic; v = scientific verifier; s = shared processes. “full” is the training-evidence arm, not a guarantee that all three switches are enabled.

## T1-hard shortlist and actual frozen replay

Six candidates were frozen for inspection from their equations and development results before evaluating new curves. Reused the seven previously defined canonical probes without changing the schedules, parameters, equations or initializers. All 16 training and four validation trajectories were replayed for every shortlisted candidate. All 162 rollouts were finite; all 12 pooled development scores reproduced their saved values within 2.45e-15.

The archive omits the original split fingerprints. The locally saved same-cell public data, its fingerprints, and runtime are recorded in replay-plan.json. Score reproduction provides a strong consistency check, not a reconstructed original data hash.

| Candidate | Task | Record type | Train | Validation | Split meal | Initial Gt −20%, with meal |
|---|---|---|---:|---:|---:|---:|
| Brief-only R9 | cell06_seed1_brief_only_c0v1s1 | selected at R9 | .01485 | .01603 | .02154 | .16972 |
| Brief-only R10 | cell06_seed1_brief_only_c0v1s1 | retained through R14 | .01484 | .01546 | .02077 | .16657 |
| Brief-only R14 | cell06_seed1_brief_only_c0v1s1 | trial, not retained | .01216 | .02220 | .01274 | .03445 |
| Brief-only R4 | cell06_seed1_brief_only_c1v1s1 | retained through R5 | .05182 | .07377 | .05077 | .05593 |
| Full R4 | cell06_seed1_full_c1v1s0 | trial; shared processes off | .04676 | .06701 | .04730 | .05634 |
| Full R14 | cell06_seed1_full_c0v1s1 | selected | .01604 | .05844 | .01317 | .05144 |

All entries are absolute-trajectory NMSE using the training scale. Split meal is 30 g at minute 60 plus 30 g at minute 90; its control is 60 g at minute 60. The Gt intervention changes initial **tissue** glucose by 20%, holding initial plasma glucose and other physical states fixed; it is not a mass-conserving redistribution. Gt is supplied as an auxiliary in T1-hard. No observed plasma-glucose samples reset the rollout after initialization.

### Accurate meal-response family: Brief-only R9/R10/R14

R9 has two latent states. Renaming them A and H for readability, its fitted equations are approximately

```
A'  = 0.0751386 u - 0.0385572 A
H'  = 0.0385572 A - 0.0511580 H
Gp' = A - 0.0288195 Gp + 0.0385572 Gt
      + (0.0119285 Gt - 1.49231) H
A(0) = -1.549456; H(0) = 31.177344
```

The positive meal path has explicit memory. The model fits many meal trajectories well. But A begins negative, and the effect of H depends on Gt. The name `insulin` in the original model is not a demonstrated physiological insulin concentration: the state has unspecified units/description and is inferred from meals alone. Its initial Gt sensitivity is much larger than the reference exchange coefficient.

R10 adds a third state with the same driven dynamics as A but an independent initial value. This changes the transient degrees of freedom without adding an independent meal-input timescale. It is the actual retained endpoint, so it should be the main unassisted endpoint in a figure, not silently replaced by the better-performing trial after seeing probes.

R14 adds another delayed effect and improves split-meal absolute NMSE to .01274. Its initial `meal_absorption` is -119.8534; the separate `meal_glucose` starts at 17.8436. It is not a literal nonnegative absorption model. The relative squared error of its response to Gt −20% is 2.84, despite absolute trajectory NMSE .03445. R10's corresponding response error is 12.45. Both can look close on an absolute glucose axis while having the wrong sensitivity to initial tissue glucose.

These models are useful **meal-response demonstrations with explicit caveats**, not evidence of complete physiological recovery. R14 must remain labeled an exploratory trial selected for inspection, with critic disabled. It was not the selected final endpoint.

### Cleaner reduced models: Brief-only R4 and Full R4

Rescale the sole meal state into its contribution B to Gp. This is an exact coordinate change, with no refit or pruning.

Full R4:

```
B'  = 0.0949650 u - 0.00802076 B;  B(0) = 1.509686
Gp' = B - 0.0709602 Gp + 0.0904661 Gt
```

Brief-only R4:

```
B'  = 0.0864003 u - 0.0134820 B;   B(0) = 1.133964
Gp' = B - 0.0459638 Gp + 0.0594458 Gt
```

Both have a positive initialized meal state, positive meal forcing, negative state decay, positive tissue-to-plasma contribution and negative plasma clearance. The meal-memory time constants are 124.7 and 74.2 minutes respectively. For orientation, the reference plasma exchange terms are `+0.079 Gt - 0.065 Gp`; Full R4 is close in these two coefficients. However, the reference also contains production, utilization and excretion. Those are unavailable auxiliaries in T1-hard and are only implicitly approximated here. The models do not recover the full reference equation.

Their relative squared response errors for Gt −20% are .0245 and .0457 respectively, substantially better than the meal-optimized Brief-only models. Their split-meal absolute fits are less accurate, around .05. They are the most defensible starting points for a parsimonious reduced-model illustration, especially if the scientific focus is sensitivity to initial tissue glucose. Their meal response still starts earlier than the reference.

The Full candidate has shared processes disabled and is a fitted trial. Brief-only R4 was selected and retained through R5 in the critic/verifier/shared-processes-enabled lineage. Neither result establishes a final pruned winner. Do not relabel these as the full c1v1s1 pipeline or as the earlier assisted R4.

### Full R14 requires particular caution

Although split-meal NMSE is .01317, `I(0)=-1663.939` and `X(0)=203.000`. The effective tissue coefficient is `.146903 + .0000923564 I`, which is **-.006773 initially**. Thus nonnegative coefficients and apparently positive tissue terms do not guarantee a positive fitted tissue effect. This is not my preferred mechanistic illustration.

## Other cells and signs

T1-easy's best-fitting lineage has a positive tissue coefficient but also negative EGP, positive utilization/excretion, and a negative direct meal contribution. A representative low-error model has

```
M'  = 0.09830 u - 0.01681 M
Gp' = M + 0.1242 Gt - 0.0910 Gp - 1.9493 EGP
      + 2.7763 Uii + 155.773 E - 0.4813 u
```

I found no fitted T1-easy model with all of positive direct Gt and EGP coefficients, negative direct Uii coefficient and negative plasma self coefficient. This is a narrowly stated linear-coefficient screen, not a theorem that every possible reduced model is scientifically invalid. Some higher-error models have correct production/utilization signs but omit tissue return; other models use the wrong tissue direction. Refitting parameters under their present constraints cannot necessarily fix the missing or hard-coded wrong mechanisms.

Across all 688 fitted records, no saved parameter declared positive/nonnegative violated that domain; positive declarations also had no zero values. No explicit finite bounds are present in these candidate declarations. Negative real coefficients and negative latent initializers are allowed by the declarations. This is a final-declaration audit, not a reconstruction of each original topology proposal. Physiological sign errors remain possible through the outer operator, inner expression, freely signed parameters or fitted state values.

T2-easy has no ready whole-model demonstration. Its best total training NMSE is .6846; individual training NMSEs are Gp=.4520, I=.8084, U=.7934. Among 154 fitted records, 140 have no syntactic meal-to-I or Gp-to-I path; 13 have meal-to-I; one has Gp-to-I, but that term is overwhelmed by insulin clearance near 2.34e9 and the total fit is poor. These are descriptive fitted-equation facts. The public task requires delayed insulin action, not a specific secretory equation, so these findings must not become fabricated deterministic prompt requirements.

T2-hard has one more interesting Full c1v1s1 R5 model with meal-driven I, a delayed insulin sink and meal absorption. Its train/validation NMSE is .3683/.3976. But the fitted insulin-memory decay time is about 10.4 million minutes and the initial memory is -18,844, supplying a compensating source through a nominal sink. It is a research/repair lead, not a ready physiological example. The R6 successor was unfitted in the snapshot.

## Recommended next action

1. Keep the retained Brief-only R10 and its R14 trial as the strongest new meal-curve leads, with the trial/critic/initialization limitations visible.
2. Keep the two one-state R4 models for a cleaner equation-based interpretation and initial-tissue-glucose response comparison. Do not claim they beat every external baseline.
3. Retrieve a fresh export when the critic-enabled lineages and pruning finish. The current archive stops those lineages much earlier than the critic-disabled ones. In particular retrieve `cell06_seed1_brief_only_c1v1s1` and `cell06_seed1_full_c1v1s0`, plus the fully enabled Full lineages.
4. Do not start a broad T2 refitting sweep for the paper figure from this evidence. The presently exported T2 limitations are structural and initial-state issues as well as optimization issues.
5. Existing frozen probes are exploratory; choosing a candidate after viewing them is not an independent benchmark test. Any subsequent repair must be reported as a new diagnostic experiment, with parameters fixed before its final intervention evaluation.

## Files and validation

All generated files are local and uncommitted under `artifacts/dalla-all-models-review-2026-09-25/`:

- `screen.json`: all 1,334 descriptive equation screens.
- `unfitted-without-fitted-counterpart.json`: the 37 unmatched proposals.
- `shortlisted-models.json`: six exact candidates, fitted vectors, initializers and source occurrences.
- `shortlist-curves.csv`, `shortlist-probes.csv`: trajectories and metrics for plotting.
- `shortlist-trajectories.png/.svg`: comparison of training, validation and interventions.
- `brief-r10-all-train.png`, `brief-r10-all-validation.png`: every development trajectory for the retained endpoint.
- `replay-plan.json`, `replay-summary.json`, `verification.json`: sealed replay identity, results and independent numerical checks.

A mistaken human-readable R14 alias for the Full R4 candidate was corrected to the source-recorded R4. Superseded seals are retained locally; equations, parameters and cached rollouts were unchanged. The deliverable uses the corrected labels.

No fitting, pruning, live LLM calls, remote sessions or test-data access occurred. Runtime implementation was not changed. All 162 rollouts passed independent metric checks, and deterministic cache replay reproduced the results. Focused regression command `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/test_collect_component_models.py tests/test_t1_external_interventions.py`: **20 passed in 1.28 s**. Full repository Ruff still reports 37 existing issues under `analysis/claude`; those unrelated files were not modified.
