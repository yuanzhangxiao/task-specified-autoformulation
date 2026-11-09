# What the learned hidden states contributed

On the 48 g meal intervention, one of the two fully enabled historical T2-hard models needed its evolving hidden states to produce its glucose response. Its relative squared response error was **0.053** with the model intact and **1.000** when both hidden states followed their fasting trajectories. The other seed achieved **0.481** in both conditions. We compared the same frozen equations, coefficients and initial conditions within each seed. A zero-change response scores one. [Paired glucose trajectories](local/clamp/paired-glucose-clamps.pdf) · [Every clamp and numerical status](local/clamp/latent_clamp.csv)

## Independent responses

| T2-hard, Full prompt with critic and verifier | Seed 0 | Seed 1 |
|---|---:|---:|
| Meal glucose response error, intact → all hidden states clamped | 0.481 → 0.481 | 0.053 → 1.000 |
| Short fasting insulin pulse, glucose response error, intact → clamped | 199.889 → 199.889 | 1.105 → 1.000 |
| Pulse glucose peak change in the intact model, mg/kg | +35.585 at 300 min | −3.863 at 300 min |
| Pulse generated-insulin response error | 3.488 | 1.819 |
| Pulse insulin-dependent disposal response error, scalar train-calibrated probe | 0.999 | 1.637 |

The simulator reached its glucose peak change of −2.839 mg/kg at minute 148. In seed 0, the fitted glucose equation adds a positive insulin × mediator term. Current insulin multiplying the mediator's pre-existing value accounts for +35.585 mg/kg in the integrated glucose response. The model has an additional state named X that no longer feeds a target. In seed 1, holding just the delayed insulin state at its fasting trajectory erased the pulse response. Holding just the other latent state preserved it. Seed 1 learned an input-dependent dynamical pathway, although the timing and magnitude of that response were poor. These are interventions on the model's own equations. Direct physical correspondence between the learned states and tissue disposal remains unverified. [Equation attribution and decoder controls](docs/ICLR_2027_LATENT_REVIEW.md) · [All physical readout decisions](local/latent/physical_readout_review.csv)

The retrospective cohort comprises 16 learned fits from two algorithm seeds, both prompt variants, and four critic/verifier configurations. All 16 beat zero change for meal glucose; none beat it for short high-pulse glucose. Under the Full prompt, seed 1's meal glucose error was 0.053 with both components enabled, 0.442 with the critic removed, 0.493 with the verifier removed and 0.445 with both removed. Seed 0 gave 0.481, 0.403, 0.464 and 0.442 in the same order. These paired results show the seed dependence of the component comparison. The full tables include both generated targets, all six interventions, the remaining configurations, and the two distinct development-matched pairs. [Target effects](local/latent/target_effects.csv) · [Matched pairs](local/latent/matched_pairs.csv)

**Numerical checks.** We simultaneously integrated a control model, the intact treated model and the treated model with chosen hidden states constrained to the control trajectory. Both models received only permitted public inputs and initial readings. The control states evolved continuously on the same solver steps as the treatment, including all forcing corners. We ran Radau and BDF, required agreement on the paired effects and reproduced the original intact target rollouts. All 80 target/clamp records passed across the 32 model/intervention jobs. For confirmation, the same evaluator is prepared for both hard anchors, Full and fixed-state, three seeds per arm. It accepts the separate frozen confirmation roster.

## Earlier T1 state ablation

In the separate round-12 named T1-easy cohort, seed 0's glucose peak came at minute 122 with latent dynamics, minute 61 without them, and minute 129 in the simulator. Split-meal response RSE was 0.277 for Full and 1.735 for the no-latent fit. This is the one matched named no-latent seed with a retained model. In the conditional equation assay we fixed plasma glucose and the supplied auxiliaries at their basal values and measured the meal-driven glucose flux. The two named Full seeds produced only 14.2% and 15.5% of the reference area (medians across the tested meals). The isolated flux result limits the interpretation of the accurate peak timing. [Round-12 free responses](local/earlier-t1/paired_responses.csv) · [Conditional meal flux](local/earlier-t1/meal_flux.csv)

## Structure, provenance and reference fits

At visit six we could recover both fitted structures for 38 of 80 planned historical seed pairs. Fifteen pairs had the same target-reachable state count and 17 had the same public input-to-target reachability. No pair had identical full signed equation topology. The per-seed files retain all 160 planned lineages. Only seven Full-prompt lineages for the fully enabled treatment had joint fit and functional scores at every one of the first six visits; their median validation NMSE fell from 0.762 to 0.397 and their mean functional fraction fell from 0.786 to 0.643. [Fixed-cohort curves and historical scope](docs/ICLR_2027_HISTORICAL_FOLLOWUP.md) · [Manuscript corrections](local/historical/manuscript_corrections.csv)

Seven of nine coordinates reached a bound in each of the three original perturbed-T1 reference fits. Widening the coefficient limits with the same canonical equations and 50-evaluation budget brought validation NMSE from 0.8651 to 0.2956. All six sensitivity starts reached approximately the same result. This is a development-only, privileged fit. [Parameter values and bounds](local/historical/reference_parameters.csv) · [Transfer diagnostics](local/historical/canonical_structure_transfer_and_conditioning.csv)

## Remaining experiments

The live development pilot and the 48 anchor, 114 additional stress, and 16 corrected historical CSTR search endpoints require ACES. Numerical failures and unavailable models remain in every denominator. The 126-row stress report reuses 12 anchor endpoints. Corrected historical CSTR will be scored after freeze on its corrected held-out schedules and separately with the same functional rubric across all 16 endpoints. The original nine-case post-pruning freeze has not been recovered locally. The prospective fixed-state comparison, automatic reaction to a wrong specification and discovery scaling remain unmeasured. [Exact ACES commands](docs/ICLR_2027_ACES_EXECUTION.md) · [Updated draft reviewer replies](docs/ICLR_2027_REVIEWER_RESPONSES.md)

The analyst package includes the prior evidence and candidate trajectories, the new numerical tables, plots and per-fit/solver receipts. The archived base commit is in `source-commit.txt`, and the new analysis commit is in `local/source-commit.txt`. Use the separate ACES deployment archive for search and the original checksummed analyst ZIP specified by the runbook for confirmation scoring.

## Reproduce the hidden-state clamp analysis

From the analysis checkout, set `AF_FIDELITY` to the extracted analyst archive's `fidelity` directory. It contains the matching saved subject trajectories. Use a fresh output directory and the numerical dependencies declared by the repository.

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m scripts.iclr_latent_clamp --fidelity "$AF_FIDELITY" \
  --output "$PWD/output/iclr-clamp-reproduction" --workers 3
.venv/bin/python -m scripts.plot_iclr_latent_clamp \
  --clamp "$PWD/output/iclr-clamp-reproduction" --fidelity "$AF_FIDELITY" \
  --output "$PWD/output/iclr-clamp-reproduction/paired-glucose-clamps.pdf"
~~~
