# Evidence audit for the interventional-discrimination subsection

The proposed subsection is `docs/INTERVENTIONAL_DISCRIMINATION.tex`.
This audit recomputes the supplied draft's comparisons from the locally available
round-12 endpoint artifacts. It does not submit experiments, open held-out test
observations, or infer results for ablations that were not run.

## Sources and endpoint rule

- Primary local report: `analysis/claude/data/rd4_summary.json`.
  SHA-256: `16514495a2daebbade51de1bbe4872a6e16445953260463efdf2557dcdd37089`.
- Cross-check: `analysis/claude/data/rd4_rounds.csv`.
  SHA-256: `d83e7758407d3813282270c1ed7368cadb3e98141beb9a7ea6e79215e362900b`.
- Campaign plan: `67108d6fe6b9b70b9d0a9e0e30a0aa6fbc39cccd21e3d8a8174ebeb25d6b0cf1`.
- These two artifacts agree exactly on all 176 rows' retained train and
  validation scores. Select global round 12, use retained rather than trial
  scores, and pair on exact cell and seed. Never choose the most favorable round.
- The report declares `test_metrics_used: false`. It contains rounds 9--12,
  including an imported round-9 checkpoint followed by three additional visits.
  Earlier phases changed the revision interface. These are continued-development
  endpoints, not a pristine experiment with one unchanged protocol throughout.
- `configs/review_deadline_v2.json` and `docs/REVIEW_DEADLINE.md` establish the
  six-cell roster and limited ablation assignments.
- `src/autoformalism/rebuttal/review_deadline_pipeline.py`, functions
  `_brief` and `certificates`, specifies the experimental overlays.
  Prediction-only (`no_spec`) withholds scientific context, requirements and
  target dependencies while retaining permitted channels and allowing latent
  states. Withheld scientific predicates do not gate its selection. The
  no-latent arm retains the scientific specification but restricts differential
  states to directly observed targets.
- `docs/PHASE_B_EXACT_BENCHMARK_PROTOCOL.md` and the schedule definitions in
  `src/autoformalism/benchmarks/phase_b_generation.py` distinguish training and
  validation input/initial-condition protocols. No private equations or test
  observations are needed for the comparisons here.

The local report is available evidence, not an independent rerun of its saved
models. Its input files are existing local experiment artifacts and are not
added to this documentation commit.

## Coverage across the six cells

Each cell has two planned seeds. Entries below count finite retained round-12
endpoints. A dash means the arm was not assigned, not that it failed.

| Cell | Full | Brief-only | Refit-only | Prediction-only | No-latent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dalla Man, canonical, named | 2 | 1 | 2 | -- | 1 |
| Dalla Man, canonical, obfuscated | 2 | 2 | 2 | 2 | -- |
| Dalla Man, perturbed, named | 2 | 2 | 2 | -- | -- |
| Dalla Man, perturbed, obfuscated | 2 | 2 | 2 | -- | -- |
| CSTR, canonical, named | 2 | 2 | 2 | -- | 2 |
| Alien device, functional | 2 | 2 | 2 | 2 | -- |

These are six cells from three generating families, not six independent systems.
The missing named-canonical seed-1 brief-only and no-latent endpoints are closed
lineages. No prediction-only result exists in this roster for CSTR or the other
three Dalla Man cells.

## Specification ablation: every available matched endpoint

Here `g = validation NMSE / training NMSE`, and `g ratio` is prediction-only
divided by full. Values in the table are rounded only for display.

| Cell | Seed | Full train | Full validation | Prediction-only train | Prediction-only validation | g ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical obfuscated Dalla Man | 0 | 0.148121 | 0.274970 | 0.043576 | 0.124052 | 1.533518 |
| Canonical obfuscated Dalla Man | 1 | 0.157364 | 0.239654 | 0.171618 | 0.337849 | 1.292651 |
| Alien device | 0 | 0.943496 | 0.926122 | 0.671737 | 0.661337 | 1.002986 |
| Alien device | 1 | 0.772779 | 0.627191 | 0.472221 | 0.655052 | 1.709172 |

Findings:

- The original 4/4 direction and median g ratio 1.4130848608 are arithmetically
  correct. One change is only 0.30% (alien seed 0).
- Prediction-only has worse absolute validation NMSE in 2/4 pairs, not 4/4.
  It is better on both training and validation in the other two pairs.
- The only specification-ablation training/validation rank reversal is alien
  seed 1: prediction-only has 38.89% lower training NMSE but 4.44% higher
  validation NMSE (absolute difference 0.0278607).
- Canonical obfuscated seed 1 has relatively close training scores (prediction-
  only is 9.06% worse) and 40.97% worse validation. This is a validation gap, but
  not a ranking reversal or equal-training-fit comparison.
- None of the four prediction-only endpoints exhibits catastrophic or above-one
  validation NMSE. The available data do not establish the desired clean example
  of low training error followed by very high interventional validation error.

## Latent-state ablation: distinct scientific question

| Cell | Seed | Full train | Full validation | No-latent train | No-latent validation | Validation ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical named Dalla Man | 0 | 0.333330 | 0.263820 | 0.457120 | 0.445399 | 1.688269 |
| CSTR | 0 | 0.132255 | 0.014008 | 0.122933 | 0.031975 | 2.282557 |
| CSTR | 1 | 0.159257 | 0.004177 | 0.155628 | 0.003326 | 0.796286 |

At CSTR seed 0, no-latent has 7.05% lower training NMSE and 128.26% higher
validation NMSE. Both validation errors remain low in absolute terms; this is
not a catastrophic failure. At seed 1, no-latent is better on both metrics.
CSTR's public graph check is reported as unresolved for both arms, not certified
scientific success. This example concerns the hypothesis class and cannot be
used as evidence that withholding the specification caused the difference.

## Why the draft's ratio argument was removed

For an ablation A and full method F,

`g_A / g_F = (validation_A / validation_F) / (training_A / training_F)`.

Pairing holds the benchmark cell and random-seed label fixed. Dividing by
training error does not hold fit quality fixed or isolate a causal effect of the
specification. A larger ratio can result from much better training fit even when
validation improves. The canonical obfuscated seed-0 comparison is an actual
example. Ratios are also unstable near zero training error.

The other ratios in the supplied draft also reproduce:

| Arm versus full | Finite pairs | g ratio greater than one | Median g ratio |
| --- | ---: | ---: | ---: |
| Prediction-only | 4 | 4 | 1.413085 |
| No-latent | 3 | 2 | 1.231080 |
| Brief-only | 11 | 4 | 0.692690 |
| Refit-only | 12 | 6 | 0.859444 |

They are descriptive statistics, not evidence that brief-only or refit-only
cannot change intervention responses. Their absolute validation scores do
change. Refit-only can change fitted coefficients, and thus input response,
without changing the equations. Brief-only can change the proposed model.
Different arm coverage also prevents treating these aggregate ratios as a
matched comparison among all four ablations.

## What the intervention language supports

- Training and validation use 16 and 4 trajectories, respectively. Validation
  contains unseen interpolating conditions, not necessarily out-of-range
  extrapolations. Dalla Man also includes an initial-condition validation case,
  so it is inaccurate to attribute every validation difference only to schedules.
- Examples of held-from-training validation schedules are a 75-g meal at minute
  30 in Dalla Man; intermediate-amplitude steps, pulses and sinusoids in CSTR;
  and new pulse durations/amplitudes and sinusoidal periods in the alien device.
- A dynamical model can simulate a permitted external-input intervention by
  changing the input path, fixing learned equations/parameters and evolving its
  state from permitted initial information. Later target observations cannot
  reset its state. Causal initial maps evaluate new initial observations without
  fitting new validation-specific parameters.
- Where auxiliary trajectories are supplied, the prediction is conditional on
  those trajectories. This does not establish autonomous intervention behavior
  for every subsystem or identify arbitrary interventions on latent variables.
- Scientific requirements constrain the intended model and guide search. They
  do not by themselves prove the truth of the equations or unique causal
  identification. Prediction-only models can also simulate input changes.
- Validation selects retained candidates, so these scores are development
  evidence. They are held out from numerical parameter estimation, not from the
  complete model-selection process. No independent test conclusion is claimed.

## Recommended interpretation and next evidence

Use the benchmark-only subsection as a measured motivation: training accuracy
does not determine response accuracy under changed input conditions. Keep the
constructed equal-driver example separately if an exact non-identifiability
demonstration is needed; it supplies a known mechanism that the benchmarks do
not establish. Its near-zero-error contrast cannot be presented as the observed
effect size of the benchmark ablation.

To test a general specification advantage, first extend prediction-only to all
six cells under a fixed protocol and matched budgets, reporting every endpoint
and failure. Predefine comparisons based on absolute validation loss and paired
training/validation changes rather than selecting favorable examples or ratio
thresholds afterward. For an independent interventional claim, freeze selection
before evaluating a separate intervention set. No new experiment is launched by
this documentation change.
