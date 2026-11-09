# T2 independent starts reproduce the insulin timing and recovery errors

The short- and long-memory fits of `c15-3` found the same early insulin peaks. The shallow- and steep-sigmoid fits of `c15-2` found the same late insulin rebound. The discrepancies are ready to use as numerical feedback for construction.

## Four independent fits

The 16 public training trajectories provide glucose, insulin and disposal. Each pair shared the archived parent's common starting coefficients and two training initial values per trajectory. One starting coordinate differed per pair. Each run then optimized all free coordinates, including the shape coordinate and all 32 training initials. Each optimizer had access only to its own start and its own verified checkpoints.

For meal memory, the initial time constants were 10 and 120 min. For glucose-dependent secretion, the initial slopes were 0.25 and 4 divided by the pooled training glucose standard deviation of 49.916 mg/kg. Both sigmoid midpoints started at the median training glucose of 172.689 mg/kg and could move during fitting.

| Child and start | Starting NMSE | NMSE after the first 2,000 evaluations | Final training NMSE | Final shape | Evaluations used |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c15-3`, 10 min | 3.46622 | 0.11673206 | 0.116730861 | 63.088 min | 2,634 |
| `c15-3`, 120 min | 2.55166 | 0.11673112 | 0.116730894 | 63.218 min | 3,517 |
| `c15-2`, slope 0.0050084 | 12.97257 | 0.12940935 | 0.129409330 | slope 0.0196341 | 2,970 |
| `c15-2`, slope 0.0801345 | 11.34912 | 0.12940934 | 0.129409333 | slope 0.0196340 | 2,984 |

The first two blocks allowed 1,000 whole-training evaluations each, with the prior campaign's pooled NMSE objective followed by the sampled-domain penalty. Two more blocks allowed up to 2,000 evaluations each with the same penalty. Each final block reached the solver's relative objective-change tolerance. The table gives unpenalized NMSE. For all four runs the lowest-error checkpoint also passed the dense check on all three outputs. Original-equation Radau and DOP853 replays agreed on the measured grids, and resuming the four terminal runs reconstructed identical results.

Across all targets and training times, the root mean squared difference between the two fitted predictions was 0.000317 pooled target standard deviations for `c15-3` and 0.0000175 for `c15-2`. The sigmoid midpoints finished at 306.842 and 306.844 mg/kg. In `c15-3`, individual gains differed. Since $M(0)=0$ and $\dot M=k_r r-M/\tau$, the coefficient multiplying the meal-memory convolution in the insulin equation is the product of the memory input gain, the memory-to-production gain and the outer production gain. This product was 0.125727 and 0.125749.

## Trajectory discrepancies to send to the proposer

| Training observation | `c15-3`, short / long starts | `c15-2`, shallow / steep starts |
| --- | --- | --- |
| Insulin peak after each shifted 90 g meal. Observed absolute minutes 101, 161, 221 | Both 68, 128, 187 | Both 87, 154, 211 |
| Insulin at minute 300 after the 150 g meal. Observed 22.62 pmol/L | 29.29 / 29.33 pmol/L | 92.35 / 92.35 pmol/L |

- **Meal memory.** The meals begin at minutes 0, 60 and 120. Both fits predict insulin peaks 67–68 minutes after meal onset. The observed delay is 101 minutes on all three schedules. A revised candidate should explain the additional 33–34 minutes of delay while retaining the late recovery already captured by this family.
- **Sigmoid secretion.** Both fits put the shifted peaks within 7–14 minutes of the observations. After the 150 g meal, insulin rises by 60.67–60.68 pmol/L between minutes 240 and 300 in the fitted models. The measured increase is 0.91 pmol/L. A revised candidate should account for that late balance of production and removal while preserving the shifted-meal timing.

These are reproducible training residuals across the two tested starts per model and the earlier fits. The evidence supports giving these specific discrepancies to construction. The global optimum, unique recovery of every coefficient, and forecast accuracy with eight-minute prefix initialization remain open.

[Trajectory figure](../../output/t2-breadth-independent-starts-20261109/insulin-trajectories.png) · [All four fitted equations and parameters](../../output/t2-breadth-independent-starts-20261109/fitted-models.json) · [Machine-readable comparison](../../output/t2-breadth-independent-starts-20261109/comparison.json) · [Portable review package](../../output/t2-breadth-independent-starts-20261109/t2-independent-starts-review.tar.gz)

The subsequent equation audit traces the early peaks to meal-driven secretion
and the late rebound to excess glucose supply, and checks the paired infusion.
[Term-level feedback and matched infusion controls](2026-11-09-t2-insulin-equation-attribution.md)

## Reproducing the numerical run

The completed run used two local CPU workers and the same frozen fitting backend as the matched November 8 campaign. From the project checkout with the original handoff and numerical snapshot available, run the following commands. An existing result directory resumes its recorded operations.

```bash
export PYTHONPATH="$PWD/output/t2-breadth-fitting-20261108/verified-source/src:$PWD"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python scripts/run_t2_breadth_multistart.py \
  --archive output/t2-breadth-results-20261108/orion-fitting-handoff.tar.gz \
  --root output/t2-breadth-independent-starts-20261109 --workers 2
```

The frozen plan is `31a962352ba61c98299fe7b589f8dfb7a9f800f9f55491c6e66b04070b640b14`. It records the complete starting vectors, training identity, code identity, schedule and selection split.
