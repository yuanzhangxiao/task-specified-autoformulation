# Judge calibration analysis

## Question and protocol

This milestone asks whether the historical LLM judge is a sufficiently
reliable measurement instrument, whether its six categories should be
reweighted, and whether such a reweighting should replace the production
model-selection rule.

The analysis uses the frozen 336-call adversarial study: 28 matched candidate
pairs, two judge families, and three independent repetitions per candidate.
Each pair contains a valid candidate and a deterministically mutated candidate.
No trajectory test metric, hidden trajectory, or private simulator reference
enters a judge call or a calibration feature.

The primary analysis excludes `narrative_equation_mismatch`. That mutation
changes candidate prose but does not deterministically change its equations,
so treating it as a known wrong-dynamics label would confound narrative
consistency with structural correctness. Results over all seven mutation types
remain available as sensitivity analysis.

Calibration is leakage-safe at the benchmark level. A standardized
ridge-logistic model is fitted on the six category scores from three benchmarks
and evaluated on the held-out fourth benchmark. This is repeated for every
benchmark and separately for each judge family.

## Primary results

The primary dynamics-only set contains 24 matched pairs per judge family.

| Judge | Historical pair accuracy | 95% bootstrap CI | Historical AUROC | Calibrated pair accuracy | Calibrated AUROC | Historical Brier | Calibrated Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.6 Flash | 91.7% | 79.2–100% | 0.966 | 91.7% | 0.917 | 0.246 | 0.088 |
| GPT-5.6 Terra | 100.0% | 100–100% | 0.979 | 100.0% | 0.986 | 0.154 | 0.041 |

Calibration sharply reduces Brier error, meaning its output is a better
probability of distinguishing these synthetic defects. It does not improve
pairwise selection accuracy. For Gemini, held-out AUROC decreases. Therefore
the calibrated coefficients should not be interpreted as universally valid
model-selection weights.

The simple mean of mechanism/state adequacy and parsimony/interpretability also
does not improve pair accuracy: it obtains 91.7% for Gemini and 100% for GPT.
Mechanism/state adequacy is the strongest scientifically relevant individual
category (91.7% and 100% pair accuracy). Constraint compliance is also highly
discriminative, but it substantially overlaps deterministic validation.
Data/causal consistency is weak as currently elicited, with pair accuracy of
58.3% for Gemini and 87.5% for GPT.

## Reliability

| Judge | Repeat ICC(1,1) | Mean within-candidate SD |
|---|---:|---:|
| Gemini 3.6 Flash | 0.975 | 0.0167 |
| GPT-5.6 Terra | 0.941 | 0.0249 |

The cross-family Spearman correlation of aggregate scores is 0.810. Thus the
instrument is repeatable and broadly consistent across families, while still
showing material provider dependence.

The hardest genuine dynamics mutations are wrong regulator signs and wrong
mediator/target coupling. The prose-only mutation produces near-zero margins
and must not be cited as evidence of dynamics sensitivity.

## Decision

1. Keep validation NMSE as the production selection criterion. The frozen
   selector confirmation found no multidimensional benefit from increasing the
   historical judge's selection influence, and this calibration study supplies
   no contrary evidence.
2. Do not deploy the fitted logistic coefficients as selector weights. They
   calibrate a small synthetic discrimination task, not the full vector of
   predictive accuracy, structural recovery, hidden-state accuracy, and
   parsimony.
3. For a prospective method version, remove deterministically checkable
   syntax, closure, availability, mapping, explicit constraints, and raw term
   count from the LLM's responsibility.
4. Retain a narrower advisory rubric for mechanistic coherence, plausible
   latent-state roles, unsupported scientific shortcuts, and whether added
   complexity has a scientific justification. Report its categories rather
   than presenting the aggregate as structural recovery.
5. Before giving that narrower score greater selection influence, validate it
   against blinded expert ratings and a larger, correctly labeled adversarial
   set. Include more sign and mediator defects, intervention-level defects, and
   hard negatives with plausible prose but incorrect equations.

## Reproduction

```bash
python scripts/analyze_judge_calibration.py \
  --scores artifacts/rebuttal/analysis/adversarial/adversarial_judge_scores.csv \
  --output-root artifacts/rebuttal/judge_calibration_v1
```

The generated directory contains fixed-score metrics, leave-one-benchmark-out
metrics and coefficients, repeat reliability, mutation sensitivity, a manifest,
and a Markdown report.

## Limitations

- There are only four benchmarks and 24 primary matched pairs per judge.
- The labels are deterministic mutations, not blinded expert judgments.
- Pairwise discrimination is not the same estimand as scientific model quality
  or downstream model-selection utility.
- The fixed historical rubric cannot be retroactively changed without new LLM
  calls; this analysis supports design of a future tagged method version.
