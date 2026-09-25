# R4 sign diagnostic: constraints held, but limited intervention recovery

## Finding

The assisted sign constraints were respected. They did not produce a robust
mechanistic demonstration in this run. The repaired model improves the change
caused by splitting a meal, but exaggerates responses to initial tissue-glucose
perturbations and predicts the wrong direction for the fasting +20% case.

The source is the user's `Downloads/models (1).json`, exported from
`dalla-sign-diagnostic-r4-v1`. This is **T1 perturbed named easy, seed 1**, not
the canonical cell of the earlier Sol/no-specification comparison. Both arms
receive the auxiliaries permitted by this easy task. No new fitting, model
selection, proposer call, test split, or reference generation was performed.

## Verification and numerical outcomes

Both result seals, requests, lowered candidates, initialization plans, parameter
inventories, and train/validation content identities were checked. Independent
free rollouts of all 16 training and four validation trajectories per model
reproduce exported aggregate NMSEs to within 4.1e-16. All 14 intervention
rollouts completed. Repeating the inspection preserved all 118 existing JSON
files byte-for-byte. The seven exploratory cases and perturbed reference arrays
were imported unchanged from the previous rescue inspection.

| Endpoint | Training NMSE | Validation NMSE |
| --- | ---: | ---: |
| Assisted sign-constrained R4 | 0.107146 | 0.081134 |
| Unchanged-sign control, refitted/pruned | 0.070776 | 0.068110 |

The repaired model's training trajectory errors range from 0.0273 to 0.5014;
its four validation trajectory errors range from 0.0665 to 0.1193. A pooled
NMSE does not establish that every training curve matches well.

## Fitted equations and pruning

Rounded repaired equations are:

```text
X'  = 0.000845599 meal - 0.000691988 X
Y'  = 30.288665 X - 0.158843 Y
Gp' = 5603.109 Y + 1.99063e-15 EGP - 4.28605e-10 Uii
      + 3759.737 Gt - 8402.397 Gp
X(0) = 0.942878; Y(0) = 164.620720
```

Gp uses its permitted observed initial value. Latent initials remain shared
training-fitted values. The saved zeros in base-candidate placeholders are not
the actual lowered initial values used in the rollouts.

Four surviving fixed gains pass sign/domain checks. The fifth, excretion, was
pruned along with disconnected state M. E is identically zero in these training
and validation data; dropping it is a local simplification, not evidence that
excretion is physiologically unnecessary. No relevant sign constraint was
violated. Gt was intentionally left unrestricted; its negative stored parameter
multiplies an explicit minus, giving the positive effective coefficient above.

Production and utilization coefficients are nearly zero. Their use is optional
under this public task, so this does not violate the graph requirement for a
meal-to-Gp path. Correct signs alone do not ensure scientifically identified
contributions or time constants.

The fitted Gp relaxation time is 0.000119 minutes (0.00714 seconds), compared
with one-minute observations. Apart from the initial fast transient, its equation
therefore approximates `Gp = 0.666847 Y + 0.447460 Gt`. This is an inference
from the fitted linear equations, not a claim that the model is exactly
algebraic. The X and Y time constants are about 1445 and 6.30 minutes. The
near-zero EGP/Uii terms and rapid Gp response explain why passing the sign
check alone is insufficient for the desired demonstration.

The unchanged control retains a negative production coefficient (-865.73 EGP)
and a positive effective utilization coefficient (+13266.78 Uii). Uii is
constant at 1 in training, so this term can supply an offset rather than identify
a utilization response. Better aggregate validation error does not resolve that
mechanistic issue.

## All seven exploratory probes

These change initial **tissue** glucose Gt or meal spacing, not an externally
specified insulin infusion. Each physical reference supplies its own consistent
auxiliary trajectories to the learned models. Parameters and latent initialization
rules are frozen. The reference is the perturbed variant throughout.

| Probe | Repaired absolute NMSE | Control absolute NMSE | Repaired response error | Control response error |
| --- | ---: | ---: | ---: | ---: |
| Fasting, baseline Gt | 0.060107 | 0.038268 | — | — |
| Fasting, Gt -20% | 0.134272 | 0.079442 | 84.356 | 33.982 |
| Fasting, Gt +20% | 0.066180 | 0.025264 | 39.479 | 18.486 |
| One 60 g meal, baseline Gt | 0.074636 | 0.086737 | — | — |
| One meal, Gt -20% | 0.086972 | 0.089722 | 5.246 | 2.056 |
| One meal, Gt +20% | 0.114946 | 0.103213 | 6.973 | 3.310 |
| Two 30 g meals, 30 minutes apart | 0.058706 | 0.060421 | 0.222 | 0.388 |

Response error is the squared error of the change from its matched control,
divided by the squared reference change. It differs from ordinary NMSE; a model
predicting zero change scores 1. Small reference changes can amplify this ratio,
so also inspect physical amplitudes and absolute curves. For fasting Gt -20% and
+20%, the reference peak changes have magnitudes 1.71 and 1.79 mg/kg, whereas the
repaired model's peaks are 11.42 and 11.85 mg/kg. In the +20% case the reference
decreases while the model increases. For split meals, peak change magnitudes are
19.64 (reference), 17.24 (repaired), and 25.64 mg/kg (control).

This is a partial success for the meal-spacing response, not a demonstration
that sign repair/pruning yields uniformly better intervention behavior. Pruning
mostly removed inactive/disconnected structure; it did not create a substantial
predictive improvement. The corrected run also changes gain starts when domains
change, so this paired experiment does not isolate sign constraints from optimizer
initialization effects or establish the best achievable fit of either structure.

## Numerical health and reporting correction

Some initializer stages reached iteration or wall-clock limits. Later fitting
stages nevertheless yielded finite rollouts. The repaired pruning initializers
report convergence, but the retained public result still records
`native_optimizer_converged: null`. Neither `status: complete` nor
`budget_exhausted: false` certifies global or final-optimizer convergence.

The diagnostic metadata at `d28c408` incorrectly said pruning selected on
training. The implementation actually fits parameters on training, retains
seed/refit using training error, ranks removal candidates using training
contributions, and **accepts pruning using validation NMSE** against the paired
baseline. This was the existing frozen policy; no algorithm was changed here.
The metadata and documentation are corrected for future runs, while the uploaded
frozen result is preserved verbatim. No rerun is needed for this correction.

After the prose correction, 17 diagnostic tests and the synthetic diagnostic
smoke were run. Changed code passes Ruff; repository-wide Ruff retains 37
unrelated issues in `analysis/claude`. The larger test suite was not rerun.

## Deliverables and next step

Local outputs are under `artifacts/dalla-sign-diagnostic-inspection-2026-09-25/`:
`summary.json`, `curves.csv`, `intervention-comparison.png`/`.pdf`, sealed replay
files, input copy and the inspection script. The figure displays both fasting
Gt perturbations and meal spacing; the table above reports every probe.

Keep the queued automatic GPU review intact and compare its decisions after it
finishes. These results do not justify another identical refit by themselves.
Any later revision should address the fitted response/time-scale behavior using
permitted evidence, without converting inspected exploratory probe outcomes
into claims about untouched test generalization. No existing experiment, prompt,
benchmark data or scheduler record was modified.
