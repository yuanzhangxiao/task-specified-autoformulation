# Matched refits of the two T2-easy positive insulin revisions

Both revised models reduced pooled error on every one of the four development trajectories relative to their own jointly refitted restoring parent. Slot 015 had the lowest pooled errors. Most of its gain after transfer came from refitting the glucose and disposal equations with insulin held fixed. Glucose predictions improved substantially. Disposal and peak timing still warrant construction feedback.

## What was fitted

The four models are from slots 002 and 015 of Astra’s trajectory-guided function experiment. Each child replaces its parent’s insulin law with first-order clearance and a nonnegative production function of supplied tissue glucose, administered insulin and supplied endogenous glucose production (EGP). The EGP association is specific to the available signals in this experiment. These four development schedules contain no administered insulin, so they cannot establish the response to an insulin dose.

All four source vectors replayed successfully on all 16 training events. For the parents, the largest per-target NMSE discrepancy from the source fits was below 3 × 10⁻¹⁶. For the children, the archived independent insulin sum of squared errors was 7,519,120.27 and the coupled replay gave 7,519,120.26. These sources supply an archived reference for the complete parent fits and the children's insulin channel.

The matched procedure first held every coefficient in the insulin equation fixed. Both graphs initialize insulin from the measured initial concentration, and their insulin equations depend only on insulin and supplied channels. Refitting the other equations and the existing latent initializer therefore leaves each insulin trajectory fixed. The second stage released all coefficients. Both stages used free rollout on the complete 16-event training cohort, train-derived channel scales, analytic derivatives of the RK4 rollout, two starts and at most 80 residual evaluations per start. The first start was the incumbent and the second displaced active coefficients by 20% in predetermined directions. Every candidate for retention was scored again with the source rollout checks. Child insulin rollouts were also checked against an independent scalar integrator. The production weights stayed nonnegative, the clearance timescale stayed above the division guard, and the basis knots stayed fixed.

The training criterion averaged squared residuals over all time samples after normalizing each channel, then gave equal weight to Gp, I and U. Every training-selected vector was frozen before it was scored on the four previously used development events. The campaign accessed only the registered training and development splits.

## Pooled NMSE through the two fitting stages

Each pair of numbers gives training / development NMSE.

| Slot and insulin law | Supplied start | Insulin law fixed | Joint refit |
|---|---:|---:|---:|
| 002 restoring | 0.36257 / 0.30115 | 0.36257 / 0.30114 | 0.36257 / 0.30115 |
| 002 positive | 0.32593 / 0.27240 | 0.30319 / 0.24156 | 0.28528 / 0.20666 |
| 015 restoring | 0.19864 / 0.12232 | 0.19864 / 0.12231 | 0.19864 / 0.12232 |
| 015 positive | 0.14800 / 0.09871 | 0.13659 / 0.07599 | 0.13621 / 0.07627 |

In slot 015, the fixed-insulin refit lowered development Gp NMSE from 0.10977 to 0.03936 while U rose from 0.13351 to 0.13576. Joint fitting made a small additional gain on training and gave a similar development score. In slot 002, releasing the insulin parameters made a larger difference. Its fitted clearance timescale moved from 5.00 to 1.02 min, compared with 5.49 min in slot 015. These are effective fitted timescales under each coupled model.

## Joint refits by target

| Model | Train Gp | Train I | Train U | Dev Gp | Dev I | Dev U | Max dev insulin error (pmol/L) | Min dev insulin (pmol/L) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 002 restoring | 0.324 | 0.230 | 0.534 | 0.309 | 0.142 | 0.453 | 201.8 | 0.81 |
| 002 positive | 0.207 | 0.149 | 0.500 | 0.109 | 0.045 | 0.466 | 68.2 | 22.59 |
| 015 restoring | 0.149 | 0.229 | 0.218 | 0.094 | 0.142 | 0.131 | 201.5 | 0.76 |
| 015 positive | 0.048 | 0.127 | 0.234 | 0.038 | 0.056 | 0.135 | 101.1 | 23.67 |

Pooled NMSE favored each child on 4/4 matched development events. Gp improved on 4/4 in each pair, I on 4/4 for slot 002 and 3/4 for slot 015, and U on 2/4 and 0/4, respectively. Each joint child exceeded the 1 pmol/L maximum absolute insulin error criterion on all 16 training events and all four development events.

On the later meal in `val_001`, observed insulin at minute 122 was 271.0 pmol/L. Slot 002 predicted 69.2 with its restoring law and 202.8 with its positive law. The fast rise was better captured, but the same child peaked at minute 177 and 344.1 pmol/L. The observation peaked at minute 189 and 364.5 pmol/L. Slot 015 positive peaked at minute 173 and 369.7 pmol/L. Across all four events, the median absolute peak-time error was 25.5 min for both children and 24.5 min for both parents. The no-meal event has a small observed range, so the timing of its global maximum is particularly sensitive. The report includes event-level peaks and boundary indicators alongside the trajectories.

For the next construction review, distinguish three questions. Can the insulin law produce the rapid rise on the later meal, can it place the peaks across schedules, and can the downstream insulin-action and disposal equations convert that insulin history into the observed U? The improvement addresses the first question in part. The measured residuals localize the remaining problems but cannot select a unique replacement term.

## Fitting reliability and reproducibility

All 16 searches stopped on relative cost change before the evaluation cap (11–49 evaluations). For every model and stage, the two starts agreed within 3 × 10⁻¹² in independently replayed pooled training NMSE. The solver's reported first-order optimality values remained between 0.021 and 0.403, above the requested gradient tolerance. This supports repeatability of the attained objective under these starts. Global optimality, unique coefficients, new-cohort accuracy and physiological causation remain open.

From the repository root with the original handoff zip and public intakes in place, run

```bash
env PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m scripts.fit_positive_insulin_handoff --jobs 2
```

The output root is `output/t2-orion-positive-refits-20261123-v1/`. The sealed `fitted-models.json` contains the equations, initializer, named coefficients and updated scalar verification model for both refit stages. The accompanying `training-freeze.json` records all optimizer attempts. The two split reports contain scores, peak time and height, concentration extrema, independent replay diagnostics and full observed and predicted trajectories. The validation figure compares the joint endpoints. The fitting command resumes completed stage/start checkpoints.
