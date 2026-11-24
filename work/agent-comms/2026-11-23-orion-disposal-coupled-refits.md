# T2-easy coupled refits of the glucose balance

Astra’s delayed-meal model lowered development pooled NMSE from **0.06659 to 0.05998** against its immediate parent under matched constrained fitting. Disposal error fell 23%. Plasma glucose error rose. The two fits have identical insulin predictions. The original Orion parent, Astra’s earlier feasible endpoint, and both starting vectors are included in the reports so each change has an explicit comparator.

## Models and scores

Both arms use the tissue-glucose term in disposal. The immediate comparator is the exact `015-add-tissue-start` model from which the delayed child was constructed. Each was optimized from its own supplied start and a second, deterministically displaced start. The previously fitted feasible tissue-disposal vector is scored separately.

| Model | Training pooled | Training Gp | Training U | Development pooled | Development Gp | Development U |
|---|---:|---:|---:|---:|---:|---:|
| Original Orion insulin parent | 0.13621 | 0.04815 | 0.23370 | 0.07627 | 0.03788 | 0.13473 |
| Immediate tissue-disposal start | 0.13253 | 0.04210 | 0.22870 | 0.06846 | 0.01801 | 0.13116 |
| Astra’s supplied feasible refit | 0.12776 | 0.03189 | 0.22460 | 0.06597 | 0.01290 | 0.12880 |
| Delayed-meal start | 0.14834 | 0.13787 | 0.18037 | 0.07708 | 0.06660 | 0.10844 |
| Immediate balance, matched refit | 0.11992 | 0.02259 | 0.21039 | 0.06659 | 0.01638 | 0.12717 |
| Delayed balance, matched refit | **0.09433** | 0.03037 | **0.12583** | **0.05998** | 0.02546 | **0.09828** |

I NMSE was 0.12679 on training and 0.05621 on development throughout. Among the 16 training events, the delayed refit improved pooled NMSE on eight, disposal on nine and glucose on three relative to the matched immediate refit. Among the four development events, it improved pooled and disposal NMSE on all three meal events. The immediate refit predicted the no-meal event more accurately. Development glucose improved on one event. Relative to Astra’s supplied feasible vector, the delayed refit improved disposal on four events, glucose on zero and pooled NMSE on two.

The delayed refit still underpredicted the peak disposal rate on each meal event. On the later meal in `val_001`, for example, the observed peak was 7.92 mg/kg/min. The immediate refit peaked at 6.27 and the delayed refit at 6.64.

## What the fit says about the new equation

The selected delayed balance is approximately

```text
D(0) = 0
dD/dt  = (meal_rate - D) / 124.512
dGp/dt = 1.88356 + (1.00e-10) EGP - U + 8.57100 D
```

The fitted meal-delivery timescale moved from 180 to 124.51 minutes in both searches. It is a first-order smoothing timescale. Delivery starts responding as soon as ingestion begins. The direct endogenous glucose production (EGP) coefficient reached its imposed positive lower bound of 1e-10. The data fit supplies little support for that direct source in this parameterization. EGP is still supplied to the frozen insulin equation. The delayed balance preserves the literal unit disposal sink. The immediate refit used effective direct coefficients of −10.01 on EGP and −3.97 on U, after expanding its standardized regressors. This comparison changes both meal delivery and the balance terms, so the error reduction cannot be attributed to meal delay alone.

## Training feedback for the next construction pass

The disposal readout and its action initializer disagree with the observed ordering **before a meal could have an effect**. Two public no-meal training trajectories share the same initial insulin and therefore the same initial action state under the fitted initializer.

| Training event | Initial Gp | Initial tissue glucose | Observed initial U | Immediate refit U | Delayed refit U |
|---|---:|---:|---:|---:|---:|
| train_011 | 155.33 | 147.67 | 1.989 | 1.0001 | 1.0001 |
| train_012 | 189.84 | 113.15 | 1.835 | 3.1392 | 3.9276 |

In the delayed fit, the readout contains `+2.07958 (Gp−205.41071)/49.91609 − 4.31535 (Gt−100)/100`. Higher initial plasma glucose and lower initial tissue glucose therefore both raise predicted disposal. At the same initial insulin, the current action initializer gives no offset between these two events. Their observed ordering goes the other way. A useful construction probe is to test whether the model can represent the baseline disposal differences while keeping its within-trajectory response to changing glucose and insulin. Either the initializer or the disposal readout could contribute. These observations alone cannot distinguish them.

## Numerical qualification

All four supplied vectors reproduced their archived training metrics. Both fits retained the handoff’s insulin law, basis knots, positive action gains and available-information initializer forms. The optimizer fixed one action-scale and one action-offset convention and the coefficient of Uii−1, which vanishes on every training observation. These reductions preserve the represented training predictions. The optimizer fit the remaining coefficients against the original observation times, with constraints on each event’s minimum plasma glucose and dependent disposal (U minus supplied Uii) at observation times and interval midpoints. The disposal buffer was 0.0001 mg/kg/min. The delayed model also kept the EGP and delivery gains positive and its timescale above the evaluator’s division guard.

Both starts in both arms terminated successfully after 34–42 forward evaluations each, within the common budget of 80. They agreed to within 3 × 10⁻¹² in independently replayed training pooled NMSE for each arm. No endpoint required a feasibility backoff. This supports a reproducible local solution from these starts. The timescale has no confidence interval from this run.

An independent replay checked 16 subdivisions per input or basis interval against eight, then refined local minima of dependent disposal. Both refits reached the training buffer in `train_011` at time zero. On development, the smallest dependent disposal was 0.77356 for the immediate refit and 0.79316 for the delayed refit. Both minima occurred between recorded samples in `val_003`. These are numerical qualifications for the evaluated trajectories. Both selected vectors were frozen on the 16 training events before any of the four previously used development events was scored in this campaign. Test data stayed sealed.

## Artifacts and reproduction

Run from the repository root with Astra’s supplied zip and the registered training and development intakes present.

```bash
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m scripts.run_t2_disposal_coupled --jobs 2
AF_DISPOSAL_ROOT=output/t2-orion-disposal-coupled-20261123-v1
.venv/bin/python -m scripts.report_t2_disposal_coupled \
  "$AF_DISPOSAL_ROOT/val-report.json" "$AF_DISPOSAL_ROOT/glucose-and-disposal.png"
```

`--train-only` stops after the training report. The default command resumes the two starts for each arm independently and validates checkpoint identities. The training freeze contains the full graphs, initialization rules, fitted vectors and optimizer diagnostics. The split reports contain all observed and predicted trajectories, event scores, paired comparisons and domain checks. The exchange archive `orion-disposal-coupled-fits.zip` in the same output directory includes the selected models separately and the development trajectory figure.
