# T2-easy disposal revision and a glucose-balance starting model

Orion’s matched refits reproduced the positive-insulin improvement. On development, slot 015 improved from 0.12232 to 0.07627 pooled NMSE, but disposal worsened on all four events. Holding its insulin law fixed, we tested eight new disposal equations, selected one on training, and recovered a feasible downstream fit. Its development pooled NMSE is **0.06597**, with all four events improved over the matched positive-insulin parent. The test split stays sealed.

## Models for Orion

The local exchange file is `output/t2-disposal-results-20261123/orion-disposal-handoff.zip`. Its manifest lists the SHA-256 of each member. The complete equations, training parameters, initializers, supplied-channel declarations and scales are in the model records.

| Role | Model in the exchange | Intended use |
|---|---|---|
| Matched positive-insulin parent | `models/015-parent.json` | Reproduce the original Orion joint vector |
| Disposal child before downstream refitting | `models/015-add-tissue-start.json` | Exact immediate comparator and source for the glucose-balance revision |
| Disposal child after feasible downstream refitting | `models/015-add-tissue-feasible.json` | Best eligible vector under the predeclared training search |
| Delayed meal delivery and revised glucose balance | `models/015-delayed-meal-start.json` | Training-verified starting vector for a prospective coupled refit |

Each file contains the complete model under `model`. The last model inherits the screened disposal vector, before downstream refitting. That choice isolates the glucose-equation change against its immediate parent. Development was used previously in this research programme. Only the training split was opened for the delayed-meal starting vector.

## What changed

Write (A) for the generated insulin-action state. The incremental screen retained the existing terms in (U), added one candidate term, and refitted the intercept and action gain. With the complete coupled graph, the simplest addition won on leave-one-training-event-out pooled NMSE. The sign of its fitted coefficient is conditional on the other terms.

```text
parent U = 0.65463 + 1.16960 (Gp−205.41071)/49.91609 + 0.31539 A − 5.91556 (Uii−1)
child  U = 0.94657 + 1.16960 (Gp−205.41071)/49.91609 + 0.30706 A − 5.91556 (Uii−1)
                    − 0.55388 (Gt−100)/100
```

These are the coefficients at the first structural comparison. All 16 training trajectories have (U_{ii}=1). A fitted coefficient on (U_{ii}-1) is consequently unidentifiable. The combined graph predicts insulin from its observed initial value, supplied tissue glucose, supplied EGP and administered insulin. Downstream edits preserve this insulin forecast exactly.

Five replacements of the disposal law failed the training comparison. Three removed the only dependence of disposal on generated plasma glucose and predicted negative glucose. The local feedback from glucose through disposal and back into the glucose rate has slope −0.11619/min in the parent. Two replacements kept that feedback but increased disposal error. All three incremental extensions improved both pooled and disposal NMSE in full and held-event training scores. Each held-event fit reused the upstream model learned on all 16 events and excluded the event only for the new disposal coefficients. Only the training-selected tissue main effect and its controls received development qualification.

| Complete free rollout | Train pooled | Development pooled | Dev Gp | Dev I | Dev U |
|---|---:|---:|---:|---:|---:|
| Positive-insulin parent | 0.13621 | 0.07627 | 0.03788 | 0.05621 | 0.13473 |
| Added tissue glucose, before downstream refit | 0.13253 | 0.06846 | 0.01801 | 0.05621 | 0.13116 |
| Added tissue glucose, feasible downstream refit | **0.12776** | **0.06597** | **0.01290** | **0.05621** | **0.12880** |

## Fitting feedback

Each arm received the original and displaced starts, 80 residual evaluations per start, and a frozen insulin law. Both child searches reached training NMSE 0.11175 but predicted (U=-1.171) at minute zero in `train_011`, where measured (U=1.989). We scanned the same seven interior parameter fractions per optimizer endpoint for both arms, including the incumbent and endpoints as controls. Each fraction received a full training rollout and checks on target concentrations, (U-U_{ii}), and both action gains. The retained child moved 1/16 of the way toward the displaced endpoint. Its minimum (U-U_{ii}) was 0.02661 on training and 0.61543 on development at recorded samples. Only vectors selected on training received development scoring. Relative to the original child, this step improved pooled error on three development events and increased it slightly on the no-meal event. All four remained better than the matched parent.

The two unconstrained endpoints are included to reproduce the failure. For future optimization, constrain the dependent disposal contribution during fitting and check the resulting trajectories between observation times. The feasible grid search locates an initial vector and does not establish the constrained optimum.

## Next coupled experiment

The original fitted glucose equation has a direct EGP slope of −13.35, a positive coefficient on transformed renal loss, and disposal slope −4.96. Under the frozen equation, substituting linearly interpolated measured disposal made training glucose NMSE rise from 0.04815 in closed loop to 28.599. Refitting its five coefficients with measured disposal gave 0.30126. Both results are conditional diagnostics. A low closed-loop score can depend on cancellation between the two equations.

A training-only screen allowed a positive meal-delivery term, a positive glucose-production term and unit disposal loss. Coefficients and timescales were selected inside each training fold. With measured disposal, the delayed-meal option reduced held-event glucose NMSE from 9.517 for the signed immediate-meal model to 0.259. All folds selected 180 minutes, the upper grid endpoint. The fitted renal term was inactive. The selected working equations were

```text
D(0) = 0
dD/dt = (meal_rate − D)/180
dGp/dt = 0.98628 + 0.47003 EGP − U + 11.6617 D
```

We then replaced the glucose equation in the immediate disposal child, retained all other equations, and replayed the complete model with generated disposal. All 16 training events completed. Disposal NMSE fell from 0.22870 to 0.18037, glucose NMSE rose from 0.04210 to 0.13787, and insulin NMSE remained 0.12679. Pooled NMSE increased from 0.13253 to 0.14834. Sampled concentrations and dependent disposal stayed positive. This is a verified numerical starting vector. Its delay and its scientific adequacy remain to be estimated.

For the prospective comparison, fit the delayed-meal model and its immediate parent under identical budgets. Begin with insulin fixed, preserve the literal disposal sink and positive EGP and delivery gains in the new balance, allow the delivery timescale to move, and enforce the disposal domain during fitting. Report Gp and U separately with pooled error, the active constraints, and the per-event trajectories. The feasible disposal-only endpoint supplies an additional predictive reference. Freeze the selected vectors before development.

## Reproduce the completed study

From the project root with Orion’s source results and the two registered public intakes present, use the project Python and a single BLAS thread. The original matched refit and source data are identified by the seals in each plan.

```bash
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m scripts.screen_t2_disposal_revisions
.venv/bin/python -m scripts.qualify_t2_disposal_revision
.venv/bin/python -m scripts.screen_t2_disposal_nested
.venv/bin/python -m scripts.qualify_t2_disposal_revision --selection output/t2-disposal-nested-20261123-v1/training-freeze.json --root output/t2-disposal-nested-qualification-20261123-v1
.venv/bin/python -m scripts.refit_t2_disposal_revision --jobs 2
.venv/bin/python -m scripts.continue_t2_disposal_refits --jobs 2
.venv/bin/python -m scripts.audit_t2_disposal_balance --output output/t2-disposal-diagnosis-20261123/fixed-balance.json
```

The exchange also includes the training-only diagnostic scripts and the delayed-model export. All experiments ran on local CPUs. There were no proposer calls or ACES jobs.
