# External T1 frozen-model replay

The uploaded `baseline-replay-subjects.tar.gz` is usable. Its two member hashes
match the preceding results-package manifest. It contains 443 adapted subjects:
102 SINDy, 102 PySR, 120 Sol, 119 D3. All 36 endpoints are present in the three
fixed demonstration cells: canonical named, canonical obfuscated, and perturbed
named T1-easy, with four methods and three repetitions each.

All 36 endpoints completed 16 training, four validation, and seven previously
defined exploratory intervention rollouts: 972 trajectories. Parameters,
process definitions and causal initializers were frozen. D3 retained its native
increment semantics and one-minute sampling. Supplied auxiliary trajectories
were available under the original T1-easy contract. No fitting, provider calls,
or test-trajectory access occurred; existing test endpoint scores were discarded.
Repeating the command reproduced the identical sealed summary without new solves.

Plan: `5ed42ff123b77925484cc2ee154149bedcbe92ed331f989e5974a1841d15de52`.
Implementation and command: `docs/T1_EXTERNAL_INTERVENTION_REPLAY.md`.
Full results and plots: `artifacts/t1-external-intervention-replay-2026-09-25-v1/`.
Plotting handoff archive: `transfers/t1-baseline-intervention-replay-20260925.tar.gz`.
No experiment artifacts are committed.

## Perturbed named T1-easy

The control is 60 g at 60 min. The intervention splits it into 30 g at 60 min
and 30 g at 90 min. Response RSE scores the split-minus-control prediction
against the corresponding reference difference; a predicted zero change scores 1.

| Model | Training NMSE | Validation NMSE | Split-meal NMSE | Response RSE |
|---|---:|---:|---:|---:|
| R4 assisted sign repair + pruning | 0.10715 | 0.08113 | 0.05871 | 0.22162 |
| R4 unchanged paired control + pruning | 0.07078 | 0.06811 | 0.06042 | 0.38790 |
| Sol repetition 0 | 0.80328 | 0.07432 | 0.01203 | 0.92076 |
| Sol repetition 1 | 0.47952 | 0.03348 | 0.05228 | 1.26068 |
| Sol repetition 2 | 23.93320 | 62.97079 | 38.18464 | 54.95773 |

R4's response RSE is lower than every external endpoint in this matching cell.
However, Sol repetitions 0 and 1 have lower absolute split-meal NMSE. Neither
fits the entire training set comparably well. This is a useful response-shape
comparison, not the clean matched-good-training example we originally sought.

The R4 diagnostic is assisted. Its correct meal-spacing response is largely
mediated by supplied tissue glucose, and it previously failed the initial-tissue-
glucose probes. It must not be presented as autonomous full-mechanism recovery.

## Canonical obfuscated T1-easy

| Model | Training NMSE | Split-meal NMSE | Response RSE |
|---|---:|---:|---:|
| Previous R9 rescue, before new sign repair | 0.01841 | 0.01340 | 0.11846 |
| Sol no-latent, repetition 0 | 0.03677 | 0.01347 | 0.07488 |
| Sol three-latent, repetition 1 | 0.04118 | 0.02679 | 0.36627 |
| Sol repetition 2 | 0.16232 | 0.01864 | 0.53558 |

The no-latent Sol model remains strong on meal spacing. We cannot use this case
to claim that missing latent memory necessarily causes failure. R9 improves on
the three-latent endpoint but has unresolved sign concerns until the separate
canonical rescue returns. SINDy/PySR/D3 fit training poorly in these cells and
therefore do not supply the desired matched-training comparison either.

The existing no-specification example remains relevant, with training NMSE
0.04358 and response RSE about 1.135, but has three latent states. Its failure
must be discussed using its equations and omitted supplied mechanisms, not by
calling it memoryless.

## Next decision

Inspect the separately submitted canonical R9/R2 sign-rescue outcomes. Replay
all repaired/control endpoints on this unchanged seven-case suite, retaining
both successful and adverse interventions. Do not alter the external baseline
parameters or redefine the probes to produce a preferred ranking.

The plotting package contains all development/probe curves, the complete frozen
baseline models with test scores removed, all 36 metric rows, and PNG/SVG views.
Keep canonical and perturbed dynamics separate in figures. These exploratory
diagnostics are not independent confirmatory test estimates.
