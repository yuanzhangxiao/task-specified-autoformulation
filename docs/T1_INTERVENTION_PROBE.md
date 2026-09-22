# Exploratory T1 intervention discrimination

This is a separate diagnostic suite, not a modification of the benchmark or its
test split. It freezes previously selected models, chooses new interventions
from their equations, and generates all reference and candidate responses without
refitting. No proposer or selection controller receives these outcomes.

The source is the output of `scripts/replay_t1_curves.py`. The eligible inventory
contains every available canonical obfuscated T1 easy model with aggregate
free-rollout training NMSE below 0.05: Sol repetitions 0/1 and No-specification
seed 0. This is an exploratory screening criterion, not a preregistered benchmark
threshold. The primary comparison is the two Sol models. All three eligible
models and every case are retained in the output.

The two Sol equations use very different coefficients on the supplied tissue
glucose channel: 0.9735222 and 0.0471445671. The latter also has three meal-memory
states. No-specification has three latent states but ignores all supplied
auxiliaries. With its fitted initialization unchanged, its output is a function
of meal input and initial plasma glucose only. Consequently it cannot respond
to a preparation that changes initial tissue glucose while keeping those two
arguments fixed. This is a missing dependency, not proof of insufficient latent
dimension.

Inspection of the frozen source data shows that all 16 training and four
validation trajectories satisfy `Gp(0) + Gt(0) = 302.990063489` mg/kg to CSV
precision. The original design therefore does not directly test independent
variation of the two initial masses. The new preparation leaves that line while
retaining the same target initial value and meal schedule. This explains the
choice of intervention; it does not assert exact observational equivalence of
the fitted models. The meal-memory states are inactive in the fasting probe,
so its outcome cannot isolate a benefit of those states.

Seven cases are frozen before evaluating new outcomes:

1. No meal, basal initial state.
2. No meal, initial tissue glucose reduced by 20%.
3. No meal, initial tissue glucose increased by 20%.
4. A 60 g meal at 60 minutes, basal initial state.
5. The same meal, initial tissue glucose reduced by 20%.
6. The same meal, initial tissue glucose increased by 20%.
7. Two 30 g meals at 60 and 90 minutes, basal initial state.

Every other initial physiological state and every physical parameter remains
unchanged. Independent tissue preparation changes initial total glucose mass;
it is intentionally different from the existing release's mass-conserving
plasma/tissue redistribution. This is an additional synthetic initial-condition
family, not an existing benchmark input schedule or a proposed clinical protocol.
The full reference dynamics regenerate plasma glucose and every supplied
auxiliary consistently. No auxiliary curve is artificially held at its old value.

The reference uses the trusted canonical Dalla Man equations, with independently
checked Radau and DOP853 integrations (relative tolerance 1e-10, absolute tolerance
1e-12, maximum step 0.5 min). Both integrations must agree on every exported
public channel within 2e-5 in its native units. Saved candidates use the production
free-rollout evaluator, fixed parameters, original causal initializers and no
observed-target resets.

Ordinary trajectory NMSE retains the original training scale. Paired-effect
error is reported separately:

`sum((predicted_delta - reference_delta)**2) / sum(reference_delta**2)`.

Here each delta subtracts the matching unperturbed control at every time point.
A zero-effect prediction scores one for a non-negligible reference response.
Near-zero reference effects are labeled uninformative, rather than divided by an
arbitrary small constant. Failures remain failures. Neither this measure nor peak
response agreement establishes full scientific correctness.

From the project checkout:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.probe_t1_interventions freeze \
  --source artifacts/t1-curves-round12-2026-09-22 \
  --output artifacts/t1-intervention-probe

PYTHONPATH=src .venv/bin/python -m scripts.probe_t1_interventions run \
  --output artifacts/t1-intervention-probe

MPLCONFIGDIR=/tmp/t1-matplotlib PYTHONPATH=src \
  .venv/bin/python -m scripts.plot_t1_intervention_probe \
  --source artifacts/t1-curves-round12-2026-09-22 \
  --root artifacts/t1-intervention-probe

PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_t1_intervention_probe.py tests/test_t1_curve_replay.py \
  tests/test_baseline_validation.py
```

Plans, references and model trajectories are sealed, and source/runtime changes
invalidate whole-run resume. Completed work is loaded from checkpoints. The
output contains `plan.json`, `summary.json`, `curves.csv`, all seven reference
bundles, all 21 model rollouts and four PNG/SVG inspection sheets. The overview
shows two illustrative training trajectories and paired initial-condition
responses; separate sheets include every training trajectory, every absolute
diagnostic trajectory and every paired response over the full horizon.

Interpretation must remain specific: this can demonstrate that good training
fit does not determine response to independently varied initial conditions.
It cannot establish a Full-method advantage, a universal benefit of memory, or
that the current T1 requirement checker would reject a model omitting tissue
glucose. A task specification must state which intervention family the model is
intended to support. Report the meal-spacing outcome as well as the tissue-glucose
outcomes, and distinguish response error from aggregate trajectory NMSE.
