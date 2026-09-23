# Four-model T1 intervention plots and data handoff

The requested comparison uses canonical, obfuscated T1-easy throughout. It adds
our Brief-only seed-1 round-13 **unretained fitted trial** to the two Sol models
and our no-specification seed-0 model. It does not include the perturbed named
Full round-13 model, which belongs to a different benchmark.

## Data and figure package

Local output: `artifacts/t1-four-model-comparison-2026-09-23/`.
Shareable archive: `artifacts/t1-four-model-comparison-2026-09-23.zip`.

- `absolute.csv`: all seven original diagnostic cases, in native glucose-mass
  units (mg/kg), at every saved time sample.
- `relative_change.csv`: all five intervention/control comparisons. Each model
  subtracts **its own** paired-control prediction; the reference subtracts its
  reference control. These are additive changes, not percentages, normalized
  changes, or deviations from the initial sample.
- `training_validation.csv`: all 16 training and four validation trajectories
  for every model. The two training panels in Claude's figure are `split=train`,
  `case_id=002` and `case_id=010`. Read case IDs as strings.
- `scores.csv`: the original saved per-trajectory NMSE values, using the original
  training scale; these do not score the relative changes.
- `manifest.json`: exact model IDs, diagnostic definitions, original plan
  identities, checks, and SHA-256 hashes of every source file.
- `absolute.*` and `relative_change.*`: full 300-minute views. The absolute plot
  also includes the unperturbed control to expose baseline errors. Files with
  `_first120min` show the window used in Claude's original figure. PDF, SVG, and
  PNG formats are provided.

The columns `reference`, `brief_only_r13`, `sol_no_latent`, `sol_three_latent`,
and `no_spec` are consistent across the three trajectory CSVs. The Brief-only
curve is green; the two Sol curves and no-specification keep Claude's colors.
Relative changes are not a substitute for showing absolute accuracy.

The original intervention plan is
`8035569cea1a51f09bf0f895f6e425eafe937a8ac49341c39fe472398ab3b72d`.
The Brief-only source model is
`a276318babbbc64b105676549d20eb813157492c7909fd3ce662abe604f3ee7e`.
The physical intervention changes only initial tissue glucose by +/-20%, holding
initial plasma glucose and all other physical states fixed. The supplied
auxiliary trajectories are regenerated coherently by the original reference
simulation. This is an exploratory initial-condition experiment, not a new
insulin schedule, registered test result, or controlled ablation of memory.

## Reproduction and verification

Run from the project checkout with the original local replay artifacts present:

```bash
MPLCONFIGDIR=/tmp/autoformalism-matplotlib .venv/bin/python \
  scripts/plot_t1_four_model_interventions.py \
  --output artifacts/t1-four-model-comparison-2026-09-23

PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_t1_four_model_plot.py tests/test_t1_intervention_probe.py \
  tests/test_t1_curve_replay.py tests/test_t1_brief_only_replay.py \
  tests/test_t1_full_interventions.py
```

The exporter checks original artifact seals, identical physical references and
time grids, finite complete predictions, and identical intervention plans.
Tests cover different-reference rejection and subtraction of each model's own
baseline. All 24 relevant tests pass. The changed Python files pass Ruff;
repository-wide `ruff check .` reports the same 37 pre-existing issues in
unrelated `analysis/claude` files.

No new integration, parameter fitting, reference simulation, LLM call, or test
access is needed. Existing checkpoint reuse avoids source-version drift from
rerunning older experiment code. Models and conditions are unchanged. The
Brief-only model's addition is post hoc and must not be presented as a retained
benchmark endpoint or as uniform mechanistic recovery.
