# Frozen T1 trajectory inspection

`scripts/replay_t1_curves.py` replays the supplied canonical T1 easy archives.
It covers both named and obfuscated cells, all available selected round-12
Full/No-latent/No-specification endpoints, and every available SINDy, PySR and
raw-data-agent repetition. It does not select a new endpoint or fit parameters.

The round-12 archive has `plan.json` and eight
`results/<task>/round_03/result.json` files. The baseline archive has `plan.json`
and `summary.json`. The historical plan identities are explicit constants in the
script. Unexpected archive paths, links, oversized files and changed saved inputs
are rejected. Baseline public-data fingerprints must match the data carried by
the corresponding construction plan. Original baseline source files are not in
these archives; this verifies the frozen adapted models, not their raw sources.

Run from the project checkout, using the downloaded archives:

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_t1_curves.py \
  --ours /absolute/path/to/t1-round12.tar.gz \
  --baselines /absolute/path/to/baseline-validation-models.tar.gz \
  --output artifacts/t1-curves-round12 \
  --workers 4

MPLCONFIGDIR=/tmp/t1-matplotlib PYTHONPATH=src \
  .venv/bin/python scripts/plot_t1_curves.py \
  --root artifacts/t1-curves-round12

PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_t1_curve_replay.py tests/test_cstr_curve_replay.py \
  tests/test_baseline_validation.py
```

This is a new diagnostic replay in the current runtime. The source-tree identity,
script identity, numerical settings and archive hashes are frozen separately from
the historical results. The solver is Radau at relative tolerance `1e-7` and
absolute tolerance `1e-9`, with 120 seconds per trajectory. Historical scores stay
in the report for comparison. Changed source/runtime identities require a new
output directory; an interrupted trajectory marker requires inspection. Completed
trajectory checkpoints resume without another simulation.

The observed target supplies its initial value only. Supplied auxiliaries and
external inputs retain their public availability contract. Latent initialization
uses the saved policy and parameters. There are no measured-target resets,
new initializer fits, model revisions, test-data reads or private-reference reads.
Training and validation errors use the same pooled training standard deviation,
include initial samples and weight by sample count. A failed trajectory prevents
a complete aggregate score.

Outputs:

- `summary.json`: training/validation scores, historical comparisons, state counts
  and forcing symbols, with unavailable models retained explicitly.
- `models.json`: exact lowered equations, parameter values, contexts and provenance.
- `curves.csv`: every model, split, trajectory, timestamp, reference and prediction.
- `trajectory_scores.csv`: individual trajectory scores and success status.
- `replays/`: sealed per-trajectory outputs, including causal initial states.
- `figures/`: PNG and SVG inspection sheets for ablations and external repetitions.

The plot labels `v01` as plasma-glucose mass for the specific frozen canonical
T1 obfuscated cell. Its target and time arrays were independently checked against
named `Gp`; this is an interpretation for this diagnostic, not a proposer input.

Do not infer that absent latent states imply an absent effective mechanism: T1
easy supplies tissue glucose and several rates as auxiliary trajectories. A model
can use those channels as dynamic mediators. Likewise, different validation scores
alone do not identify which equation caused the difference. Inspect training
curves, all repetitions, actual dependencies and no-meal controls before choosing
a paper illustration. These two archives contain no specification-rejected
candidate and no available Phase-B D3 subject for the selected cells.

## Brief-only equation inspection

`scripts/replay_t1_brief_only.py` accepts the separate eight-lineage brief-only
archive. It exports every retained model's exact equations and fitted parameters
before replay. All seven available endpoints use their own original training and
validation arrays; the missing named canonical seed remains explicit. Only the
three canonical endpoints additionally receive the seven already frozen synthetic
interventions from `T1_INTERVENTION_PROBE.md`. Perturbed endpoints are never
evaluated against canonical reference curves. No coefficients or initializers
are re-estimated, and neither the original experiment nor its selected models
are overwritten.

```bash
PYTHONPATH=src .venv/bin/python -m scripts.replay_t1_brief_only freeze \
  --archive /absolute/path/to/t1-brief-only-round12.tar.gz \
  --probe artifacts/t1-intervention-probe-2026-09-22 \
  --output artifacts/t1-brief-only-inspection

PYTHONPATH=src .venv/bin/python -m scripts.replay_t1_brief_only run \
  --output artifacts/t1-brief-only-inspection --workers 4

MPLCONFIGDIR=/tmp/t1-matplotlib PYTHONPATH=src \
  .venv/bin/python -m scripts.plot_t1_brief_only \
  --root artifacts/t1-brief-only-inspection \
  --source artifacts/t1-curves-round12-2026-09-22 \
  --probe artifacts/t1-intervention-probe-2026-09-22

PYTHONPATH=src .venv/bin/pytest -q tests/test_t1_brief_only_replay.py \
  tests/test_t1_curve_replay.py tests/test_t1_intervention_probe.py \
  tests/test_t1_full_interventions.py tests/test_baseline_validation.py
```

The inspection focus, canonical obfuscated seed 1, was identified from existing
scores before the new simulation. Its complete sixteen training and four
validation trajectories are shown alongside both previously inspected Sol fits.
The renderer imports those original comparator records without new simulation,
binds their artifact/model identities, and requires identical reference values
and times before overlaying curves. `comparison_curves.csv` and the copied
comparator records make the resulting plots independently reusable.

Equation screening must use effective fitted signs, active coefficient magnitudes
and initial conditions. Shared decay rates can make multiple latent filters
redundant, and supplied physiological auxiliaries can carry input history. Small
ordinary trajectory NMSE does not certify the learned response to independently
changed auxiliary states. Ground-truth comparisons here are post-selection
diagnostics; private equations do not enter proposal generation or fitting.

As in the other diagnostics, source/runtime drift refuses whole-run resume. If
another task changes unrelated source files during inspection, preserve the
original plan and results. Reconstruct and verify the exact frozen source before
testing resume; do not silently update its identity or bypass the guard.
