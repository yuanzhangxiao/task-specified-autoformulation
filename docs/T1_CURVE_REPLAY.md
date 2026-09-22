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
