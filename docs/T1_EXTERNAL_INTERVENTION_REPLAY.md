# Frozen external T1 intervention replay

`scripts/replay_t1_external_interventions.py` replays the supplied external-model
archive on the previously generated seven T1 exploratory intervention cases.
It includes all four methods and all three repetitions in canonical named,
canonical obfuscated, and perturbed named T1-easy: 36 endpoints in total.
Selection uses cell/method/repetition identity, never an existing test score.

The archive contains 443 adapted models across the larger benchmark matrix.
The script verifies both JSONL file hashes against the supplied package manifest,
validates each selected candidate hash, and imports complete parameter vectors,
process definitions, initial conditions and execution semantics. Existing private
endpoint fields are discarded. Original training-file hashes are absent from the
adapted subjects; the development data are joined by benchmark identity and
checked public channel contract. The local public plan is independently hashed.

No coefficients or initializers are refitted. Continuous models use the existing
bounded ODE replay. D3 uses the existing restricted native increment interpreter,
`x_next = x + f`, on the same one-minute sample grid as its training data.
Later target measurements never reset predicted target states. Both paths retain
their existing access to supplied auxiliaries. All 16 training and four validation
trajectories are replayed, using the training standard deviation for normalization.

The meal comparison is 60 g at 60 min versus 30 g at 60 min and 30 g at 90 min.
Absolute trajectory NMSE and response error measure different things. The latter
is the squared error of the predicted split-minus-single trajectory, divided by
the squared magnitude of the corresponding reference change. A predicted zero
change has response error 1, when the reference change is nonzero.

The other saved cases change initial tissue glucose by ±20%, with and without
a meal. They change tissue glucose alone, not a mass-preserving redistribution.
Canonical and perturbed models use their own matching reference dynamics.

These are exploratory demonstration diagnostics, already inspected during
development. They are not new independent test estimates. T1-easy supplies
intervention-specific auxiliary trajectories; successful conditional prediction
does not establish autonomous recovery of the full physiological system.

## Local replay

From the project checkout, with the previously downloaded public/reference
artifacts present:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/replay_t1_external_interventions.py \
  --archive /Users/yuanzhangxiao/Downloads/baseline-replay-subjects.tar.gz \
  --public-plan artifacts/t1-curves-round12-2026-09-22/inputs/ours/plan.json \
  --reference-root artifacts/dalla-rescue-inspection-2026-09-24 \
  --root artifacts/t1-external-intervention-replay-2026-09-25-v1 \
  --workers 2
```

The sealed plan binds source/runtime, script, inputs and cases. Every completed
trajectory is checkpointed. Repeating the command with the same identity reads
completed records. Interrupted continuous rollouts retain a marker requiring
inspection. The summary preserves failed endpoints; missing/failed curves do not
become zero errors. `curves.csv` contains all seven cases for every endpoint.

Focused checks, including analytic continuous/discrete smoke cases:

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_t1_external_interventions.py tests/test_t1_curve_replay.py \
  tests/test_t1_intervention_probe.py tests/test_d3_rollout.py
```
