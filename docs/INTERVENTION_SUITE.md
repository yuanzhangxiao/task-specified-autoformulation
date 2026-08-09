# Frozen intervention suite

## Purpose

Phase A3 evaluates already selected models under controlled distribution shifts.
Private simulator specifications and resulting reference trajectories are final
evaluation inputs only. They are never available to proposal generation,
parameter fitting, pruning, model selection, or judge scoring.

The prespecified suite is
`configs/interventions/phase_a3_v1.json`. Its stable fingerprint is
`8a77a7ccd9bd44e9286bde10d12654d36e4efe9df9556a29dcebb6b274c528de`.
Changing any case definition creates a different suite and must be reported as
a new version rather than silently replacing this one.

Two additive frozen suites complete Phase A3:

- `phase_a3_noise_v1.json`, fingerprint
  `129d95526afa415223db7a428b6e69bb4b93511812f15aa5b1d914a41f3d6e33`;
- `phase_a3_dalla_man_v1.json`, expanded-template fingerprint
  `13ccb45664c46f6e66876fead16e36d486fc899b94222a39288ef256aef96d3b`.

## Initial coverage

The first frozen milestone covers Benchmarks 5 and 6 because their private
system specifications are complete, executable, and independent of method
performance. Each benchmark has four cases:

1. extrapolated input magnitudes with multiple events;
2. shifted initial conditions, unseen event timing, and a longer horizon;
3. simulator-parameter shifts with a sparser observation grid;
4. a denser observation grid with deterministic five-percent measurement
   noise relative to each state's clean trajectory range.

The output includes the clean private state trajectory, its deterministic noisy
observation counterpart where applicable, and the forcing trajectory. Reference
files and source system specifications are SHA-256 hashed in the generated
manifest.

## Generation

```bash
python scripts/freeze_intervention_suite.py \
  --suite configs/interventions/phase_a3_v1.json \
  --data-root data_raw \
  --output-root artifacts/interventions/phase_a3_v1
```

Generated references are private experiment artifacts and should not be
committed. The command is deterministic and safe to resume by rerunning: files
are written atomically, and the manifest records every output hash.

## Frozen-model evaluation

`scripts/evaluate_intervention_suite.py` runs selected models against the
generated forcing trajectories without structural or parameter refitting. It
reports clean-reference target MSE, train-scale-normalized MSE, degradation from
in-distribution test NMSE, and explicit simulation failures. When a case adds
measurement noise, noisy observations provide the permitted lagged history but
the prediction is scored against the clean private reference.

Canonical Autoformalism checkpoints are evaluated directly. Equation baselines
are wrapped in the same restricted expression compiler. Native D3 checkpoints
retain their frozen Adam-fitted coefficients and teacher-forced one-slot update
without a sampling-interval multiplier. Because native D3 historically
treated proposer parameter ranges as initialization metadata and could optimize
past them, the evaluation adapter may widen a metadata bound just enough to
admit the saved coefficient; it never clamps or refits that coefficient.

The original and B1-perturbed Dalla Man generators are now extracted into the
side-effect-free `rebuttal.dalla_man` module. Regression tests reproduce the
private notebook trajectories before the separately frozen
`phase_a3_dalla_man_v1` interventions are generated. Semantic and obfuscated
representations use the same underlying private dynamics but retain their own
frozen discovered models.

The final evaluator also reports permutation/sign/scale-invariant hidden-state
alignment, hidden-state coverage, qualitative response direction, response
shape correlation, and normalized peak-timing error. The additional
`phase_a3_noise_v1` suite measures one-, ten-, and twenty-percent noise using a
fixed noise realization per benchmark.
