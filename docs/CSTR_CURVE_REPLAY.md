# Frozen CSTR trajectory replay

`scripts/replay_cstr_curves.py` replays the four retained CSTR endpoints in the
user-supplied round-12 archive: Full and No latent, seeds 0 and 1. It verifies
the archive members, artifact seals, pinned plan identity, lowered model hash,
and complete fitted parameter vector. It does not fit parameters, call an LLM,
or open test data. This is a narrow historical replay, not a general evaluator.

Run from the project checkout with its Python environment:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/replay_cstr_curves.py \
  --archive /path/to/cstr-round12.tar.gz \
  --output artifacts/cstr-curves-round12

MPLCONFIGDIR=/tmp/cstr-matplotlib .venv/bin/python \
  scripts/plot_cstr_curves.py --root artifacts/cstr-curves-round12

PYTHONPATH=src .venv/bin/pytest -q tests/test_cstr_curve_replay.py
```

Completed trajectory checkpoints resume without simulation. Changed input,
source, script, runtime or numerical settings require a new output directory.
An interrupted trajectory marker requires inspection rather than an automatic
retry. Every trajectory must succeed for the aggregate score to be available.

The replay integrates with Radau, relative tolerance `1e-7`, absolute tolerance
`1e-9`, and a 300-second limit per trajectory. It uses the retained causal
initializers, without observed-target resets. The measured auxiliary channels
`C(t)` and `Tj(t)` remain supplied under the task contract. Thus these are
conditional temperature rollouts, not autonomous predictions of all three
physical states. NMSE uses the pooled training temperature standard deviation
and includes the initial sample; trajectory scores are sample-weighted.

The September 22 replay completed 80 trajectories and reproduced all eight
saved train/validation scores (maximum discrepancy below `6e-14`). Exports:

- `trajectories.csv`: seed, arm, split, trajectory, target, time, observed,
  predicted, residual, training scale.
- `inputs_auxiliaries.csv`: public inputs and supplied auxiliary channels.
- `trajectory_metrics.csv`, `summary.json`: individual and aggregate scores.
- `models.json`: exact retained equations, parameters and provenance.
- `figures/`: complete training/validation contact sheets, residuals and an
  overview, in PNG and editable SVG. The overview includes every validation
  trajectory and two training cases chosen by reference excursion, independent
  of model errors; it is not a substitute for the complete contact sheets.

## Reference-data issue found during inspection

These plots are currently **inspection artifacts**, not verified evidence for
the paper's intervention claim. In the saved CSTR cell, all eleven forced
training step/pulse cases and three of four validation cases have temperature
excursions of only about `1e-9 K`, despite changing forcings. The remaining
validation case is sinusoidal and has a roughly `12.03 K` excursion.

`phase_b_generation._simulate_cstr` calls `_integrate`, which uses LSODA without
segmenting at forcing discontinuities or bounding its step size. A synthetic
diagnostic using that exact helper, `dy/dt = 1_[10,15)(t) - y`, `y(0)=0`,
on the same 0-to-30 grid returned an identically zero trajectory. The helper
evaluated the RHS at approximately 0, 0.003, 0.006 and 30, missing the forcing
window. The analytic peak is `1-exp(-5)`, approximately 0.993262. A separate
diagnostic with maximum step 0.05 agreed with the analytic solution within
`1.6e-8`. Four other CSTR forcing windows showed the same missed-forcing failure.

This reproduces a concrete failure mode consistent with the saved data; it is
not a regenerated CSTR benchmark or a complete audit of the historical run.
No benchmark files, generator, model parameters or historical scores were
changed. Before publication, audit actual CSTR reference integration at every
forcing boundary, then version any necessary corrected dataset and rerun all
affected methods on that same version. Do not replace reference curves alone
while retaining scores from the old data.

The archive has no CSTR No specification run, no rejected-candidate histories,
and no external-baseline equations. These comparisons cannot be generated from
these files. Both seeds must remain visible: seed 0 favors Full on aggregate
validation, while seed 1 favors No latent. Validation also participated in
model selection, so this is exploratory comparison rather than a test estimate.
