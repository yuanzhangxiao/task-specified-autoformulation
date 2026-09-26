# CSTR curves reproduced; reference forcing needs an audit before Figure 1

The user supplied `cstr-round12.tar.gz`. I replayed the exact retained Full and
No latent models for seeds 0 and 1, with their saved parameter vectors and
causal initializers. All 80 trajectories completed; all eight train/validation
aggregate NMSEs reproduced within `6e-14`. There was no refitting, LLM call,
test-data access, remote session, or benchmark modification.

The exports are in
`artifacts/cstr-curves-round12-2026-09-22/`, including long-form trajectories,
public forcings/auxiliaries, per-trajectory scores, models and PNG/SVG contact
sheets. `docs/CSTR_CURVE_REPLAY.md` gives reproduction commands. Use the
complete sheets when interpreting the overview, which only shows two training
examples but all four validation cases.

## Scores reproduced

| Seed | Arm | Training NMSE | Validation NMSE |
|---|---|---:|---:|
| 0 | Full | 0.132255 | 0.0140082 |
| 0 | No latent | 0.122933 | 0.0319746 |
| 1 | Full | 0.159257 | 0.00417653 |
| 1 | No latent | 0.155628 | 0.00332571 |

Seed 0 Full wins 3/4 validation trajectories; seed 1 No latent wins 4/4. The
single nonflat validation trajectory (`validation_002`, sinusoidal jacket
forcing) is fitted reasonably closely by both arms: seed-0 NMSE is 0.0137282
versus 0.0518265. This is not the clean example of one arm tracking all training
curves and the other diverging on validation. Full's seed-0 training NMSE is
0.9542 on the feed-concentration sine and 0.6390 on the feed-temperature sine.

## Most important finding: reference traces can miss entire forcing windows

Raw stored arrays, not just plots, show nearly constant `T`, `C` and `Tj` in
every step/pulse protocol. All eleven forced training step/pulse cases and
three validation cases have target excursions around `1e-9 K`. The input
columns do change. For validation:

| Trajectory | Forcing | Input excursion | Reference T excursion |
|---|---|---|---:|
| validation_000 | Cf step, 6–22 min | 0.125 | about 1e-9 K |
| validation_001 | Tf pulse, 10–15 min | 5 K | about 1e-9 K |
| validation_002 | Tjf sine, period 16 min | 12 K | 12.0309 K |
| validation_003 | Cf/Tf step, 7–20 min | 0.1 / 4 K | about 1e-9 K |

In `src/autoformalism/benchmarks/phase_b_generation.py`, `_simulate_cstr`
supplies time-dependent forcings to `_integrate`. That helper runs LSODA over
the entire horizon with `t_eval`, without segmenting at discontinuities or
setting `max_step`. Output sampling does not require RHS evaluation at each
sample time.

I tested the helper on the analytically soluble synthetic ODE
`y' = 1_[10,15)(t) - y`, `y(0)=0`, using the same 0–30 min, 0.1-min output
grid. It returned all zeros, evaluating its RHS only at approximately 0,
0.003, 0.006 and 30 min. The correct peak is `1-exp(-5) = 0.993262`.
A diagnostic solve with `max_step=0.05` matched the analytic solution within
`1.6e-8`. All five distinct CSTR forcing windows tested reproduced the missed
forcing failure. Details are in `forcing_discontinuity_diagnostic.json`;
raw-data ranges are in `reference_range_audit.json`.

This establishes a concrete failure mode consistent with the historical
arrays. It does not claim that the actual private CSTR simulation was rerun
or that every other benchmark is affected. Do not label these stored reference
curves verified ground truth for intervention discrimination yet.

## Recommended next work for the coding agent

1. Independently reproduce the actual CSTR generator with the existing private
   generator configuration in an isolated diagnostic location. Instrument RHS
   call times; compare with integration restarted at every known input jump.
   Preserve all historical files. Keep private reference details out of
   proposer inputs and model-selection feedback.
2. Verify discontinuity handling analytically, then audit every benchmark that
   shares `_integrate` and every dataset/export derived from affected protocols.
   Include the prediction integrator in the audit; do not assume it is immune.
3. Report the affected dataset identities and splits before proposing a repair.
   If confirmed, create a separately versioned corrected benchmark and rerun
   affected methods under identical data. Replacing reference lines or scoring
   old fitted models alone does not repair training performed on bad data.
4. Until then, use these plots only to inspect the historical experiments.
   Do not build the paper's intervention claim around the seed-0 aggregate
   advantage. Preserve the contrary seed-1 result in any comparison.

## Plotting and remaining comparisons

`trajectories.csv` has columns `seed, arm, split, trajectory_id, target, time,
observed, predicted, residual, training_scale`. `residual = predicted-observed`.
`inputs_auxiliaries.csv` contains the exact public input and supplied auxiliary
arrays. These are conditional `T` rollouts using `C(t)` and `Tj(t)`; they are
not autonomous reconstructions of all physical states. Validation was used
for selection. No latent does not automatically mean mechanism noncompliance.

No CSTR No specification run exists in this campaign. The archive also lacks
rejected-candidate histories and external-baseline model artifacts, so none
of those curves has been invented or inferred from scalar scores. Retrieve
those exact models and contexts separately if still needed after the data audit.

## Code and verification

Committed and pushed replay/plot scripts, focused tests and documentation as
`c7d1b84` on `codex/prefit-aces-v1`. Generated artifacts and this local
communication note are not committed.

- Four focused tests passed, including an analytic causal-initialization
  rollout, exact resume without simulation, complete scoring, and unsafe-archive
  rejection.
- All 80 real replay trajectories succeeded; all eight saved aggregate scores
  matched. Running the replay again reused the completed checkpoints exactly.
- PNG contact sheets and the original-forcing/reference comparison were
  rendered and visually inspected; CSV, SVG and a checksummed ZIP are included.
- Focused Ruff checks passed. Repository-wide `ruff check .` reported 37
  existing issues in unrelated `analysis/claude` files; these were not edited.
- The repository-wide pytest run was interrupted after 1502 passed, eight
  skipped (Torch unavailable), one failure and one setup error. Both diagnostics
  were source/freeze identity mismatches while independent work changed the
  shared source tree. The two affected tests passed individually afterward
  (`2 passed in 10.48s`). This is not a claim that the full suite passed.
