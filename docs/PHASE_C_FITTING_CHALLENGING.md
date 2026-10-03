# Phase C fitting M6: harder fixed-equation cases

Approved 2026-10-02. Protocol `phase-c-fitting-challenging-1`.
Config: `configs/phase_c_fitting_challenging_v1.json`.

## Question

M5 established that caching the formulation preserves cold-solve results and
that good primal starts can reduce repeated-solve work. It did not exercise
screening's veto: every first solve was already successful or acceptable.
M6 keeps all four numerical strategies and both primal-transfer policies,
and moves to a larger nonlinear hidden-state problem. It does not assume that
screening will win or that any particular start will produce a bad checkpoint.

This is a fitting study with supplied correct equations. No construction, LLM
calls, test access, new benchmark trajectories or production-default changes.
The unchanged Phase C development-2 release supplies all observations and inputs.
Do not compare this assisted study to autonomous model-discovery results.

## Cases and fitted quantities

| Case | States | Fitted quantities | Development trajectories |
|---|---:|---|---|
| Coupled noiseless basin | 2; downstream observed | 2 hydraulic coefficients; public initial gauge | 6 training, 3 validation |
| Alien-device hard | 6; only output observed | 13 dynamic factors and 5 shared latent initials | 16 training, 4 validation |

Alien-device has coupled latent dynamics, saturating nonlinearities and product
interactions. The 13 free dynamic factors are six decay magnitudes, two input
gain magnitudes, one input saturation scale and four output gain magnitudes.
Outer signs stay fixed. The five latent initials are signed, shared fitted values;
the observed output starts at its measured initial observation. No hidden initial
value or hidden trajectory is supplied to an optimizer.

Internal linear couplings, internal nonlinear gains, and internal/output tanh
shapes remain fixed at the supplied equation witness. This is substantial,
explicit assistance. It anchors the latent coordinates and avoids testing an
unrestricted parameterization with a latent-scale gauge. We are estimating a
specified parameter block conditional on the fixed block, not recovering every
coefficient of the original generator. The input tanh scale remains unknown,
so even the dynamic parameter block is not wholly linear conditional on states.

There are 601 samples per alien training trajectory. Rollout fitting optimizes
18 variables. Initial two-node Radau collocation additionally optimizes 115,200
latent/observed nodal variables and imposes 115,200 dynamical equalities before
parameter bounds. Initial values are functions of the 18 fitted quantities and
observations, not extra per-trajectory parameters. Actual variable/constraint
counts, native iterations, mesh refinements and formulation times are retained
by each backend. This is a meaningful size increase over the small controls.

## Qualification before fitting

The exporter checks the release plan/summary seals and the byte hashes of public
train/validation/specification/prompt assets and the alien equation witness.
It lifts numeric factors through parsed expression trees and preserves all other
terms. Tests compare the lifted and original vector fields at matched parameters.
It never searches for a model or selects a start using reference error.

The Delta preparation job repeats two gates:

1. At reference parameters, both development splits must replay with NMSE at most
   1e-6; independent Radau and DOP853 solutions must agree within 1e-4 training
   standard deviations.
2. Output sensitivities to **all fitted coefficients and initials**, under the
   actual training inputs, must have full column-normalized numerical rank with
   smallest/largest singular-value ratio greater than 1e-8.

Local preflight found full rank in both cases: ratio 0.006420 for alien-device
and 0.876688 for basin. These are local excitation checks, not proofs of global
identifiability, uniqueness, or robustness to noise. Ill-conditioning away from
the reference and nonconvex optimization can still matter. A failed gate blocks
the fitting array; it is not silently removed from the roster.

Independent local replay of the lifted requests at reference parameters also
passed: alien train/validation NMSE 2.50e-18/4.31e-19 and maximum solver difference
4.09e-9; basin 2.42e-17/3.68e-18 and 2.77e-6. These are attainable-reference
checks, not fitted results. The sealed input content hash is
`97aacd4d137ad3631b11798795ad857d4febcae5780565d351277806119adfc6`.

## Matched strategies and starts

The two cases, three starts, and six arms give **36 CPU tasks**:

1. `rollout_only`: optimize coefficients and initials through accurate integration.
2. `collocation_only`: direct collocation with existing adaptive-mesh policy.
3. `adaptive_shooting`: multiple shooting with the existing integration/mesh policy.
4. `collocation_rollout`: collocation initialization followed by rollout refinement.
5. `cache_last_primal`: two fixed-mesh solves, transferring the latest finite,
   parameter-bounded primal checkpoint.
6. `cache_screened_primal`: the same two-solve schedule, transferring the best
   checkpoint satisfying the M5 training-rollout and scaled-defect checks;
   otherwise reuse the original cold start.

The last two import no duals. Their comparison isolates the primal screening
policy. Comparisons to the first four are comparisons of numerical strategy
bundles, since native schedules differ. The old cache-only/rebuild arms remain
implemented but are not repeated in this roster; M5 already isolated that cost.

Generic alien guesses are 0.1 for decays, 0.5 for gains and 1 for the input scale;
basin guesses are 30 for each coefficient. Three deterministic starts multiply
these by exp(Uniform(-w,w)), with w=0.5,1,1.5. Alien latent initials are drawn from
Uniform(-w,w), independently of reference values. Seeds are 20261002+i. These are
progressively broader starts, not guaranteed orderings of optimization difficulty.
All six methods get the same start, coordinates, physical node guesses and data
within each pair. Every start is reported; no best-of-three selection.

## Budgets and interpretation

- Each fit has a 900-second total wall ceiling including construction, screening,
  native work and checkpoint handling, and at most 900 counted rollout calls.
- The two cache arms each allow two native solves of at most 300 seconds and
  200 iterations, with 20-second per-point screens, within that same total ceiling.
  They retain M5's scaled-defect limit 1e-6 and 0.1% improvement over the original
  training score. A nodally inconsistent checkpoint may still be retained as a
  final parameter endpoint if its actual rollout is best.
- Independent endpoint replay gets a separate 300-second evaluation allowance.
  Validation and reference values are absent from fitting worker payloads.
- Delta allocates one CPU and 16 GiB per task, at most two tasks concurrently,
  with 25-minute fit allocations. There are no GPUs or API credentials.
- Completed results are reused. An interrupted fit keeps its artifacts and becomes
  an interrupted endpoint; a rerun cannot reset its consumed budget. A completed
  backend can resume independent scoring. An uncertain scheduler response blocks
  resubmission until the existing job identity is reconciled.

Report train/validation NMSE, unavailable rollouts, all dynamic coefficient errors,
shared-initial absolute errors, native status, budgets, and timing separately.
Prediction thresholds are 1e-6 for basin and 0.01 for alien, with independent
solver agreement required. All dynamic coefficients must have relative error at
most 1% for coefficient recovery. The zero-valued alien reference initials use
absolute error at most 0.01 in the anchored latent coordinates; relative error
at zero is undefined. These numerical thresholds are frozen reporting criteria,
not scientific guarantees. Hidden-trajectory accuracy is not reported by M6.

Inspect the cache arms' saved `warm_start_decision.json` and `attempts.json`:
did screening actually veto a checkpoint, why, and did the fallback help? If no
veto occurs, record that limitation rather than claiming protection was tested.
The two-solve diagnostic deliberately keeps both attempts. Skipping an already
adequate second solve and general production scheduling remain subsequent work.

## Delta commands

Upload the supplied `phase-c-fitting-m6.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`, then:

```bash
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/fitting-m6-code"
tar -xzf "$AF_BASE/phase-c-fitting-m6.tar.gz" -C "$AF_BASE/fitting-m6-code"
bash "$AF_BASE/fitting-m6-code/scripts/hpc/submit_phase_c_fitting_challenging_delta.sh"
```

The wrapper uses the existing Delta fitting Python and CasADi dependency paths.
It prints their versions, checks bundle hashes and freezes the current runtime.
Only set `AF_PYTHON`/`AF_CASADI_ROOT` if those installations moved. Do not install
new dependencies into an environment used by running jobs.

After completion:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/fitting-m6-code/scripts/hpc/inspect_phase_c_fitting_challenging_delta.sh
```

This prints scheduler status and group summaries, and writes a timestamped review
archive for download under `fitting-challenging-v1`. Use a new `AF_CAMPAIGN` only
for an explicitly changed experiment, not to bypass uncertain submissions or
interrupted budgets. No cluster session or submission was performed locally.

Exit: account for all 36 endpoints, inspect prediction/parameter/initial recovery
and screening decisions for every start, and explain numerical failures before
changing equations or promoting a fitter.
