# M31: fitting feedback and controlled diagnosis qualification

Implemented October 13, 2026. This is an opt-in evidence interface, not a new
optimizer or an automatic specification-revision algorithm. It follows the
[M30 review](PHASE_C_FITTING_M30_RESULTS_2026-10-12.md) and prepares the numerical
boundary described in the updated [fallible-specification handoff](VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md).
Construction prompts, scientific requirements, benchmark data and production
defaults are unchanged. No experiment is submitted by the agent.

## Question and scope

Can construction distinguish an accurate fit, ambiguous parameters, an unfinished
search, a locally stationary search, an unavailable rollout, and a restricted
output problem whose best achievable error remains poor?

An accurate fit is an adequate witness for those equations on those observations.
An established poor optimum concerns the fixed candidate's output equations and
declared domains. Neither one establishes the truth or falsity of a natural-
language specification. In particular, a construction that omitted drainage can
fail even when the specification correctly requires drainage.

M31 supplies training-only evidence and advisory next actions. It does not launch
LLM calls, alter equations, add physical constraints, change parameter values, or
read validation/test observations. The future constructor must receive executable
equations and observation maps before using numerical feedback; topology alone
is insufficient. Claims-to-equation bindings remain a later construction boundary.

## Candidate-bound interface

`autoformalism.fitting.fitting_feedback.run(problem, fit, backend, folder, policy)`
accepts the existing typed `TrainingProblem` (or explicit piecewise variant),
`FitResult`, and its sealed backend. It rejects mismatched training problems,
parameter vectors, policies, starts and result hashes. Its durable packet binds:

- The request, lowered candidate, training arrays, retained vector, original fit
  and backend, fitting policy, diagnostic policy, code and numerical runtime.
- Historical fitting costs separately from new diagnostic work.
- Current numerical accuracy separately from historical accuracy flags.
- A saved search witness only when its vector matches the retained vector.

The packet contains no validation scores, private coefficient errors or oracle
fixture labels. `specification_verdict` is always `not_assessed`, automatic
specification revision is false, and confidence probability is null. The JSON
schema is `schemas/fitting-feedback-v1.schema.json`.

| Diagnosis | Advisory action and scope |
|---|---|
| `accurate_local_evidence` | Retain with local sensitivity evidence; no global uniqueness or coefficient-accuracy claim. |
| `accurate_ambiguity` | Examine weak active directions or independently verified alternative vectors. |
| `accurate_unresolved` | Predictions are accurate; parameter information is unavailable or derivative scope is insufficient. |
| `solved_restricted_problem_poor_fit` | Review these equations; an all-inner/fixed-output witness concerns only their restricted output problem. |
| `stationary_poor_fit` | Consider equation review or different starts; local stationarity is not a global optimum. |
| `search_improving` | Consider continuing from the retained point; recent saved progress is not a promise of eventual recovery. |
| `search_incomplete` | Diversify or extend search; no adequate solution or impossibility certificate was found. |
| `rollout_unverified` | Check numerical execution or input contracts; no scientific verdict. |

No retained vector means no fabricated residual or parameter score. A source
construction failure remains `source_unavailable` in the campaign denominator.
An optional new diagnostic timeout does not erase an independently verified
historical incumbent; parameter information then remains unavailable. A completed
new replay that disagrees with the historical metrics triggers numerical review.

## Diagnostic computation and limitations

Each retained vector receives one additional allowance of 180 seconds, covering
symbolic setup and up to three fixed-vector evaluations: DOP853 outputs, Radau
outputs, and a joint output Jacobian. Relative/absolute tolerances are
`1e-12`/`1e-14`. The existing deadline checks apply; the scheduler additionally
bounds the worker. This is not a new optimization allowance.

The two rollouts must agree before residuals support numerical conclusions.
Reproducing historical metrics within tolerance is not sufficient for an accuracy
label: aggregate and worst-trajectory fitting targets and solver agreement are
checked on the new rollout. A failed derivative check preserves completed output
checks while leaving sensitivity unavailable.

Residual summaries contain per-target NMSE, signed normalized mean error, the
worst trajectory, and the largest residual's time/location. They use the existing
training-only pooled standard deviation, floored at `1e-12`, and residual sign
`prediction - observation`. They do not send every sample or infer which term
caused an error. No peak/decay causal explanation is manufactured from these
statistics.

AST-based output dependencies include expanded processes, upstream states and
causal initial-map parameters. Certified output-inactive parameters are listed
separately. They are not estimated, removed, or counted as weak active directions.
Small/canceling sampled sensitivities never remove a parameter. Reachability is
syntactic and can overestimate dependence; it does not certify required mechanisms
or feasibility of disconnected dynamics.

The scaled Jacobian includes active physical parameters and shared initializer
coefficients jointly. Rank, singular values, condition number and a weak direction
are local screens. Actual alternative vectors from independent saved checks can
provide ambiguity witnesses; absence of alternatives does not prove uniqueness.
Piecewise branch derivatives are explicitly labeled and cannot alone produce an
unqualified local-sensitivity success. Full profile minimization, uncertainty
calibration, noisy-data stopping, and claim-level identifiability are not added.

## Frozen qualification roster

There are sixteen tasks, all with one CPU and no GPU:

1. Eleven entries from M30's `output_aware` arm, including the two construction
   failures and the candidate without a retained finite endpoint. The nine
   available fitting handoffs are imported without refitting. These are September
   30 archived constructions and known controls, not Astra's current topologies.
2. Five small newly synthesized basin controls. These use analytic training
   observations and genuine bounded fitting, not mocked fitting outcomes.

| Synthetic control | Intended distinction |
|---|---|
| Adequate drainage equation | Accurate prediction and locally resolved gain/rate. |
| Same observations; candidate omits drainage | Restricted output optimum is poor although an adequate specification-compatible model exists. |
| Two parallel inflow gains | Accurate prediction with only the gain sum identifiable. |
| Correct nonlinear drainage; eight-call search | Computational uncertainty, with an independently checked adequate witness kept outside the fitter. |
| Explicit synthetic no-drainage claim | Same candidate fitting evidence as omitted drainage; a separate reviewed monotonicity argument contradicts this narrowly defined claim family. |

The false-claim control explicitly stipulates `h'=g*u`, `g>=0`, nonnegative input
and directly observed depth. Every admissible trajectory is nondecreasing.
Decreasing noiseless observations therefore contradict this class independently
of whether optimization failed. That certificate is stored only in evaluator
labels; the fitting packet still says `specification_verdict=not_assessed`.
The omitted-drainage and false-claim controls intentionally give identical inputs
to the fitter. A fitting algorithm cannot distinguish their external claim
contexts from numerical error alone.

The synthetic fit ceilings are four blocks, ten calls and five seconds per block
(two calls for the eight-call control), with separate five-second verification
and sensitivity allowances. They do not change the imported campaign's policies.
Expected outcomes, adequate witness coefficients and analytic optima never enter
the feedback function. Reference witnesses qualify the fixtures; they are not
discovery results or extra benchmark wins.

## Accounting, resume and exit criteria

Plans and results are sealed. Resume verifies identities and never renews an
interrupted diagnostic allowance. Unknown time/calls stay unknown and the full
allowance remains charged. Completed diagnostic reuse performs no new fitting
or replay. Historical artifacts are never edited.

Reports preserve missing and unavailable entries. They separate diagnostic costs,
new synthetic fitting costs, and historical fitting costs. `status=complete`
means accounting records exist; qualification additionally checks expected
fixture diagnoses and packet boundaries. A completed campaign can fail its
qualification. No broad efficiency or scientific-validity claim follows from
passing these interface controls.

Exit criteria: all sixteen entries accounted for; five controlled diagnoses
match their evaluator-only expectations; no packet accuses or revises a
specification; parameter values remain fixed during feedback; resume and source
identity checks pass; unknown usage is not converted to zero. Investigate any
lost numerical accuracy before connecting this interface to live construction.

## Delta commands

Upload `transfers/phase-c-fitting-m31.tar.gz` into
`/work/hdd/bibo/yxiao2/phase_c/`. The delivery contains committed source, tests,
configuration, `SOURCE_COMMIT`, `SHA256SUMS`, and the frozen training-only M30
export. It needs neither an API key nor a new Python environment.

```bash
cd /work/hdd/bibo/yxiao2/phase_c
tar -xzf phase-c-fitting-m31.tar.gz
bash phase-c-fitting-m31/scripts/hpc/submit_phase_c_fitting_feedback_delta.sh
```

Default output: `/work/hdd/bibo/yxiao2/phase_c/fitting-feedback-v1`.
The CPU array has sixteen tasks, three concurrent, a 30-minute scheduler ceiling,
and 16 GB per task. The ceiling is not a requested optimization duration.
`AF_CONCURRENCY` changes this array's throttle. Existing confirmed submissions
resume; unconfirmed scheduler replies require reconciliation before resubmission.

Without a packaged input file, the launcher can export from
`AF_FEEDBACK_SOURCE` (default `fitting-poor-fit-v1` on Delta). `AF_FEEDBACK_INPUTS`
can name an existing frozen export. Export copies only training problems and
fitting records; it does not regenerate data or copy post-fit evaluation metrics.

Inspect partial or final results and create the review archive:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m31/scripts/hpc/inspect_phase_c_fitting_feedback_delta.sh
```

Download the exact `review-*.tar.gz` path printed by that command.

## Verification

Local tests exercise real small fits, exact output profiling, active/inactive
separation, shared-initializer and multi-output dependencies, ambiguity, restricted
claim certificates, candidate/point/policy mismatch, replayed accuracy thresholds,
piecewise derivative scope, interrupted/unknown accounting, deterministic resume,
source mutation, and fail-closed submission intent handling. Full archived-model
diagnostics remain the user-run Delta qualification. No production fitting or
construction defaults are changed.

The relevant fitting regression selection passed 99 tests. After the final
interface refinements, the three new test modules passed 36 tests. Scoped Ruff,
CLI smoke checks and shell syntax checks passed. Repository-wide `ruff check .`
still reports 37 unrelated pre-existing findings in `analysis/claude/`.
An archived independent-basin smoke replay completed with the expected restricted
poor-fit diagnosis; all eleven archive entries export successfully. These checks
do not replace the sixteen-task Delta qualification.
