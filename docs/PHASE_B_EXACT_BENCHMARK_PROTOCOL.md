# Phase-B exact benchmark protocol

## Status and scope

This document freezes the pre-generation protocol for Phase-B v1. It does not
modify or replace historical benchmark data or prompts. New public assets may
be generated only after their private simulations pass the gates below and
before any discovery method is evaluated.

Every family has two mechanism/observability difficulty tiers. Semantic controls are
orthogonal:

- Dalla Man and CSTR: named versus obfuscated;
- alien device: functional task description versus opaque task description.

Semantic pairs use identical numerical data, channel roles, and splits. The
alien functional condition describes roles and required behavior but never
reveals the generated equations, coefficients, private state names, or graph.
Its purpose is task-information ablation, not detection of model retrieval.

### Why the complete matrix has 40 cells

There are six scientific tasks, but ten task-dynamics cases because the four
Dalla Man tasks are each simulated under canonical and perturbed dynamics:

- Dalla Man: `4 tasks x 2 dynamics x 2 semantic variants x 2 tiers = 32`;
- CSTR: `1 task x 2 semantic variants x 2 tiers = 4`;
- alien device: `1 task x 2 semantic variants x 2 tiers = 4`.

Thus the suite has 40 benchmark cells before methods or seeds. “Ten cases”
means eight Dalla task-dynamics cases plus CSTR and alien device; it does not
mean ten distinct task specifications.

## Common split and evaluation rules

| Item | Easy | Hard |
|---|---:|---:|
| Training trajectories | 16 | 16 |
| Validation trajectories | 4 | 4 |
| Test trajectories | 6 | 6 |
| Mechanistic burden | Focused/task-local | Coupled/end-to-end |
| Public dynamic information | Richer task-relevant channels | Minimal channels that pass identifiability gates |
| Excitation and input schedules | Identical across tiers | Identical across tiers |

Rules:

1. Structure and global parameters are shared across every trajectory.
2. Validation protocols interpolate between training regimes. Test protocols
   include prespecified input and initial-condition extrapolations.
3. Semantic pairs share row-for-row numerical data and differ only in public
   names/descriptions.
4. Easy and hard use the same trajectory identifiers, input schedules, initial
   conditions, sampling, duration, and noise condition. They differ only in
   required mechanisms and public targets/auxiliaries.
5. Training/validation may be regenerated only from the prespecified simulator
   design and before method results are inspected. Test assets are generated
   once and sealed.
6. Primary evaluation is unseen-condition free rollout. Fixed horizons and
   one-step prediction are secondary and diagnostic, respectively.

Unless a task-specific row overrides them, Dalla Man trajectories last 300 min
at 1 min sampling. CSTR lasts 30 normalized time units and alien device lasts
60, both at 0.1-unit sampling.

Durations are system-specific rather than numerically matched across unrelated
time units. Dalla Man contains slow insulin-delay and action rates
(`ki=0.0079/min`, approximately 127 min, and `p2U=0.0331/min`, approximately
30 min) in addition to meal absorption (`kabs=0.057/min`, approximately 18 min),
so 300 min is needed to observe both the event response and substantial
recovery. The CSTR's principal flow/exchange rates are approximately 1--2.5 per
normalized time unit, while the selected alien system's decay rates are
approximately 0.069--0.175, corresponding to time constants of roughly
6--15 units. Their 30- and 60-unit trajectories therefore already span several
dominant time constants. Final durations must be confirmed by simulator-only
response/recovery diagnostics; reaching exact equilibrium is not required.

### Free simulation versus rolling-origin K-step-ahead prediction

The two metrics answer different questions and use different initialization
frequency.

For an `H`-sample rolling-origin forecast, every admissible sample `i` is a
forecast origin:

1. observations through and including sample `i` are available;
2. candidate states directly corresponding to public observed channels are set
   to their observed values at `i`;
3. private latent truth is never supplied. The model's latent state at `i` is
   obtained by causally propagating it from the trajectory start while using
   only observations available through `i`;
4. a cloned model state predicts samples `i+1` through `i+H` without target or
   observed-state resets;
5. declared future external inputs and any full-horizon auxiliary trajectories
   listed as supplied by the tier remain available.

This is repeated through the last origin with a complete `H`-sample future. If
samples are numbered 1 through 300 and `H=30`, origin 1 predicts 2--31, origin
2 predicts 3--32, and origin 270 predicts 271--300. Following the standard
K-step-ahead convention, the headline `K=30` error compares the lead-30 endpoint
at each origin, then averages over origins and equally over trajectories. The
train-derived normalization scale is fixed for all horizons. We additionally
retain the lead-wise curve for `1,...,30` and its mean as an integrated-horizon
diagnostic, but do not call that mean the 30-step-ahead NMSE. Overlapping origins
are part of the forecasting estimand; uncertainty calculations cluster by
trajectory rather than treating origins as independent replicates. Model
parameters and structure remain frozen at every origin; only causal state
estimation is updated.

For **unseen-condition free simulation** (also called free-run or
infinity-step-ahead simulation), the model receives the held-out
trajectory's public initial observations at sample 1 and its declared latent
initialization. It then predicts samples 2--300 in one uninterrupted simulation
without any target/observed-state reset. Declared external inputs and supplied
auxiliary trajectories remain available, but no future target does. NMSE is
calculated over all predicted samples. This is the primary test of long-term
conditional-autonomous validity and numerical stability.

Thus K-step prediction measures repeated operational forecasting after causal
state estimation, whereas free simulation asks whether one initialization
can support the entire unseen trajectory. A 100%-length rolling-origin window
exists only at the first origin and coincides with free simulation; shorter
rolling-origin forecasts do not.

These names follow established forecasting and system-identification usage:
rolling-origin evaluation moves the forecast origin across time; K-step-ahead
prediction uses measured history before that origin; and simulation uses only
inputs plus initial conditions, without later measured-output correction.

References:

- Tashman (2000), “Out-of-sample tests of forecasting accuracy,”
  <https://doi.org/10.1016/S0169-2070(00)00065-0>.
- MathWorks System Identification documentation, “Simulate and Predict
  Identified Model Output,”
  <https://www.mathworks.com/help/ident/ug/definition-simulation-and-prediction.html>.
- Brunton, Proctor, and Kutz (2016), “Discovering governing equations from data
  by sparse identification of nonlinear dynamical systems,”
  <https://doi.org/10.1073/pnas.1517384113>.
- d'Ascoli et al. (2024), “ODEFormer: Symbolic Regression of Dynamical Systems
  with Transformers,” <https://arxiv.org/abs/2310.05573>.

## Dalla Man tasks

Canonical and perturbed dynamics use the same protocols. Each is released in a
named and a deterministically obfuscated representation.

### Public channels and required mechanisms

| Task | Easy: targets; auxiliaries | Easy required mechanisms | Hard: targets; auxiliaries | Hard required mechanisms |
|---|---|---|---|---|
| T1 meal appearance | `Gp`; `EGP,Uii,E,Gt` | Causal meal memory and appearance contribution | `Gp`; `Gt` | Meal memory plus a closed glucose balance sufficient for `Gp` |
| T2 absorption action | `Gp,I,U`; `EGP,Uii,E,Gt` | Delayed insulin-dependent disposal | `Gp,I`; `Uii` | Joint meal appearance, delayed disposal, and glucose balance |
| T3 hepatic regulation | `Gp,I,EGP,U`; `Uii,E,Gt,Ipo` | Delayed hepatic regulation separated from disposal | `Gp,I,EGP`; none | Joint meal, disposal, hepatic regulation, and observed balances |
| T4 flux portrait | `Gp,I`; `Uii,E,Gt,Ipo` | Coherent `Ra,U,EGP,S` flux portrait | `Gp,I`; none | Coupled task-compatible flux portrait; uniqueness is not claimed |

T1 hidden evaluation is input-output equivalence of the meal kernel. T2 and T3
use identifiable hidden-subspace recovery. T4 uses flux compatibility and
intervention behavior rather than coordinate-wise recovery.

### Dalla Man intervention schedules

Both tiers use these same 16 training trajectories:

1. basal, no meal;
2. single meals of 30, 60, 90, 120, and 150 g at minute 0;
3. 90 g meals at minute 60 and minute 120;
4. two-meal schedules 60+30 g at minutes 0/90, 45+45 g at 0/60, and
   30+60 g at 0/120;
5. initial `Gp` shifts of -10% and +10%, with the simulator's compensating
   tissue-state adjustment;
6. initial plasma/liver-insulin shifts of -15% and +15%;
7. one task-specific orthogonal intervention: insulin pulse for T2, insulin
   pulse plus clamp-like glucose forcing for T3/T4, and a third separated meal
   for T1.

If the hard public-channel/mechanism design fails a task's identifiability gate,
the tier is not released. The remedy is chosen without discovery-model results:
add the minimum task-relevant public channel or weaken the recovery claim to an
identifiable subspace. The input schedules are not changed independently by
tier.

Validation uses four unseen interpolations: 75 g at minute 30, 105 g at minute
90, 30+75 g at minutes 0/90, and a combined +5% `Gp`/-7.5% insulin initial
shift. Test uses six extrapolations: 20 g, 180 g, 90 g at minute 180, three
30 g meals at minutes 0/60/120, combined -12% `Gp`/+20% insulin, and a stronger
task-specific orthogonal intervention.

For T3/T4, insulin and clamp-like interventions are declared external inputs;
their private induced fluxes are never supplied. For T4 easy, the listed
auxiliaries are supplied trajectories, not private target fluxes.

The private generator represents these orthogonal interventions in explicit
model-native derivative units: glucose forcing is in `mg kg^-1 min^-1` and
enters `dGp/dt`, while insulin forcing is in `pmol kg^-1 min^-1` and enters
`dIp/dt`. Every piecewise-constant segment is recorded in the private protocol
manifest. These are declared public inputs for tasks that use them, not hidden
reference fluxes.

## CSTR task

The private reference remains the three-state controlled reactor with
concentration `C`, reactor temperature `T`, jacket temperature `Tj`, and inputs
`Cf,Tf,Tjf`. Named and obfuscated versions share identical data.

| Tier | Targets | Auxiliaries | Required mechanisms |
|---|---|---|---|
| Easy | `T` | `C,Tj` | Feed-temperature contribution, reaction heat contribution, and jacket exchange in the `T` balance |
| Hard | `T` | none | Coupled reactant/reaction dynamics, reactor heat balance, jacket dynamics, and three input pathways |

CSTR mechanism presence and intervention behavior are evaluated component-wise,
but quantitative hidden recovery is claimed only on the two-dimensional
locally identifiable mechanism subspace. The private pre-release audit found
that the remaining mechanism directions are individually consequential under
ablation but too weakly conditioned for unique coordinate-wise recovery, even
when all three private states are inspected.

Inputs are expressed in normalized admissible coordinates `[-1,1]`, then
mapped deterministically to simulator-safe physical ranges. Both tiers use the
same 16 training trajectories: zero input; positive/negative single-input steps for all
three inputs; single-input pulses for all three; slow sinusoids for all three;
two pairwise step combinations; and one initial-state shift along the dominant
sensitivity direction.

Validation uses four unseen intermediate-amplitude step/pulse combinations.
Test uses six cases: amplitudes 20% outside the training envelope, a faster
multisine, a delayed pulse, an opposing `Tf/Tjf` combination, an initial-state
extrapolation, and a combined input/initial-condition shift.

## Alien-device task

The system is procedurally generated and screened without using discovery
performance. The functional prompt may describe input memory, persistent
internal response, nonlinear feedback, and output generation only when those
roles are verified from the private generator. The opaque prompt states the
same output obligation and available symbols but removes those functional
roles. It does not serve as an anti-memorization condition.

| Tier | Targets | Auxiliaries | Required mechanisms |
|---|---|---|---|
| Easy | output `y` | two sensitivity-selected telemetry channels | Input-driven memory and its causal contribution to `y` |
| Hard | output `y` | none | Input memory, persistent latent coupling, nonlinear feedback, and output generation |

The alien hard tier evaluates quantitative recovery on its three-dimensional
identifiable mechanism subspace. All four required mechanisms remain separate
structural and intervention criteria; the fourth local direction is
consequential under ablation but dominated in the output-sensitivity spectrum.

The semantic pair within each tier is numerically identical. Inputs use the
normalized range `[-1,1]`. Both tiers use the same 16 training trajectories:
zero input; four
four-unit pulses with amplitudes `-1,-0.5,0.5,1`; three steps at
`-0.75,0.5,1`; three sinusoids with periods 8, 16, and 32; one linear chirp
from period 32 to 8; two separated multi-pulse schedules; and two
sensitivity-selected initial-state shifts.

Validation uses four unseen pulse-amplitude/duration and sinusoidal-frequency
interpolations. Test uses six cases: a 20% amplitude extrapolation, unseen fast
frequency, chirp, delayed multi-pulse schedule, initial-condition
extrapolation, and combined input/initial-condition shift.

## Pre-release identifiability and shortcut gates

All gates use only the private simulator and training/validation designs. No
discovery-method output or test trajectory is inspected.

| Gate | Easy threshold | Hard threshold |
|---|---:|---:|
| Scaled output-sensitivity rank | Full claimed rank at relative singular-value threshold `1e-3` | Same |
| Sensitivity condition number on claimed subspace | <= 1,000 | <= 5,000 |
| Stable rank | >= 50% of claimed dimension | >= 35% of claimed dimension |
| Required-mechanism ablation separability | Validation discrepancy NMSE >=0.20 for every ablation | >=0.15 |
| Input-design Gram minimum/maximum eigenvalue | >= `1e-3` | Same shared design |
| Persistence 30-min NMSE, Dalla Man | >=0.25 | >=0.25 |
| Persistence full-horizon NMSE, CSTR/alien | >=0.5 | >=0.5 |
| Finite reference rollouts | 100% | 100% |

For equivalence-class tasks, “claimed rank” refers to the prespecified
identifiable mechanism subspace, not all private simulator parameters. A tier
that fails is redesigned before release; thresholds are not relaxed after
observing method performance.

Here ablation discrepancy NMSE compares the ablated private simulator against
the intact private simulator on validation protocols, using channel scales
estimated from intact training trajectories. It is an absolute normalized
separation criterion, not a percentage increase relative to the intact
simulator's zero self-error.

## Implementation and release state

The executable private-generation path is
`src/autoformalism/benchmarks/phase_b_generation.py`, with command-line entry
point `scripts/generate_phase_b_private_references.py`. It constructs the
frozen 16 training, 4 validation, and 6 test protocols for each family and
executes trusted Dalla Man, CSTR, and alien-device simulators. Generated
bundles are explicitly marked private and unavailable to discovery methods.

Test generation is not the command-line default. `--seal-test` must be used
only once after all pre-release audits pass. The basic report deliberately
fixes `standalone_release_ready=false`, so an executable simulator is never
mistaken for a released benchmark. The task-specific audit subsequently passed
all gates for all 20 distinct numerical cells; details and prespecified design
remedies are recorded in `PHASE_B_PRE_RELEASE_AUDIT.md`.

The non-registered public staging path is
`src/autoformalism/benchmarks/phase_b_public.py`, with command-line entry point
`scripts/stage_phase_b_public_assets.py`. Each package contains typed public
channel roles, proposer and judge prompts, one tidy CSV per available split,
file hashes, and a channel-name-independent numeric commitment. Named and
obfuscated Dalla Man/CSTR pairs preserve the same task burden in semantic or
anonymous language. Functional and opaque alien-device pairs differ in whether
verified functional roles are stated. Paired variants are projected from the
same private arrays.

Static proposer prompts use a natural domain-expert framing and retain a common
six-section scaffold: task specification, available data, modeling
requirements, constraints/plausibility, intended use, and required response.
They say that the model should explain mechanisms and generalize to new inputs
and initial conditions, but do not expose implementation-specific numerical
fitting or rollout machinery. The controller's iterative feedback—not the
static domain prompt—must label every numerical result explicitly as
free-rollout NMSE or a stated fixed-horizon error. All cells use one identical
judge prompt and rubric.

Sections A and B retain benchmark-specific prose where it improves scientific
clarity. Requirements identify the task-relevant causal roles without revealing
a more detailed equation graph in a semantic-control prompt than in its named
counterpart. The natural-language static prompt states the scientific objective
without duplicating controller-level evaluation instructions. Named Dalla Man
and CSTR channels include declared physical or relative units. Obfuscated
channels omit units because they could reveal the source model. This is an
intentional semantic-control difference; prompt organization, response
requirements, data, and judge instructions remain paired.

Public CSVs expose only `trajectory_id`, `t`, and canonical numeric channels
declared as targets, auxiliaries, or external inputs. Structured schedule
objects, duplicate encodings such as `meal_schedule`, and descriptive protocol
labels are excluded. Split-local trajectory identifiers such as `train_000`
carry no intervention semantics. This removes a recurrent source of invalid LLM
variables at the data boundary rather than relying solely on parser repair.

The staging writer omits test by default, excludes the private-to-public channel
mapping from serialization, labels the package `staging_not_registered`, and
runs a fail-closed leakage audit. The audit rejects undeclared CSV columns,
private mappings, private specification filenames, hidden dimensions, private
state names, and semantic family terms in obfuscated/opaque assets. Public test
sealing and production-registry integration remain separate final milestones.
`scripts/audit_phase_b_public_suite.py` staged all 40 train/validation cells,
verified every leakage report, checked name-independent numeric commitments for
all semantic pairs, and confirmed that zero test cells were sealed.

Manual review of all 40 proposer prompts and the shared judge prompt is now
complete. Fresh generator output is byte-identical to every reviewed prompt.
The functional/opaque Alien Device pairs preserve the same tier-specific task
burden; opaque target `v01(t)` is identified only as the primary output, and no
private state identity, equation, or application domain is disclosed.

## Optional noise robustness and replication

The primary suite uses deterministic clean simulator trajectories for
structural comparison. Noise is not part of easy/hard and is not required for
the primary matrix. If a separate robustness appendix is run, it applies the
same frozen channel-scaled noise level to both tiers and semantic variants.
Noise seeds remain separate from proposal, fitting, and data-design seeds.

Run inexpensive methods across the complete 40-cell factorial first. Hosted-LLM
runs use sequential seeds and prespecified stopping intervals. The untouched
confirmatory subset is frozen only after the data, prompts, rollout-aware
selection, and identifiability reports pass review.
