# Joint and alternating collocation experiment

Protocol `fitter-methods-4` tests conditional linear coefficient fitting inside
collocation on the existing synthetic recovery fixtures. It is an experimental
adapter, not a production fitter change. The proposer, scientific equations,
parameter signs, zero initial conditions, benchmark data and finalized prompts
are unchanged. No LLM calls, hidden-state labels or test data are used.

## Question and controls

The previous Delta portfolio experiment recovered the observed trajectories in
12/12 C+S fits, but one short-pilot policy selected a slowly varying, nearly
feedback-free solution and stopped before recovering the active dynamics.
The new experiment asks whether explicitly solving the small coefficient block
helps the initializer avoid these difficulties.

| Route | Initializer | Refinement |
| --- | --- | --- |
| C+S | Existing exact Radau collocation in physical parameters; native convergence required, otherwise ordinary-start fallback. | Existing forward-sensitivity least squares. |
| J+S | Joint optimization of lifted node values and rate/weight coefficients, with the fixed penalized objective below. | The same forward-sensitivity least squares. |
| A+S | Alternate a bounded linear least-squares coefficient solve and a nonlinear node-value solve on exactly the J+S objective. | The same forward-sensitivity least squares. |

J+S versus A+S isolates the block strategy. C+S versus either new route changes
the parameter coordinates, algebraic lifting, scaling and constraint treatment
together; it cannot identify which of those changes caused an improvement.
There is no variable projection or adaptive penalty schedule in this milestone.

## Why node lifting is necessary

The hidden fixture is

```text
v01 = c + k*f^2/(1+f^2) + m + k_p*p + k_u*u01
m' = -m/tau + k_u*u01
p' = -p/tau_p + m
f' = k*v01^2/(1+v01^2) - f/tau_f
m(0) = p(0) = f(0) = 0
```

Fixing only `m`, `p` and `f` does not make the full system affine in its
parameters: `v01` itself depends on the weights and occurs inside nonlinear
feedback. The initializer therefore introduces an independent value `q` for
the declared algebraic process `v01` at each collocation stage. It retains
`q - (c + k*f^2/(1+f^2) + m + k_p*p + k_u*u01)` as an algebraic residual.
Feedback uses the estimated `q`, never a substituted noisy observation.

Declared time constants become positive decay rates. For ordinary time constants
this is `r=1/tau`, so `-m/tau` becomes `-r*m`. The existing restricted evaluator
floors positive denominators below `1e-12`; the exact runtime-preserving map is
therefore `r=1/max(tau,1e-12)`. Values below that floor have the same evaluated
equations for supported simple-divisor occurrences. Conversion back uses the
canonical representative `tau=1/r`, checks the original domains, and does not
permit zero rates or infinite physical values.

The adapter accepts this conversion only when every use of a declared time
constant is a simple divisor. It copies and rewrites the validated restricted
AST, then uses the existing safe translator. It rejects a time constant in
another role and rejects any remaining nonlinear coefficient dependence.
No `eval`, `exec` or unrestricted symbolic lambdification is used.

With node values fixed, the seven shared coefficients are `c`, `k`, `k_p`,
`k_u`, `1/tau`, `1/tau_f` and `1/tau_p`, in the candidate's recorded ordering.
The same `k` and `k_u` remain shared across their occurrences. The offset stays
signed; all other weights retain their original domains. The numerical rate
mapping preserves the denominator floor rather than changing the scientific
sign contract.

## One objective for the two new routes

Both routes use two-stage Radau IIA, order three, at each existing input/sample
interval. The internal stage is at one third of the interval and the endpoint
is the second stage. Inputs are linearly interpolated, consistent with the
existing forcing contract. Initial model states remain fixed at zero.

There are 2,880 free state values and 964 algebraic process values across the
four training trajectories: **3,844 node values plus seven coefficients**.
The initial process value is a decision variable with an algebraic residual;
the initial state is not. No extra scientific state or trajectory-specific
physical parameter is introduced.

For normalized observation residuals `e`, Radau defects `d`, and algebraic
residuals `a`, the fixed initializer objective is

```text
F(nodes, coefficients) = 0.5 * [mean(e^2) + 10000*mean(d^2) + 10000*mean(a^2)].
```

Observation units use the existing training-observation scale. Each state's
unit is `max(1, RMS of its ordinary-start rollout guesses)`. Each process unit
is the larger of the observation scale and the RMS of its model-generated
guesses. These units, the mesh, the ordinary parameter vector and the initial
nodes are frozen before either optimizer starts. They never use generating
latent trajectories. Means count individual scalar residuals within each of
the three groups. All components are logged separately.

Exact dynamic equalities would generally pin the state block for fixed
parameters and make an independently updated coefficient block infeasible.
Finite penalties allow the alternating initializer to move. They also allow
defects and discretization bias, which is why node loss is never an accepted
trajectory score. C+S retains its previous exact equalities as the reference.

## Alternating updates and acceptance

For fixed nodes, the residual vector is `A(nodes)*beta - b(nodes)`. CasADi
checks conditional affinity and constructs this design matrix analytically.
The coefficient step uses SciPy's bounded linear least-squares solver (BVLS),
with column scaling and a dense factorization of the small seven-column
system. It does not form an explicit normal-equation inverse. With sign/domain
bounds and possible rank deficiency, a single unconstrained closed-form
formula is not generally sufficient.

The report records the solver status, scaled projected-gradient KKT residual,
active bounds, column scales, rank and condition estimate. A coefficient
proposal must be finite, within its bounds, pass the KKT check and not increase
the same objective. Rank deficiency is reported; it is not interpreted as
unique physical parameter recovery.

Then IPOPT updates node values with coefficients fixed for at most 15
iterations. Up to 12 coefficient/node cycles run, subject to the common
120-second initializer budget and a small-progress stop. J+S instead updates
nodes and scaled coefficients together with IPOPT for at most 150 iterations.
Both use tolerance `1e-7`, the same objective, strict coefficient bounds, and
the same finite, in-domain, nonincreasing-iterate acceptance rule.

A usable initializer need not have native convergence. A+S and J+S retain
accepted finite iterates even after a native limit; `last_nonlinear_block_success`
reports native success separately. C+S retains its previous strict-convergence
fallback. Any improvement over C+S must be interpreted with that difference
in acceptance policy in mind.

## Fresh rollouts, budgets and checkpoints

Only physical parameters are passed to refinement. All fitted node values are
discarded. The original expanded ODE model rolls out from the original fixed
initial states, and the same forward-sensitivity Jacobian drives the existing
least-squares refinement. This removes lifted algebraic freedom and penalized
defects from the final model evaluation.

Every logical fit has **600 seconds total**, including at most **120 seconds
for its initializer**, followed by the existing 150-evaluation refinement
ceiling. Each route actually runs and pays for its own initializer; there is
no shared initializer job. Reference generation and final independent replay
checks have separate budgets. Setup/reporting is outside `total_fit_seconds`;
native initializer process startup and saved-iterate I/O are inside it.

Every improving J+S/A+S native iterate and every accepted coefficient block is
saved as a numeric array with a digest plus an atomic JSON checkpoint. The
checkpoint identifies the source experiment, candidate, training fingerprint,
ordinary start, domains, settings, penalty and algorithm settings. On resume,
completed fits and initializers are reused; an interrupted initializer resumes
from its latest accepted point and charges previously recorded elapsed time.
A native hard limit can retain that point without granting a fresh initializer
budget. Corrupted arrays or changed identities are rejected.

IPOPT's internal state is not serialized. An interrupted node block restarts
its native solver at the saved point. The existing sensitivity runner reuses
completed fit checkpoints, but an interrupted refinement attempt restarts
from the frozen refinement start. A machine or scheduler kill between
checkpoints can leave unrecorded work; this is deterministic experiment
configuration and checkpoint reuse, not bitwise native-iteration replay or
perfect accounting of externally killed processes.

## Frozen matrix and interpretation

The ordinary matrix remains two systems (moderate/separated), noise fractions
0/0.03, and three paired seeds: 20260909, 20260910, 20260911. It gives 12 pairs
and 36 fits. An additional noiseless separated stress pair runs all three
routes from the actual saved Delta initializer parameter vector previously
shown by the user:

```json
{"c":-0.7454970473406565,"k":0.02109269311390669,"k_p":2.274881443811928,"k_u":0.9030140854701865,"tau":0.045988457871128506,"tau_f":99958.15055240729,"tau_p":6.197502526242358}
```

This reuses parameters only, not any saved latent node values. It is an opened
development stress test motivated by a known failure, not an untouched holdout.
Stress results are separated from the ordinary three-replicate aggregates.

The full submission has **2 numerical guards + 39 fits = 41 array tasks**, plus
a summary job. Each task requests one CPU and 8 GB; concurrency is two and no
GPU is requested. The sum of worker ceilings is about **9.47 CPU-hours**, plus
summary. Actual usage can be much lower with early convergence. ACES is not
needed, so the proposer can continue there independently.

Guards check reference accuracy and sensitivity agreement at generating
parameters, every broad start and the separated stress start. They also check
the lifted/rate equations against the existing expanded evaluator along these
rollouts. Final fits undergo independent production Radau, BDF and tighter
Radau replay on both development splits.

`complete` means numerical replay verification. Output recovery separately
requires clean-reference training **and** validation NMSE <= `1e-4`. Clean
scores, known parameter aliases and validation data never enter fitting,
initializer acceptance or coefficient/node updates. Named latent variables
and unique physical parameters are not recovery gates.

## Results to review

`summary.md` includes every fit, ordinary and stress aggregates, and 13
J+S/A+S matching checks. These compare initial node/scaling identities,
objective values, penalties, training fingerprints and ordinary parameter
vectors. The initializer table shows initial/final objective, component errors,
cycle counts and accepted/rejected coefficient steps. `compact.json` includes
coefficient KKT/rank reports. `details.md` and `summary.json` retain full results.

Accepted node snapshots and progress are under each new route's
`results/<fit-name>/block_initializer/`. They are diagnostic artifacts, not
model predictions. Missing and failed tasks remain in report denominators.

Judge the ordinary pairs first: recovery counts, paired elapsed time, residual
calls, fallbacks and agreement of independent replays. Then inspect the known
stress pair separately. A lower initializer objective alone is insufficient;
A+S can decrease that objective monotonically and still enter a poor rollout
basin or make slower progress than J+S.

## Running on Delta

Use the pinned commit and isolated-checkout command supplied in the handoff.
The existing Python environment and CasADi 3.7.2 target are reused; no new
package installation is required.

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-fitter-methods-v4 && bash scripts/hpc/submit_fitter_methods_v4_delta.sh
```

The launcher submits guards, then fits, then summary. Repeating a completed
submission returns its recorded job IDs. A partial submission stops for queue
reconciliation rather than silently duplicating jobs. Source/runtime drift
and mismatched checkpoints remain rejected.

When the jobs finish, paste:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-methods-v4/summary.md
```

To regenerate a report without submitting jobs:

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-fitter-methods-v4 && PYTHONPATH=src:/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps /projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python scripts/run_fitter_methods.py summarize --output /work/hdd/bibo/yxiao2/phase_b/fitter-methods-v4
```

## Limits of this milestone

The adapter currently targets the synthetic scalar-observation fixture and
supported expressions with fixed numeric initials. An arbitrary proposer
candidate can have nonlinear shape parameters, different observation contracts,
additional algebraic structure or unsupported time-constant use; conditional
linearity must be audited, not inferred from parameter names. The rate bounds
exclude unrepresentable reciprocal values and preserve the existing denominator
floor. No new scientific parameter bounds are inferred.

Alternating optimization is local and can stall along strongly coupled
node/parameter directions. The fixed penalty and mesh are experimental
choices, not validated general defaults. Variable projection, penalty
continuation, arbitrary-candidate support and production promotion require
separate milestones after reviewing this comparison.

## Local verification and development smoke results

The full `pytest` run reported 1,191 passed, three optional Torch tests skipped,
and one existing multiple-shooting test exceeding its 30-second native wall
limit while full-size smokes were running concurrently. That unchanged test
passed on a separate rerun in 13.72 seconds. All 1,192 applicable tests therefore
passed across these validation runs. The new collocation tests passed in the
full suite. Ruff, shell syntax and diff checks also passed.

Tests cover conditional affinity, signed/bounded coefficient solving, rank
deficiency, denominator-floor equivalence, rejection of unsupported nonlinear
coefficients and validation input, native node optimization, accepted-iterate
retention after a consumed budget, corrupted/stale checkpoints, completed-fit
resume, and the 41-task launcher with duplicate-submission protection.

Both full-size numerical guards passed, including every broad start and the
stress vector. Lifted-versus-expanded equation discrepancies were at most
`4.45e-16` under the guard's normalized comparison. Five full-size fits passed
all independent production replay checks:

| Opened development case | Route | Refinement residual calls | Clean validation NMSE |
| --- | --- | ---: | ---: |
| Moderate, noiseless, broad replicate 0 | J+S | 44 | 1.46e-20 |
| Moderate, noiseless, broad replicate 0 | A+S | 29 | 1.47e-20 |
| Separated, noiseless, saved stress vector | C+S | 31 | 0.00206117 |
| Separated, noiseless, saved stress vector | J+S | 19 | 2.01e-21 |
| Separated, noiseless, saved stress vector | A+S | 46 | 0.00206117 |

The two completed J+S/A+S pairs used identical initial node/scaling identities
and objectives. Alternating coefficient steps passed the bounded solve checks
and decreased their objective. Nevertheless, A+S did not escape the known
stress basin in this local run. J+S did. This supports keeping the joint
formulation control and provides no basis to promote alternating updates.
The penalty, cycles, iteration limits and frozen Delta matrix were not tuned
after these outcomes. Local wall times were measured under concurrent regression
tests and are not used to claim a runtime advantage.
