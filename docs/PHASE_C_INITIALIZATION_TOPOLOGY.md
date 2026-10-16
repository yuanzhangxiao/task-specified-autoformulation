# Initial conditions at the topology boundary

The opt-in `saved-topology-initialization-1` milestone separates **what initial
information a state uses** from **the analytic formula applied to that
information**. Public benchmark text and historical constructor policies are
unchanged. This is a live qualification on saved topologies, not a new fitted
campaign or a claim of scientific recovery.

## Scientific distinction

For a latent differential state,

\[
\dot z=f(z,y,u;\theta),\qquad
z^{(n)}(t_0)=g(y^{(n)}(t_0),u^{(n)}(t_0),c^{(n)};\eta).
\]

The displayed arguments are examples, not mandatory dependencies. An ODE
describes change; an initializer describes the preparation/starting information.
For example, `z'=u` can start at `z(0)=0`. If equilibrium is scientifically
justified, `z'=-a*z+b*u` can start at `b*u(0)/a`; equilibrium is not presumed.
An initializer can also infer hidden state from an initial measurement absent
from the RHS. Initializer dependencies are therefore chosen independently.

1. **Topology boundary:** the proposer chooses a shared unknown initial value,
   or a shared map of explicitly listed permitted initial measurements and
   covariates. Missing/unknown choices remain unfinished. Algebraic variables
   have no independent initial value. Observed initial rules remain fixed.
2. **Interaction construction:** the existing causal-initialization constructor
   chooses analytic formulas/parameters after RHS functions. The new consumer
   checks their mode and exact declared public-symbol set. It never copies RHS
   dependencies or silently widens the chosen initial-information set.
3. **Fitting:** existing shared initializer parameters are fitted using training
   data. No per-trajectory validation initial fitting is introduced.

The contract permits only recorded t0 values, fixed covariates and initial time.
It excludes future observations, trajectory IDs and unlisted prehistory. This
is an explicit runtime modeling scope, not a theorem that all scientific tasks
can be initialized from these values. Broader histories require another contract.

## Bookkeeping and matched trajectories

`initialization=null` retains records; a non-null handoff replaces the complete
record. Joint topology/initializer edits are validated before either commits.
Removing a state or making it algebraic requires removing its record in the same
reply; a newly added latent differential state requires a new choice. Runtime
state aliases translate state names only, never public t0 channel names.
Known runtime shared-value/map rules are copied unchanged into the effective
interaction handoff with their evidence provenance; proposer choices remain
separately recorded. Export is withheld when a diagnostic rule has no supported
analytic consumer, rather than inventing a substitute initial policy.

Every diagnostic recomputation retains historical matched-trajectory witnesses.
Checks compare forcing through the mismatch time, so a later forcing difference
cannot erase an earlier contradiction. While a choice is unresolved, the family
check conditions on all permitted t0 information; this does not invent a retained
initializer. Once a choice is declared, its own information set is used.
The prompt labels such hypothetical maximal-family conditioning explicitly; it
does not present those temporary arguments as an initializer already selected.

A changed initializer can distinguish previously matched trajectories without
changing the graph. A previous restriction is resolved only when every relevant
saved witness has a matching, current `conditioning_distinction` receipt. Unknown
or missing evidence remains unresolved. Distinguishing initial conditions is a
necessary-condition result, not proof of a good fit or a correct initializer.

## Live qualification

Use all 15 final drafts from the v2 saved-repair study. These are the latest actual
predecessors, with their original witnesses, not hypothetical replay chains or
equal-budget controls. No score-based selection occurs. Each episode allows at
most three dedicated dependency calls **inside** the existing allowance of ten
total calls and 262,144 charged tokens; subsequent repairs may jointly edit the
topology and initializer records. Cached requests, exact predecessors, schemas,
seeds, contracts and receipts are verified on deterministic resume.

The result separates structural eligibility, initializer completeness, conditional
scientific checks and availability of an `interaction_source`. That source can
be passed to `run_staged_functions`; its two metadata fields automatically select
the causal initializer consumer. Structural/dependency readiness alone does not
authorize promotion. This pilot does not execute interaction construction or fit.

```bash
export PYTHONPATH=.:src
.venv/bin/python scripts/saved_topology_repair.py initialize \
  --source /path/to/saved-actionable-repair-v2 \
  --root /path/to/new-initialization-inputs
.venv/bin/python scripts/saved_topology_repair.py preview \
  --root /path/to/new-initialization-inputs --output /path/to/prompt-review
```

The delivered Jetstream bundle includes the frozen public inputs and compatibility
requests. Run `scripts/jetstream/run_saved_topology_repair.sh` with
`AF_JS2_CAMPAIGN_ROOT` set to a fresh campaign directory; `run`, `status`, and
`inspect` retain their existing meanings. No benchmark files are modified and
no test data is opened.

## Gradual interaction complexity

The separate opt-in helper documented in `PHASE_C_JOINT_DEPENDENCY_ADVICE.md`
can ask whether a joint function should be investigated when verified training
discrepancy and scoped numerical evidence remain under an additive unary model.
It does not impose an argument-count quota, infer a missing edge from NMSE, or
modify a model. It is ready for a later fitting-to-construction experiment;
the topology-only pilot has no fitted evidence and does not invoke it.

Remaining limits: dependency declarations do not establish scientific validity;
exact parsed symbol use does not prove effective dependence after fitting;
the new handoff is opt-in at the saved topology boundary, not a changed default
for every historical fresh-construction policy. Live initializer quality and
subsequent numerical performance remain to be measured.
