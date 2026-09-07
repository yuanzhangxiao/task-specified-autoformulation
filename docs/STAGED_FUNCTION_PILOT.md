# Conditional function-generation handoff

The topology implementation has a conditional pass for this next development
milestone. The 20B v2 probe compiled five of six public constructions and four
passed configured public graph checks. Independent public-scientific review
selected Dalla seed 1 and anonymous-system seed 1 for a bounded function pilot.
This does not establish production topology reliability or numerical model quality.

`scientific-staged-functions-1` freezes those exact source results. The configuration
pins the original topology-plan digest and both result digests. Freeze verifies
the original terminal identity and result, then embeds the public brief,
scientific inventory/equations and compiled topology into a new source-bound
plan. All generated outputs stay outside Git. Neither test trajectories nor
private reference artifacts are used.
The new plan binds the function CLI, shared serving launcher and Slurm wrappers
as well as the Python package. Tensor parallelism must match the frozen platform.

## Provider contract

For each existing term, the runtime supplies the full public scientific brief,
inventory, equation sketch, selected LHS/mode/source set/assembly sign/role,
accepted functions and shared parameter registry. Only this response is allowed:

```json
{
  "expression": "gain * x * u",
  "parameters": [{"name": "gain", "role": "scale"}]
}
```

The expression uses exactly the selected scientific sources. Parameter names
and roles are model choices; numeric values, ranges, scope and term IDs are
runtime responsibilities and rejected as provider fields. Every used parameter,
including reused parameters, is declared locally; roles must agree across terms.
The inherited compiler rejects real-valued `coefficient` roles because topology
owns outer polarity. It does not prove whole-function positivity or monotonicity.

The restricted parser checks grammar and resource limits before any AST alias
mapping. Generated auxiliary scientific names are renamed structurally to the
compiler aliases. Provider text is never executed as Python. Each tentative
assignment is checked for topology compatibility before replacing the draft;
failed calls leave previous functions unchanged.

After functions, each latent state has a separate initializer request:

```json
{"initial": {"fixed_value": 0.0}}
```

or

```json
{"initial": {"expression": "baseline_input"}}
```

The schema has disjoint modes. Initial expressions can use only supplied
auxiliaries, exogenous inputs, covariates and time. Generated auxiliaries,
targets, states, processes and fitted parameters are excluded. Observed-state
initializers remain compiler-derived. These are current compiler/protocol
limits, not claims about which initializers are scientifically possible.

Every call is cached before local validation; rejected visible responses and
diagnostics enter a bounded retry. Cached accepted calls reconstruct exactly
the same draft after interruption. Allocation drain leaves partial checkpoints
without a terminal failure, and uncertain calls are not silently reissued.
The complete draft passes the ordinary candidate validator and expression
compiler. Completion is distinct from numerical stability, scientific
correctness and fit quality; no parameter fitting occurs in this pilot.

The selected term also includes a runtime-rendered `assembly_template`, such as
`d(x)/dt = ... - (FUNCTION)` or `output = ... + (FUNCTION)`. The reply fills
only the parenthesized inner slot. For a subtractive relaxation term, `x / tau`
becomes `-(x / tau)` after assembly; returning `-x / tau` would instead create
growth. Genuine grouped laws can contain internal subtraction. The runtime
preserves their expressions and applies the topology sign once; it does not
silently remove negations or impose universal nonnegativity on inner functions.
The prompt asks for scientific functional laws, preserving required nonlinear
behavior and introducing unknown time scales or gains when needed. It permits
identity expressions for scientifically known, unit-compatible balance terms.

## Frozen pilot

`configs/staged_function_probe_v1.json` retains the established 20B, low-reasoning,
single-H100 serving settings. Each of the two selected topologies has function
seeds 0, 1 and 2. Two separate toy diagnostics exercise multivariate source sets
and generated auxiliaries. These eight tasks share one warm model server in a
45-minute allocation; each construction retains the three-attempt local repair
limit and existing per-task call/token budgets.
The worker has a separate 25-minute window and a four-minute drain margin;
the allocation also covers bounded server startup and cleanup. The minimum
first-attempt workload is 80 requests. Launcher identity and tensor parallelism
are checked before the server starts and launcher identity is rechecked by the worker.

Report six conditional public handoff outcomes and two diagnostics separately,
including first-pass versus repaired completion, errors, complete candidate
artifacts, all physical calls and measured tokens. The two reviewer-selected
topologies are not a random sample; these outcomes cannot estimate end-to-end
construction success. Keep the original topology failures in their own report.

```sh
PYTHONPATH=src python scripts/staged_function_campaign.py freeze \
  --config configs/staged_function_probe_v1.json \
  --topology-plan /path/to/topology-v2/plan.json \
  --topology-results /path/to/topology-v2/results \
  --output /path/to/function-run/plan.json
```

The existing shared server launcher selects the topology or function worker
from a closed protocol list; earlier source-pinned runs remain reproducible.
Delta CPU fitting/diagnostics follow after authentication is available and the
function results are reviewed. No two-H100 job is needed for this gate.

## Matched function follow-up

The initial pilot completed six of six selected-topology public tasks and two
of two diagnostics at the compiler gate. Scientific review found no public
candidate ready for fitting: subtractive functions repeated their outer signs,
turning sinks and relaxation into growth. The anonymous-system candidates also
omitted the required nonlinear feedback. Dalla candidates had no free
parameters, and anonymous-system candidates had only an offset, leaving unknown
rates and gains fixed to one. These outcomes do not establish a function-stage
scientific pass, and the raw candidates will not receive a fitting budget.

`configs/staged_function_probe_v2.json` defines a separate matched development
rerun. It preserves both source topology digests, all three function seeds,
both diagnostics and the serving/budget settings. The changed treatment is the
explicit assembly slot and scientific-law prompt guidance. The new frozen plan
binds its own code and launcher identities; the v1 configuration and artifacts
remain unchanged. Review assembled signs, the required nonlinear pathway,
parameterization, candidate duplicates and initialization before numerical work.
These scientific reviews remain separate from deterministic compiler completion.
Typed scientific-obligation audits, unit propagation and topology revisions are
deferred until this narrow correction is measured.

## Verification

Tests cover strict minimal schemas, successful multivariate functions,
missing/extra sources, unsafe grammar, unused/colliding/conflicting parameters,
generated auxiliary aliasing, causal initializers, atomic local repair,
deterministic replay and drain/resume, source-result tampering and frozen
campaign identity. Run full pytest, Ruff, the incremental-construction smoke,
the actual freeze CLI, and launcher shell syntax checks before deployment.
