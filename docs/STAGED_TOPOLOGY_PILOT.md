# Scientific staged topology: first implementation milestone

The development protocol `scientific-staged-topology-1` implements the agreed
variable identification -> inventory check -> equation topology flow. The
initial objective is reliable, valid topology construction. It does not run
function generation, fitting, a scientific judge, or any test/private endpoint.

The runtime provides public scientific context and chooses a fixed agenda:
required mechanisms with their affected targets, then completion of every
public target. Level 1 returns variable names, definitions and scientific roles.
Available data roles and candidate definitions are separate. Consistent reuse
keeps the first role description; similar scientific hypotheses are not merged.

Level 2 defines one generated variable per call using grouped sources, an outer
assembly sign and a scientific role. Each call sees the full inventory and
accepted equation sketch. Its source enum and term limit are enforced by the
same Pydantic-derived schema sent to the provider. An alternative inventory
revision response is retained as unresolved; this version does not automatically
backtrack. Local contract repairs receive the rejected visible response and
diagnostics. Routing, qualitative response assertions, shared-transfer rules and
the grouped/joint comparison variants are deferred.

The compiler creates internal identities and target mappings. An algebraically
generated auxiliary uses a compiler-only alias so it cannot be resolved as its
supplied trajectory. Differential auxiliaries also use aliases and are generated
without observed-state resets. Variable and term roles survive compilation.
Auxiliary supervision is not introduced. Purely algebraic models are not supported by the
existing ODE compiler; such a proposal is retained as a compilation failure.

The versioned public brief keeps Sections A-E of the frozen benchmark prompt,
positive mechanism requirements and explicit target-composition dependencies.
The older mass/rate keyword heuristic is not imported as a mandatory choice of
differential versus algebraic definition. Frozen benchmark prompts, data and
legacy evaluation artifacts are unchanged. Necessary graph paths are reported
separately from basic validity and are not proof of scientific correctness.

## Verification and the first GPU probe

Use the repository Python environment with `PYTHONPATH=src`. Tests cover
success, missing targets/equations, unavailable variables, algebraic cycles,
dynamic feedback, grouped sources, generated auxiliaries, inventory revisions,
repair, allocation draining, terminal call caching and deterministic replay.
Run full pytest, Ruff, the existing incremental-construction smoke and the new
campaign CLI smoke before deployment. External test fixtures can be supplied by
uncommitted read-only links in an isolated worktree; do not commit those links.

`configs/staged_topology_probe_v1.json` pins 20B, low reasoning, temperature 0.2,
8192 output tokens per call, three attempts per step, at most 64 physical
requests and a 262144-token conservative budget per construction. Missing usage
is budgeted by request bytes plus the output allowance; actual provider tokens
and unmeasured requests are reported separately from that budget charge. A new call
must fit the remaining conservative token reservation. An interrupted physical
call is recorded as uncertain and is never reissued under the same key; any
further attempt consumes the next deterministic retry key.

The first allocation requests one ACES H100 for 45 minutes. It runs a simple
Level-2 fixture, Dalla T2 named easy and the opaque hard device at seed zero,
then generated-auxiliary and two-memory fixtures. These five logical tasks
share one model server. Fixtures never enter benchmark completion denominators.
There are no functions or numerical data in the requests. A source fingerprint,
frozen task manifest and complete physical request records bind deterministic
resume. The worker drains before its deadline and does not repeat terminal
tasks. Server startup is bounded to ten minutes. GPU startup, provider time and
resource information are stored separately. Cache identity includes the pinned
model revision, serving image digest and platform through the frozen plan.
The launcher verifies the image and passes the frozen model revision to vLLM.
A same-worker Delta A40 launcher
is included, with submission dependent on refreshed NCSA authentication.

Freeze on the deployment source before submitting:

```sh
PYTHONPATH=src python scripts/staged_topology_campaign.py freeze \
  --config configs/staged_topology_probe_v1.json \
  --public-root /path/to/public-prompt-v3 --repository . \
  --output /path/to/run/plan.json
```

The scheduler launchers require explicit repository, Python, output, image,
Hugging Face, compute-cache and short IPC-root environment paths. They do not
submit jobs themselves. The 120B campaign follows a reviewed implementation
probe; a failure in 20B scientific modeling does not by itself establish a
120B limitation.
