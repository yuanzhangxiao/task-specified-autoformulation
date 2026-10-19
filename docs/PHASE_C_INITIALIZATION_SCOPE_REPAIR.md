# Initializer record scope and unresolved-conflict feedback

The first five-seed repair diagnostic completed ten episodes on two fixed saved
T2-easy models: four accepted initializer graphs with deferred evidence, five
proposer requests for topology changes, and one exhausted repair allowance.
All twenty requests returned normally, consuming 100,848 observed tokens.
All four accepted choices selected an independent trajectory root for an existing
latent state. No fitted recovery was demonstrated. The five topology requests
are proposer judgments, not proofs that initialization cannot resolve the issue.

The unsuccessful replies repeatedly removed an invalid initializer for an
algebraic output, then retained the original valid-but-conflicting latent map.
Other replies attempted to introduce new clearance coefficients through initializer
records. The deterministic gate correctly withheld acceptance in both situations.

## Opt-in presentation change

`scope-aware-2` leaves scientific checks, schemas, allowed initializers and budgets
unchanged. The default `legacy-1` wording remains available for historical replay.
The new system message and repair payload:

- State explicitly that a model cannot be accepted with an active demonstrated
  model-data conflict. Removing a separate invalid record does not resolve it.
- Distinguish algebraic outputs, processes, undeclared targets, fixed observed
  initials and fixed known initials from editable latent differential initials.
  An initializer record cannot introduce a dynamics parameter or ODE term.
- Show the existing editable initials relevant to each matched-subsystem
  conflict, tracing the current topology and transitive initializer dependencies.
  This is a scope explanation, not an automatically selected scientific fix.
- Retain unaffected records, while permitting revision of a schema-valid rule
  that causes the conflict. Invalid replies retain the last verified model and
  conflict; their raw declarations do not become the current graph.
- Distinguish initializer targets and `state_initials` from recorded measurements
  in `public_initials`. A listed initial measurement of an algebraic output remains
  an allowed map input, without giving that output its own initializer.

The messages are generated from declarations, runtime information permissions and
current evidence. There are no benchmark-name or coefficient-name heuristics.
Unknown or deferred evidence remains unresolved. Independent trajectory roots
require the existing observation-prefix authorization; this milestone does not
choose a prefix duration, estimate a root, construct a formula or establish
scientific correctness of the explanation.

## Bounded comparison

The new sealed campaign protocol is `saved-construction-initialization-repair-2`.
It uses the **same original two saved models and initializer handoffs**, proposal
seeds 0–4, settings and budgets as the first repair experiment. It does not start
from the accepted or failed final choices of that experiment. Results and request
namespaces are separate; old checkpoints and costs remain intact.

Each of ten episodes has at most three physical requests, 393,216 charged tokens,
and 8,192 output tokens per call. Report every seed. The experiment tests whether
more specific feedback improves permitted initialization choices, not whether the
models predict accurately. There is no automatic follow-up, topology edit, fitting,
test access or promotion.

The prepare command accepts `--feedback-policy scope-aware-2`; execution and
inspection read the policy from the sealed plan. Unknown policies or an attempt
to reuse an incompatible output root are rejected. Resume validates the exact
cached requests and retains previously charged work.

## ACES commands

From the supplied source bundle in group scratch:

```bash
bash scripts/hpc/start_initialization_scope_repair_aces.sh run
bash scripts/hpc/start_initialization_scope_repair_aces.sh status
bash scripts/hpc/start_initialization_scope_repair_aces.sh inspect
```

The default output is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-initialization-scope-repair-seeds5-v1`.
The existing one-H100, one-hour launcher and node-local compiler cache are reused.
`inspect` writes `inspection.tar.gz` offline. Use the bundle's `RUN.md` for the
complete upload, integrity-check and launch commands.
