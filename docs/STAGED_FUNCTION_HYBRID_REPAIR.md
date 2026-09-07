# Staged function batching with interaction-local repair

The function-granularity experiment showed a real trade-off. Same-LHS equation
batches reduced requests and tokens, but the opaque-system batches omitted the
required nonlinear dependence in all three seeds. Fully atomic construction
preserved that dependence more often but approximately doubled transport cost.
In the completed development run, batches produced 6/6 complete models with 42
requests and 120,054 observed tokens, versus 5/6, 86, and 245,656 for atomic
generation. On the opaque topology, however, required nonlinear syntax appeared
in 0/3 complete batch models and 2/2 complete atomic models.

The hybrid policy keeps one coherent batch per left-hand side and adds three
runtime controls:

1. Before a provider call, the runtime derives a narrow, typed functional
   obligation for every immutable interaction. An interaction whose reviewed
   scientific role explicitly says `nonlinear`, `saturation`, `sigmoid`, or
   `threshold` must contain syntax-level nonlinear dependence on a displayed
   source. The check recognizes an admitted nonlinear function, a non-unit
   power, a source-dependent denominator, or a product of sources. This is a
   deterministic minimum contract, not a claim of scientific correctness.
2. After a schema-valid equation batch, each ordered slot is compiled and
   audited independently. Valid slots are retained. Only a failing slot is
   resubmitted as an atomic interaction with the rejected expression and exact
   deterministic diagnostic. A schema-invalid outer batch still receives a
   bounded batch-level retry because no ordered slot can be trusted.
3. Provider parameter names are local mnemonics. The runtime deterministically
   namespaces every fitted identity by interaction before compiling the full
   model. Repeated spellings cannot accidentally couple different equations.
   Intentional shared physical parameters require a future explicit typed
   sharing action.

Latent initial-value calls remain separate and use the same policy as the
earlier granularity experiment. The topology, source groups, left-hand sides,
outer signs, and public task contract remain immutable.

## Frozen targeted pilot

The first campaign uses only the reviewed opaque-system topology. It runs three
seeds with GPT-OSS-20B at low reasoning and 8,192 maximum output tokens on one
ACES H100. The source function plan, topology digest, serving image, model
revision, inference settings, context limit, modeling limits, and seeds match
the earlier atomic-versus-batch experiment. No parameter fitting, judge, test
data, or private reference is used.

The summary retains all planned seeds and separately reports model completion,
batch-term acceptance, atomic-repair activation and success, required
nonlinearity syntax, accidental cross-LHS parameter sharing, physical requests,
tokens, latency, and the final equations. It does not define an automatic
winner. The scientific question is whether the hybrid recovers the opaque
nonlinearity of the atomic arm while retaining most of the equation-batch
resource reduction.

## Remaining limitations

- Obligation derivation is deliberately conservative and lexical. It only
  enforces explicit reviewed role language; it does not infer new scientific
  requirements.
- Syntax-level nonlinear dependence can still be scientifically wrong.
- Terms with invalid nested response schemas cannot be individually recovered
  from the outer batch and therefore require a batch retry.
- Interaction-local parameters forbid intentional sharing for now. A later
  schema must represent a shared physical identity explicitly rather than infer
  it from a repeated name.
- This pilot evaluates function construction only. Fitting and feedback-routed
  model refinement remain separate prospective milestones.
