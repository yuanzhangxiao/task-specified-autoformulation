# Staged function-generation granularity experiment

This development-only experiment compares two provider task granularities while
holding the reviewed topology, public scientific brief, model settings, seed,
and latent-initialization stage fixed:

- `atomic_interaction`: one function reply for each immutable topology term;
- `equation_batch`: one ordered, atomic reply for all terms on one left-hand
  side.

The batch response is deliberately small: it contains only the ordered scalar
functions and their parameter identities and qualitative roles. The runtime
owns the left-hand side, term order, grouped source sets, outer signs, topology,
and binding. If one function in a batch fails deterministic validation, none of
that batch is accepted and the complete batch is retried with the diagnostic.

Latent initial values remain separate per-state calls in both arms. This keeps
the comparison about function construction rather than initialization. No
parameter fitting, scientific judge, test data, or private reference is used.

The launch order is counterbalanced by seed so neither arm is always first on
the warm server. The summary keeps every planned Dalla Man and opaque-system
trial in its denominator. It reports function and latent-initial acceptance
separately, completion, provider resources, candidate equations, and auditable
syntax facts. Scientific conclusions about unit
consistency, justified parameter sharing, direction inside grouped laws, and
mechanistic adequacy still require human review. No scalar winner is declared
automatically.

Scientific-feedback routing is intentionally deferred to a separate matched
experiment after selecting a generation granularity. Mixing the two changes in
one campaign would make a gain impossible to attribute.
