# Staged function pre-fit handoff

This development campaign consumes the exact six topology artifacts that passed
`scientific-staged-prefunction-hybrid-2`. It does not regenerate variables,
equations, topology, or interaction signs.

For each frozen topology, the function stage:

1. requests one ordered function batch per left-hand side;
2. validates each term independently against its exact source set, sign slot,
   restricted grammar, and functional obligation;
3. retains valid batch terms and requests atomic repair only for invalid terms;
4. requests latent initializers separately; and
5. compiles the complete candidate and emits a deterministic pre-fit
   certificate.

Positive and negative topology terms use runtime-owned outer assembly signs.
Unrestricted terms use a signed inner function and remain proposer-owned. The
runtime's certified outer-gain role repair applies only to fixed-sign terms and
only when the restricted AST proves one direct whole-term scalar gain.

The deterministic certificate checks exact target mapping, interaction-function
coverage, preservation of frozen sources and signs, latent-initializer coverage,
and syntax-level nonlinear obligations. It does not claim scientific adequacy,
dimensional validity, parsimony, or empirical fit.

The plan embeds and hashes every source result, records the source plan and
artifact-ledger hashes, supports per-task checkpoint/resume, and exposes all
failed attempts. Parameter fitting, the scientific judge, test data, private
references, and automatic winner selection are excluded.
