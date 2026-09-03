# Public mechanism-compliance signals

Mechanism evaluation exposes two separate deterministic endpoints. They are not
combined into a weighted score.

## Graph mechanism compliance

`graph_mechanism_compliance` is the primary scientific signal. For each
publicly required mechanism, the evaluator searches the candidate's directed
dependency graph for candidate-owned state or process components that:

- receive every required public driver;
- contribute through a directed path to every required target; and
- provide latent dynamic memory when the public requirement calls for memory.

The check does not depend on proposer-supplied mechanism names or tags. It
allows alternative latent-state names and alternative restricted expressions.
Sign requirements that cannot yet be certified from graph structure remain
ambiguous and make the separate completeness flag false.

## Mechanism annotation compliance

`mechanism_annotation_compliance` measures whether proposer-supplied mechanism
tags identify components that satisfy the same graph predicates. This is a
metadata-quality signal: it helps route feedback, explain models, and attach
public mechanism identities to an otherwise valid graph. It is not evidence
that an unconnected or scientifically invalid structure is correct.

When graph compliance succeeds but annotations do not, the evaluator may emit
an `annotation_repair`. A repair is marked unambiguous only when graph evidence
identifies one preferred component. The repair carries provenance and does not
change equations, parameters, or target mappings. Ambiguous cases remain for
proposer clarification.

## Compatibility fields

The historical `mechanism_compliance` and
`mechanism_compliance_complete` fields remain available. They are exact aliases
of graph mechanism compliance and its completeness flag; they are never an
average or blend of graph and annotation scores. New reports print graph and
annotation endpoints explicitly.
