# Public mechanism-compliance signals

The legacy evaluator exposes deterministic **structural** endpoints. They are not
full scientific mechanism compliance and are not combined into a weighted score.
The saved-model audit in [PUBLIC_MECHANISM_AUDIT.md](PUBLIC_MECHANISM_AUDIT.md)
reports these separately from evidence-bearing public equation reviews and
behavioral evidence availability. Its new reports omit the misleading alias
`mechanism_compliance`; historical artifacts and runtime gates remain unchanged.

## Graph mechanism compliance

`graph_mechanism_compliance` is a structural requirement signal. For each
publicly required mechanism, the evaluator searches the candidate's directed
dependency graph for candidate-owned state or process components that:

- receive every required public driver;
- contribute through a directed path to every required target; and
- provide latent dynamic memory when the public requirement calls for memory.

The check does not depend on proposer-supplied mechanism names or tags. It
allows alternative latent-state names and alternative restricted expressions.
Sign requirements that cannot yet be certified from graph structure remain
ambiguous and make the separate completeness flag false.

The graph evaluator does not receive fitted values or trajectories. It cannot
establish nonzero pathway activity, correct timescales, physically appropriate
nonlinear functions, or scientific correctness. A 100% graph score means only
that all predicates actually encoded in the specification passed. Annotation
aliases such as `nonlinear_feedback` do not create additional predicates. High
scores in graph-filtered model populations are also a consequence of selection.

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
