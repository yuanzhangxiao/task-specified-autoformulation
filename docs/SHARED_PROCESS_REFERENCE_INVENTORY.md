# Private reference inventory for shared-process development

Diagnostic code review, not proposer material. No trajectory files or test scores
were used for this inventory. Counts refer to the complete reference equations;
they are **not required-model sizes or completeness scores for reduced candidates**.
Keep this document out of construction, critic and numerical-feedback packets.

| Reference family | Concrete reuse in the full equations | Interpretation |
| --- | --- | --- |
| Canonical Dalla Man | Seven directed same-unit transfer contributions listed below | Five groups if the two reversible pairs are each counted as one process |
| Perturbed Dalla Man | The same seven transfer contributions; the glucose exchange laws are changed | Same sharing structure, different functions |
| CSTR | Reaction law used in concentration and temperature equations; temperature difference used for reactor/jacket exchange | Two shared processes; consumer coefficients contain conversions, so equality of derivative magnitudes is not required |
| Alien device | A common nonlinear input response is distributed by the input vector; latent skew coupling has an antisymmetric coefficient structure | Shared response/coefficient structure; do not infer material transfer or conservation from it |

Dalla Man's seven directed witnesses in `src/autoformalism/rebuttal/dalla_man.py`
(`_rhs_and_derived`, the `derivative` array and `dgp`/`dgt` assembly) are:

| Contribution | Depletes | Supplies |
| --- | --- | --- |
| Gastric grinding `kgri * Qsto1` | `Qsto1` | `Qsto2` |
| Gastric emptying `kempt * Qsto2` | `Qsto2` | `Qgut` |
| Plasma-to-tissue glucose exchange | `Gp` | `Gt` |
| Tissue-to-plasma glucose exchange | `Gt` | `Gp` |
| `m2 * Ip` | `Ip` | `Il` |
| `m1 * Il` | `Il` | `Ip` |
| Portal secretion `gamma * Ipo` | `Ipo` | `Il` |

These are an explicit witness inventory, not an exhaustive count of every shared
subexpression. Gut disappearance and plasma appearance also share absorption
dynamics with bioavailability and body-weight conversion; they are not an eighth
literal equal-and-opposite same-unit term. Delay filters such as the `I1`/`Id`
chain must not be labeled physical transfers simply because a term occurs with
opposite signs.

The CSTR witnesses are in `src/autoformalism/benchmarks/phase_b_generation.py`,
`_simulate_cstr`: one `reaction` value enters `C'` negatively and `T'` through
`source_gain`; `(T-Tj)` enters `T'` and `Tj'` with distinct exchange coefficients.
The distinct coefficients encode the quantities represented by the states and
must not be forced equal by the proposed compiler.

The alien reference in the same module, `_simulate_alien`, shares
`tanh(input_scale * forcing)` across its input-vector entries. Its `skew @ latent`
structure is a different relationship: opposite matrix entries multiply different
states. This is not the same unsigned flux subtracted from one state and added to
another. The first milestone does not automatically discover or enforce that
matrix constraint.

The six deadline cells include four Dalla Man cells (two underlying dynamics,
each named and obfuscated), one CSTR cell, and one alien cell. Thus five cells,
representing two of the three families, have clear physical-transfer or converted
reaction/exchange witnesses in their full references. All three families have
some reusable algebraic structure. These are different counts and claims.

The public tasks are reduced, partially observed tasks. For example, easy CSTR
supplies `C` and `Tj` as auxiliaries while targeting `T`. A valid candidate may
therefore contain only the `T` equation; it need not duplicate the full reference's
two-sided balance. Likewise, a Dalla Man glucose task does not require recovery
of the full insulin subsystem. Count sharing opportunities only among equations
that the candidate actually needs to generate. Do not add hidden reference
requirements to obtain a higher reuse count.

Next step: obtain the saved-model inventory for both `full` and `brief_only`,
then manually classify its witnesses as already shared, safe factoring,
scientifically justified tying, or independent processes. Only the last review
can support statements about missed sharing in discovered models. The automated
syntax counts alone cannot establish those statistics.
