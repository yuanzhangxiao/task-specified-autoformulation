# Review of the twelve construction-only models

The construction and local-preservation gate passed. The saved equations do not
justify declaring the scientific pre-fitting work complete. Several problems are
visible in the assembled models without running a fitter: a growing mode labeled
as delayed insulin action, repeated disposal contributions, and an explicitly
required nonlinear mechanism missed by zero local nonlinearity obligations.

This is a manual review of candidate structures and their saved audit facts, not
a scientific-judge experiment, a new production policy, or a fitting result.

## Evidence and limits

- User-provided export: `/Users/yuanzhangxiao/Downloads/model-review.json`.
- Export SHA-256:
  `84fc26496888fe574760aa56dc88466aaca820f6cf3e587891282386a1f4a443`.
- Subsequently supplied frozen context:
  `/Users/yuanzhangxiao/Downloads/review-context.json`, SHA-256
  `cad00ca42bccd36d222fe1e8f8f92f6eb8f0198c746aa3a35a935541d4b12ca4`.
- Campaign: `prefit-construction-audit-1`, ACES root
  `/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1`.
- Reported campaign commit: `c7d0b0523b03eb54f688423b288393280399e77f`.
- Reported plan hash:
  `c722b0a2de2736150b2f82db93973126a63a480b4c8b82f15dbd76e33b334ab2`.
- Twelve models: two public benchmark cells, three seeds, two evidence arms.
- All twelve canonical candidates were read and schema-validated. Their state
  equations, algebraic processes, initializers and changed slots were inspected.
- The second export supplies both frozen briefs, validation contexts, target
  contracts, mechanism specifications and public asset digests. Its reported
  plan hash matches the campaign. Both cells' prompt digests agree across their
  asset ledger, target contract and mechanism specification. The target contracts
  equal the registered v2 contracts; the mechanism specifications equal the
  registered v1 specifications. All four schema types validate for both cells.
- These projections still exclude the complete original topology, unchanged local
  replies and provider cache. They cannot independently reproduce the full ACES
  reconstruction/provenance audit or attribute each defect to its first proposal
  stage. The complete plan hash cannot be recomputed from a partial projection.
  The saved certificates report that reconstruction passed.
- No trajectory data, validation/test values, private reference equations,
  scientific judge or fitted parameter values were used in this review.

The current local proposer prompts are not a substitute for the frozen campaign
brief. Their hashes differ from the frozen public contracts. The anonymous-system
nonlinearity question is now resolved by the actual frozen text, not by the
current local prompt, a keyword flag alone or an inferred hidden mechanism.

## Resolution using the frozen requirements

The frozen anonymous-system task explicitly requires a causal internal pathway
connecting input memory, persistent coupling, nonlinear feedback and output
generation. Requirement `input_memory_output_pathway` repeats this obligation.
Evidence seed 1 has only linear ODEs and affine initialization, with no algebraic
nonlinear process. It therefore lacks a stated required feature. Fitting its
constant coefficients cannot add nonlinear feedback. This is a high-confidence
whole-model requirement mismatch, beyond the earlier keyword-based advisory flag.
It does not establish that the other eleven models pass scientific review.

The Dalla Man context confirms both the delayed insulin-action requirement and
the meaning of `U` as total disposal including the supplied `Uii` contribution.
That strengthens the interpretation of the growing delayed-action mode and
repeated disposal terms as scientific concerns. The exact expression facts are
separate from deciding how to repair the model; they do not establish global
numerical infeasibility or automatically change the runtime's rejection policy.

Two qualifications matter:

- The frozen target-contract metadata labels `U` as `instantaneous_process`,
  while four candidates make it a state. However, the scientific constructor
  deliberately does not import the older mass/rate representation inference as
  a mandatory rule. Its brief retains target dependencies, not the expected
  representation field. The prose identifies a rate but does not explicitly
  prohibit dynamics for that rate. Report this difference between contract
  metadata and the construction policy; do not silently turn it into four new
  deterministic failures. Review the actual disposal equations and their units.
- The frozen `meal_pathway` has `requires_dynamic_memory=false`; only the
  insulin-action pathway requires memory. A direct meal contribution is allowed.
  The absolute-time exponential in brief seed 2 still needs a scientific
  explanation, but absence of a meal-storage state is not a missing mandatory
  feature. This narrows the earlier meal-memory concern.

The frozen prose also explicitly calls `Gp` nonnegative. Empty candidate
constraint lists do not demonstrate that this property is enforced; they also
do not prove that a particular simulated trajectory becomes negative. Preserve
this distinction between a required property, a missing guarantee and an
observed violation.

## What the aggregate results establish

| Measure | Brief only | Training evidence |
| --- | ---: | ---: |
| Constructed and deterministic audit passed | 6/6 | 6/6 |
| Function slots | 78 | 48 |
| Valid slots preserved after certified normalization | 68 | 45 |
| Slots sent for atomic repair | 10 | 3 |
| Atomic-repair incidence per function slot | 12.8% | 6.3% |
| Physical requests | 88 | 60 |
| Reported tokens | 252,254 | 525,843 |

The evidence arm uses 31.8% fewer requests but 2.08 times the tokens. It also
constructs fewer slots, particularly for the anonymous system. Smaller structures
can be useful or can omit mechanisms; these counts do not decide which happened.
There are six matched seed/cell pairs, not twelve independent scientific tasks.
This is an evidence-arm comparison with the same controller, not a matched
before/after comparison against the historical 11/12 constructions.

The thirteen repaired slots consist of twelve source mismatches and one local
nonlinearity violation. The latter changes `k_f*f` into `k_f*sigmoid(f)` in
anonymous-system brief-only seed 2. Final accepted replies fix those local
violations. This does not establish that each assembled equation is scientifically
appropriate, nor does thirteen slots mean thirteen individual retry calls.

The nineteen slots counted as parameter-role changes split into:

- Eleven slots with a same-name `coefficient` to `nonnegative_coefficient` change.
  Every one has the recorded certified outer-gain normalization. This narrows the
  gain domain in accordance with the selected sign contract; it is not unexplained
  proposer role drift.
- Eight slots whose parameter-name-to-role dictionary changes without a role
  change for a surviving name. These involve parameter replacement, renaming,
  addition or removal during repair. Their exact functional changes remain
  reviewable, but the aggregate counter should not call them all role drift.

## Initialization and identity mappings

All twenty latent-state boundaries use `mode=map`, with globally shared
coefficients to be fitted on training. Anonymous-system maps use initial `u01`
and/or `v01`; Dalla Man maps use initial `I`, sometimes with `Uii` or
`meal_event_g`. No map uses a trajectory identifier, future target samples or a
per-time latent fit.
Different available initial values can therefore produce different latent
initials while using the same learned coefficients at validation time.

Repeated local formulas such as `a + b*u01 + c*v01` do not force different latent
states to share coefficients: compilation namespaces them by state. The stored
`0.1` values are optimizer guesses, not learned initial conditions. Map adequacy,
coefficient identifiability, positivity and held-out performance remain untested.
Where available initial features coincide, the deterministic map still gives the
same latent initial state; there is no new preparation-history information.

Identity observation mappings are legitimate aliases here. For example,
`observation U = U` can expose the computed process `U = Uii + A`, and
`observation I = I` can expose a state generated by its ODE. These are not copies
of future measured targets. Likewise, the bare term `Uii` is sensible in an
algebraic sum of disposal rates. Its placement in `dU/dt` has a different meaning
and must be reviewed at the equation level.

For the two algebraic-U models, initial `U` is not automatically matched by an
observed-state boundary. Their latent maps omit initial `U`; hence the identity
observation mapping alone does not guarantee agreement with measured `U(0)`.
This is a boundary-consistency question, not evidence of validation leakage.

## Concrete Dalla Man findings

In the following equations, `q` denotes `meal_event_g`; parameter suffixes are
shortened only for readability. Separate saved parameter names remain separate
parameters. No hidden reference model is used.

### Evidence seed 2: a growing mode, repeated sinks and reversed disposal

The candidate contains

```text
dI_delay/dt = k*I + I_delay/tau,       tau > 0
dU/dt       = -Uii - I_delay
dGp/dt      = k_meal*q - U - Uii - Uii - Uii
```

The `I` equation evolves independently of `I_delay`. Two trajectories with the
same inputs and `I` but different `I_delay(0)` have
`delta_I_delay(t) = delta_I_delay(0)*exp(t/tau)`. This positive mode exists for
every finite allowed positive `tau`; changing the fitted value of `k` cannot
turn it into relaxation. A large time constant can hide growth over a short
window, but cannot change its sign. This is not proof of finite-time blow-up or
of failure to obtain a low finite-horizon fitting error.

The three identical `-Uii` contributions are an exact expression fact. Whether
separate physiological fluxes were intended requires the frozen term roles, but
as compiled they are three copies of the same supplied channel with fixed unit
coefficients. `dU/dt` also decreases with both the insulin-independent rate and
the state described as delayed insulin action. These are strong scientific
concerns about the assembled model, not grammar errors.

### Brief seed 1: the disposal definition and balance disagree

```text
U      = Uii + A
dGp/dt = q - U - Uii - U
        = q - 3*Uii - 2*A
```

The algebraic target definition is meaningful, yet the glucose balance subtracts
total disposal twice and the insulin-independent component once more. Local
source coverage and successful function repair do not check this model-level
accounting. The candidate's glucose-state description also mentions production
and excretion, but those supplied channels do not appear in the equation.
Omitting an optional auxiliary is not itself an error; the explanation and the
actual equation need to agree.

### Four models integrate the disposal-rate target

Brief seed 0 and all three evidence seeds make `U` a dynamic state. Brief seeds
1 and 2 instead use algebraic sums `Uii + A` and `Uii + Uid`.

A rate-valued quantity can legitimately be modeled dynamically; the variable's
name alone must not force an algebraic representation. The specific equations
are the concern:

- Brief seed 0 has `dU/dt = Uii + I_act/tau` and no U-dependent return term.
  Sustained positive contributions accumulate in `U` rather than directly
  defining total disposal.
- Evidence seed 0 has `dX/dt = I/tau` and
  `dU/dt = Uii/tau_Uii + X/tau_X`. Both equations integrate their drivers;
  `X` has no self-relaxation. Under sustained positive insulin this can yield
  growing action and disposal instead of a bounded delayed response.
- Evidence seed 1 has a conventional grouped relaxation law
  `dI_mem/dt = (I-I_mem)/tau`, but `dU/dt = -Uii + I_mem/tau_U` still needs a
  justified rate-dynamics interpretation and scaling.
- Evidence seed 2 has the problems described above.

All canonical state, process and parameter units are `unspecified`, and every
candidate has an empty constraints list. Thus the export does not establish
dimensional consistency or nonnegativity. These are missing guarantees, not
proof that every trajectory violates a constraint.

### Brief seed 2: meal timing and overlapping contributions

The meal process is `k*q(t)*exp(-t/tau)`, and the glucose RHS adds another term
of that form with independent parameters. At a fixed finite time, each is zero
whenever the current meal pulse is zero: neither term stores past meals.
Shifting a pulse by `Delta` changes the exponential factor by
`exp(-Delta/tau)` at corresponding event-relative times. Unless an absolute-time
effect is explicitly intended, this does not represent a repeatable absorption
response tied to each meal. A direct instantaneous meal effect could be a
deliberate approximation, but an exponential multiplier alone is not memory.

The glucose balance also subtracts `U`, `Uii`, `U_mem` and `I`, while
`U = Uii + Uid`. This mixes an already assembled disposal rate with its
components and other modeled quantities. The two meal terms have different
parameters, so they should be reported as overlapping pathways, not falsely
called exact duplicate expressions.

## Anonymous-system findings

Brief-only seeds 0, 1 and 2 all introduce three latent states (`m`, `p`, `f`),
with 13, 16 and 16 function slots and 22, 25 and 25 total parameters. Evidence
seeds use one, two and one latent states, with 6, 8 and 8 slots and 9, 13 and 11
parameters. No particular latent dimension can be certified from this comparison.

Evidence seed 1 is entirely linear in the states and input; its initial maps are
affine. It has a state described as mediating nonlinear coupling, but no nonlinear
term. Its saved facts explicitly report:

```text
required_nonlinearity_mentioned_publicly = true
required_nonlinearity_has_syntax_evidence = false
nonlinear_obligation_count = 0
nonlinear_obligation_pass_count = 0
nonlinear_obligations_satisfied = true
```

The code explains this result: local obligations are derived from each selected
term's scientific-role text, and the certificate checks that all such obligations
were met. Zero satisfied out of zero required passes. The separate public-text
flag is a keyword-based review fact, not an adjudicated requirement. It cannot
by itself authorize a new deterministic rejection. With the second export, the
explicit frozen nonlinear-feedback requirement supplies the missing evidence:
this entirely linear candidate does not meet it.

This is a gap between global task/variable claims and local function obligations.
The frozen public requirement is now attached to the review. A targeted repair
must add an appropriate nonlinear mechanism through a term or topology revision;
merely deleting a proposer claim cannot waive the public requirement. Adding an
arbitrary square merely to satisfy a keyword would not establish the right
feedback mechanism.

The other five anonymous-system models contain powers, a sigmoid or bilinear
coupling. Their nonlinear syntax does not certify feedback direction, boundedness
or adequacy under new inputs. The report's negative-outer-sign relaxation flag
also cannot diagnose stability: a valid term like `-I/tau` can sit inside an
unrestricted outer slot. Effective expressions and parameter domains matter.

## Next milestone

Proceed to a bounded whole-model diagnosis and targeted-repair evaluation using
these saved candidates, before claiming pre-fitting scientific readiness.

1. Bind the supplied frozen public brief and requirement IDs to each diagnostic;
   retain original topology/term roles in the repair artifact. Keep typed ODEs
   and algebraic definitions visible. Public requirements must not disappear
   when a proposer omits a keyword from a local term's role.
2. Report exact structural facts separately from interpretations: repeated
   contributions, algebraically expanded balances, effective self-signs when
   provable, initializer/mapping consistency, and global requirement coverage.
   Parameter-dependent questions stay unresolved; semantic concerns remain
   advisory unless an explicit deterministic contract applies.
3. Route feedback to the component that owns the issue. A rate/state change or
   removal of a duplicate pathway needs topology/equation repair; a permitted
   signed-function correction can stay local. Preserve unrelated valid slots,
   and revalidate the dependency-affected model as one transaction. Local validity
   does not permanently freeze a slot against a separate, supported scientific
   revision.
4. Evaluate correction of the named concern, new violations, unrelated changes,
   calls/tokens and exact resume. Include valid grouped relaxation and legitimate
   identity terms as controls. Do not score success merely by another compiler
   pass, or treat these twelve models as twelve independent scientific tasks.

The existing campaign remains frozen. Its local repair and initialization
results stand; the review supplies new examples for the next diagnostic layer.
No fitter default, benchmark prompt, scientific-judge policy or model artifact
is changed by this document. Astra's optimizer diagnostic remains separate.

The frozen review context has now been supplied; no further context upload is
needed for these conclusions. For reproducibility, its export command is one
physical line:

```bash
jq '{protocol, artifact_sha256, cells: (.cells | with_entries(.value |= {assets, brief, context, target_contract, mechanism_spec}))}' /scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1/plan.json > /scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1/review-context.json
```

For an exact stage-origin investigation, retain the corresponding
`results/TASK_ID/construction.json` as well; the changed-slot-only export omits
the unchanged slots' original term-role declarations.

## Verification

All twelve exported candidates pass the canonical schema. A read-only inspection
of their parsed expressions and parameter dictionaries confirms the counts and
examples above. Verification on the checkout based on `965d1d2`:

- Full pytest: 1,654 passed, three optional Torch skips, 632.32 seconds.
- `ruff check .`: passed.
- `smoke_prefit_construction_audit.py`: passed, including exact resume and
  construction/audit for both synthetic arms, with zero live LLM calls or fitting.
- Documentation whitespace checks: passed.

For the documentation-only frozen-context follow-up, the audit, staged-function
runner and topology-contract regression groups passed: 69 tests in 8.54 seconds.
Ruff, documentation whitespace checks and the construction-only resume smoke
also passed. Read-only schema/digest comparisons checked both exported contexts,
and expression inspection reconfirmed that anonymous evidence seed 1 has no
nonlinear state/input dependence or nonlinear initialization. Runtime source is
unchanged from the preceding full-suite verification.

The first review commit changed this document and its link from
`PREFIT_CONSTRUCTION_AUDIT.md`. The context follow-up changes only this document;
unrelated checkout edits and the supplied experiment artifacts remain untouched.
