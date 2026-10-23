# Directional feedback and saved-model family checks

23 October 2026. This is a construction milestone. Orion's numerical work remains
separate: unfinished optimization does not establish inadequate equations, and
these checks do not rank parent and child NMSEs.

## Feedback changes

Protocol `construction-feedback-revision-4` preserves the previous protocols and
adds the following opt-in behavior:

- A derivative witness distinguishes **model-implied change bound** from
  **observed change interval**. If the model permits an increase of at most 1.95
  while the recorded increase exceeds 2.27, feedback says the upper bound is too
  low. Lowering it cannot repair that witness. The runtime does not prescribe the
  replacement scientific equation.
- Repetition is detected from equation edits and new parameter declarations,
  independently of hypothesis wording, citation IDs and advisory source lists.
  Matching equation labels and whitespace normalize for this comparison. This is
  syntactic bookkeeping, not a general algebraic-equivalence claim.
- Feedback distinguishes the unchanged starting candidate from the rejected trial.
  All repairs remain complete patches relative to that starting candidate.
- Displayed nested evidence headings can resolve to `training_consistency` as a
  citation alias. The audit records normalization; unknown citations still fail,
  and citation normalization cannot bypass the scientific checks.
- Type/parameter-edit errors show the actual eligible equation names. Their local
  guidance explains that an ODE may depend on its own state, an existing parameter
  is not an editable equation, and `Z'=u` integrates `u` rather than making `Z=u`.

## Additional necessary conditions

The implementations inspect actual expanded processes, output mappings, dependency
closures and the scoped initializer graph. They use bounded exact rational
arithmetic. They never optimize, sample parameter guesses, run a numerical rollout,
or substitute measured target histories for hidden trajectories.

1. **Shared initializer/readout family.** At time zero, write the supported
   observation relations as `y = offset + A theta`, with shared coordinates
   `theta`. An exact left-null vector of `A` eliminates all these coordinates.
   A weighted observation discrepancy exceeding the authenticated observation
   error bounds and numerical guard proves incompatibility. Relevant free
   trajectory roots, nonlinear shared-coordinate expressions, uncertain
   conditioning observations and exceeded resource limits defer the check.
2. **Affine trajectory superposition.** Under identical fully specified forcing,
   shared affine ODEs and affine readouts preserve an initial-state linear
   combination whose weights sum to zero. The check obtains initial conditions
   from actual maps and, where exact, algebraic output constraints. Unknown initial
   combinations, differing forcing and unsupported nonlinear dependencies defer.
3. **One-dimensional initial-state variation.** If a closed shared linear
   subsystem has identical forcing and at most one possibly different initial
   coordinate, its trajectory-difference matrix has rank at most one. A witnessed
   nonzero 2-by-2 determinant contradicts this model class. The varying coordinate
   may be an arbitrary trajectory-specific latent initial value; it need not be
   equal across trajectories. More unknown initial directions defer this check.

The last two checks currently require exact conditioning and target observations;
nonzero or unknown error bounds are not treated as zero. Their numerical guards
are inherited from the frozen comparison policy. The superposition certificate
conditions on exact initial reproduction. It is **not a lower bound on achievable
forecast NMSE** when initial discrepancies and unbounded dynamics are allowed.
More generally, an exact-reproduction conflict does not establish a practically
large fitting error. A nonzero scientific acceptance tolerance would require a
separate, predeclared protocol rather than a threshold chosen after seeing results.

Unsupported checks are reported as deferred. A revised model may avoid a proof
by introducing an unsupported nonlinear dependence; that is not evidence that
the scientific problem is solved. The full report retains coverage and assumptions.

## Bounded saved-case experiment

Six saved candidates cover the observed defects, selected without NMSE:

| Source episode | Saved trial | Reason |
| --- | --- | --- |
| P03-r0 | Last rejected trial | Reversed interpretation of the glucose upper bound |
| P06-r2 | Last rejected trial | Repeated insulin equation edits and wrong direction |
| P01-r0 | Accepted trial | Shared initializer family and integrator interpretation |
| P05-r2 | Accepted trial | Type/decay confusion and closed linear subsystem |
| P00-r1 | Retained model | Affine superposition with an inferred latent initial root |
| P06-r1 | Accepted trial | Closed linear subsystem after simpler coefficients |

Each starting candidate receives three fresh seeds and at most three calls per
seed: **18 episodes, at most 54 generation requests**, at most 98,304 charged
tokens per episode. A new namespace binds the selected candidate, source receipts,
policy and code. Checkpoints resume within the same allowance. No fitting,
promotion or automatic follow-up is launched.

The two rejected trials are reconstructed only to recover typed mathematical
objects; this does not relabel them historically accepted. The original eight
training projections and their provenance remain intact. Starting assessments are
sealed separately and verified again, so changed checks are distinguishable from
changes to equations. The saved source is packaged as one archive to limit ACES
inode consumption; it contains data, never executable recovery code.

The edit scope still fixes the variable roster, component kinds, observation
mappings, existing parameter roles and initialization graph. Existing dynamic and
algebraic definitions may change, including their dependencies. If a necessary
repair exceeds this scope, the conflict remains blocking and the proposer should
describe the wider change. Passing these finite checks makes a candidate eligible
for independent fitting; it does not certify scientific validity.

## Inspection

The generated prompt preview shows the exact first requests for all six saved
candidates, the response schema and a marked repair example. The returned package
contains complete replies, equations, parameter-use audits, qualified witnesses,
deferred-check coverage, source provenance and new costs separately from the source
run. Inspect all seeds and equation changes before deciding on another full run.

## Returned ACES diagnostic: 23 October 2026

Job `2248535` completed all 18 episodes using GPT-OSS-20B, low reasoning,
temperature 0.2. All six intentionally selected starting models are T2-easy
candidates; this is not a fresh benchmark-suite success estimate. The returned
archive SHA-256 is
`85cf2d544746e11ac8d92ea8f597e3943f964117a48d0942254af60cb32ce7e0`;
plan `e0095b1c34b270441729b3f0e3ab53981f9e756e474f33fb2f94a44c6b1f4264`.

No revision was accepted. All 54 generation calls were recorded, using 561,583
observed tokens with no missing usage. The 54 replies comprise 20 edited trials
that reached consistency checks but remained blocked, two no-edit responses
requesting wider changes, 31 interface/schema rejections, and one output-token
truncation. All other responses finished normally. There were no fitting packets,
new numerical rollouts, optimizer calls, model promotions or test-data access.

The runtime and evidence replay exactly under the delivered `cf21fa0` source:
all 18 completed episodes, sealed source selection, unchanged controls,
request/reply records and tokenization receipts verified. The existing focused
regression suite passed 81 tests.

The main actionable findings are:

- Family diagnostics suggest an additional independent initial-state direction,
  while this experiment freezes variable and initializer declarations. Of 53
  parseable replies, 25 attempt nonexistent dynamic components, including six
  uses of `initialization_graph` as an equation name. The system does explain
  wider-scope reporting, but the local diagnostic and response schema do not
  make escalation sufficiently distinct from an applicable edit.
- A basal insulin source removes the simple nonincrease implication but leaves
  a shared scalar subsystem unable to distinguish identical initial states and
  forcing. Adding a linear term in an already-present state also does not create
  an independent initial-state direction. Multiplying a zero input contributes
  nothing on the witnessed interval. These were substantive proposer errors.
- Six intermediate trials no longer expose their original directional witness,
  but their function checks defer and other conflicts remain. This is not a
  scientific pass. Positive utilization-as-glucose-source proposals particularly
  need independent mechanistic scrutiny.
- Twelve normalized patches repeated earlier edits; eight later requests could
  explicitly warn about repetition. Six then changed their patch and two repeated
  again. No next reply passed. Malformed expressions can evade this fingerprint.
- All seven `REVISION_SCHEMA` rejections contain duplicate component edits.
  Name the duplicates and unavailable components together in future feedback.
- P00/P01 each have six affine-family conflicts across Gp/I/U; prompt compression
  shows two representative witnesses, both Gp, while retaining the full target
  list. Prefer strong witnesses with clear affected-target coverage. The first
  threshold crossing can be tiny even when later discrepancies are substantial.

No certificate calculation error was found in the audited cases. However, the
frozen guard is not an authenticated simulator/export error bound, and exact
reproduction conflicts do not establish practically large optimized NMSE. Keep
Orion's original-model numerical work separate. In particular, do not replace
those original fitting candidates with these rejected trials.

Recommended follow-up, not implemented by this analysis: scope-aware actionable
feedback plus a distinct wider-revision outcome; stronger per-target witnesses;
then a matched 20B/120B comparison on the same saved cases and budget. State or
initializer additions require their proper validated construction interface, not
relaxed unknown-symbol acceptance. Do not silently broaden one comparison arm.

The local report at `output/directional-family-analysis-20261023/REPORT.html`
contains training-witness plots, all 54 replies and exact prompts, per-seed
outcomes, independent counts and frozen-code replay evidence. Generated reports
and raw responses are not committed.
