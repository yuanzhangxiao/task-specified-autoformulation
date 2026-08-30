# Hybrid scientific judge protocol

## Status and boundary

The hybrid protocol is calibration-only. It does not change production search,
candidate ranking, benchmark prompts, or finalized benchmark data. It evaluates
the canonical executable candidate after behavior-preserving repair and records
the raw proposal repair ledger separately.

Completely unreferenced processes and unused parameters may be removed before
judging only when removal cannot alter executable behavior. Retained components
that lack a path to requested targets remain in the candidate and are certified
to the judge. Repair frequency is a proposer-contract diagnostic, not a
scientific model score.

## Authority and provenance

The runtime extracts task-required mechanism bullets only from the public prompt
and freezes them as `ScientificRequirement` records. Absence of the explicit
public marker creates no inferred requirement. Requirement sources are
`domain_expert`, `benchmark`, `proposer`, and `runtime`; enforcement is `hard`,
`soft`, or `descriptive`.

Mechanism tags attached to candidate states and processes become proposer-owned
`ProposerClaim` records. They cannot create a task requirement or earn task
coverage merely by existing. They are evaluated only for scientific support,
connectivity, and consistency.

## Deterministic facts

For both candidates, the runtime constructs the same directed expression graph
and certifies:

- paths from declared task inputs to requested targets;
- paths from retained states and processes to requested targets;
- equation-level drivers of latent states;
- candidate-owned mechanism claims and their component subjects.

The runtime owns absolute pass/fail results for task-input reachability, claimed
component reachability, latent incoming pathways, and latent target influence.
These facts contain no fit data, trajectory data, hidden mechanism, mutation
label, or scientific verdict.

## LLM absolute assessments

The judge sees Candidate A and Candidate B together in randomized order but
answers each requested predicate separately for each candidate with `pass`,
`fail`, `indeterminate`, or `not_applicable`. Required mechanisms are always
applicable. Every answer cites candidate or certified-fact evidence.

For every public task requirement, the judge answers:

1. whether an identifiable candidate structure represents the requirement;
2. whether that representation has scientifically relevant target influence.

Candidate-wide semantic questions assess source roles, sink roles, semantic
flux duplication, conflicting claims, accumulator justification, meaningful
delays, appropriate saturation, and support for proposer-owned claims.

The provider must return exactly the runtime-requested criterion/subject keys.
Post-schema validation rejects missing, duplicate, or invented units.

## Direct comparative residual

The same blinded call also asks which candidate is better on three irreducibly
relative properties:

- parsimony while satisfying the public task;
- fewer unsupported scientific assumptions;
- mechanistic interpretability.

These answers use `candidate_a`, `candidate_b`, `tie`, or `indeterminate`. They
are never converted into absolute pass/fail claims. Candidate order is reversed
and repeated during calibration.

## Generic conjunctions

The runtime, not the LLM, instantiates generic group templates. Instances come
only from public requirements and submitted candidate structure; hidden ground
truth never creates a group. The initial templates are:

- one group per public user/benchmark requirement;
- task-input connectivity;
- proposer-claim integrity;
- balance semantics;
- latent-state validity;
- delay and saturation claim validity.

Atomic predicates inside a group are combined with logical AND. For example,
a public requirement is complete only when it is both represented and
functionally connected. Atomic predicates are not also added as independent
bonuses.

The candidate's conjunctive score is the weighted fraction of complete,
determined, applicable groups. The partial score is the weighted atomic pass
fraction and is used only as a small tiebreak signal:

```text
shaped = (conjunctive + epsilon * partial) / (1 + epsilon)
```

Hard public requirements additionally produce an explicit eligibility status.
One candidate satisfying all hard requirements defeats a candidate with a
known hard-requirement failure. Missing information remains indeterminate and
reduces reported coverage.

The final pair decision is exploratory and calibration-owned:

```text
decision(A, B) = shaped(A) - shaped(B)
               + lambda * comparative_residual(A, B)
```

The partial weight, comparative weight, and tie threshold are recorded CLI
parameters. They must be chosen on calibration training data and frozen before
held-out evaluation. They are not production defaults.

## Question-level truth

`build_hybrid_judge_label_template.py` creates one versioned label record per
pair without requiring domain-expert review at this protocol-engineering stage.
Deterministic items are filled from the canonical graph. Each controlled
mutation type has an explicit, typed contract containing only the semantic and
comparative consequences guaranteed by its construction. For example, a
duplicated-flux mutation certifies failure of the no-duplication predicate, but
does not claim to establish whether an unrelated saturation law is appropriate.

Questions not certified by either the runtime or that mutation's contract stay
`unlabeled` with source `not_scored_by_mutation_contract`. They remain in the
judge request for protocol diagnostics but are excluded from question-level
accuracy. Canonically identical candidates receive comparative ties. Unknown
mutation types fail label generation instead of inheriting guessed labels.
Every scored label includes its exact runtime or mutation-contract provenance
and rationale. Gold labels, mutation types, and baseline identities are never
sent to the judge.

The analyzer reports:

- deterministic runtime consistency as an integrity check, never as LLM
  accuracy;
- mutation-targeted LLM absolute verdict accuracy;
- exact paired semantic accuracy when both sides are contract-labeled;
- direct comparative question accuracy;
- absolute-only, comparative-only, and combined preference accuracy;
- pair-aggregated accuracy;
- order consistency;
- certified-label coverage and criterion-specific results.

An overall baseline/mutation preference is insufficient for atomic evaluation:
a globally degraded candidate may genuinely improve one narrow predicate. This
calibration therefore reports both the broad pair-level test and the narrower
certified atomic subset. It does not claim full domain-expert validation of every
scientific rubric question.

## Reproducibility and safety

Every call remains content-addressed, cached, logged, order-reversed,
checkpointable, and shard-resumable. Schema/format failures may receive bounded
contract-only repair attempts. No repair prompt contains a scientific answer,
fit result, hidden model, or expected label. The LLM never emits an aggregate
score or overall winner.

If all bounded provider attempts fail, the runner writes a versioned terminal
record to `hybrid_judge_failures.jsonl` and continues with the next logical call.
It does not parse hidden reasoning or impute a score. Resume treats successful
CSV rows and terminal failure records as disjoint completed outcomes. Shard merge
requires their union to cover the planned calls. Analysis reports scientific
accuracy conditional on a valid structured response, structured-response success
rate, and end-to-end accuracy that counts a terminal response failure as an
incorrect judge decision.

Each shard also stores an immutable `hybrid_judge_run_manifest.json` containing
the pair-file digest, selected pair identifiers, sharding rule, model settings,
seeds, repetition count, and scoring configuration. Resume fails closed if the
current configuration differs or if saved outcome keys do not belong to the
selected shard. This prevents an expired shell environment from silently mixing
an older calibration pair set into an existing output directory.

For seeded Ollama calls, provider attempt 1 uses the registered repetition seed.
If that attempt fails, repair attempt `n` uses `seed + n - 1`. This deterministic
fallback prevents an empty-content response from reproducing forever under the
same seed while preserving the original seed for every ordinary successful call.
The actual attempt number and sampling seed are stored with the raw response;
all attempts still count as one logical judge call.

If Ollama returns reasoning but leaves `message.content` empty, subsequent repair
attempts use Ollama's OpenAI-compatible chat endpoint with JSON mode and
`reasoning_effort: none`, a strict `json_schema` response format, and an explicit
final-content instruction. If that endpoint wraps the object in prose, the runtime
accepts it only when exactly one embedded object independently satisfies the full
Pydantic schema. Expected-unit post-validation remains mandatory, so this recovery
changes only the provider emission channel and does not weaken the trusted boundary
or infer an answer from hidden reasoning.

The baseline-structure-held-out evaluation later showed that every accepted
response completed on the first native JSON-schema attempt and no
reasoning-disabled repair attempt recovered an empty response. The explicit
`json_schema_native_retry` ablation tests the minimal alternative: every attempt
uses `/api/chat`, the same compact schema, and low reasoning; only the
deterministic sampling seed and contract-only finalization hint change. The
historical `json_schema` behavior remains available as the matched control.
`configs/hybrid_judge_native_retry_pilot_v1.json` freezes the four-pair,
40-logical-call pilot and its relative expansion gate before new calls are made.

The four-pair pilot improved structured-response success from `1/40` in the
matched historical control to `37/40`, with `0.892` conditional combined
accuracy and perfect pair aggregation. Expansion over all ten baseline-
structure-held-out pairs produced `97/100` valid responses. At the frozen
adaptive operating point, paired coverage was `1.000`, conditional pair
accuracy was `0.920`, end-to-end pair accuracy was `0.920`, and order
consistency was `0.800`; all four numerical gates passed. The result is an
engineering validation because transport recovery was designed after observing
the original held-out completion failure, not an untouched confirmatory paper
holdout.

The residual three failures all occurred on one wrong-meal-sink pair after ten
native attempts, with populated reasoning and empty final content. The next
manifest-pinned ablation, `json_schema_openai_thinking_retry`, returns to the
OpenAI-compatible endpoint while retaining `reasoning_effort: low`. This changes
one factor relative to native retry and one factor relative to the historical
reasoning-disabled fallback. Its ten calls are frozen in
`configs/hybrid_judge_openai_thinking_retry_pilot_v1.json`.

That endpoint ablation completed `10/10` calls, confirming that disabling
GPT-OSS reasoning was the main cause of the historical empty repair responses.
It did not preserve judgment stability: combined accuracy was `0.800`, order
consistency was `0.600`, and errors again favored Candidate A when the baseline
occupied B. The OpenAI-compatible Ollama endpoint is therefore not selected as
the repair transport. A separately pinned vLLM container smoke test is the next
cross-runtime diagnostic; it must succeed on one A40 before any vLLM client is
added to the calibration runner.

### Ollama protocol-completion transport ablation

The calibration runner supports `--ollama-response-mode json_schema` (the frozen
control) and `--ollama-response-mode tool_call` (the protocol-completion
alternative). Tool-call mode supplies exactly one `submit_structured_response`
function whose arguments use the same compact JSON Schema as the control. A valid
response must contain exactly one call to that function; its arguments then pass
the unchanged Pydantic and expected-unit validation. Ordinary message content and
the thinking channel are never used as substitute answers.

Transport mode is included in the content-addressed cache key and immutable shard
manifest, so the two conditions require separate output roots and cannot be mixed
during resume. This ablation changes only how the model submits its assessment:
the blinded candidates, scientific questions, seeds, repetitions, repair limits,
scoring, and failure accounting remain fixed. Reports compare structured-response
success, conditional scientific accuracy, end-to-end accuracy, latency, and token
usage before any production integration.

The initial experiment retains five repetitions in both orders. Existing output
supports offline ablations of absolute-only, comparative-only, and combined
decisions before any production integration. The operating-point analyzer also
enumerates every frozen repetition subset for one call in either single order and
for one through five repetitions in both orders. It clusters uncertainty by pair
rather than treating repeated calls as independent scientific examples.

The frozen four-pair transport pilot found that always-on tool calls removed the
two remaining structured-response failures (`40/40` versus `38/40`) but reduced
conditional combined preference accuracy from `1.000` to `0.825`, order
consistency from `1.000` to `0.650`, and repeat ICC from `0.895` to `0.575`.
All material preference errors favored Candidate A when the valid baseline was
Candidate B. Averaging both candidate orders improved tool-mode accuracy to
`18/20`, but did not match the JSON-schema control's `18/18`. Always-on tool mode
therefore failed the predeclared expansion gate and is not a production default.

The explicit `json_schema_tool_fallback` mode instead preserves the general
scientific prompt and uses the existing JSON-schema path for attempts 1 through
`max_attempts - 1`. The final bounded attempt switches to the single validated
tool only when an earlier attempt received the typed `empty_provider_content`
diagnostic. Ordinary schema-validation failures do not activate the tool. The
confirmation configuration uses eleven total attempts so the first ten exactly
preserve the frozen JSON control budget and only a terminal case incurs one extra
tool generation. Accepted score rows record the transport, provider-attempt count,
and successful sampling seed.

An offline cascade simulation retained all 38 accepted JSON-schema judgments and
filled only their two terminal failures from the tool pilot. It achieved `40/40`
response success, `1.000` combined conditional and end-to-end accuracy, `1.000`
order consistency, and `0.911` repeat ICC. This simulation motivates, but does
not replace, a fresh end-to-end 40-call confirmation of the bounded cascade.

The first end-to-end cascade confirmation reached 29 of 40 calls before its
two-hour allocation expired: 6 native-JSON successes, 15 final-tool successes,
and 8 terminal failures. Every terminal failure contained a complete tool call
but used the invalid key `ver verdict` in place of `verdict` inside one or more
evidence-bearing assessment objects. The scientific evidence and verdict literal
were otherwise present. This is classified as a provider serialization defect,
not a scientific repair opportunity.

The tool parser therefore permits one exact deterministic key repair. It renames
`ver verdict` only when the object also contains evidence, the correct `verdict`
key is absent, and the value is one of the closed-schema verdict literals. It
does not use fuzzy matching, does not repair collisions, and does not change a
verdict value or scientific statement. Full Pydantic and expected-unit validation
still follow. The accepted score row and raw event record the repair count.

The completed corrected continuation produced 39 structured responses and one
recorded provider failure in 40 planned calls. Of the successful responses, 29
used tool fallback and 10 used native JSON schema; four responses required 20
exact key repairs. The completion rate improved from the frozen JSON control's
`0.950` to `0.975`, but conditional combined preference accuracy fell from
`1.000` to `0.667`, end-to-end accuracy from `0.950` to `0.650`, and order
consistency from `1.000` to `0.368`. The remaining failure emitted no tool call,
so no safe deterministic repair was possible. The fallback therefore failed the
scientific-stability gate. JSON schema is the primary calibration transport;
tool modes remain diagnostics and will not be expanded to the 250-call protocol.

Run the offline call-budget analysis on a complete merged JSON-schema result:

```bash
python scripts/analyze_hybrid_operating_points.py \
  --scores hybrid_judge_scores.csv \
  --failures hybrid_judge_failures.jsonl \
  --labels hybrid_labels.jsonl \
  --protocol-config configs/hybrid_judge_protocol_v1.json \
  --output hybrid_judge_operating_points.json
```

The failure ledger calls candidate orientation `order`, with values `baseline_a`
and `baseline_b`; `baseline_position` separately records `A` or `B`. There is no
`candidate_order` field.

The same analyzer evaluates a symmetry-preserving adaptive policy. It begins with
one JSON-schema call in each orientation and emits no decision unless both
orientations succeed. Only a missing orientation advances to the next frozen
sampling seed; a successful orientation is not called again. Retry limits from
one through all five frozen seeds are evaluated for every possible starting seed.
The report includes paired-response coverage, conditional and end-to-end pair
accuracy, expected and maximum calls per pair, retry activation, order
consistency, decision variability, and pair-clustered confidence intervals. This
adaptive analysis avoids counting a single position-biased response as a complete
pair judgment while avoiding unconditional repeated calls.

### Frozen baseline-structure-held-out evaluation

`configs/hybrid_judge_protocol_v1.json` freezes the first held-out operating
point: JSON-schema transport, no tool fallback, both candidate orientations,
retry of only a missing orientation, at most five distinct seeds per orientation,
abstention unless both orientations succeed, arithmetic-mean aggregation after
orientation normalization, and the calibrated scoring/tie parameters. The
frozen calibration point used 2.400 logical calls per pair on average and reached
`0.960` paired coverage, `0.992` conditional pair accuracy, and `0.952`
end-to-end pair accuracy. These are selection-set measurements, not held-out
performance claims.

The first held-out evaluation uses newly selected candidate structures while
retaining the already audited mutation contracts. The builder fingerprints the
canonical repaired executable baseline with identifiers and summaries excluded,
rejects every structure present in the calibration pair file, removes duplicate
new baselines, requires all four base mutations, and then adds the retained
disconnection control. Its manifest records the calibration-pair digest and the
selected structural fingerprints. This is baseline-structure-held-out, not
mutation-family-held-out; it tests transfer across candidate structures but not
across previously unseen defect definitions.
Run-directory-derived source pair IDs may have appeared in an earlier pair file
even when a rerun selected a new canonical structure. After the structural
holdout check passes, the builder therefore assigns evaluation-local pair IDs
from the unseen structural fingerprint, mutation type, and source ID. Identifier
reuse alone neither admits nor rejects a baseline; canonical structural overlap
remains the exclusion criterion.

Before results are opened, the held-out gate is frozen as:

- conditional pair accuracy at least `0.90`;
- end-to-end pair accuracy at least `0.85`;
- symmetric paired-response coverage at least `0.90`;
- order consistency at least `0.80`.

Build two unseen baselines (ten pairs after augmentation) from a run root not used
to create the calibration pairs:

```bash
python scripts/build_hybrid_judge_heldout_pairs.py \
  --runs-root /path/to/unused/runs \
  --data-root /path/to/public/data \
  --calibration-pairs /path/to/calibration/pairs.jsonl \
  --baseline-count 2 \
  --output /path/to/heldout/pairs.jsonl

python scripts/build_hybrid_judge_label_template.py \
  --pairs /path/to/heldout/pairs.jsonl \
  --data-root /path/to/public/data \
  --output /path/to/heldout/hybrid_labels.jsonl
```

The held-out measurement collects all five frozen seeds in both orientations so
the adaptive policy can be replayed exactly without outcome-dependent new calls.
The operating-point analyzer receives `--protocol-config
configs/hybrid_judge_protocol_v1.json`; it marks exactly one adaptive row as the
frozen selection. No prompt, scoring weight, threshold, transport, retry rule, or
mutation contract may change after the held-out calls begin.

`build_hybrid_judge_pairs.py` preserves the original frozen pairs as controls
and adds a retained disconnected pathway for every old repaired-away
disconnection. The retained variant uses a driven, relaxing latent module whose
components survive canonicalization but have no target path. This tests the
production judge boundary without confusing a harmless discarded declaration
with an executable scientific defect.

### Frozen vLLM-low diagnostic boundary

The ten-pair vLLM-low evaluation completed all `100/100` structured responses,
removing provider completion as a confounder. Its per-call combined accuracy was
`0.810`, full ten-call pair aggregation was `0.900`, and order consistency was
`0.740`. Under the predeclared missing-orientation-only adaptive protocol, perfect
response success reduced the operating point to one call in each orientation;
pair accuracy was `0.880`. It therefore missed the existing `0.90` conditional
pair-accuracy and `0.80` order-consistency gates even though transport reliability
was perfect. Absolute-only preference accuracy was `0.500`, while comparative-only
accuracy was `0.810`; runtime certification remained `1.000`.

`analyze_hybrid_diagnostics.py` attributes this result without new LLM calls. It
validates that stored decisions exactly match the frozen hard-requirement override,
absolute shaped-score delta, comparative residual weight, and tie threshold before
producing any diagnostic. It reports pair and mutation margins, candidate-order
means, repetition consistency, certified atomic-question accuracy, and a generic
aggregation-sensitivity grid. Candidate pairs are grouped by canonical baseline
structure, and weight sensitivity includes leave-one-baseline-structure-out folds.
Because these outcomes have already been opened, every alternative aggregation
and cross-structure estimate is exploratory. A revised weight or threshold must be
frozen and tested on new baseline structures before it becomes a performance claim
or enters search.

`audit_hybrid_judge_evidence.py` is the next read-only boundary. For every
mutation-contract label, it normalizes Candidate A/B back to baseline/mutated,
checks the certified verdict, and preserves the exact public evidence attached to
an incorrect answer. The JSONL contains every error; the Markdown report groups
actual-verdict patterns and candidate-order counts and shows a bounded set of
representative rationales. It does not inspect hidden reasoning, infer missing
answers, alter scores, or issue new LLM requests. Prompt or deterministic-fact
changes follow only after these stored rationales identify the recurring error.

The frozen audit found that GPT-OSS frequently asserted a positive sign when an
added negative input term was visible, treated an exactly repeated additive term
as acceptable merely because algebra could combine it, and declared comparative
ties from equal declaration counts while overlooking changed equations. These
errors occurred in both candidate orientations. They motivate protocol version
`hybrid-judge-protocol-2` and structural-fact schema `structural-facts-2`.

The new facts remain syntax-only and symmetric. For every process and state
equation, runtime provides the source expression, flattened top-level additive
terms with explicit polarity, symbol membership in those signed terms, and groups
of exactly repeated same-polarity AST terms. The facts do not label a term as a
scientific source, sink, assumption, or duplicated physical flux. The judge must
still interpret those facts against the public task and proposer claims. The
general rubric now states that every signed occurrence must be inspected, exact
repetition cannot be excused solely by coefficient simplification, and equation-
level changes count in comparative assumption and parsimony judgments even when
declaration counts are equal.

`configs/hybrid_judge_vllm_facts_pilot_v1.json` freezes a matched development
ablation over the two wrong-sink and two duplicated-flux pairs. It reuses the same
20B model, low reasoning, five seeds, both orientations, response schema, retry
limit, and score weights. The manifest records both protocol and fact-schema
versions so an old output directory cannot silently resume under the new prompt.
Because the pair outcomes were already opened, a passing result only authorizes
freezing an unchanged protocol for new baseline structures.

The matched facts pilot completed all `40/40` calls and correctly selected the
baseline for all four pairs, but it failed the atomic and stability gates. Exact
duplicate-flux accuracy improved from `0.55` to `0.95`, while source-role
accuracy fell from `0.45` to `0.25`, targeted comparative accuracy was `0.45`,
and order consistency was `0.55`. Syntax-only repetition facts therefore solve
an evidence-location problem, but signed-symbol polarity alone does not solve
scientific role inference.

### Sign-blinded atomic occurrence development protocol

Protocol version `hybrid-judge-protocol-3-atomic-occurrence` adds a separate
structured scientific call before the ordinary blinded comparison. Runtime
flattens every state equation and generated process into top-level additive
occurrences, removes only
the certified outer sign from the first-stage request, and supplies the unsigned
term, governed quantity, public prompt, public symbol contract, component
definitions, and proposer-owned claims. The first-stage judge must infer
`positive_contribution`, `negative_contribution`, `context_dependent`, or
`insufficient_public_information`. It is never shown the mutation label,
expected answer, reference model, fitted trajectory, or hidden benchmark fact.

Runtime then compares a determinate inferred direction with the privately held
certified polarity. This compatibility operation is deterministic, but its
scientific premise retains explicit LLM provenance. Candidate-wide source and
sink verdicts in atomic mode are derived from those compatibility checks rather
than requested again from the second-stage judge. Indeterminate scientific
directions remain indeterminate; they are not converted into failures.

The same first-stage call receives pairs of exact same-polarity repeated
expressions without calling them scientific duplicates. It classifies each pair
as the same physical contribution, distinct contributions, or insufficiently
specified. A determinate same-contribution answer forces failure of the
nonduplication predicate. Otherwise, the second-stage broader semantic verdict
is retained so nonexact equivalent pathways remain judgeable.

The second-stage blinded comparison receives the frozen atomic inferences and
runtime polarity-compatibility outcomes. It answers all other absolute units and
the same three comparative questions. This prevents a later comparative tie from
silently ignoring an atomic mismatch while leaving the scientific inference with
the LLM. Both calls are separately content-addressed, cached, logged, and
schema-validated. A failure in either stage is a terminal failure for that planned
judgment, annotated with its stage; merge-safe score and failure files are created
before provider calls.

`configs/hybrid_judge_atomic_occurrence_v1.json` freezes the next development
factorial. GPT-OSS 20B and GPT-OSS 120B receive the same four opened pairs, public
prompt, atomic protocol, low reasoning, temperature, seeds, orientations, retry
budget, schema, and scoring. The 20B condition uses one A40; 120B uses four A40s
with tensor parallelism. Each model produces 40 paired judgments and 80 logical
LLM stages. This is a model-scale and protocol diagnostic, not a confirmatory
paper holdout. If the development gates pass, the selected protocol must be
frozen unchanged before testing genuinely unseen baseline structures.

### Frozen canonical-structure confirmation

The matched factorial selected the 120B, low-reasoning, two-stage atomic
protocol. Its confirmation is defined by
`configs/hybrid_judge_atomic_confirmation_v1.json`; this file is frozen before
any confirmation call. It preserves the public prompt, model, serving endpoint,
atomic and hybrid schemas, temperature, five seeds, both orientations, ten-attempt
provider budget, group scoring, comparative weight, and tie threshold. It changes
only the canonical baseline structures supplied to the judge.

`build_hybrid_judge_confirmation_pairs.py` inventories completed model-selection
summaries across one or more run roots. It canonicalizes and deduplicates their
selected structures, then excludes every structure appearing as the valid member
of any supplied historical pair file. The exclusion files, hashes, pair counts,
and total unique excluded structures are written to a persistent manifest. The
output contains exactly two prespecified mutations for each selected unseen
structure: duplicated glucose flux and a wrong-sign meal occurrence. Confirmation
pair identifiers use a separate namespace. Mutation labels and certified polarity
remain hidden from both LLM stages.

The primary confirmation has two unseen baseline structures, four pairs, five
seeds, and both orientations: 40 paired judgments and 80 separately logged LLM
stages. It passes only if all predeclared gates pass: response success at least
`0.95`, perfect four-pair aggregation, order consistency at least `0.80`, each
targeted atomic accuracy at least `0.75`, and targeted comparative-question
accuracy at least `0.60`. `analyze_hybrid_judge_confirmation.py` applies these
thresholds mechanically. A failed gate is reported as a failed confirmation;
changing a prompt, model, weight, or threshold creates a new development protocol
and requires another fresh canonical-structure holdout.

The frozen confirmation passed every gate: `0.975` response success, `1.000`
pair-aggregate accuracy, `0.842` order consistency, `1.000` accuracy for each of
the wrong-sink and exact-repeat atomic tasks, and `0.932` targeted comparative
accuracy. One of 40 planned judgments failed because the second-stage response
included every requested absolute unit plus redundant candidate-level source and
sink units already owned by the atomic stage.

Protocol version 3 therefore has one contract-only implementation repair. In
atomic mode, and only when the requested unit set is otherwise exact, the runtime
may discard both redundant `source_roles_consistent:candidate` and
`sink_roles_consistent:candidate` units. Missing units, only one of the two role
units, or any other extra unit remain errors. The response is rebuilt and
revalidated; the separate cache role, manifest repair version, removed keys, and
repair count preserve provenance. This does not alter the frozen scientific
prompt, atomic inference, verdicts, evidence, comparative questions, weights, or
decision rule.

### Equivalence and non-ordered tradeoff development

`configs/hybrid_judge_equivalence_tradeoff_v1.json` freezes the next development
milestone over the two now-open confirmation structures. Each structure produces
one mathematically identical pair by reordering top-level additive terms and
three pairs whose members carry different controlled defects. Equivalence calls
and all three comparative questions have a certified tie target. Tradeoff overall
preferences have no gold winner: only mutation-certified per-side wrong-sink,
duplicate-flux, or one-sided-accumulator failures are scored.

The experiment retains the confirmed 120B low-reasoning atomic protocol, five
seeds, both orientations, scoring weights, and tie threshold. Its predeclared
development gates cover response completion, equivalence tie recognition,
mutation-certified atomic accuracy, order consistency, orientation bias, and
repeat variance. Tradeoff winner counts are descriptive. Passing this milestone
does not establish a calibrated continuous search score; it supports freezing an
interpretation for a later new-structure validation.

The development run completed all 80 paired judgments. It achieved perfect
equivalence-tie, mutation-certified atomic, and labeled comparative accuracy;
order consistency was `0.900` and mean repeat decision SD was `0.055`. The
predeclared outcome was still **FAIL** because mean tradeoff orientation bias was
`0.115`, above the frozen `0.100` maximum. Attribution localized every directional
inconsistency to wrong-sink versus duplicate-flux pairs. In that comparison,
`mechanistic_interpretability` often preferred the displayed Candidate A in both
orientations. This is a position-sensitive relative judgment between two distinct
known defects, not an atomic-detection error.

`analyze_hybrid_symmetric_aggregation.py` is the next post-hoc development step.
It reuses frozen rows and compares three identity-normalized rules without new LLM
calls: paired final-decision mean, strict question consensus, and orientation-
interval abstention. Question consensus accepts a verdict only if both candidate
orders agree after mapping them to the same candidate identities; disagreements
become indeterminate and the unchanged runtime scorer is rerun. The interval rule
uses the mean decision as its center and half the A/B decision gap as uncertainty,
withholding a result whenever that interval crosses the tie boundary. Reports keep
coverage, equivalence truth, repeat stability, and disagreement counts distinct.
Tradeoff preferences remain descriptive and cannot establish rule accuracy.

### Frozen question-consensus validation

The development analysis selected `paired_question_consensus` without changing
the judge prompt, model, atomic questions, score weights, or tie threshold. The
rule first normalizes both orientations to the same candidate identities. An
absolute or comparative answer survives only when both orientations agree;
otherwise that unit becomes `indeterminate`. Deterministic graph facts must agree
exactly. The standard hybrid scorer is then rerun on the consensus result. Half
the normalized A/B final-decision gap is retained as an uncertainty diagnostic;
it does not trigger automatic abstention in this protocol.

`configs/hybrid_judge_consensus_validation_v1.json` freezes the confirmatory
contract. `build_hybrid_judge_consensus_validation_pairs.py` excludes every
canonical baseline structure present in any supplied opened pair file and selects
exactly two unseen structures. Each structure generates seven pairs:

1. one exact additive-reordering equivalence with certified tie truth;
2. three baseline-versus-single-defect pairs for wrong meal sign, exact repeated
   glucose flux, and unjustified one-sided accumulation;
3. two monotonic pairs in which a wrong-sign or repeated-flux candidate is
   compared with the same candidate plus an unjustified accumulator; and
4. one wrong-sign-versus-repeated-flux tradeoff with no overall winner label.

The monotonic labels require only set inclusion of controlled defects: the first
member's defect is preserved in the second, which adds one certified accumulator
defect. They do not assert which distinct scientific defect is worse. Tradeoff
preferences remain descriptive. Mutation contracts are held privately for
evaluation and are never shown to either LLM stage.

The planned run has 14 pairs, five seeds, and both orientations: 140 judgments
and 280 separately logged LLM stages on the frozen 120B low-reasoning vLLM
protocol. It passes only if every predeclared completion, coverage, tie,
dominance, mutation-certified atomic, stability, and orientation-uncertainty gate
passes. A failed gate cannot be repaired by retuning on these structures. Search
objective integration remains a separate later milestone even if validation
passes.

The validation did pass all gates. Of 140 planned judgments, 139 returned valid
responses. This yielded `0.986` paired-seed coverage, `0.983` labeled accuracy,
`1.000` equivalence-tie accuracy, `0.980` known-dominance accuracy, `1.000`
known-dominance pair accuracy, and `1.000` accuracy for each targeted atomic
family. Modal preference consistency was `0.986`, repeat decision SD was `0.007`,
mean orientation half-gap was `0.017`, and comparative disagreement rate was
`0.058`. These are confirmatory results for the frozen judge aggregation, not a
license to tune its questions or weights.

`configs/hybrid_judge_consensus_operating_point_v1.json` defines the subsequent
post-hoc call-budget analysis. It makes no LLM requests. For every canonical pair,
each stored seed is rotated into the first position. Candidate configurations
request one through five complete paired seeds under one through five maximum
distinct seed attempts. An incomplete seed is replaced only by rerunning both
orientations at the next seed. It is never combined across mismatched seeds, and
scientific disagreement does not activate a retry. When multiple complete paired
seeds are requested, their question-consensus decision values are averaged before
the frozen tie threshold is applied.

The analyzer reports expected and maximum judge operations and logical LLM
stages, response-replacement activation, decision coverage, clustered accuracy,
equivalence, dominance, repeat SD, and modal consistency. Selection first requires
every frozen scientific and coverage gate, then minimizes expected logical LLM
stages, maximum logical stages, requested complete seeds, and maximum attempted
seeds in that order. Unlabeled tradeoff winners remain excluded. The selected row
is a development choice for a later search-integration pilot, not another held-out
scientific result.

The selected operating point requests one complete paired seed and permits one
whole-pair replacement at a second distinct seed. Frozen rotation analysis
estimated `2.029` judge operations (`4.057` logical atomic/comparative stages)
per comparison, with a maximum of four operations/eight stages. It retained full
decision coverage, `0.983` labeled accuracy, perfect equivalence, `0.980`
known-dominance accuracy, and perfect pair-aggregated dominance.

`configs/hybrid_search_objective_pilot_v1.json` freezes the subsequent online
development boundary. The existing production defaults are unchanged. In the
new opt-in mode, the first fitted candidate seeds an incumbent and each later
round supplies exactly one challenger. Both judge orientations use the same
seed; a terminal failure in either orientation discards that incomplete pair and
retries both orientations once. Scientific disagreement never triggers a retry.
Question consensus is recomputed before a bounded challenger science preference
is combined with symmetric relative validation-NMSE improvement. The only new
tradeoff parameter is the science weight. A positive combined preference replaces
the incumbent; zero, indeterminate science, or exhausted response attempts retain
it. Comparisons against different incumbents are never treated as a shared global
score. The mode requires a one-member beam, records its full challenge ledger for
deterministic resume, and is structurally unable to open test data.

The first eight-round plumbing run exposed an aggregation pathology rather than
a new judge failure. After A/B question consensus, one comparative answer could
remain determined while two became `indeterminate`. Version 1 averaged only the
determined answers, so that single answer received the entire comparative
weight. Version 2 instead encodes comparative answers as `+1` for candidate A,
`-1` for candidate B, and `0` for both tie and indeterminate, then divides by the
fixed set of three schema-required comparative questions. Indeterminate evidence
therefore remains neutral but still consumes its share of the denominator. This
is a deterministic aggregation change: it does not alter the prompt, questions,
LLM answers, absolute groups, weights, or tie threshold. The legacy exclusion
rule remains available only to reproduce frozen version-1 analyses.

`configs/hybrid_search_objective_pilot_v2.json` freezes this rule before the
next development search. Checkpoints record both the new protocol version and
the indeterminate policy so version-1 and version-2 runs cannot silently resume
across the scoring boundary.

### Model-semantic extension: target mappings and initialization

The second objective-pilot candidate exposed two scientific contracts that the
validated judge did not ask about. An observation mapping can report only one
component of a public target while the dynamics use that symbol as a component,
and an absolute observed state can be fixed to zero without any public or model
justification. These are not parser failures: both candidates can compile and
fit. They therefore require scientific evaluation after deterministic repair.

Protocol version
`hybrid-judge-protocol-4-target-mapping-initialization` adds two atomic,
benchmark-general absolute questions:

- `target_mapping_semantically_consistent`: does each mapping generate the
  complete public quantity rather than silently omit or double-count a known
  component or contradict the model's own use of that symbol?
- `initialization_semantically_consistent`: is each initialization compatible
  with absolute-versus-deviation semantics and any explicitly available initial
  observation, with fixed zero requiring public or candidate justification?

Both questions explicitly require `indeterminate` when the public task and the
candidate do not supply enough semantics. The hidden generator and mutation
contract are never included in either LLM prompt. The questions are opt-in, so
all previously frozen protocol-3 rows and scores remain reproducible.

The completed v1 calibration is retained but adjudicated invalid rather than
relabeled. Its target-mapping baseline used `Uii + U` without first checking
whether process `U` already contained `Uii`; on the selected structures this
could double-count the supplied component. Its fixed-zero observed-state
mutation was also behaviorally inert: the simulator initializes an
identity-mapped state from its observed channel at each causal boundary.
`hybrid_judge_model_semantics_validation_v1_adjudication.json` records both
construction errors and excludes the v1 accuracy from protocol and paper claims.

Initialization is now resolved at the deterministic contextual-repair boundary.
For a one-step causal task, an identity observation mapping such as channel `I`
mapped to state `I` causes any submitted fixed, ranged, or analytic initializer
to be replaced by the authoritative observed-channel binding. Open-loop tasks
retain their explicit initializers. Versioned structural facts expose the
effective runtime binding, but no LLM initialization question is included in
protocol v2.

`hybrid-judge-protocol-5-target-mapping-certified` retains only the target-mapping
question. Its builder fails closed unless the target process is a unique process,
does not already reference the supplied component `Uii`, the complete mapping has
exactly symbols `{Uii, U}`, the omission has exactly `{U}`, and the candidates are
otherwise identical. The frozen v2 configuration uses the established 120B
low-reasoning atomic protocol, both orientations, five seeds, paired question
consensus, and the neutral fixed denominator. These are development pairs because
prior structures may be reused and that reuse is recorded; passing permits only a
later versioned search pilot.

The v2 run completed all calls but failed its scientific gates: labeled accuracy
was `0.100`, pair aggregation was `0.000`, and target-mapping absolute accuracy was
`0.150`. Post-run review found that the public prompt called `U` only "glucose
disposal rate." It did not define `U` as the total of insulin-independent and
insulin-dependent contributions. Because hidden component semantics cannot serve
as a judge label when the public contract leaves them ambiguous, v2 is retained as
evidence of prompt underspecification rather than evidence that the judge cannot
evaluate a defined total.

The frozen v3 development rerun isolates that interpretation. It reuses the exact
v2 pair bytes, candidates, mutation labels, model, seeds, retries, scoring, and
aggregation. An audited overlay changes exactly three proposer-prompt phrases: it
defines `U` as total glucose utilization/disposal, identifies `Uii` as the supplied
insulin-independent contribution, and connects the already-required delayed
insulin pathway to the insulin-dependent contribution to `U`. Numeric tables and
the judge prompt are byte-identical to v2. The original release remains unchanged,
and the overlay records hashes for the source prompt, revised prompt, source pairs,
and every unchanged release file. Labels are regenerated because public-requirement
identifiers depend on prompt text. Search integration remains blocked unless every
unchanged v2 gate passes.

The v3 clarification fixed the scientific judgment on every usable paired trial:
labeled accuracy, pair aggregation, and target-mapping absolute accuracy were all
`1.000`. The run still failed the predeclared all-gates rule. One sampling seed
returned the wrong atomic unit namespace in both orientations—eight invented
repeat-comparison identifiers instead of the eight requested signed-occurrence
identifiers—so response success and paired coverage were `0.900`. The validator
failed closed; it did not infer an occurrence mapping from the response prose.
Mean repeat SD (`0.057`), mean orientation half-gap (`0.113`), and comparative
disagreement (`0.333`) also narrowly exceeded their frozen limits.

The frozen v4 follow-up changes representation, not the public scientific
contract. The former process `U`, which carries the insulin-dependent mechanism,
is renamed safely to `Uid` throughout the candidate AST. A new process `U` is the
same-named observed total in both candidates. The complete candidate defines
`U = Uii + Uid`; the controlled omission defines `U = Uid`; both map observed
channel `U` to process `U`. Certification requires a unique insulin-claimed
`Uid`, excludes `Uii` from `Uid`, compiles both candidates, and proves that only
the total-process expression differs. The revised v3 prompt, 120B judge, five
seeds, both orientations, fixed-denominator scoring, question consensus, and all
validation thresholds remain frozen.

V4 failed its scientific and stability gates despite using the cleaner names.
Response success was `0.950`, but paired coverage was `0.900`, target-mapping
absolute accuracy was `0.778`, paired-question-consensus accuracy was `0.667`,
and modal consistency was `0.675`. The stored evidence exposed two independent
failure modes. Four successful calls treated `U -> U` as sufficient and did not
resolve the process definition. In several other calls, the absolute unit
correctly said that `U = Uid` omitted `Uii`, yet direct comparative questions
still caused a tie or preference for the incomplete model.

`hybrid-judge-protocol-6-recursive-hard-target-contract` repairs both interfaces
without encoding a benchmark-specific decomposition. The judge receives one
general instruction: recursively resolve mapped states and processes, never
treat identity naming as proof of completeness, compare the resolved expression
with the public target definition, and cite the mapping plus the determining
definitions. The runtime adds a separate `target_contract` group with hard
enforcement. If question consensus yields pass for exactly one candidate and fail
for the other, the passing candidate wins before comparative residuals. Equal or
indeterminate target verdicts produce no hard override. The old target prompt and
soft balance-group placement remain the default for reproducibility.

Before new calls, the frozen v4 rows are rescored offline with hard enforcement
only. That measures aggregation independently of the recursive instruction. The
matched v5 call experiment then reuses the exact v4 pair and label bytes, public
prompt, model, seeds, weights, thresholds, and question consensus; only the
recursive instruction and versioned hard target contract change.

V5 produced the correct paired winner for every usable trial and improved the
target-mapping result, but failed the frozen reliability gates. The valid model
passed in all 19 successful orientations. The incomplete model failed in 17 of
19 orientations; its two false passes occurred only when the baseline was shown
first, while the swapped presentation correctly found the omission. Here an
orientation means one presentation order of the same two candidate identities:
the runtime asks A/B and B/A, then maps both answers back to those identities.
One additional B/A call failed before comparison because all nine requested
atomic occurrence units were omitted after ten provider attempts.

`hybrid-judge-protocol-7-fail-closed-target-contract` makes two versioned,
general repairs. Ordinary scientific-question disagreements remain
indeterminate. For a hard public requirement only, pass requires pass in both
orientations, while a fail in either orientation yields fail; this is recorded
as an orientation disagreement even though the contract fails closed. Separately,
the atomic provider still receives all bounded retries. If its final parsed
response contains only missing runtime-owned units and no unexpected units, the
runtime fills each omission with `insufficient_public_information`, records the
exact identifiers and repair counts, and continues. It never invents a direction
or repeated-flux relation, and any unexpected identifier remains terminal.

## Target-only completeness protocols

The V6 matched run answered every target-completeness question correctly among
complete paired trials, but one orientation failed in the unrelated atomic
stage after returning five repeat identifiers when the runtime requested none.
This showed that target completeness inherited failure modes from scientific
questions it did not require. V6 remains frozen and is not retrospectively
repaired.

`target-completeness-judge-1` tested whether a fully isolated candidate was
sufficient context. It was not: all 20 structured responses completed, every
valid target passed, but the incomplete target also passed in all ten mutated-
candidate calls. Candidate verdict accuracy was `0.500`, so this protocol is a
frozen negative result and its fresh confirmation remains correctly blocked.

`paired-target-completeness-judge-1` is the fair, isolated evaluation of the
scientific logic that worked in V6. Each call contains both blinded candidates,
all public target identifiers, canonical observation mappings, recursively
resolvable definitions, and deterministic structural facts. For each target it
returns separate absolute `pass`, `fail`, or `indeterminate` answers for A and
B. It cannot emit atomic occurrence answers, repeat relations, comparative
criteria, a winner, or a numeric score.

Each repetition is an atomic two-orientation transaction. The same candidate
pair is presented A/B and B/A at one seed, then normalized to stable baseline
and mutated identities. For each candidate/target unit, pass/pass becomes pass,
any fail becomes fail, and every other combination becomes indeterminate. If
either orientation fails structured validation after bounded provider retries,
the other orientation is discarded even if valid and both orientations are
retried at the next seed. Seed blocks are separated by the provider-attempt
limit so provider repairs and pair-level retries never reuse a sampling seed.
No missing scientific verdict is repaired.

The matched development run copies the exact V6 pairs and labels and uses the
same public prompt and model. Pair labels are opened only by the offline
analyzer. A pass authorizes a separately frozen fresh-structure confirmation;
it does not itself enable the production search gate.

The matched V8 run failed. All 20 orientation calls returned valid structured
responses, but the incomplete total `U = Uid` failed in only three of the ten
paired trials. Candidate-verdict accuracy was `0.650`, incomplete-target fail
rate was `0.300`, and neither pair passed the aggregate gate. The stored
evidence shows that the model usually treated presence of the requested delayed
insulin pathway as sufficient and ignored the separately stated fact that total
`U` includes the supplied `Uii` contribution. This is a scientific
interpretation failure, not a transport failure.

Post-freeze review also found that the older V6 candidate identifiers contained
`baseline` and `omitted` labels. V8 correctly blinded those identifiers, so the
apparent V6 success is not admissible evidence for target-completeness accuracy.
V6 and V8 remain frozen development diagnostics and are not production gates.

## Deterministic public target contract

Production target feasibility is now owned by a typed, deterministic public
contract. Every benchmark uses the same schema: each public target must have an
explicit observation mapping, and a target may list public channels whose
contribution is explicitly required by the public prompt. Evaluation recursively
traverses candidate process, state, and mapping definitions. An identity mapping
therefore proves only that a named quantity is exposed; it does not bypass the
composition check.

For the named easy Dalla-Man tasks where `U` is publicly defined as total
disposal and `Uii` is supplied as its insulin-independent contribution, the
contract requires a directed dependency path `Uii -> ... -> target:U`. Thus
`U = Uii + Uid` passes and `U = Uid` fails. The contract does not prescribe the
name or formula of the insulin-dependent component. Required-mechanism
representation, driver paths, and dynamic memory remain separate mechanism
compliance predicates, while scientific tradeoffs remain with the paired LLM
judge.

The 40 Phase-B contract instances are prompt-hash committed under
`configs/target_eval/phase_b_v1`. They are derived from public specifications,
contain no private equations or test data, and enter search as a hard feasibility
gate before fitting. Pruning may not remove a required target dependency; if a
pruned form violates the contract, search retains the valid unpruned form.

Production pairwise judge payloads now replace proposer candidate identifiers,
parent identifiers, and change summaries with neutral A/B metadata. Runtime
checkpoints retain the real lineage outside the provider payload. Frozen
calibration artifacts are not rewritten.

The earlier one-candidate fresh confirmation is frozen by
`configs/target_completeness_fresh_confirmation_v1.json`. Preparation fails
unless the exact V7 development configuration and pair bytes are certified by
a passing `target-completeness-judge-analysis-1` artifact. It then selects two
eligible proposer structures after canonicalizing away the clean-name total
wrapper and excludes every previously opened structure for the same benchmark.
This prevents `U`/`Uid` bookkeeping from making an already opened proposer
structure appear fresh. The confirmation repeats the unchanged one-candidate,
all-public-target protocol for 20 logical calls and applies the same frozen
gates. Because its V7 prerequisite failed, it is retained only for
reproducibility and is not run. V8 requires a new confirmation manifest after
the matched paired target-only development gate passes.
