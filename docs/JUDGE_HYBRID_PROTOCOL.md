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
