# Phase-B final evaluation contract

## Purpose

The LLM judge is part of Autoformalism: it supplies public-information
scientific feedback and participates in the frozen search/selection protocol.
It is not, however, the source of hidden benchmark truth. Final evaluation
therefore reports separate deterministic, behavioral, private-mechanism, and
qualitative endpoints rather than collapsing them into one weighted score.

The same final evaluator applies to Autoformalism, raw-data frontier agents,
SINDy, PySR, D3, and ablations after each method has frozen its model without
test or private-reference access.

## Endpoint vector

1. **Runtime validity:** deterministic schema, expression, symbol, target,
   initialization, causal-availability, and static expression-domain checks.
   Numerical integration failures are retained separately with the behavioral
   endpoint because they require fitted parameters and trajectories.
2. **Public mechanism compliance:** the fraction of task-required mechanisms
   for which every applicable public graph predicate is certified.
3. **Target behavior:** held-out free-rollout target NMSE and per-target NMSE.
4. **Mechanism recovery:** fraction of private reference mechanisms represented
   by a compatible declared candidate component.
5. **Conditional hidden error:** linearly aligned response-subspace test NMSE
   only for recovered mechanisms, with the alignment fitted on training data.
6. **Intervention behavior:** private intervention target NMSE, response
   direction, response-shape correlation, and peak-timing error.
7. **Complexity and reliability:** states, latent states, processes, parameters,
   additive terms, source completion, replay-complete fitted-parameter coverage,
   and failure counts.
8. **Qualitative LLM assessment:** optional protocol, requested/successful call
   counts, and absolute verdict counts. This remains separate from deterministic
   correctness and is not interpreted as a gold-label accuracy metric on
   natural candidates.

No overall weighted score is defined.

Runtime validity is recomputed by the evaluator from the frozen candidate and a
serialized public `ValidationContext`; a method cannot supply or override its
own validity flag. The same restricted parser and candidate validator used by
the runtime check target mappings, equation closure, declared symbols, causal
channel availability, initial conditions, constraints, and expression domains.
Invalid candidates retain stable diagnostics, cannot carry available private
scores, and do not proceed to public-mechanism scoring.

Before any private endpoint is opened, every method is converted to a common
content-addressed `FrozenEvaluationSubject`. It records the exact source
artifact hash, canonical candidate hash, adapter, fitted global parameters,
and any fitted global initial conditions. Parameterization status is reported
as `available`, `partial`, `not_required`, or `missing`; a structure is never
silently called replayable when fitted scalars are absent. Older Autoformalism
summaries that omitted optimized global initial conditions are therefore
retained as `partial`, while new summaries serialize those values.

## Conjunctive public mechanism compliance

For public mechanism requirement \(m\), the evaluator instantiates only the
applicable predicates:

\[
C_m = D_m R_m P_m M_m S_m,
\]

where \(D_m\) is an explicit declared component, \(R_m\) is required-driver
ancestry, \(P_m\) is a directed path to the requested target, \(M_m\) is dynamic
memory when required, and \(S_m\) is a certified sign/role predicate when the
public specification defines one. The product is implemented as a conjunction,
not floating-point multiplication. One failed predicate fails that mechanism;
an uncertified required predicate makes the mechanism ambiguous and therefore
not certified compliant.

The model-level metric is

\[
C_{\mathrm{task}} = \frac{1}{|\mathcal M|}
\sum_{m\in\mathcal M} \mathbf{1}[C_m\text{ is certified satisfied}].
\]

This grouping avoids an all-model zero: failure of one requirement does not
erase other fully compliant mechanisms. Historical `mechanism_coverage` and
atomic `structural_validity` remain available for exact reproduction, but the
new `mechanism_compliance` field is the prospective endpoint.

Every public mechanism specification must be frozen independently of candidate
outputs. It may encode only information stated in the public task. It must not
require simulator state names, exact equations, or private edges.

The Phase-B v1 specifications are generated from the Section-A task bullets in
the frozen typed public contract. There is exactly one small conjunctive group
per public task bullet: 24 cells have one group, eight have two, and eight have
three. Each group stores its verbatim public source text, only public channel
identifiers, and the SHA-256 of the complete proposer prompt. The generated
bundle contains 40 specifications under
`configs/mechanism_eval/phase_b_v1/specs` and a manifest that commits to every
file. Named cells may use named scientific aliases stated in their prompts;
obfuscated and opaque cells use only their public generic vocabulary.

These contracts deliberately do not turn every noun in a compound task bullet
into a separate hidden-truth requirement. For example, the T4 flux portrait and
the controlled-balance task remain one public requirement each. The evaluator
can certify explicit declaration, public-driver ancestry, target reachability,
and expressly required dynamic memory. It does not claim to deterministically
distinguish two scientifically different nonlinear functions with the same
public dependency graph; hidden-mechanism and intervention endpoints evaluate
those differences after model freeze.

## Private ground-truth boundary

Target test trajectories, hidden reference trajectories, private mechanism
identities, and interventions may be opened only after the selected model is
frozen. The typed `FrozenEvaluationSubject` rejects available private endpoints
unless `private_metrics_opened_after_freeze` is true.

The common post-freeze target protocol is **unseen-condition free rollout**.
For each benchmark and tier, target standard deviations are fitted once from
the public training split. The frozen equations, global parameters, and fitted
global initial conditions are then replayed on every sealed test trajectory
with no parameter fitting, no trajectory-specific initial-state fitting, and
no measured-target resets after the initial sample. Aggregate target NMSE is
reported only when every requested trajectory integrates successfully. A
partial rollout remains a failure, while the successful and failed trajectory
counts remain visible. Each trajectory also has the same prespecified numerical
wall-time limit (300 seconds by default); exceeding it is a replay failure, not
a penalty-imputed score.

Each sealed Phase-B test trajectory also defines one intervention/distribution-
shift case because it uses an unseen forcing schedule. The evaluator reports
all-target NMSE for that case and direction, centered-shape correlation, and
peak-timing error for the public primary target. These per-case values remain
separate from aggregate target NMSE.

Hidden NMSE is conditional on structural recovery. Models without a compatible
declared mechanism are recovery failures and do not receive a favorable hidden
trajectory error. Alignment parameters are fitted using training trajectories
only and evaluated once on test trajectories.

### Private mechanism-response subspace

Phase-B hidden recovery is evaluated in response space rather than by matching
simulator state names. The private contract reuses the mechanism groups,
claimed dimensions, and `1e-3` fractional perturbation frozen by the
pre-release identifiability gates. For every private mechanism group, the
trusted simulator supplies one normalized local target-response direction. For
every explicitly tagged candidate state or process matched by the public
mechanism contract, the evaluator supplies one candidate direction by
multiplying that process expression—or the tagged state's complete derivative
equation—by `1 + 1e-3`. The frozen candidate parameters and initial conditions
are unchanged.

The same frozen relative singular-value tolerance, `1e-3`, is used for the
pre-release reference-rank audit, candidate recovery rank, and least-squares
pseudoinverse truncation. A numerically negligible direction therefore cannot
pass candidate recovery under a looser scoring threshold than the one used to
certify the benchmark.

Let `R_train` and `R_test` contain the private response directions and let
`C_train` and `C_test` contain the candidate directions. The right singular
vectors of `R_train` define the predeclared `r`-dimensional private mechanism
subspace. The evaluator fits only

\[
B^* = \arg\min_B \|C_{train} B - R_{train}V_r\|_F^2
\]

and reports on sealed test interventions

\[
\frac{\|C_{test}B^* - R_{test}V_r\|_F^2}
     {\|R_{test}V_r\|_F^2 + \epsilon}.
\]

This is invariant to candidate mechanism naming, permutation, sign, scale, and
invertible linear mixing. It is not claimed to identify nonlinear coordinate
transformations. A candidate is structurally recovered only when all public
mechanism conjunctions pass and the candidate response matrix has rank at
least `r`; otherwise hidden NMSE is withheld. Dalla Man T4 remains explicitly
not applicable because its frozen contract uses flux compatibility and
intervention behavior rather than hidden-coordinate recovery.

## Implementation

- `src/autoformalism/rebuttal/mechanisms.py` computes legacy atomic metrics and
  prospective conjunctive compliance.
- `src/autoformalism/rebuttal/final_evaluation.py` defines the frozen subject,
  recomputed runtime certification, separate endpoints, privacy guards,
  deterministic complexity, and summary.
- `src/autoformalism/rebuttal/final_evaluation_adapters.py` converts frozen
  Autoformalism summaries, GPT raw-data-agent runs, SINDy, PySR, and D3 results
  to that shared contract.
- `scripts/export_phase_b_frozen_subjects.py` executes a prespecified JSONL
  adapter manifest and writes both successful subjects and an all-request
  outcome ledger. It loads train/validation only to reconstruct the public
  validation context.
- `scripts/evaluate_candidate_deterministically.py` evaluates one method-neutral
  candidate and public validation context without opening test or private data.
- `scripts/evaluate_phase_b_postfreeze.py` writes a content-addressed freeze
  receipt before test access, then performs frozen-parameter free rollout with
  per-subject checkpoints and deterministic resume. It supports independent
  shards.
- `scripts/merge_phase_b_postfreeze.py` verifies the frozen-input and shard
  hashes, complete shard coverage, outcome identities, and immutable public
  subject fields before restoring the original subject order.
- `src/autoformalism/rebuttal/phase_b_hidden_subspace.py` defines the private
  response-subspace contract, ground-truth and candidate direction generation,
  structural/rank recovery gate, and training-only alignment.
- `scripts/evaluate_phase_b_hidden_subspace.py` performs checkpointed,
  shardable private evaluation after common target replay.
- `scripts/merge_phase_b_hidden_subspace.py` verifies and merges those shards
  without allowing any public or fitted field to change.
- `scripts/audit_phase_b_hidden_subspace_contracts.py` checks the private
  target-response rank and named/obfuscated identity of selected frozen cells
  before any method outputs are scored.
- `scripts/assemble_phase_b_final_evaluation.py` validates a method-independent
  JSONL subject manifest, joins frozen public mechanism specifications, and
  writes JSONL, CSV, JSON, Markdown, and provenance outputs.
- `scripts/evaluate_hidden_mechanisms.py` computes train-aligned/test-scored
  hidden mechanism NMSE.
- `scripts/evaluate_intervention_suite.py` and the intervention modules compute
  private distribution-shift endpoints.

Example assembly:

```bash
python scripts/assemble_phase_b_final_evaluation.py \
  --subjects /path/to/frozen_evaluation_subjects.jsonl \
  --mechanism-config-root configs/mechanism_eval/phase_b_v1/specs \
  --output-root /path/to/final_evaluation
```

Before assembly, a method-neutral source request file can be adapted with:

```bash
python scripts/export_phase_b_frozen_subjects.py \
  --requests /path/to/source_adapter_requests.jsonl \
  --data-root /path/to/public/phase_b \
  --output-root /path/to/frozen_subjects
```

Each JSONL request has `request_id`, `source_kind`, and `source_path`. The
accepted source kinds are `autoformalism`, `raw_data_agent`, `sindy`, `pysr`,
and `d3`. Autoformalism paths point to `summary.json`; raw-agent paths point to
the run directory; the remaining paths point to `result.json`. Existing test
scores embedded in legacy source artifacts are deliberately ignored. The
common post-freeze evaluator is the only component allowed to populate private
target, hidden-mechanism, or intervention endpoints.

### Two-cell end-to-end development pilot

`configs/phase_b_final_evaluation_pilot_v1.json` freezes a small integration
matrix before its Autoformalism searches: the named Dalla Man T2 easy cell and
the opaque alien-system hard cell, three repetitions each, for Autoformalism and
the primary fitted GPT-5.6 raw-data-agent baseline. The GPT artifacts are reused
without a new provider call. Autoformalism uses its frozen
`incumbent_relative_hybrid` search and exactly returned final parameters.

The pilot is explicitly development-only. It verifies the plumbing and failure
accounting before expanding to 40 cells; it is not a confirmatory benchmark
claim. `scripts/prepare_phase_b_final_evaluation_pilot.py` resolves the exact 12
source paths, verifies the public identity of every available run, hashes every
available relevant source file, and verifies the exact successful
hidden-contract audit v2 plus its companion digest. The request ledger always
contains the whole planned cross-product. A terminal method run without a
summary remains a content-addressed missing-source outcome rather than being
retried until success or silently omitted. This freeze occurs before test replay
or candidate-specific private evaluation.

`scripts/hpc/phase_b_final_evaluation_pilot_v1.slurm` then runs the existing
checkpointed stages in order: source adaptation, sealed unseen-condition target
replay, intervention scoring, private response-subspace evaluation, and final
assembly. No parameter is refitted and no LLM judge is called during final
evaluation. `scripts/summarize_phase_b_final_evaluation_pilot.py` reports source
completion, runtime validity, public mechanism compliance, target NMSE, hidden
recovery and conditional hidden NMSE, intervention behavior, and complexity as
separate endpoints. Unavailable endpoints remain missing with explicit coverage;
there is no penalty imputation or weighted overall score.

For a single process, run the common test replay after source adaptation:

```bash
python scripts/evaluate_phase_b_postfreeze.py \
  --subjects /path/to/frozen_subjects/frozen_evaluation_subjects.jsonl \
  --public-data-root /path/to/public/phase_b \
  --output-root /path/to/postfreeze
```

For an HPC array, pass the same frozen subject file and output root to every
task with `--shard-index "$SLURM_ARRAY_TASK_ID" --shard-count N`, then merge:

```bash
python scripts/merge_phase_b_postfreeze.py \
  --subjects /path/to/frozen_subjects/frozen_evaluation_subjects.jsonl \
  --input-root /path/to/postfreeze \
  --output-root /path/to/postfreeze
```

The merge output `postfreeze_subjects.jsonl` is the input to final assembly.
To add the conditional private hidden endpoint first, run:

```bash
python scripts/evaluate_phase_b_hidden_subspace.py \
  --subjects /path/to/postfreeze/postfreeze_subjects.jsonl \
  --public-data-root /path/to/public/phase_b \
  --private-data-root data_raw \
  --mechanism-config-root configs/mechanism_eval/phase_b_v1/specs \
  --output-root /path/to/hidden
```

Use `--shard-index` and `--shard-count` analogously, then merge with
`scripts/merge_phase_b_hidden_subspace.py`. The merged
`hidden_subjects.jsonl` becomes the final-assembly input.

Before the first production run, audit representative frozen contracts. The
audit must pass the claimed target-response rank and exact semantic-pair
identity checks; a failure stops the hidden evaluation rather than changing a
rank threshold after seeing method outputs:

```bash
python scripts/audit_phase_b_hidden_subspace_contracts.py \
  --benchmark-id phase_b_dalla_man_t1_canonical_named_easy \
  --benchmark-id phase_b_anonymous_system_t1_canonical_obfuscated_easy \
  --benchmark-id phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy \
  --benchmark-id phase_b_anonymous_system_task_canonical_obfuscated_easy \
  --public-data-root /path/to/public/phase_b \
  --private-data-root data_raw \
  --output /path/to/hidden_contract_audit.json
```

The audit separates provenance from numerical sensitivity. CSTR and Alien
private system files and protocol suites must match committed SHA-256 values.
Public split manifests and prompt hashes independently identify the released
inputs. Dalla--Man and CSTR additionally compare fresh nominal rollouts to the
public release within a frozen cross-solver tolerance. Alien does not use that
last numerical comparison because long-horizon adaptive integration is not
portable across SciPy versions; its reference directions subtract nominal and
perturbed rollouts generated together in the same environment. This exception
does not relax any rank, recovery, or hidden-NMSE threshold.

The audit command writes a standard companion `<audit>.sha256` file after every
complete run. A passing digest is retained with the final experiment manifest.
Because the contract and outcome schemas are versioned, changing the rank
tolerance invalidates older cached hidden outcomes instead of silently mixing
scoring rules.

The committed public specifications can be regenerated and audited without
opening test trajectories or using private reference mechanisms in the emitted
contract:

```bash
python scripts/build_phase_b_public_mechanism_specs.py \
  --suite configs/benchmarks/phase_b_suite_v1.json \
  --data-root data_raw \
  --output-root configs/mechanism_eval/phase_b_v1
```

Structure-only evaluation of one already-frozen candidate requires no test data:

```bash
python scripts/evaluate_candidate_deterministically.py \
  --candidate /path/to/candidate.json \
  --validation-context /path/to/public_validation_context.json \
  --mechanism-spec /path/to/frozen_public_mechanism_spec.json \
  --subject-id method_benchmark_rep0 \
  --method method \
  --benchmark-id benchmark \
  --tier hard \
  --repetition 0 \
  --output /path/to/deterministic_record.json
```

Omitting `--mechanism-spec` reports that endpoint as `missing`; it never treats
an absent contract as a zero or as not applicable.

The command fails when a subject says public mechanism evaluation is applicable
but the benchmark/tier has no frozen specification. During specification
development only, `--allow-missing-mechanism-specs` retains those endpoints as
explicitly `missing` rather than scoring them as zero or not applicable.

The prospective mechanism conjunction records separate declared-component,
driver-ancestry, per-target path, dynamic-memory, and sign predicates. Historical
`mechanism_coverage` and `structural_validity` retain their previous definitions.
Every required target must have its own certified path; reaching one of several
listed targets is insufficient.

## Remaining implementation milestones

1. Run the frozen two-cell, three-repetition paired-question-consensus versus
   no-judge search integration ablation. Freeze all 12 planned source outcomes,
   including failures, before sending both arms through the common evaluator.
   This experiment attributes the effect of judge-guided selection and feedback;
   it does not tune the judge or define an overall score.
2. Analyze the candidate-specific V7 target-completeness run and perform the
   predeclared fresh-structure confirmation before enabling the hard target
   contract in production search.
3. Run the private mechanism-response implementation on Delta and audit the
   prespecified ground-truth rank, semantic-pair identity, recovery coverage,
   numerical failures, and runtime before enabling it for the full comparison.
   The legacy coordinate-matching scripts remain excluded.
4. Run every method through the common source adapter, post-freeze replay,
   hidden-subspace evaluator, and final assembler.
5. Use the LLM judge only as required by Autoformalism search and a small,
   predeclared qualitative comparison; report its call budget and coverage
   separately.

The integration ablation is frozen by
`configs/phase_b_search_integration_ablation_v1.json`. Its preparation command
writes an immutable 12-task ledger before any search call. The GPU array reads
only that ledger. The judge tasks run first and populate the shared
content-addressed proposer cache; the dependent no-judge tasks therefore reuse
identical initial requests before their feedback histories diverge. After
search, a second freeze records every available or
missing summary with a distinct `autoformalism:paired_question_consensus` or
`autoformalism:no_judge` method label and verifies the exact successful hidden
contract audit before the CPU evaluator can open test or private data.
