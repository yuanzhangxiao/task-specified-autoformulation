# Frozen LLM assets and candidate pool

This manifest records the first post-rebuttal, zero-new-API-call asset freeze.
It was generated on 2026-08-04 from the local `artifacts/` tree. Generated
manifests and cache copies live under `artifacts/rebuttal/frozen_assets_v1/` and
are intentionally excluded from Git.

## LLM cache resolution

- Cache files inspected: 107
- Valid cache files: 107
- Unique request hashes: 77
- Duplicate copies: 30
- Malformed files: 0
- Metadata-only conflicts: 17
- Semantic conflicts: 1
- Safely resolved requests: 76
- Recorded usage across unique requests: 225,499 input tokens, 84,927 output
  tokens, and 310,426 total tokens

For the 17 metadata-only conflicts, all copies have an identical validated
`parsed_response`; only the raw provider response differs. The resolver selects
the lexicographically first source path, records that source, and writes one
canonical cache entry.

The semantic conflict is request hash
`96ea27f26b52199576edb095c206e25bd3250a77fcfd5de032137af641726672`.
Its five copies contain two different validated Qwen 3 8B proposals: one copy
uses states `Gp`, `Gt`, `EGP`, `Uii`, `E`, and `meal_effect`; four copies use
rate-named states such as `Gp_rate`. It is excluded from the resolved replay
cache. Neither variant is silently preferred. Source-specific frozen
checkpoints remain available for post hoc analysis.

The resolved replay cache therefore contains 76 entries. Experiment and
LLM-based baseline CLIs support `--llm-cache-only` and `--llm-cache-root`.
Cache-only clients do not initialize hosted-provider SDK clients and raise a
dedicated `LLMCacheMissError` before transport on a missing hash.

## Frozen candidate pool

- Valid completed round candidates: 1,036
- Candidates with judge evidence: 653
- Unique structural hashes: 997
- Covered benchmark/tier/seed cells: 90 of 90
- Candidate-pool SHA-256:
  `cecc3b75e82d4edd42a8aaa1a8b7f1d47358c68a34a3956199af2ca2bc7eca6b`

Coverage is complete for all six benchmarks, all three tiers, and seeds 0--4.
The number of valid candidates per cell ranges from 1 to 62. Coverage here means
that at least one valid completed round checkpoint is available; it does not
assert that every original run completed successfully or that all cells have
equal candidate budgets. Those differences are preserved in
`candidate_pool_completeness.csv` and must be considered in pooled analyses.

## Generated files

- `artifacts/rebuttal/frozen_assets_v1/cache_audit/llm_cache_audit.json`
- `artifacts/rebuttal/frozen_assets_v1/cache_audit/llm_cache_manifest.csv`
- `artifacts/rebuttal/frozen_assets_v1/cache_audit/llm_cache_resolution.json`
- `artifacts/rebuttal/frozen_assets_v1/resolved_cache/*.json`
- `artifacts/rebuttal/frozen_assets_v1/candidate_pool/candidate_pool.jsonl`
- `artifacts/rebuttal/frozen_assets_v1/candidate_pool/run_inventory.csv`
- `artifacts/rebuttal/frozen_assets_v1/candidate_pool/candidate_pool_completeness.csv`
- `artifacts/rebuttal/frozen_assets_v1/candidate_pool/artifact_validation_report.json`

## Reproduction commands

```bash
python scripts/audit_llm_assets.py artifacts \
  --output-root artifacts/rebuttal/frozen_assets_v1/cache_audit \
  --resolved-cache-root artifacts/rebuttal/frozen_assets_v1/resolved_cache

python scripts/index_experiment_artifacts.py artifacts \
  --output-root artifacts/rebuttal/frozen_assets_v1/candidate_pool
```

The cache audit exits nonzero for malformed entries or semantic conflicts. When
`--resolved-cache-root` is supplied, it still writes a safe cache containing
only unambiguous requests and records every exclusion.

For a cache-only experiment replay, add:

```bash
  --llm-cache-only \
  --llm-cache-root artifacts/rebuttal/frozen_assets_v1/resolved_cache
```

The proposer/judge provider names, model identifiers, output-token limit,
prompts, response schema, and target-channel contract must match the original
request because all are part of the content-addressed request hash. A mismatch
fails closed and does not fall back to a live provider.
