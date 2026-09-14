# Fitting-free pre-fitting experiments

This follow-up isolates construction-interface behavior while the fitter is being
finalized. It does not rerun numerical fitting or reinterpret optimizer failures
as structural defects. Benchmark prompts and tables are unchanged.

## Milestone 1: historical response replay

`scripts/prefit_response_replay.py` reads a terminal
`prefit-matched-construction-1` root and writes a sealed replay corpus outside it.
The source plan, construction results, initializer checkpoints and referenced
cached requests are checked. File hashes bind the replay to the source; missing
requests and changed artifacts fail explicitly. No CSV or fit result is opened.
Only final visible provider JSON becomes a proposed expression.

The replay rechecks interaction-function slots and causal initializers using the
production validators. Functions retain historically accepted preceding slots;
initializers retain the actual historical equations and accepted boundaries.
Every saved attempt is independent. A newly admissible response never replaces
the historical parent of a later response. The assignment-normalization view is
compared with a strict-RHS view under the same current validators. This is not an
exact re-execution of every old runtime version.

A schema-valid equation batch was historically accepted before its individual
slots were checked. Its replay rows explicitly identify that different scope;
batch acceptance must not be compared with full slot validity as a regression.
Topology requests, malformed batch shapes and missing final JSON remain in an
explicit unreplayed-request inventory. They are not counted as successful replay
or silently selected as live repair cases. This first pilot does not evaluate
topology repair or scientific correctness.

Historical construction completion, normalization recoveries and remaining
deterministic violations have separate counts. Scientific status is
`not_assessed`. Replay does not establish saved live calls, avoided future
failures, scientific improvement, or a revised historical construction rate.

From the project checkout with its Python environment:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_prefit_replay.py tests/test_causal_initialization_construction.py tests/test_staged_function_runner.py
PYTHONPATH=src python scripts/smoke_prefit_replay.py
PYTHONPATH=src python scripts/prefit_response_replay.py --source /scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1-e7ffd12 --output /scratch/user/u.yx126462/phase_b/prefit-response-replay-v1.json
```

The smoke creates temporary synthetic public observations and prescribed cached
responses, then exercises the actual constructor and replay. It performs no fit
or live LLM call and does not modify benchmark assets.

## Milestone 2: matched local feedback

The next runner will freeze remaining invalid slots and valid controls from the
replay, with identical source contexts, normalization, response schemas, model
settings, seeds and budgets in both arms. The treatment is the feedback payload:
ordinary error text versus named diagnostics with explicit scope and syntax facts.
Routing, scientific judging and new full-model construction are later milestones.
