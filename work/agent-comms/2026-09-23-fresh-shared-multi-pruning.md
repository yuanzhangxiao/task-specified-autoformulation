# Fresh Dalla integration

Implements the user's requested restart of T1-hard, T2-easy and T2-hard with the
current general shared-process constructor and final pruning, plus joint-output
fitting, compact response evidence, explicit target-contract repair, v8 immutable
parameter declarations and tolerant incumbent selection. Scientific critic off.

Six Full lineages (two seeds), fresh R0 followed by R1–14. Reuses public development
assets only; imports no historical candidates. Pruning runs once after R14, using
Astra's existing training ranking and paired unchanged/pruned comparison. Numerical
profile is the already implemented collocation-multi-target-v1. General construction
limits match shared_process_integration_v1, which makes this an integration run,
not an isolated comparison against the historical pipeline.

Implementation, budgets, ACES commands, outputs and uncertain-submission recovery:
[docs/DALLA_SHARED_MULTI_PRUNING.md](../../docs/DALLA_SHARED_MULTI_PRUNING.md).

One H100 per 20B proposer; CPU fitting/pruning. No critic or remote session launched
by this task. Official GPT-OSS-120B documentation supports single-H100 inference;
the current two-GPU critic configuration is a separate serving choice.

Verification: focused integration/regression suite 107 passed; follow-up fresh and
response-evidence checks 20 passed; the shared-process pilot file excluded by the
repository wrapper passed separately (11). `scripts/run_tests.sh full` passed:
3107 passed initially, five initial failures/errors passed serially, then all 66
timing-sensitive cases passed; eight Torch-dependent tests skipped. Some parallel
failures were frozen-source guards during the final small edit, rather than
scientific test failures. Integrated synthetic smoke passed with real joint-output
search fits and both pruning comparison arms. All changed Python files pass Ruff;
whole-tree Ruff still flags 37 pre-existing issues under untracked analysis/claude.
Shell syntax and the new server protocol dispatch check passed.

The downloaded v7 plan's public prompt hashes match current target contracts for
all three ACES cells. Loose local benchmark folders have different prompt versions
and were not substituted into the campaign. No remote jobs were submitted here.
