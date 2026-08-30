# Phase-B search integration v2

This development confirmation is the first matched search experiment using the
production public-input boundary. It supersedes the v1 search-ablation launch,
whose named T2 cell used the pre-clarification proposer prompt and did not pass a
public-target contract into search.

The v2 freeze binds each selected cell to:

- the versioned public-prompt v3 overlay;
- the exact prompt SHA-256;
- the exact deterministic public-target contract SHA-256;
- the validated paired-question-consensus GPT-OSS-120B protocol;
- the passed hidden-response-subspace contract audit; and
- a common resource-accounting policy.

Both arms retain deterministic schema, restricted-expression, symbol-closure,
algebraic-cycle, observation-mapping, public-target-dependency, parameter, and
initialization checks. A public-target-contract failure rejects a proposal before
parameter fitting. The `no_judge` arm disables only LLM scientific comparison and
selects by validation error; it does not disable deterministic execution or final
evaluation.

The judge arm is launched first so that the matched no-judge arm can reuse the
same initial proposer request. Logical token usage includes cache-restored calls,
while actual uncached provider attempts and tokens are reported separately. Each
task records process wall time, CPU time, peak resident memory, allocated CPU
core-hours, allocated GPU-hours, GPU inventory, cache use, retries, and provider
latency. Local GPT-OSS is not assigned a fictitious zero-dollar price: hardware
time is reported and monetary cost remains unpriced. Slurm submission and start
timestamps are saved separately so queue time is never mixed with execution.

No test or private data is opened during search. All 12 planned sources,
including missing or terminal outcomes, are frozen before the common post-freeze
evaluation computes sealed target NMSE, deterministic mechanism compliance,
hidden response-subspace NMSE, intervention behavior, and model complexity. No
weighted overall evaluation score is defined.

## Delta launch

Prepare the v3 prompt overlay first, then run:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python

cd "$repo"
git pull --ff-only origin main

export AF_REPO_ROOT="$repo"
export AF_PYTHON="$python_bin"
export AF_PUBLIC_DATA_ROOT=/work/hdd/bibo/$USER/phase_b/inputs/public-prompt-v3
export AF_TARGET_CONTRACT_ROOT="$repo/configs/target_eval/phase_b_v1"
export AF_PROMPT_OVERLAY_CONFIG="$repo/configs/phase_b_public_prompt_overlay_v3.json"
export AF_OUTPUT_ROOT=/work/hdd/bibo/$USER/phase_b/search-integration-ablation-v2

scripts/hpc/submit_phase_b_search_integration_ablation_v2.sh
```

The submission command fails if the output root already has a submission
manifest. Resume or diagnose that frozen launch instead of silently creating a
second matrix.

If full request hashes differ, distinguish ephemeral transport metadata from
the actual initial candidate before declaring the matched trial invalid:

```bash
python scripts/check_phase_b_search_initial_comparability.py \
  --plan /path/to/frozen/plan.json \
  --search-root /path/to/search-integration-ablation-v2 \
  --output /path/to/initial_comparability.json
```

Equality is evaluated within each benchmark, tier, and repetition. Different
seeds are neither expected nor required to produce the same candidate. The
report separately records the full transport-sensitive request hash, exact
parsed proposer response, identity-insensitive candidate content, and the
post-validation round-zero structural hash.
