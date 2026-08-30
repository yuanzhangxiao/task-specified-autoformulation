# Phase-B public prompt overlay

The deterministic public-target evaluator is bound to the exact proposer prompt
shown to a discovery method. The registered Phase-B release predates the reviewed
clarification that named easy target `U` is total glucose disposal and that
supplied `Uii` is its insulin-independent contribution. Production search must
therefore use a versioned copy-on-write overlay, not modify the registered release
and not bypass the prompt hash check.

The overlay has deliberately narrow authority:

- all 40 registered production cells must be present;
- every non-`proposer_prompt.txt` file is byte-identical to the source release;
- a prompt already committed by its target contract is copied unchanged;
- only the four reviewed named easy T2/T3 prompts may be revised;
- every revised prompt must equal the SHA-256 committed by its target contract;
- any other prompt mismatch, staging bundle, missing sealed test, symlink, hidden
  path, altered contract, or altered existing overlay fails closed.

The common judge prompt is not changed. The output manifest records the source
and overlay inventories, the target-contract bundle digest, and every changed
benchmark identifier. Running the preparation command again verifies the existing
overlay; it never overwrites a divergent artifact.

## Delta preparation and verification

Run from the checked-out repository:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
source_public=/projects/bibo/$USER/phase_b/inputs/public
overlay=/work/hdd/bibo/$USER/phase_b/inputs/public-prompt-v2
contracts="$repo/configs/target_eval/phase_b_v1"

cd "$repo"

"$python_bin" scripts/prepare_phase_b_public_prompt_overlay.py \
  --source-data-root "$source_public" \
  --output-data-root "$overlay" \
  --target-contract-root "$contracts" \
  --config configs/phase_b_public_prompt_overlay_v2.json

jq '{status, cell_count, changed_prompt_count, changed_benchmark_ids,
     non_proposer_files_byte_identical,
     target_contract_manifest_sha256}' \
  "$overlay/prompt_overlay_manifest.json"
```

The expected result is `status: ready`, `cell_count: 40`,
`non_proposer_files_byte_identical: true`, and exactly four changed named easy
T2/T3 identifiers. If a released prompt was already revised, it is valid for the
changed count to be smaller; every resulting prompt must still match its contract.

Verify the original release remained unchanged and the clarified T2 prompt is
active only in the overlay:

```bash
benchmark=phase_b_dalla_man_t2_canonical_named_easy

sha256sum \
  "$source_public/phase_b_v1/$benchmark/proposer_prompt.txt" \
  "$overlay/phase_b_v1/$benchmark/proposer_prompt.txt"

jq -r '.public_prompt_sha256' \
  "$contracts/specs/$benchmark.json"

grep -nE \
  'total glucose utilization|supplied insulin-independent|insulin-dependent contribution' \
  "$overlay/phase_b_v1/$benchmark/proposer_prompt.txt"
```

Then exercise the same data root and contract through the authoritative search
entry point without making an LLM call:

```bash
"$python_bin" scripts/run_experiment.py \
  --data-root "$overlay" \
  --benchmark-id "$benchmark" \
  --tier easy \
  --public-target-contract "$contracts/specs/$benchmark.json" \
  --dry-run
```

Every production launcher must set:

```bash
export AF_PUBLIC_DATA_ROOT=/work/hdd/bibo/$USER/phase_b/inputs/public-prompt-v2
export AF_TARGET_CONTRACT_ROOT=/projects/bibo/$USER/repos/autoformalism-v21/configs/target_eval/phase_b_v1
```

and pass
`--public-target-contract "$AF_TARGET_CONTRACT_ROOT/specs/${benchmark_id}.json"`
to `scripts/run_experiment.py`. This guarantees that proposal generation and the
deterministic feasibility gate consume the same versioned public task.

## Baseline matching

The overlay does not retroactively change completed baseline calls. Before final
cross-method comparison, rerun any GPT-5.6 raw-data-agent trials whose benchmark
identifier appears in `changed_benchmark_ids`, using this same overlay. Trials for
unchanged prompts remain matched and do not need to be repeated.
