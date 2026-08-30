# GPT-5.6 raw-agent public-prompt v3 refresh

The original fitted raw-agent matrix contains 40 Phase-B cells with three
repetitions. Public-prompt v3 changed exactly ten named Dalla Man cells. This
refresh reruns only those ten cells and retains the original fitted-model
contract and agent budget, yielding 30 new calls.

The refresh has a separate output root. It never overwrites or silently reuses
the pre-v3 calls. Before submission, it verifies the prompt-overlay manifest,
all ten revised prompt hashes, and the train and validation files. Test files
are not read. A later composition manifest may reuse the other 90 historical
runs only after verifying their prompt hashes against public-prompt v3.

Run on Delta:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python

cd "$repo"
git pull --ff-only origin main

export AF_REPO_ROOT="$repo"
export AF_PYTHON="$python_bin"
export AF_PUBLIC_DATA_ROOT=/work/hdd/bibo/$USER/phase_b/inputs/public-prompt-v3
export AF_OUTPUT_ROOT=/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-prompt-v3-refresh-v1

scripts/hpc/submit_phase_b_raw_data_agent_prompt_v3_refresh.sh
```

If `OPENAI_API_KEY` is absent, the submission script requests it without
echoing it. The key is exported to Slurm but is never written to a repository
file, frozen manifest, or experiment output.
