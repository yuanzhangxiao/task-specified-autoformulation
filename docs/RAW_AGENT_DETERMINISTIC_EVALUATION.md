# GPT-5.6 raw-agent deterministic evaluation

This evaluation composes the 90 unchanged-prompt fitted-model runs from
`raw-data-agent-fitted-v1` with the 30 public-prompt-v3 refresh runs from
`raw-data-agent-fitted-prompt-v3-refresh-v1`. The composition is frozen by
content hash before any test or private reference is opened.

The evaluation uses the same method-neutral deterministic pipeline as
Autoformalism:

1. adapt each provider-fitted candidate without parameter refitting;
2. validate runtime and public mechanism compliance;
3. replay the frozen model on sealed test trajectories;
4. evaluate the claimed hidden response subspace against private references;
5. report intervention behavior and model complexity;
6. report provider latency, tool calls, and tokens from frozen artifacts.

Endpoints remain separate. No weighted overall score is defined, and no LLM is
called. Monetary cost is reported as unavailable because it was not embedded
in the provider artifacts; the pipeline does not reconstruct historical cost
from a mutable price table.

The Delta submission chain uses one preparation job, 24 post-freeze CPU
shards, one merge job, 24 hidden-evaluation CPU shards, and one final merge and
report job. Per-subject failures are retained as outcomes.

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
cd "$repo"
git pull --ff-only origin main

export AF_REPO_ROOT="$repo"
export AF_PYTHON=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
export AF_PUBLIC_DATA_ROOT=/work/hdd/bibo/$USER/phase_b/inputs/public-prompt-v3
export AF_PRIVATE_DATA_ROOT="$repo/data_raw"
export AF_HISTORICAL_RAW_ROOT=/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-v1
export AF_REFRESH_RAW_ROOT=/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-prompt-v3-refresh-v1
export AF_HIDDEN_AUDIT=/work/hdd/bibo/$USER/phase_b/hidden-contract-audit-v2/hidden_contract_audit.json
export AF_OUTPUT_ROOT=/work/hdd/bibo/$USER/phase_b/raw-agent-deterministic-evaluation-v1

scripts/hpc/submit_phase_b_raw_agent_deterministic_evaluation.sh
```

The preparation job must report 120 requested sources. Missing or malformed
source artifacts remain explicit source-completion failures instead of being
silently dropped.
