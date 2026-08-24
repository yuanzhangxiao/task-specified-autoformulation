# ACCESS and NAIRR operational setup

## Resource roles

Use the allocations as complementary systems:

| Resource | Primary role | Avoid using it for |
|---|---|---|
| NCSA Delta CPU | fitting arrays, replay, baselines, metrics | persistent services |
| NCSA Delta GPU | open-weight proposer/judge inference | CPU-only fitting |
| TAMU ACES | independent replication, overflow CPU jobs, later accelerator studies | long work on login nodes |
| Jetstream2 CPU | short-lived development/orchestration VM and transfer staging | always-on batch capacity |
| Jetstream2 storage | durable shared project staging | the only copy of results |

The award is for **Delta**, not DeltaAI. Use Delta hostnames, documentation,
partitions, and accounts unless ACCESS later explicitly adds DeltaAI.

## Freeze before production

Remote production must use one committed immutable tag. Do not deploy a dirty
checkout. The public Phase-B bundle and portable candidate pool are separate
versioned inputs with SHA-256 manifests. Never transfer `.autoformalism-secrets`
for replay or other no-LLM jobs.

Prepare locally after the code is committed and tagged:

```bash
cd /Users/yuanzhangxiao/Projects/autoformalism
python scripts/release_phase_b_public_suite.py \
  --private-data-root /PATH/TO/PRIVATE_REFERENCES \
  --public-data-root /tmp/phase-b-public-release
python scripts/export_portable_candidate_pool.py \
  artifacts/rebuttal/frozen_assets_v1/candidate_pool/candidate_pool.jsonl \
  /tmp/phase-b-portable-candidate-pool.jsonl \
  --report /tmp/phase-b-portable-candidate-pool.report.json
tar -C /tmp -czf phase-b-replay-inputs.tar.gz \
  phase-b-public-release \
  phase-b-portable-candidate-pool.jsonl \
  phase-b-portable-candidate-pool.report.json
shasum -a 256 phase-b-replay-inputs.tar.gz
```

The portable export preserves the frozen 1,036-candidate membership while
embedding warm-start parameters; remote jobs do not depend on Mac paths.

## Delta: first login and identity check

The easiest first login is Delta Open OnDemand. Direct SSH uses the NCSA
username listed in the ACCESS profile, the NCSA password, and Duo MFA:

```bash
ssh -o PreferredAuthentications=keyboard-interactive,password \
  NCSA_USERNAME@login.delta.ncsa.illinois.edu
```

On Delta, run only these read-only checks first:

```bash
hostname
id
accounts
module reset
module load gcc python
python --version
git --version
echo "WORK=$WORK"
echo "SCRATCH=$SCRATCH"
```

Record the exact CPU and GPU project names printed by `accounts`; they are
usually distinct. Use the CPU account for replay and the GPU account only for
GPU jobs.

### Delta checkout and environment

Use project/work storage rather than filling `$HOME`:

```bash
module reset
module load gcc python
mkdir -p "$WORK/repos" "$WORK/venvs" "$WORK/phase_b" "$WORK/logs"
git clone git@github.com:yuanzhangxiao/task-specified-autoformulation.git \
  "$WORK/repos/autoformalism"
cd "$WORK/repos/autoformalism"
git checkout FROZEN_TAG
python -m venv "$WORK/venvs/autoformalism-FROZEN_TAG"
source "$WORK/venvs/autoformalism-FROZEN_TAG/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
```

If GitHub SSH is not configured, use the HTTPS repository URL. Do not enter a
GitHub password into a job script.

Transfer the small replay bundle with `scp` or the Open OnDemand file manager;
use Globus for larger future datasets. Extract it under `$WORK/phase_b/inputs`.

### Delta interactive pilot

Replace `CPU_ACCOUNT` with the CPU project returned by `accounts`:

```bash
srun --account=CPU_ACCOUNT --partition=cpu-interactive \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8g \
  --time=00:20:00 --pty bash
```

Inside the allocation, activate the environment and run one two-candidate
smoke test. Verify that it writes a manifest with `uses_llm_calls=false` and
`uses_test_data=false` before submitting the array.

### Delta replay array

On the login node:

```bash
cd "$WORK/repos/autoformalism"
mkdir -p logs "$WORK/phase_b/replay-v1"
export AF_REPO_ROOT="$WORK/repos/autoformalism"
export AF_PYTHON="$WORK/venvs/autoformalism-FROZEN_TAG/bin/python"
export AF_PUBLIC_DATA_ROOT="$WORK/phase_b/inputs/phase-b-public-release"
export AF_CANDIDATE_POOL="$WORK/phase_b/inputs/phase-b-portable-candidate-pool.jsonl"
export AF_OUTPUT_ROOT="$WORK/phase_b/replay-v1"
sbatch --account=CPU_ACCOUNT --partition=cpu \
  --export=ALL scripts/hpc/phase_b_replay_array.slurm
```

Monitor without repeatedly reading every result file:

```bash
squeue -u "$USER"
sacct -X -S today --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
find "$AF_OUTPUT_ROOT" -name replay_manifest.json -type f | wc -l
```

For low wall-clock latency, use the sharded pilot workflow. It screens ten
deterministically assigned candidate shards per cell, then starts one merge and
refinement task per cell only after every screening task succeeds:

```bash
export AF_OUTPUT_ROOT="$WORK/phase_b/replay-sharded-v1"
screen_job=$(sbatch --parsable \
  --account=CPU_ACCOUNT --partition=cpu --export=ALL \
  scripts/hpc/phase_b_replay_sharded_array.slurm)
merge_job=$(sbatch --parsable --dependency="afterok:${screen_job}" \
  --account=CPU_ACCOUNT --partition=cpu --export=ALL \
  scripts/hpc/phase_b_replay_merge_array.slurm)
printf 'screen_job=%s merge_job=%s\n' "$screen_job" "$merge_job"
```

The screening array contains 40 single-core tasks: four pilot cells times ten
shards. The merge array contains four single-core tasks. A merge fails closed
if a shard is absent, incomplete, duplicated, or covers an unexpected artifact.
The merged result remains development-only and records no LLM or test-data use.

```bash
squeue -j "$screen_job,$merge_job"
sacct -X -j "$screen_job,$merge_job" \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
find "$AF_OUTPUT_ROOT/final" -name replay_manifest.json -type f | sort
```

Validate that the four sharded selections match the corresponding unsharded
pilot before expanding the cell map to the complete 40-cell suite.

### Delta local-proposer pilot

First create the development-only compatibility and call-budget manifest on a
CPU login node. This command makes no LLM calls and does not open test data:

```bash
python scripts/plan_phase_b_candidate_generation.py \
  --public-data-root "$AF_PUBLIC_DATA_ROOT" \
  --candidate-pool "$AF_CANDIDATE_POOL" \
  --output "$AF_PROJECT/phase_b/inputs/candidate-generation-plan-v1.json"
```

The frozen v1 audit has 4 cells with exact reusable structures and 36 cells
that require generation. Its first-pass budget is 8 local proposer calls per
cell (320 total), no generation-stage judge calls, and at most 4 conditional
hosted rescue calls per cell. The hosted cap is not a reservation or a required
number of calls.

Delta supports Apptainer and NVIDIA GPU passthrough with `--nv`. Pull the Ollama
container once into project storage, record its checksum, and store model blobs
outside home:

```bash
mkdir -p "$AF_PROJECT/containers" "$AF_PROJECT/ollama-models" \
  "$AF_WORK/apptainer-cache"
export APPTAINER_CACHEDIR="$AF_WORK/apptainer-cache"
apptainer pull "$AF_PROJECT/containers/ollama.sif" docker://ollama/ollama:latest
sha256sum "$AF_PROJECT/containers/ollama.sif" \
  > "$AF_PROJECT/containers/ollama.sif.sha256"
```

Populate the model directory in a short A40 interactive allocation before the
pilot. Use a nondefault port so independent jobs never share an Ollama server:

```bash
srun --account=bibo-delta-gpu --partition=gpuA40x4-interactive \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64g \
  --gpus-per-node=1 --time=00:45:00 --pty bash

export OLLAMA_HOST=127.0.0.1:21434
export OLLAMA_MODELS=/projects/bibo/$USER/ollama-models
apptainer exec --nv \
  --bind "/projects/bibo/$USER:/projects/bibo/$USER" \
  --env "OLLAMA_HOST=$OLLAMA_HOST" --env "OLLAMA_MODELS=$OLLAMA_MODELS" \
  /projects/bibo/$USER/containers/ollama.sif ollama serve \
  > /tmp/ollama-model-pull.log 2>&1 &
ollama_pid=$!
sleep 10
apptainer exec --nv \
  --bind "/projects/bibo/$USER:/projects/bibo/$USER" \
  --env "OLLAMA_HOST=$OLLAMA_HOST" --env "OLLAMA_MODELS=$OLLAMA_MODELS" \
  /projects/bibo/$USER/containers/ollama.sif ollama pull gpt-oss:20b
kill "$ollama_pid"
exit
```

Then submit one twelve-proposal, no-judge, development-only pilot:

```bash
export AF_OLLAMA_IMAGE="$AF_PROJECT/containers/ollama.sif"
export AF_OLLAMA_MODELS="$AF_PROJECT/ollama-models"
export AF_LOCAL_MODEL="gpt-oss:20b"
export AF_OUTPUT_ROOT="$AF_WORK/phase_b/local-generation-pilot-v8"
mkdir -p logs "$AF_OUTPUT_ROOT"
sbatch --account=bibo-delta-gpu --partition=gpuA40x4 --export=ALL \
  scripts/hpc/phase_b_local_generation_pilot.slurm
```

The pilot targets named canonical T2 easy, a cell with no exact reusable
structure. Acceptance requires at least one deterministically valid fitted
candidate, a final checkpoint at `development_complete`, no test metrics, zero
judge events, and twelve logical proposer rounds. Provider attempts and repair
attempts are reported separately and may exceed twelve because each logical round
has a bounded contract-repair budget. Do not launch all 40
cells until this structured-output and fitting pilot passes.

After the single-seed pilot passes, run a five-seed diagnostic on the same
development-only cell before expanding to the full suite. Each array element
uses its array index as the experiment seed and retains the twelve-round
logical proposal budget:

```bash
export AF_OUTPUT_ROOT="$AF_WORK/phase_b/local-generation-diagnostic-v2"
mkdir -p logs "$AF_OUTPUT_ROOT"
sbatch --account=bibo-delta-gpu --partition=gpuA40x4 --array=0-4 \
  --export=ALL scripts/hpc/phase_b_local_generation_pilot.slurm
```

The five seeds share the pinned Ollama model directory but create distinct run
directories and LLM caches under `AF_OUTPUT_ROOT`. Each array index is also
passed to Ollama as a recorded sampling seed with temperature 0.2. Stagnation
patience equals the twelve-round budget so every seed can use its full proposal
allowance. Compare valid-fit rate, best validation normalized MSE, selected
structural family, repair accounting, and soft-constraint violations. This
diagnostic remains development-only, performs no judge calls, and does not open
test data.

For the prospective scientific-judge weighted-objective pilot, use a new output
root and export the complete configuration into the submitted jobs. Judge calls
increase runtime, so request two hours:

```bash
export AF_OUTPUT_ROOT="$AF_WORK/phase_b/local-generation-weighted-v1"
export AF_ENABLE_JUDGE=true
export AF_SELECTION_POLICY=normalized_weighted_sum
export AF_JUDGE_WEIGHT=0.25
mkdir -p logs "$AF_OUTPUT_ROOT"
sbatch --account=bibo-delta-gpu --partition=gpuA40x4 --time=02:00:00 \
  --array=0-4 --export=ALL \
  scripts/hpc/phase_b_local_generation_pilot.slurm
```

The job script validates that weighted selection cannot accidentally run with
the judge disabled. The frozen selection and summary record the policy, weight,
raw judge score, normalized fit and judge components, and combined objective.

On Delta, the Slurm script derives the standard project, work, checkout,
virtual-environment, public-data, container, and model paths from
`SLURM_JOB_USER`. Login-shell exports are optional overrides, not required job
inputs, so a disconnected or expired login session does not affect an already
submitted job.

### Delta vLLM judge reasoning pilot

After the one-A40 vLLM smoke test passes, compare low and high GPT-OSS reasoning
on the same four frozen difficult pairs, five seeds, and both candidate
orientations. The eight array tasks are self-contained: tasks 0--3 run low
reasoning and tasks 4--7 run high reasoning, with one pair per task. Each task
builds a job-local Apptainer sandbox from the persistent OCI cache, avoiding the
Delta `mksquashfs` failure observed when converting this image to SIF.

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
gpu_account="$(accounts | awk '/gpu/ {print $1; exit}')"
sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_hybrid_judge_vllm_reasoning.slurm
```

The input is the existing held-out pair and label set under
`$AF_WORK/phase_b/judge-hybrid-heldout-v1`; no new labels or mutations are
generated. After all eight tasks finish, merge and analyze low and high
separately:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
pilot=/work/hdd/bibo/$USER/phase_b/judge-hybrid-vllm-reasoning-pilot-v1
labels=/work/hdd/bibo/$USER/phase_b/judge-hybrid-heldout-v1/hybrid_labels.jsonl

for effort in low high; do
  root="$pilot/$effort"
  "$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
    --inputs "$root"/shards/shard_*/hybrid_judge_scores.csv \
    --failure-inputs "$root"/shards/shard_*/hybrid_judge_failures.jsonl \
    --output "$root/hybrid_judge_scores.csv" \
    --failure-output "$root/hybrid_judge_failures.jsonl" \
    --expected 40
  "$python_bin" "$repo/scripts/analyze_hybrid_judge.py" \
    --scores "$root/hybrid_judge_scores.csv" \
    --failures "$root/hybrid_judge_failures.jsonl" \
    --labels "$labels" \
    --output "$root/hybrid_judge_metrics.json"
done
```

Compare response success, conditional and end-to-end preference accuracy,
question-level accuracy, order consistency, repeat ICC/SD, provider attempts,
and elapsed time. This is a serving-runtime and reasoning-effort ablation only;
it does not change the production Ollama judge or search objective.

If low reasoning is selected, do not repeat its first four pairs. Run only the
six untouched held-out pairs with the frozen expansion job:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
gpu_account="$(accounts | awk '/gpu/ {print $1; exit}')"
sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_hybrid_judge_vllm_low_expansion.slurm
```

The six array tasks produce 60 new calls. Merge them with the previously merged
40-call low-reasoning stress subset, rejecting any duplicate key, then analyze
the complete ten-pair/100-call held-out set:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
pilot=/work/hdd/bibo/$USER/phase_b/judge-hybrid-vllm-reasoning-pilot-v1
expansion=/work/hdd/bibo/$USER/phase_b/judge-hybrid-vllm-low-expansion-v1
final=/work/hdd/bibo/$USER/phase_b/judge-hybrid-vllm-low-full-v1
labels=/work/hdd/bibo/$USER/phase_b/judge-hybrid-heldout-v1/hybrid_labels.jsonl

"$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
  --inputs \
    "$pilot/low/hybrid_judge_scores.csv" \
    "$expansion"/shards/shard_*/hybrid_judge_scores.csv \
  --failure-inputs \
    "$pilot/low/hybrid_judge_failures.jsonl" \
    "$expansion"/shards/shard_*/hybrid_judge_failures.jsonl \
  --output "$final/hybrid_judge_scores.csv" \
  --failure-output "$final/hybrid_judge_failures.jsonl" \
  --expected 100

"$python_bin" "$repo/scripts/analyze_hybrid_judge.py" \
  --scores "$final/hybrid_judge_scores.csv" \
  --failures "$final/hybrid_judge_failures.jsonl" \
  --labels "$labels" \
  --output "$final/hybrid_judge_metrics.json"

"$python_bin" "$repo/scripts/analyze_hybrid_operating_points.py" \
  --scores "$final/hybrid_judge_scores.csv" \
  --failures "$final/hybrid_judge_failures.jsonl" \
  --labels "$labels" \
  --protocol-config "$repo/configs/hybrid_judge_vllm_low_protocol_v1.json" \
  --output "$final/hybrid_judge_operating_points.json"
```

This final merge is valid only when it reports exactly 100 unique outcomes and
zero conflicting preference keys.

Diagnose pair, mutation, order, repetition, and certified-question errors without
making another LLM call:

```bash
pairs=/work/hdd/bibo/$USER/phase_b/judge-hybrid-heldout-v1/pairs.jsonl

"$python_bin" "$repo/scripts/analyze_hybrid_diagnostics.py" \
  --scores "$final/hybrid_judge_scores.csv" \
  --failures "$final/hybrid_judge_failures.jsonl" \
  --labels "$labels" \
  --pairs "$pairs" \
  --output "$final/hybrid_judge_diagnostics.json"

cat "$final/hybrid_judge_diagnostics.md"

"$python_bin" "$repo/scripts/audit_hybrid_judge_evidence.py" \
  --scores "$final/hybrid_judge_scores.csv" \
  --labels "$labels" \
  --pairs "$pairs" \
  --output "$final/hybrid_judge_evidence_errors.jsonl" \
  --summary "$final/hybrid_judge_evidence_audit.md"

cat "$final/hybrid_judge_evidence_audit.md"
```

The diagnostic reconstructs the frozen score before evaluating a fixed generic
grid of comparative weights and tie thresholds. It groups mutations by
canonical baseline structure and reports leave-one-structure-out sensitivity.
All alternative aggregation results are explicitly post-hoc and cannot be used
as confirmatory performance for a revised protocol on the same pairs.
The separate evidence audit retains every incorrect certified verdict and its
stored rationale in JSONL, while the Markdown report groups verdict patterns by
criterion, mutation, and candidate order and shows bounded representative
examples. Hidden reasoning is never parsed.

The evidence audit identified two development targets: wrong source/sink polarity
and exact repeated flux accounting. Run the frozen 40-call matched vLLM-low pilot
on only the two pairs from each mutation family:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
gpu_account="$(sacct -X -n -j 21372879 -o Account%40 | awk 'NF {print $1; exit}')"
sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_hybrid_judge_vllm_facts_pilot.slurm
```

The four array tasks each request one A40 and make ten calls for one pair. After
all four finish, merge and analyze the 40 treatment outcomes:

```bash
facts=/work/hdd/bibo/$USER/phase_b/judge-hybrid-vllm-facts-pilot-v1

"$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
  --inputs "$facts"/shards/shard_*/hybrid_judge_scores.csv \
  --failure-inputs "$facts"/shards/shard_*/hybrid_judge_failures.jsonl \
  --output "$facts/hybrid_judge_scores.csv" \
  --failure-output "$facts/hybrid_judge_failures.jsonl" \
  --expected 40

"$python_bin" "$repo/scripts/analyze_hybrid_judge.py" \
  --scores "$facts/hybrid_judge_scores.csv" \
  --failures "$facts/hybrid_judge_failures.jsonl" \
  --labels "$labels" \
  --output "$facts/hybrid_judge_metrics.json"

"$python_bin" "$repo/scripts/analyze_hybrid_diagnostics.py" \
  --scores "$facts/hybrid_judge_scores.csv" \
  --failures "$facts/hybrid_judge_failures.jsonl" \
  --labels "$labels" \
  --pairs "$pairs" \
  --output "$facts/hybrid_judge_diagnostics.json"

"$python_bin" "$repo/scripts/audit_hybrid_judge_evidence.py" \
  --scores "$facts/hybrid_judge_scores.csv" \
  --labels "$labels" \
  --pairs "$pairs" \
  --output "$facts/hybrid_judge_evidence_errors.jsonl" \
  --summary "$facts/hybrid_judge_evidence_audit.md"
```

This is development reuse of opened pairs, not a new held-out claim. Model,
reasoning effort, seeds, orders, retries, schema, and scoring remain matched to
the old control. Only the general algebraic facts and rubric wording change.

The facts treatment improved exact-repeat detection but did not pass source-role,
comparative, or order-consistency gates. Run the frozen two-stage atomic protocol
with both model sizes from any fresh login; all persistent paths are reconstructed
inside the scripts:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
gpu_account="$(accounts | awk '/gpu/ {print $1; exit}')"

sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_hybrid_judge_vllm_atomic_20b.slurm

sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_hybrid_judge_vllm_atomic_120b.slurm
```

The 20B job is a four-task array with one A40 per task. The 120B job is one
four-A40 task using tensor parallelism, which avoids repeatedly loading the large
model. Each model performs 40 paired judgments; every judgment has one
sign-blinded atomic call and one full hybrid comparison call. Both use low
reasoning, temperature `0.2`, seeds `10000` through `10004`, both candidate
orientations, and unchanged scoring.

After both jobs finish, merge each model and then the matched comparison:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
atomic=/work/hdd/bibo/$USER/phase_b/judge-hybrid-atomic-v1
heldout=/work/hdd/bibo/$USER/phase_b/judge-hybrid-heldout-v1
labels="$heldout/hybrid_labels.jsonl"
pairs="$heldout/pairs.jsonl"

for model in gpt-oss-20b gpt-oss-120b; do
  root="$atomic/$model"
  "$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
    --inputs "$root"/shards/shard_*/hybrid_judge_scores.csv \
    --failure-inputs "$root"/shards/shard_*/hybrid_judge_failures.jsonl \
    --output "$root/hybrid_judge_scores.csv" \
    --failure-output "$root/hybrid_judge_failures.jsonl" \
    --expected 40

  "$python_bin" "$repo/scripts/analyze_hybrid_judge.py" \
    --scores "$root/hybrid_judge_scores.csv" \
    --failures "$root/hybrid_judge_failures.jsonl" \
    --labels "$labels" \
    --output "$root/hybrid_judge_metrics.json"

  "$python_bin" "$repo/scripts/analyze_atomic_evidence.py" \
    --scores "$root/hybrid_judge_scores.csv" \
    --output "$root/atomic_evidence_metrics.json"
done

comparison="$atomic/matched-comparison"
mkdir -p "$comparison"
"$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
  --inputs "$atomic"/gpt-oss-*/hybrid_judge_scores.csv \
  --failure-inputs "$atomic"/gpt-oss-*/hybrid_judge_failures.jsonl \
  --output "$comparison/hybrid_judge_scores.csv" \
  --failure-output "$comparison/hybrid_judge_failures.jsonl" \
  --expected 80

"$python_bin" "$repo/scripts/analyze_hybrid_judge.py" \
  --scores "$comparison/hybrid_judge_scores.csv" \
  --failures "$comparison/hybrid_judge_failures.jsonl" \
  --labels "$labels" \
  --output "$comparison/hybrid_judge_metrics.json"

"$python_bin" "$repo/scripts/analyze_atomic_evidence.py" \
  --scores "$comparison/hybrid_judge_scores.csv" \
  --output "$comparison/atomic_evidence_metrics.json"

cat "$comparison/hybrid_judge_summary.md"
cat "$comparison/atomic_evidence_metrics.md"
```

Decompose the persisted decisions into atomic-sensitive, other-absolute, and
direct-comparative contributions without making new LLM calls:

```bash
"$python_bin" "$repo/scripts/analyze_hybrid_decision_decomposition.py" \
  --scores \
    "$atomic/gpt-oss-20b/hybrid_judge_scores.csv" \
    "$atomic/gpt-oss-120b/hybrid_judge_scores.csv" \
  --failures \
    "$atomic/gpt-oss-20b/hybrid_judge_failures.jsonl" \
    "$atomic/gpt-oss-120b/hybrid_judge_failures.jsonl" \
  --labels "$labels" \
  --output "$comparison/hybrid_decision_decomposition.json"

cat "$comparison/hybrid_decision_decomposition.md"
```

The atomic-sensitive marginal is a counterfactual diagnostic: it removes the
source-role, sink-role, and semantic-duplication assessments and recomputes the
existing nonlinear group score. It does not claim that all remaining absolute
answers have certified question-level labels.

Run comparative-criterion leave-one-out, single-criterion, and frozen
weight/threshold sensitivity analyses on the same rows:

```bash
"$python_bin" "$repo/scripts/analyze_hybrid_comparative_ablation.py" \
  --scores \
    "$atomic/gpt-oss-20b/hybrid_judge_scores.csv" \
    "$atomic/gpt-oss-120b/hybrid_judge_scores.csv" \
  --failures \
    "$atomic/gpt-oss-20b/hybrid_judge_failures.jsonl" \
    "$atomic/gpt-oss-120b/hybrid_judge_failures.jsonl" \
  --labels "$labels" \
  --output "$comparison/hybrid_comparative_ablation.json"

cat "$comparison/hybrid_comparative_ablation.md"
```

The mutation-contract-labeled-only configuration is an offline diagnostic upper
bound. It uses label provenance to choose a criterion subset and therefore must
never be used during ordinary judging or search.

This remains a matched development factorial over opened pairs. Mutation labels
are used only by the offline analyzers after calls complete. A selected protocol
must be frozen unchanged before any new-structure confirmation.

## ACES: first login and identity check

Use the ACES portal at `https://portal-aces.hprc.tamu.edu`. The portal's SSHCA
application creates a 49-hour SSH credential if direct SSH is desired. In the
portal shell, run:

```bash
hostname
id
myproject -l
myproject -m
showquota
echo "HOME=$HOME"
echo "SCRATCH=$SCRATCH"
echo "PROJECT=$PROJECT"
sinfo -o '%P %a %l %c %m %G'
module spider Python
```

Set the NAIRR project as default only after matching its local account number:

```bash
myproject -d ACES_ACCOUNT_NUMBER
```

Use `$PROJECT` for the shared durable checkout/inputs and `$SCRATCH` for active
outputs. Home is only 10 GB; user scratch is 1 TB and is not backed up. ACES
also provides a 5 TB allocation project directory.

### ACES environment and pilot

Load the exact Python version reported by `module spider Python`, then:

```bash
module purge
module load Python/AVAILABLE_VERSION
mkdir -p "$PROJECT/repos" "$PROJECT/venvs" "$SCRATCH/phase_b" "$PROJECT/phase_b"
git clone git@github.com:yuanzhangxiao/task-specified-autoformulation.git \
  "$PROJECT/repos/autoformalism"
cd "$PROJECT/repos/autoformalism"
git checkout FROZEN_TAG
python -m venv "$PROJECT/venvs/autoformalism-FROZEN_TAG"
source "$PROJECT/venvs/autoformalism-FROZEN_TAG/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

Submit the same array script only after a serial pilot passes:

```bash
cd "$PROJECT/repos/autoformalism"
mkdir -p logs "$SCRATCH/phase_b/replay-replication"
export AF_REPO_ROOT="$PROJECT/repos/autoformalism"
export AF_PYTHON="$PROJECT/venvs/autoformalism-FROZEN_TAG/bin/python"
export AF_PUBLIC_DATA_ROOT="$PROJECT/phase_b/inputs/phase-b-public-release"
export AF_CANDIDATE_POOL="$PROJECT/phase_b/inputs/phase-b-portable-candidate-pool.jsonl"
export AF_OUTPUT_ROOT="$SCRATCH/phase_b/replay-replication"
sbatch --account=ACES_ACCOUNT_NUMBER --partition=cpu \
  --export=ALL scripts/hpc/phase_b_replay_array.slurm
```

Do not run fitting on ACES login nodes. Copy completed results from scratch to
project storage and to a second system.

## Jetstream2: short-lived development VM

In Exosphere:

1. Add an ACCESS account and select the Jetstream2 CPU allocation.
2. Create a named Ubuntu instance, initially `m3.medium` (8 vCPU, 30 GB RAM).
3. Assign a public IP and add an SSH public key during creation.
4. Leave the generated cloud-init/boot script unchanged.
5. Create/attach storage or a Manila share for persistent inputs/results.

An `m3.medium` consumes 8 SUs per running hour. Shelve or delete it when not in
use; Jetstream capacity is not a free always-on service. Exosphere creates the
`exouser` account, which has sudo access and supports a web shell. Native SSH is:

```bash
ssh exouser@PUBLIC_IP
```

Install system packages, clone the frozen tag, make a venv, and run tests as on
Google Cloud. Keep durable data on the allocated volume/share, not only on the
instance root disk. Jetstream is useful for orchestration and development, but
Delta/ACES Slurm arrays are the correct place for the production replay.

## GPU phase (after CPU replay)

Do not consume GPU hours until the CPU replay, candidate budgets, prompts, and
model-serving design are frozen. On Delta, use `accounts` to select the GPU
account and inspect current partitions with `sinfo`. Delta recommends
Apptainer/NGC containers for GPU Python workloads. Store the Apptainer cache
outside home and pin the image digest. Begin with one short interactive GPU
smoke test, record model/token throughput and memory, then write a separate
checkpointed inference array. Never place API keys or model tokens in a Slurm
script or repository.

## Acceptance checklist

For each system, do not start production until all are true:

- hostname, project account, storage paths, and quotas are recorded;
- frozen Git tag and input SHA-256 values match;
- environment imports `autoformalism` from the intended checkout;
- full tests pass and known Ruff exceptions are recorded;
- a two-candidate no-LLM pilot completes;
- output is checkpointed and resume works;
- thread counts are pinned to one for one-core fitting jobs;
- no secret is present in source, logs, or job scripts;
- results are copied to a second storage system.
