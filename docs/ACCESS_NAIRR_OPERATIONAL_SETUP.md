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

### Frozen 120B unseen-structure confirmation

The 120B atomic protocol passed the opened-pair development gates. Build the
confirmation input from selected models whose canonical structures have never
appeared in an earlier pair artifact. The commands below conservatively exclude
every `pairs.jsonl` under the Phase-B root, not merely the final calibration and
held-out files. They also inventory every available local-generation run root and
deduplicate repeated structures. Run this once, inspect and retain the manifest,
then do not add or remove exclusions after judge calls begin:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
phase=/work/hdd/bibo/$USER/phase_b
confirm="$phase/judge-hybrid-atomic-confirmation-v1"
data=/projects/bibo/$USER/phase_b/inputs/public

mkdir -p "$confirm"
runs_args=()
while IFS= read -r path; do
  runs_args+=(--runs-root "$path")
done < <(
  find "$phase" -mindepth 2 -maxdepth 2 -type d -name runs \
    -path '*local-generation*' | sort
)
exclusion_args=()
while IFS= read -r path; do
  exclusion_args+=(--exclude-pairs "$path")
done < <(
  find "$phase" -type f -name pairs.jsonl \
    ! -path "$confirm/*" | sort
)

(( ${#runs_args[@]} > 0 )) || { echo "no local-generation run roots" >&2; exit 1; }
(( ${#exclusion_args[@]} > 0 )) || { echo "no historical pair files" >&2; exit 1; }

"$python_bin" "$repo/scripts/build_hybrid_judge_confirmation_pairs.py" \
  "${runs_args[@]}" \
  --data-root "$data" \
  "${exclusion_args[@]}" \
  --baseline-count 2 \
  --output "$confirm/pairs.jsonl" \
  --manifest "$confirm/confirmation_pairs_manifest.json"

"$python_bin" "$repo/scripts/build_hybrid_judge_label_template.py" \
  --pairs "$confirm/pairs.jsonl" \
  --data-root "$data" \
  --output "$confirm/hybrid_labels.jsonl"

cp "$repo/configs/hybrid_judge_atomic_confirmation_v1.json" \
  "$confirm/protocol_config.json"
wc -l "$confirm/pairs.jsonl" "$confirm/hybrid_labels.jsonl"
jq . "$confirm/confirmation_pairs_manifest.json"
sha256sum "$confirm/pairs.jsonl" "$confirm/hybrid_labels.jsonl" \
  "$confirm/confirmation_pairs_manifest.json" "$confirm/protocol_config.json" \
  > "$confirm/frozen_inputs.sha256"
```

The two line counts must both be four. The manifest must report two selected
baseline fingerprints, four selected pair IDs, the two frozen mutation types,
and a nonempty exclusion-file list. If fewer than two unseen structures remain,
stop and generate additional development-only candidates under new seeds; do not
relax the exclusion boundary.

Submit the self-contained four-A40 120B job from any fresh login:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
gpu_account="$(accounts | awk '/gpu/ {print $1; exit}')"
sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_hybrid_judge_vllm_atomic_confirmation_120b.slurm
```

After completion, merge exactly 40 outcomes and apply the standard, atomic, and
predeclared confirmation analyses:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
confirm=/work/hdd/bibo/$USER/phase_b/judge-hybrid-atomic-confirmation-v1
root="$confirm/gpt-oss-120b"

"$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
  --inputs "$root"/shards/shard_*/hybrid_judge_scores.csv \
  --failure-inputs "$root"/shards/shard_*/hybrid_judge_failures.jsonl \
  --output "$root/hybrid_judge_scores.csv" \
  --failure-output "$root/hybrid_judge_failures.jsonl" \
  --expected 40

"$python_bin" "$repo/scripts/analyze_hybrid_judge.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$confirm/hybrid_labels.jsonl" \
  --output "$root/hybrid_judge_metrics.json"

"$python_bin" "$repo/scripts/analyze_atomic_evidence.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --output "$root/atomic_evidence_metrics.json"

"$python_bin" "$repo/scripts/analyze_hybrid_judge_confirmation.py" \
  --hybrid-metrics "$root/hybrid_judge_metrics.json" \
  --atomic-metrics "$root/atomic_evidence_metrics.json" \
  --protocol-config "$confirm/protocol_config.json" \
  --output "$root/confirmation_result.json"

cat "$root/confirmation_result.md"
```

Report the frozen result whether it passes or fails. Do not use these outcomes to
tune the three comparative questions, `0.25` comparative weight, `0.05` partial
weight, or `0.05` tie threshold. Equivalent-model and mechanism-tradeoff pairs are
a subsequent calibration milestone, not part of this confirmation.

### Equivalence and non-ordered tradeoff development

After opening and reporting the confirmation result, build the next development
set from its two baseline structures. This set has known truth only for exact
equivalence and mutation-certified atomic defects; it deliberately does not assign
a gold winner to defect-vs-defect tradeoffs:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
phase=/work/hdd/bibo/$USER/phase_b
confirm="$phase/judge-hybrid-atomic-confirmation-v1"
scorecal="$phase/judge-hybrid-equivalence-tradeoff-v1"
data=/projects/bibo/$USER/phase_b/inputs/public

mkdir -p "$scorecal"
"$python_bin" "$repo/scripts/build_hybrid_judge_equivalence_tradeoff_pairs.py" \
  --source-pairs "$confirm/pairs.jsonl" \
  --data-root "$data" \
  --baseline-count 2 \
  --output "$scorecal/pairs.jsonl" \
  --manifest "$scorecal/equivalence_tradeoff_pairs_manifest.json"

"$python_bin" "$repo/scripts/build_hybrid_judge_label_template.py" \
  --pairs "$scorecal/pairs.jsonl" \
  --data-root "$data" \
  --output "$scorecal/hybrid_labels.jsonl"

cp "$repo/configs/hybrid_judge_equivalence_tradeoff_v1.json" \
  "$scorecal/protocol_config.json"
wc -l "$scorecal/pairs.jsonl" "$scorecal/hybrid_labels.jsonl"
jq . "$scorecal/equivalence_tradeoff_pairs_manifest.json"
sha256sum "$scorecal/pairs.jsonl" "$scorecal/hybrid_labels.jsonl" \
  "$scorecal/equivalence_tradeoff_pairs_manifest.json" \
  "$scorecal/protocol_config.json" > "$scorecal/frozen_inputs.sha256"
```

Both line counts must be eight. The manifest must show two source structures,
eight unique pair IDs, one equivalence type, three tradeoff types, and status
`frozen_before_judge_calls`. Then submit the self-contained four-A40 job:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
gpu_account="$(accounts | awk '/gpu/ {print $1; exit}')"
sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_hybrid_judge_vllm_equivalence_tradeoff_120b.slurm
```

After completion, merge exactly 80 outcomes and apply both the standard report
and the predeclared equivalence/tradeoff analysis:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
scorecal=/work/hdd/bibo/$USER/phase_b/judge-hybrid-equivalence-tradeoff-v1
root="$scorecal/gpt-oss-120b"

"$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
  --inputs "$root"/shards/shard_*/hybrid_judge_scores.csv \
  --failure-inputs "$root"/shards/shard_*/hybrid_judge_failures.jsonl \
  --output "$root/hybrid_judge_scores.csv" \
  --failure-output "$root/hybrid_judge_failures.jsonl" \
  --expected 80

"$python_bin" "$repo/scripts/analyze_hybrid_judge.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$scorecal/hybrid_labels.jsonl" \
  --output "$root/hybrid_judge_metrics.json"

"$python_bin" "$repo/scripts/analyze_equivalence_tradeoff_judge.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$scorecal/hybrid_labels.jsonl" \
  --protocol-config "$scorecal/protocol_config.json" \
  --output "$root/equivalence_tradeoff_result.json"

cat "$root/equivalence_tradeoff_result.md"
```

Report pass or fail without altering the frozen thresholds. A tradeoff preference
count is not an accuracy result because no overall tradeoff winner was labeled.

If the orientation-bias gate fails, compare symmetry-preserving aggregators over
the frozen calls before making another LLM request. This is explicitly post-hoc
development analysis, not a reinterpretation of the frozen gate:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
scorecal=/work/hdd/bibo/$USER/phase_b/judge-hybrid-equivalence-tradeoff-v1
root="$scorecal/gpt-oss-120b"

"$python_bin" "$repo/scripts/analyze_hybrid_symmetric_aggregation.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$scorecal/hybrid_labels.jsonl" \
  --protocol-config "$scorecal/protocol_config.json" \
  --output "$root/symmetric_aggregation_analysis.json"

cat "$root/symmetric_aggregation_analysis.md"
```

The report compares paired final-decision averaging, strict question consensus,
and uncertainty-aware abstention. It must preserve equivalence truth and report
coverage, repeat stability, raw orientation gaps, and question disagreements.
Do not count unlabeled tradeoff winners as accuracy or select a rule because it
prefers a particular defect.

### Frozen fresh-structure question-consensus validation

After selecting strict question consensus from the opened equivalence/tradeoff
calls, build the final validation set from canonical structures absent from every
opened pair file. All roots are reconstructed in this single shell block so a
later login does not depend on exported session variables:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
phase=/work/hdd/bibo/$USER/phase_b
data=/projects/bibo/$USER/phase_b/inputs/public
validation="$phase/judge-hybrid-consensus-validation-v1"

cd "$repo"
mkdir -p "$validation"

mapfile -t run_roots < <(
  find "$phase" -type d -name runs -path "$phase/local-generation-*" | sort
)
runs_args=()
for root in "${run_roots[@]}"; do
  runs_args+=(--runs-root "$root")
done

mapfile -t exclusion_files < <(
  find "$phase" -type f -name pairs.jsonl \
    ! -path "$validation/*" | sort
)
exclusion_args=()
for path in "${exclusion_files[@]}"; do
  exclusion_args+=(--exclude-pairs "$path")
done

"$python_bin" "$repo/scripts/build_hybrid_judge_consensus_validation_pairs.py" \
  "${runs_args[@]}" \
  --data-root "$data" \
  "${exclusion_args[@]}" \
  --baseline-count 2 \
  --output "$validation/pairs.jsonl" \
  --manifest "$validation/consensus_validation_pairs_manifest.json"

"$python_bin" "$repo/scripts/build_hybrid_judge_label_template.py" \
  --pairs "$validation/pairs.jsonl" \
  --data-root "$data" \
  --output "$validation/hybrid_labels.jsonl"

cp "$repo/configs/hybrid_judge_consensus_validation_v1.json" \
  "$validation/protocol_config.json"
wc -l "$validation/pairs.jsonl" "$validation/hybrid_labels.jsonl"
jq . "$validation/consensus_validation_pairs_manifest.json"
sha256sum "$validation/pairs.jsonl" "$validation/hybrid_labels.jsonl" \
  "$validation/consensus_validation_pairs_manifest.json" \
  "$validation/protocol_config.json" > "$validation/frozen_inputs.sha256"
```

Both line counts must be 14. The manifest must report two selected baseline
fingerprints, seven pair types, 14 unique pair IDs, a nonempty exclusion list,
and status `frozen_before_judge_calls`. If the builder finds fewer than two unseen
structures, stop and generate new development-only candidates; do not remove an
exclusion file or reuse an opened structure.

Submit the self-contained four-A40 job. The script reconstructs all persistent
paths inside the job, so logout does not unset `AF_PROJECT` or related variables:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
gpu_account="$(accounts | awk '/gpu/ {print $1; exit}')"
sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_hybrid_judge_vllm_consensus_validation_120b.slurm
```

After completion, merge exactly 140 outcomes, create the standard report, form
identity-normalized question consensus, and apply the frozen gates:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
validation=/work/hdd/bibo/$USER/phase_b/judge-hybrid-consensus-validation-v1
root="$validation/gpt-oss-120b"

"$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
  --inputs "$root"/shards/shard_*/hybrid_judge_scores.csv \
  --failure-inputs "$root"/shards/shard_*/hybrid_judge_failures.jsonl \
  --output "$root/hybrid_judge_scores.csv" \
  --failure-output "$root/hybrid_judge_failures.jsonl" \
  --expected 140

"$python_bin" "$repo/scripts/analyze_hybrid_judge.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$validation/hybrid_labels.jsonl" \
  --output "$root/hybrid_judge_metrics.json"

"$python_bin" "$repo/scripts/analyze_hybrid_symmetric_aggregation.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$validation/hybrid_labels.jsonl" \
  --protocol-config "$validation/protocol_config.json" \
  --output "$root/symmetric_aggregation_analysis.json"

"$python_bin" "$repo/scripts/analyze_hybrid_consensus_validation.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$validation/hybrid_labels.jsonl" \
  --symmetric-analysis "$root/symmetric_aggregation_analysis.json" \
  --protocol-config "$validation/protocol_config.json" \
  --output "$root/consensus_validation_result.json"

cat "$root/consensus_validation_result.md"
```

Report PASS or FAIL exactly as produced. The defect-tradeoff winner counts are
descriptive, and no prompt, weight, threshold, aggregation rule, or gate may be
tuned using these validation structures.

After a PASS, run the frozen call-budget analysis on the stored symmetric trials.
This is a quick CPU-only login-node analysis and makes no LLM request:

```bash
(
set -euo pipefail

repo="/projects/bibo/$USER/repos/autoformalism-v21"
validation="/work/hdd/bibo/$USER/phase_b/judge-hybrid-consensus-validation-v1"
root="$validation/gpt-oss-120b"
python_bin="/projects/bibo/$USER/venvs/autoformalism-v21/bin/python"

cd "$repo"
cp "$repo/configs/hybrid_judge_consensus_operating_point_v1.json" \
  "$validation/operating_point_config.json"
sha256sum "$validation/operating_point_config.json" \
  > "$validation/operating_point_config.sha256"

"$python_bin" "$repo/scripts/analyze_hybrid_consensus_operating_points.py" \
  --symmetric-analysis "$root/symmetric_aggregation_analysis.json" \
  --labels "$validation/hybrid_labels.jsonl" \
  --protocol-config "$validation/operating_point_config.json" \
  --output "$root/consensus_operating_points.json"

cat "$root/consensus_operating_points.md"
)
```

The selected row must pass every gate. Do not select a more expensive row merely
because it improves an unlabeled tradeoff preference, and do not make additional
judge calls for this analysis.

### Incumbent-relative hybrid-search fixed-denominator confirmation

After the frozen consensus operating point passes, run one development-only
controller confirmation. This job uses the same four-A40 vLLM 120B server for both
proposal and paired scientific judgment. That choice isolates checkpoint,
feedback, and selection plumbing; it is not a comparison with the 20B proposer
and must not be reported as such. The default eight rounds use the version-2
neutral fixed-denominator comparative rule. Test data are never opened.

The Slurm file contains the Delta GPU account, partition, and persistent path
defaults, so it does not depend on exported login-shell variables after logout:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
git pull --ff-only
mkdir -p logs
sbatch scripts/hpc/phase_b_hybrid_search_smoke_120b.slurm
```

The job automatically resumes if the same run directory already contains a
checkpoint. To restart deliberately with a different seed, give it a new output
root rather than overwriting the existing ledger:

```bash
sbatch --export=ALL,AF_SEED=1,AF_OUTPUT_ROOT=/work/hdd/bibo/$USER/phase_b/hybrid-search-smoke-v3-seed1 \
  scripts/hpc/phase_b_hybrid_search_smoke_120b.slurm
```

After reconnecting, replace `JOB_ID` below. Log paths are absolute, so the
commands work from any login directory:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
tail -n 80 "$repo/logs/phase-b-hybrid-search-smoke-JOB_ID.out"
tail -n 80 "$repo/logs/phase-b-hybrid-search-smoke-JOB_ID.err"
```

Inspect the development summary and checkpointed challenge ledger:

```bash
run=/work/hdd/bibo/$USER/phase_b/hybrid-search-smoke-v3/runs/phase_b_dalla_man_t2_canonical_named_easy_easy_seed0
jq '{status,stopping_reason,selection_validation_normalized_mse,selected_candidate_id}' \
  "$run/summary.json"

for file in "$run"/checkpoints/round_*.json; do
  jq -r '[input_filename,
    (.record.pruned_candidate.candidate_id // .candidate.candidate_id // "-"),
    (.record.incumbent_challenge.judgment.protocol_version // "seed"),
    (.record.incumbent_challenge.judgment.comparative_indeterminate_policy // "seed"),
    (.record.incumbent_challenge.fit_preference_for_challenger // "seed"),
    (.record.incumbent_challenge.science_preference_for_challenger // "seed"),
    (.record.incumbent_challenge.combined_preference_for_challenger // "seed"),
    (.record.incumbent_challenge.selected_hash // "seed")]
    | @tsv' "$file"
done
```

Passing this smoke means the end-to-end development plumbing completed and the
challenge ledger is internally consistent. It does not establish scientific
superiority, tune the fit/science weight, or authorize test access.

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

### Delta model-semantics judge validation

The model-semantic validation is a separate frozen milestone; do not reuse or
overwrite the earlier consensus-validation directory. Prepare `pairs.jsonl`,
`hybrid_labels.jsonl`, the pair manifest, and a copied protocol config under
`$AF_WORK/phase_b/judge-hybrid-model-semantics-v1`. The builder must receive all
previously opened pair files through repeated `--exclude-pairs` arguments and
enough completed run roots to find two eligible structures. The current inventory
has no eligible structure outside all earlier pair files, so preparation uses
`--allow-opened-baselines` and the manifest must report exactly two previously
opened fingerprints. This makes the result targeted development calibration, not
fresh-structure confirmation. Label generation must include
`--model-semantic-contract`.

Submit only after verifying the counts are four pairs and four labels:

```bash
cd "$AF_REPO_ROOT"
mkdir -p logs
sbatch scripts/hpc/phase_b_hybrid_judge_vllm_model_semantics_120b.slurm
```

The launcher is session-independent: the durable project, work, repository,
model, pair, output, semantic-contract, and fixed-denominator defaults are all
resolved inside the batch job. Logging out does not unset them. After completion,
merge the single shard, run the existing hybrid and symmetric analyzers, and run
`analyze_hybrid_model_semantics_validation.py`. Do not integrate the questions
into search unless the final Markdown report says `PASS` for every predeclared
gate.

The completed v1 run must not be interpreted with those gates. Post-run audit
found invalid pair construction, recorded in
`configs/hybrid_judge_model_semantics_validation_v1_adjudication.json`. Preserve
its directory unchanged.

Prepare the corrected target-only v2 milestone under a new root. Use
`build_hybrid_judge_target_mapping_pairs.py`; it fails unless process `U` excludes
`Uii` and records every certification in the manifest. Generate labels with
`--target-mapping-semantic-contract`, copy
`configs/hybrid_judge_target_mapping_validation_v2.json` to
`protocol_config.json`, verify two pairs/two labels and their hashes, then submit:

```bash
cd "$AF_REPO_ROOT"
mkdir -p logs
sbatch scripts/hpc/phase_b_hybrid_judge_vllm_target_mapping_v2_120b.slurm
```

The v2 launcher is session-independent and writes only beneath
`$AF_WORK/phase_b/judge-hybrid-target-mapping-v2`. Analyze it with the existing
hybrid, symmetric-aggregation, and model-semantics validation analyzers. The
initialization accuracy gate is intentionally absent because identity-observed
initialization is now deterministic canonicalization rather than an LLM task.

The v2 result is a failed development result because its public prompt did not
define `U` as total disposal. Run the frozen matched prompt-only v3 milestone
without changing or deleting v2 artifacts. It reuses the v2 pair file and creates
an audited copy-on-write public-data overlay containing the revised proposer
prompt. Numeric tables, candidates, judge prompt, model, seeds, and aggregation
remain fixed.

Prepare the overlay and regenerate labels because public-requirement identifiers
depend on prompt text:

```bash
cd "$AF_REPO_ROOT"

experiment_root="$AF_WORK/phase_b/judge-hybrid-target-mapping-v3-prompt-only"
source_pairs="$AF_WORK/phase_b/judge-hybrid-target-mapping-v2/pairs.jsonl"
protocol_config="$AF_REPO_ROOT/configs/hybrid_judge_target_mapping_prompt_revision_v3.json"

"$AF_PYTHON" scripts/prepare_target_mapping_prompt_revision.py \
  --source-data-root "$AF_PUBLIC_DATA_ROOT" \
  --output-data-root "$experiment_root/public" \
  --pairs "$source_pairs" \
  --protocol-config "$protocol_config" \
  --manifest "$experiment_root/prompt_revision_manifest.json"

"$AF_PYTHON" scripts/build_hybrid_judge_label_template.py \
  --pairs "$source_pairs" \
  --data-root "$experiment_root/public" \
  --target-mapping-semantic-contract \
  --output "$experiment_root/hybrid_labels.jsonl"

cp "$protocol_config" "$experiment_root/protocol_config.json"
sha256sum \
  "$source_pairs" \
  "$experiment_root/public/phase_b_v1/phase_b_dalla_man_t2_canonical_named_easy/proposer_prompt.txt" \
  "$experiment_root/hybrid_labels.jsonl" \
  "$experiment_root/prompt_revision_manifest.json" \
  "$experiment_root/protocol_config.json"
```

The preparation command is deterministic and safe to rerun. It fails if the v2
pair hash, source prompt hash, revised prompt hash, benchmark identifier, file
inventory, or any non-prompt file differs. Submit from the repository after the
two labels are present:

```bash
cd "$AF_REPO_ROOT"
mkdir -p logs
sbatch scripts/hpc/phase_b_hybrid_judge_vllm_target_mapping_v3_prompt_120b.slurm
```

The job has durable defaults for every path and may prepare or verify the same
overlay after the login session ends. Analyze the new root with the existing
hybrid, symmetric-aggregation, and model-semantics validators using its copied
`protocol_config.json`. Do not merge or overwrite v2 outputs.

The v3 prompt clarification produced perfect target-mapping accuracy on all
usable paired trials, but one seed failed the atomic unit-ID contract in both
orientations and the run narrowly missed its reliability gates. Preserve v3 and
run the clean-name v4 follow-up under a new root. The batch script is
session-independent: it verifies or prepares the revised-prompt overlay, builds
and certifies two `Uid`/`U` pairs, regenerates target-mapping labels, copies the
frozen config, and then launches vLLM.

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-v21
mkdir -p logs

git fetch origin main
git switch main
git pull --ff-only origin main

sbatch \
  scripts/hpc/phase_b_hybrid_judge_vllm_target_mapping_v4_clean_names_120b.slurm
```

The valid candidate in each pair has an insulin-dependent process `Uid`, a total
target process `U = Uii + Uid`, and mapping `U -> U`. Its mutation retains the
same names and mapping but defines `U = Uid`. Pair certification fails if the
source-pair hash or revised-prompt hash differs, if `Uid` already exists, if the
insulin mechanism is lost, if `Uid` contains `Uii`, or if any field other than
the total-process expression differs.

V4 failed because some calls stopped at `U -> U` without expanding process `U`,
and because the soft balance group allowed comparative preferences to override a
correctly detected target omission. After updating to the protocol-6 commit,
first rescore the frozen v4 calls with hard target enforcement only. This makes
no LLM requests and does not overwrite the original v4 reports:

```bash
repo=/projects/bibo/yxiao2/repos/autoformalism-v21
python_bin=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
control=/work/hdd/bibo/yxiao2/phase_b/judge-hybrid-target-mapping-v4-clean-names
root="$control/gpt-oss-120b"
v5_config="$repo/configs/hybrid_judge_target_mapping_recursive_hard_v5.json"

"$python_bin" "$repo/scripts/analyze_hybrid_symmetric_aggregation.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$control/hybrid_labels.jsonl" \
  --protocol-config "$v5_config" \
  --output "$root/hard_target_contract_rescore.json"

"$python_bin" "$repo/scripts/analyze_hybrid_model_semantics_validation.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$control/hybrid_labels.jsonl" \
  --symmetric-analysis "$root/hard_target_contract_rescore.json" \
  --protocol-config "$v5_config" \
  --output "$root/hard_target_contract_rescore_validation.json"

cat "$root/hard_target_contract_rescore.md"
cat "$root/hard_target_contract_rescore_validation.md"
```

Then submit the matched recursive-prompt experiment. Its launcher copies and
byte-compares the v4 pair and label files before starting vLLM, and all paths and
protocol flags survive logout:

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-v21
mkdir -p logs
sbatch \
  scripts/hpc/phase_b_hybrid_judge_vllm_target_mapping_v5_recursive_hard_120b.slurm
```

Do not reinterpret the offline ablation as prompt accuracy: it changes only the
deterministic enforcement of stored v4 answers. V5 is the matched test of the
recursive instruction plus the frozen hard contract.

V5 retained one atomic terminal failure and two order-specific false passes for
the incomplete target. Before making new calls, rescore its nine complete paired
trials using fail-closed hard-target consensus. This does not recover the failed
call and does not overwrite v5 reports:

```bash
repo=/projects/bibo/yxiao2/repos/autoformalism-v21
python_bin=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
source=/work/hdd/bibo/yxiao2/phase_b/judge-hybrid-target-mapping-v5-recursive-hard
root="$source/gpt-oss-120b"
v6_config="$repo/configs/hybrid_judge_target_mapping_fail_closed_v6.json"

"$python_bin" "$repo/scripts/analyze_hybrid_symmetric_aggregation.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$source/hybrid_labels.jsonl" \
  --protocol-config "$v6_config" \
  --output "$root/fail_closed_target_rescore.json"

"$python_bin" "$repo/scripts/analyze_hybrid_model_semantics_validation.py" \
  --scores "$root/hybrid_judge_scores.csv" \
  --failures "$root/hybrid_judge_failures.jsonl" \
  --labels "$source/hybrid_labels.jsonl" \
  --symmetric-analysis "$root/fail_closed_target_rescore.json" \
  --protocol-config "$v6_config" \
  --output "$root/fail_closed_target_rescore_validation.json"

cat "$root/fail_closed_target_rescore.md"
cat "$root/fail_closed_target_rescore_validation.md"
```

Then submit the matched v6 run. It reuses and byte-compares the v5 pair and label
files, keeps the recursive prompts and all numeric settings fixed, and changes
only fail-closed hard-target consensus plus the bounded neutral atomic repair:

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-v21
mkdir -p logs
sbatch \
  scripts/hpc/phase_b_hybrid_judge_vllm_target_mapping_v6_fail_closed_120b.slurm
```

The atomic repair does not silently claim success. Raw provider omissions and
the exact neutral fills remain visible in the CSV provenance fields. Do not
enable either repair in search until v6 passes every unchanged gate and a later
fresh-structure confirmation passes.

## Audit and compare the raw-data frontier-agent pilot

The primary GPT baseline returns a full fitted model. Submit the two-benchmark,
three-repetition GPT-5.6 array after updating the checkout. These are six new
paid provider calls; parameter values returned by GPT are evaluated directly
and are never refit by Autoformalism:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
git pull --ff-only origin main
mkdir -p logs
sbatch --array=0-5%6 \
  scripts/hpc/phase_b_raw_data_agent_fitted_model.slurm
```

After completion, inspect statuses and the exact-value development metrics:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
root=/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-v1

"$python_bin" "$repo/scripts/summarize_raw_data_agent_pilot.py" \
  --root "$root"
cat "$root/summary.csv"
find "$root" -mindepth 2 -maxdepth 2 -name agent_result.json -print0 | \
  xargs -0 -n1 jq '{output_contract, fitted_parameter_values, fit_method_summary}'
find "$root" -mindepth 2 -maxdepth 2 -name evaluation.json -print0 | \
  xargs -0 -n1 jq \
    '{parameter_source, parameter_refit_applied, training_metrics, validation_metrics}'
```

The six earlier GPT-5.6 outputs used the structure-only contract. Freeze a
fit-free scientific self-audit of their equations before any further numerical
optimization. Each candidate is duplicated into a blinded identity pair so the
judge provides two orientation-controlled readings of every absolute question;
comparative answers are not interpreted:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
source_root=/work/hdd/bibo/$USER/phase_b/raw-data-agent-pilot-v1
audit=/work/hdd/bibo/$USER/phase_b/raw-agent-scientific-audit-v1
mkdir -p "$audit"

"$python_bin" "$repo/scripts/build_raw_agent_scientific_audit_pairs.py" \
  --raw-runs-root "$source_root" \
  --data-root /projects/bibo/$USER/phase_b/inputs/public \
  --provider openai \
  --model gpt-5.6-sol \
  --output "$audit/pairs.jsonl" \
  --manifest "$audit/raw_agent_scientific_audit_manifest.json"
cp "$repo/configs/raw_data_agent_scientific_audit_v1.json" \
  "$audit/protocol_config.json"
wc -l "$audit/pairs.jsonl"
jq . "$audit/raw_agent_scientific_audit_manifest.json"
```

Expect six pairs and `evaluation_scope` equal to
`structure_only_no_parameter_fitting_by_judge`. Submit the one-seed, two-orientation
120B audit:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
gpu_account="$(accounts | awk '/gpu/ {print $1; exit}')"
sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_raw_agent_scientific_audit_120b.slurm
```

Merge exactly two responses per candidate and produce the question-level
scientific/task-compliance report:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
audit=/work/hdd/bibo/$USER/phase_b/raw-agent-scientific-audit-v1
root="$audit/gpt-oss-120b"
pair_count="$(wc -l < "$audit/pairs.jsonl")"
expected="$((2 * pair_count))"

"$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
  --inputs "$root"/shards/shard_*/hybrid_judge_scores.csv \
  --failure-inputs "$root"/shards/shard_*/hybrid_judge_failures.jsonl \
  --output "$root/hybrid_judge_scores.csv" \
  --failure-output "$root/hybrid_judge_failures.jsonl" \
  --expected "$expected"

"$python_bin" "$repo/scripts/summarize_raw_agent_scientific_audit.py" \
  --pairs "$audit/pairs.jsonl" \
  --scores "$root/hybrid_judge_scores.csv" \
  --output "$root/raw_agent_scientific_audit_summary.json"
jq . "$root/raw_agent_scientific_audit_summary.json"
```

The remaining commands treat the older structure-only candidates as secondary
ablations. They audit the hosted tool budget, apply a common evaluator, and run
an NMSE-blind cross-method scientific comparison.

### Expand the fitted GPT-5.6 baseline to all Phase-B cells

The full array contains 40 cells times three repetitions. It uses the existing
`raw-data-agent-fitted-v1` root so the six exact pilot calls are restored after
request-hash verification. Submit the API key explicitly because Delta does not
reliably preserve an implicitly inherited variable:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
git pull --ff-only origin main
: "${OPENAI_API_KEY:?OPENAI_API_KEY is unset}"
mkdir -p logs
FULL_AGENT_JOB="$(
  sbatch --parsable --export=ALL,OPENAI_API_KEY \
    scripts/hpc/phase_b_raw_data_agent_fitted_model_full.slurm
)"
echo "FULL_AGENT_JOB=$FULL_AGENT_JOB"
```

After all 120 tasks reach a terminal state, summarize numerical results, audit
the hosted tool budget offline, and freeze one identity self-pair per returned
candidate:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
runs=/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-v1
evaluation=/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-full-evaluation-v1
mkdir -p "$evaluation"

"$python_bin" "$repo/scripts/summarize_raw_data_agent_pilot.py" \
  --root "$runs"
"$python_bin" "$repo/scripts/audit_raw_data_agent_budget.py" \
  --root "$runs"
"$python_bin" "$repo/scripts/build_raw_agent_scientific_audit_pairs.py" \
  --raw-runs-root "$runs" \
  --data-root /projects/bibo/$USER/phase_b/inputs/public \
  --provider openai \
  --model gpt-5.6-sol \
  --output "$evaluation/pairs.jsonl" \
  --manifest "$evaluation/raw_agent_scientific_audit_manifest.json"

wc -l "$evaluation/pairs.jsonl"
jq . "$evaluation/raw_agent_scientific_audit_manifest.json"
```

The pair count equals the number of returned executable candidates and may be
less than 120 when the agent itself failed. Submit the four-shard 120B audit:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
gpu_account="$(accounts | awk '/gpu/ {print $1; exit}')"
FULL_AUDIT_JOB="$(
  sbatch --parsable --account="$gpu_account" --partition=gpuA40x4 \
    scripts/hpc/phase_b_raw_agent_scientific_audit_full_120b.slurm
)"
echo "FULL_AUDIT_JOB=$FULL_AUDIT_JOB"
```

Merge all successful and failed judge calls, summarize paired scientific
coverage, and create the complete development report:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
runs=/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-v1
evaluation=/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-full-evaluation-v1
judge="$evaluation/gpt-oss-120b"
pair_count="$(wc -l < "$evaluation/pairs.jsonl")"
expected="$((2 * pair_count))"

"$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
  --inputs "$judge"/shards/shard_*/hybrid_judge_scores.csv \
  --failure-inputs "$judge"/shards/shard_*/hybrid_judge_failures.jsonl \
  --output "$judge/hybrid_judge_scores.csv" \
  --failure-output "$judge/hybrid_judge_failures.jsonl" \
  --expected "$expected"

"$python_bin" "$repo/scripts/summarize_raw_agent_scientific_audit.py" \
  --pairs "$evaluation/pairs.jsonl" \
  --scores "$judge/hybrid_judge_scores.csv" \
  --output "$judge/raw_agent_scientific_audit_summary.json"

"$python_bin" "$repo/scripts/summarize_raw_data_agent_full_evaluation.py" \
  --runs-root "$runs" \
  --protocol-config "$repo/configs/raw_data_agent_fitted_model_full_v1.json" \
  --audit-manifest \
    "$evaluation/raw_agent_scientific_audit_manifest.json" \
  --audit-summary "$judge/raw_agent_scientific_audit_summary.json" \
  --tool-budget-csv "$runs/tool_budget_audit.csv" \
  --output-root "$evaluation"

cat "$evaluation/full_evaluation_summary.md"
jq . "$evaluation/full_evaluation_summary.json"
```

Audit the six completed GPT-5.6 calls directly from their cached Responses API
objects. This makes no API request:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
raw_root=/work/hdd/bibo/$USER/phase_b/raw-data-agent-pilot-v1

"$python_bin" scripts/audit_raw_data_agent_budget.py --root "$raw_root"
cat "$raw_root/tool_budget_audit.csv"
```

The 12-call value is a pilot operating point. To run the predeclared matched
24-call sensitivity for GPT-5.6 only, submit the six even array indices. Use a
new root so cached 12-call responses cannot be mistaken for 24-call responses:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
export AF_RAW_AGENT_ROOT=/work/hdd/bibo/$USER/phase_b/raw-data-agent-budget24-v1
export AF_RAW_AGENT_MAX_TOOL_CALLS=24
export AF_RAW_AGENT_PROTOCOL_CONFIG=$PWD/configs/raw_data_agent_budget_sensitivity_v1.json
sbatch --array=0,2,4,6,8,10%6 \
  scripts/hpc/phase_b_raw_data_agent_pilot.slurm
```

Apply the common fixed-RK4 warm-start screen and longer `solve_ivp` refit to all
six frozen GPT structures. A screen failure now falls back to the independent
default `solve_ivp` start instead of terminating the candidate. No LLM call is
made, and the v2 root prevents stale v1 failures from being resumed:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
sbatch --array=0-5%6 \
  scripts/hpc/phase_b_raw_data_agent_common_refit.slurm
```

Apply the identical evaluator to the easy-cell Autoformalism incumbent, then
summarize both kinds of source in the same table:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
data_root=/projects/bibo/$USER/phase_b/inputs/public
refit_root=/work/hdd/bibo/$USER/phase_b/raw-data-agent-common-refit-v2
reference=/work/hdd/bibo/$USER/phase_b/hybrid-search-smoke-v3/runs/phase_b_dalla_man_t2_canonical_named_easy_easy_seed0/summary.json

"$python_bin" "$repo/scripts/refit_raw_data_agent_candidate.py" \
  --source-summary "$reference" \
  --data-root "$data_root" \
  --protocol-config "$repo/configs/raw_data_agent_common_refit_v2.json" \
  --output-root "$refit_root"

"$python_bin" "$repo/scripts/summarize_raw_data_agent_common_refit.py" \
  --root "$refit_root"
```

The hard anonymous cell cannot yet enter a raw-agent-versus-method pair until a
frozen Autoformalism summary for that same benchmark and tier exists. Build the
three easy-cell unlabeled pairs now; the builder hides method identity and fit
metrics from the judge:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
raw_root=/work/hdd/bibo/$USER/phase_b/raw-data-agent-pilot-v1
judge_root=/work/hdd/bibo/$USER/phase_b/raw-data-agent-method-judge-v1
reference=/work/hdd/bibo/$USER/phase_b/hybrid-search-smoke-v3/runs/phase_b_dalla_man_t2_canonical_named_easy_easy_seed0/summary.json
mkdir -p "$judge_root"

"$python_bin" "$repo/scripts/build_raw_agent_method_pairs.py" \
  --raw-runs-root "$raw_root" \
  --reference-summary "$reference" \
  --data-root /projects/bibo/$USER/phase_b/inputs/public \
  --provider openai \
  --model gpt-5.6-sol \
  --output "$judge_root/pairs.jsonl" \
  --manifest "$judge_root/raw_agent_method_pairs_manifest.json"

cp "$repo/configs/raw_data_agent_mechanism_comparison_v1.json" \
  "$judge_root/protocol_config.json"
wc -l "$judge_root/pairs.jsonl"
jq . "$judge_root/raw_agent_method_pairs_manifest.json"
```

After checking that the manifest says `pair_truth: unlabeled`, submit the
one-repetition, two-orientation 120B comparison:

```bash
cd /projects/bibo/$USER/repos/autoformalism-v21
mkdir -p logs
gpu_account="$(accounts | awk '/gpu/ {print $1; exit}')"
sbatch --account="$gpu_account" --partition=gpuA40x4 \
  scripts/hpc/phase_b_raw_agent_method_judge_120b.slurm
```

Merge exactly two calls per frozen pair and create the descriptive scientific
report. Do not call the resulting preference an accuracy measurement:

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
experiment=/work/hdd/bibo/$USER/phase_b/raw-data-agent-method-judge-v1
root="$experiment/gpt-oss-120b"
pair_count="$(wc -l < "$experiment/pairs.jsonl")"
expected="$((2 * pair_count))"

"$python_bin" "$repo/scripts/merge_hybrid_scores.py" \
  --inputs "$root"/shards/shard_*/hybrid_judge_scores.csv \
  --failure-inputs "$root"/shards/shard_*/hybrid_judge_failures.jsonl \
  --output "$root/hybrid_judge_scores.csv" \
  --failure-output "$root/hybrid_judge_failures.jsonl" \
  --expected "$expected"

"$python_bin" "$repo/scripts/summarize_raw_agent_method_judge.py" \
  --pairs "$experiment/pairs.jsonl" \
  --scores "$root/hybrid_judge_scores.csv" \
  --output "$root/raw_agent_method_judge_summary.json"

jq . "$root/raw_agent_method_judge_summary.json"
```

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
