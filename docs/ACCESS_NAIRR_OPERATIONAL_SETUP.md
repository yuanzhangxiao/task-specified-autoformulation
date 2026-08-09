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
