# ACES commands for the internal rebuttal experiments

Use the Mac for file transfer and an ACES login shell for submission. The commands below use the configured `aces` SSH alias for `u.yx126462`, group scratch under `/scratch/group/p.nairr260351.000/u.yx126462`, and allocation `156264627414`. The launch archive contains a committed source snapshot, the checksummed 21-cell public release, the four corrected historical CSTR public files, the original public launch designs and the verified offline Harmony vocabulary. The existing scientific input archives retain their original `f02f708` provenance. The new search plans record the actual source and environment on ACES.

## 1. Transfer and prepare

On the Mac, transfer the supplied launch archive. It can also be rebuilt at its exact source commit with `bash scripts/hpc/package_iclr_aces.sh`.

~~~bash
cd /Users/yuanzhangxiao/Projects/autoformalism
ssh aces 'mkdir -p /scratch/group/p.nairr260351.000/u.yx126462/iclr-upload'
scp output/iclr-nonbaseline-aces.tar.gz aces:/scratch/group/p.nairr260351.000/u.yx126462/iclr-upload/
ssh aces
~~~

On ACES, unpack the code at an archive-specific path. Keep this working directory for the subsequent login-node commands. The final line prints the campaign directory.

~~~bash
export AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
export AF_JUDGE_GATE_ROOT="$AF_GROUP/judge-jetstream-known-cases-v1"
bundle="$AF_GROUP/iclr-upload/iclr-nonbaseline-aces.tar.gz"
digest=$(sha256sum "$bundle" | cut -d' ' -f1)
delivery="$AF_GROUP/iclr-deliveries/$digest"
mkdir -p "$delivery"
tar -xzf "$bundle" -C "$delivery"
cd "$delivery/iclr-nonbaseline-aces/code"
export AF_CAMPAIGN="$AF_GROUP/iclr-internal-$(cut -c1-12 SOURCE_COMMIT)"
bash scripts/hpc/iclr_aces.sh prepare
~~~

For an existing deployment that stopped at critic binding, keep the same code directory and campaign. Export the gate path above and repeat `bash scripts/hpc/iclr_aces.sh prepare`. Preparation reuses the sealed plans. The ACES folder named `judge-jetstream-calibration-v1` contains the one-pair transport test. The 140-orientation gate in `judge-jetstream-known-cases-v1` has plan digest `f8436d40f99c35eaede0c97205918c3dbe2bedca1ec773e287c791a08b60a017`, matching the original final-components authorization.

Preparation checks the delivery and public-file hashes, the original scientific designs and their public file hashes, the installed fitting packages, the H100 image and the pinned 20B model cache. It binds each plan to the previously successful 140-orientation Jetstream calibration, using `$AF_GROUP/judge-jetstream-known-cases-v1/{plan,summary}.json`. The historical environment is `/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python` with modules `GCCcore/13.2.0 Python/3.11.5`. Set `AF_JUDGE_GATE_ROOT`, `AF_HF_HOME`, `AF_VLLM_IMAGE` or `AF_PYTHON` before preparation if the verified assets have moved. An existing external installation of CasADi can be supplied via `AF_CASADI_ROOT`. If it is missing, this installs only the pinned CasADi wheel in group scratch. Keep this variable in every subsequent shell.

~~~bash
module load GCCcore/13.2.0 Python/3.11.5 WebProxy
export AF_CASADI_ROOT="$AF_GROUP/iclr-python311-casadi-3.7.2"
/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python -m pip install \
    --no-deps --target "$AF_CASADI_ROOT" 'casadi==3.7.2'
bash scripts/hpc/iclr_aces.sh prepare
~~~

## 2. Run the development pilot first

~~~bash
bash scripts/hpc/iclr_aces.sh submit pilot wave1
bash scripts/hpc/iclr_aces.sh status
~~~

Enter the Jetstream critic key at the hidden prompt. The compute job serves the pinned 20B proposer on loopback and contacts the calibrated 120B critic through the ACES WebProxy. The pilot uses one H100, public T2-hard, seed 97, Full and fixed-state arms, one shared initialization and one revision per arm. Its two endpoints are excluded from the 178 research endpoints. The scientific evaluators reject its plan.

After `squeue` shows that the array has left the queue:

~~~bash
bash scripts/hpc/iclr_aces.sh status pilot
bash scripts/hpc/iclr_aces.sh finish pilot
bash scripts/hpc/iclr_aces.sh cost pilot
cat "$AF_CAMPAIGN/pilot/pilot_receipt.json"
cat "$AF_CAMPAIGN/pilot/scheduler-cost.csv"
~~~

A time-limited slice may leave pending visits. In that case, run `bash scripts/hpc/iclr_aces.sh submit pilot wave2` and repeat the finish and cost commands after that wave. A repeated wave name returns the existing scheduler ID. A new wave submits only unfinished blocks and checks the preceding array has exited. An uncertain scheduler reply retains an intent file. Reconcile that intent with Slurm before repeating submission.

Job `2328659_0` exhausted the ACES group file-count quota during the first pilot's initial construction. Triton failed to create a compiler-cache entry, six proposer requests failed, and the worker hit the same quota at the pruning checkpoint. Both arms recorded an unavailable shared start. A later wave on that directory would reuse these visit records. Preserve the original directory and its scheduler and call ledgers. The storage retry uses a fresh campaign directory and the same sealed seed-97 pilot plan.

The launch wrapper now puts compiler caches in the compute job's temporary directory and tests a write under the durable campaign root before starting the proposer. Check the remaining group file count with `df -i "$AF_CAMPAIGN"`. After the pilot qualifies, measure the number of durable entries with `find "$AF_CAMPAIGN/pilot" -xdev -print | wc -l` and compare with the available quota before launching larger waves. [ACES job-local storage](https://hprc.tamu.edu/kb/User-Guides/ACES/Batch/) · [ACES storage quotas](https://hprc.tamu.edu/kb/User-Guides/ACES/Filesystems_and_Files/)

Job `2334957_0` received six successful Jetstream critic responses. The fixed-state revision's fit stopped with `EDQUOT` while creating `results/block0000_fixed_gamma/round_01/fit/result.json.tmp`. The pilot runtime directory accepted writes on Lustre metadata target 2 while the failed fit directory on target 0 rejected them. Use `lfs getdirstripe -m DIRECTORY` and test file creation in the directory named by the exception. The verified archives of two completed compiler jobs released 3,083 loose entries.

Once every remaining pilot proposal and its visit-level critic review have been saved, the numerical work can resume in the original campaign on one CPU allocation. Source the campaign activation file, then submit from the deployed recovery helper or a checkout containing it.

~~~bash
bash scripts/hpc/run_iclr_cached_pilot_aces.sh submit cpu-recovery1
~~~

The worker checks the sealed plan, scientific runtime, cached visit identities and actual fit and pruning directories. It disables proposer transport. A final prune that changes the selected equations can still require the Jetstream critic, so the launcher accepts the key at its hidden prompt. The original fit budget counts a started child with no final fit receipt as interrupted and retains its eligible parent. Search and pruning use the frozen selection rules. The worker runs the normal qualification audit after the block completes. Read the fixed-arm visit disposition before using any numerical result. Scheduler accounting includes the CPU wave under its own array ID.

For job `2334957_0`, a separate engineering replay at `$AF_GROUP/iclr-pilot-fit-r1-2334957` starts the fixed-state fit from its identical frozen request and warm start. The replay carries the three completed visits, all recorded proposer calls, all six successful critic responses and the two preceding array IDs. Its `engineering-fit-replay.json` records the inherited artifacts and the original interrupted optimizer files. The complete original attempt also has a verified on-cluster archive. Charge the extra bounded CPU fit to this engineering replay. The seed-97 endpoints are excluded from the research comparison.

~~~bash
campaign=/scratch/group/p.nairr260351.000/u.yx126462/iclr-pilot-fit-r1-2334957
env -u AF_JETSTREAM_API_KEY bash "$campaign/resume.sh"
source "$campaign/activate.sh"
iclr status pilot
~~~

The pilot receipt covers the sealed shared coefficients, model, initialization and critic identities, the fixed-state revision prompt, actual provider responses, pruning, pre-freeze restart and frozen export. Restart rechecks scientific and call checkpoint hashes with the proposer disabled. The GPU and token ledgers give measured costs for this small workload. Later visits, scientific rejection, serving restarts and the number of critique units change those costs.

If `finish pilot` writes `qualified=false` and returns an error, inspect `pilot_receipt.json`, `shared_initial_receipts.json`, `search_visits.csv`, `paired_ablation_results.csv`, and the Slurm job logs. A frozen pilot with no finite shared start is a recorded outcome of the bounded search. It leaves fit-dependent critique or pruning paths unexercised. A transport or configuration error, by contrast, needs diagnosis from the provider or scheduler receipts. Once the roster is frozen, another wave on that root cannot produce a new model. Retain that pilot and its cost. If a specific engineering defect is identified, document its fix and prospectively version a separate engineering study in a fresh root. Changing seeds or budgets in response to the outcome would change the study. Production checks the relevant operational coverage without imposing an NMSE threshold.

## 3. Submit the remaining searches

After the pilot qualifies, these commands submit one wave each. Anchor and stress arrays each permit two concurrent H100 blocks. The corrected historical CSTR wave separately requests two proposer GPUs and CPU critic, fitting and pruning services. For a two-GPU allocation, run these commands as successive waves, beginning with anchors. No command schedules later waves automatically.

~~~bash
bash scripts/hpc/iclr_aces.sh submit anchors wave1
bash scripts/hpc/iclr_aces.sh cstr wave1
bash scripts/hpc/iclr_aces.sh submit stress wave1
~~~

| Campaign | Blocks and endpoints | Distinct workload |
|---|---|---|
| Anchors | 6 blocks, 48 endpoints | 2 hard cells × 3 seeds × 8 treatments, 15 visits |
| Additional stress | 57 blocks, 114 endpoints | 19 cells × 3 seeds × Full/fixed, 15 visits |
| Corrected historical CSTR-easy | Original blocks 16–19, 16 endpoints | 2 prompts × 2 seeds × 4 original critic/verifier combinations |

The stress analysis reuses 12 anchor endpoints to report 126 Full/fixed rows. The corrected historical CSTR campaign also needs the original ACES `$AF_GROUP/final-components-v1` plan and public assets. It verifies them before staging a new campaign. The two original prompt variants and two original seeds are preserved.

One complete first-wave request reserves at most 72 H100-hours for anchors and 684 for stress, from 6 or 57 blocks at 12 hours each. The historical wave's two GPUs each request 6.5 hours. Actual charges follow Slurm accounting. Subsequent waves request additional time only for remaining work. The following computes a provisional proposer-call and token budget from the **actual** pilot. It applies its one independent initial construction and its mean over two revisions to every planned visit. Scientific rejections, different cells, language-model retries and cost from the separately metered critic can materially change the totals. Token estimates with unreported usage are withheld.

~~~bash
python3 - "$AF_CAMPAIGN/pilot" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1]) / "results"
def cost(arm, visit):
    return json.loads((root / f"block0000_{arm}/round_{visit:02d}/result.json").read_text())["cost"]
initial = cost("full", 0)
revisions = [cost(arm, 1) for arm in ("full", "fixed_gamma")]
for lane, independent_starts, revision_visits in (("anchors", 42, 672), ("stress", 57, 1596)):
    for key in ("physical_requests", "observed_tokens"):
        if key == "observed_tokens" and any(c.get("requests_with_unknown_usage") for c in [initial, *revisions]):
            print(lane, key, "unavailable")
            continue
        mean_revision = sum(c.get(key, 0) for c in revisions) / 2
        print(lane, key, round(independent_starts * initial.get(key, 0) + revision_visits * mean_revision))
PY
~~~

Inspect coverage **between** waves. Use a new wave name for each additional allocation on the same root.

~~~bash
bash scripts/hpc/iclr_aces.sh status
bash scripts/hpc/iclr_aces.sh status anchors
bash scripts/hpc/iclr_aces.sh status stress
bash scripts/hpc/iclr_aces.sh status cstr
~~~

For lanes with pending work after their previous workers exit:

~~~bash
bash scripts/hpc/iclr_aces.sh submit anchors wave2
bash scripts/hpc/iclr_aces.sh submit stress wave2
bash scripts/hpc/iclr_aces.sh cstr wave2
~~~

Use `wave3` and subsequent wave names as needed. When each planned roster is terminal, freeze it and collect the records. Terminal scientific failures are retained. Freeze refuses pending work.

~~~bash
bash scripts/hpc/iclr_aces.sh finish anchors
bash scripts/hpc/iclr_aces.sh cost anchors
bash scripts/hpc/iclr_aces.sh finish stress
bash scripts/hpc/iclr_aces.sh cost stress
bash scripts/hpc/iclr_aces.sh finish cstr
~~~

Each production root contains the complete roster, `endpoint-audit/`, `search_visits.csv`, `paired_ablation_results.csv`, `spec_sensitivity.csv`, `frozen_models.json`, `evaluation_freeze.json` and `scheduler-cost.csv`. The paired receipts detect a different copied model, training fit or critic. Slurm array accounting includes each recovery allocation once. The per-arm search costs are also recorded in the tables for the same plan and block. Capture the separate historical campaign's scheduler ledger on ACES:

~~~bash
cstr_ids=$(cat "$AF_CAMPAIGN"/historical-cstr/submissions/*/*.id | sort -u | paste -sd, -)
sacct --array -j "$cstr_ids" -P --format=JobID%45,ElapsedRaw,TotalCPU,AllocCPUS,AllocTRES%120,State \
    > "$AF_CAMPAIGN/historical-cstr/sacct.psv"
~~~

## 4. Score the frozen endpoints on ACES CPUs

On the Mac, copy the original analyst ZIP to the existing upload directory. Scoring verifies its fixed SHA-256. Use `iclr-nonbaseline-evidence.zip` from the original delivery.

~~~bash
cd /Users/yuanzhangxiao/Projects/autoformalism
scp output/iclr-nonbaseline-evidence.zip aces:/scratch/group/p.nairr260351.000/u.yx126462/iclr-upload/
~~~

On ACES, run these commands from the same deployed code directory. Each lane can be scored immediately after that lane freezes. Score anchors before stress when prioritizing the latent-state result. If reviewed physical correspondences for new model hashes are available, set `AF_READOUTS` to that evaluator-only map **before** the first anchor scoring run. The fidelity plan binds the map and selection before generating the confirmation trajectories. With no map, direct learned physical quantities remain explicitly unavailable; the training-calibrated probes and both generated targets still run.

Scoring checks for NumPy's `trapezoid` (NumPy 2+) and exercises Matplotlib's headless renderer for anchors and combined figures. If the ACES evaluator environment lacks these dependencies, the following installs the project-pinned versions and Matplotlib's dependencies in a separate directory. Retain `AF_EVAL_DEPS` in the scoring shell. Discovery uses the original fitting environment.

~~~bash
module load GCCcore/13.2.0 Python/3.11.5 WebProxy
export AF_EVAL_DEPS="$AF_GROUP/iclr-evaluator-py311"
/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python -m pip install \
    --target "$AF_EVAL_DEPS" 'numpy==2.3.5' 'matplotlib==3.10.8'
~~~

~~~bash
srun --account=156264627414 --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --mem=32G --time=12:00:00 --export=ALL bash "$PWD/scripts/hpc/iclr_aces.sh" score anchors
srun --account=156264627414 --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --mem=32G --time=12:00:00 --export=ALL bash "$PWD/scripts/hpc/iclr_aces.sh" score cstr
srun --account=156264627414 --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --mem=32G --time=12:00:00 --export=ALL bash "$PWD/scripts/hpc/iclr_aces.sh" score stress
bash scripts/hpc/iclr_aces.sh combine
~~~

Repeat the same scoring command on the same paths after a time limit. The evaluators preserve recorded outcomes and explicit interrupted or unavailable statuses. The result directories are under `$AF_CAMPAIGN/confirmation`, including `fidelity`, `anchor-comparison`, `anchor-clamps`, `specification`, `corrected-cstr-test`, `corrected-cstr-compliance`, `acquisition-anchors`, `acquisition-stress` and `stress-report`. The anchor clamp analysis measures the influence of the selected models' joint hidden coordinates under two interventions per hard cell.

## Historical archive recovery

To check whether the original ACES campaign already has final files, run this in the ACES login shell. A nonempty evaluation file can still have missing subjects. Inspect its status counts before using any score. The newer freeze and six-visit submission snapshot are distinct cohorts.

~~~bash
original="$AF_GROUP/final-components-v1"
for f in plan.json evaluation_freeze.json subjects.jsonl component_evaluation.json; do
    if [[ -f "$original/$f" ]]; then sha256sum "$original/$f"; else echo "Missing $f"; fi
done
if [[ -s "$original/component_evaluation.json" ]]; then
    jq '{planned_endpoints, counts: (.rows | group_by(.status) | map({status: .[0].status, n: length}))}' \
      "$original/component_evaluation.json"
fi
~~~

For read-only assessment of up to all nine archived cases, run `bash scripts/hpc/iclr_aces.sh archived` from the same checkout. This CPU array reads the original campaign, records every recoverable visit and runs the public functional evaluator. It makes zero provider calls. The results reside under `$AF_CAMPAIGN/archived-assessments`. Later pruning and test outcomes retain their own provenance. The September report remains a six-visit validation snapshot. [Historical recovery and merge instructions](ICLR_2027_NONBASELINE_RUNBOOK.md)

Each search or recovery wave writes its submitted job IDs. If needed, an operator can inspect individual task states and measured scheduler costs using Slurm's native tools. [Slurm array job IDs](https://slurm.schedmd.com/squeue.html) · [Slurm accounting](https://slurm.schedmd.com/sacct.html)
