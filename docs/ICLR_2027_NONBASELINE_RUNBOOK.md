# Running the internal rebuttal campaigns

Two ZIPs accompany this runbook. Use **iclr-nonbaseline-public.zip** for discovery. It contains the public specification, prompt, training and validation observations for each of 21 cells and a roster with their file hashes. **iclr-nonbaseline-evidence.zip** is for the analyst. It includes simulator-facing files, development results and the reference fits. Keep the analyst archive outside the discovery input directory. The checkout and numerical runtime used for each search are sealed on the execution host.

## Prepare and submit on ACES

Use the Git commit recorded in `source-commit.txt` inside the public ZIP. In the following commands, point `AF_REPO_ROOT` at that checkout, `AF_UPLOAD` at the directory containing the ZIPs, and `AF_ACCOUNT` at your ACES GPU allocation. Set the two paths if your historical interpreter or calibration receipt has moved. No provider requests occur during preparation.

~~~bash
module load GCCcore/13.2.0 Python/3.11.5
export AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export AF_INPUTS="$AF_GROUP/iclr-public-study-1"
export AF_CAMPAIGN="$AF_GROUP/iclr-nonbaseline-study-1"
export AF_JUDGE_GATE_ROOT="$AF_GROUP/judge-jetstream-calibration-v1"
mkdir -p "$AF_INPUTS" "$AF_CAMPAIGN"
unzip -oq "$AF_UPLOAD/iclr-nonbaseline-public.zip" -d "$AF_INPUTS"
(cd "$AF_INPUTS" && sha256sum -c SHA256SUMS --quiet)
cd "$AF_REPO_ROOT"
test "$(git rev-parse HEAD)" = "$(cat "$AF_INPUTS/source-commit.txt")"
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
bash scripts/hpc/prepare_iclr_nonbaseline.sh
for lane in anchors stress; do
  "$AF_PYTHON" -m scripts.iclr_controlled_search bind-critic \
    --root "$AF_CAMPAIGN/$lane" \
    --calibration-plan "$AF_JUDGE_GATE_ROOT/plan.json" \
    --calibration-summary "$AF_JUDGE_GATE_ROOT/summary.json"
done
read -rsp 'Jetstream critic key: ' AF_JETSTREAM_API_KEY; echo
export AF_JETSTREAM_API_KEY
export AF_SEARCH_ROOT="$AF_CAMPAIGN/anchors"
sbatch --account="$AF_ACCOUNT" --partition=gpu --array=0-5%2 --export=ALL \
  scripts/hpc/run_iclr_controlled_search_aces.sh
export AF_SEARCH_ROOT="$AF_CAMPAIGN/stress"
sbatch --account="$AF_ACCOUNT" --partition=gpu --array=0-56%2 --export=ALL \
  scripts/hpc/run_iclr_controlled_search_aces.sh
~~~

Each active block uses one H100 and starts the pinned proposer on its compute node. These are 63 **cell/seed blocks**, covering 162 lineages with up to 15 visits each. The two arrays above permit two concurrent blocks apiece. For a smaller first wave, submit anchor block 0, inspect its cost/elapsed time, then submit the rest. A slice can finish its time allocation with work pending. After the prior job exits, resubmit unfinished block indices using the same root. A completed search/fitting stage retains its original call/fit receipts. See the [paired search runbook](ICLR_2027_CONTROLLED_SEARCH.md) for treatment definitions, budgets and source checks. If an equal-parameter sensitivity is needed, the optional `--max-parameters` setting applies the same cap to both search arms. Freeze that separate study from development evidence before examining its confirmation outcomes.

Between waves, report coverage; after all planned rows finish, freeze the endpoints. The export refuses pending rows. Search and evaluation use the same parent directory for each plan, its freeze receipt, and its frozen model bundle.

~~~bash
for lane in anchors stress; do
  "$AF_PYTHON" -m scripts.iclr_controlled_search report --root "$AF_CAMPAIGN/$lane"
  "$AF_PYTHON" -m scripts.iclr_controlled_search export --root "$AF_CAMPAIGN/$lane"
done
~~~

## Score the returned models

Use a CPU allocation on ACES or Delta. If moving between systems, copy each lane's `plan.json`, `spec_variants.json`, `evaluation_freeze.json`, `frozen_models.json`, and search tables. Export `AF_CAMPAIGN` to that parent, set `AF_PYTHON` and `AF_REPO_ROOT` for the local environment, and run from the same source checkout. Only the evaluator reads private reference files. Use a fresh `AF_EVAL` directory. Unpack the public ZIP again into `$AF_EVIDENCE/stress` so its public cell files sit next to the matching evaluator manifest. The numerical dependencies are those used by the project fitting pipeline, including CasADi for the supplied-structure fits.

~~~bash
export AF_EVIDENCE=/path/to/analyst-evidence
export AF_EVAL=/path/to/fresh-confirmation-output
mkdir -p "$AF_EVIDENCE" "$AF_EVAL"
unzip -oq "$AF_UPLOAD/iclr-nonbaseline-evidence.zip" -d "$AF_EVIDENCE"
(cd "$AF_EVIDENCE" && sha256sum -c SHA256SUMS --quiet)
unzip -oq "$AF_UPLOAD/iclr-nonbaseline-public.zip" 'public/*' -d "$AF_EVIDENCE/stress"
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$AF_REPO_ROOT"
"$AF_PYTHON" -m scripts.iclr_anchor_fidelity prepare --mode confirmation \
  --models "$AF_CAMPAIGN/anchors/frozen_models.json" \
  --models "$AF_EVIDENCE/reference/frozen_models.json" \
  --public-root "$AF_EVIDENCE/phase-c-public" --stress-public-root "$AF_EVIDENCE/stress/public" \
  --data-root "$AF_EVIDENCE/evaluator-data" --root "$AF_EVAL/fidelity"
"$AF_PYTHON" -m scripts.iclr_anchor_fidelity run --root "$AF_EVAL/fidelity" \
  --data-root "$AF_EVIDENCE/evaluator-data" --workers 4
"$AF_PYTHON" -m scripts.plot_iclr_anchor_fidelity --root "$AF_EVAL/fidelity"
"$AF_PYTHON" -m scripts.iclr_specification_results --search "$AF_CAMPAIGN/anchors" \
  --public "$AF_EVIDENCE/stress/public" --fidelity "$AF_EVAL/fidelity" --output "$AF_EVAL/specification"
for lane in anchors stress; do
  if [[ "$lane" == anchors ]]; then
    reference_root="$AF_EVIDENCE/reference"
  else
    reference_root="$AF_EVIDENCE/stress/conditioned-references"
  fi
  "$AF_PYTHON" -m scripts.iclr_stress evaluate --root "$AF_EVIDENCE/stress" \
    --models "$AF_CAMPAIGN/$lane/frozen_models.json" --data-root "$AF_EVIDENCE/evaluator-data" \
    --reference-root "$reference_root" --output "$AF_EVAL/acquisition-$lane"
done
"$AF_PYTHON" -m scripts.iclr_stress report --root "$AF_EVIDENCE/stress" \
  --evaluations "$AF_EVAL/acquisition-anchors/evaluation.json" \
    "$AF_EVAL/acquisition-stress/evaluation.json" \
  --output "$AF_EVAL/stress-report"
~~~

For declared physical readouts in candidate coordinates, supply a reviewed expression, unit and evidence through `--readouts` at fidelity preparation time. Unmapped physical quantities retain their explicit status. More details on matched predictions, multistart diagnostics, noise variants, scheduler CPU/GPU cost collection, and independent reference checks appear in [reference fitting and interventions](ICLR_2027_REFERENCE_AND_INTERVENTIONS.md) and [stress studies](ICLR_2027_STRESS_STUDIES.md).

## Historical submission follow-up

The online component methods were trained on the historical CSTR data. To run its four submitted critic/verifier arms at both prompt variants and both original seeds on the corrected **easy** cell, use the original campaign at `AF_ORIGINAL` plus the four corrected public CSTR files in the analyst archive. The helper verifies the pinned original plan and original public-file identities, preserves the original prompt, and writes a separate development tree. Its 16 lineages are a distinct experiment from the new Full/fixed-representation comparison.

~~~bash
export AF_ORIGINAL="$AF_GROUP/final-components-v1"
export AF_CSTR_PUBLIC="$AF_GROUP/iclr-corrected-historical-public"
export AF_CSTR_RUN="$AF_GROUP/iclr-corrected-historical-cstr"
export AF_GPU_ACCOUNT="$AF_ACCOUNT" AF_CPU_ACCOUNT="${AF_CPU_ACCOUNT:-$AF_ACCOUNT}"
export AF_VLLM_IMAGE="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
export AF_HF_HOME="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
export AF_COMPUTE_CACHE_ROOT="$AF_GROUP/component-runtime-cache"
export AF_IPC_TMP_ROOT=/tmp/af-ipc-u.yx126462
"$AF_PYTHON" -m scripts.iclr_cstr_correction --original "$AF_ORIGINAL" \
  --corrected "$AF_EVIDENCE/corrected-easy" --output "$AF_CSTR_PUBLIC"
"$AF_PYTHON" scripts/component_campaign.py prepare --root "$AF_CSTR_RUN" \
  --public-root "$AF_CSTR_PUBLIC" --blocks 16,17,18,19
"$AF_PYTHON" scripts/component_campaign.py authorize-critic --root "$AF_CSTR_RUN" \
  --calibration-plan "$AF_JUDGE_GATE_ROOT/plan.json" --calibration-summary "$AF_JUDGE_GATE_ROOT/summary.json"
"$AF_PYTHON" scripts/submit_component_campaign.py --root "$AF_CSTR_RUN" \
  --wave wave1 --proposers 2 --critics 1 --fits 8 --pruning 2
~~~

Use the native component report to identify unfinished work. When a wave ends with tasks pending, submit the next wave with a new wave name against the same root. After all 16 final endpoints have a disposition, freeze them and evaluate the corrected held-out trajectories on a CPU allocation. The local development-only check reproduced all 16 training and four validation trajectories byte for byte.

~~~bash
"$AF_PYTHON" scripts/component_campaign.py report --root "$AF_CSTR_RUN"
"$AF_PYTHON" scripts/component_campaign.py export --root "$AF_CSTR_RUN"
"$AF_PYTHON" -m scripts.iclr_corrected_cstr_test evaluate \
  --root "$AF_CSTR_RUN" --staging "$AF_CSTR_PUBLIC" \
  --data-root "$AF_EVIDENCE/evaluator-data" --output "$AF_EVAL/corrected-cstr-test"
~~~

To extend the historical learning curves and structural-stability analysis to all nine cases, export the archived incumbents and rescore their public functional requirements on ACES. The shell worker reads recorded visits and public development data only. No LLM calls or refitting occur. A later model cannot fill a missing row in the September submission snapshot.

~~~bash
export AF_ASSESSMENTS="$AF_GROUP/iclr-historic-visit-assessments"
sbatch --account="$AF_ACCOUNT" --partition=cpu --cpus-per-task=4 --mem=16G \
  --time=12:00:00 --array=0-14%3 --export=ALL scripts/hpc/run_iclr_archived_mechanisms.sh
~~~

Bring the resulting `visit_*/functional/{plan.json,results}` and `visit_05/models/models.json` back to the analysis checkout. Use `python -m scripts.iclr_search_history --complete-bundle "$AF_ASSESSMENTS/visit_05/models/models.json" --functional-root "$AF_ASSESSMENTS" --output /path/to/new-history`. Each returned score must match the original retained model's source receipt. For exact original T2-hard discovery, point `iclr_controlled_search prepare` at `"$AF_ORIGINAL/public/phase_b_v1"` and select `phase_b_dalla_man_t2_canonical_named_hard`. Use a separate search root and record the original prompt and development hashes.
