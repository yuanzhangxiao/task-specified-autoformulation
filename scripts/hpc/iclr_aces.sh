#!/usr/bin/env bash
# Login-node commands for the internal ICLR studies. Scoring runs on a CPU allocation.
set -euo pipefail
umask 077
export AF_REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
export AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
export AF_DELIVERY=${AF_DELIVERY:-$(dirname "$AF_REPO_ROOT")}
export AF_COMMIT=$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")
export AF_ACCOUNT=${AF_ACCOUNT:-156264627414}
export AF_GPU_ACCOUNT=${AF_GPU_ACCOUNT:-$AF_ACCOUNT} AF_CPU_ACCOUNT=${AF_CPU_ACCOUNT:-$AF_ACCOUNT}
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export AF_CAMPAIGN=${AF_CAMPAIGN:-$AF_GROUP/iclr-internal-${AF_COMMIT:0:12}}
export AF_INPUTS="$AF_CAMPAIGN/public-inputs"
export AF_HARMONY_ASSETS="$AF_REPO_ROOT/assets/harmony"
export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
export AF_JUDGE_GATE_ROOT=${AF_JUDGE_GATE_ROOT:-$AF_GROUP/judge-jetstream-calibration-v1}
export AF_COMPUTE_CACHE_ROOT=${AF_COMPUTE_CACHE_ROOT:-$AF_GROUP/component-runtime-cache}
export AF_IPC_TMP_ROOT=${AF_IPC_TMP_ROOT:-/tmp/af-ipc-u.yx126462}
snap=hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/config.json
if [[ -z ${AF_HF_HOME:-} ]]; then
    export AF_HF_HOME=/scratch/user/u.yx126462/huggingface-cache
    if [[ ! -s "$AF_HF_HOME/$snap" && -s "$AF_GROUP/huggingface-cache/$snap" ]]; then
        export AF_HF_HOME="$AF_GROUP/huggingface-cache"
    fi
fi
export AF_ORIGINAL=${AF_ORIGINAL:-$AF_GROUP/final-components-v1}
export AF_CSTR_PUBLIC="$AF_CAMPAIGN/historical-cstr-public" AF_CSTR_RUN="$AF_CAMPAIGN/historical-cstr"
export AF_EVIDENCE="$AF_CAMPAIGN/analyst" AF_EVAL="$AF_CAMPAIGN/confirmation"
export AF_ASSESSMENTS="$AF_CAMPAIGN/archived-assessments"
cd "$AF_REPO_ROOT"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { "$AF_PYTHON" -m "scripts.$1" "${@:2}"; }
evaluation_env() {
    export MPLBACKEND=Agg
    export PYTHONPATH="$PYTHONPATH${AF_EVAL_DEPS:+:$AF_EVAL_DEPS}"
    "$AF_PYTHON" - "$1" <<'PY'
import sys
import numpy as np
if not callable(getattr(np, 'trapezoid', None)):
    raise SystemExit('Scoring needs NumPy with np.trapezoid. See the evaluator install in the ACES runbook.')
if sys.argv[1] in ('anchors', 'combine'):
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        FigureCanvasAgg(Figure()).draw()
    except ImportError as exc:
        raise SystemExit('Install the Matplotlib evaluator dependencies using the ACES runbook.') from exc
    print('Scoring runtime: NumPy', np.__version__, 'Matplotlib', matplotlib.__version__)
PY
}
bind() {
    run iclr_controlled_search bind-critic --root "$1" --calibration-plan "$AF_JUDGE_GATE_ROOT/plan.json" \
        --calibration-summary "$AF_JUDGE_GATE_ROOT/summary.json" >/dev/null
}
credential() {
    if [[ -z ${AF_JETSTREAM_API_KEY:-} ]]; then
        read -r -s -p 'Jetstream critic key: ' AF_JETSTREAM_API_KEY </dev/tty
        printf '\n' >/dev/tty
        export AF_JETSTREAM_API_KEY
    fi
    : "${AF_JETSTREAM_API_KEY:?missing Jetstream credential}"
}
qualify() {
    "$AF_PYTHON" - "$AF_CAMPAIGN/pilot" <<'PY'
from pathlib import Path
import sys
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read
root = Path(sys.argv[1])
r = sealed_read(root / 'pilot_receipt.json')
if not r['qualified'] or r['plan_sha256'] != sealed_read(root / 'plan.json')['artifact_sha256']:
    raise ValueError('pilot has not exercised the full live execution path; inspect its receipts')
if r['source_sha256'] != public._source_identity() or r['runtime'] != public._runtime():
    raise ValueError('source or runtime changed since the live pilot')
PY
}
lane_name() { [[ "$1" == anchors || "$1" == stress || "$1" == pilot ]]; }
case "${1:-}" in
    prepare)
        (cd "$AF_DELIVERY" && sha256sum --quiet -c SHA256SUMS)
        test -s "$AF_VLLM_IMAGE" && test -s "$AF_HF_HOME/$snap" || { echo 'Missing pinned model or image; check AF_HF_HOME and AF_VLLM_IMAGE' >&2; exit 1; }
        "$AF_PYTHON" -c 'import casadi, scipy, pandas, pydantic, numpy; print("local fitting runtime", casadi.__version__, scipy.__version__, numpy.__version__)'
        mkdir -p "$AF_INPUTS"
        unzip -oq "$AF_DELIVERY/inputs/iclr-nonbaseline-public.zip" -d "$AF_INPUTS"
        (cd "$AF_INPUTS" && sha256sum --quiet -c SHA256SUMS)
        bash scripts/hpc/prepare_iclr_nonbaseline.sh
        for lane in anchors stress; do
            run iclr_aces_pilot check-design --root "$AF_CAMPAIGN/$lane" --original "$AF_DELIVERY/inputs/$lane-design.json" --roster "$AF_INPUTS/roster.json"
        done
        run iclr_aces_pilot prepare --root "$AF_CAMPAIGN/pilot" --public "$AF_INPUTS/public"
        for lane in pilot anchors stress; do bind "$AF_CAMPAIGN/$lane"; done
        printf 'Campaign %s\n' "$AF_CAMPAIGN"
        ;;
    submit)
        lane=${2:?pilot, anchors or stress}; lane_name "$lane" || exit 2
        limit=2
        if [[ "$lane" == pilot ]]; then limit=1; else qualify; fi
        credential
        run iclr_aces_pilot submit --root "$AF_CAMPAIGN/$lane" --wave "${3:?wave name}" --account "$AF_GPU_ACCOUNT" --concurrency "$limit"
        ;;
    status)
        squeue -u "$USER" -o '%.18i %.18j %.8T %.10M %R'
        if [[ -n ${2:-} ]]; then
            if [[ "$2" == cstr ]]; then run component_campaign report --root "$AF_CSTR_RUN"
            else lane_name "$2" || exit 2; run iclr_controlled_search report --root "$AF_CAMPAIGN/$2"; fi
        fi
        ;;
    finish)
        lane=${2:?pilot, anchors, stress or cstr}
        case "$lane" in
            pilot) unset AF_JETSTREAM_API_KEY; run iclr_aces_pilot finish --root "$AF_CAMPAIGN/pilot" ;;
            anchors|stress)
                run iclr_controlled_search export --root "$AF_CAMPAIGN/$lane"
                run iclr_controlled_receipts --root "$AF_CAMPAIGN/$lane" --output "$AF_CAMPAIGN/$lane/endpoint-audit" ;;
            cstr) run component_campaign export --root "$AF_CSTR_RUN" ;;
            *) exit 2 ;;
        esac
        ;;
    cstr)
        qualify
        run iclr_cstr_correction --original "$AF_ORIGINAL" --corrected "$AF_DELIVERY/inputs/corrected-easy" --output "$AF_CSTR_PUBLIC"
        run component_campaign prepare --root "$AF_CSTR_RUN" --public-root "$AF_CSTR_PUBLIC" --blocks 16,17,18,19
        run component_campaign authorize-critic --root "$AF_CSTR_RUN" \
            --calibration-plan "$AF_JUDGE_GATE_ROOT/plan.json" --calibration-summary "$AF_JUDGE_GATE_ROOT/summary.json"
        credential
        # The historical GPU launcher inherits these two offline tokenizer settings.
        export APPTAINERENV_TIKTOKEN_ENCODINGS_BASE="$AF_HARMONY_ASSETS" SINGULARITYENV_TIKTOKEN_ENCODINGS_BASE="$AF_HARMONY_ASSETS"
        export APPTAINER_BINDPATH="$AF_HARMONY_ASSETS:$AF_HARMONY_ASSETS:ro${APPTAINER_BINDPATH:+,$APPTAINER_BINDPATH}"
        export SINGULARITY_BINDPATH="$AF_HARMONY_ASSETS:$AF_HARMONY_ASSETS:ro${SINGULARITY_BINDPATH:+,$SINGULARITY_BINDPATH}"
        run submit_component_campaign --root "$AF_CSTR_RUN" --wave "${2:?wave name}" --proposers 2 --critics 1 --fits 8 --pruning 2
        ;;
    cost)
        lane=${2:?pilot, anchors or stress}; lane_name "$lane" || exit 2
        root="$AF_CAMPAIGN/$lane"
        shopt -s nullglob
        files=("$root"/submissions/*/array.id)
        [[ ${#files[@]} -gt 0 ]] || { echo 'No confirmed array submissions' >&2; exit 1; }
        ids=$(cat "${files[@]}" | sort -u | paste -sd, -)
        sacct --array -j "$ids" -P --format=JobID%45,ElapsedRaw,TotalCPU,AllocCPUS,AllocTRES%120,State >"$root/sacct.psv"
        run iclr_stress_resources --plan "$root/plan.json" --sacct "$root/sacct.psv" --output "$root/scheduler-cost.csv"
        ;;
    score)
        [[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run score in the documented CPU allocation' >&2; exit 2; }
        lane=${2:?anchors, stress or cstr}
        unset AF_JETSTREAM_API_KEY
        evaluation_env "$lane"
        case "$lane" in
            anchors|stress) test -s "$AF_CAMPAIGN/$lane/evaluation_freeze.json" && test -s "$AF_CAMPAIGN/$lane/frozen_models.json" || { echo 'Freeze the lane before scoring' >&2; exit 1; } ;;
            cstr) test -s "$AF_CSTR_RUN/evaluation_freeze.json" ;;
            *) exit 2 ;;
        esac
        analyst_zip=${AF_ANALYST_ZIP:-"$AF_GROUP/iclr-upload/iclr-nonbaseline-evidence.zip"}
        echo "7e1a64fe6ef8e9c93f3521f8e513edc2aac14f2b8144bcb081f2d53dca315663  $analyst_zip" | sha256sum --check --quiet
        mkdir -p "$AF_EVIDENCE" "$AF_EVAL"
        (
          flock -x 9
          unzip -oq "$analyst_zip" -d "$AF_EVIDENCE"
          (cd "$AF_EVIDENCE" && sha256sum --quiet -c SHA256SUMS)
          unzip -oq "$AF_DELIVERY/inputs/iclr-nonbaseline-public.zip" 'public/*' -d "$AF_EVIDENCE/stress"
        ) 9>"$AF_EVIDENCE/.unpack.lock"
        if [[ "$lane" == cstr ]]; then
            run iclr_corrected_cstr_test evaluate --root "$AF_CSTR_RUN" --staging "$AF_CSTR_PUBLIC" --data-root "$AF_EVIDENCE/evaluator-data" --output "$AF_EVAL/corrected-cstr-test"
            run iclr_corrected_cstr_compliance --root "$AF_CSTR_RUN" --staging "$AF_CSTR_PUBLIC" --output "$AF_EVAL/corrected-cstr-compliance"
            exit
        fi
        if [[ "$lane" == anchors ]]; then
            ref="$AF_EVIDENCE/reference"
            readouts=()
            if [[ -n ${AF_READOUTS:-} ]]; then readouts+=(--readouts "$AF_READOUTS"); fi
            run iclr_anchor_fidelity prepare --mode confirmation --models "$AF_CAMPAIGN/anchors/frozen_models.json" \
                --models "$ref/frozen_models.json" --public-root "$AF_EVIDENCE/phase-c-public" --stress-public-root "$AF_EVIDENCE/stress/public" \
                --data-root "$AF_EVIDENCE/evaluator-data" --root "$AF_EVAL/fidelity" "${readouts[@]}"
            run iclr_anchor_fidelity run --data-root "$AF_EVIDENCE/evaluator-data" --root "$AF_EVAL/fidelity" --workers "${SLURM_CPUS_PER_TASK:-1}"
            run plot_iclr_anchor_fidelity --root "$AF_EVAL/fidelity"
            run iclr_specification_results --search "$AF_CAMPAIGN/anchors" --public "$AF_EVIDENCE/stress/public" --fidelity "$AF_EVAL/fidelity" --output "$AF_EVAL/specification"
            run iclr_equation_examples --search "$AF_CAMPAIGN/anchors" --output "$AF_EVAL/specification/paired-equations-seed0.md"
            run iclr_anchor_comparison --search "$AF_CAMPAIGN/anchors" --fidelity "$AF_EVAL/fidelity" --output "$AF_EVAL/anchor-comparison"
            run iclr_latent_clamp --fidelity "$AF_EVAL/fidelity" --search "$AF_CAMPAIGN/anchors" \
                --output "$AF_EVAL/anchor-clamps" --workers "${SLURM_CPUS_PER_TASK:-1}"
        else
            ref="$AF_EVIDENCE/stress/conditioned-references"
        fi
        run iclr_stress evaluate --root "$AF_EVIDENCE/stress" --models "$AF_CAMPAIGN/$lane/frozen_models.json" \
            --reference-root "$ref" --data-root "$AF_EVIDENCE/evaluator-data" --output "$AF_EVAL/acquisition-$lane"
        ;;
    combine)
        evaluation_env combine
        run iclr_stress report --root "$AF_EVIDENCE/stress" --evaluations "$AF_EVAL/acquisition-anchors/evaluation.json" \
            "$AF_EVAL/acquisition-stress/evaluation.json" --output "$AF_EVAL/stress-report"
        ;;
    archived)
        sbatch --parsable --account="$AF_CPU_ACCOUNT" --partition=cpu --cpus-per-task=4 --mem=16G \
            --time=12:00:00 --array=0-14%3 --export=ALL scripts/hpc/run_iclr_archived_mechanisms.sh
        ;;
    *) echo 'Use prepare | submit LANE WAVE | status [LANE] | finish LANE | cstr WAVE | cost LANE | score LANE | combine | archived' >&2; exit 2 ;;
esac
