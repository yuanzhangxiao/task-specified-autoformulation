#!/bin/bash
# CPU preparation, one warm proposer server, then a protocol-specific CPU audit or fit.
set -euo pipefail
: "${AF_REPO_ROOT:?required}" "${AF_PYTHON:?required}" "${AF_OUTPUT_ROOT:?required}"
mode="${1:?expected prepare, construct, audit, or fit}"
case "$mode" in prepare|construct|audit|fit) ;; *) echo 'Unknown prefit worker mode' >&2; exit 2 ;; esac
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$AF_REPO_ROOT"
mkdir -p "$AF_OUTPUT_ROOT/runtime" "$AF_OUTPUT_ROOT/tmp"
export TMPDIR="$AF_OUTPUT_ROOT/tmp"
"$AF_PYTHON" scripts/prefit_construction_campaign.py verify --root "$AF_OUTPUT_ROOT"
protocol="$(jq -er '.config.protocol' "$AF_OUTPUT_ROOT/plan.json")"
case "$mode" in
  prepare)
    : "${AF_VLLM_IMAGE:?required}" "${AF_HF_HOME:?required}"
    module load WebProxy
    runtime="$(command -v apptainer || command -v singularity)"
    expected="$(jq -er '.config.serving_image_sha256' "$AF_OUTPUT_ROOT/plan.json")"
    actual="$(sha256sum "$AF_VLLM_IMAGE")"
    [[ "${actual%% *}" == "$expected" ]] || { echo 'Frozen image SHA differs' >&2; exit 2; }
    printf '%s\n' "$actual" > "$AF_OUTPUT_ROOT/runtime/image-${SLURM_JOB_ID}.sha256"
    if [[ "$protocol" == prefit-construction-audit-1 ]]; then
      tests=(tests/test_prefit_construction_audit.py tests/test_prefit_aces_submission.py tests/test_training_evidence.py tests/test_staged_topology_runner.py tests/test_staged_function_runner.py)
      smoke=scripts/smoke_prefit_construction_audit.py
    else
      tests=(tests/test_prefit_construction_campaign.py tests/test_prefit_aces_submission.py tests/test_training_evidence.py tests/test_causal_initialization_construction.py tests/test_staged_topology_runner.py tests/test_staged_function_runner.py tests/test_staged_function_prefit_campaign.py)
      smoke=scripts/smoke_prefit_construction.py
    fi
    preflight_log="$AF_OUTPUT_ROOT/runtime/preflight-${SLURM_JOB_ID}.log"
    printf 'Running preflight tests; full output: %s\n' "$preflight_log"
    if "$AF_PYTHON" -m pytest -q -p no:cacheprovider "${tests[@]}" > "$preflight_log" 2>&1; then
      tail -n 2 "$preflight_log"
    else
      preflight_status=$?
      tail -n 80 "$preflight_log" >&2
      exit "$preflight_status"
    fi
    "$AF_PYTHON" "$smoke" > "$AF_OUTPUT_ROOT/runtime/smoke-${SLURM_JOB_ID}.json"
    container_python="$("$runtime" exec "$AF_VLLM_IMAGE" sh -c 'command -v python3 || command -v python')"
    [[ "$container_python" == /* && "$container_python" != *$'\n'* ]] || { echo 'Invalid container Python path' >&2; exit 2; }
    model="$(jq -er '.config.model_settings.model' "$AF_OUTPUT_ROOT/plan.json")"
    revision="$(jq -er '.config.model_settings.model_revision' "$AF_OUTPUT_ROOT/plan.json")"
    "$runtime" exec --bind "$AF_HF_HOME:$AF_HF_HOME" --env "HF_HOME=$AF_HF_HOME" "$AF_VLLM_IMAGE" "$container_python" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2])' "$model" "$revision"
    ;;
  construct)
    module load WebProxy
    exec bash scripts/hpc/run_staged_topology_server.sh
    ;;
  audit)
    "$AF_PYTHON" scripts/prefit_construction_campaign.py audit --root "$AF_OUTPUT_ROOT" --wall-seconds 1500 > "$AF_OUTPUT_ROOT/runtime/audit-${SLURM_JOB_ID}.json"
    ;;
  fit)
    "$AF_PYTHON" scripts/prefit_construction_campaign.py fit --root "$AF_OUTPUT_ROOT" --wall-seconds 6900 > "$AF_OUTPUT_ROOT/runtime/fit-${SLURM_JOB_ID}.json"
    ;;
esac
