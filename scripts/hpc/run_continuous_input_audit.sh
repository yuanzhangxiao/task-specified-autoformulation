#!/bin/bash
# CPU-only input-contract verification. No discovery fitting or LLM requests.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}"
if [[ "${AF_AUDIT_SITE:?}" == aces ]]; then
  module load GCCcore/13.2.0 Python/3.11.5
fi
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]
fi
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${AF_REFERENCE_MODE:-audit}" == phase-c ]]; then
  "$AF_PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_phase_c_benchmarks.py tests/test_continuous_input_submission.py
  "$AF_PYTHON" scripts/prepare_phase_c_benchmarks.py build --root "$AF_OUTPUT_ROOT"
  "$AF_PYTHON" scripts/prepare_phase_c_benchmarks.py verify --root "$AF_OUTPUT_ROOT"
  echo "Completed: $AF_OUTPUT_ROOT/summary.json"
  exit 0
fi
if [[ "${AF_REFERENCE_MODE:-audit}" == refine ]]; then
  "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_local_poll.py tests/test_reference_followup.py
  "$AF_PYTHON" scripts/qualify_reference_fitting.py followup-prepare \
    --source-root "${AF_QUALIFICATION_ROOT:?}" --root "$AF_OUTPUT_ROOT/qualification"
  af_indices=("${SLURM_ARRAY_TASK_ID:-0}")
  if [[ "$AF_AUDIT_SITE" == jetstream2 ]]; then
    af_indices=({0..13})
  fi
  for af_index in "${af_indices[@]}"; do
    af_status=0
    "$AF_PYTHON" scripts/qualify_reference_fitting.py run \
      --root "$AF_OUTPUT_ROOT/qualification" --index "$af_index" || af_status=$?
    "$AF_PYTHON" scripts/qualify_reference_fitting.py followup-report \
      --root "$AF_OUTPUT_ROOT/qualification"
    [[ "$af_status" == 0 ]] || exit "$af_status"
  done
  echo "Updated: $AF_OUTPUT_ROOT/qualification/summary.json"
  exit 0
fi
if [[ "${AF_REFERENCE_MODE:-audit}" == qualify ]]; then
  "$AF_PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_reference_qualification.py tests/test_continuous_input_submission.py
  "$AF_PYTHON" scripts/release_phase_b_public_suite.py \
    --audit-root "${AF_AUDIT_ROOT:?}" --public-data-root "$AF_OUTPUT_ROOT/public"
  "$AF_PYTHON" scripts/qualify_reference_fitting.py prepare \
    --public-data-root "$AF_OUTPUT_ROOT/public" --root "$AF_OUTPUT_ROOT/qualification"
  "$AF_PYTHON" scripts/qualify_reference_fitting.py report --root "$AF_OUTPUT_ROOT/qualification"
  for af_index in 0 1 2 3; do
    "$AF_PYTHON" scripts/qualify_reference_fitting.py run \
      --root "$AF_OUTPUT_ROOT/qualification" --index "$af_index"
    "$AF_PYTHON" scripts/qualify_reference_fitting.py report --root "$AF_OUTPUT_ROOT/qualification"
  done
  echo "Completed: $AF_OUTPUT_ROOT/qualification/summary.json"
  exit 0
fi
"$AF_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_continuous_inputs.py tests/test_continuous_input_submission.py
"$AF_PYTHON" scripts/audit_reference_integrity.py \
  --output-root "$AF_OUTPUT_ROOT/audit" \
  --input-contract continuous-rates-1 --include-test-protocols
"$AF_PYTHON" scripts/audit_reference_integrity.py \
  --output-root "$AF_OUTPUT_ROOT/audit" \
  --input-contract continuous-rates-1 --include-test-protocols
echo "Audit and deterministic resume completed: $AF_OUTPUT_ROOT/audit/summary.json"
