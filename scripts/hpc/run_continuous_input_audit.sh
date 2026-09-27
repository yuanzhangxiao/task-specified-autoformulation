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
