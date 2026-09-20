#!/bin/bash
# Run the test suite without paying 25 minutes for every change.
#
#   scripts/run_tests.sh fast   parallel bulk only          (~5 min)
#   scripts/run_tests.sh full   bulk, then the serial files (~9 min)
#
# A handful of files supervise real subprocesses against wall-clock deadlines.
# Under xdist they time out and report false failures, so they run serially.
# Everything else distributes per file, which keeps a module's shared fixtures
# on one worker.

set -uo pipefail
readonly mode="${1:-full}"
readonly python_bin="${AF_TEST_PYTHON:-.venv/bin/python}"

# Verified contention-sensitive: each passes serially and fails under load.
readonly serial_files=(
  tests/test_scaled_alternating.py
  tests/test_fitter_recovery.py
  tests/test_fitter_methods.py
  tests/test_fitter_offset.py
)

deselect=()
for file in "${serial_files[@]}"; do
  deselect+=(--deselect "${file}")
done
# Untracked work from a concurrent session in this checkout, when present.
[[ -f tests/test_shared_process_pilot.py ]] &&
  deselect+=(--deselect tests/test_shared_process_pilot.py)

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
echo "== parallel bulk"
"${python_bin}" -m pytest -q -n auto --dist loadfile "${deselect[@]}" "${@:2}"
bulk=$?

serial=0
if [[ "${mode}" == full ]]; then
  echo "== serial, timing-sensitive"
  "${python_bin}" -m pytest -q "${serial_files[@]}"
  serial=$?
fi

if ((bulk == 0 && serial == 0)); then
  echo "PASS (${mode})"
  exit 0
fi
echo "FAIL (${mode}): bulk=${bulk} serial=${serial}" >&2
exit 1
