#!/bin/bash
# Prepare a Jetstream2 CPU VM to run the classical Phase C baselines (SINDy and
# PySR) with scripts/jetstream/run_phase_c_classical.sh: a Python 3.12
# environment with this checkout and PySR 1.5.9 (as on Delta), PySR's Julia
# runtime, and the public Phase C release.
#
# Run it from the pinned checkout. Rerunning is safe: each step keeps what is
# already in place and checks it. Nothing here needs a key or a password. The
# environment is separate from the LLM baselines' (setup_vm.sh), so a VM can
# hold both without one changing the other's packages.
#
#   AF_HOME            everything this installs (default ~/af)
#   AF_RELEASE_BUNDLE  the public Phase C release tarball copied from the Mac
#                      (default ~/phase-c-release-public-v2.tar.gz)

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
: "${AF_RELEASE_BUNDLE:=${HOME}/phase-c-release-public-v2.tar.gz}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly venv="${AF_HOME}/venv-classical"
readonly py="${venv}/bin/python"
readonly depot="${AF_HOME}/julia-depot-pysr-1.5.9"
readonly release="${AF_HOME}/release/phase-c-public-v2"
readonly plan="${repo}/configs/phase_c_public_baseline_jetstream_cpu_v1.json"

step() { printf '\n== %s\n' "$*"; }

step "checkout"
git -C "${repo}" diff --quiet && git -C "${repo}" diff --cached --quiet || {
  echo "the checkout at ${repo} has local changes" >&2
  exit 2
}
echo "${repo} at $(git -C "${repo}" rev-parse HEAD)"

step "uv and Python 3.12"
export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
[[ -x "${py}" ]] || uv venv --python 3.12 "${venv}"
"${py}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"${py}" --version

step "packages"
# The package's own pins, casadi for the Phase C loader, and the pysr extra
# (pysr==1.5.9), the version the Delta environment runs.
uv pip install --quiet --python "${py}" -e "${repo}[casadi,pysr]"
(
  cd "${repo}"
  PYTHONPATH="${repo}/src" "${py}" - "${repo}" <<'EOF'
import sys
from pathlib import Path

import autoformalism

where = Path(autoformalism.__file__).resolve()
assert where.is_relative_to(Path(sys.argv[1]) / "src"), f"imported {where}"
print("our package imports from this checkout")
EOF
)

step "Phase C public release"
if [[ ! -f "${release}/summary.json" ]]; then
  [[ -f "${AF_RELEASE_BUNDLE}" ]] || {
    echo "copy the release bundle to ${AF_RELEASE_BUNDLE} first" >&2
    exit 2
  }
  mkdir -p "${release}"
  # The bundle was made by macOS tar, whose extended attributes GNU tar
  # reports once per file; they carry nothing the release needs.
  tar --warning=no-unknown-keyword -xzf "${AF_RELEASE_BUNDLE}" -C "${release}"
fi
receipt="$(sha256sum "${release}/summary.json" | cut -d' ' -f1)"
expected="$("${py}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["release_summary_sha256"])' "${plan}")"
[[ "${receipt}" == "${expected}" ]] || {
  echo "the release receipt ${receipt} is not the one the plan names" >&2
  exit 2
}
echo "receipt ${receipt}"

step "PySR's Julia runtime"
# The first import downloads Julia and SymbolicRegression.jl and precompiles
# them into the depot, which every task then shares, as on Delta.
mkdir -p "${depot}"
(
  cd "${repo}"
  JULIA_DEPOT_PATH="${depot}" PYTHONPATH="${repo}/src" "${py}" \
    scripts/check_pysr_runtime.py --output "${AF_HOME}/pysr_runtime_setup.json"
)

step "record"
uv pip freeze --python "${py}" >"${AF_HOME}/setup_classical_freeze.txt"
{
  echo "utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "checkout $(git -C "${repo}" rev-parse HEAD)"
  echo "python $("${py}" --version 2>&1)"
  echo "uv $(uv --version)"
  echo "packages sha256 $(sha256sum "${AF_HOME}/setup_classical_freeze.txt" | cut -d' ' -f1)"
  echo "julia $("${py}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["julia_version"])' "${AF_HOME}/pysr_runtime_setup.json")"
  echo "release receipt ${receipt}"
  echo "flavor $(curl -s --max-time 3 http://169.254.169.254/latest/meta-data/instance-type || echo unknown)"
  echo "cpus $(nproc)"
} | tee "${AF_HOME}/setup_classical_record.txt"
echo
echo "setup complete"
