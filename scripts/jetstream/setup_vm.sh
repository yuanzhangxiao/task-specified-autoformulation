#!/bin/bash
# Prepare a Jetstream2 Ubuntu VM to run the vendored LLM baselines (LLM-SR,
# LLM-ODE) against the Jetstream2 hosted GPT-OSS endpoint.
#
# Run it from the pinned checkout. Rerunning is safe: each step keeps what is
# already in place and checks it. Nothing here needs a key or a password; the
# hosted endpoint answers Jetstream2 addresses without one, so none is stored
# on the VM, where LLM-SR executes the programs the model writes.
#
#   AF_HOME            everything this installs (default ~/af)
#   AF_RELEASE_BUNDLE  the public Phase C release tarball copied from the Mac
#                      (default ~/phase-c-release-public-v2.tar.gz)

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
: "${AF_RELEASE_BUNDLE:=${HOME}/phase-c-release-public-v2.tar.gz}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly venv="${AF_HOME}/venv"
readonly py="${venv}/bin/python"
readonly vendor="${AF_HOME}/vendor"
readonly release="${AF_HOME}/release/phase-c-public-v2"
readonly pilot_config="${repo}/configs/phase_c_llm_sr_budget_pilot_v1.json"
readonly ode_config="${repo}/configs/phase_b_llm_ode_campaign_v1.json"

step() { printf '\n== %s\n' "$*"; }
config_value() {  # file, dotted key
  python3 - "$1" "$2" <<'EOF'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
EOF
}

step "checkout"
git -C "${repo}" diff --quiet && git -C "${repo}" diff --cached --quiet || {
  echo "the checkout at ${repo} has local changes" >&2
  exit 2
}
echo "${repo} at $(git -C "${repo}" rev-parse HEAD)"

step "uv and Python 3.13"
export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
# LLM-ODE builds its programs with str.replace(count=...), new in Python 3.13.
[[ -x "${py}" ]] || uv venv --python 3.13 "${venv}"
"${py}" -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
"${py}" --version

step "packages"
# CPU-only torch: LLM-SR's profiler imports torch's TensorBoard writer and
# LLM-ODE imports torch; neither needs a GPU on this VM.
uv pip install --quiet --python "${py}" \
  --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"
# Our package with casadi, which its Phase C loader imports, then the vendored
# methods' own imports. Their published pins predate Python 3.13 (numpy 1.26,
# torch 2.0), so these are releases that install on it; the resolved set is
# recorded below.
uv pip install --quiet --python "${py}" -e "${repo}[dev,casadi]" \
  "absl-py==2.3.1" "requests==2.32.5" "tensorboard==2.20.0" \
  "findiff==0.12.1" "sympy==1.14.0" "tqdm==4.67.1"

step "upstream checkouts"
clone_pinned() {  # name, repository, commit
  local target="${vendor}/$1"
  [[ -d "${target}/.git" ]] || git clone --quiet "$2" "${target}"
  git -C "${target}" checkout --quiet --detach "$3"
  [[ "$(git -C "${target}" rev-parse HEAD)" == "$3" ]] || {
    echo "$1 is not at $3" >&2
    exit 2
  }
  echo "$1 at $3"
}
mkdir -p "${vendor}"
clone_pinned LLM-SR "$(config_value "${pilot_config}" upstream.repository)" \
  "$(config_value "${pilot_config}" upstream.commit)"
clone_pinned llm-ode "$(config_value "${ode_config}" upstream.repository)" \
  "$(config_value "${ode_config}" upstream.commit)"

step "imports"
(
  cd "${repo}"
  PYTHONPATH="${repo}/src" "${py}" - "${repo}" "${vendor}" <<'EOF'
import sys
from pathlib import Path

repo, vendor = Path(sys.argv[1]), Path(sys.argv[2])
import autoformalism

where = Path(autoformalism.__file__).resolve()
assert where.is_relative_to(repo / "src"), f"imported {where}, not this checkout"
sys.path[:0] = [str(vendor / "LLM-SR"), str(vendor / "llm-ode")]
from llmsr import config, evaluator, pipeline, profile, sampler  # noqa: F401
from llmode.llm import Llm, generate_prompt  # noqa: F401
from llmode.llmode import LlmOdeEquation  # noqa: F401
from llmode.system import System  # noqa: F401

print("our package, LLM-SR and LLM-ODE import")
EOF
)

step "Phase C public release"
if [[ ! -f "${release}/summary.json" ]]; then
  [[ -f "${AF_RELEASE_BUNDLE}" ]] || {
    echo "copy the release bundle to ${AF_RELEASE_BUNDLE} first" >&2
    exit 2
  }
  mkdir -p "${release}"
  tar -xzf "${AF_RELEASE_BUNDLE}" -C "${release}"
fi
receipt="$(sha256sum "${release}/summary.json" | cut -d' ' -f1)"
[[ "${receipt}" == "$(config_value "${pilot_config}" release_summary_sha256)" ]] || {
  echo "the release receipt ${receipt} is not the one the plans name" >&2
  exit 2
}
echo "receipt ${receipt}"

step "hosted endpoint"
PYTHONPATH="${repo}/src" "${py}" - <<'EOF'
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    JETSTREAM2_HOSTED_BASE_URL,
    JETSTREAM2_HOSTED_MODEL,
    served_model_ids,
)

served = served_model_ids(JETSTREAM2_HOSTED_BASE_URL)
assert JETSTREAM2_HOSTED_MODEL in served, served
print(f"{JETSTREAM2_HOSTED_MODEL} is served ({len(served)} models listed)")
EOF

step "record"
uv pip freeze --python "${py}" >"${AF_HOME}/setup_freeze.txt"
{
  echo "utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "checkout $(git -C "${repo}" rev-parse HEAD)"
  echo "python $("${py}" --version 2>&1)"
  echo "uv $(uv --version)"
  echo "packages sha256 $(sha256sum "${AF_HOME}/setup_freeze.txt" | cut -d' ' -f1)"
  echo "LLM-SR $(git -C "${vendor}/LLM-SR" rev-parse HEAD)"
  echo "llm-ode $(git -C "${vendor}/llm-ode" rev-parse HEAD)"
  echo "release receipt ${receipt}"
  echo "cpus $(nproc)"
} | tee "${AF_HOME}/setup_record.txt"
echo
echo "setup complete"
