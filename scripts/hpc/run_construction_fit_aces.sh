#!/bin/bash
# CPU-only execution with an import preflight before any numerical work.
set -euo pipefail
: "${AF_REPO_ROOT:?required}" "${AF_OUTPUT_ROOT:?required}"
AF_STAGE=${1:?Use prepare, submit, run, verify, report, or inspect}
shift
AF_CLI="$AF_REPO_ROOT/scripts/qualify_construction_fitting.py"
AF_SUBMIT="$AF_REPO_ROOT/scripts/submit_construction_fit_aces.py"
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AF_FIT_RUNTIME=${AF_FIT_RUNTIME:-native}
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
AF_PROBE='import ctypes, numpy, scipy, pandas, sympy, pydantic; from autoformalism.research import construction_fit_qualification'
case "$AF_FIT_RUNTIME" in
  native)
    if ! "$AF_PYTHON" -c "$AF_PROBE" >/dev/null 2>&1; then
      echo 'Scientific Python import failed; loading the compiler and Python module hierarchy.' >&2
      if ! type module >/dev/null 2>&1; then
        echo 'No module command. Load the ACES scientific environment or explicitly select AF_FIT_RUNTIME=container with AF_CPU_IMAGE.' >&2
        exit 2
      fi
      module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
      "$AF_PYTHON" -c "$AF_PROBE"
    fi
    AF_RUN=("$AF_PYTHON")
    ;;
  container)
    : "${AF_CPU_IMAGE:?Set AF_CPU_IMAGE to an existing scientific Python image}"
    [[ -f "$AF_CPU_IMAGE" ]] || { echo "Missing CPU image: $AF_CPU_IMAGE" >&2; exit 2; }
    if ! AF_CONTAINER=$(command -v apptainer || command -v singularity); then
      if [[ -n "${AF_CONTAINER_MODULE:-}" ]] && type module >/dev/null 2>&1; then
        module load "$AF_CONTAINER_MODULE"
      fi
      AF_CONTAINER=$(command -v apptainer || command -v singularity) || {
        echo 'No Apptainer/Singularity runtime. Use native scientific Python or run this launcher in a CPU allocation; no work was started.' >&2
        exit 2
      }
    fi
    AF_RUN=("$AF_CONTAINER" exec --bind "$AF_REPO_ROOT:$AF_REPO_ROOT,${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}:${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462},$AF_CPU_IMAGE:$AF_CPU_IMAGE:ro"
      --env "PYTHONPATH=$PYTHONPATH" --env PYTHONDONTWRITEBYTECODE=1
      --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1
      "$AF_CPU_IMAGE" "${AF_CONTAINER_PYTHON:-python3}")
    "${AF_RUN[@]}" -c "$AF_PROBE"
    ;;
  *) echo 'AF_FIT_RUNTIME must be native or container.' >&2; exit 2 ;;
esac
cd "$AF_REPO_ROOT"
case "$AF_STAGE" in
  prepare) exec "${AF_RUN[@]}" "$AF_CLI" prepare --root "$AF_OUTPUT_ROOT" "$@" ;;
  submit) exec "${AF_RUN[@]}" "$AF_SUBMIT" submit --root "$AF_OUTPUT_ROOT" "$@" ;;
  verify) exec "${AF_RUN[@]}" "$AF_CLI" verify --root "$AF_OUTPUT_ROOT" ;;
  run)
    AF_INDEX=${SLURM_ARRAY_TASK_ID:-${1:-}}
    [[ "$AF_INDEX" =~ ^[0-7]$ ]] || { echo 'Expected task index 0..7.' >&2; exit 2; }
    "${AF_RUN[@]}" "$AF_SUBMIT" verify --root "$AF_OUTPUT_ROOT"
    exec "${AF_RUN[@]}" "$AF_CLI" run --root "$AF_OUTPUT_ROOT" --index "$AF_INDEX"
    ;;
  report|inspect)
    if [[ -f "$AF_OUTPUT_ROOT/launcher_execution.json" ]]; then
      "${AF_RUN[@]}" "$AF_SUBMIT" verify --root "$AF_OUTPUT_ROOT"
    fi
    exec "${AF_RUN[@]}" "$AF_CLI" "$AF_STAGE" --root "$AF_OUTPUT_ROOT"
    ;;
  *) echo 'Use prepare, submit, run, verify, report, or inspect.' >&2; exit 2 ;;
esac
