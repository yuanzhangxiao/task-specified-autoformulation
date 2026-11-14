#!/usr/bin/env bash
# Staged next to the two release-pinned operations helpers, outside the scientific code.
set -eo pipefail
umask 077
ops=$(cd "$(dirname "$0")" && pwd -P)
campaign=/scratch/group/p.nairr260351.000/u.yx126462/iclr-pilot-fit-r1-2334957
mirror=/scratch/user/u.yx126462/iclr-stress-solver-archive-v1
source "$campaign/activate.sh"
set -u
case ${1:-} in check|dry-run|submit) ;; *) echo "Use: bash $0 check | dry-run | submit" >&2; exit 2 ;; esac
(cd "$ops" && sha256sum --quiet -c SHA256SUMS)
exec bash -c '
    ops=$1; action=$2; mirror=$3
    source "$0" status >/dev/null
    qualify
    module load WebProxy
    export PYTHONPATH="$ops:$PYTHONPATH"
    exec "$AF_PYTHON" "$ops/scripts/iclr_stress_remainder.py" "$action" --root "$AF_CAMPAIGN/stress" \
        --mirror "$mirror" --ops "$ops" --account "$AF_GPU_ACCOUNT"
' "$AF_DELIVERY/code/scripts/hpc/iclr_aces.sh" "$ops" "$1" "$mirror"
