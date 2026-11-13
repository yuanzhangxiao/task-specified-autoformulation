#!/bin/bash -l
# Slurm runs a spool copy; the first argument identifies the original deployment.
set -eu
ops=${1:?operator directory}
export AF_GROUP=${2:?scratch group} AF_T1_FULL_ROOT=${3:?prepared result root}
exec /bin/bash "$ops/run.sh" finish
