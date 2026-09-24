# Jetstream assessment on existing labeled judge cases

Reuse all 14 frozen consensus-validation cases: two canonical structures, each
with an equivalent pair, five controlled dominance mutations, and one unlabeled
defect tradeoff. Five seeds (10000–10004) and both orientations give 70 paired
trials, 140 orientation judgments and ordinarily 280 HTTP requests. Each stage
retains ten provider attempts: at most 2,800 physical requests. No extra seeds
are selected from scientific outcomes.

Both stages reuse production `PairedHybridJudge`. Current corrected sign evidence,
blinded candidate metadata, fixed comparative indeterminate denominator and the
successful strict atomic-ID transport contract apply. Scientific prompts, low
reasoning, temperature 0.2 and the 6,144-token output allowance are unchanged.
The historical original protocol is reported separately: its sign evidence,
metadata and denominator differed. Differences cannot be attributed to hardware
or provider alone. Managed model revision and setting forwarding remain unverified.

Labels stay in the offline evaluator and never enter judge requests. The exporter
reads existing pair/label/judge-score files and the original public development
loader. That loader reads train/validation to recover public numeric-channel
context. The portable file contains no trajectories, fitted NMSE, private equations
or test data. Historical judge scores are evaluator-only. No proposer, optimizer,
model promotion, automatic follow-up or cluster job cancellation runs.

Passing the pre-existing gates supports use on these **known cases**, not a fresh
holdout calibration or broad scientific correctness. Unlabeled tradeoff winners
do not enter accuracy. Missing, indeterminate, detected defects and false passes
are counted separately. Coverage always includes every planned trial.

## Export on Delta (CPU only)

Set `AF_COMMIT` to the full commit supplied with this change. Run on **Delta**.
The existing project Python environment is reused; no GPU allocation is needed.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set the supplied full commit first}"
  AF_GIT=/projects/bibo/yxiao2/repos/sign-recheck-source.git
  AF_REPO_ROOT=/projects/bibo/yxiao2/repos/jetstream-calibration-${AF_COMMIT:0:7}
  AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
  AF_CAL=/work/hdd/bibo/yxiao2/phase_b/judge-hybrid-consensus-validation-v1
  AF_EXPORT=/work/hdd/bibo/yxiao2/phase_b/judge-jetstream-calibration-inputs-v1.json
  [[ -d "$AF_GIT" ]] || {
    git clone --bare --single-branch --branch codex/prefit-aces-v1 \
      https://github.com/yuanzhangxiao/task-specified-autoformulation.git "$AF_GIT"
  }
  git --git-dir="$AF_GIT" fetch origin codex/prefit-aces-v1
  git --git-dir="$AF_GIT" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git --git-dir="$AF_GIT" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
  export PYTHONDONTWRITEBYTECODE=1
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/judge_jetstream_calibration.py" export \
    --source "$AF_CAL" \
    --data-root /projects/bibo/yxiao2/phase_b/inputs/public \
    --output "$AF_EXPORT"
)
```

Download the single JSON as
`~/Downloads/judge-jetstream-calibration-inputs-v1.json`. The exporter requires
all 14 cases, matching labels, the frozen config and all 140 historical outcomes.
If a historical path is unavailable, locate the existing files; do not regenerate
or replace cases. There is no fallback that chooses another dataset.

## Run on your Mac

Use the same full `AF_COMMIT`, in a separate output from the one-pair pilot.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set the supplied full commit first}"
  AF_BASE=/Users/yuanzhangxiao/Projects/autoformalism
  AF_REPO_ROOT="$AF_BASE/tmp/jetstream-calibration-${AF_COMMIT:0:7}"
  AF_PYTHON="$AF_BASE/.venv/bin/python"
  AF_ROOT="$HOME/Downloads/judge-jetstream-calibration-v1"
  AF_INPUT="$HOME/Downloads/judge-jetstream-calibration-inputs-v1.json"
  [[ -f "$AF_INPUT" ]]
  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
  export PYTHONDONTWRITEBYTECODE=1
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/judge_jetstream_calibration.py" freeze \
    --source "$AF_INPUT" --root "$AF_ROOT"
  caffeinate -i "$AF_PYTHON" "$AF_REPO_ROOT/scripts/judge_jetstream_calibration.py" run \
    --root "$AF_ROOT"
)
```

Enter the key at the hidden prompt; it is never saved in artifacts. Progress is
printed for every orientation and HTTP attempt. Keep the terminal open. Repeating
the block resumes unstarted orientations; completed and interrupted orientations
are terminal and do not receive a fresh budget. A provider failure stops the
invocation before consuming the remaining suite. Inspect logs before resuming.
Do not delete markers to retry a failed unit. Each HTTP attempt retains its
900-second timeout. `caffeinate` prevents idle sleep, not closing the laptop.

## Inspect

After `run` (including Ctrl+C), `SUMMARY.md` and `summary.json` are updated.
For a live partial snapshot, run the pinned CLI with `report --root ...`; this
needs no key and makes no network request.

```bash
AF_ROOT="$HOME/Downloads/judge-jetstream-calibration-v1"
cat "$AF_ROOT/SUMMARY.md"
jq '{status, status_counts, known_case_gate_passed,
     fresh_holdout_calibration_established,
     cost: (.cost | del(.calls)),
     gates: .current.gate_assessment.checks,
     absolute_coverage: .current.absolute_coverage}' "$AF_ROOT/summary.json"
```

The gate verdict stays null until all orientations are terminal. Finishing can
still mean failed gates. Unknown interrupted usage is not zero. Detailed sealed
outcomes, caches, events and wire calls live in `orientations/pair-*/seed-*/*/`.

## ACES probe

The supplied `ac049` probe passed kernels on both H100s in inherited and clean
environments. It verifies CUDA access on that node/container combination, not
model loading or tensor-parallel inference; it does not diagnose the `ac096`
failure. No serving settings need changing solely because of this probe. An
ACES retry should use the existing frozen campaign's original archive, container
and manifest, and fresh Slurm review/report IDs. Keep the Delta pending job
`22343450` unchanged.

The following runs on **ACES** and stops if a sign-recheck review/report is already
active. It locates the original archive by the commit recorded in the submission
manifest, then verifies the frozen plan. Excluding the previously failed node is
a diagnostic retry, not a conclusion that the node is defective.

```bash
(
  set -euo pipefail
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_OUTPUT_ROOT="$AF_GROUP/judge-sign-recheck-v1"
  AF_MANIFEST="$AF_OUTPUT_ROOT/submission_manifest.json"
  export AF_COMMIT="$(jq -er '.identity.commit' "$AF_MANIFEST")"
  export AF_PYTHON="$(jq -er '.identity.python' "$AF_MANIFEST")"
  export AF_VLLM_IMAGE="$(jq -er '.identity.image' "$AF_MANIFEST")"
  export AF_HF_HOME="$(jq -er '.identity.hf_home' "$AF_MANIFEST")"
  AF_REPO_ROOT=''
  for AF_MARKER in "$AF_GROUP"/repos/*/SOURCE_COMMIT \
    /scratch/user/u.yx126462/repos/*/SOURCE_COMMIT; do
    [[ -f "$AF_MARKER" ]] || continue
    if [[ "$(cat "$AF_MARKER")" == "$AF_COMMIT" ]]; then
      AF_REPO_ROOT="${AF_MARKER%/SOURCE_COMMIT}"
      break
    fi
  done
  : "${AF_REPO_ROOT:?Original pinned archive was not found}"
  export AF_REPO_ROOT PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
  module load GCCcore/13.2.0 Python/3.11.5
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/judge_sign_recheck.py" verify --root "$AF_OUTPUT_ROOT"
  AF_QUEUE="$(squeue --me --noheader --format='%j')"
  if printf '%s\n' "$AF_QUEUE" | grep -Eq '^sign-recheck-(review|report|retry)'; then
    echo 'A sign-recheck review/report is active; no duplicate submitted.'
    exit 1
  fi
  AF_JOB="$(sbatch --parsable --nodes=1 --ntasks=1 --export=ALL \
    --account=156264627414 --partition=gpu --gres=gpu:h100:2 \
    --cpus-per-task=8 --mem=128G --time=08:00:00 --exclude=ac096 \
    --job-name=sign-recheck-retry \
    --output="$AF_OUTPUT_ROOT/logs/retry-%j.out" \
    --error="$AF_OUTPUT_ROOT/logs/retry-%j.err" \
    "$AF_REPO_ROOT/scripts/hpc/run_judge_sign_recheck_aces.sh" review 0)"
  AF_JOB="${AF_JOB%%;*}"
  [[ "$AF_JOB" =~ ^[0-9]+$ ]]
  printf 'ACES retry review: %s\n' "$AF_JOB"
  sbatch --parsable --nodes=1 --ntasks=1 --export=ALL \
    --account=156264627414 --partition=cpu --cpus-per-task=1 --mem=4G \
    --time=00:15:00 --dependency="afterany:$AF_JOB" \
    --job-name=sign-recheck-report \
    --output="$AF_OUTPUT_ROOT/logs/retry-report-%j.out" \
    --error="$AF_OUTPUT_ROOT/logs/retry-report-%j.err" \
    "$AF_REPO_ROOT/scripts/hpc/run_judge_sign_recheck_aces.sh" report 0
)
```

Keep both new Slurm IDs. The original submission manifest stays unchanged;
`retry-*` logs identify this execution. If submission returns an uncertain reply,
inspect `squeue` before repeating: a job may already exist. Review checkpoints
remain authoritative; an interrupted review never gets a fresh budget. Inspect
the same campaign `SUMMARY.md` after the report finishes.
