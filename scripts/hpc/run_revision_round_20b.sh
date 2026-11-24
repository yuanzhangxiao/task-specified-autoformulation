#!/bin/bash
#SBATCH --account=156264627414
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=ac042,ac096
#SBATCH --job-name=af-revision-round
set -euo pipefail
af_dir=${1:?Path to experiment directory}
af_stage=${2:?Use generate or rank}
case "$af_stage" in generate|rank) ;; *) exit 2 ;; esac
af_group=/scratch/group/p.nairr260351.000/u.yx126462
af_hf=$af_group/huggingface-cache
af_assets=$af_group/phase-c-pointwise-repair-20b-v1/assets/harmony
af_image=/scratch/user/$USER/containers/vllm-openai-v0.27.1.sif
af_revision=6cee5e81ee83917806bbde320786a8fb61efebee
test -f "$af_dir/$af_stage.json"
test -f "$af_hf/hub/models--openai--gpt-oss-20b/snapshots/$af_revision/config.json"
af_container=$(command -v apptainer || command -v singularity)
module load GCCcore/13.2.0 Python/3.11.5
af_tmp=$(mktemp -d "/tmp/af-order-search-${SLURM_JOB_ID}.XXXXXX")
af_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
af_url=http://127.0.0.1:$af_port
setsid "$af_container" exec --nv --bind "$af_hf:$af_hf,$af_assets:$af_assets:ro,$af_tmp:$af_tmp" \
  --env "HF_HOME=$af_hf,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,TIKTOKEN_ENCODINGS_BASE=$af_assets" \
  --env "XDG_CACHE_HOME=$af_tmp,TRITON_CACHE_DIR=$af_tmp/triton,TORCHINDUCTOR_CACHE_DIR=$af_tmp/torch,VLLM_CACHE_ROOT=$af_tmp/vllm" \
  "$af_image" vllm serve openai/gpt-oss-20b --revision "$af_revision" --host 127.0.0.1 \
  --port "$af_port" --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.90 \
  > "$af_dir/server-$SLURM_JOB_ID.log" 2>&1 &
af_pid=$!
trap 'kill -TERM -- "-$af_pid" 2>/dev/null || true; wait "$af_pid" || true; rm -rf -- "$af_tmp"' EXIT
for af_tick in $(seq 1 240); do
  if curl --noproxy '*' -sf "$af_url/v1/models" > "$af_dir/models-$SLURM_JOB_ID.json"; then break; fi
  if ! kill -0 "$af_pid" 2>/dev/null; then tail -n 35 "$af_dir/server-$SLURM_JOB_ID.log"; exit 1; fi
  sleep 2
done
python3 -c 'import json, sys; assert [v["id"] for v in json.load(open(sys.argv[1]))["data"]]==["openai/gpt-oss-20b"]' "$af_dir/models-$SLURM_JOB_ID.json"
python3 "$af_dir/revision_round_provider.py" "$af_dir" "$af_stage" "$af_url"
if [[ "$af_stage" = generate && "${AF_WAIT_FOR_RANK:-0}" = 1 ]]; then
  for af_tick in $(seq 1 240); do
    if [[ -f "$af_dir/rank.json" ]]; then
      python3 "$af_dir/revision_round_provider.py" "$af_dir" rank "$af_url"
      break
    fi
    sleep 2
  done
fi
