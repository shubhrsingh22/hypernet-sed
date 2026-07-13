#!/usr/bin/env bash
# Run models across seeds using a per-GPU job queue: every GPU always pulls the
# next pending (experiment, seed) job as soon as its current one finishes, so no
# GPU idles (unlike a fixed batch-and-wait schedule).
#
# Usage:
#   scripts/run_seeds.sh "42 1337 2024 7 2718" "hcrnn32 hcrnn64 hcrnn128 hcrnn256" "0 1 2 3"
#
# Args (all optional):
#   $1  space-separated seeds            (default: "42 1337 2024 7 2718")
#   $2  space-separated experiments      (default: all baselines + hyper sweep)
#   $3  space-separated GPU ids to use   (default: "0 1 2 3")
#
# Env vars:
#   PRECISION      Lightning precision (default: bf16-mixed; use 32 for fp32)
#   SLOTS_PER_GPU  concurrent runs per GPU (default 1). Launch-bound models like
#                  the HyperLSTM use <25% of a GPU, so 3 packs them efficiently.
#   NWORKERS       dataloader workers per run (default 8; lower when packing)
#   EXTRA          extra Hydra overrides appended to every run
set -uo pipefail

SEEDS=(${1:-42 1337 2024 7 2718})
EXPERIMENTS=(${2:-crnn_bi crnn_uni hcrnn32 hcrnn64 hcrnn128 hcrnn256})
GPUS=(${3:-0 1 2 3})
PRECISION="${PRECISION:-bf16-mixed}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
NWORKERS="${NWORKERS:-8}"
EXTRA="${EXTRA:-}"

# Expand GPUs into slots (each GPU repeated SLOTS_PER_GPU times).
SLOT_GPU=()
for gpu in "${GPUS[@]}"; do
  for ((s = 0; s < SLOTS_PER_GPU; s++)); do SLOT_GPU+=("$gpu"); done
done
NSLOTS=${#SLOT_GPU[@]}

# Build the job list (experiment x seed).
JOBS=()
for exp in "${EXPERIMENTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    JOBS+=("${exp}:${seed}")
  done
done
echo "Queued ${#JOBS[@]} jobs across GPUs: ${GPUS[*]} (precision=${PRECISION})"

mkdir -p logs
LAST_PID=""
launch() {  # $1=gpu  $2=job  -> sets global LAST_PID
  local gpu="$1" exp="${2%%:*}" seed="${2##*:}"
  echo ">>> [gpu ${gpu}] experiment=${exp} seed=${seed}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
      experiment="${exp}" seed="${seed}" trainer.precision="${PRECISION}" ${EXTRA} \
      || echo "!!! FAILED: experiment=${exp} seed=${seed}"
  ) > "logs/console_${exp}_seed${seed}.log" 2>&1 &
  LAST_PID=$!
}

declare -A GPU_PID
idx=0
# Initial fill: one job per GPU.
for gpu in "${GPUS[@]}"; do
  if (( idx < ${#JOBS[@]} )); then
    launch "$gpu" "${JOBS[$idx]}"
    GPU_PID[$gpu]=$LAST_PID
    idx=$((idx + 1))
  fi
done

# Poll: when a GPU's job finishes, launch the next pending job on it.
while :; do
  running=0
  for gpu in "${GPUS[@]}"; do
    pid="${GPU_PID[$gpu]:-}"
    [ -z "$pid" ] && continue
    if kill -0 "$pid" 2>/dev/null; then
      running=1
    else
      if (( idx < ${#JOBS[@]} )); then
        launch "$gpu" "${JOBS[$idx]}"
        GPU_PID[$gpu]=$LAST_PID
        idx=$((idx + 1))
        running=1
      else
        GPU_PID[$gpu]=""
      fi
    fi
  done
  (( running == 0 )) && break
  sleep 15
done

echo "All runs finished. Aggregate: python -m src.aggregate_results --logs_dir logs --out_dir results"
