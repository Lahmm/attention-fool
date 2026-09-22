#!/usr/bin/env bash
set -euo pipefail

TASK_PY=/root/miniconda3/envs/att-atk/bin/python
TASK_PREFIX=prd1000
TASK_SEED=20260907

run_one() {
  local name=$1
  local model=$2
  local batch_size=$3
  local checkpoints=$4
  local ratios=$5
  local opponent_strength=$6
  local attack_dir="outputs/attack/${TASK_PREFIX}_${name}_seed${TASK_SEED}"
  local csv_path="outputs/csv/outputs_attack_${TASK_PREFIX}_${name}_seed${TASK_SEED}.csv"

  if [[ -d "$attack_dir" ]]; then
    local image_count
    image_count=$(find "$attack_dir" -maxdepth 1 -type f -name 'adv_*.png' | wc -l)
    if [[ "$image_count" -ne 1000 ]] ||
       [[ ! -f "$attack_dir/attack_params.json" ]] ||
       [[ ! -f "$attack_dir/gradient_diagnostics.json" ]] ||
       [[ ! -f "$attack_dir/replay_manifest.json" ]]; then
      echo "Refusing to overwrite incomplete attack directory: $attack_dir" >&2
      return 1
    fi
  else
    "$TASK_PY" main.py \
      --attack-method progressive \
      --whitebox-model "$model" \
      --checkpoints "$checkpoints" \
      --drop-ratios "$ratios" \
      --opponent-noise-strength "$opponent_strength" \
      --batch-size "$batch_size" \
      --max-attacked-samples 1000 \
      --seed "$TASK_SEED" \
      --output-dir "$attack_dir"
  fi

  if [[ ! -f "$csv_path" ]]; then
    "$TASK_PY" transfer_eval.py \
      --image-dir "$attack_dir" \
      --amp \
      --exp-name "${TASK_PREFIX}_${name}_seed${TASK_SEED}"
  fi
}

run_one vit vit_base_patch16_224 96 \
  block3,block10 \
  0.051020408163,0.051020408163 \
  0.2

run_one cait cait_s24_224 48 \
  block17,block23 \
  0.010204081633,0.142857142857 \
  0.2

run_one pit pit_b_224 96 \
  stage2_block1,stage3_block2,stage3_block3 \
  0.02081165,0.03125,0.09375 \
  0.4

run_one visformer visformer_small 48 \
  stage2_block1,stage3_block1 \
  0.209183673469,0.204081632653 \
  0.4
