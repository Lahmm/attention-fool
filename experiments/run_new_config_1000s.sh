#!/usr/bin/env bash
set -euo pipefail

TASK_PY=/root/miniconda3/envs/att-atk/bin/python
TASK_PREFIX=newconfig1000
TASK_SEED=20260907

run_one() {
  local name=$1
  local model=$2
  local batch_size=$3
  local checkpoints=$4
  local ratios=$5
  local score_mode=$6
  local opponent_strength=$7
  local attack_dir="outputs/attack/${TASK_PREFIX}_${name}_s1000_offset0_seed${TASK_SEED}"
  local csv_path="outputs/csv/outputs_attack_${TASK_PREFIX}_${name}_s1000_offset0_seed${TASK_SEED}.csv"

  echo "===== ${name}: attack ====="
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
    echo "Attack already complete: ${name}"
  else
    "$TASK_PY" main.py \
      --attack-method progressive \
      --whitebox-model "$model" \
      --checkpoints "$checkpoints" \
      --drop-ratios "$ratios" \
      --progressive-patch-selector high \
      --score-window-ratio 0.5 \
      --progressive-score-mode "$score_mode" \
      --score-global-noise-strength 0.2 \
      --opponent-noise-strength "$opponent_strength" \
      --batch-size "$batch_size" \
      --max-attacked-samples 1000 \
      --sample-offset 0 \
      --seed "$TASK_SEED" \
      --output-dir "$attack_dir"
  fi

  echo "===== ${name}: transfer ====="
  if [[ -f "$csv_path" ]]; then
    echo "Transfer CSV already complete: ${name}"
  else
    "$TASK_PY" transfer_eval.py \
      --image-dir "$attack_dir" \
      --amp \
      --exp-name "${TASK_PREFIX}_${name}_s1000_offset0_seed${TASK_SEED}"
  fi
}

run_one vit_b3_b10_c10_10 vit_base_patch16_224 96 \
  block3,block10 \
  0.051020408163,0.051020408163 \
  cosine 0.2

run_one cait_b17_b23_c02_28_projection cait_s24_224 48 \
  block17_gap,block23_gap \
  0.010204081633,0.142857142857 \
  gap_projection 0.2

run_one pit_s2b1_s3b2_s3b3_c05_02_06_opp04 pit_b_224 96 \
  stage2_block1,stage3_block2,stage3_block3 \
  0.02081165,0.03125,0.09375 \
  cosine 0.4

run_one vis_s2b1_s3b1_c41_10_projection_opp04 visformer_small 48 \
  stage2_block1,stage3_block1 \
  0.209183673469,0.204081632653 \
  gap_projection 0.4
