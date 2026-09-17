#!/usr/bin/env bash
set -euo pipefail

TASK_PY=/root/miniconda3/envs/att-atk/bin/python
TASK_SEED=20260907

run_one() {
  local name=$1
  local checkpoints=$2
  local ratios=$3
  local attack_dir="outputs/attack/${name}_s1000_offset0_seed${TASK_SEED}"
  local csv_path="outputs/csv/outputs_attack_${name}_s1000_offset0_seed${TASK_SEED}.csv"

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
    echo "Attack already complete: $name"
  else
    "$TASK_PY" main.py \
      --attack-method progressive \
      --whitebox-model cait_s24_224 \
      --checkpoints "$checkpoints" \
      --drop-ratios "$ratios" \
      --progressive-patch-selector high \
      --progressive-score-mode gap_projection \
      --score-global-noise-strength 0.2 \
      --opponent-noise-strength 0.2 \
      --batch-size 48 \
      --max-attacked-samples 1000 \
      --sample-offset 0 \
      --seed "$TASK_SEED" \
      --output-dir "$attack_dir"
  fi

  if [[ -f "$csv_path" ]]; then
    echo "Transfer CSV already complete: $name"
  else
    "$TASK_PY" transfer_eval.py \
      --image-dir "$attack_dir" \
      --amp \
      --exp-name "${name}_s1000_offset0_seed${TASK_SEED}"
  fi
}

run_one caitproj1000_k3_b5_b17_b23_c10_10_10 \
  block5_gap,block17_gap,block23_gap \
  0.051020408163,0.051020408163,0.051020408163

run_one caitproj1000_k2_b17_b23_c15_15 \
  block17_gap,block23_gap \
  0.076530612245,0.076530612245
