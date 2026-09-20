#!/usr/bin/env bash
set -euo pipefail

TASK_PY=/root/miniconda3/envs/att-atk/bin/python
TASK_PREFIX=selectorgnoise1000
TASK_SEED=20260920
TOTAL_RUNS=36
RUN_INDEX=0

TARGET_MODELS=(
  vit_base_patch16_224
  levit_256
  pit_b_224
  deit_base_patch16_224
  tnt_s_patch16_224
  convit_base
  visformer_small
  cait_s24_224
  inception_v3
  inception_v4
  inception_resnet_v2
  resnet101
  inception_v3_adv
  inception_resnet_v2_adv
)

attack_is_complete() {
  local attack_dir=$1
  local model=$2
  local selector=$3
  local noise_strength=$4
  [[ -d "$attack_dir" ]] || return 1
  local image_count
  image_count=$(find "$attack_dir" -maxdepth 1 -type f -name 'adv_*.png' | wc -l)
  [[ "$image_count" -eq 1000 ]] || return 1
  [[ -f "$attack_dir/attack_params.json" ]] || return 1
  [[ -f "$attack_dir/gradient_diagnostics.json" ]] || return 1
  [[ -f "$attack_dir/replay_manifest.json" ]] || return 1
  "$TASK_PY" - "$attack_dir/attack_params.json" "$model" "$selector" "$noise_strength" <<'PY'
import json
import sys

path, model, selector, noise_strength = sys.argv[1:]
params = json.loads(open(path, encoding="utf-8").read())
expected = {
    "whitebox_model": model,
    "progressive_patch_selector": selector,
    "score_global_noise_strength": float(noise_strength),
    "seed": 20260920,
    "max_attacked_samples": 1000,
}
raise SystemExit(0 if all(params.get(key) == value for key, value in expected.items()) else 1)
PY
}

csv_is_complete() {
  local csv_path=$1
  [[ -f "$csv_path" ]] || return 1
  "$TASK_PY" - "$csv_path" "${TARGET_MODELS[@]}" <<'PY'
import csv
import sys

path, *models = sys.argv[1:]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
complete = any(
    row.get("adv_image_count", "").strip() == "1000"
    and all(row.get(model, "").strip() for model in models)
    for row in rows
)
raise SystemExit(0 if complete else 1)
PY
}

run_one() {
  local source=$1
  local model=$2
  local batch_size=$3
  local selector=$4
  local noise_label=$5
  local noise_strength=$6
  local stem="${TASK_PREFIX}_${source}_${selector}_gnoise${noise_label}_adapter_defaults_s1000_seed${TASK_SEED}"
  local attack_dir="outputs/attack/${stem}"
  local csv_path="outputs/csv/outputs_attack_${stem}.csv"

  RUN_INDEX=$((RUN_INDEX + 1))
  echo "===== [${RUN_INDEX}/${TOTAL_RUNS}] ${source} selector=${selector} global_noise=${noise_label} ====="
  if attack_is_complete "$attack_dir" "$model" "$selector" "$noise_strength"; then
    echo "Attack already complete: ${stem}"
  elif [[ -e "$attack_dir" ]]; then
    echo "Refusing to overwrite incomplete or mismatched attack directory: $attack_dir" >&2
    return 1
  else
    "$TASK_PY" main.py \
      --attack-method progressive \
      --whitebox-model "$model" \
      --progressive-patch-selector "$selector" \
      --score-global-noise-strength "$noise_strength" \
      --batch-size "$batch_size" \
      --max-attacked-samples 1000 \
      --seed "$TASK_SEED" \
      --output-dir "$attack_dir"
  fi

  if csv_is_complete "$csv_path"; then
    echo "Transfer already complete: ${stem}"
  else
    "$TASK_PY" transfer_eval.py \
      --image-dir "$attack_dir" \
      --amp \
      --exp-name "$stem"
  fi
}

run_source() {
  local source=$1
  local model=$2
  local batch_size=$3
  local selector
  for selector in high low extreme-high extreme-low; do
    run_one "$source" "$model" "$batch_size" "$selector" off 0.0
    run_one "$source" "$model" "$batch_size" "$selector" on 0.2
  done
  # Uniform random routing never reads the global score token, so score-global
  # noise is inapplicable. Run this control once with its strength set to zero.
  run_one "$source" "$model" "$batch_size" random ignored 0.0
}

echo "Starting ${TOTAL_RUNS} attacks and transfer evaluations with seed ${TASK_SEED}."
echo "All checkpoint, drop-ratio, score-mode, and opponent-noise settings come from adapters."
run_source vit vit_base_patch16_224 96
run_source cait cait_s24_224 48
run_source pit pit_b_224 96
run_source visformer visformer_small 48

"$TASK_PY" experiments/summarize_selector_global_noise_matrix.py
echo "===== ALL ${TOTAL_RUNS} ATTACKS, TRANSFERS, AND THE MATRIX REPORT COMPLETED ====="
