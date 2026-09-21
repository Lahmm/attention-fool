#!/usr/bin/env bash
set -euo pipefail

TASK_PY=/root/miniconda3/envs/att-atk/bin/python
TASK_PREFIX=selectorisolation500
TOTAL_RUNS=24
RUN_INDEX=0
SEEDS=(20260921 20260922 20260923)
TARGET_MODELS=(
  vit_base_patch16_224 levit_256 pit_b_224 deit_base_patch16_224
  tnt_s_patch16_224 convit_base visformer_small cait_s24_224
  inception_v3 inception_v4 inception_resnet_v2 resnet101
  inception_v3_adv inception_resnet_v2_adv
)

attack_is_complete() {
  local attack_dir=$1 model=$2 selector=$3 seed=$4
  [[ -d "$attack_dir" ]] || return 1
  [[ $(find "$attack_dir" -maxdepth 1 -type f -name 'adv_*.png' | wc -l) -eq 500 ]] || return 1
  [[ -f "$attack_dir/attack_params.json" ]] || return 1
  [[ -f "$attack_dir/gradient_diagnostics.json" ]] || return 1
  [[ -f "$attack_dir/replay_manifest.json" ]] || return 1
  "$TASK_PY" - "$attack_dir/attack_params.json" "$model" "$selector" "$seed" <<'PY'
import json
import sys

path, model, selector, seed = sys.argv[1:]
params = json.loads(open(path, encoding="utf-8").read())
expected = {
    "whitebox_model": model,
    "progressive_patch_selector": selector,
    "score_global_noise_strength": 0.0,
    "opponent_noise_strength": 0.0,
    "gaussian_alpha": 0.0,
    "max_attacked_samples": 500,
    "seed": int(seed),
}
raise SystemExit(0 if all(params.get(key) == value for key, value in expected.items()) else 1)
PY
}

transfer_is_complete() {
  local csv_path=$1 predictions_path=$2
  [[ -f "$csv_path" && -f "$predictions_path" ]] || return 1
  "$TASK_PY" - "$csv_path" "$predictions_path" "${TARGET_MODELS[@]}" <<'PY'
import csv
import json
import sys

csv_path, predictions_path, *models = sys.argv[1:]
with open(csv_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
csv_complete = any(
    row.get("adv_image_count", "").strip() == "500"
    and all(row.get(model, "").strip() for model in models)
    for row in rows
)
artifact = json.loads(open(predictions_path, encoding="utf-8").read())
predictions = artifact.get("predictions", {})
prediction_complete = (
    len(artifact.get("sample_ids", [])) == 500
    and len(artifact.get("labels", [])) == 500
    and set(predictions) == set(models)
    and all(len(predictions[model]) == 500 for model in models)
)
raise SystemExit(0 if csv_complete and prediction_complete else 1)
PY
}

run_one() {
  local source=$1 model=$2 batch_size=$3 selector=$4 seed=$5
  local stem="${TASK_PREFIX}_${source}_${selector}_s500_seed${seed}"
  local attack_dir="outputs/attack/${stem}"
  local csv_path="outputs/csv/outputs_attack_${stem}.csv"
  local predictions_path="${attack_dir}/transfer_predictions.json"

  RUN_INDEX=$((RUN_INDEX + 1))
  echo "===== [${RUN_INDEX}/${TOTAL_RUNS}] ${source} selector=${selector} seed=${seed} ====="
  if attack_is_complete "$attack_dir" "$model" "$selector" "$seed"; then
    echo "Attack already complete: ${stem}"
  elif [[ -e "$attack_dir" ]]; then
    echo "Refusing to overwrite incomplete or mismatched attack directory: $attack_dir" >&2
    return 1
  else
    "$TASK_PY" main.py \
      --attack-method progressive \
      --whitebox-model "$model" \
      --progressive-patch-selector "$selector" \
      --score-global-noise-strength 0 \
      --opponent-noise-strength 0 \
      --gaussian-alpha 0 \
      --batch-size "$batch_size" \
      --max-attacked-samples 500 \
      --seed "$seed" \
      --output-dir "$attack_dir"
  fi

  if transfer_is_complete "$csv_path" "$predictions_path"; then
    echo "Transfer already complete: ${stem}"
  else
    "$TASK_PY" transfer_eval.py \
      --image-dir "$attack_dir" \
      --amp \
      --predictions-output "$predictions_path" \
      --exp-name "$stem"
  fi
}

for seed in "${SEEDS[@]}"; do
  for selector in high random; do
    run_one vit vit_base_patch16_224 96 "$selector" "$seed"
    run_one cait cait_s24_224 48 "$selector" "$seed"
    run_one pit pit_b_224 96 "$selector" "$seed"
    run_one visformer visformer_small 48 "$selector" "$seed"
  done
done

"$TASK_PY" experiments/summarize_selector_isolation_500s.py
echo "===== ALL ${TOTAL_RUNS} ISOLATION ATTACKS, TRANSFERS, AND REPORT COMPLETED ====="
