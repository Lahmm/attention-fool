#!/usr/bin/env bash
set -euo pipefail

TASK_PY=/root/miniconda3/envs/att-atk/bin/python
TASK_PREFIX=followup192
TASK_SEED=20260907

run_one() {
  local name=$1
  local model=$2
  local batch_size=$3
  local checkpoints=$4
  local ratios=$5
  local score_mode=$6
  local opponent_strength=$7
  local attack_dir="outputs/attack/${TASK_PREFIX}_${name}_s192_offset0_seed${TASK_SEED}"
  local csv_path="outputs/csv/outputs_attack_${TASK_PREFIX}_${name}_s192_offset0_seed${TASK_SEED}.csv"

  if [[ -d "$attack_dir" ]]; then
    local image_count
    image_count=$(find "$attack_dir" -maxdepth 1 -type f -name 'adv_*.png' | wc -l)
    if [[ "$image_count" -ne 192 ]] ||
       [[ ! -f "$attack_dir/attack_params.json" ]] ||
       [[ ! -f "$attack_dir/gradient_diagnostics.json" ]] ||
       [[ ! -f "$attack_dir/replay_manifest.json" ]]; then
      echo "Refusing to overwrite incomplete attack directory: $attack_dir" >&2
      return 1
    fi
    echo "Attack already complete: $name"
  else
    "$TASK_PY" main.py \
      --whitebox-model "$model" \
      --checkpoints "$checkpoints" \
      --drop-ratios "$ratios" \
      --progressive-patch-selector high \
      --score-window-ratio 0.5 \
      --progressive-score-mode "$score_mode" \
      --score-global-noise-strength 0.2 \
      --opponent-noise-strength "$opponent_strength" \
      --batch-size "$batch_size" \
      --max-attacked-samples 192 \
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
      --exp-name "${TASK_PREFIX}_${name}_s192_offset0_seed${TASK_SEED}"
  fi
}

# Same-revision public baselines.
run_one b_vit vit_base_patch16_224 96 block3,block10 0.051020408163,0.051020408163 cosine 0.2
run_one b_cait cait_s24_224 48 block5_gap,block17_gap,block23_gap 0.051020408163,0.051020408163,0.051020408163 cosine 0.2
run_one b_pit pit_b_224 96 stage2_block1,stage3_block2,stage3_block3 0.02081165,0.03125,0.09375 cosine 0.2
run_one b_vis visformer_small 48 stage2_block1,stage3_block1 0.209183673469,0.204081632653 cosine 0.2

# E1: CaiT late-heavy K2 and K3 schedules, total budget fixed at 30.
run_one e1_cait_k2_c02_28 cait_s24_224 48 block17_gap,block23_gap 0.010204081633,0.142857142857 cosine 0.2
run_one e1_cait_k2_c05_25 cait_s24_224 48 block17_gap,block23_gap 0.025510204082,0.127551020408 cosine 0.2
run_one e1_cait_k2_c10_20 cait_s24_224 48 block17_gap,block23_gap 0.051020408163,0.102040816327 cosine 0.2
run_one e1_cait_k2_c15_15 cait_s24_224 48 block17_gap,block23_gap 0.076530612245,0.076530612245 cosine 0.2
run_one e1_cait_k3_c05_10_15 cait_s24_224 48 block5_gap,block17_gap,block23_gap 0.025510204082,0.051020408163,0.076530612245 cosine 0.2
run_one e1_cait_k3_c05_05_20 cait_s24_224 48 block5_gap,block17_gap,block23_gap 0.025510204082,0.025510204082,0.102040816327 cosine 0.2
run_one e1_cait_k3_c10_05_15 cait_s24_224 48 block5_gap,block17_gap,block23_gap 0.051020408163,0.025510204082,0.076530612245 cosine 0.2

# E3: label-free, gradient-independent GAP score definitions on Visformer.
run_one e3_vis_score_loo visformer_small 48 stage2_block1,stage3_block1 0.209183673469,0.204081632653 gap_leave_one_out_cosine 0.2
run_one e3_vis_score_projection visformer_small 48 stage2_block1,stage3_block1 0.209183673469,0.204081632653 gap_projection 0.2
run_one e3_vis_score_channel_rms visformer_small 48 stage2_block1,stage3_block1 0.209183673469,0.204081632653 gap_channel_rms_cosine 0.2

# E4: opponent noise coarse sweep. Strength 0.2 is shared with each baseline.
for strength in 0.0 0.1 0.3 0.4; do
  tag=${strength/./}
  run_one "e4_vit_opp_${tag}" vit_base_patch16_224 96 block3,block10 0.051020408163,0.051020408163 cosine "$strength"
  run_one "e4_cait_opp_${tag}" cait_s24_224 48 block5_gap,block17_gap,block23_gap 0.051020408163,0.051020408163,0.051020408163 cosine "$strength"
  run_one "e4_pit_opp_${tag}" pit_b_224 96 stage2_block1,stage3_block2,stage3_block3 0.02081165,0.03125,0.09375 cosine "$strength"
  run_one "e4_vis_opp_${tag}" visformer_small 48 stage2_block1,stage3_block1 0.209183673469,0.204081632653 cosine "$strength"
done

# E6: Visformer K2 equal-ratio strength scan; 41/10 is the shared baseline.
run_one e6_vis_ratio_c20_05 visformer_small 48 stage2_block1,stage3_block1 0.102040816327,0.102040816327 cosine 0.2
run_one e6_vis_ratio_c29_07 visformer_small 48 stage2_block1,stage3_block1 0.147959183673,0.142857142857 cosine 0.2
run_one e6_vis_ratio_c49_12 visformer_small 48 stage2_block1,stage3_block1 0.25,0.244897959184 cosine 0.2
run_one e6_vis_ratio_c59_15 visformer_small 48 stage2_block1,stage3_block1 0.301020408163,0.30612244898 cosine 0.2
