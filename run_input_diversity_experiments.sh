#!/bin/bash
# Input Diversity 四种方案的 ASR 实验
# 固定 baseline 参数: epsilon=16/255, steps=10, MI=1.0, feature_layer=12
#   token_score_cls_noise=True, HighScore, ZeroToken, OppChannel, r=0.3, 100 samples

set -euo pipefail
cd /root/autodl-tmp/attention-fool

BASE_DIR="outputs/attack/lmdss_ablation"
COMMON_ARGS=(
  --max-attacked-samples 100 --steps 10
  --whitebox-model vit_base_patch16_224
  --epsilon 0.06274509803921569
  --guide-aug --guide-aug-method token_patch_dropout
  --guide-aug-copies 20
  --patch-dropout-ratio 0.3
  --patch-dropout-score-mode high
  --patch-dropout-fill-mode zero_noise
  --patch-dropout-noise-mode opponent_channel_gaussian
  --token-score-cls-noise --feature-layer 12
  --mode attack
)

run_exp() {
  local exp_name=$1; shift
  local output_dir="${BASE_DIR}/${exp_name}"

  echo ""
  echo "============================================================"
  echo "  Experiment: ${exp_name}"
  echo "  Output:     ${output_dir}"
  echo "  Args:       $@"
  echo "============================================================"

  echo "[1/2] Generating adversarial samples..."
  python main.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    "$@"

  echo "[2/2] Evaluating transfer ASR..."
  python transfer_eval.py \
    --image-dir "${output_dir}" \
    --prefix adv_ \
    --exp-name "${exp_name}" \
    --batch-size 256
}

# ============================================================
# 方案二 (P0): Cross-Patch Counterfactual Transport — rotate180
# ============================================================
for alpha in 0.10 0.20 0.30; do
  alpha_str="${alpha//./_}"
  run_exp "plan2_transport_rotate180_${alpha_str}_s100" \
    --cross-patch-transport-mode rotate180 \
    --cross-patch-transport-alpha "${alpha}"
done

# ============================================================
# 方案四 (P1): Kept-Token Orthogonal Residual — pair_swap
# ============================================================
for alpha in 0.05 0.10 0.20; do
  alpha_str="${alpha//./_}"
  run_exp "plan4_rotation_pairswap_${alpha_str}_s100" \
    --kept-token-rotation-mode pair_swap \
    --kept-token-rotation-alpha "${alpha}"
done

# ============================================================
# 方案一 (P2): Phase Pair — 10 groups × 2 views, plain pair mean
# ============================================================
run_exp "plan1_phase_pair_10x2_mean_s100" \
  --input-diversity-groups 10 \
  --input-diversity-views-per-group 2 \
  --input-diversity-phase-shift-set "4,4;8,8;12,12" \
  --input-diversity-pair-aggregation mean

# ============================================================
# 方案三 (P3): Pair-Difference Gradient
# ============================================================
for lam in 0.05 0.10; do
  lam_str="${lam//./_}"
  run_exp "plan3_pairdiff_10x2_${lam_str}_s100" \
    --input-diversity-groups 10 \
    --input-diversity-views-per-group 2 \
    --input-diversity-phase-shift-set "4,4;8,8;12,12" \
    --input-diversity-pair-aggregation difference_mix \
    --input-diversity-lambda-difference "${lam}"
done

echo ""
echo "============================================================"
echo "  All experiments completed!"
echo "  Results saved to: outputs/csv/"
echo "============================================================"
