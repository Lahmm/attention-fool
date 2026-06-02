#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/attack/lazyagg/wavelet_ablation"
LOG_DIR="${ROOT}/logs"
GUIDE_MODELS="deit_base_patch16_224,pit_s_224,cait_s24_224"
BLACK_BOX_MODELS="deit_base_patch16_224,beit_base_patch16_224,swin_tiny_patch4_window7_224,pvt_v2_b2,cait_s24_224,levit_256,pit_s_224,crossvit_15_240"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/att-atk/bin/python}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_DIR"

run_one() {
  local exp_name="$1"    # e.g. wavelet_ablation_dim_fg_low
  local dim_flag="$2"    # 1 = --dim, 0 = no --dim
  local area="$3"        # foreground / background / all
  local method="$4"      # wavelet_noise_low / wavelet_noise_high
  local out_dir="${ROOT}/${exp_name}"

  local dim_arg=""
  if [ "$dim_flag" = "1" ]; then
    dim_arg="--dim"
  fi

  echo "========================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $exp_name (area=$area, method=$method, dim=$dim_flag)"
  echo "========================================"

  "$PYTHON_BIN" main.py \
    --mode attack \
    --max-attacked-samples 500 \
    --steps 40 \
    --ti-sigma 0 \
    $dim_arg \
    --mi \
    --mi-decay 1.0 \
    --guide-aug \
    --guide-aug-area "$area" \
    --guide-aug-method "$method" \
    --guide-aug-copies 3 \
    --guide-aug-strength 0.2 \
    --attention-guide-models "$GUIDE_MODELS" \
    --attention-guide-type qk_cls \
    --attention-guide-build-method patch \
    --layers=0,1,4,9,11 \
    --output-dir "$out_dir" \
    2>&1 | tee "${LOG_DIR}/${exp_name}.attack.log"

  "$PYTHON_BIN" transfer_eval.py \
    --image-dir "$out_dir" \
    --prefix adv_ \
    --model-name "$BLACK_BOX_MODELS" \
    --batch-size 128 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --amp \
    --exp-name "$exp_name" \
    2>&1 | tee "${LOG_DIR}/${exp_name}.transfer.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $exp_name"
}

# ========== DIM ON ==========
run_one "wavelet_ablation_dim_fg_low"    1 "foreground" "wavelet_noise_low"
run_one "wavelet_ablation_dim_bg_high"   1 "background" "wavelet_noise_high"
run_one "wavelet_ablation_dim_all_low"   1 "all"        "wavelet_noise_low"
run_one "wavelet_ablation_dim_all_high"  1 "all"        "wavelet_noise_high"
run_one "wavelet_ablation_dim_fg_high"   1 "foreground" "wavelet_noise_high"
run_one "wavelet_ablation_dim_bg_low"    1 "background" "wavelet_noise_low"

# ========== DIM OFF ==========
run_one "wavelet_ablation_nodim_fg_low"    0 "foreground" "wavelet_noise_low"
run_one "wavelet_ablation_nodim_bg_high"   0 "background" "wavelet_noise_high"
run_one "wavelet_ablation_nodim_all_low"   0 "all"        "wavelet_noise_low"
run_one "wavelet_ablation_nodim_all_high"  0 "all"        "wavelet_noise_high"
run_one "wavelet_ablation_nodim_fg_high"   0 "foreground" "wavelet_noise_high"
run_one "wavelet_ablation_nodim_bg_low"    0 "background" "wavelet_noise_low"

echo ""
echo "========================================"
echo "All 12 ablation experiments completed."
echo "========================================"
