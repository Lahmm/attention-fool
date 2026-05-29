#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/attack/lazyagg/lowfreq"
LOG_DIR="${ROOT}/logs"
GUIDE_MODELS="deit_base_patch16_224,pit_s_224,cait_s24_224"
BLACK_BOX_MODELS="deit_base_patch16_224,beit_base_patch16_224,swin_tiny_patch4_window7_224,pvt_v2_b2,cait_s24_224,levit_256,pit_s_224,crossvit_15_240"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/att-atk/bin/python}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_DIR"

run_one() {
  local exp_name="$1"
  local methods="$2"
  local out_dir="${ROOT}/${exp_name}"

  "$PYTHON_BIN" main.py \
    --mode attack \
    --max-attacked-samples 500 \
    --steps 40 \
    --ti-sigma 0 \
    --dim \
    --mi \
    --mi-decay 1.0 \
    --guide-aug \
    --guide-aug-area background \
    --guide-aug-method "$methods" \
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
    --batch-size 256 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --amp \
    --exp-name "$exp_name" \
    2>&1 | tee "${LOG_DIR}/${exp_name}.transfer.log"
}

run_one "ours_dim_background_patch_pre_fpridx_lowpass_gauss_mifgsm_s40_500" "lowpass_gauss"
run_one "ours_dim_background_patch_pre_fpridx_laplacian_low_mifgsm_s40_500" "laplacian_low"
run_one "ours_dim_background_patch_pre_fpridx_fft_lowboost_mifgsm_s40_500" "fft_lowboost"
run_one "ours_dim_background_patch_pre_fpridx_illumination_low_mifgsm_s40_500" "illumination_low"
run_one "ours_dim_background_patch_pre_fpridx_lowfreq_all_mifgsm_s40_500" "lowpass_gauss,laplacian_low,fft_lowboost,illumination_low"
