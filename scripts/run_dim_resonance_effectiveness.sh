#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-outputs/attack/lazyagg/dim_resonance_effectiveness}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/att-atk/bin/python}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
STEPS="${STEPS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
GUIDE_MODELS="${GUIDE_MODELS:-deit_base_patch16_224,pit_s_224,cait_s24_224}"
TARGET_MODELS="${TARGET_MODELS:-deit_base_patch16_224,beit_base_patch16_224,swin_tiny_patch4_window7_224,pvt_v2_b2,cait_s24_224,levit_256,pit_s_224,crossvit_15_240}"
LAYERS="${LAYERS:-0,1,4,9,11}"
GUIDE_TYPE="${GUIDE_TYPE:-qk_cls}"
GUIDE_BUILD="${GUIDE_BUILD:-patch}"
GUIDE_AREA="${GUIDE_AREA:-background}"
GUIDE_STRENGTH="${GUIDE_STRENGTH:-0.2}"
GUIDE_COPIES="${GUIDE_COPIES:-3}"

run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
    return
  fi
  "$@"
}

attack_and_eval() {
  local name="$1"
  local methods="$2"
  local out_dir="${ROOT}/${name}"

  if [[ ! -d "$out_dir" || "${FORCE_ATTACK:-0}" == "1" ]]; then
    run_cmd "$PYTHON_BIN" main.py \
      --mode attack \
      --max-attacked-samples "$MAX_SAMPLES" \
      --steps "$STEPS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --ti-sigma 0 \
      --dim \
      --mi \
      --mi-decay 1.0 \
      --guide-aug \
      --guide-aug-area "$GUIDE_AREA" \
      --guide-aug-method "$methods" \
      --guide-aug-copies "$GUIDE_COPIES" \
      --guide-aug-strength "$GUIDE_STRENGTH" \
      --attention-guide-models "$GUIDE_MODELS" \
      --attention-guide-type "$GUIDE_TYPE" \
      --attention-guide-build-method "$GUIDE_BUILD" \
      --layers="$LAYERS" \
      --output-dir "$out_dir"
  fi

  run_cmd "$PYTHON_BIN" transfer_eval.py \
    --image-dir "$out_dir" \
    --prefix adv_ \
    --model-name "$TARGET_MODELS" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$EVAL_WORKERS" \
    --prefetch-factor 4 \
    --amp \
    --exp-name "dim_resonance_${name}"
}

# Same DIM/MI/no-normalize-grad/patch-qk/fpridx-background protocol as the
# current strongest family; only the guide augmentation method changes.
attack_and_eval "reference_djf" "dropout,jitter,freq"
attack_and_eval "dim_resonance_only" "dim_resonance"
attack_and_eval "dim_resonance_djf" "dropout,jitter,freq,dim_resonance"
attack_and_eval "fft_lowboost_only" "fft_lowboost"
attack_and_eval "fft_lowboost_djf" "dropout,jitter,freq,fft_lowboost"

# Plain DIM-MI baseline with the same steps and no gradient normalization.
BASELINE_DIR="${ROOT}/dim_mi_noaug"
if [[ ! -d "$BASELINE_DIR" || "${FORCE_ATTACK:-0}" == "1" ]]; then
  run_cmd "$PYTHON_BIN" main.py \
    --mode attack \
    --max-attacked-samples "$MAX_SAMPLES" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --ti-sigma 0 \
    --dim \
    --mi \
    --mi-decay 1.0 \
    --output-dir "$BASELINE_DIR"
fi
run_cmd "$PYTHON_BIN" transfer_eval.py \
  --image-dir "$BASELINE_DIR" \
  --prefix adv_ \
  --model-name "$TARGET_MODELS" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$EVAL_WORKERS" \
  --prefetch-factor 4 \
  --amp \
  --exp-name "dim_resonance_dim_mi_noaug"

run_cmd "$PYTHON_BIN" scripts/summarize_dim_resonance_effectiveness.py --root "$ROOT"
