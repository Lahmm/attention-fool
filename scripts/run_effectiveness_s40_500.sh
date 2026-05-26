#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/attack/lazyagg/effectiveness"
GUIDE_MODELS="deit_base_patch16_224,pit_s_224,cait_s24_224"
METHODS="dropout,jitter,freq"

run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
    return
  fi

  "$@"
}

run_cmd python main.py \
  --mode attack \
  --max-attacked-samples 500 \
  --steps 40 \
  --ti-sigma 0 \
  --output-dir "${ROOT}/baseline_ifgsm_s40_500"

run_cmd python main.py \
  --mode attack \
  --max-attacked-samples 500 \
  --steps 40 \
  --ti-sigma 0 \
  --mi \
  --mi-decay 1.0 \
  --normalize-grad \
  --output-dir "${ROOT}/baseline_mifgsm_s40_500"

run_cmd python main.py \
  --mode attack \
  --max-attacked-samples 500 \
  --steps 40 \
  --ti-sigma 0 \
  --dim \
  --output-dir "${ROOT}/dim_ifgsm_s40_500"

run_cmd python main.py \
  --mode attack \
  --max-attacked-samples 500 \
  --steps 40 \
  --ti-sigma 0 \
  --dim \
  --mi \
  --mi-decay 1.0 \
  --normalize-grad \
  --output-dir "${ROOT}/dim_mifgsm_s40_500"

for DIM_NAME in nodim dim; do
  for AREA in foreground background all; do
    for BUILD in pixel patch; do
      for GUIDE_TYPE_NAME in post pre; do
        for LAYER_NAME in lastsix fpridx; do
          EXTRA_ARGS=()

          if [[ "$DIM_NAME" == "dim" ]]; then
            EXTRA_ARGS+=(--dim)
          fi

          if [[ "$GUIDE_TYPE_NAME" == "post" ]]; then
            GUIDE_TYPE="postsoftmax_cls"
          else
            GUIDE_TYPE="qk_cls"
          fi

          if [[ "$LAYER_NAME" == "lastsix" ]]; then
            LAYERS="-6,-5,-4,-3,-2,-1"
          else
            LAYERS="0,1,4,9,11"
          fi

          run_cmd python main.py \
            --mode attack \
            --max-attacked-samples 500 \
            --steps 40 \
            --ti-sigma 0 \
            --mi \
            --mi-decay 1.0 \
            --normalize-grad \
            --guide-aug \
            --guide-aug-area "$AREA" \
            --guide-aug-method "$METHODS" \
            --guide-aug-copies 3 \
            --guide-aug-strength 0.2 \
            --attention-guide-models "$GUIDE_MODELS" \
            --attention-guide-type "$GUIDE_TYPE" \
            --attention-guide-build-method "$BUILD" \
            --layers="$LAYERS" \
            "${EXTRA_ARGS[@]}" \
            --output-dir "${ROOT}/ours_${DIM_NAME}_${AREA}_${BUILD}_${GUIDE_TYPE_NAME}_${LAYER_NAME}_mifgsm_s40_500"
        done
      done
    done
  done
done
