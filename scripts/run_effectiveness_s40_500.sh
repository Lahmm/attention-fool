#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/attack/lazyagg/effectiveness"
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
  --output-dir "${ROOT}/dim_mifgsm_s40_500"

for DIM_NAME in nodim dim; do
  EXTRA_ARGS=()

  if [[ "$DIM_NAME" == "dim" ]]; then
    EXTRA_ARGS+=(--dim)
  fi

  run_cmd python main.py \
    --mode attack \
    --max-attacked-samples 500 \
    --steps 40 \
    --ti-sigma 0 \
    --mi \
    --mi-decay 1.0 \
    --guide-aug \
    --guide-aug-method "$METHODS" \
    --guide-aug-copies 3 \
    --guide-aug-strength 0.2 \
    "${EXTRA_ARGS[@]}" \
    --output-dir "${ROOT}/ours_${DIM_NAME}_mifgsm_s40_500"
done
