#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-outputs/dim_bg_image_gradient_frequency_math}"
mkdir -p "$ROOT"
COMMON=(--output-dir "$ROOT" --seeds 0,1 --max-samples 100 --trace-steps 1,10,20,40 --dim-samples 8 --batch-size 4)
{
  python dim_bg_image_gradient_frequency_math.py experiment "${COMMON[@]}"
  python dim_bg_image_gradient_frequency_math.py report "${COMMON[@]}"
} 2>&1 | tee "$ROOT/dim_bg_image_gradient_frequency_run.log"
test -s "$ROOT/operator_frequency_response.npz"
test -s "$ROOT/region_band_gradient_metrics.npz"
test -s "$ROOT/image_gradient_frequency_link_metrics.npz"
test -s "$ROOT/factorial_interaction_metrics.npz"
test -s "$ROOT/dim_bg_image_gradient_frequency_report.json"
test -s "$ROOT/dim_bg_image_gradient_frequency_conclusion_zh.md"
test -s "$ROOT/dim_bg_image_gradient_frequency_run.log"
