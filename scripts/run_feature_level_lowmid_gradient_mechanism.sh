#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-outputs/feature_level_lowmid_gradient_mechanism}"
mkdir -p "$ROOT"
COMMON=(--output-dir "$ROOT" --seeds 0,1 --max-samples 100 --trace-steps 1,10,20,40 --batch-size 4)
{
  python feature_level_lowmid_gradient_mechanism.py experiment "${COMMON[@]}"
  python feature_level_lowmid_gradient_mechanism.py report "${COMMON[@]}"
} 2>&1 | tee "$ROOT/feature_level_lowmid_gradient_run.log"
test -s "$ROOT/feature_region_lowmid_metrics.npz"
test -s "$ROOT/augmentation_band_effect_metrics.npz"
test -s "$ROOT/dim_jacobian_frequency_response.npz"
test -s "$ROOT/dim_cross_band_leakage_metrics.npz"
test -s "$ROOT/feature_level_lowmid_gradient_report.json"
test -s "$ROOT/feature_level_lowmid_gradient_conclusion_zh.md"
test -s "$ROOT/feature_level_lowmid_gradient_run.log"
