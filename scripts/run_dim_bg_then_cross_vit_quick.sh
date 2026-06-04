#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-outputs/quick_serial}"
DIM_ROOT="$ROOT/dim_bg_mechanism"
CROSS_ROOT="$ROOT/cross_vit_quick"
mkdir -p "$ROOT"
bash scripts/run_dim_bg_mechanism_quick.sh "$DIM_ROOT"
test -s "$DIM_ROOT/method_high_frequency_ranking.json"
test -s "$DIM_ROOT/dim_bg_mechanism_report.json"
bash scripts/run_cross_vit_component_experiment.sh "$CROSS_ROOT"
test -s "$CROSS_ROOT/final_report.json"
python combined_conclusion.py --dim-root "$DIM_ROOT" --cross-root "$CROSS_ROOT" --output "$ROOT/combined_conclusion.md"
test -s "$ROOT/combined_conclusion.md"
