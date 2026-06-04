#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-outputs/cross_vit_quick}"
BASELINE_ROOT="${BASELINE_ROOT:-$ROOT/baselines}"
COMMON=(--output-dir "$ROOT" --seeds 0,1 --trace-steps 1,10,20,40 --discovery-samples 15 --total-samples 50 --candidate-count 2 --bootstrap-repeats 3000 --batch-size 4 --eval-batch-size 32)
for SEED in 0 1; do
  DIR="$BASELINE_ROOT/baseline_seed_$SEED"
  [[ -f "$DIR/manifest.json" ]] || python causal_analysis.py mi-switch --output-dir "$DIR" --seed "$SEED" --max-samples 50 --batch-size 4
done
[[ -f "$ROOT/selected_candidates.json" ]] || python cross_vit_components.py screen "${COMMON[@]}"
[[ -f "$ROOT/manifest.json" ]] || python cross_vit_components.py confirm-attacks "${COMMON[@]}" --baseline-root "$BASELINE_ROOT" --candidate-file "$ROOT/selected_candidates.json"
[[ -f "$ROOT/evaluation.pt" ]] || python cross_vit_components.py confirm-evaluate "${COMMON[@]}" --candidate-file "$ROOT/selected_candidates.json"
python cross_vit_components.py report "${COMMON[@]}" --candidate-file "$ROOT/selected_candidates.json"
test -s "$ROOT/final_report.json"
