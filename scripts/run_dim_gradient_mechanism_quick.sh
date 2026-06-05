#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-outputs/dim_gradient_mechanism_quick}"
mkdir -p "$ROOT"
COMMON=(--output-dir "$ROOT" --seeds 0,1 --max-samples 100 --trace-steps 1,10,20,40 --dim-samples 8 --batch-size 4)
{
  python dim_gradient_mechanism.py experiment "${COMMON[@]}"
  python dim_gradient_mechanism.py report "${COMMON[@]}"
} 2>&1 | tee "$ROOT/dim_gradient_mechanism_run.log"
test -s "$ROOT/gradient_mechanism_metrics.npz"
test -s "$ROOT/dim_gradient_mechanism_report.json"
test -s "$ROOT/dim_gradient_mechanism_conclusion.md"
test -s "$ROOT/dim_gradient_mechanism_run.log"
