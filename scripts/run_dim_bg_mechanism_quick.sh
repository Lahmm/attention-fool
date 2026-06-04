#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-outputs/dim_bg_mechanism_quick}"
COMMON=(--output-dir "$ROOT" --seeds 0,1 --max-samples 100)
[[ -f "$ROOT/method_high_frequency_ranking.json" ]] || python dim_bg_mechanism.py rank "${COMMON[@]}"
python dim_bg_mechanism.py experiment "${COMMON[@]}"
python dim_bg_mechanism.py report "${COMMON[@]}"
test -s "$ROOT/method_high_frequency_ranking.json"
test -s "$ROOT/dim_bg_mechanism_report.json"
