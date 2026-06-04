#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/cross_vit_components}"

python cross_vit_components.py screen --output-dir "$ROOT"
python cross_vit_components.py confirm-attacks --output-dir "$ROOT" --candidate-file "$ROOT/selected_candidates.json"
python cross_vit_components.py confirm-evaluate --output-dir "$ROOT" --candidate-file "$ROOT/selected_candidates.json"
python cross_vit_components.py report --output-dir "$ROOT" --candidate-file "$ROOT/selected_candidates.json"
