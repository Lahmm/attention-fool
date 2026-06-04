import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class QuickSerialSmokeTests(unittest.TestCase):
    def test_orchestrator_runs_dim_then_cross_and_supports_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bin_dir = root / "bin"; bin_dir.mkdir(); log = root / "calls.log"
            fake = bin_dir / "python"
            fake.write_text(r'''#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$CALL_LOG"
script="$1"; shift
out=""; mode=""
[[ $# -gt 0 ]] && mode="$1"
while [[ $# -gt 0 ]]; do
  [[ "$1" == "--output-dir" ]] && { out="$2"; shift 2; continue; }
  [[ "$1" == "--output" ]] && { out="$2"; shift 2; continue; }
  shift
done
case "$script:$mode" in
  dim_bg_mechanism.py:rank) mkdir -p "$out"; echo '{}' > "$out/method_high_frequency_ranking.json" ;;
  dim_bg_mechanism.py:experiment) mkdir -p "$out/runs" ;;
  dim_bg_mechanism.py:report) mkdir -p "$out"; echo '{}' > "$out/dim_bg_mechanism_report.json" ;;
  causal_analysis.py:trace) mkdir -p "$out"; echo '{}' > "$out/manifest.json" ;;
  cross_vit_components.py:screen) mkdir -p "$out"; echo '{}' > "$out/selected_candidates.json" ;;
  cross_vit_components.py:confirm-attacks) mkdir -p "$out"; echo '{}' > "$out/manifest.json" ;;
  cross_vit_components.py:confirm-evaluate) mkdir -p "$out"; echo x > "$out/evaluation.pt" ;;
  cross_vit_components.py:report) mkdir -p "$out"; echo '{}' > "$out/final_report.json" ;;
  combined_conclusion.py:*) mkdir -p "$(dirname "$out")"; echo conclusion > "$out" ;;
esac
''', encoding="utf-8")
            fake.chmod(0o755); output = root / "output"
            env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CALL_LOG": str(log)}
            command = ["bash", "scripts/run_dim_bg_then_cross_vit_quick.sh", str(output)]
            subprocess.run(command, check=True, env=env, cwd=Path(__file__).parents[1])
            first = log.read_text().splitlines(); self.assertIn("dim_bg_mechanism.py rank", first[0])
            experiment = next(x for x in first if "dim_bg_mechanism.py experiment" in x)
            self.assertIn("--max-samples 50", experiment); self.assertIn("--gradient-probes 2", experiment)
            self.assertIn("background:forward-only", experiment); self.assertNotIn("backward-only", experiment)
            self.assertLess(next(i for i, x in enumerate(first) if "dim_bg_mechanism.py report" in x), next(i for i, x in enumerate(first) if "cross_vit_components.py screen" in x))
            subprocess.run(command, check=True, env=env, cwd=Path(__file__).parents[1])
            second = log.read_text().splitlines(); self.assertEqual(sum("dim_bg_mechanism.py rank" in x for x in second), 2)
            self.assertEqual(sum("cross_vit_components.py screen" in x for x in second), 1)
            self.assertTrue((output / "combined_conclusion.md").is_file())


if __name__ == "__main__": unittest.main()
