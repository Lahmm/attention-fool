"""Supplement DeiT-B distilled using the ViT PRD settings and FPR targets."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fpr_targets"
IMAGES = ROOT / "outputs/attack/prd_fpr_targets_deit_distilled_seed20260907"
MODEL = "deit_base_distilled_patch16_224"

def run(args, log):
    with log.open("a") as handle:
        subprocess.run([sys.executable, "-u", *args], cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)

def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.loads((ROOT / "results/prd_run_artifacts/vit/attack_params.json").read_text())
    if not (IMAGES / "attack_params.json").exists():
        if IMAGES.exists() and any(IMAGES.iterdir()):
            raise RuntimeError(f"Inspect incomplete output before restarting: {IMAGES}")
        print("Generating DeiT-B distilled: 1000 samples", flush=True)
        run(["main.py", "--whitebox-model", MODEL, "--batch-size", "8",
             "--seed", str(params["seed"]), "--max-attacked-samples", "1000",
             "--steps", str(params["steps"]), "--epsilon", str(params["epsilon"]),
             "--step-size", str(params["step_size"]),
             "--input-diversity-groups", str(params["input_diversity_groups"]),
             "--mi-decay", str(params["mi_decay"]),
             "--gaussian-sigma", str(params["gaussian_sigma"]),
             "--gaussian-alpha", str(params["gaussian_alpha"]),
             "--checkpoints", ",".join(params["checkpoints"]),
             "--drop-ratios", ",".join(map(str, params["drop_ratios"])),
             "--opponent-noise-strength", str(params["opponent_noise_strength"]),
             "--output-dir", str(IMAGES)], OUT / "deit_distilled_attack.log")
    assert len(list(IMAGES.glob("adv_*.png"))) == 1000
    predictions = OUT / "deit_distilled_predictions.json"
    if not predictions.exists():
        print("Evaluating DeiT-B distilled: 13 FPR targets", flush=True)
        run(["transfer_eval.py", "--image-dir", str(IMAGES), "--protocol", "fpr",
             "--batch-size", "32", "--num-workers", "4", "--no-tf32",
             "--strict-model-loading", "--no-record", "--predictions-output", str(predictions)],
            OUT / "deit_distilled_transfer.log")
    data = json.loads(predictions.read_text())
    reference = json.loads((OUT / "vit_predictions.json").read_text())
    assert data["sample_ids"] == reference["sample_ids"]
    assert data["labels"] == reference["labels"] and len(data["labels"]) == 1000
    assert set(data["predictions"]) == set(reference["predictions"])
    row = {"source": MODEL}
    for target, pred in data["predictions"].items():
        assert len(pred) == 1000
        row[target] = sum(p != y for p, y in zip(pred, data["labels"])) / 10
    targets = list(data["predictions"])
    assert MODEL not in targets  # All thirteen FPR targets are black-box for this source.
    row["overall"] = row["black_box"] = sum(row[t] for t in targets) / len(targets)
    with (OUT / "deit_distilled_transfer_asr.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)
    print("Complete: " + str(OUT / "deit_distilled_transfer_asr.csv"), flush=True)

if __name__ == "__main__":
    main()
