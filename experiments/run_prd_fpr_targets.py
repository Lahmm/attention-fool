"""Run current four-source PRD on 1000 images against FPR Table 2 targets."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fpr_targets"
SOURCES = {"vit": "vit_base_patch16_224", "cait": "cait_s24_224",
           "pit": "pit_b_224", "visformer": "visformer_small"}
TARGETS = ["vit_base_patch16_224", "cait_s24_224", "pit_b_224", "visformer_small",
           "swin_tiny_patch4_window7_224", "deit_tiny_patch16_224", "coat_tiny",
           "resnet18", "vgg16", "densenet121", "efficientnet_b0",
           "mobilenet_v3_small", "resnext50_32x4d"]

def run(args, log):
    with log.open("a") as handle:
        subprocess.run([sys.executable, "-u", *args], cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    rows = []
    for source, model in SOURCES.items():
        images = ROOT / "outputs/attack" / f"prd_fpr_targets_{source}_seed20260907"
        if not (images / "attack_params.json").exists():
            if images.exists() and any(images.iterdir()):
                raise RuntimeError(f"Incomplete output requires inspection: {images}")
            params = json.loads((ROOT / f"results/prd_run_artifacts/{source}/attack_params.json").read_text())
            print(f"Generating {source}: 1000 samples", flush=True)
            run(["main.py", "--whitebox-model", model, "--batch-size", "8",
                 "--seed", "20260907", "--max-attacked-samples", "1000",
                 "--checkpoints", ",".join(params["checkpoints"]),
                 "--drop-ratios", ",".join(map(str, params["drop_ratios"])),
                 "--opponent-noise-strength", str(params["opponent_noise_strength"]),
                 "--output-dir", str(images)], OUT / f"{source}_attack.log")
        if len(list(images.glob("adv_*.png"))) != 1000:
            raise RuntimeError(f"Expected 1000 adversarial images: {images}")
        predictions = OUT / f"{source}_predictions.json"
        if not predictions.exists():
            print(f"Evaluating {source}: 13 targets", flush=True)
            run(["transfer_eval.py", "--image-dir", str(images), "--protocol", "fpr",
                 "--batch-size", "32", "--num-workers", "4", "--no-tf32",
                 "--strict-model-loading", "--no-record", "--predictions-output", str(predictions)],
                OUT / f"{source}_transfer.log")
        data = json.loads(predictions.read_text())
        if len(data["labels"]) != 1000 or set(data["predictions"]) != set(TARGETS):
            raise RuntimeError(f"Incomplete predictions: {predictions}")
        row = {"source": source}
        for target in TARGETS:
            pred = data["predictions"][target]
            if len(pred) != 1000:
                raise RuntimeError(f"Incomplete {source}/{target}")
            row[target] = 100 * sum(p != y for p, y in zip(pred, data["labels"])) / 1000
        row["overall"] = sum(row[t] for t in TARGETS) / 13
        row["black_box"] = sum(row[t] for t in TARGETS if t != model) / 12
        rows.append(row)
        with (OUT / "transfer_asr.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", *TARGETS, "overall", "black_box"])
            writer.writeheader()
            writer.writerows(rows)
        print(row, flush=True)
    print("Complete: " + str(OUT / "transfer_asr.csv"), flush=True)

if __name__ == "__main__":
    main()
