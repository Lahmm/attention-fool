import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_COLS = [
    "levit_256",
    "pit_b_224",
    "deit_base_patch16_224",
    "tnt_s_patch16_224",
    "convit_base",
    "visformer_small",
    "cait_s24_224",
    "inception_v3",
    "inception_v4",
    "inception_resnet_v2",
    "resnet101",
    "inception_v3_adv",
    "inception_resnet_v2_adv",
    "inception_v3_adv_3",
    "inception_v3_adv_4",
]
VIT_MODEL_COLS = set(MODEL_COLS[:7])
CNN_MODEL_COLS = set(MODEL_COLS[7:])
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Record transfer-eval results into an csv file.")
    parser.add_argument("repo_path", help="Repository root.")
    parser.add_argument("exp_name", help="Experiment name.")
    parser.add_argument("params", help="JSON object with experiment parameters.")
    parser.add_argument(
        "adv_dir",
        help="Adversarial sample directory under outputs/attack, e.g. outputs/attack/fftcc.",
    )
    return parser.parse_args()


def get_git_head(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()[:8]


def parse_eval_output(output: str) -> dict[str, float]:
    results = {}
    for line in output.strip().split("\n"):
        if line.startswith("model="):
            parts = line.split()
            model = parts[0].split("=")[1]
            for part in parts:
                if part.startswith("ASR="):
                    results[model] = float(part.split("=")[1])
    return results


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def architecture_avg_asr(asr_by_model: dict[str, float]) -> dict[str, float | None]:
    vit_asrs = [
        float(value) for model, value in asr_by_model.items() if model in VIT_MODEL_COLS
    ]
    cnn_asrs = [
        float(value) for model, value in asr_by_model.items() if model in CNN_MODEL_COLS
    ]
    return {
        "avg_vit": average(vit_asrs),
        "avg_cnn": average(cnn_asrs),
    }


def format_avg(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def validate_adv_dir(repo_path: Path, adv_dir_arg: str) -> Path:
    repo_root = repo_path.resolve()
    adv_dir = Path(adv_dir_arg).expanduser()
    if not adv_dir.is_absolute():
        adv_dir = repo_root / adv_dir
    adv_dir = adv_dir.resolve()

    attack_root = (repo_root / "outputs" / "attack").resolve()
    try:
        adv_dir.relative_to(attack_root)
    except ValueError as exc:
        raise ValueError(f"adv_dir must be under {attack_root}: {adv_dir}") from exc

    if not adv_dir.is_dir():
        raise ValueError(f"adv_dir does not exist or is not a directory: {adv_dir}")

    image_count = count_images(adv_dir)
    if image_count == 0:
        raise ValueError(f"adv_dir contains no adversarial image files: {adv_dir}")

    return adv_dir


def count_images(adv_dir: Path) -> int:
    return sum(
        1
        for path in adv_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def csv_path_for_adv_dir(repo_path: Path, adv_dir: Path) -> Path:
    relative = adv_dir.resolve().relative_to(repo_path.resolve())
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", relative.as_posix()).strip("_")
    csv_dir = repo_path / "outputs" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir / f"{stem}.csv"


def record_results(
    repo_path: Path,
    exp_name: str,
    params_dict: dict[str, Any],
    adv_dir_arg: str | Path,
    asr_by_model: dict[str, float],
) -> Path:
    """Record ASR values defined as one minus adversarial-set accuracy."""
    repo_path = repo_path.expanduser().resolve()
    adv_dir = validate_adv_dir(repo_path, str(adv_dir_arg))
    results: dict[str, Any] = dict(asr_by_model)
    if not results:
        raise ValueError("no ASR results to record")

    avg_asr = sum(float(value) for value in asr_by_model.values()) / len(asr_by_model)
    architecture_avgs = architecture_avg_asr(asr_by_model)
    relative_adv_dir = adv_dir.relative_to(repo_path)
    results["avg"] = avg_asr
    results.update(architecture_avgs)
    results["git_head"] = get_git_head(repo_path)
    results["exp_name"] = exp_name
    results["adv_dir"] = relative_adv_dir.as_posix()
    results["adv_image_count"] = count_images(adv_dir)
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results.update(params_dict)

    import pandas as pd

    df = pd.DataFrame([results])
    meta_cols = [
        "exp_name",
        "timestamp",
        "git_head",
        "adv_dir",
        "adv_image_count",
        "avg",
        "avg_vit",
        "avg_cnn",
    ]
    dynamic_model_cols = MODEL_COLS + [
        key for key in asr_by_model.keys() if key not in MODEL_COLS
    ]
    param_cols = [key for key in params_dict.keys() if key not in dynamic_model_cols + meta_cols]
    final_cols = [col for col in (meta_cols + param_cols + dynamic_model_cols) if col in df.columns]
    df = df[final_cols]

    csv_path = csv_path_for_adv_dir(repo_path, adv_dir)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")
    print(f"Avg ASR: {avg_asr:.4f}")
    print(f"ViT Avg ASR: {format_avg(architecture_avgs['avg_vit'])}")
    print(f"CNN Avg ASR: {format_avg(architecture_avgs['avg_cnn'])}")
    for model_name in MODEL_COLS:
        if model_name in results:
            print(f"  {model_name}: {results[model_name]:.4f}")
    return csv_path


def main() -> None:
    args = parse_args()
    repo_path = Path(args.repo_path).expanduser().resolve()
    params_dict = json.loads(args.params)

    eval_output = sys.stdin.read()
    results = parse_eval_output(eval_output)
    if not results:
        print("ERROR: no results parsed")
        sys.exit(1)

    record_results(
        repo_path=repo_path,
        exp_name=args.exp_name,
        params_dict=params_dict,
        adv_dir_arg=args.adv_dir,
        asr_by_model=results,
    )


if __name__ == "__main__":
    main()
