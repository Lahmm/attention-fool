import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODEL_COLS = [
    "deit_base_patch16_224",
    "beit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "pvt_v2_b2",
    "cait_s24_224",
    "levit_256",
    "pit_s_224",
    "crossvit_15_240",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Record transfer-eval results into an Excel file.")
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


def excel_path_for_adv_dir(repo_path: Path, adv_dir: Path) -> Path:
    relative = adv_dir.resolve().relative_to(repo_path.resolve())
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", relative.as_posix()).strip("_")
    excel_dir = repo_path / "outputs" / "excel"
    excel_dir.mkdir(parents=True, exist_ok=True)
    return excel_dir / f"{stem}.xlsx"


def main() -> None:
    args = parse_args()
    repo_path = Path(args.repo_path).expanduser().resolve()
    adv_dir = validate_adv_dir(repo_path, args.adv_dir)
    params_dict = json.loads(args.params)

    eval_output = sys.stdin.read()
    results = parse_eval_output(eval_output)
    if not results:
        print("ERROR: no results parsed")
        sys.exit(1)

    avg_asr = sum(results.values()) / len(results)
    relative_adv_dir = adv_dir.relative_to(repo_path)
    results["avg"] = avg_asr
    results["git_head"] = get_git_head(repo_path)
    results["exp_name"] = args.exp_name
    results["adv_dir"] = relative_adv_dir.as_posix()
    results["adv_image_count"] = count_images(adv_dir)
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results.update(params_dict)

    import pandas as pd

    df = pd.DataFrame([results])
    meta_cols = ["exp_name", "timestamp", "git_head", "adv_dir", "adv_image_count", "avg"]
    param_cols = [key for key in params_dict.keys() if key not in MODEL_COLS + meta_cols]
    final_cols = [col for col in (meta_cols + param_cols + MODEL_COLS) if col in df.columns]
    df = df[final_cols]

    excel_path = excel_path_for_adv_dir(repo_path, adv_dir)
    if excel_path.exists():
        existing = pd.read_excel(excel_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_excel(excel_path, index=False)
    print(f"Saved to {excel_path}")
    print(f"Avg ASR: {avg_asr:.4f}")
    for model_name in MODEL_COLS:
        if model_name in results:
            print(f"  {model_name}: {results[model_name]:.4f}")


if __name__ == "__main__":
    main()
