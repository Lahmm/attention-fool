"""Cross-timestep LM-DSS 500-sample transfer experiment.

Compares four branches:
  baseline_mi                        – DIM + MI + guide_aug (no rotation)
  rotation_mi                        – DIM + MI + guide_aug + low/mid rotation
  cross_step_dss_sign_rotation_mi    – cross-timestep DSS (sign) + rotation
  cross_step_dss_cos_rotation_mi     – cross-timestep DSS (cos) + rotation
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

TARGET_MODELS = (
    "deit_base_patch16_224",
    "beit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "pvt_v2_b2",
    "cait_s24_224",
    "levit_256",
    "pit_s_224",
    "crossvit_15_240",
)
GUIDE_MODELS = "deit_base_patch16_224,pit_s_224,cait_s24_224"
GUIDE_METHODS = "dropout,jitter,freq"

BRANCHES = (
    {
        "name": "baseline_mi",
        "description": "DIM + MI + guide_aug background, no rotation, no DSS.",
        "extra": [],
    },
    {
        "name": "rotation_mi",
        "description": "DIM + MI + guide_aug background + low/mid rotation (strength=0.5).",
        "extra": [
            "--lowmid-grad-tuning",
            "--lowmid-grad-rotation-strength", "0.5",
        ],
    },
    {
        "name": "cross_step_dss_sign_rotation_mi",
        "description": "Cross-timestep LM-DSS (sign, thresh=0.67) + low/mid rotation + MI.",
        "extra": [
            "--lowmid-dss-filter",
            "--lowmid-dss-consistency", "sign",
            "--lowmid-dss-agreement-threshold", "0.67",
            "--lowmid-grad-tuning",
            "--lowmid-grad-rotation-strength", "0.5",
        ],
    },
    {
        "name": "cross_step_dss_cos_rotation_mi",
        "description": "Cross-timestep LM-DSS (cos) + low/mid rotation + MI.",
        "extra": [
            "--lowmid-dss-filter",
            "--lowmid-dss-consistency", "cos",
            "--lowmid-grad-tuning",
            "--lowmid-grad-rotation-strength", "0.5",
        ],
    },
)


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def csv_path_for_adv_dir(repo: Path, adv_dir: Path) -> Path:
    relative = adv_dir.resolve().relative_to(repo.resolve())
    stem = relative.as_posix().replace("/", "_")
    return repo / "outputs" / "csv" / f"{stem}.csv"


def read_asr_csv(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty ASR csv: {path}")
    row = rows[-1]
    return {
        "csv": path.as_posix(),
        "avg": float(row["avg"]),
        "models": {model: float(row[model]) for model in TARGET_MODELS if model in row},
    }


def build_attack_cmd(
    repo: Path,
    args: argparse.Namespace,
    branch: dict,
    adv_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--mode", "attack",
        "--max-attacked-samples", str(args.max_samples),
        "--steps", str(args.steps),
        "--ti-sigma", "0",
        "--dim",
        "--mi",
        "--mi-decay", "1.0",
        "--guide-aug",
        "--guide-aug-area", "background",
        "--guide-aug-method", GUIDE_METHODS,
        "--guide-aug-copies", "3",
        "--guide-aug-strength", "0.2",
        "--attention-guide-models", GUIDE_MODELS,
        "--attention-guide-type", "qk_cls",
        "--attention-guide-build-method", "patch",
        "--layers", "0,1,4,9,11",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--prefetch-factor", str(args.prefetch_factor),
        "--output-dir", adv_dir.relative_to(repo).as_posix(),
        *branch["extra"],
    ]


def build_eval_cmd(
    repo: Path,
    args: argparse.Namespace,
    branch: dict,
    adv_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "transfer_eval.py",
        "--image-dir", adv_dir.relative_to(repo).as_posix(),
        "--batch-size", str(args.eval_batch_size),
        "--num-workers", str(args.num_workers),
        "--prefetch-factor", str(args.prefetch_factor),
        "--amp",
        "--exp-name", branch["name"],
    ]


def build_conclusion(report: dict) -> str:
    branches = report["branches"]
    baseline = branches["baseline_mi"]["avg"]
    rotation = branches["rotation_mi"]["avg"]

    lines = [
        "# Cross-Timestep LM-DSS 500 样本迁移实验结论",
        "",
        f"- 样本数: `{report['max_samples']}`。",
        f"- 步数: `{report['steps']}`。",
        f"- baseline_mi (DIM+MI+guide_aug) Avg ASR: `{baseline:.4f}`",
        f"- rotation_mi (+ rotation) Avg ASR: `{rotation:.4f}`, Δbaseline `{rotation - baseline:+.4f}`",
    ]

    for name in ("cross_step_dss_sign_rotation_mi", "cross_step_dss_cos_rotation_mi"):
        result = branches[name]
        lines.append(
            f"- {name} Avg ASR: `{result['avg']:.4f}`, "
            f"Δbaseline `{result['avg'] - baseline:+.4f}`, "
            f"Δrotation `{result['avg'] - rotation:+.4f}`"
        )

    lines.extend(["", "## 解释", ""])

    sign_result = branches["cross_step_dss_sign_rotation_mi"]
    cos_result = branches["cross_step_dss_cos_rotation_mi"]

    best_name = max(branches, key=lambda n: branches[n]["avg"])
    if best_name in ("cross_step_dss_sign_rotation_mi", "cross_step_dss_cos_rotation_mi"):
        lines.append(
            f"跨时间步 LM-DSS ({best_name}) 在 500 样本上优于单纯 rotation "
            f"({branches[best_name]['avg'] - rotation:+.4f})，"
            f"说明将一致性参考从'跨增强样本'切换为'跨优化步动量'有效解决了与输入多样性的矛盾。"
        )
    else:
        lines.append(
            "跨时间步 LM-DSS 在 500 样本上没有超过单纯 rotation，需要进一步分析原因。"
        )

    if sign_result["avg"] > cos_result["avg"]:
        lines.append("Sign 一致性在该设置下优于 cos 一致性。")
    else:
        lines.append("Cos 一致性在该设置下优于 sign 一致性。")

    lines.append("")
    lines.append("## 逐模型 ASR")
    lines.append("")
    lines.append("| Model | baseline_mi | rotation_mi | sign_dss | cos_dss |")
    lines.append("|-------|------------|-------------|----------|---------|")
    for model in TARGET_MODELS:
        b = branches["baseline_mi"]["models"].get(model, 0)
        r = branches["rotation_mi"]["models"].get(model, 0)
        s = branches["cross_step_dss_sign_rotation_mi"]["models"].get(model, 0)
        c = branches["cross_step_dss_cos_rotation_mi"]["models"].get(model, 0)
        lines.append(f"| {model} | {b:.4f} | {r:.4f} | {s:.4f} | {c:.4f} |")
    lines.append("")

    return "\n".join(lines)


def write_report(
    repo: Path,
    args: argparse.Namespace,
    branch_results: dict[str, dict[str, object]],
) -> None:
    report = {
        "protocol": "cross_step_dss_s500_v1",
        "max_samples": args.max_samples,
        "steps": args.steps,
        "branches": branch_results,
        "target_models": list(TARGET_MODELS),
        "config": {
            "dim": True,
            "mi": True,
            "mi_decay": 1.0,
            "guide_aug": True,
            "guide_aug_area": "background",
            "guide_aug_method": GUIDE_METHODS,
            "guide_aug_copies": 3,
            "guide_aug_strength": 0.2,
            "attention_guide_models": GUIDE_MODELS,
            "attention_guide_type": "qk_cls",
            "attention_guide_build_method": "patch",
            "layers": "0,1,4,9,11",
            "ti_sigma": 0,
            "normalize_grad": False,
            "project_each_step": True,
            "epsilon": "16/255",
        },
        "branch_descriptions": {b["name"]: b["description"] for b in BRANCHES},
    }

    out = repo / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_step_dss_s500_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "cross_step_dss_s500_conclusion_zh.md").write_text(
        build_conclusion(report), encoding="utf-8"
    )

    metrics = out / "cross_step_dss_s500_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["branch", "avg", *TARGET_MODELS, "csv"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in branch_results.items():
            row = {"branch": name, "avg": result["avg"], "csv": result["csv"]}
            row.update(result["models"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    attack_root = repo / "outputs" / "attack" / "lazyagg" / "cross_step_dss_s500"
    branch_results: dict[str, dict[str, object]] = {}

    for branch in BRANCHES:
        name = branch["name"]
        adv_dir = attack_root / name
        csv_path = csv_path_for_adv_dir(repo, adv_dir)

        if args.skip_existing and csv_path.exists():
            print(f"skip existing branch: {name} ({csv_path})", flush=True)
        else:
            run(build_attack_cmd(repo, args, branch, adv_dir), dry_run=args.dry_run)
            run(build_eval_cmd(repo, args, branch, adv_dir), dry_run=args.dry_run)

        if not args.dry_run:
            branch_results[name] = read_asr_csv(csv_path)

    if not args.dry_run:
        write_report(repo, args, branch_results)


if __name__ == "__main__":
    main()
