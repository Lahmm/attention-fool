"""Run 100-sample ablations for LM-DSS sign-filter + rotation + MI."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
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
CONTROL_BRANCH = "control_lm_dss_sign_filter_rotation_mi"


@dataclass(frozen=True)
class Branch:
    name: str
    description: str
    guide_aug_area: str = "background"
    mi: bool = True
    rotation: bool = True


BRANCHES = (
    Branch(
        name=CONTROL_BRANCH,
        description="LM-DSS sign-filter + low/mid rotation + MI, background guide augmentation.",
    ),
    Branch(
        name="ablate_aug_all",
        description="Only change guide augmentation area from background to all.",
        guide_aug_area="all",
    ),
    Branch(
        name="ablate_no_mi",
        description="Only remove MI momentum.",
        mi=False,
    ),
    Branch(
        name="ablate_sign_filter_only",
        description="Only remove low/mid rotation, keeping sign-filter and MI.",
        rotation=False,
    ),
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


def branch_attack_flags(branch: Branch) -> list[str]:
    flags = [
        "--lowmid-dss-filter",
        "--lowmid-dss-consistency",
        "sign",
        "--lowmid-dss-agreement-threshold",
        "0.67",
    ]
    if branch.mi:
        flags.extend(["--mi", "--mi-decay", "1.0"])
    if branch.rotation:
        flags.extend(["--lowmid-grad-tuning", "--lowmid-grad-rotation-strength", "0.5"])
    return flags


def build_attack_cmd(repo: Path, args: argparse.Namespace, branch: Branch, adv_dir: Path) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--mode",
        "attack",
        "--max-attacked-samples",
        str(args.max_samples),
        "--steps",
        str(args.steps),
        "--ti-sigma",
        "0",
        "--dim",
        "--guide-aug",
        "--guide-aug-area",
        branch.guide_aug_area,
        "--guide-aug-method",
        GUIDE_METHODS,
        "--guide-aug-copies",
        "3",
        "--guide-aug-strength",
        "0.2",
        "--attention-guide-models",
        GUIDE_MODELS,
        "--attention-guide-type",
        "qk_cls",
        "--attention-guide-build-method",
        "patch",
        "--layers",
        "0,1,4,9,11",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--output-dir",
        adv_dir.relative_to(repo).as_posix(),
        *branch_attack_flags(branch),
    ]


def build_eval_cmd(repo: Path, args: argparse.Namespace, branch: Branch, adv_dir: Path) -> list[str]:
    return [
        sys.executable,
        "transfer_eval.py",
        "--image-dir",
        adv_dir.relative_to(repo).as_posix(),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--amp",
        "--exp-name",
        branch.name,
    ]


def build_conclusion(report: dict[str, object]) -> str:
    branches = report["branches"]
    control = branches[CONTROL_BRANCH]
    best_name = max(branches, key=lambda name: branches[name]["avg"])
    lines = [
        "# LM-DSS Sign-Filter Rotation MI 100 样本消融结论",
        "",
        f"- 样本数: `{report['max_samples']}`。",
        f"- control Avg ASR: `{control['avg']:.4f}`。",
    ]
    for name, title in (
        ("ablate_aug_all", "全图增强"),
        ("ablate_no_mi", "不加 MI"),
        ("ablate_sign_filter_only", "只做 sign-filter"),
    ):
        result = branches[name]
        lines.append(
            f"- {title} Avg ASR: `{result['avg']:.4f}`，相对 control `{result['delta_vs_control']:+.4f}`。"
        )
    lines.extend(["", "## 解释", ""])
    if branches["ablate_aug_all"]["delta_vs_control"] > 0:
        lines.append("全图增强在这 100 个样本上优于 background-only，说明当前 guide 区域限制可能过强，需要进一步复验。")
    else:
        lines.append("全图增强没有超过 background-only，说明当前收益主要来自 background 区域的定向扰动，而不是简单扩大增强区域。")
    if branches["ablate_no_mi"]["delta_vs_control"] > 0:
        lines.append("去掉 MI 后 ASR 更高，说明当前 rotation/sign-filter 的单步方向可能已经足够稳定，MI 可能稀释了部分有效变化。")
    else:
        lines.append("去掉 MI 后 ASR 下降，说明 MI 对当前 LM-DSS sign-filter rotation 分支仍然是有效组成。")
    if branches["ablate_sign_filter_only"]["delta_vs_control"] > 0:
        lines.append("只做 sign-filter 更好，说明 low/mid rotation 在该设置下可能引入了与迁移方向不一致的额外旋转。")
    else:
        lines.append("只做 sign-filter 没有超过 control，说明 rotation 在 sign-filter 后仍提供了额外迁移收益。")
    lines.append(f"本轮最佳分支是 `{best_name}`，Avg ASR 为 `{branches[best_name]['avg']:.4f}`。")
    return "\n".join(lines) + "\n"


def write_report(repo: Path, args: argparse.Namespace, branch_results: dict[str, dict[str, object]]) -> None:
    control_avg = branch_results[CONTROL_BRANCH]["avg"]
    for name, result in branch_results.items():
        delta = float(result["avg"] - control_avg)
        result["delta_vs_control"] = delta
        result["relative_delta_vs_control"] = 0.0 if control_avg == 0 else float(delta / control_avg)
    report = {
        "protocol": "lm_dss_sign_filter_ablation_s100_v1",
        "max_samples": args.max_samples,
        "steps": args.steps,
        "control_branch": CONTROL_BRANCH,
        "branches": branch_results,
        "target_models": TARGET_MODELS,
        "config": {
            "dim": True,
            "guide_aug_method": GUIDE_METHODS,
            "guide_aug_copies": 3,
            "guide_aug_strength": 0.2,
            "attention_guide_models": GUIDE_MODELS,
            "attention_guide_type": "qk_cls",
            "attention_guide_build_method": "patch",
            "layers": "0,1,4,9,11",
            "ti_sigma": 0,
            "normalize_grad": False,
            "lowmid_dss_filter": True,
            "lowmid_dss_consistency": "sign",
            "lowmid_dss_agreement_threshold": 0.67,
            "control": {
                "guide_aug_area": "background",
                "mi": True,
                "mi_decay": 1.0,
                "lowmid_grad_tuning": True,
                "lowmid_grad_rotation_strength": 0.5,
            },
        },
        "ablation_design": {branch.name: branch.description for branch in BRANCHES},
    }
    out = repo / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lm_dss_sign_filter_ablation_s100_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "lm_dss_sign_filter_ablation_s100_conclusion_zh.md").write_text(
        build_conclusion(report), encoding="utf-8"
    )
    metrics = out / "lm_dss_sign_filter_ablation_s100_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "branch",
            "avg",
            "delta_vs_control",
            "relative_delta_vs_control",
            *TARGET_MODELS,
            "csv",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in branch_results.items():
            row = {
                "branch": name,
                "avg": result["avg"],
                "delta_vs_control": result["delta_vs_control"],
                "relative_delta_vs_control": result["relative_delta_vs_control"],
                "csv": result["csv"],
            }
            row.update(result["models"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    attack_root = repo / "outputs" / "attack" / "lazyagg" / "lm_dss_sign_ablation_s100"
    branch_results = {}
    for branch in BRANCHES:
        adv_dir = attack_root / branch.name
        csv_path = csv_path_for_adv_dir(repo, adv_dir)
        if args.skip_existing and csv_path.exists():
            print(f"skip existing branch: {branch.name} ({csv_path})", flush=True)
        else:
            run(build_attack_cmd(repo, args, branch, adv_dir), dry_run=args.dry_run)
            run(build_eval_cmd(repo, args, branch, adv_dir), dry_run=args.dry_run)
        if not args.dry_run:
            branch_results[branch.name] = read_asr_csv(csv_path)
    if not args.dry_run:
        write_report(repo, args, branch_results)


if __name__ == "__main__":
    main()
