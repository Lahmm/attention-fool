"""Run no-step-projection 100-sample MI/no-MI ablations for LM-DSS."""
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
PROJECTION_ON_MI_AVG_ASR = 0.8375
PROJECTION_ON_NO_MI_AVG_ASR = 0.6613


@dataclass(frozen=True)
class Branch:
    name: str
    description: str
    mi: bool = True


BRANCHES = (
    Branch(
        name="no_step_projection_aug_all_mi",
        description="No per-step epsilon projection, all-area guide augmentation, LM-DSS sign-filter + low/mid rotation + MI.",
        mi=True,
    ),
    Branch(
        name="no_step_projection_aug_all_no_mi",
        description="No per-step epsilon projection, all-area guide augmentation, LM-DSS sign-filter + low/mid rotation, no MI.",
        mi=False,
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
        "--no-step-projection",
        "--dim",
        "--guide-aug",
        "--guide-aug-area",
        "all",
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
    mi = branches["no_step_projection_aug_all_mi"]
    no_mi = branches["no_step_projection_aug_all_no_mi"]
    gap = report["mi_minus_no_mi_gap"]
    projection_on_gap = report["projection_on_reference"]["mi_minus_no_mi_gap"]
    lines = [
        "# No-Step-Projection 下 MI / no-MI 100 样本消融结论",
        "",
        f"- 样本数: `{report['max_samples']}`。",
        f"- no-step-projection + MI Avg ASR: `{mi['avg']:.4f}`。",
        f"- no-step-projection + no-MI Avg ASR: `{no_mi['avg']:.4f}`。",
        f"- no-step-projection MI - no-MI gap: `{gap:+.4f}`。",
        f"- projection-on 参考 gap: `{projection_on_gap:+.4f}`，其中 MI `{PROJECTION_ON_MI_AVG_ASR:.4f}`，no-MI `{PROJECTION_ON_NO_MI_AVG_ASR:.4f}`。",
        "",
        "## 解释",
        "",
    ]
    if gap > projection_on_gap:
        lines.append("去掉每步 epsilon projection 后，MI/no-MI 差距变大，说明无 projection 加重了 no-MI 历史梯度方向保留不足的问题。")
    elif gap < projection_on_gap:
        lines.append("去掉每步 epsilon projection 后，MI/no-MI 差距变小，说明 `x_t` 中累积的扰动历史可能部分补偿了 no-MI 缺失的梯度历史。")
    else:
        lines.append("去掉每步 epsilon projection 后，MI/no-MI 差距与 projection-on 参考一致，说明 projection 开关没有明显改变 MI 依赖强度。")
    lines.append("机制判断需要结合 `mi_no_step_projection_mechanism.py` 的 target alignment 与 history component 指标。")
    return "\n".join(lines) + "\n"


def write_report(repo: Path, args: argparse.Namespace, branch_results: dict[str, dict[str, object]]) -> None:
    mi_avg = float(branch_results["no_step_projection_aug_all_mi"]["avg"])
    no_mi_avg = float(branch_results["no_step_projection_aug_all_no_mi"]["avg"])
    gap = mi_avg - no_mi_avg
    projection_on_gap = PROJECTION_ON_MI_AVG_ASR - PROJECTION_ON_NO_MI_AVG_ASR
    for result in branch_results.values():
        result["delta_vs_no_step_mi"] = float(result["avg"] - mi_avg)
    report = {
        "protocol": "lm_dss_no_step_projection_mi_ablation_s100_v1",
        "max_samples": args.max_samples,
        "steps": args.steps,
        "branches": branch_results,
        "mi_avg_asr": mi_avg,
        "no_mi_avg_asr": no_mi_avg,
        "mi_minus_no_mi_gap": gap,
        "projection_on_reference": {
            "mi_avg_asr": PROJECTION_ON_MI_AVG_ASR,
            "no_mi_avg_asr": PROJECTION_ON_NO_MI_AVG_ASR,
            "mi_minus_no_mi_gap": projection_on_gap,
            "gap_delta_no_step_minus_projection_on": gap - projection_on_gap,
        },
        "target_models": TARGET_MODELS,
        "config": {
            "project_each_step": False,
            "dim": True,
            "guide_aug_method": GUIDE_METHODS,
            "guide_aug_area": "all",
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
            "lowmid_grad_tuning": True,
            "lowmid_grad_rotation_strength": 0.5,
        },
        "ablation_design": {branch.name: branch.description for branch in BRANCHES},
    }
    out = repo / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lm_dss_no_step_projection_mi_ablation_s100_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "lm_dss_no_step_projection_mi_ablation_s100_conclusion_zh.md").write_text(
        build_conclusion(report), encoding="utf-8"
    )
    metrics = out / "lm_dss_no_step_projection_mi_ablation_s100_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "branch",
            "avg",
            "delta_vs_no_step_mi",
            *TARGET_MODELS,
            "csv",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in branch_results.items():
            row = {
                "branch": name,
                "avg": result["avg"],
                "delta_vs_no_step_mi": result["delta_vs_no_step_mi"],
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
    attack_root = repo / "outputs" / "attack" / "lazyagg" / "lm_dss_no_step_projection_s100"
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
