"""Run the 100-sample LM-DSS low/mid filter experiment."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from causal_analysis import selected_batches, seed_all
from main import ANNOTATIONS_PATH, IMAGE_DIR, create_attacker
from nets import build_vit_model
from utils import DEVICE, load_data

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
    {"name": "baseline_no_rotation_mi", "extra": []},
    {"name": "rotation_mi", "extra": ["--lowmid-grad-tuning", "--lowmid-grad-rotation-strength", "0.5"]},
    {
        "name": "lm_dss_sign_filter_rotation_mi",
        "extra": [
            "--lowmid-dss-filter",
            "--lowmid-dss-consistency",
            "sign",
            "--lowmid-dss-agreement-threshold",
            "0.67",
            "--lowmid-grad-tuning",
            "--lowmid-grad-rotation-strength",
            "0.5",
        ],
    },
    {
        "name": "lm_dss_cos_filter_rotation_mi",
        "extra": [
            "--lowmid-dss-filter",
            "--lowmid-dss-consistency",
            "cos",
            "--lowmid-grad-tuning",
            "--lowmid-grad-rotation-strength",
            "0.5",
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


def build_conclusion(report: dict[str, object]) -> str:
    branches = report["branches"]
    rotation = branches.get("rotation_mi", {}).get("avg")
    lines = [
        "# LM-DSS Low/Mid 符号过滤 100 样本实验结论",
        "",
        f"- 样本数: {report['max_samples']}。",
        f"- baseline_no_rotation_mi Avg ASR: `{branches['baseline_no_rotation_mi']['avg']:.4f}`。",
        f"- rotation_mi Avg ASR: `{branches['rotation_mi']['avg']:.4f}`。",
        f"- lm_dss_sign_filter_rotation_mi Avg ASR: `{branches['lm_dss_sign_filter_rotation_mi']['avg']:.4f}`，相对 rotation `{branches['lm_dss_sign_filter_rotation_mi']['delta_vs_rotation']:+.4f}`。",
        f"- lm_dss_cos_filter_rotation_mi Avg ASR: `{branches['lm_dss_cos_filter_rotation_mi']['avg']:.4f}`，相对 rotation `{branches['lm_dss_cos_filter_rotation_mi']['delta_vs_rotation']:+.4f}`。",
        "",
        "## 解释",
        "",
    ]
    best_name = max(branches, key=lambda name: branches[name]["avg"])
    if best_name.startswith("lm_dss") and rotation is not None:
        lines.append("LM-DSS filter 在 100 样本上优于单纯 rotation，建议进入 500 样本复验。")
    else:
        lines.append("LM-DSS filter 在 100 样本上没有超过单纯 rotation，当前应保留为机制诊断或继续调整阈值。")
    return "\n".join(lines) + "\n"


def tensor_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).flatten(1).sum(1)


def tensor_norm(a: torch.Tensor) -> torch.Tensor:
    return a.flatten(1).norm(p=2, dim=1)


def sign_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.sign().eq(b.sign()).to(torch.float32).flatten(1).mean(1)


def band_components(attacker, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lowmid = sum((attacker._fft_project_grad(grad, band) for band in range(6)), torch.zeros_like(grad))
    high = sum((attacker._fft_project_grad(grad, band) for band in range(6, 8)), torch.zeros_like(grad))
    return lowmid, high


def lowmid_ratio(attacker, grad: torch.Tensor) -> torch.Tensor:
    lowmid, high = band_components(attacker, grad)
    lowmid_energy = lowmid.square().flatten(1).sum(1)
    high_energy = high.square().flatten(1).sum(1)
    return lowmid_energy / (lowmid_energy + high_energy).clamp_min(1e-20)


def branch_extra(name: str) -> list[str]:
    for branch in BRANCHES:
        if branch["name"] == name:
            return list(branch["extra"])
    raise KeyError(name)


def make_diagnostic_attacker(num_classes: int, branch_name: str):
    source = build_vit_model(num_classes=num_classes, model_name="vit_base_patch16_224")
    guides = tuple(build_vit_model(num_classes=num_classes, model_name=name) for name in GUIDE_MODELS.split(","))
    kwargs = {
        "model": source,
        "epsilon": 16 / 255,
        "step_size": None,
        "steps": 40,
        "layers": (0, 1, 4, 9, 11),
        "ti_sigma": 0,
        "dim": True,
        "mi": True,
        "mi_decay": 1.0,
        "normalize_grad": False,
        "attention_guide_models": guides,
        "attention_guide_type": "qk_cls",
        "attention_guide_build_method": "patch",
        "attention_guide_patch_size": 16,
        "guide_aug": True,
        "guide_aug_area": "background",
        "guide_aug_methods": tuple(GUIDE_METHODS.split(",")),
        "guide_aug_copies": 3,
        "guide_aug_strength": 0.2,
    }
    extra = branch_extra(branch_name)
    if "--lowmid-grad-tuning" in extra:
        kwargs["lowmid_grad_tuning"] = True
        kwargs["lowmid_grad_rotation_strength"] = 0.5
    if "--lowmid-dss-filter" in extra:
        kwargs["lowmid_dss_filter"] = True
        kwargs["lowmid_dss_consistency"] = extra[extra.index("--lowmid-dss-consistency") + 1]
        if "--lowmid-dss-agreement-threshold" in extra:
            kwargs["lowmid_dss_agreement_threshold"] = float(extra[extra.index("--lowmid-dss-agreement-threshold") + 1])
    return source, create_attacker(**kwargs)


def trace_source_diagnostics(attacker, images, labels, branch_name: str) -> list[dict[str, object]]:
    images, labels = images.to(attacker.device), labels.to(attacker.device)
    clean = attacker._denormalize(images).detach()
    guide = attacker._build_guide_pixel_map(images, clean.size(-1))
    adv = clean.clone().detach()
    momentum = torch.zeros_like(clean)
    rows = []
    for step_idx in range(attacker.steps):
        step = step_idx + 1
        grad_pixels = adv.detach().requires_grad_(True)
        if attacker.lowmid_dss_filter:
            raw, term_grads = attacker._attack_grad_terms(grad_pixels, labels, guide)
        else:
            raw, term_grads = attacker._attack_grad(grad_pixels, labels, guide), None
        raw = attacker._smooth_grad(attacker._normalize_guided_grad(raw, guide))
        if term_grads is not None:
            term_grads = tuple(attacker._smooth_grad(attacker._normalize_guided_grad(term, guide)) for term in term_grads)
        filtered = attacker._apply_lowmid_dss_filter(raw, term_grads)
        rotated = attacker._tune_lowmid_gradient(filtered)
        momentum = attacker.decay * momentum + rotated
        update = momentum
        raw_ratio = lowmid_ratio(attacker, raw)
        filtered_ratio = lowmid_ratio(attacker, filtered)
        rotated_ratio = lowmid_ratio(attacker, rotated)
        for idx in range(images.size(0)):
            rows.append({
                "branch": branch_name,
                "step": step,
                "raw_lowmid_ratio": float(raw_ratio[idx]),
                "filtered_lowmid_ratio": float(filtered_ratio[idx]),
                "rotated_lowmid_ratio": float(rotated_ratio[idx]),
                "filter_delta_lowmid_ratio": float(filtered_ratio[idx] - raw_ratio[idx]),
                "rotation_delta_lowmid_ratio": float(rotated_ratio[idx] - filtered_ratio[idx]),
                "sign_raw_filtered_agree": float(sign_agreement(raw, filtered)[idx]),
                "sign_filtered_rotated_agree": float(sign_agreement(filtered, rotated)[idx]),
                "sign_filtered_momentum_after_agree": float(sign_agreement(filtered, momentum)[idx]),
                "raw_update_sign_change": float(1.0 - sign_agreement(raw, update)[idx]),
            })
        with torch.no_grad():
            adv = adv + attacker.step_size * update.sign()
            delta = torch.clamp(adv - clean, -attacker.epsilon, attacker.epsilon)
            adv = torch.clamp(clean + delta, 0.0, 1.0)
    return rows


def collect_source_diagnostics(repo: Path, args) -> dict[str, dict[str, float]]:
    seed_all(0)
    loader, num_classes = load_data(IMAGE_DIR, ANNOTATIONS_PATH, args.batch_size, args.num_workers, args.prefetch_factor, 224)
    source = build_vit_model(num_classes=num_classes, model_name="vit_base_patch16_224")
    selector_args = Namespace(max_samples=args.max_samples)
    samples = [(x.cpu(), y.cpu()) for x, y, _idx in selected_batches(selector_args, source, loader)]
    del source, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    for branch in BRANCHES:
        branch_name = branch["name"]
        source, attacker = make_diagnostic_attacker(num_classes, branch_name)
        for images, labels in samples:
            rows.extend(trace_source_diagnostics(attacker, images, labels, branch_name))
        del source, attacker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = repo / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "lm_dss_lowmid_filter_s100_source_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else ["branch"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    keys = [key for key in rows[0] if key not in ("branch", "step")] if rows else []
    for branch in BRANCHES:
        branch_rows = [row for row in rows if row["branch"] == branch["name"]]
        summary[branch["name"]] = {key: float(np.mean([row[key] for row in branch_rows])) for key in keys}
    return summary


def write_report(repo: Path, args, branch_results: dict[str, dict[str, object]], source_summary: dict[str, dict[str, float]] | None = None) -> None:
    rotation_avg = branch_results["rotation_mi"]["avg"]
    for result in branch_results.values():
        result["delta_vs_rotation"] = float(result["avg"] - rotation_avg)
    report = {
        "protocol": "lm_dss_lowmid_filter_s100_v1",
        "max_samples": args.max_samples,
        "steps": args.steps,
        "branches": branch_results,
        "source_summary": source_summary or {},
        "target_models": TARGET_MODELS,
        "config": {
            "dim": True,
            "guide_aug_area": "background",
            "guide_aug_method": GUIDE_METHODS,
            "guide_aug_copies": 3,
            "attention_guide_models": GUIDE_MODELS,
            "attention_guide_type": "qk_cls",
            "attention_guide_build_method": "patch",
            "layers": "0,1,4,9,11",
            "ti_sigma": 0,
            "normalize_grad": False,
            "mi": True,
            "mi_decay": 1.0,
        },
    }
    out = repo / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lm_dss_lowmid_filter_s100_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "lm_dss_lowmid_filter_s100_conclusion_zh.md").write_text(
        build_conclusion(report), encoding="utf-8"
    )
    metrics = out / "lm_dss_lowmid_filter_s100_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["branch", "avg", "delta_vs_rotation", *TARGET_MODELS, "csv"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in branch_results.items():
            row = {
                "branch": name,
                "avg": result["avg"],
                "delta_vs_rotation": result["delta_vs_rotation"],
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
    parser.add_argument("--skip-source-diagnostics", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    attack_root = repo / "outputs" / "attack" / "lazyagg" / "lm_dss_s100"
    branch_results = {}
    for branch in BRANCHES:
        name = branch["name"]
        adv_dir = attack_root / name
        csv_path = csv_path_for_adv_dir(repo, adv_dir)
        if not (args.skip_existing and csv_path.exists()):
            attack_cmd = [
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
                "--mi",
                "--mi-decay",
                "1.0",
                "--guide-aug",
                "--guide-aug-area",
                "background",
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
                adv_dir.as_posix(),
                *branch["extra"],
            ]
            run(attack_cmd, dry_run=args.dry_run)
            eval_cmd = [
                sys.executable,
                "transfer_eval.py",
                "--image-dir",
                adv_dir.as_posix(),
                "--batch-size",
                str(args.eval_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--prefetch-factor",
                str(args.prefetch_factor),
                "--amp",
                "--exp-name",
                name,
            ]
            run(eval_cmd, dry_run=args.dry_run)
        if not args.dry_run:
            branch_results[name] = read_asr_csv(csv_path)
    if not args.dry_run:
        source_summary = {} if args.skip_source_diagnostics else collect_source_diagnostics(repo, args)
        write_report(repo, args, branch_results, source_summary)


if __name__ == "__main__":
    main()
