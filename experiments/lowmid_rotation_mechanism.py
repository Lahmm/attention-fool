"""Low/mid gradient-rotation mechanism experiment.

This script records matched attack-step gradients before and after
low/mid-gradient rotation, then relates the spectral change to momentum
dilution and target-gradient alignment.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attack import LMDSSAttacker
from causal_analysis import MAIN_TARGETS, _target_normalize, seed_all, selected_batches
from gradient_analysis import FFT_BANDS, fft_project
from main import ANNOTATIONS_PATH, IMAGE_DIR, create_attacker, parse_model_names
from nets import build_vit_model
from utils import DEVICE, load_data

PROTOCOL = "lowmid_rotation_mechanism_v1"
LOWMID_BANDS = tuple(range(6))
HIGH_BANDS = tuple(range(6, len(FFT_BANDS) - 1))
DEFAULT_TRACE_STEPS = tuple(range(1, 41))
DEFAULT_TARGET_STEPS = (1, 5, 10, 20, 40)
DEFAULT_GUIDES = ("deit_base_patch16_224", "pit_s_224", "cait_s24_224")
DEFAULT_ASR = {
    "rotation_mi": 0.7825,
    "rotation_no_mi": 0.6643,
    "best_no_rotation_mi": 0.8015,
}


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).flatten(1).sum(1)


def tensor_norm(a: torch.Tensor) -> torch.Tensor:
    return a.flatten(1).norm(p=2, dim=1)


def tensor_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return tensor_dot(a, b) / (tensor_norm(a) * tensor_norm(b)).clamp_min(eps)


def sign_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.sign().eq(b.sign()).to(torch.float32).flatten(1).mean(1)


def project_bands(x: torch.Tensor, bands: tuple[int, ...]) -> torch.Tensor:
    if not bands:
        return torch.zeros_like(x)
    return sum((fft_project(x, band) for band in bands), torch.zeros_like(x))


def band_energy(x: torch.Tensor, bands: tuple[int, ...]) -> torch.Tensor:
    return project_bands(x, bands).square().flatten(1).sum(1)


def spectrum_metrics(x: torch.Tensor) -> dict[str, torch.Tensor]:
    lowmid = band_energy(x, LOWMID_BANDS)
    high = band_energy(x, HIGH_BANDS)
    total = lowmid + high
    return {
        "lowmid_energy": lowmid,
        "high_energy": high,
        "total_energy": total,
        "total_norm": tensor_norm(x),
        "lowmid_ratio": lowmid / total.clamp_min(1e-20),
    }


def summarize_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def build_conclusion_zh(report: dict[str, object]) -> str:
    source = report["source_summary"]
    target = report.get("target_summary", {})
    mi = source.get("rotation/mi_on", {})
    no_mi = source.get("rotation/mi_off", {})
    mi_delta = mi.get("delta_lowmid_ratio")
    no_mi_delta = no_mi.get("delta_lowmid_ratio")
    mi_sign = mi.get("rotation_update_sign_change")
    no_mi_sign = no_mi.get("rotation_update_sign_change")
    target_delta = target.get("rotation/mi_on", {}).get("delta_target_cos")
    lines = [
        "# Low/Mid Rotation 机制实验结论",
        "",
        f"- 协议: `{report['protocol']}`。",
        f"- 已知 ASR: MI rotation Avg ASR = {DEFAULT_ASR['rotation_mi']:.4f}, no-MI rotation Avg ASR = {DEFAULT_ASR['rotation_no_mi']:.4f}, 历史最强 no-rotation MI Avg ASR = {DEFAULT_ASR['best_no_rotation_mi']:.4f}。",
        f"- MI 分支中，rotation 的平均 low/mid ratio 改变量为 `{mi_delta}`；no-MI 分支为 `{no_mi_delta}`。",
        f"- MI 分支中，rotation 对最终 `sign(update)` 的平均改变量为 `{mi_sign}`；no-MI 分支为 `{no_mi_sign}`。",
        f"- 目标模型 alignment 的 `cos(rot_grad,target)-cos(raw_grad,target)` 平均值为 `{target_delta}`。",
        "",
        "## 解释",
        "",
    ]
    if mi_delta is not None and mi_delta > 0:
        lines.append("rotation 确实提高了 source-side 输入梯度的中低频能量占比。")
    else:
        lines.append("当前记录没有支持 rotation 稳定提高 source-side 中低频能量占比。")
    if mi_sign is not None and no_mi_sign is not None and mi_sign < no_mi_sign:
        lines.append("MI 分支中，单步 rotation 的方向变化被 momentum 累积稀释，最终 `sign(update)` 的改变小于 no-MI。")
    if target_delta is not None and target_delta <= 0:
        lines.append("虽然频谱比例上升，但 target-gradient alignment 没有同步提升，说明频谱比例本身不是迁移性的充分条件。")
    else:
        lines.append("迁移性仍应结合 target-gradient alignment 判断；频谱比例本身不是迁移性的充分条件。")
    return "\n".join(lines) + "\n"


def csv_asr_summary(paths: tuple[str, ...]) -> dict[str, object]:
    result = {}
    for item in paths:
        path = Path(item)
        if not path.exists():
            result[path.name] = {"exists": False}
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        result[path.name] = {"exists": True, "rows": len(rows), "columns": list(rows[0]) if rows else []}
    return result


def make_attacker(num_classes: int, *, mi: bool, rotation: bool):
    source = build_vit_model(num_classes=num_classes, model_name="vit_base_patch16_224")
    guides = tuple(build_vit_model(num_classes=num_classes, model_name=name) for name in DEFAULT_GUIDES)
    attacker = create_attacker(
        model=source,
        epsilon=16 / 255,
        step_size=None,
        steps=40,
        layers=(0, 1, 4, 9, 11),
        ti_sigma=0,
        dim=True,
        mi=mi,
        mi_decay=1.0,
        attention_guide_models=guides,
        attention_guide_type="qk_cls",
        attention_guide_build_method="patch",
        attention_guide_patch_size=16,
        guide_aug=True,
        guide_aug_area="background",
        guide_aug_methods=("dropout", "jitter", "freq"),
        guide_aug_copies=3,
        lowmid_grad_tuning=rotation,
        lowmid_grad_rotation_strength=0.5,
    )
    return source, attacker


def trace_attack(attacker: LMDSSAttacker, images: torch.Tensor, labels: torch.Tensor, branch: str, keep_steps: set[int]):
    images, labels = images.to(attacker.device), labels.to(attacker.device)
    clean = attacker._denormalize(images).detach()
    needs_guide = attacker.guide_aug and attacker.guide_aug_area != "all"
    guide = attacker._build_guide_pixel_map(images, clean.size(-1)) if needs_guide else None
    adv = clean.clone().detach()
    momentum = torch.zeros_like(clean)
    rows = []
    for step_idx in range(attacker.steps):
        step = step_idx + 1
        grad_pixels = adv.detach().requires_grad_(True)
        raw = attacker._attack_grad(grad_pixels, labels, guide)
        raw = attacker._smooth_grad(raw)
        rot = attacker._tune_lowmid_gradient(raw)
        previous = momentum
        momentum = attacker.decay * momentum + rot
        update = momentum if attacker.use_momentum else rot
        if step in keep_steps:
            rows.append({
                "branch": branch,
                "step": step,
                "x_t": adv.detach().cpu(),
                "raw_grad": raw.detach().cpu(),
                "rot_grad": rot.detach().cpu(),
                "momentum_before": previous.detach().cpu(),
                "momentum_after": momentum.detach().cpu(),
                "update_grad": update.detach().cpu(),
                "guide_map": None if guide is None else guide.detach().cpu(),
            })
        with torch.no_grad():
            adv = adv + attacker.step_size * update.sign()
            adv = torch.clamp(adv, 0.0, 1.0)
    return rows


def source_rows_from_trace(trace_rows: list[dict[str, object]], indices: torch.Tensor) -> list[dict[str, object]]:
    rows = []
    for trace in trace_rows:
        raw = trace["raw_grad"]
        rot = trace["rot_grad"]
        update = trace["update_grad"]
        momentum = trace["momentum_after"]
        raw_s = spectrum_metrics(raw)
        rot_s = spectrum_metrics(rot)
        for sample_idx, image_idx in enumerate(indices.tolist()):
            rows.append({
                "scope": "source",
                "branch": trace["branch"],
                "step": trace["step"],
                "image_index": int(image_idx),
                "raw_lowmid_ratio": float(raw_s["lowmid_ratio"][sample_idx]),
                "rot_lowmid_ratio": float(rot_s["lowmid_ratio"][sample_idx]),
                "delta_lowmid_ratio": float(rot_s["lowmid_ratio"][sample_idx] - raw_s["lowmid_ratio"][sample_idx]),
                "raw_lowmid_energy": float(raw_s["lowmid_energy"][sample_idx]),
                "rot_lowmid_energy": float(rot_s["lowmid_energy"][sample_idx]),
                "raw_high_energy": float(raw_s["high_energy"][sample_idx]),
                "rot_high_energy": float(rot_s["high_energy"][sample_idx]),
                "raw_total_norm": float(raw_s["total_norm"][sample_idx]),
                "rot_total_norm": float(rot_s["total_norm"][sample_idx]),
                "cos_raw_rot": float(tensor_cos(raw, rot)[sample_idx]),
                "sign_raw_rot_agree": float(sign_agreement(raw, rot)[sample_idx]),
                "cos_rot_momentum_after": float(tensor_cos(rot, momentum)[sample_idx]),
                "sign_rot_momentum_after_agree": float(sign_agreement(rot, momentum)[sample_idx]),
                "rotation_update_sign_change": float(1.0 - sign_agreement(raw, update)[sample_idx]),
            })
    return rows


def target_alignment_rows(trace_rows, labels, indices, target_models):
    rows = []
    labels = labels.to(DEVICE)
    for model_name in target_models:
        model = build_vit_model(num_classes=1000, model_name=model_name).eval()
        for trace in trace_rows:
            pixels = trace["x_t"].to(DEVICE).detach().requires_grad_(True)
            raw = trace["raw_grad"].to(DEVICE)
            rot = trace["rot_grad"].to(DEVICE)
            loss = F.cross_entropy(model(_target_normalize(model, pixels), return_attn=False), labels)
            tgt = torch.autograd.grad(loss, pixels)[0].detach()
            raw_lm, rot_lm, tgt_lm = project_bands(raw, LOWMID_BANDS), project_bands(rot, LOWMID_BANDS), project_bands(tgt, LOWMID_BANDS)
            raw_hi, rot_hi, tgt_hi = project_bands(raw, HIGH_BANDS), project_bands(rot, HIGH_BANDS), project_bands(tgt, HIGH_BANDS)
            for sample_idx, image_idx in enumerate(indices.tolist()):
                rows.append({
                    "scope": "target",
                    "branch": trace["branch"],
                    "step": trace["step"],
                    "image_index": int(image_idx),
                    "target_model": model_name,
                    "raw_target_dot": float(tensor_dot(raw, tgt)[sample_idx]),
                    "rot_target_dot": float(tensor_dot(rot, tgt)[sample_idx]),
                    "delta_target_dot": float((tensor_dot(rot, tgt) - tensor_dot(raw, tgt))[sample_idx]),
                    "raw_target_cos": float(tensor_cos(raw, tgt)[sample_idx]),
                    "rot_target_cos": float(tensor_cos(rot, tgt)[sample_idx]),
                    "delta_target_cos": float((tensor_cos(rot, tgt) - tensor_cos(raw, tgt))[sample_idx]),
                    "raw_target_sign_agree": float(sign_agreement(raw, tgt)[sample_idx]),
                    "rot_target_sign_agree": float(sign_agreement(rot, tgt)[sample_idx]),
                    "raw_lowmid_target_dot": float(tensor_dot(raw_lm, tgt_lm)[sample_idx]),
                    "rot_lowmid_target_dot": float(tensor_dot(rot_lm, tgt_lm)[sample_idx]),
                    "raw_lowmid_target_cos": float(tensor_cos(raw_lm, tgt_lm)[sample_idx]),
                    "rot_lowmid_target_cos": float(tensor_cos(rot_lm, tgt_lm)[sample_idx]),
                    "raw_high_target_dot": float(tensor_dot(raw_hi, tgt_hi)[sample_idx]),
                    "rot_high_target_dot": float(tensor_dot(rot_hi, tgt_hi)[sample_idx]),
                    "raw_high_target_cos": float(tensor_cos(raw_hi, tgt_hi)[sample_idx]),
                    "rot_high_target_cos": float(tensor_cos(rot_hi, tgt_hi)[sample_idx]),
                })
        del model
        release_cuda()
    return rows


def aggregate_report(rows: list[dict[str, object]], args) -> dict[str, object]:
    source_keys = ("delta_lowmid_ratio", "cos_raw_rot", "sign_raw_rot_agree", "cos_rot_momentum_after", "sign_rot_momentum_after_agree", "rotation_update_sign_change")
    target_keys = ("delta_target_dot", "delta_target_cos", "raw_target_cos", "rot_target_cos", "raw_lowmid_target_cos", "rot_lowmid_target_cos", "raw_high_target_cos", "rot_high_target_cos")
    report = {
        "protocol": PROTOCOL,
        "config": vars(args),
        "fft_bands": FFT_BANDS,
        "lowmid_bands": LOWMID_BANDS,
        "high_bands": HIGH_BANDS,
        "known_asr": DEFAULT_ASR,
        "source_summary": {},
        "target_summary": {},
        "csv_asr_inputs": csv_asr_summary(tuple(args.asr_csv)),
    }
    for branch in sorted({str(row["branch"]) for row in rows}):
        branch_rows = [row for row in rows if row.get("branch") == branch]
        report["source_summary"][branch] = summarize_rows([row for row in branch_rows if row.get("scope") == "source"], source_keys)
        report["target_summary"][branch] = summarize_rows([row for row in branch_rows if row.get("scope") == "target"], target_keys)
    return report


def write_outputs(rows: list[dict[str, object]], report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lowmid_rotation_mechanism_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "lowmid_rotation_mechanism_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "lowmid_rotation_mechanism_conclusion_zh.md").write_text(
        build_conclusion_zh(report), encoding="utf-8"
    )


def run_experiment(args) -> None:
    seed_all(args.seed)
    loader, num_classes = load_data(args.image_dir, args.annotations_path, args.batch_size, args.num_workers, 2, args.img_size)
    source, _ = make_attacker(num_classes, mi=False, rotation=False)
    samples = []
    for images, labels, indices in selected_batches(args, source, loader):
        samples.append((images.cpu(), labels.cpu(), indices.cpu()))
    del source, loader
    release_cuda()

    rows = []
    for mi in (False, True):
        for rotation in (False, True):
            branch = f"{'rotation' if rotation else 'baseline'}/{'mi_on' if mi else 'mi_off'}"
            _source, attacker = make_attacker(num_classes, mi=mi, rotation=rotation)
            remaining_targets = args.target_samples
            for images, labels, indices in samples:
                trace = trace_attack(attacker, images, labels, branch, set(args.trace_steps))
                rows.extend(source_rows_from_trace(trace, indices))
                target_trace = [row for row in trace if row["step"] in set(args.target_steps)]
                if remaining_targets > 0 and target_trace:
                    take = min(int(remaining_targets), images.size(0))
                    sliced_trace = []
                    for item in target_trace:
                        sliced = dict(item)
                        for key in ("x_t", "raw_grad", "rot_grad", "momentum_before", "momentum_after", "update_grad"):
                            sliced[key] = sliced[key][:take]
                        sliced_trace.append(sliced)
                    rows.extend(target_alignment_rows(sliced_trace, labels[:take], indices[:take], args.target_models))
                    remaining_targets -= take
            del _source, attacker
            release_cuda()
    report = aggregate_report(rows, args)
    write_outputs(rows, report, Path(args.output_dir))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--target-samples", type=int, default=100)
    parser.add_argument("--trace-steps", type=parse_model_names, default=DEFAULT_TRACE_STEPS)
    parser.add_argument("--target-steps", type=parse_model_names, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--target-models", type=parse_model_names, default=MAIN_TARGETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--asr-csv",
        action="append",
        default=[
            "outputs/csv/outputs_attack_lazyagg_lowmid_grad_rotation_ours_dim_background_patch_pre_fpridx_lowmidrot_mifgsm_s40_500.csv",
            "outputs/csv/outputs_attack_lazyagg_lowmid_grad_rotation_ours_dim_background_patch_pre_fpridx_lowmidrot_ifgsm_s40_500.csv",
        ],
    )
    args = parser.parse_args()
    args.trace_steps = tuple(int(step) for step in args.trace_steps)
    args.target_steps = tuple(int(step) for step in args.target_steps)
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
