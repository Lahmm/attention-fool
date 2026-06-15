"""Trace why MI improves LM-DSS sign-filter rotation transferability."""
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

PROTOCOL = "mi_gradient_accumulation_mechanism_v1"
LOWMID_BANDS = tuple(range(6))
HIGH_BANDS = tuple(range(6, len(FFT_BANDS) - 1))
DEFAULT_GUIDES = ("deit_base_patch16_224", "pit_s_224", "cait_s24_224")
DEFAULT_TRACE_STEPS = tuple(range(1, 41))
DEFAULT_TARGET_STEPS = (1, 5, 10, 20, 40)
KNOWN_ASR = {
    "control_lm_dss_sign_filter_rotation_mi": 0.8250,
    "ablate_no_mi": 0.66125,
    "ablate_sign_filter_only": 0.78625,
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


def sign_change(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - sign_agreement(a, b)


def project_bands(x: torch.Tensor, bands: tuple[int, ...]) -> torch.Tensor:
    if not bands:
        return torch.zeros_like(x)
    return sum((fft_project(x, band) for band in bands), torch.zeros_like(x))


def band_energy(x: torch.Tensor, bands: tuple[int, ...]) -> torch.Tensor:
    return project_bands(x, bands).square().flatten(1).sum(1)


def energy_ratio(x: torch.Tensor, bands: tuple[int, ...]) -> torch.Tensor:
    band = band_energy(x, bands)
    total = sum((band_energy(x, (band_idx,)) for band_idx in range(len(FFT_BANDS) - 1)), torch.zeros_like(band))
    return band / total.clamp_min(1e-20)


def masked_energy_ratio(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.full((x.size(0),), float("nan"), device=x.device)
    mask = mask.to(x.device, x.dtype).clamp(0.0, 1.0)
    masked = (x * mask).square().flatten(1).sum(1)
    total = x.square().flatten(1).sum(1).clamp_min(1e-20)
    return masked / total


def summarize_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        out[key] = float(np.mean(values)) if values else None
    return out


def make_attacker(num_classes: int, *, mi: bool, steps: int = 40):
    source = build_vit_model(num_classes=num_classes, model_name="vit_base_patch16_224")
    guides = tuple(build_vit_model(num_classes=num_classes, model_name=name) for name in DEFAULT_GUIDES)
    attacker = create_attacker(
        model=source,
        epsilon=16 / 255,
        step_size=None,
        steps=steps,
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
        guide_aug_strength=0.2,
        lowmid_dss_filter=True,
        lowmid_dss_consistency="sign",
        lowmid_dss_agreement_threshold=0.67,
        lowmid_grad_tuning=True,
        lowmid_grad_rotation_strength=0.5,
    )
    return source, attacker


def _process_grad_terms(
    attacker: LMDSSAttacker,
    raw: torch.Tensor,
    term_grads: tuple[torch.Tensor, ...],
    guide: torch.Tensor | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    after_ti = attacker._smooth_grad(raw)
    processed_terms = tuple(attacker._smooth_grad(term) for term in term_grads)
    after_filter = attacker._apply_lowmid_dss_filter(after_ti, processed_terms)
    after_rotation = attacker._tune_lowmid_gradient(after_filter)
    return after_ti, processed_terms, after_filter, after_rotation


def trace_attack(
    attacker: LMDSSAttacker,
    images: torch.Tensor,
    labels: torch.Tensor,
    branch: str,
    *,
    manual_accumulator: bool = False,
    keep_steps: set[int] | None = None,
    save_full_tensors: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, torch.Tensor]], torch.Tensor]:
    keep_steps = set(DEFAULT_TRACE_STEPS if keep_steps is None else keep_steps)
    images, labels = images.to(attacker.device), labels.to(attacker.device)
    clean = attacker._denormalize(images).detach()
    needs_guide = attacker.guide_aug and attacker.guide_aug_area != "all"
    guide = attacker._build_guide_pixel_map(images, clean.size(-1)) if needs_guide else None
    adv = clean.clone().detach()
    accumulator = torch.zeros_like(clean)
    metric_rows: list[dict[str, object]] = []
    tensor_rows: list[dict[str, torch.Tensor]] = []
    prev_grads: list[torch.Tensor] = []

    for step_idx in range(attacker.steps):
        step = step_idx + 1
        x_t = adv.detach()
        grad_pixels = x_t.requires_grad_(True)
        raw_mean, term_grads = attacker._attack_grad_terms(grad_pixels, labels, guide)
        after_ti, processed_terms, after_filter, after_rotation = _process_grad_terms(attacker, raw_mean, term_grads, guide)
        accumulator_before = accumulator
        accumulator = attacker.decay * accumulator + after_rotation
        update = accumulator if (attacker.use_momentum or manual_accumulator) else after_rotation
        history = accumulator - after_rotation
        with torch.no_grad():
            adv = adv + attacker.step_size * update.sign()
            adv = torch.clamp(adv, 0.0, 1.0)
            x_next = adv.detach()

        if step in keep_steps:
            sign_current = after_rotation.sign()
            sign_history = history.sign()
            sign_update = update.sign()
            override = sign_history.ne(0) & sign_current.ne(0) & sign_history.ne(sign_current) & sign_update.eq(sign_history)
            amplify = sign_history.eq(sign_current) & sign_current.ne(0) & accumulator.abs().gt(after_rotation.abs())
            row_base = {
                "scope": "source",
                "branch": branch,
                "step": step,
                "history_norm": tensor_norm(history),
                "current_norm": tensor_norm(after_rotation),
                "update_norm": tensor_norm(update),
                "history_lowmid_ratio": energy_ratio(history, LOWMID_BANDS),
                "history_high_ratio": energy_ratio(history, HIGH_BANDS),
                "current_lowmid_ratio": energy_ratio(after_rotation, LOWMID_BANDS),
                "update_lowmid_ratio": energy_ratio(update, LOWMID_BANDS),
                "current_update_sign_change": sign_change(after_rotation, update),
                "history_update_cos": tensor_cos(history, update),
                "current_update_cos": tensor_cos(after_rotation, update),
                "history_current_cos": tensor_cos(history, after_rotation),
                "history_override_ratio": override.flatten(1).float().mean(1),
                "consensus_amplify_ratio": amplify.flatten(1).float().mean(1),
                "history_background_energy_ratio": masked_energy_ratio(history, 1.0 - guide if guide is not None else None),
                "history_foreground_energy_ratio": masked_energy_ratio(history, guide),
            }
            if prev_grads:
                prev = torch.stack(prev_grads, dim=0)
                row_base["history_persistence_sign_agree"] = prev.sign().eq(sign_current.unsqueeze(0)).float().flatten(2).mean(2).mean(0)
                row_base["history_persistence_cos"] = torch.stack([tensor_cos(item, after_rotation) for item in prev], dim=0).mean(0)
            else:
                row_base["history_persistence_sign_agree"] = torch.full((images.size(0),), float("nan"), device=images.device)
                row_base["history_persistence_cos"] = torch.full((images.size(0),), float("nan"), device=images.device)
            for idx in range(images.size(0)):
                metric_rows.append({key: (float(value[idx]) if isinstance(value, torch.Tensor) else value) for key, value in row_base.items()})
            if save_full_tensors:
                tensor_rows.append({
                    "branch": branch,
                    "step": torch.tensor(step),
                    "clean": clean.detach().cpu(),
                    "x_t": x_t.detach().cpu(),
                    "x_next": x_next.detach().cpu(),
                    "delta_t": (x_t - clean).detach().cpu(),
                    "guide_pixel_map": torch.empty(0) if guide is None else guide.detach().cpu(),
                    "term_grad_raw": torch.stack([term.detach().cpu() for term in term_grads], dim=0),
                    "term_grad_processed": torch.stack([term.detach().cpu() for term in processed_terms], dim=0),
                    "grad_raw_mean": raw_mean.detach().cpu(),
                    "grad_after_ti": after_ti.detach().cpu(),
                    "grad_after_dss_filter": after_filter.detach().cpu(),
                    "grad_after_rotation": after_rotation.detach().cpu(),
                    "momentum_before": accumulator_before.detach().cpu(),
                    "momentum_after": accumulator.detach().cpu(),
                    "history_component": history.detach().cpu(),
                    "update_grad": update.detach().cpu(),
                    "update_sign": update.sign().detach().cpu(),
                    "counterfactual_no_mi_sign": after_rotation.sign().detach().cpu(),
                    "counterfactual_mi_sign": accumulator.sign().detach().cpu(),
                })
        prev_grads.append(after_rotation.detach())
    return metric_rows, tensor_rows, adv.detach()



def compress_trace_tensors(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.detach().cpu().to(torch.float16)
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: compress_trace_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compress_trace_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(compress_trace_tensors(item) for item in value)
    return value

def save_trace_payload(output_dir: Path, branch: str, batch_idx: int, payload: dict[str, object]) -> None:
    branch_dir = output_dir / "mi_gradient_accumulation_trace" / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(compress_trace_tensors(payload), branch_dir / f"batch_{batch_idx:04d}.pt")


def target_alignment_rows(
    tensor_rows: list[dict[str, torch.Tensor]],
    labels: torch.Tensor,
    indices: torch.Tensor,
    target_models: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = labels.to(DEVICE)
    for model_name in target_models:
        model = build_vit_model(num_classes=1000, model_name=model_name).eval()
        for trace in tensor_rows:
            pixels = trace["x_t"].to(DEVICE).detach().requires_grad_(True)
            current = trace["grad_after_rotation"].to(DEVICE)
            update = trace["update_grad"].to(DEVICE)
            history = trace["history_component"].to(DEVICE)
            loss = F.cross_entropy(model(_target_normalize(model, pixels), return_attn=False), labels)
            target = torch.autograd.grad(loss, pixels)[0].detach()
            for sample_idx, image_idx in enumerate(indices.tolist()):
                rows.append({
                    "scope": "target",
                    "branch": str(trace["branch"]),
                    "step": int(trace["step"]),
                    "image_index": int(image_idx),
                    "target_model": model_name,
                    "current_target_cos": float(tensor_cos(current, target)[sample_idx]),
                    "update_target_cos": float(tensor_cos(update, target)[sample_idx]),
                    "history_target_cos": float(tensor_cos(history, target)[sample_idx]),
                    "current_lowmid_target_cos": float(tensor_cos(project_bands(current, LOWMID_BANDS), project_bands(target, LOWMID_BANDS))[sample_idx]),
                    "update_lowmid_target_cos": float(tensor_cos(project_bands(update, LOWMID_BANDS), project_bands(target, LOWMID_BANDS))[sample_idx]),
                    "history_lowmid_target_cos": float(tensor_cos(project_bands(history, LOWMID_BANDS), project_bands(target, LOWMID_BANDS))[sample_idx]),
                    "update_vs_current_target_cos_delta": float((tensor_cos(update, target) - tensor_cos(current, target))[sample_idx]),
                })
        del model
        release_cuda()
    return rows


def aggregate_report(rows: list[dict[str, object]], args) -> dict[str, object]:
    source_keys = (
        "history_norm",
        "current_norm",
        "update_norm",
        "history_lowmid_ratio",
        "history_high_ratio",
        "current_lowmid_ratio",
        "update_lowmid_ratio",
        "current_update_sign_change",
        "history_update_cos",
        "current_update_cos",
        "history_current_cos",
        "history_override_ratio",
        "consensus_amplify_ratio",
        "history_background_energy_ratio",
        "history_foreground_energy_ratio",
        "history_persistence_sign_agree",
        "history_persistence_cos",
    )
    target_keys = (
        "current_target_cos",
        "update_target_cos",
        "history_target_cos",
        "current_lowmid_target_cos",
        "update_lowmid_target_cos",
        "history_lowmid_target_cos",
        "update_vs_current_target_cos_delta",
    )
    report = {
        "protocol": PROTOCOL,
        "config": vars(args),
        "known_asr": KNOWN_ASR,
        "fft_bands": FFT_BANDS,
        "lowmid_bands": LOWMID_BANDS,
        "high_bands": HIGH_BANDS,
        "source_summary": {},
        "target_summary": {},
    }
    for branch in sorted({str(row["branch"]) for row in rows}):
        branch_rows = [row for row in rows if row.get("branch") == branch]
        report["source_summary"][branch] = summarize_rows([row for row in branch_rows if row.get("scope") == "source"], source_keys)
        report["target_summary"][branch] = summarize_rows([row for row in branch_rows if row.get("scope") == "target"], target_keys)
    return report


def build_conclusion_zh(report: dict[str, object]) -> str:
    source = report["source_summary"]
    target = report["target_summary"]
    mi = source.get("control_mi", {})
    no_mi = source.get("no_mi", {})
    manual = source.get("manual_accumulator_no_mi_flag", {})
    mi_target = target.get("control_mi", {})
    lines = [
        "# MI 梯度累计机制实验结论",
        "",
        f"- 已知 100 样本 ASR: MI `{KNOWN_ASR['control_lm_dss_sign_filter_rotation_mi']:.4f}`，no-MI `{KNOWN_ASR['ablate_no_mi']:.4f}`。",
        f"- MI 分支中，当前单步梯度与最终 update sign 的平均差异为 `{mi.get('current_update_sign_change')}`。",
        f"- no-MI 分支中，该差异为 `{no_mi.get('current_update_sign_change')}`，因为 update 直接等于当前步梯度。",
        f"- MI 的历史分量 low/mid energy ratio 为 `{mi.get('history_lowmid_ratio')}`，high ratio 为 `{mi.get('history_high_ratio')}`。",
        f"- 历史方向覆盖当前单步方向的比例为 `{mi.get('history_override_ratio')}`，同向放大的比例为 `{mi.get('consensus_amplify_ratio')}`。",
        f"- 历史分量与 target gradient 的平均 cosine 为 `{mi_target.get('history_target_cos')}`；MI update 相对当前梯度的 target-cos 增量为 `{mi_target.get('update_vs_current_target_cos_delta')}`。",
        "",
        "## 解释",
        "",
        "MI 提升不是因为模型反传变了，而是因为 `momentum_after - current_grad` 这部分历史输入梯度被保留下来并参与 `sign(update)`。",
        "no-MI 每一步只使用当前 `grad_after_rotation`，因此所有在前几步出现过、但当前步减弱或反向的历史方向都会从 update 中消失。",
        "如果 `manual_accumulator_no_mi_flag` 与 `control_mi` 的指标接近，说明不用 CLI 的 `--mi` 也能实现同样效果；关键机制是跨 step 状态累计，而不是参数名本身。",
    ]
    if manual:
        lines.append(f"手动累计分支的 update/current sign 差异为 `{manual.get('current_update_sign_change')}`，可用于验证这一点。")
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict[str, object]], report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "mi_gradient_accumulation_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "mi_gradient_accumulation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "mi_gradient_accumulation_conclusion_zh.md").write_text(
        build_conclusion_zh(report), encoding="utf-8"
    )


def run_experiment(args) -> None:
    seed_all(args.seed)
    output_dir = Path(args.output_dir)
    loader, num_classes = load_data(args.image_dir, args.annotations_path, args.batch_size, args.num_workers, 2, args.img_size)
    source, _ = make_attacker(num_classes, mi=False, steps=args.steps)
    samples = []
    selected = 0
    for images, labels, indices in selected_batches(args, source, loader):
        samples.append((images.cpu(), labels.cpu(), indices.cpu()))
        selected += images.size(0)
        if selected >= args.max_samples:
            break
    del source, loader
    release_cuda()

    rows: list[dict[str, object]] = []
    branches = (
        ("control_mi", True, False),
        ("no_mi", False, False),
        ("manual_accumulator_no_mi_flag", False, True),
    )
    for branch, use_mi, manual in branches:
        _source, attacker = make_attacker(num_classes, mi=use_mi, steps=args.steps)
        traced = 0
        target_remaining = args.target_samples
        for batch_idx, (images, labels, indices) in enumerate(samples):
            take = min(images.size(0), max(0, args.trace_samples - traced))
            if take > 0:
                trace_images = images[:take]
                trace_labels = labels[:take]
                trace_indices = indices[:take]
                metric_rows, tensor_rows, adv = trace_attack(
                    attacker,
                    trace_images,
                    trace_labels,
                    branch,
                    manual_accumulator=manual,
                    keep_steps=set(args.trace_steps),
                    save_full_tensors=True,
                )
                rows.extend(metric_rows)
                save_trace_payload(output_dir, branch, batch_idx, {
                    "branch": branch,
                    "indices": trace_indices,
                    "labels": trace_labels,
                    "adv": adv.cpu(),
                    "traces": tensor_rows,
                })
                target_rows = [row for row in tensor_rows if int(row["step"]) in set(args.target_steps)]
                if target_remaining > 0 and target_rows:
                    rows.extend(target_alignment_rows(target_rows, trace_labels, trace_indices, args.target_models))
                    target_remaining -= take
                traced += take
            if take < images.size(0):
                metric_rows, _tensor_rows, _adv = trace_attack(
                    attacker,
                    images[take:],
                    labels[take:],
                    branch,
                    manual_accumulator=manual,
                    keep_steps=set(args.trace_steps),
                    save_full_tensors=False,
                )
                rows.extend(metric_rows)
        del _source, attacker
        release_cuda()
    report = aggregate_report(rows, args)
    write_outputs(rows, report, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--trace-samples", type=int, default=20)
    parser.add_argument("--target-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--trace-steps", type=parse_model_names, default=DEFAULT_TRACE_STEPS)
    parser.add_argument("--target-steps", type=parse_model_names, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--target-models", type=parse_model_names, default=MAIN_TARGETS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.trace_steps = tuple(int(step) for step in args.trace_steps)
    args.target_steps = tuple(int(step) for step in args.target_steps)
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
