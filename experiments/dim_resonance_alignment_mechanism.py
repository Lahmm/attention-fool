"""DIM-resonance alignment mechanism analysis.

This diagnostic asks whether a guide augmentation amplifies the same
source-gradient direction that DIM contributes, and whether that amplified
component is also aligned with black-box target gradients.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from causal_analysis import MAIN_TARGETS, _target_normalize, build_baseline, seed_all, selected_batches
from gradient_analysis import FFT_BANDS, fft_project
from main import ANNOTATIONS_PATH, IMAGE_DIR, parse_model_names
from nets import build_vit_model
from utils import DEVICE, load_data

PROTOCOL = "dim_resonance_alignment_mechanism_v1"
BAND_GROUPS = {
    "low": (0, 1, 2),
    "mid": (3, 4, 5),
    "low_mid": (0, 1, 2, 3, 4, 5),
    "high": (6, 7),
}
VARIANTS = {
    "dim_only": {"dim": True, "guide_aug": False, "methods": ("dropout",), "dim_adjoint_echo": False},
    "reference_djf": {"dim": True, "guide_aug": True, "methods": ("dropout", "jitter", "freq"), "dim_adjoint_echo": False},
    "dim_resonance_only": {"dim": True, "guide_aug": True, "methods": ("dim_resonance",), "dim_adjoint_echo": False},
    "dim_resonance_djf": {"dim": True, "guide_aug": True, "methods": ("dropout", "jitter", "freq", "dim_resonance"), "dim_adjoint_echo": False},
    "dim_adjoint_echo_only": {"dim": True, "guide_aug": False, "methods": ("dropout",), "dim_adjoint_echo": True},
    "dim_adjoint_echo_djf": {"dim": True, "guide_aug": True, "methods": ("dropout", "jitter", "freq"), "dim_adjoint_echo": True},
}


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def attacker_options(attacker, **options):
    previous = {name: getattr(attacker, name) for name in options}
    try:
        for name, value in options.items():
            setattr(attacker, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(attacker, name, value)


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1)


def tensor_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    af, bf = flatten(a), flatten(b)
    return (af * bf).sum(1) / (af.norm(dim=1) * bf.norm(dim=1)).clamp_min(eps)


def band_project_sum(x: torch.Tensor, bands: tuple[int, ...]) -> torch.Tensor:
    out = torch.zeros_like(x)
    for band in bands:
        out = out + fft_project(x, band)
    return out


def gradient_for_variant(attacker, pixels, labels, guide, config) -> torch.Tensor:
    with attacker_options(
        attacker,
        input_diversity=config["dim"],
        guide_aug=config["guide_aug"],
        guide_aug_methods=config["methods"],
        guide_aug_area="background",
        dim_adjoint_echo=config.get("dim_adjoint_echo", False),
    ):
        probe = pixels.detach().requires_grad_(True)
        return attacker._attack_grad(probe, labels, guide).detach()


def target_gradient(model, pixels, labels) -> torch.Tensor:
    probe = pixels.detach().requires_grad_(True)
    logits = model(_target_normalize(model, probe), return_attn=False)
    loss = F.cross_entropy(logits, labels)
    return torch.autograd.grad(loss, probe)[0].detach()


def source_rows_for_pair(variant: str, grad: torch.Tensor, dim_grad: torch.Tensor) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for group, bands in BAND_GROUPS.items():
        vg = band_project_sum(grad, bands)
        dg = band_project_sum(dim_grad, bands)
        vf, df = flatten(vg), flatten(dg)
        dim_energy = df.square().sum(1).clamp_min(1e-20)
        variant_energy = vf.square().sum(1).clamp_min(1e-20)
        projection_gain = (vf * df).sum(1) / dim_energy
        norm_gain = variant_energy.sqrt() / dim_energy.sqrt()
        orthogonal_energy = (vf - projection_gain.view(-1, 1) * df).square().sum(1) / dim_energy
        sign_agreement = (vg.sign() == dg.sign()).to(torch.float32).flatten(1).mean(1)
        rows.append({
            "scope": "source_dim_alignment",
            "variant": variant,
            "model": "source",
            "band_group": group,
            "dim_cos": float(tensor_cos(vg, dg).mean().item()),
            "dim_projection_gain": float(projection_gain.mean().item()),
            "dim_norm_gain": float(norm_gain.mean().item()),
            "orthogonal_energy_over_dim": float(orthogonal_energy.mean().item()),
            "dim_sign_agreement": float(sign_agreement.mean().item()),
        })
    return rows


def source_rows_for_increment(variant: str, grad: torch.Tensor, reference: torch.Tensor, dim_grad: torch.Tensor) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    delta = grad - reference
    for group, bands in BAND_GROUPS.items():
        inc = band_project_sum(delta, bands)
        dg = band_project_sum(dim_grad, bands)
        inf, df = flatten(inc), flatten(dg)
        dim_energy = df.square().sum(1).clamp_min(1e-20)
        projection_gain = (inf * df).sum(1) / dim_energy
        rows.append({
            "scope": "increment_vs_reference",
            "variant": variant,
            "model": "source",
            "band_group": group,
            "increment_dim_projection": float(projection_gain.mean().item()),
            "increment_dim_cos": float(tensor_cos(inc, dg).mean().item()),
            "increment_norm_over_dim": float(inf.norm(dim=1).div(df.norm(dim=1).clamp_min(1e-20)).mean().item()),
        })
    return rows


def target_rows_for_increment(variant: str, grad: torch.Tensor, reference: torch.Tensor, dim_grad: torch.Tensor, target_grad: torch.Tensor, model_name: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    delta = grad - reference
    for group, bands in BAND_GROUPS.items():
        inc = band_project_sum(delta, bands)
        dg = band_project_sum(dim_grad, bands)
        tg = band_project_sum(target_grad, bands)
        inf, tf = flatten(inc), flatten(tg)
        target_dot = (inf * tf).sum(1)
        rows.append({
            "scope": "increment_target_alignment",
            "variant": variant,
            "model": model_name,
            "band_group": group,
            "increment_target_cos": float(tensor_cos(inc, tg).mean().item()),
            "increment_target_dot": float(target_dot.mean().item()),
            "positive_increment_target_dot_fraction": float((target_dot > 0).to(torch.float32).mean().item()),
            "increment_dim_cos": float(tensor_cos(inc, dg).mean().item()),
        })
    return rows


def target_rows_for_pair(variant: str, grad: torch.Tensor, dim_grad: torch.Tensor, target_grad: torch.Tensor, model_name: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for group, bands in BAND_GROUPS.items():
        vg = band_project_sum(grad, bands)
        dg = band_project_sum(dim_grad, bands)
        tg = band_project_sum(target_grad, bands)
        vf, df, tf = flatten(vg), flatten(dg), flatten(tg)
        dim_target_dot = (df * tf).sum(1)
        variant_target_dot = (vf * tf).sum(1)
        target_delta = variant_target_dot - dim_target_dot
        rows.append({
            "scope": "target_alignment",
            "variant": variant,
            "model": model_name,
            "band_group": group,
            "target_cos": float(tensor_cos(vg, tg).mean().item()),
            "dim_target_cos": float(tensor_cos(dg, tg).mean().item()),
            "target_cos_delta_vs_dim": float((tensor_cos(vg, tg) - tensor_cos(dg, tg)).mean().item()),
            "target_dot_delta_vs_dim": float(target_delta.mean().item()),
            "positive_target_dot_delta_fraction": float((target_delta > 0).to(torch.float32).mean().item()),
        })
    return rows


def aggregate_rows(rows: list[dict[str, float | str]]) -> dict[str, dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (str(row["scope"]), str(row["variant"]), str(row["band_group"]))
        bucket = grouped.setdefault(key, {})
        for metric, value in row.items():
            if metric in {"scope", "variant", "model", "band_group"}:
                continue
            bucket.setdefault(metric, []).append(float(value))
    summary: dict[str, dict[str, float | int | str]] = {}
    for (scope, variant, band_group), metrics in grouped.items():
        item: dict[str, float | int | str] = {"scope": scope, "variant": variant, "band_group": band_group, "n": len(next(iter(metrics.values()), []))}
        for metric, values in metrics.items():
            item[metric] = float(np.mean(values))
        summary[f"{scope}/{variant}/{band_group}"] = item
    return summary


def build_conclusion(summary: dict[str, dict[str, float | int | str]], effectiveness: dict[str, float] | None = None) -> str:
    def get(scope: str, variant: str, group: str, metric: str) -> float:
        return float(summary[f"{scope}/{variant}/{group}"][metric])

    variants = [name for name in VARIANTS if name != "dim_only"]
    rows = []
    increment_rows = []
    for variant in variants:
        rows.append((
            variant,
            get("source_dim_alignment", variant, "low_mid", "dim_projection_gain"),
            get("source_dim_alignment", variant, "high", "dim_projection_gain"),
            get("target_alignment", variant, "low_mid", "target_cos_delta_vs_dim"),
            get("target_alignment", variant, "high", "target_cos_delta_vs_dim"),
        ))
        if variant != "reference_djf":
            increment_rows.append((
                variant,
                get("increment_vs_reference", variant, "low_mid", "increment_dim_projection"),
                get("increment_vs_reference", variant, "high", "increment_dim_projection"),
                get("increment_target_alignment", variant, "low_mid", "increment_target_dot"),
                get("increment_target_alignment", variant, "high", "increment_target_dot"),
            ))
    best_lowmid_gain = max(rows, key=lambda item: item[1])
    best_target_delta = max(rows, key=lambda item: item[3])
    best_increment_lowmid = max(increment_rows, key=lambda item: item[1]) if increment_rows else None
    best_increment_target = max(increment_rows, key=lambda item: item[3]) if increment_rows else None
    supported_increments = [item for item in increment_rows if item[1] > 0 and item[3] > 0]
    supported_increments.sort(key=lambda item: (item[3], item[1]), reverse=True)
    lines = [
        "# DIM Resonance Alignment Mechanism",
        "",
        "## 数学判据",
        "令 DIM 随机 resize/pad 线性化为 J，DIM 源梯度含有 E[J^T ∇L(f(Jx), y)]。如果某增强 A 的梯度 g_A 真的放大 DIM 给出的可迁移方向，则在低/中频子空间 P_LM 上应满足 projection_gain = <P_LM g_A, P_LM g_DIM> / ||P_LM g_DIM||^2 > 1，并且 target-gradient alignment 也应同步改善。",
        "",
        "## 关键证据",
    ]
    for variant, lm_gain, high_gain, lm_delta, high_delta in rows:
        eff = ""
        if effectiveness and variant in effectiveness:
            eff = f", avg ASR={effectiveness[variant]:.6f}"
        lines.append(f"- {variant}: low/mid DIM 投影增益={lm_gain:.6g}, high 投影增益={high_gain:.6g}, low/mid target-cos 增量={lm_delta:.6g}, high target-cos 增量={high_delta:.6g}{eff}。")
    lines.append("")
    lines.append("## 相对 reference_djf 的新增分量")
    for variant, lm_inc, high_inc, lm_tgt, high_tgt in increment_rows:
        lines.append(f"- {variant}: 新增 low/mid DIM 投影={lm_inc:.6g}, 新增 high DIM 投影={high_inc:.6g}, 新增 low/mid target-dot={lm_tgt:.6g}, 新增 high target-dot={high_tgt:.6g}。")
    lines.extend([
        "",
        f"最强 low/mid DIM 投影增益: {best_lowmid_gain[0]} ({best_lowmid_gain[1]:.6g})。",
        f"最强 low/mid target 对齐增量: {best_target_delta[0]} ({best_target_delta[3]:.6g})。",
        f"相对 reference 的最强新增 low/mid DIM 投影: {best_increment_lowmid[0] if best_increment_lowmid else 'none'} ({best_increment_lowmid[1] if best_increment_lowmid else 0:.6g})。",
        f"相对 reference 的最强新增 low/mid target-dot: {best_increment_target[0] if best_increment_target else 'none'} ({best_increment_target[3] if best_increment_target else 0:.6g})。",
        "",
        "## 结论",
    ])
    if supported_increments:
        names = ", ".join(f"{item[0]}(DIM投影={item[1]:.6g}, target-dot={item[3]:.6g})" for item in supported_increments)
        lines.append(f"相对 reference_djf 同时给出正 low/mid DIM 同向新增投影和正 low/mid target-dot 的机制支持候选: {names}。")
    elif best_lowmid_gain[0] == best_target_delta[0]:
        lines.append(f"{best_lowmid_gain[0]} 同时最大化绝对 low/mid DIM 投影增益和 target-cos 增量，但还需要检查相对 reference 的新增分量是否为正。")
    else:
        lines.append("当前机制证据尚未找到同时具有正 DIM 同向新增投影和正 target 对齐新增量的候选；需要继续搜索或结合真实 ASR 判断。")
    lines.append("注意：projection_gain 只证明相对 DIM 方向的放大，target-cos/dot 才是迁移相关性证据；二者必须一起解释。")
    return "\n".join(lines) + "\n"


def read_effectiveness(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    result = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("variant") or row.get("name") or row.get("method")
            avg = row.get("avg_asr") or row.get("avg")
            if name and avg:
                result[name] = float(avg)
    return result


def run(args) -> None:
    seed_all(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loader, num_classes = load_data(args.image_dir, args.annotations_path, args.batch_size, args.num_workers, 2, args.img_size)
    source, attacker = build_baseline(num_classes)
    source.eval()
    attacker.guide_aug_copies = args.guide_aug_copies
    attacker.guide_aug_strength = args.guide_aug_strength
    rows: list[dict[str, float | str]] = []
    processed = 0
    for images, labels, _indices in selected_batches(args, source, loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        guide = attacker._build_guide_pixel_map(attacker._normalize(images), images.size(-1))
        grads = {name: gradient_for_variant(attacker, images, labels, guide, config) for name, config in VARIANTS.items()}
        dim_grad = grads["dim_only"]
        for name, grad in grads.items():
            if name == "dim_only":
                continue
            rows.extend(source_rows_for_pair(name, grad, dim_grad))
            if name != "reference_djf":
                rows.extend(source_rows_for_increment(name, grad, grads["reference_djf"], dim_grad))
        for model_name in args.target_models:
            model = build_vit_model(num_classes=1000, model_name=model_name)
            model.eval()
            target_grad = target_gradient(model, images, labels)
            for name, grad in grads.items():
                if name == "dim_only":
                    continue
                rows.extend(target_rows_for_pair(name, grad, dim_grad, target_grad, model_name))
                if name != "reference_djf":
                    rows.extend(target_rows_for_increment(name, grad, grads["reference_djf"], dim_grad, target_grad, model_name))
            del model
            release_cuda()
        processed += int(images.size(0))
        print(f"processed {processed}/{args.max_samples}")
        del grads, dim_grad, images, labels, guide
        release_cuda()
    summary = aggregate_rows(rows)
    effectiveness = read_effectiveness(args.effectiveness_csv)
    report = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "samples": processed,
        "target_models": list(args.target_models),
        "band_groups": {key: list(value) for key, value in BAND_GROUPS.items()},
        "variants": {key: {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()} for key, config in VARIANTS.items()},
        "summary": summary,
        "effectiveness": effectiveness,
    }
    with (out / "dim_resonance_alignment_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out / "dim_resonance_alignment_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "dim_resonance_alignment_conclusion_zh.md").write_text(build_conclusion(summary, effectiveness), encoding="utf-8")
    print(f"wrote {out / 'dim_resonance_alignment_report.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/analysis/dim_resonance_alignment_mechanism")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-models", type=parse_model_names, default=MAIN_TARGETS)
    parser.add_argument("--guide-aug-copies", type=int, default=3)
    parser.add_argument("--guide-aug-strength", type=float, default=0.2)
    parser.add_argument("--effectiveness-csv", default="outputs/analysis/dim_resonance_effectiveness_summary.csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
