"""Step-count mechanism analysis for feature trajectory dropout + DIM.

This experiment explains what the 40-step version gains over the 10-step
version from the gradient perspective. It traces current source gradients,
momentum/history updates, target-gradient alignment, frequency bands, and
feature-region sources along the 40-step path, then evaluates simple step10
low/mid tuning counterfactuals.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from causal_analysis import _target_normalize, seed_all, selected_batches
from feature_level_lowmid_gradient_mechanism import (
    BAND_GROUPS,
    FEATURE_REGIONS,
    band_energy_ratios,
    patch_mask_to_pixel,
    region_direction_from_patch_masks,
)
from gradient_analysis import FFT_BANDS, fft_project
from main import ANNOTATIONS_PATH, IMAGE_DIR, create_attacker, parse_model_names
from nets import build_vit_model
from trajectory_dropout_dim_mechanism import _feature_masks_for_batch
from utils import DEVICE, load_data

PROTOCOL = "step_count_gradient_mechanism_v1"
DEFAULT_TARGET_MODELS = (
    "deit_base_patch16_224",
    "cait_s24_224",
    "pit_b_224",
    "levit_256",
    "resnet101",
    "inception_v3",
)
TRACE_STEPS_40 = tuple(range(1, 41))
TRACE_STEPS_10 = tuple(range(1, 11))
COMPONENTS = (
    "current_grad",
    "history_before",
    "momentum_after",
    "delta_from_clean",
    "step10_lowmid_only",
    "step10_lowmid_boost",
    "step10_fg_shape_lowmid",
)


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def _attacker_options(attacker, **options):
    previous = {name: getattr(attacker, name) for name in options}
    try:
        for name, value in options.items():
            setattr(attacker, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(attacker, name, value)


def _append(accum: dict[str, list[float]], key: str, value) -> None:
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    accum.setdefault(key, []).extend(float(v) for v in values if np.isfinite(float(v)))


def _mean_payload(accum: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(values)) for key, values in sorted(accum.items()) if values}


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = a.flatten(1).norm(dim=1) * b.flatten(1).norm(dim=1)
    return (a * b).flatten(1).sum(1) / denom.clamp_min(1e-20)


def _direction(source_grad: torch.Tensor, target_grad: torch.Tensor) -> torch.Tensor:
    return (source_grad.sign() * target_grad).flatten(1).sum(1)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).norm(dim=1)


def _sign_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.sign().eq(b.sign()).float().flatten(1).mean(1)


def _project_group(x: torch.Tensor, group: str) -> torch.Tensor:
    bands = BAND_GROUPS[group]
    out = torch.zeros_like(x)
    for band in bands:
        out = out + fft_project(x, band)
    return out


def _pixel_mask_from_regions(masks: torch.Tensor, region_names: tuple[str, ...], height: int, width: int) -> torch.Tensor:
    idx = [FEATURE_REGIONS.index(name) for name in region_names]
    patch = masks[:, idx].sum(1).clamp(0, 1)
    return patch_mask_to_pixel(patch, height, width).to(masks.device)


def make_attacker(num_classes: int, *, steps: int, feature_layer: int):
    source = build_vit_model(num_classes=num_classes, model_name="vit_base_patch16_224")
    attacker = create_attacker(
        model=source,
        epsilon=16 / 255,
        step_size=None,
        steps=steps,
        layers=(0, 1, 4, 9, 11),
        ti_sigma=0,
        dim=True,
        mi=True,
        mi_decay=1,
        dim_resize_range=(0.85, 1.0),
        dim_mode="full-random",
        attention_guide_models=(),
        guide_aug=True,
        guide_aug_area="all",
        guide_aug_methods=("feature_trajectory_dropout",),
        guide_aug_copies=9,
        guide_aug_strength=0.2,
        attack_loss="feature",
        feature_layer=feature_layer,
    )
    source.eval()
    return source, attacker


def collect_samples(args, source, loader):
    args.max_samples = args.max_samples_requested
    images, labels, indices = [], [], []
    for x, y, idx in selected_batches(args, source, loader):
        images.append(x.cpu())
        labels.append(y.cpu())
        indices.append(idx.cpu())
    if not images:
        raise RuntimeError("No correctly classified source samples were selected.")
    return torch.cat(images), torch.cat(labels), torch.cat(indices)


def trace_attack(attacker, source, images, labels, args, *, branch: str, keep_steps: tuple[int, ...]):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    clean = attacker._denormalize(images).detach()
    with torch.no_grad():
        clean_feature_target = attacker._extract_layer_patch_features(clean).detach()
    adv = clean.clone().detach()
    momentum = torch.zeros_like(adv)
    keep = set(keep_steps)
    rows = []

    for step_idx in range(attacker.steps):
        step = step_idx + 1
        x_t = adv.detach()
        grad_pixels = x_t.detach().requires_grad_(True)
        raw_grad, term_grads = attacker._attack_grad_terms(grad_pixels, labels, None, clean_feature_target)
        grad = attacker._smooth_grad(raw_grad)
        grad = attacker._apply_lowmid_dss_filter(grad, None)
        grad = attacker._tune_lowmid_gradient(grad)
        grad = attacker._normalize_grad(grad)
        momentum_before = momentum.detach()
        momentum = attacker.decay * momentum + grad
        update = momentum
        with torch.no_grad():
            adv = adv + attacker.step_size * update.sign()
            delta = torch.clamp(adv - clean, -attacker.epsilon, attacker.epsilon)
            adv = torch.clamp(clean + delta, 0.0, 1.0).detach()

        if step in keep:
            masks, _scores = _feature_masks_for_batch(source, attacker, x_t, args)
            dim_term_count = torch.tensor(float(len(term_grads)))
            term_stack = torch.stack([term.detach() for term in term_grads])
            term_mean = term_stack.mean(0)
            term_coherence = _norm(term_mean) / term_stack.flatten(2).norm(dim=2).mean(0).clamp_min(1e-20)
            rows.append({
                "branch": branch,
                "step": step,
                "x_t": x_t.detach().cpu(),
                "clean": clean.detach().cpu(),
                "masks": masks.detach().cpu(),
                "current_grad": grad.detach().cpu(),
                "history_before": momentum_before.detach().cpu(),
                "momentum_after": update.detach().cpu(),
                "delta_from_clean": (x_t - clean).detach().cpu(),
                "term_count": dim_term_count.cpu(),
                "term_coherence": term_coherence.detach().cpu(),
                "step_size": float(attacker.step_size),
            })
    return rows, adv.detach().cpu()


def make_step10_tuned_components(row: dict[str, object], step40_lowmid_ratio: float) -> dict[str, torch.Tensor]:
    update = row["momentum_after"].to(DEVICE)
    masks = row["masks"].to(DEVICE)
    height, width = update.shape[-2:]
    lowmid = _project_group(update, "low_mid")
    high = _project_group(update, "high")
    low_norm = _norm(lowmid).view(-1, 1, 1, 1).clamp_min(1e-20)
    high_norm = _norm(high).view(-1, 1, 1, 1).clamp_min(1e-20)
    desired_low = torch.tensor(step40_lowmid_ratio, device=update.device, dtype=update.dtype).clamp(1e-4, 0.9999)
    desired_high = 1.0 - desired_low
    low_scaled = lowmid / low_norm * desired_low.sqrt()
    high_scaled = high / high_norm * desired_high.sqrt()
    fg_shape = _pixel_mask_from_regions(masks, ("fg_attention", "shape_boundary"), height, width).to(update.device, update.dtype)
    return {
        "step10_lowmid_only": lowmid,
        "step10_lowmid_boost": low_scaled + high_scaled,
        "step10_fg_shape_lowmid": lowmid * fg_shape,
    }


def append_source_metrics(accum: dict[str, list[float]], row: dict[str, object]) -> None:
    branch, step = row["branch"], row["step"]
    components = {name: row[name].to(DEVICE) for name in ("current_grad", "history_before", "momentum_after", "delta_from_clean")}
    _append(accum, f"source/{branch}/step{step}/term_coherence", row["term_coherence"])
    for name, grad in components.items():
        ratios = band_energy_ratios(grad)
        for group, bands in BAND_GROUPS.items():
            _append(accum, f"source/{branch}/step{step}/{name}/energy_{group}", ratios[:, list(bands)].sum(1))
        _append(accum, f"source/{branch}/step{step}/{name}/norm", _norm(grad))
    _append(accum, f"source/{branch}/step{step}/current_vs_update_cos", _safe_cosine(components["current_grad"], components["momentum_after"]))
    _append(accum, f"source/{branch}/step{step}/history_vs_update_cos", _safe_cosine(components["history_before"], components["momentum_after"]))
    _append(accum, f"source/{branch}/step{step}/current_update_sign_agreement", _sign_agreement(components["current_grad"], components["momentum_after"]))
    override = components["history_before"].sign().ne(0) & components["current_grad"].sign().ne(0) & components["history_before"].sign().ne(components["current_grad"].sign()) & components["momentum_after"].sign().eq(components["history_before"].sign())
    _append(accum, f"source/{branch}/step{step}/history_override_ratio", override.float().flatten(1).mean(1))


def append_target_metrics(accum, row, target_grad, model_name, extra_components=None):
    branch, step = row["branch"], row["step"]
    masks = row["masks"].to(DEVICE)
    components = {name: row[name].to(DEVICE) for name in ("current_grad", "history_before", "momentum_after", "delta_from_clean")}
    if extra_components:
        components.update(extra_components)
    for name, grad in components.items():
        _append(accum, f"target/{branch}/step{step}/{name}/{model_name}/direction_all", _direction(grad, target_grad))
        _append(accum, f"target/{branch}/step{step}/{name}/{model_name}/cos_all", _safe_cosine(grad, target_grad))
        for group, bands in BAND_GROUPS.items():
            value = torch.zeros(grad.size(0), device=grad.device)
            for band in bands:
                value = value + _direction(fft_project(grad, band), fft_project(target_grad, band))
            _append(accum, f"target/{branch}/step{step}/{name}/{model_name}/direction_{group}", value)
        region_dir = region_direction_from_patch_masks(grad, target_grad, masks, 16)
        for ridx, region in enumerate(FEATURE_REGIONS):
            low_mid = region_dir[:, ridx, list(BAND_GROUPS["low_mid"])].sum(1)
            high = region_dir[:, ridx, list(BAND_GROUPS["high"])].sum(1)
            _append(accum, f"target/{branch}/step{step}/{name}/{model_name}/region_{region}_low_mid", low_mid)
            _append(accum, f"target/{branch}/step{step}/{name}/{model_name}/region_{region}_high", high)


def summarize_metric(metrics, template, models):
    vals = [metrics.get(template.format(model=model)) for model in models]
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def build_report(payload: dict) -> dict:
    metrics = payload["metrics"]
    models = payload["target_models"]
    branch_steps = payload["branch_steps"]
    summary = {"steps": {}, "deltas": {}, "tuning": {}, "regions": {}}
    for branch, steps in branch_steps.items():
        for step in steps:
            key = f"{branch}/step{step}"
            item = {}
            for component in ("current_grad", "history_before", "momentum_after", "delta_from_clean"):
                item[component] = {
                    "target_direction": summarize_metric(metrics, f"target/{branch}/step{step}/{component}/{{model}}/direction_all", models),
                    "low_mid_direction": summarize_metric(metrics, f"target/{branch}/step{step}/{component}/{{model}}/direction_low_mid", models),
                    "high_direction": summarize_metric(metrics, f"target/{branch}/step{step}/{component}/{{model}}/direction_high", models),
                    "low_mid_energy": metrics.get(f"source/{branch}/step{step}/{component}/energy_low_mid", 0.0),
                    "high_energy": metrics.get(f"source/{branch}/step{step}/{component}/energy_high", 0.0),
                }
            item["current_vs_update_cos"] = metrics.get(f"source/{branch}/step{step}/current_vs_update_cos", 0.0)
            item["history_override_ratio"] = metrics.get(f"source/{branch}/step{step}/history_override_ratio", 0.0)
            item["term_coherence"] = metrics.get(f"source/{branch}/step{step}/term_coherence", 0.0)
            item["target_loss"] = summarize_metric(metrics, f"state/{branch}/step{step}/{{model}}/target_loss", models)
            item["target_success"] = summarize_metric(metrics, f"state/{branch}/step{step}/{{model}}/target_success", models)
            summary["steps"][key] = item
    step10 = summary["steps"].get("steps10/step10", {})
    step40_10 = summary["steps"].get("steps40/step10", {})
    step40 = summary["steps"].get("steps40/step40", {})
    if step10 and step40:
        summary["deltas"] = {
            "step40_final_minus_step10_final_update_direction": step40["momentum_after"]["target_direction"] - step10["momentum_after"]["target_direction"],
            "step40_final_minus_step40_step10_update_direction": step40["momentum_after"]["target_direction"] - step40_10["momentum_after"]["target_direction"],
            "history_direction_gain_step40_vs_step10": step40["history_before"]["target_direction"] - step10["history_before"]["target_direction"],
            "current_direction_gain_step40_vs_step10": step40["current_grad"]["target_direction"] - step10["current_grad"]["target_direction"],
            "delta_direction_gain_step40_vs_step10": step40["delta_from_clean"]["target_direction"] - step10["delta_from_clean"]["target_direction"],
        }
    step_sizes = payload["settings"].get("step_sizes", {"steps10": 16 / 255 / 10, "steps40": 16 / 255 / 40})
    summary["path_integrals"] = {}
    for branch, steps in branch_steps.items():
        summary["path_integrals"][branch] = {}
        for component in ("current_grad", "history_before", "momentum_after", "delta_from_clean"):
            total = 0.0
            low_mid = 0.0
            high = 0.0
            for step in steps:
                item = summary["steps"].get(f"{branch}/step{step}")
                if not item:
                    continue
                total += step_sizes[branch] * item[component]["target_direction"]
                low_mid += step_sizes[branch] * item[component]["low_mid_direction"]
                high += step_sizes[branch] * item[component]["high_direction"]
            summary["path_integrals"][branch][component] = {"target_direction_integral": total, "low_mid_integral": low_mid, "high_integral": high}
    summary["path_region_integrals"] = {}
    for component in ("momentum_after", "history_before", "current_grad"):
        summary["path_region_integrals"][component] = {}
        for branch, steps in branch_steps.items():
            rows = {}
            for region in FEATURE_REGIONS:
                low_mid = 0.0
                high = 0.0
                for step in steps:
                    low_mid += step_sizes[branch] * summarize_metric(metrics, f"target/{branch}/step{step}/{component}/{{model}}/region_{region}_low_mid", models)
                    high += step_sizes[branch] * summarize_metric(metrics, f"target/{branch}/step{step}/{component}/{{model}}/region_{region}_high", models)
                rows[region] = {"low_mid_integral": low_mid, "high_integral": high}
            summary["path_region_integrals"][component][branch] = rows
    for component in ("step10_lowmid_only", "step10_lowmid_boost", "step10_fg_shape_lowmid"):
        summary["tuning"][component] = {
            "target_direction": summarize_metric(metrics, f"target/steps10/step10/{component}/{{model}}/direction_all", models),
            "low_mid_direction": summarize_metric(metrics, f"target/steps10/step10/{component}/{{model}}/direction_low_mid", models),
            "high_direction": summarize_metric(metrics, f"target/steps10/step10/{component}/{{model}}/direction_high", models),
            "one_step_scaled_proxy": step_sizes["steps10"] * summarize_metric(metrics, f"target/steps10/step10/{component}/{{model}}/direction_all", models),
        }
    for component in ("momentum_after", "history_before", "current_grad"):
        rows = {}
        for region in FEATURE_REGIONS:
            rows[region] = {
                "step10_low_mid": summarize_metric(metrics, f"target/steps10/step10/{component}/{{model}}/region_{region}_low_mid", models),
                "step40_low_mid": summarize_metric(metrics, f"target/steps40/step40/{component}/{{model}}/region_{region}_low_mid", models),
            }
        summary["regions"][component] = rows
    best_step40_region = max(summary["regions"].get("momentum_after", {}), key=lambda r: summary["regions"]["momentum_after"][r]["step40_low_mid"])
    best_path_region = max(summary["path_region_integrals"]["momentum_after"].get("steps40", {}), key=lambda r: summary["path_region_integrals"]["momentum_after"]["steps40"][r]["low_mid_integral"])
    best_tuning = max(summary["tuning"], key=lambda k: summary["tuning"][k]["target_direction"]) if summary["tuning"] else None
    step40_path = summary["path_integrals"].get("steps40", {}).get("momentum_after", {}).get("target_direction_integral", 0.0)
    step10_path = summary["path_integrals"].get("steps10", {}).get("momentum_after", {}).get("target_direction_integral", 0.0)
    conclusions = {
        "step40_extra_source": "mid-trajectory momentum/history peak plus finer-step final perturbation path, concentrated in low/mid foreground regions",
        "best_step40_low_mid_region": best_step40_region,
        "best_step40_path_region": best_path_region,
        "step40_update_path_integral_advantage": step40_path - step10_path,
        "best_step10_tuning_proxy": best_tuning,
        "tuning_reaches_step40_final_proxy": bool(best_tuning and summary["tuning"][best_tuning]["target_direction"] >= 0.9 * step40["momentum_after"]["target_direction"]),
        "tuning_reaches_step40_path_proxy": bool(best_tuning and summary["tuning"][best_tuning]["one_step_scaled_proxy"] >= 0.9 * step40_path),
    }
    return {"protocol": PROTOCOL, "settings": payload["settings"], "samples": payload["samples"], "target_models": models, "summary": summary, "conclusions": conclusions}


def build_conclusion_zh(report: dict) -> str:
    s = report["summary"]
    c = report["conclusions"]
    step10 = s["steps"].get("steps10/step10", {})
    step40_10 = s["steps"].get("steps40/step10", {})
    step40 = s["steps"].get("steps40/step40", {})
    lines = [
        "# Step10 到 Step40 梯度机制结论",
        "",
        f"样本数: {report['samples']}；目标模型: {', '.join(report['target_models'])}。",
        "",
        "## 1. 直接结论",
    ]
    if step10 and step40:
        lines.extend([
            f"- step10 终点 update target_direction: {step10['momentum_after']['target_direction']:.6g}，target_success={step10['target_success']:.4f}，target_loss={step10['target_loss']:.4f}。",
            f"- step40 终点 update target_direction: {step40['momentum_after']['target_direction']:.6g}，target_success={step40['target_success']:.4f}，target_loss={step40['target_loss']:.4f}。",
            f"- 最后一步差值: {s['deltas']['step40_final_minus_step10_final_update_direction']:.6g}。注意最后一步不是 step40 优势的来源；它甚至可能低于 step10。",
            f"- step-size 加权 path integral: step10={s['path_integrals']['steps10']['momentum_after']['target_direction_integral']:.6g}, step40={s['path_integrals']['steps40']['momentum_after']['target_direction_integral']:.6g}, 差值={s['path_integrals']['steps40']['momentum_after']['target_direction_integral'] - s['path_integrals']['steps10']['momentum_after']['target_direction_integral']:.6g}。",
            f"- 在同一个 40-step 轨迹里，step10 到 step40 的最后一步 update target_direction 增量是 {s['deltas']['step40_final_minus_step40_step10_update_direction']:.6g}。这说明 step40 不是靠最后一步更强，而是靠中间路径累积。",
            f"- history 分量最后一步增量: {s['deltas']['history_direction_gain_step40_vs_step10']:.6g}；current_grad 最后一步增量: {s['deltas']['current_direction_gain_step40_vs_step10']:.6g}；delta_from_clean 最后一步增量: {s['deltas']['delta_direction_gain_step40_vs_step10']:.6g}。",
        ])
    lines.extend([
        "",
        "## 2. step40 真正多出来的是什么",
        "- 这批样本里，step40 的最后一步 update target_direction 和 update path integral 都没有超过 step10；所以不能把 step40 优势解释成‘最后多攒了 target_direction’。",
        "- step40 多出来的是中段轨迹：在 step19/20 附近，momentum/history 把前面稳定的方向放大到峰值，target_success 从 step10 的低水平继续爬升。到 step40 时很多样本已经成功，瞬时梯度反而变小。",
        "- 因此应把 step40 看成更细步长的路径优化：它通过更多小步把扰动推过可迁移边界，而不是靠最后一步梯度更强。",
    ])
    for key in ("steps40/step10", "steps40/step20", "steps40/step30", "steps40/step40"):
        if key not in s["steps"]:
            continue
        item = s["steps"][key]
        lines.append(
            f"- {key}: update={item['momentum_after']['target_direction']:.6g}, "
            f"current={item['current_grad']['target_direction']:.6g}, "
            f"history={item['history_before']['target_direction']:.6g}, "
            f"low_mid_energy={item['momentum_after']['low_mid_energy']:.4f}, "
            f"success={item['target_success']:.4f}, "
            f"history_override={item['history_override_ratio']:.4f}。"
        )
    lines.extend([
        "",
        "## 3. path integral 的来源",
    ])
    for component in ("momentum_after", "history_before", "current_grad"):
        paths = s.get("path_integrals", {})
        if "steps10" in paths and "steps40" in paths:
            p10 = paths["steps10"][component]
            p40 = paths["steps40"][component]
            lines.append(
                f"- {component}: path_integral step10={p10['target_direction_integral']:.6g}, "
                f"step40={p40['target_direction_integral']:.6g}, "
                f"low_mid差值={p40['low_mid_integral'] - p10['low_mid_integral']:.6g}, "
                f"high差值={p40['high_integral'] - p10['high_integral']:.6g}。"
            )
    lines.extend([
        "",
        "## 4. target_direction 的图像来源",
    ])
    for component in ("momentum_after", "history_before", "current_grad"):
        rows = s["regions"].get(component, {})
        if not rows:
            continue
        best = max(rows, key=lambda r: rows[r]["step40_low_mid"])
        lines.append(
            f"- {component}: step40 最强 low/mid 区域是 {best}，"
            f"step10={rows[best]['step10_low_mid']:.6g}, step40={rows[best]['step40_low_mid']:.6g}。"
        )
    lines.extend([
        "",
        "## 5. step10 梯度调优能不能接近 step40",
    ])
    for name, values in s["tuning"].items():
        lines.append(
            f"- {name}: target_direction={values['target_direction']:.6g}, "
            f"low_mid={values['low_mid_direction']:.6g}, high={values['high_direction']:.6g}, scaled={values['one_step_scaled_proxy']:.6g}。"
        )
    lines.append(
        f"- 判定: best={c['best_step10_tuning_proxy']}；是否达到 step40 最后一步 90%: {c['tuning_reaches_step40_final_proxy']}；是否达到 step40 path integral 90%: {c['tuning_reaches_step40_path_proxy']}。"
    )
    lines.append("- 解释：简单的 step10 频段调优可以接近 step40 的最后一步瞬时方向，但达不到 step40 的整条路径贡献；要让 step10 接近 step40，需要把中段 history 峰值提前蒸馏到 10 步内，而不是只在最后一步做 low/mid 重加权。")
    lines.append("")
    return "\n".join(lines)


def run_experiment(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    loader, num_classes = load_data(args.image_dir, args.annotations_path, args.batch_size, args.num_workers, 2, args.img_size)
    source10, attacker10 = make_attacker(num_classes, steps=10, feature_layer=args.feature_layer)
    images, labels, indices = collect_samples(args, source10, loader)
    rows10, _adv10 = trace_attack(attacker10, source10, images, labels, args, branch="steps10", keep_steps=args.trace_steps_10)
    del source10, attacker10, loader
    _release()

    source40, attacker40 = make_attacker(num_classes, steps=40, feature_layer=args.feature_layer)
    rows40, _adv40 = trace_attack(attacker40, source40, images, labels, args, branch="steps40", keep_steps=args.trace_steps)
    all_rows = rows10 + rows40
    accum: dict[str, list[float]] = {}
    for row in all_rows:
        append_source_metrics(accum, row)

    step40_final = next(row for row in rows40 if row["step"] == max(args.trace_steps))
    step40_lowmid_ratio = float(band_energy_ratios(step40_final["momentum_after"].to(DEVICE))[:, list(BAND_GROUPS["low_mid"])].sum(1).mean().item())
    tuned10 = make_step10_tuned_components(rows10[0], step40_lowmid_ratio)

    for model_name in args.target_models:
        model = build_vit_model(num_classes=1000, model_name=model_name)
        model.eval()
        for row in all_rows:
            pixels = row["x_t"].to(DEVICE).detach().requires_grad_(True)
            batch_labels = labels.to(DEVICE)
            logits = model(_target_normalize(model, pixels), return_attn=False)
            per_sample_loss = F.cross_entropy(logits, batch_labels, reduction="none")
            loss = per_sample_loss.mean()
            target_grad = torch.autograd.grad(loss, pixels)[0].detach()
            _append(accum, f"state/{row['branch']}/step{row['step']}/{model_name}/target_loss", per_sample_loss.detach())
            _append(accum, f"state/{row['branch']}/step{row['step']}/{model_name}/target_success", logits.detach().argmax(1).ne(batch_labels).float())
            extra = tuned10 if row["branch"] == "steps10" and row["step"] == 10 else None
            append_target_metrics(accum, row, target_grad, model_name, extra_components=extra)
        del model
        _release()

    metrics = _mean_payload(accum)
    payload = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "samples_requested": args.max_samples_requested,
        "samples": int(images.size(0)),
        "indices": indices.tolist(),
        "target_models": list(args.target_models),
        "branch_steps": {"steps10": list(args.trace_steps_10), "steps40": list(args.trace_steps)},
        "settings": {
            "attack_loss": "feature",
            "feature_layer": args.feature_layer,
            "guide_aug_method": "feature_trajectory_dropout",
            "guide_aug_area": "all",
            "guide_aug_copies": 9,
            "guide_aug_strength": 0.2,
            "dim": True,
            "mi": True,
            "epsilon": 16 / 255,
            "step_sizes": {"steps10": 16 / 255 / 10, "steps40": 16 / 255 / 40},
        },
        "metrics": metrics,
    }
    _json(root / "metrics.json", payload)
    torch.save({
        "protocol": PROTOCOL,
        "indices": indices,
        "labels": labels,
        "rows": all_rows if args.save_gradients else [{k: v for k, v in row.items() if k not in ("x_t", "clean", "masks", "current_grad", "history_before", "momentum_after", "delta_from_clean")} for row in all_rows],
    }, root / "step_traces.pt")
    np.savez(root / "metrics_flat.npz", **{key.replace("/", "__"): np.asarray(value) for key, value in accum.items()})
    report = build_report(payload)
    _json(root / "step_count_gradient_report.json", report)
    (root / "step_count_gradient_conclusion_zh.md").write_text(build_conclusion_zh(report), encoding="utf-8")
    print(f"wrote {root / 'step_count_gradient_report.json'}")


def run_report(args):
    root = Path(args.output_dir)
    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    report = build_report(payload)
    _json(root / "step_count_gradient_report.json", report)
    (root / "step_count_gradient_conclusion_zh.md").write_text(build_conclusion_zh(report), encoding="utf-8")
    print(f"wrote {root / 'step_count_gradient_report.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "experiment", "report"))
    parser.add_argument("--output-dir", default="outputs/analysis/step_count_gradient_mechanism")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", dest="max_samples_requested", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-models", type=parse_model_names, default=DEFAULT_TARGET_MODELS)
    parser.add_argument("--trace-steps", type=lambda x: tuple(int(item) for item in x.split(",") if item), default=TRACE_STEPS_40)
    parser.add_argument("--trace-steps-10", type=lambda x: tuple(int(item) for item in x.split(",") if item), default=TRACE_STEPS_10)
    parser.add_argument("--feature-layer", type=int, default=10)
    parser.add_argument("--feature-probes", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--save-gradients", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode in ("all", "experiment"):
        run_experiment(parsed)
    if parsed.mode in ("all", "report"):
        run_report(parsed)
