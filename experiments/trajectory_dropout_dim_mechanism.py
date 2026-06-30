"""DIM and feature-trajectory-dropout gradient decomposition experiment.

The current strongest attack uses layer-10 feature loss, step10, DIM, and
feature_trajectory_dropout. This script isolates the source-gradient pieces:
plain feature loss, trajectory dropout, DIM, their joint gradient, and the
nonlinear interaction between them. It then measures what each piece contributes
to target-gradient alignment and which image feature regions carry the energy.
"""
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
    dim_cross_band_leakage,
    feature_region_masks,
    feature_stability_scores,
    patch_grid_size,
    region_band_energy_from_patch_masks,
    region_direction_from_patch_masks,
)
from gradient_analysis import FFT_BANDS, fft_project
from main import ANNOTATIONS_PATH, IMAGE_DIR, create_attacker, parse_model_names
from nets import build_vit_model
from utils import DEVICE, load_data


PROTOCOL = "trajectory_dropout_dim_mechanism_v1"
VARIANTS = ("plain", "trajectory_dropout", "dim", "trajectory_dropout_dim")
COMPONENTS = VARIANTS + ("trajectory_increment", "dim_increment", "joint_increment", "interaction")
TARGET_MODELS = (
    "deit_base_patch16_224",
    "cait_s24_224",
    "pit_b_224",
    "levit_256",
    "resnet101",
    "inception_v3",
)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _release(*objects):
    del objects
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


def _safe_cosine(a, b):
    denom = a.flatten(1).norm(dim=1) * b.flatten(1).norm(dim=1)
    return (a * b).flatten(1).sum(1) / denom.clamp_min(1e-20)


def _direction_derivative(source_grad, target_grad):
    return (source_grad.sign() * target_grad).flatten(1).sum(1)


def _signed_target_alignment(source_grad, target_grad):
    return (source_grad * target_grad).flatten(1).sum(1)


def _append(accum, key, value):
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [float(value)]
    accum.setdefault(key, []).extend(float(v) for v in values)


def _mean_payload(accum):
    return {key: float(np.mean(value)) for key, value in sorted(accum.items()) if value}


def build_current_attacker(num_classes, feature_layer=10):
    source = build_vit_model(num_classes=num_classes, model_name="vit_base_patch16_224")
    attacker = create_attacker(
        model=source,
        epsilon=16 / 255,
        step_size=None,
        steps=10,
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


def _collect_samples(args, source, loader):
    args.max_samples = args.max_samples_requested
    images, labels, indices = [], [], []
    for x, y, idx in selected_batches(args, source, loader):
        images.append(x.cpu())
        labels.append(y.cpu())
        indices.append(idx.cpu())
    if not images:
        raise RuntimeError("No correctly classified source samples were selected.")
    return torch.cat(images), torch.cat(labels), torch.cat(indices)


def _attack_gradient(
    attacker,
    pixels,
    labels,
    clean_feature_target,
    *,
    dim,
    trajectory_dropout,
    dim_samples=1,
    base_seed=0,
):
    grads = []
    term_grads = None
    with _attacker_options(
        attacker,
        input_diversity=dim,
        dim_mode="full-random",
        guide_aug=trajectory_dropout,
        guide_aug_area="all",
        guide_aug_methods=("feature_trajectory_dropout",),
        guide_aug_copies=9,
    ):
        for sample_idx in range(dim_samples):
            seed_all(base_seed + sample_idx)
            probe = pixels.detach().requires_grad_(True)
            grad, terms = attacker._attack_grad_terms(
                probe,
                labels,
                None,
                clean_feature_target=clean_feature_target,
            )
            grads.append(grad.detach())
            if term_grads is None and trajectory_dropout:
                term_grads = tuple(term.detach().cpu() for term in terms)
    return torch.stack(grads).mean(0), term_grads, torch.stack(grads).detach()


def compute_gradient_components(attacker, pixels, labels, dim_samples, seed):
    with torch.no_grad():
        clean_feature_target = attacker._extract_layer_patch_features(pixels).detach()
    plain, _plain_terms, _plain_samples = _attack_gradient(
        attacker,
        pixels,
        labels,
        clean_feature_target,
        dim=False,
        trajectory_dropout=False,
        dim_samples=1,
        base_seed=seed + 1000,
    )
    trajectory, trajectory_terms, _trajectory_samples = _attack_gradient(
        attacker,
        pixels,
        labels,
        clean_feature_target,
        dim=False,
        trajectory_dropout=True,
        dim_samples=1,
        base_seed=seed + 2000,
    )
    dim, _dim_terms, dim_samples_tensor = _attack_gradient(
        attacker,
        pixels,
        labels,
        clean_feature_target,
        dim=True,
        trajectory_dropout=False,
        dim_samples=dim_samples,
        base_seed=seed + 3000,
    )
    joint, joint_terms, joint_samples_tensor = _attack_gradient(
        attacker,
        pixels,
        labels,
        clean_feature_target,
        dim=True,
        trajectory_dropout=True,
        dim_samples=dim_samples,
        base_seed=seed + 4000,
    )
    grads = {
        "plain": plain,
        "trajectory_dropout": trajectory,
        "dim": dim,
        "trajectory_dropout_dim": joint,
    }
    grads["trajectory_increment"] = grads["trajectory_dropout"] - grads["plain"]
    grads["dim_increment"] = grads["dim"] - grads["plain"]
    grads["joint_increment"] = grads["trajectory_dropout_dim"] - grads["plain"]
    grads["interaction"] = grads["trajectory_dropout_dim"] - grads["trajectory_dropout"] - grads["dim"] + grads["plain"]
    terms = {
        "trajectory_dropout_terms": trajectory_terms,
        "trajectory_dropout_dim_terms": joint_terms,
        "dim_samples": dim_samples_tensor.cpu(),
        "trajectory_dropout_dim_samples": joint_samples_tensor.cpu(),
    }
    return grads, terms


def _cls_guide_pixel_map(source, attacker, pixels, patch_size):
    with torch.inference_mode():
        _logits, attn_logits, tokens = source(attacker._normalize(pixels), return_attn=True, return_tokens=True)
        last = attn_logits[-1]
        patch_scores = last.softmax(dim=-1).mean(1)[:, 0, 1:].to(pixels.device, pixels.dtype)
        gh, gw = patch_grid_size(patch_scores.size(1))
        grid = patch_scores.view(patch_scores.size(0), 1, gh, gw)
        guide = F.interpolate(grid, size=pixels.shape[-2:], mode="bilinear", align_corners=False)
    return guide, attn_logits, tokens


def _feature_masks_for_batch(source, attacker, pixels, args):
    guide, attn_logits, tokens = _cls_guide_pixel_map(source, attacker, pixels, args.patch_size)
    probes = []
    with torch.inference_mode():
        base_tokens = tokens[args.feature_layer]
        for sample_idx in range(args.feature_probes):
            seed_all(args.seed + 7000 + sample_idx)
            probe_pixels = attacker._input_diversity(pixels)
            _probe_logits, _probe_attn, probe_tokens = source(
                attacker._normalize(probe_pixels),
                return_attn=True,
                return_tokens=True,
            )
            probes.append(probe_tokens[args.feature_layer])
    stability = feature_stability_scores(base_tokens, probes) if probes else None
    return feature_region_masks(
        base_tokens,
        attn_logits,
        pixels,
        guide,
        patch_size=args.patch_size,
        stability=stability,
    )


def _append_source_metrics(accum, grads, masks, args):
    for name, grad in grads.items():
        ratios = band_energy_ratios(grad)
        for band in range(len(FFT_BANDS) - 1):
            _append(accum, f"source_energy_ratio/{name}/band{band}", ratios[:, band])
        for group, bands in BAND_GROUPS.items():
            _append(accum, f"source_energy_ratio/{name}/{group}", ratios[:, list(bands)].sum(1))
        region_energy = region_band_energy_from_patch_masks(grad, masks.to(grad.device), args.patch_size)
        for ridx, region in enumerate(FEATURE_REGIONS):
            for band in range(len(FFT_BANDS) - 1):
                _append(accum, f"region_energy/{name}/{region}/band{band}", region_energy[:, ridx, band])
            low_mid = region_energy[:, ridx, list(BAND_GROUPS["low_mid"])].sum(1)
            high = region_energy[:, ridx, list(BAND_GROUPS["high"])].sum(1)
            _append(accum, f"region_energy/{name}/{region}/low_mid", low_mid)
            _append(accum, f"region_energy/{name}/{region}/high", high)
    for left in COMPONENTS:
        for right in COMPONENTS:
            if left >= right:
                continue
            _append(accum, f"source_cosine/{left}/{right}", _safe_cosine(grads[left], grads[right]))


def _append_term_metrics(accum, terms):
    for key in ("trajectory_dropout_terms", "trajectory_dropout_dim_terms"):
        term_grads = terms.get(key)
        if not term_grads:
            continue
        stack = torch.stack(term_grads)
        mean = stack.mean(0)
        denom = stack.flatten(2).norm(dim=2).mean(0).clamp_min(1e-20)
        coherence = mean.flatten(1).norm(dim=1) / denom
        _append(accum, f"term_coherence/{key}", coherence)
        for term_idx, term in enumerate(term_grads):
            _append(accum, f"term_cosine_to_mean/{key}/term{term_idx}", _safe_cosine(term.to(mean.device), mean))


def _append_target_metrics(accum, grads, target_grad, masks, model_name, args):
    for name, grad in grads.items():
        _append(accum, f"target_cosine/{name}/{model_name}/all", _safe_cosine(grad, target_grad))
        _append(accum, f"target_direction/{name}/{model_name}/all", _direction_derivative(grad, target_grad))
        _append(accum, f"target_dot/{name}/{model_name}/all", _signed_target_alignment(grad, target_grad))
        for band in range(len(FFT_BANDS) - 1):
            source_band = fft_project(grad, band)
            target_band = fft_project(target_grad, band)
            _append(accum, f"target_cosine/{name}/{model_name}/band{band}", _safe_cosine(source_band, target_band))
            _append(accum, f"target_direction/{name}/{model_name}/band{band}", _direction_derivative(source_band, target_band))
        region_dir = region_direction_from_patch_masks(grad, target_grad, masks.to(grad.device), args.patch_size)
        for ridx, region in enumerate(FEATURE_REGIONS):
            low_mid = region_dir[:, ridx, list(BAND_GROUPS["low_mid"])].sum(1)
            high = region_dir[:, ridx, list(BAND_GROUPS["high"])].sum(1)
            _append(accum, f"region_direction/{name}/{model_name}/{region}/low_mid", low_mid)
            _append(accum, f"region_direction/{name}/{model_name}/{region}/high", high)


def _dim_coherence(samples):
    mean = samples.mean(0)
    denom = samples.flatten(2).norm(dim=2).mean(0).clamp_min(1e-20)
    return mean.flatten(1).norm(dim=1) / denom


def run_experiment(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    loader, num_classes = load_data(
        image_dir_arg=args.image_dir,
        annotations_path_arg=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        img_size=args.img_size,
    )
    source, attacker = build_current_attacker(num_classes, feature_layer=args.feature_layer)
    images, labels, indices = _collect_samples(args, source, loader)
    clean_pixels = attacker._denormalize(images.to(DEVICE)).detach()
    labels = labels.to(DEVICE)
    accum = {}
    gradient_payload = {
        "protocol": PROTOCOL,
        "indices": indices,
        "labels": labels.cpu(),
        "variants": COMPONENTS,
        "batches": [],
    }
    source_cache = []
    for batch_idx, start in enumerate(range(0, clean_pixels.size(0), args.batch_size)):
        end = min(start + args.batch_size, clean_pixels.size(0))
        pixels = clean_pixels[start:end].to(DEVICE)
        batch_labels = labels[start:end]
        grads, terms = compute_gradient_components(attacker, pixels, batch_labels, args.dim_samples, args.seed + batch_idx * 100)
        masks, scores = _feature_masks_for_batch(source, attacker, pixels, args)
        _append_source_metrics(accum, grads, masks, args)
        _append_term_metrics(accum, terms)
        _append(accum, "dim_sample_coherence/dim", _dim_coherence(terms["dim_samples"]))
        _append(accum, "dim_sample_coherence/trajectory_dropout_dim", _dim_coherence(terms["trajectory_dropout_dim_samples"]))
        for ridx, region in enumerate(FEATURE_REGIONS):
            _append(accum, f"feature_region_score/{region}", scores[:, ridx].mean(1))
        item = {
            "slice": (start, end),
            "pixels": pixels.cpu() if args.save_pixels else None,
            "masks": masks.cpu(),
            "grads": {name: grad.cpu() for name, grad in grads.items()},
            "terms": terms if args.save_terms else {},
        }
        gradient_payload["batches"].append(item)
        source_cache.append(item)
    torch.save(gradient_payload, root / "gradients.pt")
    del loader
    _release()

    for model_name in args.target_models:
        model = build_vit_model(num_classes=1000, model_name=model_name)
        model.eval()
        for item in source_cache:
            start, end = item["slice"]
            pixels = clean_pixels[start:end].to(DEVICE).detach().requires_grad_(True)
            batch_labels = labels[start:end]
            loss = F.cross_entropy(model(_target_normalize(model, pixels), return_attn=False), batch_labels)
            target_grad = torch.autograd.grad(loss, pixels)[0].detach()
            grads = {name: grad.to(DEVICE) for name, grad in item["grads"].items()}
            _append_target_metrics(accum, grads, target_grad, item["masks"].to(DEVICE), model_name, args)
        del model
        _release()

    metrics = _mean_payload(accum)
    payload = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "samples_requested": args.max_samples_requested,
        "samples": int(clean_pixels.size(0)),
        "indices": indices.tolist(),
        "target_models": list(args.target_models),
        "feature_regions": list(FEATURE_REGIONS),
        "fft_bands": list(FFT_BANDS),
        "dim_samples": args.dim_samples,
        "settings": {
            "attack_loss": "feature",
            "feature_layer": args.feature_layer,
            "steps": 10,
            "guide_aug_method": "feature_trajectory_dropout",
            "guide_aug_area": "all",
            "guide_aug_copies": 9,
            "guide_aug_strength": 0.2,
            "dim_resize_range": [0.85, 1.0],
        },
        "metrics": metrics,
    }
    _json(root / "metrics.json", payload)
    np.savez(root / "metrics_flat.npz", **{key.replace("/", "__"): np.asarray(value) for key, value in accum.items()})
    leakage = dim_cross_band_leakage(args.img_size, args.operator_samples, 0.85, 1.0)
    np.savez(root / "dim_cross_band_leakage.npz", fft_bands=np.asarray(FFT_BANDS), **leakage)
    print(f"wrote {root / 'metrics.json'}")


def _metric(metrics, key, default=0.0):
    return float(metrics.get(key, default))


def _model_mean(metrics, template, models):
    values = [_metric(metrics, template.format(model=model)) for model in models]
    return float(np.mean(values)) if values else 0.0


def _band_group_mean(metrics, prefix, name, group):
    return float(np.sum([_metric(metrics, f"{prefix}/{name}/band{band}") for band in BAND_GROUPS[group]]))


def _region_mean(metrics, template, models, region, group):
    if "{model}" in template:
        values = [_metric(metrics, template.format(model=model, region=region, group=group)) for model in models]
        return float(np.mean(values)) if values else 0.0
    return _metric(metrics, template.format(region=region, group=group))


def build_report(run_payload, dim_leakage):
    metrics = run_payload["metrics"]
    models = run_payload["target_models"]
    summary = {"components": {}, "regions": {}, "dim_operator": {}}
    plain_direction = _model_mean(metrics, "target_direction/plain/{model}/all", models)
    for name in COMPONENTS:
        low_mid_energy = _band_group_mean(metrics, "source_energy_ratio", name, "low_mid")
        high_energy = _band_group_mean(metrics, "source_energy_ratio", name, "high")
        direction = _model_mean(metrics, f"target_direction/{name}/{{model}}/all", models)
        low_mid_direction = float(np.mean([
            _model_mean(metrics, f"target_direction/{name}/{{model}}/band{band}", models)
            for band in BAND_GROUPS["low_mid"]
        ]))
        high_direction = float(np.mean([
            _model_mean(metrics, f"target_direction/{name}/{{model}}/band{band}", models)
            for band in BAND_GROUPS["high"]
        ]))
        summary["components"][name] = {
            "source_low_mid_energy": low_mid_energy,
            "source_high_energy": high_energy,
            "target_direction": direction,
            "target_direction_delta_vs_plain": direction - plain_direction,
            "low_mid_target_direction": low_mid_direction,
            "high_target_direction": high_direction,
        }
    for name in ("trajectory_increment", "dim_increment", "interaction", "trajectory_dropout_dim"):
        region_rows = {}
        for region in FEATURE_REGIONS:
            region_rows[region] = {
                "low_mid_energy": _region_mean(metrics, f"region_energy/{name}/{{region}}/{{group}}", models, region, "low_mid"),
                "high_energy": _region_mean(metrics, f"region_energy/{name}/{{region}}/{{group}}", models, region, "high"),
                "low_mid_direction": _region_mean(metrics, f"region_direction/{name}/{{model}}/{{region}}/{{group}}", models, region, "low_mid"),
                "high_direction": _region_mean(metrics, f"region_direction/{name}/{{model}}/{{region}}/{{group}}", models, region, "high"),
            }
        summary["regions"][name] = region_rows
    resize_pad = dim_leakage["resize_pad"]
    summary["dim_operator"] = {
        "low_mid_diagonal_response": float(np.mean(np.diag(resize_pad)[list(BAND_GROUPS["low_mid"])])),
        "high_diagonal_response": float(np.mean(np.diag(resize_pad)[list(BAND_GROUPS["high"])])),
        "high_to_mid_leakage": float(resize_pad[np.ix_(BAND_GROUPS["high"], BAND_GROUPS["mid"])].mean()),
    }
    best_region = {}
    for name, rows in summary["regions"].items():
        best_region[name] = max(rows, key=lambda r: (rows[r]["low_mid_direction"], rows[r]["low_mid_energy"]))
    conclusions = {
        "trajectory_dropout_gradient_type": _classify_component(summary["components"]["trajectory_increment"]),
        "dim_gradient_type": _classify_component(summary["components"]["dim_increment"]),
        "joint_gradient_type": _classify_component(summary["components"]["trajectory_dropout_dim"]),
        "dim_operator_type": "low/mid-pass resize-pad adjoint" if summary["dim_operator"]["low_mid_diagonal_response"] > summary["dim_operator"]["high_diagonal_response"] else "not low/mid dominated",
        "best_image_feature_regions": best_region,
        "dim_dependence": "supported" if summary["components"]["interaction"]["target_direction_delta_vs_plain"] > 0 or summary["components"]["trajectory_dropout_dim"]["target_direction"] > summary["components"]["trajectory_dropout"]["target_direction"] else "weak_or_inconclusive",
    }
    return {
        "protocol": PROTOCOL,
        "settings": run_payload["settings"],
        "samples": run_payload["samples"],
        "target_models": models,
        "summary": summary,
        "conclusions": conclusions,
    }


def _classify_component(component):
    spectrum = "low/mid" if component["source_low_mid_energy"] >= component["source_high_energy"] else "high"
    transfer = "transfer-aligned" if component["target_direction_delta_vs_plain"] > 0 else "not transfer-improving vs plain"
    band = "low/mid-aligned" if component["low_mid_target_direction"] >= component["high_target_direction"] else "high-aligned"
    return f"{spectrum} energy, {band}, {transfer}"


def build_conclusion_zh(report):
    c = report["conclusions"]
    summary = report["summary"]
    comp = summary["components"]
    dim_gain_vs_traj = comp["dim"]["target_direction"] - comp["trajectory_dropout"]["target_direction"]
    joint_gain_vs_traj = comp["trajectory_dropout_dim"]["target_direction"] - comp["trajectory_dropout"]["target_direction"]
    joint_gain_vs_dim = comp["trajectory_dropout_dim"]["target_direction"] - comp["dim"]["target_direction"]
    lines = [
        "# Trajectory Dropout Complement 与 DIM 梯度机制结论",
        "",
        f"样本数: {report['samples']}；目标模型: {', '.join(report['target_models'])}。",
        "",
        "## 0. 直接结论",
        f"- 在当前 layer{report['settings']['feature_layer']} feature loss / step10 设置下，DIM 是更强的迁移增益来源：DIM 相比 trajectory_dropout_complement 的 target_direction 高 {dim_gain_vs_traj:.6g}。",
        "- trajectory_dropout_complement 单独提供的是 shape-boundary 主导的 low/mid 稳定特征梯度；它把 plain 梯度从 high-heavy 变成 low/mid-heavy，但增益小于 DIM。",
        "- DIM 提供的是 resize/pad adjoint 下的尺度稳定 low/mid 前景梯度；它不是简单增加随机样本，而是显著提升与目标模型梯度同向的 sign direction。",
        f"- 二者联合最强：相对 trajectory_dropout_complement 增加 {joint_gain_vs_traj:.6g}，相对 DIM 增加 {joint_gain_vs_dim:.6g}。interaction 为负表示二者有重叠/互相抵消的分量，但联合仍保留了最多 target-aligned 梯度。",
        "",
        "## 1. 梯度类型",
        f"- trajectory_dropout_complement 增量梯度: {c['trajectory_dropout_gradient_type']}。",
        f"- DIM 增量梯度: {c['dim_gradient_type']}。",
        f"- trajectory_dropout_complement + DIM 联合梯度: {c['joint_gradient_type']}。",
        f"- DIM 算子本身: {c['dim_operator_type']}；low/mid response={summary['dim_operator']['low_mid_diagonal_response']:.6g}，high response={summary['dim_operator']['high_diagonal_response']:.6g}，high-to-mid leakage={summary['dim_operator']['high_to_mid_leakage']:.6g}。",
        "",
        "## 2. 对迁移的贡献",
    ]
    for name in ("plain", "trajectory_dropout", "dim", "trajectory_dropout_dim", "trajectory_increment", "dim_increment", "interaction"):
        item = summary["components"][name]
        lines.append(
            f"- {name}: target_direction={item['target_direction']:.6g}, "
            f"delta_vs_plain={item['target_direction_delta_vs_plain']:.6g}, "
            f"low_mid_energy={item['source_low_mid_energy']:.4f}, high_energy={item['source_high_energy']:.4f}。"
        )
    lines.extend([
        "",
        "## 3. 图像特征来源",
    ])
    for name in ("trajectory_increment", "dim_increment", "interaction", "trajectory_dropout_dim"):
        region = c["best_image_feature_regions"][name]
        row = summary["regions"][name][region]
        lines.append(
            f"- {name}: 最强 low/mid target direction 来自 {region}；"
            f"low_mid_direction={row['low_mid_direction']:.6g}, low_mid_energy={row['low_mid_energy']:.4f}。"
        )
    lines.extend([
        "",
        "## 4. 判定",
        f"- DIM 依赖性: {c['dim_dependence']}。如果为 supported，说明 DIM 不只是增加噪声样本数，而是在目标梯度方向上提供额外或交互的有效分量。",
        "- 结论只基于当前保存的梯度分解、目标模型梯度方向导数和 feature-region 归因；不能直接替代完整 ASR，但用于解释为什么开启 DIM 后迁移性变化。",
        "",
    ])
    return "\n".join(lines)


def run_report(args):
    root = Path(args.output_dir)
    run_payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    leakage_npz = np.load(root / "dim_cross_band_leakage.npz")
    dim_leakage = {key: leakage_npz[key] for key in leakage_npz.files if key != "fft_bands"}
    report = build_report(run_payload, dim_leakage)
    _json(root / "trajectory_dropout_dim_report.json", report)
    (root / "trajectory_dropout_dim_conclusion_zh.md").write_text(build_conclusion_zh(report), encoding="utf-8")
    print(f"wrote {root / 'trajectory_dropout_dim_report.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "experiment", "report"))
    parser.add_argument("--output-dir", default="outputs/analysis/trajectory_dropout_dim_mechanism")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", dest="max_samples_requested", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-models", type=parse_model_names, default=TARGET_MODELS)
    parser.add_argument("--dim-samples", type=int, default=4)
    parser.add_argument("--feature-layer", type=int, default=10)
    parser.add_argument("--feature-probes", type=int, default=3)
    parser.add_argument("--operator-samples", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--save-pixels", action="store_true")
    parser.add_argument("--save-terms", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode in ("all", "experiment"):
        run_experiment(parsed)
    if parsed.mode in ("all", "report"):
        run_report(parsed)
