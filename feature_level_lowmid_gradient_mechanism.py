"""Feature-level low/mid input-gradient mechanism experiment.

This experiment separates transfer-relevant low/mid gradients by feature-level
patch structure instead of relying only on foreground/background attention.
It also diagnoses single augmentation methods and the DIM operator response.
"""
import argparse
import gc
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from causal_analysis import MAIN_TARGETS, _target_normalize, build_baseline, seed_all, selected_batches
from gradient_analysis import FFT_BANDS, fft_project, run_analyzed_attack
from main import ANNOTATIONS_PATH, IMAGE_DIR, parse_model_names
from nets import build_vit_model
from utils import DEVICE, load_data

PROTOCOL = "feature_level_lowmid_gradient_mechanism_v1"
TRACE_STEPS = (1, 10, 20, 40)
BAND_GROUPS = {"low": (0, 1, 2), "mid": (3, 4, 5), "high": (6, 7), "low_mid": (0, 1, 2, 3, 4, 5)}
FEATURE_REGIONS = ("fg_attention", "bg_attention", "texture_stable", "shape_boundary", "contextual_background", "texture_noise")
AUG_METHODS = ("dropout", "jitter", "freq", "fft_lowboost", "illumination_low", "band_noise_low", "band_noise_mid", "white_noise", "band_noise_high")
DIM_MODES = ("resize_only", "pad_only", "resize_pad", "nearest_resize", "bicubic_resize")


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


def fft_band_count():
    return len(FFT_BANDS) - 1


def band_energy_ratios(x):
    total = x.square().flatten(1).sum(1).clamp_min(1e-20)
    return torch.stack([fft_project(x, band).square().flatten(1).sum(1) / total for band in range(fft_band_count())], dim=1)


def patch_energy_map(x, patch_size):
    return F.avg_pool2d(x.square().sum(1, keepdim=True), kernel_size=patch_size, stride=patch_size).flatten(1)


def patch_band_energy(x, patch_size):
    return torch.stack([patch_energy_map(fft_project(x, band), patch_size) for band in range(fft_band_count())], dim=1)


def low_mid_high_patch_energy(x, patch_size):
    bands = patch_band_energy(x, patch_size)
    low_mid = bands[:, list(BAND_GROUPS["low_mid"])].sum(1)
    high = bands[:, list(BAND_GROUPS["high"])].sum(1)
    total = bands.sum(1).clamp_min(1e-20)
    return low_mid / total, high / total


def patch_grid_size(num_patches):
    side = int(round(num_patches ** 0.5))
    if side * side != num_patches:
        raise ValueError(f"Expected square ViT patch grid, got {num_patches} patches.")
    return side, side


def patch_mask_to_pixel(mask, height, width):
    gh, gw = patch_grid_size(mask.size(1))
    grid = mask.view(mask.size(0), 1, gh, gw).to(torch.float32)
    return F.interpolate(grid, size=(height, width), mode="nearest")


def _patch_tokens(tokens):
    if tokens.ndim != 3 or tokens.size(1) < 2:
        raise ValueError("tokens must have shape [B, CLS+patches, D].")
    return tokens[:, 1:, :]


def cls_patch_attention(attn_logits):
    if attn_logits is None:
        raise ValueError("attn_logits are required for CLS-to-patch attention scores.")
    last = attn_logits[-1]
    weights = last.softmax(dim=-1).mean(1)
    return weights[:, 0, 1:]


def feature_stability_scores(base_tokens, probe_tokens):
    base = F.normalize(_patch_tokens(base_tokens), dim=-1)
    probes = [F.normalize(_patch_tokens(tokens), dim=-1) for tokens in probe_tokens]
    return torch.stack([(base * probe).sum(-1) for probe in probes]).mean(0)


def quantile_mask(score, q=0.67, high=True):
    threshold = torch.quantile(score.detach(), q if high else 1.0 - q, dim=1, keepdim=True)
    return score >= threshold if high else score <= threshold


def feature_region_masks(tokens, attn_logits, pixels, guide_pixel_map, patch_size=16, stability=None, q=0.67):
    patches = _patch_tokens(tokens)
    batch, num_patches, _hidden = patches.shape
    height, width = pixels.shape[-2:]
    guide_patch = F.avg_pool2d(guide_pixel_map.to(pixels.device, pixels.dtype).clamp(0, 1), kernel_size=patch_size, stride=patch_size).flatten(1)
    if guide_patch.shape != (batch, num_patches):
        guide_patch = F.interpolate(guide_pixel_map.to(pixels.device, pixels.dtype), size=patch_grid_size(num_patches), mode="bilinear", align_corners=False).flatten(1)

    cls_attn = cls_patch_attention(attn_logits).to(pixels.device, pixels.dtype)
    norm = patches.norm(dim=-1)
    fg_weight = guide_patch / guide_patch.sum(1, keepdim=True).clamp_min(1e-20)
    centroid = (patches * fg_weight.unsqueeze(-1)).sum(1)
    sim_to_fg = (F.normalize(patches, dim=-1) * F.normalize(centroid, dim=-1).unsqueeze(1)).sum(-1)
    low_mid_ratio, high_ratio = low_mid_high_patch_energy(pixels, patch_size)
    low_mid_image = sum(fft_project(pixels, band) for band in BAND_GROUPS["low_mid"])
    dy = low_mid_image[..., 1:, :] - low_mid_image[..., :-1, :]
    dx = low_mid_image[..., :, 1:] - low_mid_image[..., :, :-1]
    edge = F.pad(dx.abs(), (0, 1, 0, 0)).mean(1, keepdim=True) + F.pad(dy.abs(), (0, 0, 0, 1)).mean(1, keepdim=True)
    edge_patch = F.avg_pool2d(edge, kernel_size=patch_size, stride=patch_size).flatten(1)
    if stability is None:
        stability = norm / norm.mean(1, keepdim=True).clamp_min(1e-20)

    fg_attention = guide_patch >= torch.quantile(guide_patch, 0.67, dim=1, keepdim=True)
    bg_attention = guide_patch <= torch.quantile(guide_patch, 0.33, dim=1, keepdim=True)
    texture_stable = quantile_mask(stability, q=q, high=True) & (high_ratio <= torch.quantile(high_ratio, 0.67, dim=1, keepdim=True))
    shape_boundary = quantile_mask(edge_patch * low_mid_ratio, q=q, high=True) & (high_ratio <= torch.quantile(high_ratio, 0.67, dim=1, keepdim=True))
    context_score = 0.5 * cls_attn + 0.5 * sim_to_fg
    contextual_background = bg_attention & quantile_mask(context_score, q=0.60, high=True)
    texture_noise = quantile_mask(high_ratio, q=q, high=True) & quantile_mask(stability, q=0.60, high=False)
    masks = torch.stack([fg_attention, bg_attention, texture_stable, shape_boundary, contextual_background, texture_noise], dim=1).to(pixels.dtype)
    scores = torch.stack([guide_patch, 1 - guide_patch, stability, edge_patch * low_mid_ratio, context_score, high_ratio * (1 - stability)], dim=1)
    if masks.shape[-1] != (height // patch_size) * (width // patch_size):
        raise ValueError("patch mask shape does not match image patch grid.")
    return masks, scores


def region_band_energy_from_patch_masks(grad, patch_masks, patch_size):
    values = []
    for band in range(fft_band_count()):
        energy = patch_energy_map(fft_project(grad, band), patch_size)
        denom = energy.sum(1).clamp_min(1e-20)
        values.append((patch_masks * energy.unsqueeze(1)).sum(-1) / denom.unsqueeze(1))
    return torch.stack(values, dim=2)


def region_direction_from_patch_masks(source_grad, target_grad, patch_masks, patch_size):
    height, width = source_grad.shape[-2:]
    rows = []
    for band in range(fft_band_count()):
        source_band = fft_project(source_grad, band).sign()
        target_band = fft_project(target_grad, band)
        by_region = []
        for ridx in range(patch_masks.size(1)):
            pixel_mask = patch_mask_to_pixel(patch_masks[:, ridx], height, width).to(source_grad.device, source_grad.dtype)
            by_region.append((pixel_mask * source_band * pixel_mask * target_band).flatten(1).sum(1))
        rows.append(torch.stack(by_region, dim=1))
    return torch.stack(rows, dim=2)


def classify_report_rule(energy_delta, derivative_delta, model_positive_count, min_models=6):
    if derivative_delta > 0 and model_positive_count >= min_models:
        return "supported"
    if energy_delta > 0 and not (derivative_delta > 0 and model_positive_count >= min_models):
        return "spectrum_only"
    if derivative_delta > 0:
        return "association_only"
    return "inconclusive"


def _dim_transform_adjoint_probe(x, mode, scale=0.875, top=0, left=0, resize_mode="bilinear"):
    height, width = x.shape[-2:]
    if mode == "pad_only":
        small = max(1, int(round(height * scale)))
        cropped = x[..., top:top + small, left:left + small]
        return F.pad(cropped, (left, width - small - left, top, height - small - top))
    if mode in ("resize_only", "nearest_resize", "bicubic_resize"):
        interpolation = {"resize_only": resize_mode, "nearest_resize": "nearest", "bicubic_resize": "bicubic"}[mode]
        small = max(1, int(round(height * scale)))
        kwargs = {} if interpolation == "nearest" else {"align_corners": False}
        return F.interpolate(F.interpolate(x, size=(small, small), mode=interpolation, **kwargs), size=(height, width), mode=interpolation, **kwargs)
    if mode == "resize_pad":
        small = max(1, int(round(height * scale)))
        resized = F.interpolate(x, size=(small, small), mode=resize_mode, align_corners=False)
        padded = F.pad(resized, (left, width - small - left, top, height - small - top))
        cropped = padded[..., top:top + small, left:left + small]
        return F.interpolate(cropped, size=(height, width), mode=resize_mode, align_corners=False)
    raise ValueError(f"Unsupported DIM diagnostic mode: {mode}")


def dim_cross_band_leakage(img_size=224, samples=16, resize_min=0.85, resize_max=1.0, modes=DIM_MODES, device=None):
    device = device or DEVICE
    eye = []
    for band in range(fft_band_count()):
        noise = torch.randn(1, 3, img_size, img_size, device=device)
        component = fft_project(noise, band)
        component = component / component.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-20)
        eye.append(component)
    result = {}
    for mode in modes:
        matrix = torch.zeros(fft_band_count(), fft_band_count(), device=device)
        for _ in range(samples):
            scale = float(torch.empty((), device=device).uniform_(resize_min, resize_max))
            small = max(1, int(round(img_size * scale)))
            pad = img_size - small
            top = int(torch.randint(0, pad + 1, (), device=device)) if pad > 0 else 0
            left = int(torch.randint(0, pad + 1, (), device=device)) if pad > 0 else 0
            for src_band, component in enumerate(eye):
                transformed = _dim_transform_adjoint_probe(component, mode, scale=scale, top=top, left=left)
                denom = transformed.square().flatten(1).sum(1).clamp_min(1e-20)
                for dst_band in range(fft_band_count()):
                    matrix[src_band, dst_band] += (fft_project(transformed, dst_band).square().flatten(1).sum(1) / denom).mean()
        result[mode] = (matrix / float(samples)).detach().cpu().numpy()
    return result


def _collect_samples(args, source, loader):
    args.max_samples = args.max_samples_requested
    images, labels, indices, sizes = [], [], [], []
    for x, y, idx in selected_batches(args, source, loader):
        images.append(x.cpu()); labels.append(y.cpu()); indices.append(idx.cpu()); sizes.append(x.size(0))
    return torch.cat(images), torch.cat(labels), torch.cat(indices), sizes


def _collect_trace(args, source, attacker, seed):
    path = Path(args.output_dir) / "traces" / f"seed_{seed}.pt"
    if path.exists() and not args.force:
        payload = torch.load(path, map_location="cpu")
        if payload.get("protocol") == PROTOCOL and payload.get("samples_requested") == args.max_samples_requested:
            return payload
    seed_all(seed)
    loader, _num_classes = load_data(args.image_dir, args.annotations_path, args.batch_size, args.num_workers, 2, args.img_size)
    clean, labels, indices, sizes = _collect_samples(args, source, loader)
    rows, start = [], 0
    for size in sizes:
        end = start + size
        traces = []
        run_analyzed_attack(attacker, clean[start:end], labels[start:end], trace_callback=traces.append, diagnostics=False)
        keep = {row["step"]: row for row in traces if row["step"] in args.trace_steps}
        for step in args.trace_steps:
            row = keep[step]
            rows.append({"step": step, "x_t": row["x_t"].cpu(), "guide_map": row["guide_map"].cpu() if row["guide_map"] is not None else None})
        start = end
    payload = {"protocol": PROTOCOL, "seed": seed, "samples_requested": args.max_samples_requested, "indices": indices, "labels": labels, "batch_sizes": sizes, "trace_steps": tuple(args.trace_steps), "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _source_grad(attacker, pixels, labels, guide, *, dim=False, guide_aug=False, methods=("dropout", "jitter", "freq")):
    with _attacker_options(attacker, input_diversity=dim, guide_aug=guide_aug, guide_aug_methods=methods, guide_aug_area="background"):
        probe = pixels.detach().requires_grad_(True)
        return attacker._attack_grad(probe, labels, guide).detach()


def _feature_masks_for_batch(source, attacker, pixels, guide, args):
    with torch.inference_mode():
        normalized = attacker._normalize(pixels)
        _logits, attn, tokens = source(normalized, return_attn=True, return_tokens=True)
        base_tokens = tokens[-1]
        probes = []
        for method in ("jitter", "freq")[:args.feature_probes]:
            aug = attacker._augment_full_image(pixels, method)
            _aug_logits, _aug_attn, aug_tokens = source(attacker._normalize(aug), return_attn=True, return_tokens=True)
            probes.append(aug_tokens[-1])
        stability = feature_stability_scores(base_tokens, probes) if probes else None
        return feature_region_masks(base_tokens, attn, pixels, guide, patch_size=args.patch_size, stability=stability)


def run_experiment(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    leakage = dim_cross_band_leakage(args.img_size, args.operator_samples, args.dim_resize_min, args.dim_resize_max)
    np.savez(root / "dim_cross_band_leakage_metrics.npz", fft_bands=np.asarray(FFT_BANDS), **{k: v for k, v in leakage.items()})
    response = {mode: np.diag(matrix) for mode, matrix in leakage.items()}
    np.savez(root / "dim_jacobian_frequency_response.npz", fft_bands=np.asarray(FFT_BANDS), **response)

    for seed in args.seeds:
        path = root / "runs" / f"seed_{seed}.json"
        if path.exists() and not args.force:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol") == PROTOCOL and payload.get("samples_requested") == args.max_samples_requested:
                continue
        seed_all(seed)
        loader, num_classes = load_data(args.image_dir, args.annotations_path, args.batch_size, args.num_workers, 2, args.img_size)
        source, attacker = build_baseline(num_classes)
        source.eval()
        attacker.guide_aug_area = "background"
        attacker.guide_aug_methods = ("dropout", "jitter", "freq")
        attacker.guide_aug_copies = 3
        attacker.guide_aug_strength = 0.2
        trace = _collect_trace(args, source, attacker, seed)
        labels = trace["labels"]
        accum = {}
        cache = {}
        sample_start = 0
        for batch_idx, offset in enumerate(range(0, len(trace["rows"]), len(args.trace_steps))):
            rows = trace["rows"][offset:offset + len(args.trace_steps)]
            batch_size = trace["batch_sizes"][batch_idx]
            batch_labels = labels[sample_start:sample_start + batch_size].to(DEVICE)
            for row in rows:
                step = row["step"]
                pixels = row["x_t"].to(DEVICE)
                guide = row["guide_map"].to(DEVICE) if row["guide_map"] is not None else attacker._build_guide_pixel_map(attacker._normalize(pixels), pixels.size(-1))
                masks, scores = _feature_masks_for_batch(source, attacker, pixels, guide, args)
                plain = _source_grad(attacker, pixels, batch_labels, guide, dim=False, guide_aug=False)
                method_grads = {method: _source_grad(attacker, pixels, batch_labels, guide, dim=False, guide_aug=True, methods=(method,)) for method in args.aug_methods}
                combo = _source_grad(attacker, pixels, batch_labels, guide, dim=False, guide_aug=True, methods=("dropout", "jitter", "freq"))
                cache[(offset, step)] = {"plain": plain.cpu(), "masks": masks.cpu(), "combo": combo.cpu(), **{f"method_{k}": v.cpu() for k, v in method_grads.items()}}
                region_energy = region_band_energy_from_patch_masks(plain, masks.to(DEVICE), args.patch_size)
                for ridx, region in enumerate(FEATURE_REGIONS):
                    for band in range(fft_band_count()):
                        accum.setdefault(f"step{step}_region_energy_{region}_band{band}", []).extend(region_energy[:, ridx, band].cpu().tolist())
                    accum.setdefault(f"step{step}_region_score_{region}", []).extend(scores[:, ridx].mean(1).detach().cpu().tolist())
                plain_ratio = band_energy_ratios(plain)
                for method, grad in {"combo": combo, **method_grads}.items():
                    delta = band_energy_ratios(grad) - plain_ratio
                    for band in range(fft_band_count()):
                        accum.setdefault(f"step{step}_aug_{method}_energy_delta_band{band}", []).extend(delta[:, band].cpu().tolist())
            sample_start += batch_size
        del loader
        _release()
        for model_name in args.target_models:
            model = build_vit_model(num_classes=1000, model_name=model_name)
            model.eval()
            sample_start = 0
            for batch_idx, offset in enumerate(range(0, len(trace["rows"]), len(args.trace_steps))):
                rows = trace["rows"][offset:offset + len(args.trace_steps)]
                batch_size = trace["batch_sizes"][batch_idx]
                batch_labels = labels[sample_start:sample_start + batch_size].to(DEVICE)
                for row in rows:
                    step = row["step"]
                    pixels = row["x_t"].to(DEVICE).detach().requires_grad_(True)
                    loss = F.cross_entropy(model(_target_normalize(model, pixels), return_attn=False), batch_labels)
                    target_grad = torch.autograd.grad(loss, pixels)[0].detach()
                    item = cache[(offset, step)]
                    masks = item["masks"].to(DEVICE)
                    plain = item["plain"].to(DEVICE)
                    region_dir = region_direction_from_patch_masks(plain, target_grad, masks, args.patch_size)
                    for ridx, region in enumerate(FEATURE_REGIONS):
                        for band in range(fft_band_count()):
                            accum.setdefault(f"step{step}_{model_name}_region_direction_{region}_band{band}", []).extend(region_dir[:, ridx, band].cpu().tolist())
                    for key, grad_cpu in item.items():
                        if not key.startswith("method_") and key != "combo":
                            continue
                        grad = grad_cpu.to(DEVICE)
                        for band in range(fft_band_count()):
                            source_band = fft_project(grad, band).sign()
                            target_band = fft_project(target_grad, band)
                            plain_band = fft_project(plain, band).sign()
                            delta = ((source_band - plain_band) * target_band).flatten(1).sum(1)
                            accum.setdefault(f"step{step}_{model_name}_aug_{key.replace('method_', '')}_direction_delta_band{band}", []).extend(delta.cpu().tolist())
                sample_start += batch_size
            del model
            _release()
        summary = {key: float(np.mean(value)) for key, value in accum.items() if value}
        _json(path, {"protocol": PROTOCOL, "seed": seed, "samples_requested": args.max_samples_requested, "samples": int(len(trace["indices"]),), "indices": trace["indices"].tolist(), "trace_steps": list(args.trace_steps), "target_models": list(args.target_models), "feature_regions": list(FEATURE_REGIONS), "aug_methods": list(args.aug_methods), "fft_bands": list(FFT_BANDS), "metrics": summary})
        print(f"wrote {path}")
        del source, attacker
        _release()


def _mean_metric(runs, key):
    values = [run["metrics"][key] for run in runs if key in run["metrics"]]
    return None if not values else float(np.mean(values))


def _sum_bands(runs, template, step, bands):
    values = [_mean_metric(runs, template.format(step=step, band=band)) for band in bands]
    values = [value for value in values if value is not None]
    return None if not values else float(np.sum(values))


def build_report(runs, dim_response, dim_leakage):
    target_models = runs[0]["target_models"]
    steps = runs[0]["trace_steps"]
    min_models = min(6, len(target_models))
    summary = {}
    for step in steps:
        item = {"regions": {}, "augmentations": {}}
        for region in FEATURE_REGIONS:
            energy = _sum_bands(runs, f"step{{step}}_region_energy_{region}_band{{band}}", step, BAND_GROUPS["low_mid"]) or 0.0
            positives = 0
            direction_values = []
            for model in target_models:
                direction = np.mean([_mean_metric(runs, f"step{step}_{model}_region_direction_{region}_band{band}") or 0.0 for band in BAND_GROUPS["low_mid"]])
                direction_values.append(direction)
                positives += int(direction > 0)
            item["regions"][region] = {"low_mid_energy": float(energy), "low_mid_direction": float(np.mean(direction_values)), "positive_models": positives}
        for method in runs[0]["aug_methods"] + ["combo"]:
            energy_delta = _sum_bands(runs, f"step{{step}}_aug_{method}_energy_delta_band{{band}}", step, BAND_GROUPS["low_mid"]) or 0.0
            high_delta = _sum_bands(runs, f"step{{step}}_aug_{method}_energy_delta_band{{band}}", step, BAND_GROUPS["high"]) or 0.0
            positives = 0
            derivative_values = []
            for model in target_models:
                derivative = np.mean([_mean_metric(runs, f"step{step}_{model}_aug_{method}_direction_delta_band{band}") or 0.0 for band in BAND_GROUPS["low_mid"]])
                derivative_values.append(derivative)
                positives += int(derivative > 0)
            derivative_delta = float(np.mean(derivative_values))
            item["augmentations"][method] = {"low_mid_energy_delta": float(energy_delta), "high_energy_delta": float(high_delta), "low_mid_direction_delta": derivative_delta, "positive_models": positives, "rule": classify_report_rule(float(energy_delta), derivative_delta, positives, min_models)}
        summary[str(step)] = item
    final = summary[str(steps[-1])]
    best_region = max(final["regions"], key=lambda r: (final["regions"][r]["positive_models"], final["regions"][r]["low_mid_direction"], final["regions"][r]["low_mid_energy"]))
    fg_bg_best = max(("fg_attention", "bg_attention"), key=lambda r: final["regions"][r]["low_mid_direction"])
    feature_beats_fg_bg = final["regions"][best_region]["low_mid_direction"] > final["regions"][fg_bg_best]["low_mid_direction"] and best_region not in ("fg_attention", "bg_attention")
    aug_supported = [method for method, values in final["augmentations"].items() if values["rule"] == "supported"]
    dim_resize_pad = dim_leakage["resize_pad"]
    high_to_high = float(np.mean(np.diag(dim_resize_pad)[list(BAND_GROUPS["high"])]))
    low_mid_to_low_mid = float(np.mean(np.diag(dim_resize_pad)[list(BAND_GROUPS["low_mid"])]))
    offdiag_mid = float(dim_resize_pad[np.ix_(BAND_GROUPS["high"], BAND_GROUPS["mid"])].mean())
    conclusions = {
        "best_feature_region_for_low_mid": best_region if feature_beats_fg_bg else "fg_bg_baseline_not_beaten",
        "recommended_augmentations": aug_supported or ["evidence_inconclusive"],
        "dim_mechanism": "resize/pad adjoint is low/mid pass with high-band attenuation and measurable high-to-mid leakage" if low_mid_to_low_mid > high_to_high else "evidence_inconclusive",
        "dim_low_mid_response": low_mid_to_low_mid,
        "dim_high_response": high_to_high,
        "dim_high_to_mid_leakage": offdiag_mid,
    }
    return {"protocol": PROTOCOL, "fft_bands": list(FFT_BANDS), "seeds": [run["seed"] for run in runs], "summary": summary, "dim_frequency_response": {k: v.tolist() for k, v in dim_response.items()}, "dim_cross_band_leakage": {k: v.tolist() for k, v in dim_leakage.items()}, "conclusions": conclusions, "runs": runs}


def build_conclusion_zh(report):
    c = report["conclusions"]
    final_step = sorted(report["summary"], key=lambda x: int(x))[-1]
    aug = report["summary"][final_step]["augmentations"]
    lines = [
        "# Feature-Level Low/Mid 输入梯度机制结论",
        "",
        f"1. 最能区分 low/mid transfer-relevant 梯度的 feature-level 区域: {c['best_feature_region_for_low_mid']}。如果为 fg_bg_baseline_not_beaten，则说明新增 feature 分区未稳定优于 fg/bg baseline。",
        f"2. 推荐增强: {', '.join(c['recommended_augmentations'])}。推荐只基于 low/mid target direction derivative，不只基于 energy。",
        "3. 单方法增强频段结论:",
    ]
    for method, values in aug.items():
        dominant = "low/mid" if values["low_mid_energy_delta"] >= values["high_energy_delta"] else "high"
        lines.append(f"   - {method}: energy 更偏 {dominant}; transfer rule={values['rule']}; low/mid direction delta={values['low_mid_direction_delta']:.6g}")
    lines.extend([
        f"4. DIM 数学机制: {c['dim_mechanism']}。resize_pad low/mid response={c['dim_low_mid_response']:.6g}, high response={c['dim_high_response']:.6g}, high-to-mid leakage={c['dim_high_to_mid_leakage']:.6g}。",
        "5. 若某项规则未满足，结论保持 evidence_inconclusive，不把 energy increase 等同于迁移提升。",
        "",
    ])
    return "\n".join(lines)


def run_report(args):
    root = Path(args.output_dir)
    runs = []
    for seed in args.seeds:
        path = root / "runs" / f"seed_{seed}.json"
        if not path.exists():
            raise RuntimeError(f"Missing run metrics: {path}")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    dim_response_npz = np.load(root / "dim_jacobian_frequency_response.npz")
    dim_leakage_npz = np.load(root / "dim_cross_band_leakage_metrics.npz")
    dim_response = {key: dim_response_npz[key] for key in dim_response_npz.files if key != "fft_bands"}
    dim_leakage = {key: dim_leakage_npz[key] for key in dim_leakage_npz.files if key != "fft_bands"}
    report = build_report(runs, dim_response, dim_leakage)
    _json(root / "feature_level_lowmid_gradient_report.json", report)
    arrays = {"fft_bands": np.asarray(FFT_BANDS)}
    for run in runs:
        for key, value in run["metrics"].items():
            arrays[f"seed{run['seed']}_{key}"] = np.asarray(value)
    np.savez(root / "feature_region_lowmid_metrics.npz", **{k: v for k, v in arrays.items() if "region_" in k or k == "fft_bands"})
    np.savez(root / "augmentation_band_effect_metrics.npz", **{k: v for k, v in arrays.items() if "aug_" in k or k == "fft_bands"})
    (root / "feature_level_lowmid_gradient_conclusion_zh.md").write_text(build_conclusion_zh(report), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "experiment", "report"))
    parser.add_argument("--output-dir", default="outputs/feature_level_lowmid_gradient_mechanism")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", dest="max_samples_requested", type=int, default=100)
    parser.add_argument("--seeds", type=lambda x: tuple(map(int, x.split(","))), default=(0, 1))
    parser.add_argument("--trace-steps", type=lambda x: tuple(map(int, x.split(","))), default=TRACE_STEPS)
    parser.add_argument("--target-models", type=parse_model_names, default=MAIN_TARGETS)
    parser.add_argument("--aug-methods", type=lambda x: tuple(item.strip() for item in x.split(",") if item.strip()), default=AUG_METHODS)
    parser.add_argument("--feature-probes", type=int, default=2)
    parser.add_argument("--operator-samples", type=int, default=16)
    parser.add_argument("--dim-resize-min", type=float, default=0.85)
    parser.add_argument("--dim-resize-max", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode in ("all", "experiment"):
        run_experiment(parsed)
    if parsed.mode in ("all", "report"):
        run_report(parsed)
