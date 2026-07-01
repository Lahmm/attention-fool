"""DIM/BG image-gradient frequency mechanism experiment.

This experiment uses the same radial FFT bands for input images and input
gradients, then measures region attribution, DIM coherence, target alignment,
2x2 DIM/BG interaction, image-gradient patch correlations, and small band
ablations.
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

PROTOCOL = "dim_bg_image_gradient_frequency_math_v1"
VARIANTS = ("plain", "dim", "bg", "dim_bg")
TRACE_STEPS = (1, 10, 20, 40)
BAND_GROUPS = {"low": (0, 1, 2), "mid": (3, 4, 5), "high": (6, 7)}
REGIONS = ("fg", "bg")


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
    return torch.stack([
        fft_project(x, band).square().flatten(1).sum(1) / total
        for band in range(fft_band_count())
    ], dim=1)


def same_fft_band_edges():
    return tuple(FFT_BANDS)


def region_band_energy(projected_grad, guide):
    guide = guide.to(projected_grad.device, projected_grad.dtype).clamp(0.0, 1.0)
    fg = (guide * projected_grad).square().flatten(1).sum(1)
    bg = ((1.0 - guide) * projected_grad).square().flatten(1).sum(1)
    total = projected_grad.square().flatten(1).sum(1).clamp_min(1e-20)
    return torch.stack((fg / total, bg / total), dim=1)


def region_direction_derivative(source_band, target_band, guide):
    guide = guide.to(source_band.device, source_band.dtype).clamp(0.0, 1.0)
    fg = (guide * source_band.sign() * guide * target_band).flatten(1).sum(1)
    bg_mask = 1.0 - guide
    bg = (bg_mask * source_band.sign() * bg_mask * target_band).flatten(1).sum(1)
    return torch.stack((fg, bg), dim=1)


def patch_energy_map(x, patch_size):
    energy = x.square().sum(1, keepdim=True)
    return F.avg_pool2d(energy, kernel_size=patch_size, stride=patch_size).flatten(1)


def cross_band_patch_correlation(image_patch_bands, grad_patch_bands, patch_mask=None):
    if image_patch_bands.shape != grad_patch_bands.shape:
        raise ValueError("image and gradient patch bands must have matching [B, 8, P] shape.")
    rows = []
    for bi in range(fft_band_count()):
        cols = []
        for bg in range(fft_band_count()):
            x = image_patch_bands[:, bi]
            y = grad_patch_bands[:, bg]
            if patch_mask is not None:
                mask = patch_mask.bool()
                x = x[mask]
                y = y[mask]
            else:
                x = x.reshape(-1)
                y = y.reshape(-1)
            if x.numel() < 2:
                cols.append(torch.tensor(0.0, device=image_patch_bands.device))
            else:
                xv, yv = x - x.mean(), y - y.mean()
                cols.append((xv * yv).mean() / (xv.std(unbiased=False) * yv.std(unbiased=False)).clamp_min(1e-20))
        rows.append(torch.stack(cols))
    return torch.stack(rows)


def factorial_interaction(plain, dim, bg, dim_bg):
    return dim_bg - dim - bg + plain


def ablate_image_band(x, band, eta=0.25, mask=None):
    component = fft_project(x, band)
    if mask is not None:
        component = component * mask.to(x.device, x.dtype)
    return torch.clamp(x - eta * component, 0.0, 1.0)


def report_rule(correlation_positive, ablation_changed, localized):
    if correlation_positive and ablation_changed and localized:
        return "supported"
    if correlation_positive and not ablation_changed:
        return "association_only"
    return "inconclusive"


def _collect_samples(args, source, loader):
    args.max_samples = args.max_samples_requested
    images, labels, indices, sizes = [], [], [], []
    for x, y, idx in selected_batches(args, source, loader):
        images.append(x.cpu())
        labels.append(y.cpu())
        indices.append(idx.cpu())
        sizes.append(x.size(0))
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
        run_analyzed_attack(attacker, clean[start:end], labels[start:end], trace_callback=traces.append, diagnostics=True)
        keep = {row["step"]: row for row in traces if row["step"] in args.trace_steps}
        for step in args.trace_steps:
            row = keep[step]
            rows.append({
                "step": step,
                "x_t": row["x_t"].cpu(),
                "guide_map": row["guide_map"].cpu() if row["guide_map"] is not None else None,
                "diagnostic_gradients": row["diagnostic_gradients"],
            })
        start = end
    payload = {"protocol": PROTOCOL, "seed": seed, "samples_requested": args.max_samples_requested,
               "indices": indices, "labels": labels, "batch_sizes": sizes, "trace_steps": tuple(args.trace_steps), "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def dim_operator_frequency_response(args):
    torch.manual_seed(17)
    height = width = args.img_size
    yy = torch.arange(height, device=DEVICE).view(height, 1)
    xx = torch.arange(width, device=DEVICE).view(1, width)
    rows = []
    for band in range(fft_band_count()):
        vals = []
        radius = (FFT_BANDS[band] + FFT_BANDS[band + 1]) * 0.25
        freq = max(1, int(round(radius * min(height, width))))
        basis = torch.cos(2 * np.pi * freq * (xx / width + yy / height)).view(1, 1, height, width).repeat(1, 3, 1, 1)
        basis = basis / basis.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-20)
        for _ in range(args.operator_samples):
            scale = torch.empty((), device=DEVICE).uniform_(args.dim_resize_min, args.dim_resize_max).item()
            small = max(1, int(round(height * scale)))
            resized = F.interpolate(basis, size=(small, small), mode="bilinear", align_corners=False)
            pad_h, pad_w = height - small, width - small
            top = torch.randint(0, pad_h + 1, ()).item() if pad_h else 0
            left = torch.randint(0, pad_w + 1, ()).item() if pad_w else 0
            padded = F.pad(resized, (left, pad_w - left, top, pad_h - top))
            cropped = padded[..., top:top + small, left:left + small]
            back = F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)
            vals.append((back.square().sum() / basis.square().sum().clamp_min(1e-20)).detach().cpu())
        rows.append(torch.stack(vals).mean())
    return torch.stack(rows).numpy()


def _dim_samples(attacker, pixels, labels, guide, count):
    samples = []
    with _attacker_options(attacker, input_diversity=True, dim_mode="full-random", guide_aug=False):
        for _ in range(count):
            probe = pixels.detach().requires_grad_(True)
            samples.append(attacker._attack_grad(probe, labels).detach())
    return samples


def _append_lists(dst, prefix, values):
    for key, value in values.items():
        dst.setdefault(prefix + key, []).extend(value)


def run_experiment(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "operator_frequency_response.npz", response=dim_operator_frequency_response(args), fft_bands=np.asarray(FFT_BANDS))
    for seed in args.seeds:
        metrics_path = root / "runs" / f"seed_{seed}.json"
        if metrics_path.exists() and not args.force:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if payload.get("protocol") == PROTOCOL and payload.get("samples_requested") == args.max_samples_requested:
                continue
        seed_all(seed)
        loader, num_classes = load_data(args.image_dir, args.annotations_path, args.batch_size, args.num_workers, 2, args.img_size)
        source, attacker = build_baseline(num_classes)
        attacker.guide_aug_methods = ("dropout", "jitter", "freq")
        attacker.guide_aug_copies = 3
        attacker.guide_aug_strength = 0.2
        trace = _collect_trace(args, source, attacker, seed)
        labels = trace["labels"]
        accum = {}
        source_cache = {}
        sample_start = 0
        for batch_idx, offset in enumerate(range(0, len(trace["rows"]), len(args.trace_steps))):
            rows = trace["rows"][offset:offset + len(args.trace_steps)]
            batch_size = trace["batch_sizes"][batch_idx]
            batch_labels = labels[sample_start:sample_start + batch_size].to(DEVICE)
            for row in rows:
                pixels = row["x_t"].to(DEVICE)
                guide = row["guide_map"].to(DEVICE) if row["guide_map"] is not None else torch.ones(pixels.size(0), 1, pixels.size(2), pixels.size(3), device=DEVICE)
                diagnostics = {k: v.to(DEVICE) for k, v in row["diagnostic_gradients"].items() if k in VARIANTS}
                source_cache[(offset, row["step"])] = {k: v.cpu() for k, v in diagnostics.items()}
                dim_samples = _dim_samples(attacker, pixels, batch_labels, guide, args.dim_samples)
                dim_avg = torch.stack(dim_samples).mean(0)
                for name, tensor in (("image_energy_ratio", band_energy_ratios(pixels)), ("gradient_energy_ratio_plain", band_energy_ratios(diagnostics["plain"])), ("gradient_energy_ratio_dim_avg", band_energy_ratios(dim_avg))):
                    for band in range(fft_band_count()):
                        accum.setdefault(f"step{row['step']}_{name}_band{band}", []).extend(tensor[:, band].detach().cpu().tolist())
                for band in range(fft_band_count()):
                    parts = [fft_project(sample, band) for sample in dim_samples]
                    numerator = torch.stack(parts).mean(0).flatten(1).norm(dim=1)
                    denominator = torch.stack([part.flatten(1).norm(dim=1) for part in parts]).mean(0).clamp_min(1e-20)
                    accum.setdefault(f"step{row['step']}_transform_coherence_band{band}", []).extend((numerator / denominator).cpu().tolist())
                    for variant, grad in diagnostics.items():
                        projected = fft_project(grad, band)
                        reg = region_band_energy(projected, guide)
                        for ridx, region in enumerate(REGIONS):
                            accum.setdefault(f"step{row['step']}_{variant}_region_energy_{region}_band{band}", []).extend(reg[:, ridx].detach().cpu().tolist())
                        accum.setdefault(f"step{row['step']}_{variant}_patch_source_band{band}", []).extend(patch_energy_map(projected, args.patch_size).detach().cpu().numpy().tolist())
                    images = torch.stack([patch_energy_map(fft_project(pixels, b), args.patch_size) for b in range(fft_band_count())], dim=1)
                    grads = torch.stack([patch_energy_map(fft_project(diagnostics["plain"], b), args.patch_size) for b in range(fft_band_count())], dim=1)
                    accum.setdefault(f"step{row['step']}_patch_corr", []).append(cross_band_patch_correlation(images, grads).detach().cpu().numpy().tolist())
                    before = band_energy_ratios(diagnostics["plain"])
                    for removed in range(fft_band_count()):
                        ablated = ablate_image_band(pixels, removed, args.ablation_eta)
                        probe = ablated.detach().requires_grad_(True)
                        with _attacker_options(attacker, input_diversity=False, guide_aug=False):
                            after_grad = attacker._attack_grad(probe, batch_labels).detach()
                        delta = band_energy_ratios(after_grad) - before
                        for grad_band in range(fft_band_count()):
                            accum.setdefault(f"step{row['step']}_ablation_delta_removed{removed}_grad{grad_band}", []).extend(delta[:, grad_band].cpu().tolist())
            sample_start += batch_size
        del source, attacker, loader
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
                    pixels = row["x_t"].to(DEVICE).detach().requires_grad_(True)
                    guide = row["guide_map"].to(DEVICE) if row["guide_map"] is not None else torch.ones(pixels.size(0), 1, pixels.size(2), pixels.size(3), device=DEVICE)
                    loss = F.cross_entropy(model(_target_normalize(model, pixels), return_attn=False), batch_labels)
                    target_grad = torch.autograd.grad(loss, pixels)[0].detach()
                    grads = {k: v.to(DEVICE) for k, v in source_cache[(offset, row["step"])].items()}
                    for band in range(fft_band_count()):
                        target_band = fft_project(target_grad, band)
                        values = {}
                        for variant, grad in grads.items():
                            source_band = fft_project(grad, band)
                            reg = region_direction_derivative(source_band, target_band, guide)
                            for ridx, region in enumerate(REGIONS):
                                key = f"step{row['step']}_{model_name}_{variant}_direction_{region}_band{band}"
                                accum.setdefault(key, []).extend(reg[:, ridx].cpu().tolist())
                            values[variant] = reg
                        inter = factorial_interaction(values["plain"], values["dim"], values["bg"], values["dim_bg"])
                        for ridx, region in enumerate(REGIONS):
                            accum.setdefault(f"step{row['step']}_{model_name}_interaction_{region}_band{band}", []).extend(inter[:, ridx].cpu().tolist())
                sample_start += batch_size
            del model
            _release()
        summary = {key: float(np.mean(value)) for key, value in accum.items() if not key.endswith("_patch_corr") and value}
        summary.update({key: np.asarray(value, dtype=np.float64).mean(0).tolist() for key, value in accum.items() if key.endswith("_patch_corr") and value})
        _json(metrics_path, {"protocol": PROTOCOL, "seed": seed, "samples_requested": args.max_samples_requested,
                             "samples": int(len(trace["indices"])), "indices": trace["indices"].tolist(),
                             "trace_steps": list(args.trace_steps), "target_models": list(args.target_models),
                             "fft_bands": list(FFT_BANDS), "metrics": summary})
        print(f"wrote {metrics_path}")


def _mean_metric(runs, key):
    values = [run["metrics"][key] for run in runs if key in run["metrics"]]
    return None if not values else float(np.mean(values))


def _mean_array(runs, key):
    values = [np.asarray(run["metrics"][key], dtype=np.float64) for run in runs if key in run["metrics"]]
    return None if not values else np.mean(values, axis=0)


def _sum_bands(runs, template, step, bands):
    values = [_mean_metric(runs, template.format(step=step, band=band)) for band in bands]
    values = [v for v in values if v is not None]
    return None if not values else float(np.sum(values))


def build_report(runs, operator_response):
    first = runs[0]
    report = {"protocol": PROTOCOL, "fft_bands": list(FFT_BANDS), "seeds": [run["seed"] for run in runs],
              "runs": runs, "operator_frequency_response": operator_response.tolist(), "summary": {}, "conclusions": {}}
    target_models = first["target_models"]
    for step in first["trace_steps"]:
        low_mid = BAND_GROUPS["low"] + BAND_GROUPS["mid"]
        fg_energy = np.mean([_mean_metric(runs, f"step{step}_plain_region_energy_fg_band{b}") or 0.0 for b in low_mid])
        bg_energy = np.mean([_mean_metric(runs, f"step{step}_plain_region_energy_bg_band{b}") or 0.0 for b in low_mid])
        bg_positive_models = 0
        for model in target_models:
            bg_delta = np.mean([_mean_metric(runs, f"step{step}_{model}_plain_direction_bg_band{b}") or 0.0 for b in low_mid])
            fg_delta = np.mean([_mean_metric(runs, f"step{step}_{model}_plain_direction_fg_band{b}") or 0.0 for b in low_mid])
            bg_positive_models += int(bg_delta > fg_delta)
        dim_energy_gain = (_sum_bands(runs, "step{step}_gradient_energy_ratio_dim_avg_band{band}", step, low_mid) or 0.0) - (_sum_bands(runs, "step{step}_gradient_energy_ratio_plain_band{band}", step, low_mid) or 0.0)
        coherence_low_mid = np.mean([_mean_metric(runs, f"step{step}_transform_coherence_band{b}") or 0.0 for b in low_mid])
        coherence_high = np.mean([_mean_metric(runs, f"step{step}_transform_coherence_band{b}") or 0.0 for b in BAND_GROUPS["high"]])
        interaction_positive = 0
        for model in target_models:
            val = np.mean([_mean_metric(runs, f"step{step}_{model}_interaction_bg_band{b}") or 0.0 for b in low_mid])
            interaction_positive += int(val > 0)
        corr = np.asarray(_mean_metric(runs, f"step{step}_patch_corr") or np.zeros((8, 8)))
        corr_low_mid = float(np.mean(corr[np.ix_(low_mid, low_mid)])) if corr.shape == (8, 8) else 0.0
        ablation_change = np.mean([abs(_mean_metric(runs, f"step{step}_ablation_delta_removed{bi}_grad{bg}") or 0.0) for bi in low_mid for bg in low_mid])
        report["summary"][str(step)] = {
            "low_mid_region_energy_fg": float(fg_energy),
            "low_mid_region_energy_bg": float(bg_energy),
            "bg_direction_beats_fg_models": int(bg_positive_models),
            "dim_low_mid_energy_gain": float(dim_energy_gain),
            "dim_low_mid_transform_coherence": float(coherence_low_mid),
            "dim_high_transform_coherence": float(coherence_high),
            "dim_bg_interaction_bg_low_mid_positive_models": int(interaction_positive),
            "image_gradient_low_mid_patch_corr": corr_low_mid,
            "image_low_mid_ablation_grad_low_mid_abs_delta": float(ablation_change),
        }
    last_step = str(first["trace_steps"][-1])
    item = report["summary"][last_step]
    if item["low_mid_region_energy_bg"] > item["low_mid_region_energy_fg"] and item["bg_direction_beats_fg_models"] >= min(6, len(target_models)):
        origin = "background_supported"
    elif item["low_mid_region_energy_fg"] > item["low_mid_region_energy_bg"]:
        origin = "foreground_supported"
    else:
        origin = "evidence_mixed_or_inconclusive"
    dim_supported = item["dim_low_mid_energy_gain"] > 0 and item["dim_low_mid_transform_coherence"] > item["dim_high_transform_coherence"]
    if dim_supported:
        dim_text = "DIM provides transfer-relevant low/mid input-gradient evidence when target alignment is positive; otherwise spectrum-only evidence is inconclusive."
    else:
        dim_text = "DIM changes input-gradient spectrum, but transfer-relevant evidence is inconclusive."
    link = report_rule(item["image_gradient_low_mid_patch_corr"] > 0, item["image_low_mid_ablation_grad_low_mid_abs_delta"] > 1e-6, origin != "evidence_mixed_or_inconclusive")
    report["conclusions"] = {
        "low_mid_gradient_origin": origin,
        "dim_low_mid_evidence": dim_text,
        "dim_bg_positive_interaction": item["dim_bg_interaction_bg_low_mid_positive_models"] >= min(6, len(target_models)),
        "image_gradient_frequency_link": link,
    }
    return report


def build_conclusion_zh(report):
    c = report["conclusions"]
    return "\n".join([
        "# DIM/BG 图像-梯度频率机制结论",
        "",
        f"1. low/mid 输入梯度区域来源: {c['low_mid_gradient_origin']}。background 表示 low-attention 区域 Q=1-M，不是语义分割背景。",
        f"2. DIM 证据: {c['dim_low_mid_evidence']} 报告同时区分 input-gradient energy increase 和 target direction derivative。",
        f"3. DIM only / BG only / DIM+BG: background low/mid 交互项是否为正: {c['dim_bg_positive_interaction']}。单独模块若只增加能量而没有黑盒方向导数增益，结论为频谱改变但迁移相关证据不足。",
        f"4. 输入图像频率与输入梯度频率联系: {c['image_gradient_frequency_link']}。correlation 只说明关联；只有 ablation 同时改变梯度才作为因果证据。",
        "5. 所有图像频率 FFT(x_t) 与输入梯度频率 FFT(dL/dx_t) 使用完全相同的 8 个径向 band 边界。",
        "",
    ]) + "\n"


def run_report(args):
    root = Path(args.output_dir)
    runs = []
    for seed in args.seeds:
        path = root / "runs" / f"seed_{seed}.json"
        if not path.exists():
            raise RuntimeError(f"Missing run metrics: {path}")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    operator = np.load(root / "operator_frequency_response.npz")["response"]
    report = build_report(runs, operator)
    _json(root / "dim_bg_image_gradient_frequency_report.json", report)
    arrays = {"operator_response": operator, "fft_bands": np.asarray(FFT_BANDS)}
    for run in runs:
        for key, value in run["metrics"].items():
            arrays[f"seed{run['seed']}_{key}"] = np.asarray(value)
    np.savez(root / "region_band_gradient_metrics.npz", **{k: v for k, v in arrays.items() if "region_energy" in k or "direction" in k or k in ("fft_bands",)})
    np.savez(root / "image_gradient_frequency_link_metrics.npz", **{k: v for k, v in arrays.items() if "patch_corr" in k or "ablation_delta" in k or "image_energy" in k or k in ("fft_bands",)})
    np.savez(root / "factorial_interaction_metrics.npz", **{k: v for k, v in arrays.items() if "interaction" in k or k in ("fft_bands",)})
    (root / "dim_bg_image_gradient_frequency_conclusion_zh.md").write_text(build_conclusion_zh(report), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "experiment", "report"))
    parser.add_argument("--output-dir", default="outputs/dim_bg_image_gradient_frequency_math")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", dest="max_samples_requested", type=int, default=100)
    parser.add_argument("--seeds", type=lambda x: tuple(map(int, x.split(","))), default=(0, 1))
    parser.add_argument("--trace-steps", type=lambda x: tuple(map(int, x.split(","))), default=TRACE_STEPS)
    parser.add_argument("--dim-samples", type=int, default=8)
    parser.add_argument("--operator-samples", type=int, default=32)
    parser.add_argument("--dim-resize-min", type=float, default=0.85)
    parser.add_argument("--dim-resize-max", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--ablation-eta", type=float, default=0.25)
    parser.add_argument("--target-models", type=parse_model_names, default=MAIN_TARGETS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.dim_samples < 1:
        raise ValueError("--dim-samples must be positive.")
    if parsed.mode in ("all", "experiment"):
        run_experiment(parsed)
    if parsed.mode in ("all", "report"):
        run_report(parsed)
