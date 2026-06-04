"""Quick causal DIM/background mechanism experiment with resumable JSON artifacts."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from causal_analysis import MAIN_TARGETS, _target_normalize, bh_fdr, build_baseline, seed_all, selected_batches
from gradient_analysis import fft_project, run_analyzed_attack
from main import ANNOTATIONS_PATH, IMAGE_DIR, parse_model_names
from nets import build_vit_model
from utils import DEVICE, load_data

DIM_VARIANTS = ("none", "full-random", "forward-only", "backward-only", "full-fixed", "backward-fixed")
AREAS = ("background", "all")
AUGMENTATION_METHODS = (
    "dropout", "jitter", "freq", "lowpass_gauss", "laplacian_low", "fft_lowboost", "illumination_low",
    "band_noise", "band_noise_low", "band_noise_mid", "band_noise_high",
    "colored_noise", "colored_noise_low", "colored_noise_mid", "colored_noise_high",
    "progressive_spectral_noise", "progressive_spectral_noise_low", "progressive_spectral_noise_mid", "progressive_spectral_noise_high",
    "wavelet_noise", "wavelet_noise_low", "wavelet_noise_mid", "wavelet_noise_high", "wavelet_noise_fglow_bghigh", "white_noise",
)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _loader(args):
    return load_data(image_dir_arg=args.image_dir, annotations_path_arg=args.annotations_path, batch_size=args.batch_size,
                     num_workers=args.num_workers, prefetch_factor=2, img_size=args.img_size)


def _release(*objects):
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _band_metrics(x):
    energy = x.square().flatten(1).sum(1).clamp_min(1e-20)
    low = sum(fft_project(x, band) for band in range(3))
    mid = sum(fft_project(x, band) for band in range(3, 6))
    high = sum(fft_project(x, band) for band in range(6, 8))
    return {name: value.square().flatten(1).sum(1).div(energy).detach().cpu() for name, value in (("low", low), ("mid", mid), ("high", high))}


def _collect_samples(args, source, loader):
    args.max_samples = args.max_samples_requested
    images, labels, indices, sizes = [], [], [], []
    for x, y, idx in selected_batches(args, source, loader):
        images.append(x.cpu()); labels.append(y.cpu()); indices.append(idx.cpu()); sizes.append(x.size(0))
    return torch.cat(images), torch.cat(labels), torch.cat(indices), sizes


def _configure(attacker, area, variant):
    attacker.guide_aug_area = area
    attacker.input_diversity = variant != "none"
    if variant != "none": attacker.dim_mode = variant
    attacker.reset_fixed_dim()


def _source_run(attacker, clean, labels, sizes, trace_steps):
    adv, band_rows, coherence, input_hf = [], {key: [] for key in ("low", "mid", "high")}, [], []
    start = 0
    for size in sizes:
        end = start + size
        x, y = clean[start:end].to(DEVICE), labels[start:end].to(DEVICE)
        pixels = attacker._denormalize(x).detach()
        guide = attacker._build_guide_pixel_map(x, pixels.size(-1)) if attacker.guide_aug_area != "all" else None
        samples = []
        for _ in range(3):
            probe = pixels.detach().requires_grad_(True)
            samples.append(attacker._attack_grad(probe, y, guide).detach())
        mean = torch.stack(samples).mean(0)
        coherence.extend(torch.stack([F.cosine_similarity(row.flatten(1), mean.flatten(1)) for row in samples]).mean(0).cpu().tolist())
        for key, values in _band_metrics(mean).items(): band_rows[key].extend(values.tolist())
        transformed = attacker._input_diversity(attacker._normalize(pixels)) if attacker.input_diversity else attacker._normalize(pixels)
        delta = transformed - attacker._normalize(pixels)
        input_hf.extend(_band_metrics(delta)["high"].tolist() if delta.abs().sum() else [0.0] * size)
        adv.append(run_analyzed_attack(attacker, x, y).cpu())
        start = end
    return torch.cat(adv), {
        "gradient_consistency": float(np.mean(coherence)), "input_high_frequency_ratio": float(np.mean(input_hf)),
        "gradient_band_energy": {key: float(np.mean(value)) for key, value in band_rows.items()},
    }


def _target_metrics(args, clean, adv, labels):
    clean_pixels, adv_pixels = clean.to(DEVICE) * .5 + .5, adv.to(DEVICE) * .5 + .5
    direction = (adv_pixels - clean_pixels).sign()
    model_rows = {}
    for name in args.target_models:
        model = build_vit_model(num_classes=1000, model_name=name)
        valid_rows, success_rows, derivatives = [], [], []
        for start in range(0, len(labels), args.eval_batch_size):
            end = start + args.eval_batch_size; y = labels[start:end].to(DEVICE)
            cp, ap, update = clean_pixels[start:end], adv_pixels[start:end], direction[start:end]
            with torch.inference_mode():
                valid_rows.append(model(_target_normalize(model, cp), return_attn=False).argmax(1).eq(y).cpu())
                success_rows.append(model(_target_normalize(model, ap), return_attn=False).argmax(1).ne(y).cpu())
            probe = ap.detach().requires_grad_(True)
            loss = F.cross_entropy(model(_target_normalize(model, probe), return_attn=False), y)
            grad = torch.autograd.grad(loss, probe)[0]
            derivatives.extend((update * grad).flatten(1).sum(1).detach().cpu().tolist())
        valid, success = torch.cat(valid_rows), torch.cat(success_rows)
        model_rows[name] = {"asr": float(success[valid].float().mean()) if valid.any() else None,
                            "direction_derivative": float(np.mean(np.asarray(derivatives)[valid.numpy()])) if valid.any() else None,
                            "clean_correct": int(valid.sum())}
        del model
        _release()
    return model_rows


def run_experiment(args):
    output = Path(args.output_dir) / "runs"
    for area in args.areas:
        for variant in args.variants:
            for seed in args.seeds:
                path = output / f"{area}__{variant}__seed_{seed}.json"
                if path.exists() and not args.force: continue
                seed_all(seed); loader, num_classes = _loader(args); source, attacker = build_baseline(num_classes)
                _configure(attacker, area, variant)
                clean, labels, indices, sizes = _collect_samples(args, source, loader)
                adv, source_metrics = _source_run(attacker, clean, labels, sizes, args.trace_steps)
                del attacker, source, loader
                _release()
                targets = _target_metrics(args, clean, adv, labels)
                _json(path, {"protocol": "quick_protocol", "area": area, "dim_variant": variant, "seed": seed,
                             "indices": indices.tolist(), "source_metrics": source_metrics, "targets": targets})
                print(f"wrote {path}")


def run_ranking(args):
    path = Path(args.output_dir) / "method_high_frequency_ranking.json"
    if path.exists() and not args.force: return
    seed_all(args.seeds[0]); loader, num_classes = _loader(args); source, attacker = build_baseline(num_classes)
    clean, _labels, _indices, _sizes = _collect_samples(args, source, loader); pixels = attacker._denormalize(clean.to(DEVICE))
    guide = attacker._build_guide_pixel_map(clean.to(DEVICE), pixels.size(-1)); rows, norms = [], []
    for method in AUGMENTATION_METHODS:
        delta = (attacker._guide_augmented_pixels(pixels, guide, method) - pixels).detach()
        norms.append(delta.flatten(1).norm(dim=1).cpu())
        rows.append({"method": method, "high_frequency_ratio": float(_band_metrics(delta)["high"].mean())})
        del delta
    matched_l2 = float(torch.stack(norms).median(0).values.mean())
    for row in rows: row["matched_l2"] = matched_l2
    rows.sort(key=lambda row: row["high_frequency_ratio"], reverse=True)
    _json(path, {"protocol": "quick_protocol", "l2_matching": "per-image median augmentation L2", "ranking": rows})
    del attacker, source, loader
    _release()


def run_report(args):
    root = Path(args.output_dir); rows = []
    for path in sorted((root / "runs").glob("*.json")):
        payload = json.loads(path.read_text()); targets = payload["targets"]
        rows.append({"area": payload["area"], "dim_variant": payload["dim_variant"], "seed": payload["seed"],
                     **payload["source_metrics"], "mean_asr": float(np.mean([x["asr"] for x in targets.values()])),
                     "mean_direction_derivative": float(np.mean([x["direction_derivative"] for x in targets.values()])),
                     "target_asr": {key: value["asr"] for key, value in targets.items()}})
    grouped = {}
    for row in rows:
        key = f'{row["area"]}/{row["dim_variant"]}'; grouped.setdefault(key, []).append(row)
    summary = {key: {"seeds": [x["seed"] for x in values],
                     "mean_asr": float(np.mean([x["mean_asr"] for x in values])),
                     "mean_direction_derivative": float(np.mean([x["mean_direction_derivative"] for x in values])),
                     "gradient_consistency": float(np.mean([x["gradient_consistency"] for x in values])),
                     "input_high_frequency_ratio": float(np.mean([x["input_high_frequency_ratio"] for x in values]))}
               for key, values in grouped.items()}
    _json(root / "dim_bg_mechanism_report.json", {"protocol": "quick_protocol", "seeds": list(args.seeds), "runs": rows, "summary": summary})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("mode", choices=("all", "rank", "experiment", "report"))
    p.add_argument("--output-dir", default="outputs/dim_bg_mechanism_quick"); p.add_argument("--image-dir", default=IMAGE_DIR); p.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    p.add_argument("--img-size", type=int, default=224); p.add_argument("--batch-size", type=int, default=4); p.add_argument("--eval-batch-size", type=int, default=32); p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", dest="max_samples_requested", type=int, default=100); p.add_argument("--seeds", type=lambda x: tuple(map(int, x.split(','))), default=(0, 1))
    p.add_argument("--trace-steps", type=lambda x: tuple(map(int, x.split(','))), default=(1, 10, 20, 40)); p.add_argument("--target-models", type=parse_model_names, default=MAIN_TARGETS)
    p.add_argument("--areas", type=parse_model_names, default=AREAS); p.add_argument("--variants", type=parse_model_names, default=DIM_VARIANTS); p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode in ("all", "rank"): run_ranking(args)
    if args.mode in ("all", "experiment"): run_experiment(args)
    if args.mode in ("all", "report"): run_report(args)
