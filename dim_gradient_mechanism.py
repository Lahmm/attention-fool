"""DIM source-gradient mechanism experiment with resumable artifacts."""
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

VARIANTS = ("plain", "dim_random_average", "dim_fixed", "forward_only")
TRACE_STEPS = (1, 10, 20, 40)
REGIONS = {"low": (0, 1, 2), "mid": (3, 4, 5), "high": (6, 7)}
PROTOCOL = "dim_gradient_mechanism_v1"


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


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


def _metric_template(target_models):
    return {
        "source_energy_ratio": {variant: [] for variant in VARIANTS},
        "target_cosine": {variant: {model: [] for model in target_models} for variant in VARIANTS},
        "target_direction_derivative": {variant: {model: [] for model in target_models} for variant in VARIANTS},
        "transform_coherence": {"dim_random_average": []},
    }


def _band_energy_ratios(grad):
    total = grad.square().flatten(1).sum(1).clamp_min(1e-20)
    return torch.stack([
        fft_project(grad, band).square().flatten(1).sum(1) / total
        for band in range(len(FFT_BANDS) - 1)
    ], dim=1)


def _safe_cosine(a, b):
    denom = a.flatten(1).norm(dim=1) * b.flatten(1).norm(dim=1)
    return (a * b).flatten(1).sum(1) / denom.clamp_min(1e-20)


def _direction_derivative(source_grad, target_grad):
    return (source_grad.sign() * target_grad).flatten(1).sum(1)


def _dim_gradient_samples(attacker, pixels, labels, count):
    samples = []
    with _attacker_options(attacker, input_diversity=True, dim_mode="full-random", guide_aug=False):
        for _ in range(count):
            probe = pixels.detach().requires_grad_(True)
            samples.append(attacker._attack_grad(probe, labels, None).detach())
    return samples


def compute_source_gradient_variants(attacker, pixels, labels, dim_samples=8):
    """Return plain, random-averaged DIM, fixed DIM, and forward-only gradients."""
    result = {}
    with _attacker_options(attacker, input_diversity=False, guide_aug=False):
        probe = pixels.detach().requires_grad_(True)
        result["plain"] = attacker._attack_grad(probe, labels, None).detach()
    samples = _dim_gradient_samples(attacker, pixels, labels, dim_samples)
    result["dim_random_average"] = torch.stack(samples).mean(0)
    attacker.reset_fixed_dim()
    with _attacker_options(attacker, input_diversity=True, dim_mode="full-fixed", guide_aug=False):
        probe = pixels.detach().requires_grad_(True)
        result["dim_fixed"] = attacker._attack_grad(probe, labels, None).detach()
    with _attacker_options(attacker, input_diversity=True, dim_mode="forward-only", guide_aug=False):
        probe = pixels.detach().requires_grad_(True)
        result["forward_only"] = attacker._attack_grad(probe, labels, None).detach()
    return result, samples


def fft_energy_ratio_sum(grad):
    return _band_energy_ratios(grad).sum(1)


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
    trace_path = Path(args.output_dir) / "traces" / f"seed_{seed}.pt"
    if trace_path.exists() and not args.force:
        payload = torch.load(trace_path, map_location="cpu")
        if payload.get("protocol") == PROTOCOL and payload.get("samples_requested") == args.max_samples_requested:
            return payload
    seed_all(seed)
    loader, _num_classes = load_data(
        image_dir_arg=args.image_dir,
        annotations_path_arg=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        img_size=args.img_size,
    )
    clean, labels, indices, sizes = _collect_samples(args, source, loader)
    trace_rows = []
    start = 0
    for size in sizes:
        end = start + size
        traces = []
        run_analyzed_attack(attacker, clean[start:end], labels[start:end], trace_callback=traces.append)
        keep = {row["step"]: row for row in traces if row["step"] in args.trace_steps}
        for step in args.trace_steps:
            trace_rows.append({
                "step": step,
                "x_t": keep[step]["x_t"].cpu(),
                "guide_map": None if keep[step]["guide_map"] is None else keep[step]["guide_map"].cpu(),
            })
        start = end
    payload = {
        "protocol": PROTOCOL,
        "seed": seed,
        "samples_requested": args.max_samples_requested,
        "indices": indices,
        "labels": labels,
        "batch_sizes": sizes,
        "trace_steps": tuple(args.trace_steps),
        "rows": trace_rows,
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, trace_path)
    return payload


def _init_seed_accumulators(target_models, trace_steps=TRACE_STEPS):
    return {step: {band: _metric_template(target_models) for band in range(8)} for step in trace_steps}


def _append_source_metrics(acc, step, gradients, dim_samples):
    sample_bands = [[fft_project(sample, band) for sample in dim_samples] for band in range(8)]
    for variant, grad in gradients.items():
        ratios = _band_energy_ratios(grad).cpu()
        for band in range(8):
            acc[step][band]["source_energy_ratio"][variant].extend(ratios[:, band].tolist())
    for band in range(8):
        parts = sample_bands[band]
        mean = torch.stack(parts).mean(0)
        numerator = mean.flatten(1).norm(dim=1)
        denominator = torch.stack([part.flatten(1).norm(dim=1) for part in parts]).mean(0).clamp_min(1e-20)
        acc[step][band]["transform_coherence"]["dim_random_average"].extend((numerator / denominator).cpu().tolist())


def _append_target_metrics(acc, step, gradients, target_grad, model_name):
    for band in range(8):
        target_band = fft_project(target_grad, band)
        for variant, grad in gradients.items():
            source_band = fft_project(grad, band)
            acc[step][band]["target_cosine"][variant][model_name].extend(_safe_cosine(source_band, target_band).detach().cpu().tolist())
            acc[step][band]["target_direction_derivative"][variant][model_name].extend(
                _direction_derivative(source_band, target_band).detach().cpu().tolist()
            )


def _average_nested(values):
    if isinstance(values, list):
        return float(np.mean(values)) if values else None
    return {key: _average_nested(value) for key, value in values.items()}


def run_experiment(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        metrics_path = root / "runs" / f"seed_{seed}.json"
        if metrics_path.exists() and not args.force:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if payload.get("protocol") == PROTOCOL and payload.get("samples_requested") == args.max_samples_requested:
                continue
        seed_all(seed)
        loader, num_classes = load_data(
            image_dir_arg=args.image_dir,
            annotations_path_arg=args.annotations_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=2,
            img_size=args.img_size,
        )
        source, attacker = build_baseline(num_classes)
        trace = _collect_trace(args, source, attacker, seed)
        labels = trace["labels"]
        acc = _init_seed_accumulators(args.target_models, args.trace_steps)
        source_cache = {}
        sample_start = 0
        for batch_idx, offset in enumerate(range(0, len(trace["rows"]), len(args.trace_steps))):
            rows = trace["rows"][offset:offset + len(args.trace_steps)]
            batch_size = trace.get("batch_sizes", [args.batch_size] * len(trace["rows"]))[batch_idx]
            batch_labels = labels[sample_start:sample_start + batch_size].to(DEVICE)
            for row in rows:
                pixels = row["x_t"].to(DEVICE)
                gradients, dim_samples = compute_source_gradient_variants(attacker, pixels, batch_labels, args.dim_samples)
                source_cache[(offset, row["step"])] = {name: grad.cpu() for name, grad in gradients.items()}
                _append_source_metrics(acc, row["step"], gradients, dim_samples)
            sample_start += batch_size
        del source, attacker, loader
        _release()
        for model_name in args.target_models:
            model = build_vit_model(num_classes=1000, model_name=model_name)
            sample_start = 0
            for batch_idx, offset in enumerate(range(0, len(trace["rows"]), len(args.trace_steps))):
                rows = trace["rows"][offset:offset + len(args.trace_steps)]
                batch_size = trace.get("batch_sizes", [args.batch_size] * len(trace["rows"]))[batch_idx]
                batch_labels = labels[sample_start:sample_start + batch_size].to(DEVICE)
                for row in rows:
                    pixels = row["x_t"].to(DEVICE).detach().requires_grad_(True)
                    loss = F.cross_entropy(model(_target_normalize(model, pixels), return_attn=False), batch_labels)
                    target_grad = torch.autograd.grad(loss, pixels)[0].detach()
                    gradients = {name: grad.to(DEVICE) for name, grad in source_cache[(offset, row["step"])].items()}
                    _append_target_metrics(acc, row["step"], gradients, target_grad, model_name)
                sample_start += batch_size
            del model
            _release()
        averaged = {str(step): {str(band): _average_nested(acc[step][band]) for band in range(8)} for step in args.trace_steps}
        _json(metrics_path, {
            "protocol": PROTOCOL,
            "seed": seed,
            "samples_requested": args.max_samples_requested,
            "samples": int(len(trace["indices"])),
            "indices": trace["indices"].tolist(),
            "trace_steps": list(args.trace_steps),
            "dim_samples": args.dim_samples,
            "target_models": list(args.target_models),
            "metrics": averaged,
        })
        print(f"wrote {metrics_path}")


def _delta(value, plain):
    return None if value is None or plain is None else float(value - plain)


def build_report(seed_payloads):
    report = {"protocol": PROTOCOL, "seeds": [row["seed"] for row in seed_payloads], "runs": seed_payloads, "summary": {}}
    for step in seed_payloads[0]["trace_steps"]:
        step_key = str(step)
        report["summary"][step_key] = {}
        for band in range(8):
            band_key = str(band)
            report["summary"][step_key][band_key] = {}
            for variant in VARIANTS:
                if variant == "plain":
                    continue
                energy_delta = [
                    _delta(row["metrics"][step_key][band_key]["source_energy_ratio"][variant],
                           row["metrics"][step_key][band_key]["source_energy_ratio"]["plain"])
                    for row in seed_payloads
                ]
                target_delta = {}
                positive_count = {}
                for model in seed_payloads[0]["target_models"]:
                    values = [
                        _delta(row["metrics"][step_key][band_key]["target_direction_derivative"][variant][model],
                               row["metrics"][step_key][band_key]["target_direction_derivative"]["plain"][model])
                        for row in seed_payloads
                    ]
                    target_delta[model] = float(np.mean(values))
                    positive_count[model] = all(value > 0 for value in values)
                report["summary"][step_key][band_key][variant] = {
                    "delta_energy_vs_plain_by_seed": energy_delta,
                    "delta_energy_vs_plain": float(np.mean(energy_delta)),
                    "delta_target_direction_derivative_vs_plain_by_model": target_delta,
                    "delta_target_direction_derivative_vs_plain": float(np.mean(list(target_delta.values()))),
                    "model_positive_count": int(sum(positive_count.values())),
                }
    return report


def _same_positive(values):
    return all(value > 0 for value in values)


def _same_negative(values):
    return all(value < 0 for value in values)


def classify_band(report, step, band, variant="dim_random_average", min_models=6):
    item = report["summary"][str(step)][str(band)][variant]
    classes = []
    if _same_positive(item["delta_energy_vs_plain_by_seed"]):
        classes.append("enhanced")
    if _same_negative(item["delta_energy_vs_plain_by_seed"]):
        classes.append("suppressed")
    model_values = item["delta_target_direction_derivative_vs_plain_by_model"].values()
    required_models = min(min_models, len(item["delta_target_direction_derivative_vs_plain_by_model"]))
    if float(np.mean(list(model_values))) > 0 and item["model_positive_count"] >= required_models:
        classes.append("transfer_improved")
    return classes or ["inconclusive"]


def build_conclusion(report):
    lines = ["# DIM Gradient Mechanism Conclusion", ""]
    any_strong = False
    for step in report["runs"][0]["trace_steps"]:
        enhanced, suppressed, improved = [], [], []
        for band in range(8):
            classes = classify_band(report, step, band)
            if "enhanced" in classes:
                enhanced.append(band)
            if "suppressed" in classes:
                suppressed.append(band)
            if "transfer_improved" in classes:
                improved.append(band)
        lines.append(f"## step {step}")
        lines.append(f"- enhanced bands: {enhanced or 'evidence inconclusive'}")
        lines.append(f"- suppressed bands: {suppressed or 'evidence inconclusive'}")
        lines.append(f"- transfer-improved bands: {improved or 'evidence inconclusive'}")
        any_strong = any_strong or bool(enhanced or suppressed or improved)
    lines.append("")
    lines.append("Overall: " + ("mechanism evidence present under the fixed rules." if any_strong else "evidence inconclusive"))
    return "\n".join(lines) + "\n"


def run_report(args):
    root = Path(args.output_dir)
    payloads = []
    for seed in args.seeds:
        path = root / "runs" / f"seed_{seed}.json"
        if not path.exists():
            raise RuntimeError(f"Missing run metrics: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    report = build_report(payloads)
    _json(root / "dim_gradient_mechanism_report.json", report)
    arrays = {}
    for run in payloads:
        for step in run["trace_steps"]:
            for band in range(8):
                base = run["metrics"][str(step)][str(band)]
                for metric in ("source_energy_ratio", "target_cosine", "target_direction_derivative"):
                    arrays[f"seed{run['seed']}_step{step}_band{band}_{metric}"] = np.asarray(json.dumps(base[metric]))
    np.savez(root / "gradient_mechanism_metrics.npz", **arrays)
    (root / "dim_gradient_mechanism_conclusion.md").write_text(build_conclusion(report), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("all", "experiment", "report"))
    p.add_argument("--output-dir", default="outputs/dim_gradient_mechanism_quick")
    p.add_argument("--image-dir", default=IMAGE_DIR)
    p.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", dest="max_samples_requested", type=int, default=100)
    p.add_argument("--seeds", type=lambda x: tuple(map(int, x.split(","))), default=(0, 1))
    p.add_argument("--trace-steps", type=lambda x: tuple(map(int, x.split(","))), default=TRACE_STEPS)
    p.add_argument("--dim-samples", type=int, default=8)
    p.add_argument("--target-models", type=parse_model_names, default=MAIN_TARGETS)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.dim_samples < 1:
        raise ValueError("--dim-samples must be positive.")
    if parsed.mode in ("all", "experiment"):
        run_experiment(parsed)
    if parsed.mode in ("all", "report"):
        run_report(parsed)
