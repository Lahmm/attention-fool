"""Discover and causally confirm transferable spatial-frequency gradient components."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from causal_analysis import MAIN_TARGETS, _target_normalize, bh_fdr, build_baseline, seed_all, selected_batches
from gradient_analysis import (
    FFT_BANDS,
    SCREENING_FFT_ORIENTATIONS,
    component_transform,
    fft_project,
    haar_packet_coefficients,
    haar_packet_paths,
    parse_component,
    run_analyzed_attack,
    screening_component_specs,
)
from main import ANNOTATIONS_PATH, IMAGE_DIR, parse_model_names
from nets import build_vit_model
from utils import DEVICE, load_data

SEEDS = (0, 1, 2)
TRACE_STEPS = (1, 5, 10, 20, 40)
DISCOVERY_SAMPLES = 30
TOTAL_SAMPLES = 100


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _slug(spec):
    return spec.replace(":", "_")


def _loader(args):
    return load_data(
        image_dir_arg=args.image_dir,
        annotations_path_arg=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        img_size=args.img_size,
    )


def _source_traces(args, source, attacker, loader, seed, limit):
    seed_all(seed)
    traces, sample_indices, labels, clean = [], [], [], []
    # Attack complete baseline batches so the retained discovery images consume
    # exactly the same random augmentation sequence as Full/keep/drop runs.
    args.max_samples = TOTAL_SAMPLES
    offset = 0
    for images, batch_labels, indices in selected_batches(args, source, loader):
        selected = []
        keep = min(images.size(0), limit - offset)

        def callback(trace):
            if trace["step"] in TRACE_STEPS:
                selected.append({
                    "step": trace["step"], "x_t": trace["x_t"][:keep], "gradient": trace["gradient"][:keep],
                    "labels": batch_labels[:keep].cpu(), "start": offset, "end": offset + keep,
                })

        run_analyzed_attack(attacker, images, batch_labels, trace_callback=callback)
        traces.extend(selected)
        sample_indices.append(indices[:keep].cpu())
        labels.append(batch_labels[:keep].cpu())
        clean.append(images[:keep].cpu())
        offset += keep
        if offset >= limit:
            break
    return {
        "traces": traces,
        "indices": torch.cat(sample_indices),
        "labels": torch.cat(labels),
        "clean": torch.cat(clean),
    }


def _component_measurements(source_grad, target_grad):
    """Return [batch, 1048] derivative, normalized derivative, and source-energy ratio."""
    target_energy = target_grad.square().flatten(1).sum(1).clamp_min(1e-20)
    source_energy = source_grad.square().flatten(1).sum(1).clamp_min(1e-20)
    derivative, normalized, ratio = [], [], []
    for band in range(len(FFT_BANDS) - 1):
        for orientation in SCREENING_FFT_ORIENTATIONS:
            component = fft_project(source_grad, band, orientation)
            energy = component.square().flatten(1).sum(1)
            value = (component * target_grad).flatten(1).sum(1)
            derivative.append(value)
            normalized.append(value / (energy.clamp_min(1e-20) * target_energy).sqrt())
            ratio.append(energy / source_energy)
    source_coeff = haar_packet_coefficients(source_grad)
    target_coeff = haar_packet_coefficients(target_grad)
    for path in haar_packet_paths():
        src = source_coeff[path]
        tgt = target_coeff[path]
        b, c, h, w = src.shape
        if h % 4 or w % 4:
            raise ValueError(f"Level-3 coefficient shape {(h, w)} is not divisible by 4.")
        src = src.reshape(b, c, 4, h // 4, 4, w // 4)
        tgt = tgt.reshape(b, c, 4, h // 4, 4, w // 4)
        value = (src * tgt).sum(dim=(1, 3, 5)).flatten(1)
        energy = src.square().sum(dim=(1, 3, 5)).flatten(1)
        derivative.extend(value.unbind(1))
        normalized.extend((value / (energy.clamp_min(1e-20) * target_energy[:, None]).sqrt()).unbind(1))
        ratio.extend((energy / source_energy[:, None]).unbind(1))
    return *(torch.stack(values, 1).detach().cpu().numpy() for values in (derivative, normalized, ratio)),


def _target_gradient(model, pixels, labels):
    pixels = pixels.to(DEVICE).requires_grad_(True)
    loss = F.cross_entropy(model(_target_normalize(model, pixels), return_attn=False), labels.to(DEVICE))
    return torch.autograd.grad(loss, pixels)[0].detach()


def _bootstrap_positive(values, repeats, seed):
    """Stratified bootstrap over image, seed, and model axes, ignoring invalid cells."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("values must have shape [seed, image, model].")
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        ss = rng.integers(0, values.shape[0], values.shape[0])
        ii = rng.integers(0, values.shape[1], values.shape[1])
        mm = rng.integers(0, values.shape[2], values.shape[2])
        draws[index] = np.nanmean(values[np.ix_(ss, ii, mm)])
    return {
        "mean": float(np.nanmean(values)),
        "ci_low": float(np.nanquantile(draws, 0.025)),
        "ci_high": float(np.nanquantile(draws, 0.975)),
        "p_positive": float(np.mean(draws <= 0)),
    }


def _summarize_screening(specs, derivative, normalized, energy, repeats, seed):
    rows = []
    for candidate, spec in enumerate(specs):
        values = derivative[candidate]
        norm_values = normalized[candidate]
        seed_means = values.mean(axis=(1, 2)).tolist()
        model_means = values.mean(axis=(0, 1)).tolist()
        signs = np.sign(values.mean(axis=1))
        coherence = float(np.mean([signs[a] * signs[b] for a, b in ((0, 1), (0, 2), (1, 2))]))
        significance = _bootstrap_positive(values, repeats, seed + candidate)
        row = {
            "component": spec,
            "family": spec.split(":", 1)[0],
            "mean_derivative": float(values.mean()),
            "mean_normalized_derivative": float(norm_values.mean()),
            "seed_means": seed_means,
            "positive_seeds": int(np.sum(np.asarray(seed_means) > 0)),
            "model_means": model_means,
            "positive_models": int(np.sum(np.asarray(model_means) > 0)),
            "seed_coherence": coherence,
            "mean_energy_ratio": float(energy[candidate].mean()),
            "screening_significance": significance,
        }
        row["eligible"] = row["positive_seeds"] == 3 and row["positive_models"] >= 6 and coherence > 0
        rows.append(row)
    rows.sort(key=lambda item: item["mean_normalized_derivative"], reverse=True)
    return rows


def _select_candidates(rows):
    eligible = [row for row in rows if row["eligible"]]
    selected = []
    fft = next((row for row in eligible if row["family"] == "fft"), None)
    if fft:
        selected.append(fft)
    occupied = set()
    local_count = 0
    for row in eligible:
        fields = row["component"].split(":")
        if row["family"] != "haar" or local_count >= 2:
            continue
        region = tuple(fields[-2:])
        if region not in occupied:
            selected.append(row)
            occupied.add(region)
            local_count += 1
    for row in eligible:
        if len(selected) >= 3:
            break
        if row not in selected:
            selected.append(row)
    return selected


def run_screen(args):
    loader, num_classes = _loader(args)
    source, attacker = build_baseline(num_classes)
    specs = screening_component_specs()
    shape = (len(specs), len(args.seeds), DISCOVERY_SAMPLES, len(args.target_models))
    derivative = np.full(shape, np.nan, dtype=np.float32)
    normalized = np.full(shape, np.nan, dtype=np.float32)
    energy = np.full((len(specs), len(args.seeds), DISCOVERY_SAMPLES), np.nan, dtype=np.float32)
    output = Path(args.output_dir)
    trace_dir = output / "screening_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Source and guide models are released before target models are loaded to bound GPU memory.
    for seed in args.seeds:
        torch.save(_source_traces(args, source, attacker, loader, seed, DISCOVERY_SAMPLES), trace_dir / f"seed_{seed}.pt")
    del attacker, source
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for model_index, name in enumerate(args.target_models):
        model = build_vit_model(num_classes=num_classes, model_name=name)
        for seed_index, seed in enumerate(args.seeds):
            source_data = torch.load(trace_dir / f"seed_{seed}.pt", map_location="cpu")
            sums = [np.zeros((len(specs), DISCOVERY_SAMPLES), dtype=np.float64) for _ in range(3)]
            counts = np.zeros(DISCOVERY_SAMPLES, dtype=np.int64)
            for trace in source_data["traces"]:
                target_grad = _target_gradient(model, trace["x_t"], trace["labels"])
                measurements = _component_measurements(trace["gradient"].to(DEVICE), target_grad)
                start, end = trace["start"], trace["end"]
                for metric_index in range(3):
                    sums[metric_index][:, start:end] += measurements[metric_index].T
                counts[start:end] += 1
            if not np.all(counts == len(TRACE_STEPS)):
                raise RuntimeError(f"Expected {len(TRACE_STEPS)} traces per discovery image, got {counts.tolist()}.")
            derivative[:, seed_index, :, model_index] = sums[0] / counts
            normalized[:, seed_index, :, model_index] = sums[1] / counts
            if model_index == 0:
                energy[:, seed_index, :] = sums[2] / counts
            del source_data
        np.savez_compressed(output / "screening_metrics.npz", derivative=derivative, normalized=normalized, energy=energy)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = _summarize_screening(specs, derivative, normalized, energy, args.bootstrap_repeats, args.bootstrap_seed)
    selected = _select_candidates(rows)
    _json(output / "screening_report.json", {"components": rows, "selected": selected})
    _json(output / "selected_candidates.json", {"components": [item["component"] for item in selected], "details": selected})
    print(f"Selected {len(selected)} candidates: {[item['component'] for item in selected]}")


def _collect_clean_samples(args, source, loader):
    args.max_samples = TOTAL_SAMPLES
    images, labels, indices = [], [], []
    for batch_images, batch_labels, batch_indices in selected_batches(args, source, loader):
        images.append(batch_images.cpu()); labels.append(batch_labels.cpu()); indices.append(batch_indices.cpu())
    return {
        "clean": torch.cat(images), "labels": torch.cat(labels), "indices": torch.cat(indices),
        "batch_sizes": [batch.size(0) for batch in images],
    }


def _load_baseline_adv(root, seed, expected_indices, limit=TOTAL_SAMPLES):
    batches, indices, count = [], [], 0
    for path in sorted((root / f"baseline_seed_{seed}").glob("batch_*.pt")):
        payload = torch.load(path, map_location="cpu")
        take = min(limit - count, payload["adv"].size(0))
        batches.append(payload["adv"][:take]); indices.append(payload["indices"][:take]); count += take
        if count == limit:
            break
    if count != limit:
        raise ValueError(f"Baseline root has {count}, not {limit}, samples for seed {seed}.")
    baseline_indices = torch.cat(indices)
    if not torch.equal(baseline_indices, expected_indices):
        raise ValueError(f"Baseline seed {seed} does not contain the same first {limit} source-correct images.")
    return torch.cat(batches)


def _load_candidates(args):
    if args.candidates:
        return list(args.candidates)
    payload = json.loads(Path(args.candidate_file).read_text(encoding="utf-8"))
    return payload["components"]


def run_confirm_attacks(args):
    loader, num_classes = _loader(args)
    source, attacker = build_baseline(num_classes)
    candidates = _load_candidates(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = _collect_clean_samples(args, source, loader)
    torch.save(samples, output / "samples.pt")
    for seed in args.seeds:
        seed_all(seed)
        if args.baseline_root:
            path = output / "full" / f"seed_{seed}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"adv": _load_baseline_adv(Path(args.baseline_root), seed, samples["indices"])}, path)
        for spec in candidates:
            projector = parse_component(spec)
            for condition in ("drop", "keep"):
                seed_all(seed)
                adv_batches = []
                transform = component_transform(projector, condition)
                start = 0
                for size in samples["batch_sizes"]:
                    end = start + size
                    adv_batches.append(run_analyzed_attack(
                        attacker, samples["clean"][start:end], samples["labels"][start:end], grad_transform=transform
                    ).cpu())
                    start = end
                path = output / _slug(spec) / condition / f"seed_{seed}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"adv": torch.cat(adv_batches)}, path)
    _json(output / "manifest.json", {"candidates": candidates, "seeds": list(args.seeds), "targets": list(args.target_models)})


def _evaluate_tensor(model, images, labels, batch_size):
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            pixels = images[start:start + batch_size].to(DEVICE).float() * 0.5 + 0.5
            logits = model(_target_normalize(model, pixels), return_attn=False)
            result.append(logits.argmax(1).cpu())
    return torch.cat(result)


def run_confirm_evaluation(args):
    root = Path(args.output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = torch.load(root / "samples.pt", map_location="cpu")
    labels = samples["labels"]
    results = {"indices": samples["indices"], "labels": labels, "models": {}}
    for name in args.target_models:
        model = build_vit_model(num_classes=1000, model_name=name)
        item = {"clean_correct": _evaluate_tensor(model, samples["clean"], labels, args.eval_batch_size).eq(labels)}
        item["full"] = {}
        for seed in args.seeds:
            adv = torch.load(root / "full" / f"seed_{seed}.pt", map_location="cpu")["adv"]
            item["full"][str(seed)] = _evaluate_tensor(model, adv, labels, args.eval_batch_size).ne(labels)
        item["candidates"] = {}
        for spec in manifest["candidates"]:
            conditions = {}
            for condition in ("drop", "keep"):
                conditions[condition] = {}
                for seed in args.seeds:
                    adv = torch.load(root / _slug(spec) / condition / f"seed_{seed}.pt", map_location="cpu")["adv"]
                    conditions[condition][str(seed)] = _evaluate_tensor(model, adv, labels, args.eval_batch_size).ne(labels)
            item["candidates"][spec] = conditions
        results["models"][name] = item
        del model
    torch.save(results, root / "evaluation.pt")


def _effect_tensor(evaluation, spec, condition):
    models = list(evaluation["models"])
    values = np.full((len(SEEDS), TOTAL_SAMPLES, len(models)), np.nan, dtype=np.float64)
    for model_index, name in enumerate(models):
        item = evaluation["models"][name]
        valid = item["clean_correct"].numpy().astype(bool)
        for seed_index, seed in enumerate(SEEDS):
            full = item["full"][str(seed)].numpy().astype(float)
            changed = item["candidates"][spec][condition][str(seed)].numpy().astype(float)
            values[seed_index, valid, model_index] = full[valid] - changed[valid]
    return values



def _describe_component(spec):
    fields = spec.split(":")
    if fields[0] == "fft":
        band = int(fields[1])
        return {
            "family": "global_fft", "band": band,
            "normalized_radial_frequency": [FFT_BANDS[band], FFT_BANDS[band + 1]],
            "orientation": fields[2], "spatial_location": "global",
        }
    row, col = int(fields[2]), int(fields[3])
    return {
        "family": "local_haar", "path": fields[1], "coefficient_region": [row, col],
        "approx_pixel_box": [col * 56, row * 56, (col + 1) * 56, (row + 1) * 56],
    }

def run_final_report(args):
    root = Path(args.output_dir)
    evaluation = torch.load(root / "evaluation.pt", map_location="cpu")
    screening = json.loads(Path(args.candidate_file).read_text(encoding="utf-8"))
    screen_by_spec = {item["component"]: item for item in screening.get("details", [])}
    reports, p_values = [], []
    for candidate_index, spec in enumerate(screening["components"]):
        drop = _effect_tensor(evaluation, spec, "drop")
        full_asr = []
        keep_asr = []
        common = np.ones(TOTAL_SAMPLES, dtype=bool)
        for item in evaluation["models"].values():
            common &= item["clean_correct"].numpy().astype(bool)
            for seed in SEEDS:
                full_asr.extend(item["full"][str(seed)].numpy()[item["clean_correct"]].tolist())
                keep_asr.extend(item["candidates"][spec]["keep"][str(seed)].numpy()[item["clean_correct"]].tolist())
        all_effect = _bootstrap_positive(drop, args.bootstrap_repeats, args.bootstrap_seed + candidate_index)
        held_out = _bootstrap_positive(drop[:, DISCOVERY_SAMPLES:, :], args.bootstrap_repeats, args.bootstrap_seed + 100 + candidate_index)
        common_effect = float(np.nanmean(drop[:, common, :]))
        seed_effects = np.nanmean(drop, axis=(1, 2)).tolist()
        model_effects = np.nanmean(drop, axis=(0, 1)).tolist()
        screen = screen_by_spec.get(spec, {})
        report = {
            "component": spec,
            "description": _describe_component(spec),
            "all_100_delta_drop": all_effect,
            "held_out_70_delta_drop": held_out,
            "keep_only_ratio": float(np.mean(keep_asr) / np.mean(full_asr)) if np.mean(full_asr) else None,
            "seed_delta_drop": seed_effects,
            "positive_seeds": int(np.sum(np.asarray(seed_effects) > 0)),
            "model_delta_drop": model_effects,
            "positive_models": int(np.sum(np.asarray(model_effects) > 0)),
            "common_correct_delta_drop": common_effect,
            "screening_significance": screen.get("screening_significance"),
        }
        reports.append(report); p_values.append(held_out["p_positive"])
    for report, q_value in zip(reports, bh_fdr(p_values)):
        report["held_out_70_delta_drop"]["q_positive"] = q_value
        screen_sig = report["screening_significance"] or {}
        report["confirmed"] = (
            report["held_out_70_delta_drop"]["mean"] > 0
            and report["held_out_70_delta_drop"]["ci_low"] > 0
            and q_value < 0.05
            and report["positive_seeds"] == 3
            and report["positive_models"] >= 7
            and report["common_correct_delta_drop"] > 0
            and screen_sig.get("ci_low", float("-inf")) > 0
        )
        report["status"] = "confirmed" if report["confirmed"] else (
            "model-dependent" if report["positive_models"] < 7 else "not-supported"
        )
    _json(root / "final_report.json", {"candidates": reports})
    print(json.dumps({item["component"]: item["status"] for item in reports}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("screen", "confirm-attacks", "confirm-evaluate", "report"))
    parser.add_argument("--output-dir", default="outputs/cross_vit_components")
    parser.add_argument("--candidate-file", default="outputs/cross_vit_components/selected_candidates.json")
    parser.add_argument("--candidates", type=parse_model_names, default=())
    parser.add_argument("--baseline-root", default="outputs/causal_full")
    parser.add_argument("--image-dir", default=IMAGE_DIR); parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=224); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64); parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=lambda value: tuple(map(int, value.split(","))), default=SEEDS)
    parser.add_argument("--target-models", type=parse_model_names, default=MAIN_TARGETS)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000); parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if tuple(args.seeds) != SEEDS:
        raise ValueError("The fixed protocol requires seeds 0,1,2.")
    if len(args.target_models) != 8:
        raise ValueError("The fixed protocol requires exactly eight target models.")
    {"screen": run_screen, "confirm-attacks": run_confirm_attacks, "confirm-evaluate": run_confirm_evaluation, "report": run_final_report}[args.mode](args)
