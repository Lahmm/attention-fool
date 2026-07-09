"""Analyze transferable perturbations from saved adversarial examples.

This script starts from final AE pixels, labels each sample by per-model
transfer success, and summarizes where/how successful perturbation energy is
allocated. It is intentionally model-agnostic for the pixel analysis and uses
transfer_eval's black-box model builders for per-sample success labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import transfer_eval
from main import ANNOTATIONS_PATH, IMAGE_DIR, parse_model_names
from utils import DEVICE

DEFAULT_ADV_DIR = "outputs/attack/lmdss_ablation/step20_feature10_traj_dropout_copies10_s500"
DEFAULT_OUTPUT_DIR = "outputs/analysis/ae_transfer_perturbation_step20_feature10_traj_copies10"
DEFAULT_MODELS = ("vit_base_patch16_224", *transfer_eval.DEFAULT_BLACK_BOX_MODELS)


def _load_rgb01(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        data = data.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
        return data


def _gray(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def _sobel(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kx = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = gray.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), ky)
    mag = (gx.square() + gy.square()).sqrt()
    return gx, gy, mag


def _laplacian(gray: torch.Tensor) -> torch.Tensor:
    kernel = gray.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    return F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel)


def _radial_masks(height: int, width: int, device: torch.device) -> dict[str, torch.Tensor]:
    yy = torch.linspace(-1.0, 1.0, height, device=device).view(height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=device).view(1, width)
    radius = (xx.square() + yy.square()).sqrt()
    return {
        "center": (radius <= 0.45).view(1, 1, height, width),
        "mid": ((radius > 0.45) & (radius <= 0.8)).view(1, 1, height, width),
        "border": (radius > 0.8).view(1, 1, height, width),
    }


def _fft_band_energy(delta_luma: torch.Tensor) -> dict[str, torch.Tensor]:
    bsz, _c, height, width = delta_luma.shape
    fy = torch.fft.fftfreq(height, device=delta_luma.device).view(height, 1)
    fx = torch.fft.fftfreq(width, device=delta_luma.device).view(1, width)
    radius = (fx.square() + fy.square()).sqrt().view(1, height, width)
    fft = torch.fft.fft2(delta_luma[:, 0].float(), dim=(-2, -1), norm="ortho")
    power = fft.abs().square()
    bands = {
        "fft_low_energy": radius < 0.08,
        "fft_mid_energy": (radius >= 0.08) & (radius < 0.25),
        "fft_high_energy": radius >= 0.25,
    }
    return {name: power[:, mask.expand_as(power)[0]].sum(dim=1) for name, mask in bands.items()}


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp_min(1e-12)


def collect_pairs(adv_dir: Path, clean_dir: Path, prefix: str, max_images: int | None) -> list[tuple[Path, Path, str]]:
    adv_paths = sorted(path for path in adv_dir.iterdir() if path.is_file() and path.name.startswith(prefix))
    if max_images is not None:
        adv_paths = adv_paths[:max_images]
    pairs = []
    for adv_path in adv_paths:
        original = adv_path.name[len(prefix):]
        clean_path = clean_dir / original
        if not clean_path.is_file():
            raise FileNotFoundError(f"clean image not found for {adv_path.name}: {clean_path}")
        pairs.append((adv_path, clean_path, original))
    return pairs


def _annotation_stem_index(annotations: dict[str, dict[str, int | str]]) -> dict[str, str]:
    return {Path(name).stem: name for name in annotations}


def _lookup_annotation(annotations: dict[str, dict[str, int | str]], stem_index: dict[str, str], image_name: str) -> dict[str, int | str] | None:
    direct = annotations.get(image_name)
    if direct is not None:
        return direct
    stem_key = stem_index.get(Path(image_name).stem)
    return None if stem_key is None else annotations.get(stem_key)


def compute_pixel_metrics(pairs: list[tuple[Path, Path, str]], annotations: dict[str, dict[str, int | str]], batch_size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stem_index = _annotation_stem_index(annotations)
    for start in tqdm(range(0, len(pairs), batch_size), desc="pixel metrics"):
        batch = pairs[start:start + batch_size]
        clean = torch.stack([_load_rgb01(clean_path) for _adv_path, clean_path, _name in batch])
        adv = torch.stack([_load_rgb01(adv_path) for adv_path, _clean_path, _name in batch])
        delta = adv - clean
        abs_delta = delta.abs()
        energy = delta.square().sum(dim=1, keepdim=True)
        energy_sum = energy.flatten(1).sum(dim=1)
        luma = _gray(clean)
        delta_luma = _gray(delta)
        abs_luma = delta_luma.abs()
        clean_gx, clean_gy, clean_edge = _sobel(luma)
        delta_gx, delta_gy, _delta_edge = _sobel(delta_luma)
        lap = _laplacian(luma)

        edge_flat = clean_edge.flatten(1)
        edge_threshold = torch.quantile(edge_flat, 0.80, dim=1, keepdim=True).view(-1, 1, 1, 1)
        edge_mask = clean_edge >= edge_threshold
        masks = _radial_masks(clean.size(-2), clean.size(-1), clean.device)

        edge_energy = (energy * edge_mask).flatten(1).sum(1)
        nonedge_energy = (energy * (~edge_mask)).flatten(1).sum(1)
        center_energy = (energy * masks["center"]).flatten(1).sum(1)
        mid_energy = (energy * masks["mid"]).flatten(1).sum(1)
        border_energy = (energy * masks["border"]).flatten(1).sum(1)
        pos_luma_energy = (abs_luma * (delta_luma > 0)).flatten(1).sum(1)
        neg_luma_energy = (abs_luma * (delta_luma < 0)).flatten(1).sum(1)

        clean_grad_norm = (clean_gx.square() + clean_gy.square()).sqrt()
        delta_grad_norm = (delta_gx.square() + delta_gy.square()).sqrt()
        edge_align = ((clean_gx * delta_gx + clean_gy * delta_gy).flatten(1).sum(1) /
                      (clean_grad_norm.flatten(1).norm(dim=1) * delta_grad_norm.flatten(1).norm(dim=1)).clamp_min(1e-12))
        lap_sign_agree = ((delta_luma.sign() == lap.sign()).float() * abs_luma).flatten(1).sum(1)
        lap_sign_agree = _safe_ratio(lap_sign_agree, abs_luma.flatten(1).sum(1))

        fft = _fft_band_energy(delta_luma)
        fft_total = sum(fft.values())

        for i, (_adv_path, _clean_path, original) in enumerate(batch):
            label_info = _lookup_annotation(annotations, stem_index, original)
            if label_info is None:
                raise KeyError(f"annotation not found for {original}")
            row = {
                "image_name": original,
                "class_id": int(label_info["class_id"]),
                "linf": float(abs_delta[i].max()),
                "mean_abs_delta": float(abs_delta[i].mean()),
                "l2_delta": float(delta[i].flatten().norm()),
                "luma_signed_mean": float(delta_luma[i].mean()),
                "luma_abs_mean": float(abs_luma[i].mean()),
                "positive_luma_ratio": float(_safe_ratio(pos_luma_energy[i], pos_luma_energy[i] + neg_luma_energy[i])),
                "edge_energy_ratio": float(_safe_ratio(edge_energy[i], energy_sum[i])),
                "nonedge_energy_ratio": float(_safe_ratio(nonedge_energy[i], energy_sum[i])),
                "center_energy_ratio": float(_safe_ratio(center_energy[i], energy_sum[i])),
                "mid_energy_ratio": float(_safe_ratio(mid_energy[i], energy_sum[i])),
                "border_energy_ratio": float(_safe_ratio(border_energy[i], energy_sum[i])),
                "edge_gradient_alignment": float(edge_align[i]),
                "laplacian_sign_agreement": float(lap_sign_agree[i]),
                "fft_low_ratio": float(_safe_ratio(fft["fft_low_energy"][i], fft_total[i])),
                "fft_mid_ratio": float(_safe_ratio(fft["fft_mid_energy"][i], fft_total[i])),
                "fft_high_ratio": float(_safe_ratio(fft["fft_high_energy"][i], fft_total[i])),
            }
            rows.append(row)
    return rows


def _predict_paths(model, transform, paths: list[Path], batch_size: int) -> list[int]:
    preds: list[int] = []
    for start in tqdm(range(0, len(paths), batch_size), desc="predict", leave=False):
        batch_paths = paths[start:start + batch_size]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        images = torch.stack(tensors).to(DEVICE)
        with torch.inference_mode():
            logits = model(images)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return preds


def attach_transfer_labels(rows: list[dict[str, object]], pairs: list[tuple[Path, Path, str]], model_names: Iterable[str], batch_size: int) -> tuple[list[str], list[dict[str, object]]]:
    adv_paths = [adv_path for adv_path, _clean_path, _name in pairs]
    clean_paths = [clean_path for _adv_path, clean_path, _name in pairs]
    labels = [int(row["class_id"]) for row in rows]
    usable_models: list[str] = []
    model_rows: list[dict[str, object]] = []

    for model_name in model_names:
        try:
            model, transform = transfer_eval.build_black_box_model(model_name)
        except transfer_eval.ModelUnavailableError as exc:
            print(f"skip unavailable {model_name}: {exc}")
            continue
        print(f"evaluating per-sample transfer: {model_name}")
        adv_preds = _predict_paths(model, transform, adv_paths, batch_size)
        clean_preds = _predict_paths(model, transform, clean_paths, batch_size)
        correct_adv = sum(int(pred == label) for pred, label in zip(adv_preds, labels))
        correct_clean = sum(int(pred == label) for pred, label in zip(clean_preds, labels))
        total = len(labels)
        usable_models.append(model_name)
        model_rows.append({
            "model_name": model_name,
            "total": total,
            "adv_correct": correct_adv,
            "adv_asr": 1.0 - correct_adv / max(total, 1),
            "clean_correct": correct_clean,
            "clean_acc": correct_clean / max(total, 1),
        })
        for row, adv_pred, clean_pred, label in zip(rows, adv_preds, clean_preds, labels):
            row[f"{model_name}_adv_pred"] = adv_pred
            row[f"{model_name}_clean_pred"] = clean_pred
            row[f"{model_name}_success"] = int(adv_pred != label)
            row[f"{model_name}_clean_correct"] = int(clean_pred == label)

    for row in rows:
        successes = [int(row[f"{name}_success"]) for name in usable_models]
        clean_correct = [int(row[f"{name}_clean_correct"]) for name in usable_models]
        row["transfer_count"] = sum(successes)
        row["transfer_rate"] = sum(successes) / max(len(successes), 1)
        row["clean_correct_count"] = sum(clean_correct)
    return usable_models, model_rows


def _mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return sum(vals) / max(len(vals), 1)


def summarize(rows: list[dict[str, object]], usable_models: list[str], model_rows: list[dict[str, object]]) -> dict[str, object]:
    sorted_rows = sorted(rows, key=lambda row: float(row.get("transfer_rate", 0.0)))
    n = len(sorted_rows)
    low = sorted_rows[:max(1, n // 4)]
    high = sorted_rows[-max(1, n // 4):]
    keys = [
        "linf", "mean_abs_delta", "l2_delta", "positive_luma_ratio", "edge_energy_ratio",
        "center_energy_ratio", "mid_energy_ratio", "border_energy_ratio", "edge_gradient_alignment",
        "laplacian_sign_agreement", "fft_low_ratio", "fft_mid_ratio", "fft_high_ratio",
    ]
    high_low_delta = {key: _mean(high, key) - _mean(low, key) for key in keys}
    correlations = {}
    rates = torch.tensor([float(row.get("transfer_rate", 0.0)) for row in rows])
    for key in keys:
        vals = torch.tensor([float(row[key]) for row in rows])
        vx = vals - vals.mean()
        vy = rates - rates.mean()
        correlations[key] = float((vx * vy).sum() / (vx.norm() * vy.norm()).clamp_min(1e-12))
    return {
        "num_samples": len(rows),
        "usable_models": usable_models,
        "model_asr": model_rows,
        "mean_transfer_rate": _mean(rows, "transfer_rate") if usable_models else None,
        "high_transfer_minus_low_transfer": high_low_delta,
        "metric_transfer_correlations": correlations,
        "high_transfer_threshold": float(high[0].get("transfer_rate", 0.0)) if high else None,
        "low_transfer_threshold": float(low[-1].get("transfer_rate", 0.0)) if low else None,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, object]) -> None:
    corr = summary["metric_transfer_correlations"]
    delta = summary["high_transfer_minus_low_transfer"]
    model_asr = summary["model_asr"]
    lines = [
        "# AE Transfer Perturbation Analysis",
        "",
        f"Samples: {summary['num_samples']}",
        f"Models: {', '.join(summary['usable_models'])}",
        f"Mean transfer rate: {summary['mean_transfer_rate']}",
        "",
        "## Model ASR",
    ]
    for row in model_asr:
        lines.append(f"- {row['model_name']}: adv_asr={row['adv_asr']:.4f}, clean_acc={row['clean_acc']:.4f}")
    lines.extend(["", "## High-transfer vs low-transfer metric deltas"])
    for key, value in sorted(delta.items(), key=lambda item: abs(float(item[1])), reverse=True):
        lines.append(f"- {key}: {value:.6g}")
    lines.extend(["", "## Correlation with transfer rate"])
    for key, value in sorted(corr.items(), key=lambda item: abs(float(item[1])), reverse=True):
        lines.append(f"- {key}: {value:.6g}")
    lines.extend([
        "",
        "## Working interpretation",
        "- Positive correlations indicate perturbation statistics enriched in samples that fool more target models.",
        "- Edge/FFT/center metrics are pixel-level evidence only; pair them with gradient experiments before claiming causality.",
        "- A DIM-oriented replacement for trajectory dropout should preferentially preserve low/mid, edge/foreground-stable directions while reducing high-frequency and texture-noise dependence.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze transferable perturbations from saved AE pixels.")
    parser.add_argument("--adv-dir", default=DEFAULT_ADV_DIR)
    parser.add_argument("--clean-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--prefix", default="adv_")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=parse_model_names, default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-model-eval", action="store_true")
    args = parser.parse_args()

    adv_dir = Path(args.adv_dir)
    clean_dir = Path(args.clean_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = transfer_eval.load_annotations(Path(args.annotations_path))
    pairs = collect_pairs(adv_dir, clean_dir, args.prefix, args.max_images)
    rows = compute_pixel_metrics(pairs, annotations, args.batch_size)
    usable_models: list[str] = []
    model_rows: list[dict[str, object]] = []
    if not args.skip_model_eval:
        usable_models, model_rows = attach_transfer_labels(rows, pairs, args.model_name, args.batch_size)
    summary = summarize(rows, usable_models, model_rows)
    write_csv(output_dir / "per_sample_metrics.csv", rows)
    write_csv(output_dir / "model_asr.csv", model_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir / "report.md", summary)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
