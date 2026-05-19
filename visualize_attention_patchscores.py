import argparse
import csv
import math
from pathlib import Path
from typing import Any, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nets import build_vit_model, ViTWithHook, DEFAULT_MODEL_NAME
from utils import (
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    image_2d_fft_low_high_maps,
    last_vit_foreground_background_from_tokens,
    last_vit_patch_scores_to_image_map,
    last_vit_stable_patch_frequency,
    load_data,
)

HEATMAP_CMAP = "turbo"
TITLE_FONT_SIZE = 15
COLORBAR_LABEL_SIZE = 13
COLORBAR_TICK_SIZE = 11
OVERLAP_LEGEND_FONT_SIZE = 11
PATCH_SCORE_DISPLAY_MIN = 0.5
PATCH_SCORE_DISPLAY_MAX = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention patch scores and A@V patch scores for images in one directory.")
    parser.add_argument("--image-dir",type=str,required=True,help="Directory containing images to visualize (e.g., data/clean_resized_images or outputs).")
    parser.add_argument("--pattern",type=str,default="*",help='Filename glob in --image-dir (e.g., "adv_*.png" or "clean_*.png").')
    parser.add_argument("--max-images",type=int,default=5,help="Process first N matched images.")
    parser.add_argument("--block-index",type=int,default=-1,help="Transformer block index to visualize. Supports negative index; -1 means last block.")
    parser.add_argument("--block-indices", type=parse_block_indices, default=None, help='Comma-separated block indices for CSV stats, e.g. "-6,-5,-4,-3,-2,-1".')
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="timm model name used for visualization.")
    parser.add_argument("--model-names", type=parse_model_names, default=(), help="Comma-separated model names for cross-model overlap.")
    parser.add_argument("--fft-topk", type=int, default=1, help="Per-channel Top-K stable patch count used by FFT selection.")
    parser.add_argument("--fft-alpha", type=float, default=0.6, help="FFT stability heatmap overlay opacity.")
    parser.add_argument("--image-fft-alpha", type=float, default=0.70, help="Image-space 2D FFT low/high overlay opacity.")
    parser.add_argument("--image-fft-cutoff-ratio", type=float, default=0.15, help="Circular low-pass cutoff ratio for image-space 2D FFT.")
    parser.add_argument("--image-fft-transition-ratio", type=float, default=0.04, help="Soft transition width for image-space 2D FFT low-pass mask.")
    parser.add_argument("--overlap-top-ratio", type=float, default=0.2, help="Top fraction used to mark high FFT, attention, and patch-score regions in the overlap panel.")
    parser.add_argument("--output-dir",type=str,default=None,help="Directory where visualization figures are saved.")
    parser.add_argument("--no-save-images", action="store_true", help="Only compute CSV statistics; do not save visualization images.")
    parser.add_argument("--stats-csv", type=str, default=None, help="Path for per-image/per-layer CSV statistics.")
    parser.add_argument("--summary-csv", type=str, default=None, help="Path for per-model/per-layer summary CSV statistics.")
    parser.add_argument("--cross-model-overlap", action="store_true", help="Compute cross-model top-mask overlap statistics.")
    parser.add_argument("--cross-stats-csv", type=str, default=None, help="Path for per-image/per-layer/model-pair overlap CSV.")
    parser.add_argument("--cross-summary-csv", type=str, default=None, help="Path for cross-model overlap summary CSV.")
    parser.add_argument("--cross-conclusion-csv", type=str, default=None, help="Path for cross-model overlap conclusion CSV.")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--dataset-dir", type=str, default="data/clean_resized_images")
    parser.add_argument("--annotations-path",type=str,default="data/image_name_to_class_id_and_name.json")
    return parser.parse_args()


def parse_block_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not indices:
        raise argparse.ArgumentTypeError("block indices must contain at least one integer.")
    return indices


def parse_model_names(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_model(num_classes: int, model_name: str) -> ViTWithHook:
    model = build_vit_model(num_classes=num_classes, model_name=model_name)
    model.eval()
    return model


def resolve_num_classes(
    num_classes: Optional[int],
    dataset_dir: Path,
    annotations_path: Path,
    img_size: int,
) -> int:
    if num_classes is not None:
        return num_classes
    _, inferred = load_data(
        image_dir_arg=str(dataset_dir),
        annotations_path_arg=str(annotations_path),
        batch_size=1,
        num_workers=0,
        img_size=img_size,
    )
    return inferred


def list_image_paths(image_dir: Path, pattern: str, max_images: int) -> List[Path]:
    candidates = sorted(image_dir.glob(pattern))
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = [p for p in candidates if p.is_file() and p.suffix.lower() in image_exts]
    if not paths:
        raise FileNotFoundError(f"No image found in {image_dir} with pattern {pattern}.")
    return paths[:max_images] if max_images is not None else paths


def load_rgb_image(path: Path, img_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((img_size, img_size), Image.BICUBIC)
    return np.array(image).astype(np.float32) / 255.0


def preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def preprocess_image(path: Path, img_size: int) -> torch.Tensor:
    return preprocess_rgb(load_rgb_image(path, img_size))


def resolve_block_index(block_index: int, total_blocks: int) -> int:
    resolved = block_index if block_index >= 0 else total_blocks + block_index
    if resolved < 0 or resolved >= total_blocks:
        raise ValueError(f"Invalid --block-index {block_index}, total blocks={total_blocks}.")
    return resolved


def is_square_token_count(num_tokens: int) -> bool:
    if num_tokens <= 0:
        return False
    grid_size = int(math.sqrt(num_tokens))
    return grid_size * grid_size == num_tokens


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(flat, dtype=np.float64)
    sorted_vals = flat[order]
    start = 0
    while start < flat.size:
        end = start + 1
        while end < flat.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks.reshape(values.shape)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_corr(rankdata(a), rankdata(b))


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = mask_a | mask_b
    if not union.any():
        return 0.0
    return float((mask_a & mask_b).sum() / union.sum())


def mask_mean(score_map: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.asarray(score_map, dtype=np.float64)[mask].mean())


class GenericAttentionCapture:
    """Capture qkv outputs and matching block outputs across timm ViT-like models."""

    def __init__(self, model: ViTWithHook) -> None:
        self.model = model.model
        self.module_dict = dict(self.model.named_modules())
        self.records: list[dict[str, Any]] = []
        self.handles = []
        self._discover_records()

    def _discover_records(self) -> None:
        for qkv_name, qkv_module in self.module_dict.items():
            if not qkv_name.endswith("attn.qkv"):
                continue
            attn_name = qkv_name.rsplit(".qkv", 1)[0]
            block_name = qkv_name.rsplit(".attn.qkv", 1)[0]
            attn_module = self.module_dict.get(attn_name)
            block_module = self.module_dict.get(block_name)
            if attn_module is None or block_module is None:
                continue
            record = {
                "index": len(self.records),
                "mode": "qkv",
                "qkv_name": qkv_name,
                "block_name": block_name,
                "attn_module": attn_module,
                "block_module": block_module,
                "num_heads": getattr(attn_module, "num_heads", None),
                "qkv": None,
                "attn_input": None,
                "tokens": None,
            }
            self.records.append(record)

        for q_name, q_module in self.module_dict.items():
            if not q_name.endswith("attn.q"):
                continue
            attn_name = q_name.rsplit(".q", 1)[0]
            k_module = self.module_dict.get(f"{attn_name}.k")
            v_module = self.module_dict.get(f"{attn_name}.v")
            attn_module = self.module_dict.get(attn_name)
            block_name = q_name.rsplit(".attn.q", 1)[0]
            block_module = self.module_dict.get(block_name)
            if k_module is None or v_module is None or attn_module is None or block_module is None:
                continue
            record = {
                "index": len(self.records),
                "mode": "class_attn",
                "qkv_name": q_name,
                "block_name": block_name,
                "attn_module": attn_module,
                "block_module": block_module,
                "num_heads": getattr(attn_module, "num_heads", None),
                "qkv": None,
                "attn_input": None,
                "tokens": None,
            }
            self.records.append(record)

    def __enter__(self):
        for record in self.records:
            if hasattr(record["attn_module"], "qkv"):
                self.handles.append(record["attn_module"].qkv.register_forward_hook(self._qkv_hook(record)))
            self.handles.append(record["attn_module"].register_forward_pre_hook(self._attn_pre_hook(record)))
            self.handles.append(record["block_module"].register_forward_hook(self._block_hook(record)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self) -> None:
        for record in self.records:
            record["qkv"] = None
            record["attn_input"] = None
            record["tokens"] = None

    @staticmethod
    def _qkv_hook(record: dict[str, Any]):
        def hook(_module, _inputs, output):
            record["qkv"] = output.detach() if isinstance(output, torch.Tensor) else None

        return hook

    @staticmethod
    def _attn_pre_hook(record: dict[str, Any]):
        def hook(_module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                record["attn_input"] = inputs[0].detach()

        return hook

    @staticmethod
    def _block_hook(record: dict[str, Any]):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                record["tokens"] = output.detach()
            elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                record["tokens"] = output[0].detach()
            else:
                record["tokens"] = None

        return hook


def qkv_to_attn_logits(qkv: torch.Tensor, num_heads: int | None) -> torch.Tensor:
    if qkv.ndim != 3 or num_heads is None or num_heads <= 0:
        raise ValueError(f"Unsupported qkv shape/heads: shape={tuple(qkv.shape)}, heads={num_heads}.")
    bsz, num_tokens, hidden = qkv.shape
    if hidden % (3 * int(num_heads)) != 0:
        raise ValueError(f"qkv hidden size {hidden} is not divisible by 3*num_heads={3 * int(num_heads)}.")
    head_dim = hidden // (3 * int(num_heads))
    qkv_view = qkv.reshape(bsz, num_tokens, 3, int(num_heads), head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv_view[0], qkv_view[1]
    return (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)



def build_qkv_from_attn_input(record: dict[str, Any]) -> torch.Tensor:
    x = record.get("attn_input")
    if x is None:
        raise ValueError("Missing attention input for qkv reconstruction.")
    attn_module = record["attn_module"]
    qkv_layer = getattr(attn_module, "qkv", None)
    if qkv_layer is None:
        raise ValueError("Attention module does not expose qkv.")

    q_bias = getattr(attn_module, "q_bias", None)
    if q_bias is None:
        return qkv_layer(x)

    qkv_bias = torch.cat((attn_module.q_bias, attn_module.k_bias, attn_module.v_bias))
    if getattr(attn_module, "qkv_bias_separate", False):
        qkv = qkv_layer(x)
        return qkv + qkv_bias
    return F.linear(x, weight=qkv_layer.weight, bias=qkv_bias)


def qkv_record_to_attn_logits(record: dict[str, Any]) -> torch.Tensor:
    qkv = record.get("qkv")
    if qkv is None:
        qkv = build_qkv_from_attn_input(record)
    return qkv_to_attn_logits(qkv, record["num_heads"])


def class_attn_record_to_attn_logits(record: dict[str, Any]) -> torch.Tensor:
    x = record.get("attn_input")
    if x is None:
        raise ValueError("Missing class attention input.")
    attn_module = record["attn_module"]
    num_heads = int(record["num_heads"])
    if num_heads <= 0:
        raise ValueError(f"Invalid class attention heads: {record['num_heads']}.")
    bsz, num_tokens, channels = x.shape
    head_dim = channels // num_heads
    if channels % num_heads != 0:
        raise ValueError(f"Class attention channels {channels} not divisible by heads {num_heads}.")
    q = attn_module.q(x[:, 0]).unsqueeze(1).reshape(bsz, 1, num_heads, head_dim).permute(0, 2, 1, 3)
    k = attn_module.k(x).reshape(bsz, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    scale = getattr(attn_module, "scale", head_dim ** -0.5)
    return (q * scale) @ k.transpose(-2, -1)

def flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 3:
        return tokens
    if tokens.ndim == 4:
        bsz, height, width, channels = tokens.shape
        return tokens.reshape(bsz, height * width, channels)
    raise ValueError(f"Unsupported token shape: {tuple(tokens.shape)}.")


def normalized_patch_scores(scores: torch.Tensor, img_size: int) -> np.ndarray:
    score_map = last_vit_patch_scores_to_image_map(scores, img_size=img_size, mode="bilinear")[0]
    return normalize_map(score_map.detach().cpu().numpy())


def compute_cls_stats(
    record: dict[str, Any],
    img_size: int,
    fft_topk: int,
    overlap_top_ratio: float,
) -> dict[str, Any]:
    tokens = flatten_tokens(record["tokens"])
    qkv = record["qkv"]
    num_patches = tokens.size(1) - 1
    if qkv is None:
        raise ValueError("Missing qkv output.")
    if num_patches <= 0 or not is_square_token_count(num_patches):
        raise ValueError(f"CLS mode requires square patch tokens, got token count {tokens.size(1)}.")

    attn_logits = qkv_to_attn_logits(qkv, record["num_heads"])
    attn_weights = torch.softmax(attn_logits, dim=-1)
    attn_scores = attn_weights[:, :, 0, 1:].mean(dim=1)
    attn_map = normalized_patch_scores(attn_scores, img_size)

    cls_token = tokens[:, 0, :]
    patch_tokens = tokens[:, 1:, :]
    cls_token_expanded = cls_token.unsqueeze(1).expand_as(patch_tokens)
    patch_scores = F.cosine_similarity(patch_tokens, cls_token_expanded, dim=-1)
    patch_map = normalized_patch_scores(patch_scores, img_size)

    fft_scores = last_vit_stable_patch_frequency(tokens=tokens, topk=fft_topk, has_cls_token=True)
    fft_map = normalized_patch_scores(fft_scores, img_size)

    return compute_overlap_stats(
        attn_map=attn_map,
        patch_map=patch_map,
        fft_map=fft_map,
        top_ratio=overlap_top_ratio,
        metric_mode="cls",
    )


def compute_no_cls_stats(
    record: dict[str, Any],
    img_size: int,
    fft_topk: int,
    overlap_top_ratio: float,
) -> dict[str, Any]:
    tokens = flatten_tokens(record["tokens"])
    num_patches = tokens.size(1)
    if num_patches <= 0 or not is_square_token_count(num_patches):
        raise ValueError(f"no-CLS mode requires square token grid, got token count {num_patches}.")

    token_norm = tokens.norm(dim=-1)
    attn_map = normalized_patch_scores(token_norm, img_size)
    centered = tokens - tokens.mean(dim=1, keepdim=True)
    patch_scores = centered.norm(dim=-1)
    patch_map = normalized_patch_scores(patch_scores, img_size)
    fft_scores = last_vit_stable_patch_frequency(tokens=tokens, topk=fft_topk, has_cls_token=False)
    fft_map = normalized_patch_scores(fft_scores, img_size)
    return compute_overlap_stats(
        attn_map=attn_map,
        patch_map=patch_map,
        fft_map=fft_map,
        top_ratio=overlap_top_ratio,
        metric_mode="no_cls",
    )


def compute_overlap_stats(
    attn_map: np.ndarray,
    patch_map: np.ndarray,
    fft_map: np.ndarray,
    top_ratio: float,
    metric_mode: str,
) -> dict[str, Any]:
    attn_high = top_ratio_mask(normalize_map(attn_map), top_ratio)
    patch_high = top_ratio_mask(normalize_map(patch_map), top_ratio)
    fft_high = top_ratio_mask(normalize_map(fft_map), top_ratio)
    attn_patch_high = attn_high & patch_high
    all_high = attn_patch_high & fft_high

    foreground_mean = mask_mean(fft_map, attn_patch_high)
    anchor_mean = mask_mean(fft_map, ~(attn_high | patch_high))
    return {
        "metric_mode": metric_mode,
        "attn_patch_jaccard": jaccard(attn_high, patch_high),
        "fft_capture_ratio": float(all_high.sum() / attn_patch_high.sum()) if attn_patch_high.any() else 0.0,
        "fft_precision_ratio": float(all_high.sum() / fft_high.sum()) if fft_high.any() else 0.0,
        "attn_fft_corr": pearson_corr(attn_map, fft_map),
        "patch_fft_corr": pearson_corr(patch_map, fft_map),
        "attn_patch_corr": pearson_corr(attn_map, patch_map),
        "attn_fft_spearman": spearman_corr(attn_map, fft_map),
        "patch_fft_spearman": spearman_corr(patch_map, fft_map),
        "attn_patch_spearman": spearman_corr(attn_map, patch_map),
        "foreground_mean": foreground_mean,
        "anchor_mean": anchor_mean,
        "foreground_anchor_separation": foreground_mean - anchor_mean,
    }


STATS_FIELDS = [
    "model_name",
    "image_name",
    "block_index",
    "resolved_block_index",
    "block_name",
    "metric_mode",
    "pred_class",
    "attn_patch_jaccard",
    "fft_capture_ratio",
    "fft_precision_ratio",
    "attn_fft_corr",
    "patch_fft_corr",
    "attn_patch_corr",
    "attn_fft_spearman",
    "patch_fft_spearman",
    "attn_patch_spearman",
    "foreground_mean",
    "anchor_mean",
    "foreground_anchor_separation",
    "error",
]


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "attn_patch_jaccard",
        "fft_capture_ratio",
        "fft_precision_ratio",
        "attn_fft_corr",
        "patch_fft_corr",
        "attn_patch_corr",
        "foreground_anchor_separation",
    ]
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model_name"]),
            int(row["block_index"]),
            int(row["resolved_block_index"]),
            str(row.get("metric_mode", "")),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (model_name, block_index, resolved_block_index, metric_mode), group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "model_name": model_name,
            "block_index": block_index,
            "resolved_block_index": resolved_block_index,
            "metric_mode": metric_mode,
            "records": len(group),
            "errors": sum(1 for row in group if row.get("error")),
        }
        for metric in metrics:
            values = np.array([safe_float(row.get(metric)) for row in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                out[f"{metric}_mean"] = float("nan")
                out[f"{metric}_median"] = float("nan")
                out[f"{metric}_std"] = float("nan")
                out[f"{metric}_p25"] = float("nan")
                out[f"{metric}_p75"] = float("nan")
                continue
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_median"] = float(np.median(values))
            out[f"{metric}_std"] = float(values.std(ddof=0))
            out[f"{metric}_p25"] = float(np.percentile(values, 25))
            out[f"{metric}_p75"] = float(np.percentile(values, 75))
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_csv_stats(args, model: ViTWithHook, image_paths: list[Path], output_dir: Path) -> None:
    block_indices = args.block_indices if args.block_indices is not None else (args.block_index,)
    stats_csv = Path(args.stats_csv) if args.stats_csv else output_dir / "attention_patchscores_stats.csv"
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "attention_patchscores_summary.csv"
    rows: list[dict[str, Any]] = []

    with GenericAttentionCapture(model) as capture:
        if not capture.records:
            raise RuntimeError(f"No attention qkv modules found for model {args.model_name}.")
        resolved_indices = [resolve_block_index(idx, len(capture.records)) for idx in block_indices]

        for path in image_paths:
            print(f"Processing {path}")
            original = load_rgb_image(path, args.img_size)
            tensor = preprocess_rgb(original).unsqueeze(0).to(DEVICE)
            capture.reset()
            with torch.no_grad():
                logits = model.model(tensor)
                pred = logits.argmax(dim=1).item()

            for requested_idx, resolved_idx in zip(block_indices, resolved_indices):
                record = capture.records[resolved_idx]
                row = {
                    "model_name": args.model_name,
                    "image_name": path.name,
                    "block_index": requested_idx,
                    "resolved_block_index": resolved_idx,
                    "block_name": record["block_name"],
                    "metric_mode": "",
                    "pred_class": pred,
                    "error": "",
                }
                try:
                    tokens = flatten_tokens(record["tokens"])
                    if tokens.size(1) > 1 and is_square_token_count(tokens.size(1) - 1):
                        row.update(compute_cls_stats(record, args.img_size, args.fft_topk, args.overlap_top_ratio))
                    else:
                        row.update(compute_no_cls_stats(record, args.img_size, args.fft_topk, args.overlap_top_ratio))
                except Exception as exc:  # keep long runs alive and record unsupported layers.
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)

    write_csv(stats_csv, rows, STATS_FIELDS)
    summary_rows = summarize_rows(rows)
    write_csv(summary_csv, summary_rows)
    print(f"Saved per-image stats to {stats_csv}")
    print(f"Saved summary stats to {summary_csv}")



DEFAULT_CROSS_MODEL_NAMES = (
    "vit_base_patch16_224",
    "deit_base_patch16_224",
    "beit_base_patch16_224",
    "cait_s24_224",
    "pit_s_224",
    "crossvit_15_240",
)

CROSS_METRIC_TYPES = (
    "attention",
    "patchscore",
    "stability",
    "joint_attn_patch",
    "triple",
)

CROSS_STATS_FIELDS = [
    "image_name",
    "block_index",
    "model_a",
    "model_b",
    "resolved_block_a",
    "resolved_block_b",
    "block_name_a",
    "block_name_b",
    "metric_type",
    "jaccard",
    "overlap_a_in_b",
    "overlap_b_in_a",
    "cosine_similarity",
    "random_jaccard_baseline",
    "above_random",
    "density_a",
    "density_b",
    "error",
]


def binary_cosine(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.reshape(-1).astype(np.float64)
    b = mask_b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float((a @ b) / denom)


def expected_random_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    density_a = float(mask_a.mean())
    density_b = float(mask_b.mean())
    denom = density_a + density_b - density_a * density_b
    if denom <= 1e-12:
        return 0.0
    return float((density_a * density_b) / denom)


def overlap_fraction(source: np.ndarray, target: np.ndarray) -> float:
    source_count = int(source.sum())
    if source_count == 0:
        return 0.0
    return float((source & target).sum() / source_count)


def compute_cls_maps_for_cross(
    record: dict[str, Any],
    img_size: int,
    fft_topk: int,
) -> dict[str, np.ndarray]:
    if record.get("mode") == "class_attn":
        tokens = flatten_tokens(record["attn_input"])
        attn_logits = class_attn_record_to_attn_logits(record)
    else:
        tokens = flatten_tokens(record["tokens"])
        attn_logits = qkv_record_to_attn_logits(record)
    num_patches = tokens.size(1) - 1
    if num_patches <= 0 or not is_square_token_count(num_patches):
        raise ValueError(f"Cross-model CLS overlap requires square patch tokens, got token count {tokens.size(1)}.")

    attn_weights = torch.softmax(attn_logits, dim=-1)
    attn_scores = attn_weights[:, :, 0, 1:].mean(dim=1)
    attn_map = normalized_patch_scores(attn_scores, img_size)

    cls_token = tokens[:, 0, :]
    patch_tokens = tokens[:, 1:, :]
    cls_token_expanded = cls_token.unsqueeze(1).expand_as(patch_tokens)
    patch_scores = F.cosine_similarity(patch_tokens, cls_token_expanded, dim=-1)
    patch_map = normalized_patch_scores(patch_scores, img_size)

    fft_scores = last_vit_stable_patch_frequency(tokens=tokens, topk=fft_topk, has_cls_token=True)
    stability_map = normalized_patch_scores(fft_scores, img_size)
    return {
        "attention": attn_map,
        "patchscore": patch_map,
        "stability": stability_map,
    }


def maps_to_cross_masks(maps: dict[str, np.ndarray], top_ratio: float) -> dict[str, np.ndarray]:
    attention = top_ratio_mask(normalize_map(maps["attention"]), top_ratio)
    patchscore = top_ratio_mask(normalize_map(maps["patchscore"]), top_ratio)
    stability = top_ratio_mask(normalize_map(maps["stability"]), top_ratio)
    joint = attention & patchscore
    triple = joint & stability
    return {
        "attention": attention,
        "patchscore": patchscore,
        "stability": stability,
        "joint_attn_patch": joint,
        "triple": triple,
    }


def summarize_cross_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model_a"]),
            str(row["model_b"]),
            int(row["block_index"]),
            str(row["metric_type"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    metrics = [
        "jaccard",
        "overlap_a_in_b",
        "overlap_b_in_a",
        "cosine_similarity",
        "random_jaccard_baseline",
        "above_random",
        "density_a",
        "density_b",
    ]
    for (model_a, model_b, block_index, metric_type), group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "model_a": model_a,
            "model_b": model_b,
            "block_index": block_index,
            "metric_type": metric_type,
            "records": len(group),
            "errors": sum(1 for row in group if row.get("error")),
        }
        for metric in metrics:
            values = np.array([safe_float(row.get(metric)) for row in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                out[f"{metric}_mean"] = float("nan")
                out[f"{metric}_median"] = float("nan")
                out[f"{metric}_p25"] = float("nan")
                out[f"{metric}_p75"] = float("nan")
                continue
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_median"] = float(np.median(values))
            out[f"{metric}_p25"] = float(np.percentile(values, 25))
            out[f"{metric}_p75"] = float(np.percentile(values, 75))
        summary_rows.append(out)
    return summary_rows


def cross_pair_group(model_a: str, model_b: str) -> str:
    pair = {model_a, model_b}
    if pair == {"vit_base_patch16_224", "deit_base_patch16_224"}:
        return "vit_deit"
    if "pit_s_224" in pair:
        return "includes_pit"
    if "crossvit_15_240" in pair:
        return "includes_crossvit"
    return "other_cls"


def summarize_cross_conclusion(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        group = cross_pair_group(str(row["model_a"]), str(row["model_b"]))
        key = (str(row["metric_type"]), group)
        grouped.setdefault(key, []).append(row)

    conclusion_rows = []
    for (metric_type, group), rows in sorted(grouped.items()):
        jaccard_vals = np.array([safe_float(row.get("jaccard_mean")) for row in rows], dtype=np.float64)
        above_vals = np.array([safe_float(row.get("above_random_mean")) for row in rows], dtype=np.float64)
        valid_j = jaccard_vals[np.isfinite(jaccard_vals)]
        valid_a = above_vals[np.isfinite(above_vals)]
        strong_layers = sum(1 for row in rows if safe_float(row.get("above_random_mean")) > 0.02)
        weak_layers = sum(1 for row in rows if safe_float(row.get("above_random_mean")) <= 0.0)
        conclusion_rows.append({
            "metric_type": metric_type,
            "model_pair_group": group,
            "rows": len(rows),
            "jaccard_mean": float(valid_j.mean()) if valid_j.size else float("nan"),
            "above_random_mean": float(valid_a.mean()) if valid_a.size else float("nan"),
            "strong_layers": strong_layers,
            "weak_layers": weak_layers,
        })
    return conclusion_rows


def run_cross_model_overlap(args, image_paths: list[Path], num_classes: int) -> None:
    model_names = args.model_names or DEFAULT_CROSS_MODEL_NAMES
    block_indices = args.block_indices if args.block_indices is not None else (-6, -5, -4, -3, -2, -1)
    stats_csv = Path(args.cross_stats_csv or "outputs/csv/cross_cls_overlap_stats.csv")
    summary_csv = Path(args.cross_summary_csv or "outputs/csv/cross_cls_overlap_summary.csv")
    conclusion_csv = Path(args.cross_conclusion_csv or "outputs/csv/cross_cls_overlap_conclusion.csv")

    model_entries = []
    try:
        for model_name in model_names:
            print(f"Loading {model_name}")
            model = build_model(num_classes=num_classes, model_name=model_name)
            capture = GenericAttentionCapture(model)
            capture.__enter__()
            resolved = [resolve_block_index(idx, len(capture.records)) for idx in block_indices]
            model_entries.append({
                "name": model_name,
                "model": model,
                "capture": capture,
                "resolved": resolved,
            })

        rows: list[dict[str, Any]] = []
        for path in image_paths:
            print(f"Processing {path}")
            original = load_rgb_image(path, args.img_size)
            tensor = preprocess_rgb(original).unsqueeze(0).to(DEVICE)
            per_model_masks: dict[str, dict[int, dict[str, Any]]] = {}

            for entry in model_entries:
                capture = entry["capture"]
                capture.reset()
                with torch.no_grad():
                    _logits = entry["model"].model(tensor)
                layer_masks: dict[int, dict[str, Any]] = {}
                for requested_idx, resolved_idx in zip(block_indices, entry["resolved"]):
                    record = capture.records[resolved_idx]
                    try:
                        maps = compute_cls_maps_for_cross(record, args.img_size, args.fft_topk)
                        masks = maps_to_cross_masks(maps, args.overlap_top_ratio)
                        layer_masks[requested_idx] = {
                            "error": "",
                            "resolved": resolved_idx,
                            "block_name": record["block_name"],
                            "masks": masks,
                        }
                    except Exception as exc:
                        layer_masks[requested_idx] = {
                            "error": f"{type(exc).__name__}: {exc}",
                            "resolved": resolved_idx,
                            "block_name": record["block_name"],
                            "masks": {},
                        }
                per_model_masks[entry["name"]] = layer_masks

            for model_a_idx in range(len(model_entries)):
                for model_b_idx in range(model_a_idx + 1, len(model_entries)):
                    model_a = model_entries[model_a_idx]["name"]
                    model_b = model_entries[model_b_idx]["name"]
                    for block_index in block_indices:
                        data_a = per_model_masks[model_a][block_index]
                        data_b = per_model_masks[model_b][block_index]
                        base = {
                            "image_name": path.name,
                            "block_index": block_index,
                            "model_a": model_a,
                            "model_b": model_b,
                            "resolved_block_a": data_a["resolved"],
                            "resolved_block_b": data_b["resolved"],
                            "block_name_a": data_a["block_name"],
                            "block_name_b": data_b["block_name"],
                        }
                        if data_a["error"] or data_b["error"]:
                            for metric_type in CROSS_METRIC_TYPES:
                                row = dict(base)
                                row.update({
                                    "metric_type": metric_type,
                                    "error": data_a["error"] or data_b["error"],
                                })
                                rows.append(row)
                            continue

                        for metric_type in CROSS_METRIC_TYPES:
                            mask_a = data_a["masks"][metric_type]
                            mask_b = data_b["masks"][metric_type]
                            baseline = expected_random_jaccard(mask_a, mask_b)
                            jac = jaccard(mask_a, mask_b)
                            row = dict(base)
                            row.update({
                                "metric_type": metric_type,
                                "jaccard": jac,
                                "overlap_a_in_b": overlap_fraction(mask_a, mask_b),
                                "overlap_b_in_a": overlap_fraction(mask_b, mask_a),
                                "cosine_similarity": binary_cosine(mask_a, mask_b),
                                "random_jaccard_baseline": baseline,
                                "above_random": jac - baseline,
                                "density_a": float(mask_a.mean()),
                                "density_b": float(mask_b.mean()),
                                "error": "",
                            })
                            rows.append(row)

        write_csv(stats_csv, rows, CROSS_STATS_FIELDS)
        summary_rows = summarize_cross_rows(rows)
        write_csv(summary_csv, summary_rows)
        conclusion_rows = summarize_cross_conclusion(summary_rows)
        write_csv(conclusion_csv, conclusion_rows)
        print(f"Saved cross-model stats to {stats_csv}")
        print(f"Saved cross-model summary to {summary_csv}")
        print(f"Saved cross-model conclusion to {conclusion_csv}")
    finally:
        for entry in model_entries:
            entry["capture"].__exit__(None, None, None)

def tokens_to_heatmap(token_scores: torch.Tensor, img_size: int) -> np.ndarray:
    num_tokens = token_scores.numel()
    grid_size = int(math.sqrt(num_tokens))
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a square number.")
    grid = token_scores.reshape(grid_size, grid_size)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    grid = F.interpolate(
        grid.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    return grid.squeeze().cpu().numpy()


def compute_maps(
    attn_logits: torch.Tensor,
    values: torch.Tensor,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    # attn_logits: [B, H, N, N], values: [B, H, N, d]
    attn_weights = torch.softmax(attn_logits, dim=-1)

    # Map 1: CLS -> patch attention weights
    cls_attn_scores = attn_weights[:, :, 0, 1:].mean(dim=1)[0]
    attn_map = tokens_to_heatmap(cls_attn_scores, img_size)

    # Map 2: CLS query contribution map from (A @ V), per patch token
    cls_patch_contrib = attn_weights[:, :, 0, 1:].unsqueeze(-1) * values[:, :, 1:, :]
    cls_av_scores = cls_patch_contrib.norm(dim=-1).mean(dim=1)[0]
    av_map = tokens_to_heatmap(cls_av_scores, img_size)

    return attn_map, av_map


def compute_patch_score_maps(tokens_last_block: torch.Tensor, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    # tokens_last_block: [B, N, D], use one image
    cls_token = tokens_last_block[:, 0, :]      # [B, D]
    patch_tokens = tokens_last_block[:, 1:, :]  # [B, N_patch, D]

    num_patches = patch_tokens.size(1)
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Patch token count {num_patches} is not a square number.")

    cls_token_expanded = cls_token.unsqueeze(1).expand(-1, num_patches, -1)  # [B, N_patch, D]
    patch_scores = F.cosine_similarity(patch_tokens, cls_token_expanded, dim=-1)  # [B, N_patch]

    score_map_2d = patch_scores[0].reshape(grid_size, grid_size).detach().cpu().numpy()

    upsampled = F.interpolate(
        patch_scores[0].reshape(1, 1, grid_size, grid_size),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    overlay_map = upsampled.squeeze().detach().cpu().numpy()
    return score_map_2d, overlay_map


def normalize_map(score_map: np.ndarray) -> np.ndarray:
    return (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)


def make_fft_stability_overlay(
    image: np.ndarray,
    score_map: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    score = normalize_map(score_map)
    heatmap = plt.get_cmap(HEATMAP_CMAP)(score)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * image + alpha * heatmap
    return np.clip(overlay, 0.0, 1.0), score


def make_image_frequency_overlay(
    image: np.ndarray,
    high_ratio_map: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    high_ratio = normalize_map(np.clip(high_ratio_map, 0.0, 1.0))
    high_ratio = np.clip((high_ratio - 0.5) * 1.8 + 0.5, 0.0, 1.0)
    heatmap = plt.get_cmap(HEATMAP_CMAP)(high_ratio)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * image + alpha * heatmap
    return np.clip(overlay, 0.0, 1.0), high_ratio


def top_ratio_mask(score_map: np.ndarray, top_ratio: float) -> np.ndarray:
    if not (0.0 < top_ratio <= 1.0):
        raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}.")
    threshold = np.quantile(score_map.reshape(-1), 1.0 - top_ratio)
    return score_map >= threshold


def make_mechanism_overlap_overlay(
    image: np.ndarray,
    fft_score_map: np.ndarray,
    attn_map: np.ndarray,
    patch_score_map: np.ndarray,
    top_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    fft_high = top_ratio_mask(normalize_map(fft_score_map), top_ratio)
    attn_high = top_ratio_mask(normalize_map(attn_map), top_ratio)
    patch_high = top_ratio_mask(normalize_map(patch_score_map), top_ratio)
    attn_patch_high = attn_high & patch_high
    all_high = fft_high & attn_high & patch_high

    overlay = image * 0.30
    fft_color = np.array([0.65, 0.0, 1.0], dtype=np.float32)
    attn_patch_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    both_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    overlay[fft_high] = 0.35 * overlay[fft_high] + 0.65 * fft_color
    overlay[attn_patch_high] = 0.32 * overlay[attn_patch_high] + 0.68 * attn_patch_color
    overlay[all_high] = 0.12 * overlay[all_high] + 0.88 * both_color

    attn_patch_count = float(attn_patch_high.sum())
    fft_capture_ratio = float(all_high.sum()) / attn_patch_count if attn_patch_count > 0 else 0.0
    fft_high_count = float(fft_high.sum())
    fft_precision_ratio = float(all_high.sum()) / fft_high_count if fft_high_count > 0 else 0.0
    return (
        np.clip(overlay, 0.0, 1.0),
        fft_high,
        attn_high,
        patch_high,
        fft_capture_ratio,
        fft_precision_ratio,
    )


def draw_binary_contour(
    ax,
    mask: np.ndarray,
    color: str,
    linewidth: float,
    linestyle: str = "solid",
) -> None:
    if mask.any() and (~mask).any():
        ax.contour(mask.astype(np.float32), levels=[0.5], colors=color, linewidths=linewidth, linestyles=linestyle)


def style_colorbar(cbar, label: str) -> None:
    cbar.set_label(label, fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)


def save_triptych(
    original: np.ndarray,
    attn_map: np.ndarray,
    av_map: np.ndarray,
    fft_overlay: np.ndarray,
    image_fft_overlay: np.ndarray,
    image_fft_high_ratio: np.ndarray,
    mechanism_overlap_overlay: np.ndarray,
    fft_high_mask: np.ndarray,
    attn_high_mask: np.ndarray,
    patch_high_mask: np.ndarray,
    fft_capture_ratio: float,
    fft_precision_ratio: float,
    patch_score_map_2d: np.ndarray,
    patch_score_overlay_map: np.ndarray,
    output_path: Path,
    title: str,
    image_fft_cutoff_ratio: float,
) -> None:
    fig, axes_grid = plt.subplots(
        2,
        4,
        figsize=(24, 13.5),
        gridspec_kw={"hspace": 0.46, "wspace": 0.22},
    )
    axes = axes_grid.reshape(-1)

    axes[0].imshow(original)
    axes[0].set_title("Input", fontsize=TITLE_FONT_SIZE)
    axes[0].axis("off")

    axes[1].imshow(original)
    hm1 = axes[1].imshow(attn_map, cmap=HEATMAP_CMAP, alpha=0.6, vmin=0.0, vmax=1.0)
    axes[1].set_title("Attention Scores", fontsize=TITLE_FONT_SIZE)
    axes[1].axis("off")
    cbar1 = fig.colorbar(hm1, ax=axes[1], fraction=0.046, pad=0.04)
    style_colorbar(cbar1, "Normalized score")

    axes[2].imshow(original)
    hm2 = axes[2].imshow(av_map, cmap=HEATMAP_CMAP, alpha=0.6, vmin=0.0, vmax=1.0)
    axes[2].set_title("A@V Scores", fontsize=TITLE_FONT_SIZE)
    axes[2].axis("off")
    cbar2 = fig.colorbar(hm2, ax=axes[2], fraction=0.046, pad=0.04)
    style_colorbar(cbar2, "Normalized score")

    axes[3].imshow(image_fft_overlay)
    axes[3].set_title(
        "Image 2D FFT Low/High\n"
        f"Low to high frequency ratio | cutoff {image_fft_cutoff_ratio:.2f}",
        fontsize=TITLE_FONT_SIZE,
    )
    axes[3].axis("off")
    sm_img_fft = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm_img_fft.set_array(image_fft_high_ratio)
    cbar_img_fft = fig.colorbar(
        sm_img_fft,
        ax=axes[3],
        fraction=0.046,
        pad=0.04,
    )
    style_colorbar(cbar_img_fft, "High-frequency ratio")

    axes[4].imshow(fft_overlay)
    axes[4].set_title("Token FFT Stability Selection", fontsize=TITLE_FONT_SIZE)
    axes[4].axis("off")
    sm = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar_fft = fig.colorbar(
        sm,
        ax=axes[4],
        fraction=0.046,
        pad=0.04,
    )
    style_colorbar(cbar_fft, "Selection frequency")

    axes[5].imshow(original)
    patch_score_cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    patch_score_cmap.set_bad(alpha=0.0)
    patch_score_vis = np.ma.masked_less(
        patch_score_overlay_map,
        PATCH_SCORE_DISPLAY_MIN,
    )
    hm3 = axes[5].imshow(
        patch_score_vis,
        cmap=patch_score_cmap,
        alpha=0.70,
        vmin=PATCH_SCORE_DISPLAY_MIN,
        vmax=PATCH_SCORE_DISPLAY_MAX,
        interpolation="bilinear",
    )
    axes[5].set_title(
        "Patch Score Overlay\n"
        f"Shown: {PATCH_SCORE_DISPLAY_MIN:.1f}-{PATCH_SCORE_DISPLAY_MAX:.1f} | "
        f"Raw min/max: {patch_score_map_2d.min():.3f}/{patch_score_map_2d.max():.3f}",
        fontsize=TITLE_FONT_SIZE,
    )
    axes[5].axis("off")
    cbar3 = fig.colorbar(hm3, ax=axes[5], fraction=0.046, pad=0.04)
    cbar3.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    style_colorbar(cbar3, "Normalized score")

    axes[6].imshow(mechanism_overlap_overlay)
    draw_binary_contour(axes[6], fft_high_mask, color="magenta", linewidth=2.4)
    draw_binary_contour(axes[6], attn_high_mask, color="cyan", linewidth=2.2)
    draw_binary_contour(axes[6], patch_high_mask, color="gold", linewidth=2.2)
    axes[6].set_title(
        "Mechanism Overlap\n"
        f"High A & Patch in high FFT: {fft_capture_ratio:.1%}",
        fontsize=TITLE_FONT_SIZE,
    )
    axes[6].axis("off")
    axes[6].legend(
        handles=[
            Patch(facecolor="magenta", edgecolor="black", label="High FFT"),
            Patch(facecolor="cyan", edgecolor="black", label="High Attn"),
            Patch(facecolor="gold", edgecolor="black", label="High Patch"),
            Patch(facecolor="white", edgecolor="black", label="All high"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        fontsize=OVERLAP_LEGEND_FONT_SIZE,
        framealpha=0.92,
    )
    axes[7].axis("off")

    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95), h_pad=4.0, w_pad=1.4)
    fig.subplots_adjust(hspace=0.46)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    default_output_dir = f"outputs/attention_patchscores_{args.model_name}"
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(default_output_dir)
    dataset_dir = Path(args.dataset_dir)
    annotations_path = Path(args.annotations_path)
    num_classes = resolve_num_classes(args.num_classes, dataset_dir, annotations_path, args.img_size)
    image_paths = list_image_paths(image_dir, args.pattern, args.max_images)

    if args.cross_model_overlap:
        run_cross_model_overlap(args, image_paths, num_classes)
        return

    model = build_model(num_classes=num_classes, model_name=args.model_name)

    if args.no_save_images or args.stats_csv or args.summary_csv or args.block_indices is not None:
        run_csv_stats(args, model, image_paths, output_dir)
        return

    for path in image_paths:
        print(f"Processing {path}")
        original = load_rgb_image(path, args.img_size)
        tensor = preprocess_rgb(original).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, attn_list, value_list, token_list = model(
                tensor,
                return_attn=True,
                return_values=True,
                return_tokens=True,
            )
            pred = logits.argmax(dim=1).item()

        if not attn_list:
            raise ValueError("Model did not return attention tensors.")
        if len(value_list) != len(attn_list):
            raise ValueError(
                f"Hook mismatch: got {len(attn_list)} attention tensors, {len(value_list)} value tensors."
            )
        if not token_list:
            raise ValueError("Model did not return block token outputs.")

        block_idx = resolve_block_index(args.block_index, len(attn_list))
        attn_map, av_map = compute_maps(
            attn_logits=attn_list[block_idx],
            values=value_list[block_idx],
            img_size=args.img_size,
        )
        tokens_for_block = token_list[block_idx]
        fft_maps = last_vit_foreground_background_from_tokens(
            tokens=tokens_for_block,
            topk=args.fft_topk,
            has_cls_token=True,
            img_size=args.img_size,
        )
        fft_score_map = fft_maps["score_map"][0].detach().cpu().numpy()
        fft_overlay, fft_stability_map = make_fft_stability_overlay(
            image=original,
            score_map=fft_score_map,
            alpha=args.fft_alpha,
        )
        image_fft_maps = image_2d_fft_low_high_maps(
            torch.from_numpy(original).permute(2, 0, 1),
            cutoff_ratio=args.image_fft_cutoff_ratio,
            transition_ratio=args.image_fft_transition_ratio,
        )
        image_fft_high_ratio = image_fft_maps["high_ratio"].detach().cpu().numpy()
        image_fft_overlay, image_fft_high_ratio = make_image_frequency_overlay(
            image=original,
            high_ratio_map=image_fft_high_ratio,
            alpha=args.image_fft_alpha,
        )
        patch_score_map_2d, patch_score_overlay_map = compute_patch_score_maps(
            tokens_for_block,
            args.img_size,
        )
        (
            mechanism_overlap_overlay,
            fft_high_mask,
            attn_high_mask,
            patch_high_mask,
            fft_capture_ratio,
            fft_precision_ratio,
        ) = make_mechanism_overlap_overlay(
            image=original,
            fft_score_map=fft_stability_map,
            attn_map=attn_map,
            patch_score_map=patch_score_overlay_map,
            top_ratio=args.overlap_top_ratio,
        )

        output_path = output_dir / f"{path.stem}_block_{block_idx:02d}_patchscores.png"
        title = f"{path.name} | block {block_idx} | predicted class {pred}"
        save_triptych(
            original,
            attn_map,
            av_map,
            fft_overlay,
            image_fft_overlay,
            image_fft_high_ratio,
            mechanism_overlap_overlay,
            fft_high_mask,
            attn_high_mask,
            patch_high_mask,
            fft_capture_ratio,
            fft_precision_ratio,
            patch_score_map_2d,
            patch_score_overlay_map,
            output_path,
            title,
            args.image_fft_cutoff_ratio,
        )
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
