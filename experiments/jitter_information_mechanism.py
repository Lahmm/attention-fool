"""Measure the transferable information contributed by Jitter augmentation.

The experiment separates three questions which are otherwise confounded by ASR:

1. Do nine stochastic views span diverse but repeatable source gradients?
2. Does an augmentation add a target-aligned direction outside the span of the
   other augmentations?
3. Do DIM, TI, and low/mid-frequency rotation preserve or amplify that
   target-aligned information?

The default protocol uses the same total of nine forward views for every
condition: nine copies for a single method and three copies per method for the
three-method mixture.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from causal_analysis import _target_normalize, build_baseline, seed_all, selected_batches
from gradient_analysis import fft_project
from main import ANNOTATIONS_PATH, IMAGE_DIR, parse_model_names
from nets import build_vit_model
from utils import DEVICE, load_data


PROTOCOL = "jitter_information_mechanism_v1"
METHODS = {
    "dropout": (("dropout",), 9),
    "jitter": (("jitter",), 9),
    "lowmid": (("lowmid_shift",), 9),
    "mixed": (("dropout", "jitter", "lowmid_shift"), 3),
}
BAND_GROUPS = {
    "low": (0, 1, 2),
    "mid": (3, 4, 5),
    "high": (6, 7),
}
DEFAULT_TARGETS = (
    "deit_base_patch16_224",
    "cait_s24_224",
    "inception_v3",
    "resnet101",
)
MAX_NUMPY_SEED = 2**32 - 1


def derive_seed(*values: int) -> int:
    """Combine integer coordinates into a deterministic NumPy-compatible seed."""
    seed = 0
    for value in values:
        seed = (seed * 1_000_003 + int(value)) % MAX_NUMPY_SEED
    return seed


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


def tensor_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    af, bf = a.flatten(1), b.flatten(1)
    return (af * bf).sum(1) / (af.norm(dim=1) * bf.norm(dim=1)).clamp_min(eps)


def normalized_dot(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """Target directional derivative normalized only by target energy.

    Unlike cosine, this retains the magnitude of the source update direction.
    """
    af, bf = a.flatten(1), b.flatten(1)
    return (af * bf).sum(1) / bf.square().sum(1).clamp_min(eps)


def effective_rank(term_grads: tuple[torch.Tensor, ...], eps: float = 1e-12) -> torch.Tensor:
    """Entropy effective rank of the per-image normalized view-gradient Gram matrix."""
    terms = torch.stack([term.flatten(1) for term in term_grads], dim=1)
    terms = F.normalize(terms, dim=2, eps=eps)
    gram = terms @ terms.transpose(1, 2)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum(1, keepdim=True).clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(1)
    return entropy.exp()


def view_consensus(term_grads: tuple[torch.Tensor, ...]) -> torch.Tensor:
    terms = torch.stack(term_grads, dim=1)
    mean = terms.mean(1)
    return torch.stack([tensor_cos(terms[:, index], mean) for index in range(terms.size(1))], 1).mean(1)


def span_residual(
    vector: torch.Tensor,
    basis: tuple[torch.Tensor, ...],
    ridge: float = 1e-8,
) -> torch.Tensor:
    """Remove the per-image least-squares projection onto a small gradient span."""
    vf = vector.flatten(1)
    matrix = torch.stack([item.flatten(1) for item in basis], dim=2)
    gram = matrix.transpose(1, 2) @ matrix
    scale = gram.diagonal(dim1=1, dim2=2).mean(1, keepdim=True).clamp_min(1e-20)
    eye = torch.eye(gram.size(1), device=gram.device, dtype=gram.dtype).unsqueeze(0)
    rhs = matrix.transpose(1, 2) @ vf.unsqueeze(2)
    coefficients = torch.linalg.solve(gram + ridge * scale.unsqueeze(2) * eye, rhs)
    projection = (matrix @ coefficients).squeeze(2)
    return (vf - projection).view_as(vector)


def band_energy_ratios(gradient: torch.Tensor) -> dict[str, torch.Tensor]:
    total = gradient.square().flatten(1).sum(1).clamp_min(1e-20)
    result = {}
    for name, bands in BAND_GROUPS.items():
        projected = sum((fft_project(gradient, band) for band in bands), torch.zeros_like(gradient))
        result[name] = projected.square().flatten(1).sum(1) / total
    return result


def target_gradient(model, pixels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probe = pixels.detach().requires_grad_(True)
    logits = model(_target_normalize(model, probe), return_attn=False)
    return torch.autograd.grad(F.cross_entropy(logits, labels), probe)[0].detach()


def _fixed_dim_params(attacker, pixels: torch.Tensor, seed: int):
    seed_all(seed)
    return attacker._sample_dim_params(pixels)


def gradient_ensemble(
    attacker,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    guide: torch.Tensor,
    method_name: str,
    seed: int,
    *,
    dim: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    methods, copies = METHODS[method_name]
    fixed_params = _fixed_dim_params(attacker, pixels, seed + 104729) if dim else None
    seed_all(seed)
    with attacker_options(
        attacker,
        guide_aug=True,
        guide_aug_methods=methods,
        guide_aug_copies=copies,
        input_diversity=dim,
        dim_mode="full-fixed" if dim else "full-random",
        _fixed_dim_params=fixed_params,
        dim_adjoint_echo=False,
        lowmid_grad_tuning=False,
        lowmid_dss_filter=False,
    ):
        probe = pixels.detach().requires_grad_(True)
        return attacker._attack_grad_terms(probe, labels, guide)


def processed_variants(attacker, gradient: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {"dim": gradient}
    ti = attacker._smooth_grad(gradient)
    result["dim_ti"] = ti
    with attacker_options(attacker, lowmid_grad_tuning=True, lowmid_grad_rotation_strength=0.5):
        result["dim_ti_lowmid"] = attacker._tune_lowmid_gradient(ti)
    return result


def append_values(rows: list[dict], *, batch: int, method: str, stage: str, metric: str,
                  values: torch.Tensor, model: str = "source") -> None:
    for image, value in enumerate(values.detach().cpu().tolist()):
        rows.append({
            "batch": batch,
            "image": image,
            "method": method,
            "stage": stage,
            "model": model,
            "metric": metric,
            "value": float(value),
        })


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        key = (row["method"], row["stage"], row["model"], row["metric"])
        groups.setdefault(key, []).append(float(row["value"]))
    result = []
    for (method, stage, model, metric), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        result.append({
            "method": method,
            "stage": stage,
            "model": model,
            "metric": metric,
            "n": len(values),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(values) > 1 else 0.0,
            "sem": float(array.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
        })
    return result


def _summary_value(summary: list[dict], method: str, stage: str, model: str, metric: str) -> float:
    for row in summary:
        if (row["method"], row["stage"], row["model"], row["metric"]) == (method, stage, model, metric):
            return float(row["mean"])
    raise KeyError((method, stage, model, metric))


def build_conclusion(summary: list[dict], target_models: tuple[str, ...]) -> str:
    target = "target_mean"
    lines = [
        "# Jitter 额外信息与联合机制分析",
        "",
        "## 数学定义",
        "",
        "对增强分布 A 的 9 个视图，攻击梯度为 g_A = (1/9) Σ_k J_A,k^T ∇L(T_A,k(x))。有效秩衡量视图梯度张成空间的维数，共识度与重复采样余弦衡量这些方向是否可稳定平均。",
        "",
        "令 S = span{g_dropout, g_lowmid}，Jitter 的独有分量定义为 r_J = (I-P_S)g_J。只有当 r_J 与黑盒梯度 g_t 保持正对齐时，才能解释为额外可迁移信息，而不是随机噪声。",
        "",
        "## 汇总证据",
        "",
    ]
    for method in METHODS:
        rank = _summary_value(summary, method, "raw", "source", "effective_rank")
        consensus = _summary_value(summary, method, "raw", "source", "view_consensus")
        stability = _summary_value(summary, method, "raw", "source", "resample_cos")
        raw_cos = _summary_value(summary, method, "raw", target, "target_cos")
        final_cos = _summary_value(summary, method, "dim_ti_lowmid", target, "target_cos")
        lines.append(
            f"- {method}: effective-rank={rank:.4f}, view-consensus={consensus:.4f}, "
            f"resample-cos={stability:.4f}, raw target-cos={raw_cos:.6f}, "
            f"DIM+TI+LowMid target-cos={final_cos:.6f}。"
        )
    residual_ratio = _summary_value(summary, "jitter_residual", "raw", "source", "norm_ratio")
    residual_cos = _summary_value(summary, "jitter_residual", "raw", target, "target_cos")
    residual_positive = _summary_value(summary, "jitter_residual", "raw", target, "positive_target_dot")
    lines.extend([
        "",
        "## Jitter 独有分量",
        "",
        f"- 去除 Dropout 与 LowMid 张成子空间后，Jitter 保留 {residual_ratio:.2%} 的梯度范数。",
        f"- 该残差与 {len(target_models)} 个目标模型平均梯度的 cosine 为 {residual_cos:.6f}，target-dot 为正的比例为 {residual_positive:.2%}。",
        "",
        "## 判读规则",
        "",
    ])
    if residual_cos > 0 and residual_positive > 0.5:
        lines.append("Jitter 的优势得到“子空间新颖性 + 黑盒对齐”共同支持：它并非仅复现 Dropout/LowMid，而是提供了二者未覆盖的可迁移方向。")
    else:
        lines.append("当前数据不支持把 Jitter 优势归因于额外可迁移子空间；应扩大样本或重新检查增强与后处理的交互。")
    lines.append("DIM、TI 与低中频旋转的逐级 target-cos 用于判断该信息如何与几何、平移和频率先验联合；不能仅凭源梯度范数作结论。")
    return "\n".join(lines) + "\n"


def run(args) -> None:
    seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loader, num_classes = load_data(
        args.image_dir, args.annotations_path, args.batch_size, args.num_workers, 2, args.img_size
    )
    source, attacker = build_baseline(num_classes)
    attacker.ti_sigma = args.ti_sigma
    attacker._ti_kernel = attacker._build_ti_kernel(args.ti_sigma) if args.ti_sigma > 0 else None
    rows: list[dict] = []
    processed = 0

    for batch_index, (images, labels, _indices) in enumerate(selected_batches(args, source, loader)):
        pixels = attacker._denormalize(images).detach()
        guide = attacker._build_guide_pixel_map(images, pixels.size(-1))
        raw_means, dim_variants = {}, {}
        for method_index, method in enumerate(METHODS):
            base_seed = derive_seed(args.seed, batch_index * 10_007, method_index * 101)
            raw, raw_terms = gradient_ensemble(attacker, pixels, labels, guide, method, base_seed, dim=False)
            raw_repeat, _ = gradient_ensemble(attacker, pixels, labels, guide, method, base_seed + 1, dim=False)
            dim_grad, _ = gradient_ensemble(attacker, pixels, labels, guide, method, base_seed, dim=True)
            raw_means[method] = raw
            dim_variants[method] = processed_variants(attacker, dim_grad)
            append_values(rows, batch=batch_index, method=method, stage="raw", metric="effective_rank",
                          values=effective_rank(raw_terms))
            append_values(rows, batch=batch_index, method=method, stage="raw", metric="view_consensus",
                          values=view_consensus(raw_terms))
            append_values(rows, batch=batch_index, method=method, stage="raw", metric="resample_cos",
                          values=tensor_cos(raw, raw_repeat))
            for band, values in band_energy_ratios(raw).items():
                append_values(rows, batch=batch_index, method=method, stage="raw", metric=f"energy_{band}", values=values)

        jitter_residual = span_residual(raw_means["jitter"], (raw_means["dropout"], raw_means["lowmid"]))
        residual_ratio = jitter_residual.flatten(1).norm(dim=1) / raw_means["jitter"].flatten(1).norm(dim=1).clamp_min(1e-20)
        append_values(rows, batch=batch_index, method="jitter_residual", stage="raw", metric="norm_ratio", values=residual_ratio)

        target_grads = []
        for model_name in args.target_models:
            model = build_vit_model(num_classes=num_classes, model_name=model_name)
            model.eval()
            target_grads.append(target_gradient(model, pixels, labels))
            del model
            release_cuda()
        target_mean = torch.stack([F.normalize(item.flatten(1), dim=1).view_as(item) for item in target_grads]).mean(0)

        for method in METHODS:
            for stage, gradient in (("raw", raw_means[method]), *dim_variants[method].items()):
                per_target_cos = torch.stack([tensor_cos(gradient, target) for target in target_grads]).mean(0)
                per_target_dot = torch.stack([normalized_dot(gradient, target) for target in target_grads]).mean(0)
                append_values(rows, batch=batch_index, method=method, stage=stage, model="target_mean",
                              metric="target_cos", values=per_target_cos)
                append_values(rows, batch=batch_index, method=method, stage=stage, model="target_mean",
                              metric="target_normalized_dot", values=per_target_dot)
        append_values(rows, batch=batch_index, method="jitter_residual", stage="raw", model="target_mean",
                      metric="target_cos", values=tensor_cos(jitter_residual, target_mean))
        target_dots = torch.stack([(jitter_residual * target).flatten(1).sum(1) for target in target_grads])
        append_values(rows, batch=batch_index, method="jitter_residual", stage="raw", model="target_mean",
                      metric="positive_target_dot", values=(target_dots > 0).float().mean(0))

        processed += pixels.size(0)
        print(f"processed {processed}/{args.max_samples}")
        del pixels, guide, raw_means, dim_variants, target_grads, target_mean
        release_cuda()

    summary = summarize(rows)
    raw_path = output / "jitter_information_metrics.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("batch", "image", "method", "stage", "model", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output / "jitter_information_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "stage", "model", "metric", "n", "mean", "std", "sem"))
        writer.writeheader()
        writer.writerows(summary)
    report = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "samples": processed,
        "views_per_condition": 9,
        "methods": {key: {"augmentations": list(value[0]), "copies": value[1]} for key, value in METHODS.items()},
        "target_models": list(args.target_models),
        "ti_sigma": args.ti_sigma,
        "summary": summary,
    }
    (output / "jitter_information_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "jitter_information_conclusion_zh.md").write_text(
        build_conclusion(summary, args.target_models), encoding="utf-8"
    )
    print(f"wrote {output / 'jitter_information_conclusion_zh.md'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/analysis/jitter_information_mechanism")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--target-models", type=parse_model_names, default=DEFAULT_TARGETS)
    parser.add_argument("--ti-sigma", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
