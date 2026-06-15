import argparse
import csv
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from attack import LMDSSAttacker
from main import ANNOTATIONS_PATH, DEFAULT_IMG_SIZE, IMAGE_DIR, parse_model_names
from nets import build_vit_model
from utils import DEVICE, load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Compute cross-model CLS attention/QK guide overlap statistics.")
    parser.add_argument("--models", type=parse_model_names, default=("vit_base_patch16_224", "deit_base_patch16_224", "pit_s_224", "crossvit_15_240"), help="Comma-separated timm model names.")
    parser.add_argument("--attention-guide-types", type=parse_model_names, default=("postsoftmax_cls", "qk_cls"), help="Comma-separated guide types: postsoftmax_cls,qk_cls,qk_all_queries.")
    parser.add_argument("--topk-ratio", type=float, default=0.25, help="Top patch ratio for IoU/Dice/change metrics.")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum samples to process.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--output-csv", default="outputs/attention_overlap_stats.csv")
    return parser.parse_args()


def normalize_weights(weights):
    min_vals = weights.min(dim=1, keepdim=True).values
    max_vals = weights.max(dim=1, keepdim=True).values
    return (weights - min_vals) / (max_vals - min_vals).clamp_min(1e-12)


def top_mask(scores, ratio):
    k = max(1, min(scores.size(1), int(round(scores.size(1) * ratio))))
    idx = torch.topk(scores, k=k, dim=1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(1, idx, True)


def entropy(scores):
    prob = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (-(prob * prob.clamp_min(1e-12).log()).sum(dim=1) / torch.log(torch.tensor(prob.size(1), device=prob.device))).detach().cpu()


def compactness(mask):
    n = mask.size(1)
    grid = int(n ** 0.5)
    if grid * grid != n:
        return torch.full((mask.size(0),), float("nan"), device=mask.device)
    coords = torch.stack(torch.meshgrid(torch.arange(grid, device=mask.device), torch.arange(grid, device=mask.device), indexing="ij"), dim=-1).view(-1, 2).float()
    vals = []
    for row in mask:
        pts = coords[row]
        if pts.numel() == 0:
            vals.append(torch.tensor(float("nan"), device=mask.device))
        else:
            center = pts.mean(dim=0, keepdim=True)
            vals.append(((pts - center).pow(2).sum(dim=1).sqrt().mean() / max(grid - 1, 1)))
    return torch.stack(vals).detach().cpu()


def rank_corr(a, b):
    ar = torch.argsort(torch.argsort(a, dim=1), dim=1).float()
    br = torch.argsort(torch.argsort(b, dim=1), dim=1).float()
    return F.cosine_similarity(ar - ar.mean(dim=1, keepdim=True), br - br.mean(dim=1, keepdim=True), dim=1).detach().cpu()


def pair_metrics(a, b, ratio):
    ma = top_mask(a, ratio)
    mb = top_mask(b, ratio)
    inter = (ma & mb).sum(dim=1).float()
    union = (ma | mb).sum(dim=1).float().clamp_min(1.0)
    denom = (ma.sum(dim=1) + mb.sum(dim=1)).float().clamp_min(1.0)
    return {
        "topk_iou": (inter / union).detach().cpu(),
        "dice": (2.0 * inter / denom).detach().cpu(),
        "cosine": F.cosine_similarity(a, b, dim=1).detach().cpu(),
        "rank_correlation": rank_corr(a, b),
    }


def collect_guides(models, images, guide_type, ratio):
    guides = {}
    helper = LMDSSAttacker(next(iter(models.values())), attention_guide_type=guide_type, device=DEVICE)
    for name, model in models.items():
        score = helper._collect_cls_attention_scores(model, images, attention_guide_type=guide_type)
        if score is not None:
            guides[name] = normalize_weights(score.detach())
    return guides


def main():
    args = parse_args()
    if not (0.0 < args.topk_ratio < 1.0):
        raise ValueError("--topk-ratio must be in (0, 1).")
    dataloader, num_classes = load_data(
        image_dir_arg=args.image_dir,
        annotations_path_arg=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        img_size=args.img_size,
    )
    models = {name: build_vit_model(num_classes=num_classes, model_name=name) for name in args.models}
    for model in models.values():
        model.eval()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "model_pair", "guide_type", "topk_iou", "dice", "cosine", "rank_correlation", "entropy_a", "entropy_b", "compactness_a", "compactness_b", "random_topk_iou"]
    written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for images, _labels, indices in tqdm(dataloader, desc="attention overlap"):
            if written >= args.max_samples:
                break
            remaining = args.max_samples - written
            images = images[:remaining].to(DEVICE)
            indices = indices[:remaining]
            batch_n = images.size(0)
            for guide_type in args.attention_guide_types:
                guides = collect_guides(models, images, guide_type, args.topk_ratio)
                for name_a, name_b in combinations(guides.keys(), 2):
                    a = guides[name_a]
                    b = guides[name_b]
                    if a.size(1) != b.size(1):
                        continue
                    metrics = pair_metrics(a, b, args.topk_ratio)
                    ent_a = entropy(a)
                    ent_b = entropy(b)
                    comp_a = compactness(top_mask(a, args.topk_ratio))
                    comp_b = compactness(top_mask(b, args.topk_ratio))
                    rand = normalize_weights(torch.rand_like(a))
                    rand_iou = pair_metrics(a, rand, args.topk_ratio)["topk_iou"]
                    for row_idx in range(batch_n):
                        writer.writerow({
                            "image_id": int(indices[row_idx].item()),
                            "model_pair": f"{name_a}|{name_b}",
                            "guide_type": guide_type,
                            "topk_iou": float(metrics["topk_iou"][row_idx]),
                            "dice": float(metrics["dice"][row_idx]),
                            "cosine": float(metrics["cosine"][row_idx]),
                            "rank_correlation": float(metrics["rank_correlation"][row_idx]),
                            "entropy_a": float(ent_a[row_idx]),
                            "entropy_b": float(ent_b[row_idx]),
                            "compactness_a": float(comp_a[row_idx]),
                            "compactness_b": float(comp_b[row_idx]),
                            "random_topk_iou": float(rand_iou[row_idx]),
                        })
            written += batch_n
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
