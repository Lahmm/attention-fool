from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from figure1 import compute_patch_scores
from nets import build_whitebox_model
from progressive_attack import ProgressiveRouteDisruptionAttacker
from utils import IMAGENET_MEAN, IMAGENET_STD, SEED, load_data, save_adversarial_images


CASES = ("high-fix", "high-random", "low-fix", "low-random", "random-fix", "random")
CHECKPOINT = "block10"
TOKEN_COUNT = 196
DROP_RATIO = 20 / TOKEN_COUNT


def configure_model(model):
    if model.model_name != "vit_base_patch16_224":
        raise ValueError("this experiment only supports ViT-B/16")
    model.checkpoints = (CHECKPOINT,)
    model.drop_ratios = (DROP_RATIO,)
    model.dynamic_checkpoint = False
    return model


@torch.no_grad()
def clean_boundary_scores(model, images):
    model.eval()
    images = images.to(model.device)
    pixel_mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    pixel_std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    mean = images.new_tensor(model.model_mean).view(1, 3, 1, 1)
    std = images.new_tensor(model.model_std).view(1, 3, 1, 1)
    state = model.begin_progressive_forward((images * pixel_std + pixel_mean - mean) / std)
    prefix = state.context["prefix_tokens"]
    if prefix.size(1) != 1 or state.local_tokens.size(1) != TOKEN_COUNT:
        raise ValueError("expected one CLS and a 14x14 local-token grid")
    tokens = torch.cat((prefix, state.local_tokens), dim=1)
    boundary = None
    for depth, block in enumerate(model.model.blocks, 1):
        tokens = block(tokens)
        if depth == 10:
            boundary = tokens[:, 1:].clone()
    if boundary is None:
        raise ValueError("ViT must contain at least ten blocks")
    reference = model.model.norm(tokens)[:, 0]
    return compute_patch_scores(boundary, reference).detach()


class SelectionAttacker(ProgressiveRouteDisruptionAttacker):
    """Six analysis cases implemented entirely outside the production attack."""

    def __init__(self, model, scores, case, **kwargs):
        if case not in CASES:
            raise ValueError(f"unknown experiment case: {case}")
        if scores.ndim != 2 or scores.size(1) != TOKEN_COUNT or not torch.isfinite(scores).all():
            raise ValueError("expected finite clean scores [B,196]")
        if tuple(model.checkpoints) != (CHECKPOINT,):
            raise ValueError("configure the experiment's block10 boundary first")
        super().__init__(model, dynamic_checkpoint=False, **kwargs)
        self.case = case
        self.scores = scores.detach().to(self.device)
        self.drop_count = max(1, round(TOKEN_COUNT * self.progressive_drop_ratios[0]))
        order = torch.argsort(self.scores, dim=1, stable=True)
        if case.startswith("high"):
            self.pool_indices = order[:, TOKEN_COUNT // 2:]
        elif case.startswith("low"):
            self.pool_indices = order[:, :TOKEN_COUNT // 2]
        else:
            self.pool_indices = torch.arange(TOKEN_COUNT, device=self.device).expand(scores.size(0), -1)
        if self.drop_count > self.pool_indices.size(1):
            raise ValueError("drop count exceeds the candidate pool")
        self.fixed_mask = None
        if case in ("high-fix", "low-fix"):
            chosen = order[:, -self.drop_count:] if case == "high-fix" else order[:, :self.drop_count]
            self.fixed_mask = torch.zeros_like(self.scores, dtype=torch.bool).scatter_(1, chosen, True)
        self.mask_indices = []

    def _sample_random_mask(self, *, batch_size, token_count, ratio, device):
        if batch_size != self.scores.size(0) or token_count != TOKEN_COUNT:
            raise ValueError("mask layout differs from the cached clean scores")
        count = max(1, round(token_count * ratio))
        if count != self.drop_count:
            raise ValueError("drop budget changed during the experiment")
        permutations = torch.stack([torch.randperm(token_count, device=device) for _ in range(batch_size)])
        ranks = torch.empty_like(permutations)
        ranks.scatter_(1, permutations, torch.arange(token_count, device=device).expand_as(permutations))
        pool_ranks = ranks.gather(1, self.pool_indices)
        positions = pool_ranks.topk(count, dim=1, largest=False).indices
        chosen = self.pool_indices.gather(1, positions)
        sampled = torch.zeros_like(self.scores, dtype=torch.bool).scatter_(1, chosen, True)
        if self.case == "random-fix" and self.fixed_mask is None:
            self.fixed_mask = sampled.clone()
        mask = sampled if self.fixed_mask is None else self.fixed_mask
        indices = mask.nonzero(as_tuple=False)[:, 1].reshape(batch_size, count)
        self.mask_indices.append(indices.detach().clone())
        return mask.detach()

    def parameters(self):
        params = super().parameters()
        params.update({
            "experiment": "vit_drop_selection_six_cases",
            "selection_case": self.case,
            "score_input": "clean image; no feature noise, drop, or translation",
            "score_boundary": "block10 output / block11 input; raw local tokens",
            "score_reference": "final model.norm CLS of the same clean image",
            "score_definition": "signed cosine, float32, eps=1e-8; no min-max or absolute value",
            "score_refresh": "once per clean image, before all attack steps",
            "ranking_ties": "ascending (score, patch_index); low first 98, high last 98",
            "candidate_pool_size": self.pool_indices.size(1),
            "fixed_mask_policy": (
                f"highest {self.drop_count} scores" if self.case == "high-fix" else
                f"lowest {self.drop_count} scores" if self.case == "low-fix" else
                "one uniform sample per image" if self.case == "random-fix" else None
            ),
            "mask_refresh": "once per image" if self.case.endswith("fix") else "every step/group",
            "phase_mask_policy": "spatially transform the original mask; preserve drop count",
            "random_alignment": "same full-grid permutation draws, noise coefficients and phase draws across cases",
        })
        return params


def rng_state():
    return torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def same_rng(left, right):
    return len(left[1]) == len(right[1]) and all(
        torch.equal(a, b) for a, b in zip((left[0], *left[1]), (right[0], *right[1]))
    )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def output_root(value):
    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    attack_root = (ROOT / "outputs" / "attack").resolve()
    if not path.is_relative_to(attack_root) or path == attack_root:
        raise ValueError(f"output directory must be a child of {attack_root}")
    return path


def generate(args, root):
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"output directory is not empty; choose a new directory: {root}")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    loader = load_data(args.image_dir, args.annotations_path, args.batch_size, args.num_workers)
    limit = min(args.max_attacked_samples, len(loader.dataset))
    model = configure_model(build_whitebox_model(device=device))
    root.mkdir(parents=True, exist_ok=True)
    source_files = (Path(__file__), ROOT / "progressive_attack.py", ROOT / "nets/vit.py",
                    ROOT / "nets/base.py", ROOT / "utils.py", ROOT / "transfer_eval.py", ROOT / "figure1.py")
    manifest = {
        "status": "running", "seed": SEED, "cases": args.cases, "num_images": limit,
        "checkpoint": CHECKPOINT, "drop_ratio": args.drop_ratio,
        "command": [sys.executable, *sys.argv], "arguments": vars(args),
        "torch": torch.__version__, "device": str(device),
        "model_pretrained_cfg": model.model.pretrained_cfg,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "samples": [],
    }
    for sample in loader.dataset.samples[:limit]:
        manifest["samples"].append({
            "image_name": sample["image_name"], "class_id": sample["class_id"],
            "sha256": hashlib.sha256(sample["image_path"].read_bytes()).hexdigest(),
        })
    write_json(root / "experiment.json", manifest)
    for case in args.cases:
        (root / case / "selection_traces").mkdir(parents=True)
    attacked = 0
    for batch_index, (images, labels, indices) in enumerate(loader):
        if attacked >= limit:
            break
        remaining = limit - attacked
        images, labels, indices = images[:remaining], labels[:remaining], indices[:remaining]
        names = [loader.dataset.samples[i]["image_name"] for i in indices.tolist()]
        scores = clean_boundary_scores(model, images)
        start = rng_state()
        expected_end = None
        for case in args.cases:
            restore_rng(start)
            attacker = SelectionAttacker(
                model, scores, case, drop_ratios=(args.drop_ratio,), device=device,
                steps=args.steps, epsilon=args.epsilon,
                opponent_noise_strength=args.opponent_noise_strength,
                input_diversity_groups=args.input_diversity_groups,
                input_diversity_views_per_group=args.input_diversity_views_per_group,
            )
            adversarial = attacker.attack_batch(images, labels)
            end = rng_state()
            if expected_end is not None and not same_rng(expected_end, end):
                raise RuntimeError(f"random-stream mismatch in {case}")
            expected_end = end
            case_root = root / case
            save_adversarial_images(adversarial, str(case_root), names)
            torch.save({
                "image_names": names, "clean_scores": scores.cpu(),
                "candidate_pool_indices": attacker.pool_indices.cpu(),
                "original_mask_indices": torch.stack(attacker.mask_indices).cpu(),
                "mask_axes": ["step_times_groups", "image", "selected_patch"],
            }, case_root / "selection_traces" / f"batch_{batch_index:05d}.pt")
            params = attacker.parameters()
            params.update({"num_images": attacked + len(names), "generation_batch_size": args.batch_size,
                           "status": "running", "tf32": False, "amp": False})
            write_json(case_root / "attack_params.json", params)
            print(f"{case}: {attacked + len(names)}/{limit}", flush=True)
            del adversarial, attacker
        attacked += len(names)
    for case in args.cases:
        path = root / case / "attack_params.json"
        params = json.loads(path.read_text())
        params["status"] = "complete"
        write_json(path, params)
    manifest.update(status="complete", num_images=attacked, random_streams_aligned=True)
    write_json(root / "experiment.json", manifest)


def evaluate(args, root):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest = json.loads((root / "experiment.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("attack generation is incomplete")
    for case in args.cases:
        case_root = root / case
        params = json.loads((case_root / "attack_params.json").read_text())
        if params.get("status") != "complete":
            raise ValueError(f"attack generation is incomplete for {case}")
        images = list(case_root.glob("adv_*.png"))
        if len(images) != manifest["num_images"]:
            raise ValueError(f"unexpected adversarial image count for {case}")
        subprocess.run([
            sys.executable, str(ROOT / "transfer_eval.py"), "--image-dir", str(case_root),
            "--annotations-path", manifest["arguments"]["annotations_path"],
            "--protocol", "fpr", "--batch-size", str(args.eval_batch_size), "--no-tf32",
        ], cwd=ROOT, check=True)
    summaries, rows_by_case = {}, {}
    for case in args.cases:
        case_root = root / case
        with (case_root / "transfer.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        from transfer_eval import FPR_MODELS
        if {r["target"] for r in rows} != set(FPR_MODELS) or len(rows) != 13:
            raise ValueError(f"expected all FPR13 targets for {case}")
        if any(int(r["total"]) != manifest["num_images"] for r in rows):
            raise ValueError(f"unexpected evaluation sample count for {case}")
        rows_by_case[case] = {r["target"]: float(r["asr"]) * 100 for r in rows}
        summaries[case] = json.loads((case_root / "transfer.json").read_text())["summary"]
    comparisons = []
    for case, metrics in summaries.items():
        reference = "random-fix" if case.endswith("fix") else "random"
        row = {"case": case, "overall_asr_pct": 100 * metrics["overall_asr"],
               "strict_black_box_asr_pct": 100 * metrics["strict_black_box_asr"],
               "reference": reference}
        for metric in ("overall_asr", "strict_black_box_asr"):
            row[f"{metric}_delta_pp"] = (
                100 * (metrics[metric] - summaries[reference][metric]) if reference in summaries else None
            )
        comparisons.append(row)
    with (root / "asr_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    with (root / "asr_per_target.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("target", *args.cases))
        for target in rows_by_case[args.cases[0]]:
            writer.writerow((target, *(rows_by_case[c][target] for c in args.cases)))
    for row in comparisons:
        print(json.dumps(row), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Independent ViT high/low/full-pool drop experiment; six cases.")
    parser.add_argument("--stage", choices=("attack", "evaluate", "all"), default="all")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output-dir", default="outputs/attack/vit_drop_selection_s42")
    parser.add_argument("--image-dir", default=str(ROOT / "data/clean_resized_images"))
    parser.add_argument("--annotations-path", default=str(ROOT / "data/image_name_to_class_id_and_name.json"))
    parser.add_argument("--max-attacked-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drop-ratio", type=float, default=DROP_RATIO)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=16 / 255)
    parser.add_argument("--opponent-noise-strength", type=float, default=0.2)
    parser.add_argument("--input-diversity-groups", type=int, default=10)
    parser.add_argument("--input-diversity-views-per-group", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        parser.error("cases must not repeat")
    if not math.isfinite(args.drop_ratio) or not 0 < args.drop_ratio <= 0.5:
        parser.error("drop-ratio must be finite and in (0, 0.5]")
    if min(args.max_attacked_samples, args.batch_size, args.eval_batch_size, args.threads, args.steps,
           args.input_diversity_groups) <= 0 or args.num_workers < 0:
        parser.error("sample counts, batch sizes, threads, steps and groups must be positive")
    if args.input_diversity_groups * args.input_diversity_views_per_group > 20:
        parser.error("at most 20 differentiable views per step are supported")
    if not math.isfinite(args.epsilon) or args.epsilon < 0:
        parser.error("epsilon must be finite and nonnegative")
    if not math.isfinite(args.opponent_noise_strength) or args.opponent_noise_strength < 0:
        parser.error("noise strength must be finite and nonnegative")
    args.image_dir = str(Path(args.image_dir).expanduser().resolve())
    args.annotations_path = str(Path(args.annotations_path).expanduser().resolve())
    return args


if __name__ == "__main__":
    args = parse_args()
    root = output_root(args.output_dir)
    if args.stage in ("attack", "all"):
        generate(args, root)
    if args.stage in ("evaluate", "all"):
        evaluate(args, root)
