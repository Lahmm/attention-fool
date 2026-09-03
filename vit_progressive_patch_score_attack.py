"""Standalone ViT progressive patch-drop routing attack.

This module intentionally does not alter the production attack entry points.
It implements the experimental progressive variant for a ViT-B/16 white-box:
each group first builds a three-checkpoint mask schedule on the current attack
iterate, then replays that schedule on original and phase-shifted views while
optimizing the pixel-space adversarial example.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from attack import PatchScoreAttacker
from gradient_replay import GradientReplay
from nets import build_whitebox_model
from utils import DEVICE, load_data, save_adversarial_images


MODEL_NAME = "vit_base_patch16_224"
IMAGE_DIR = "data/clean_resized_images"
ANNOTATIONS_PATH = "data/image_name_to_class_id_and_name.json"
DEFAULT_CHECKPOINTS = (3, 6, 9)
DEFAULT_DROP_RATIOS = (0.05, 0.05, 0.05)
PROGRESSIVE_PATCH_SELECTORS = ("patch_score", "random")


@dataclass(frozen=True)
class ProgressiveMaskSchedule:
    """Detached local-token masks generated along one forward trajectory."""

    checkpoints: tuple[int, ...]
    masks: tuple[torch.Tensor, ...]
    counts: tuple[int, ...]
    grid_size: tuple[int, int]

    def validate(self, *, batch_size: int, token_count: int) -> None:
        if len(self.checkpoints) != len(self.masks):
            raise ValueError("checkpoint and mask counts do not match")
        if len(self.masks) != len(self.counts):
            raise ValueError("mask and count metadata do not match")
        if self.grid_size[0] * self.grid_size[1] != token_count:
            raise ValueError("schedule grid size does not match token count")
        for mask, count in zip(self.masks, self.counts):
            if mask.shape != (batch_size, token_count):
                raise ValueError(
                    "ViT progressive masks must have shape "
                    f"[{batch_size}, {token_count}], got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                raise ValueError("progressive masks must be boolean")
            if not torch.equal(mask.sum(dim=1), torch.full((batch_size,), count, device=mask.device)):
                raise ValueError("progressive mask counts differ within a batch")


class ViTProgressivePatchScoreAttacker(PatchScoreAttacker):
    """ViT-only progressive hidden-token intervention attack.

    The parent class supplies the pixel update, momentum, replay, phase and
    opponent-noise primitives.  This subclass only replaces the production
    final-layer pixel-drop view construction with a progressive three-block
    ViT forward.
    """

    def __init__(
        self,
        model,
        *,
        checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
        drop_ratios: tuple[float, ...] = DEFAULT_DROP_RATIOS,
        patch_selector: str = "patch_score",
        score_cls_noise_strength: float = 0.2,
        opponent_noise_strength: float = 0.2,
        **kwargs,
    ) -> None:
        if getattr(model, "model_name", None) != MODEL_NAME:
            raise ValueError(
                "vit_progressive_patch_score_attack.py only supports "
                f"{MODEL_NAME!r}, got {getattr(model, 'model_name', None)!r}."
            )
        checkpoints = tuple(int(value) for value in checkpoints)
        drop_ratios = tuple(float(value) for value in drop_ratios)
        self._validate_progressive_config(checkpoints, drop_ratios)
        if patch_selector not in PROGRESSIVE_PATCH_SELECTORS:
            raise ValueError(
                "progressive patch_selector must be one of "
                f"{PROGRESSIVE_PATCH_SELECTORS}, got {patch_selector!r}"
            )
        if score_cls_noise_strength < 0:
            raise ValueError("score_cls_noise_strength must be non-negative")
        if opponent_noise_strength < 0:
            raise ValueError("opponent_noise_strength must be non-negative")
        self.progressive_checkpoints = checkpoints
        self.progressive_drop_ratios = drop_ratios
        self.progressive_patch_selector = patch_selector
        self.score_cls_noise_strength = float(score_cls_noise_strength)
        self.opponent_noise_strength = float(opponent_noise_strength)
        self._progressive_mask_counts = tuple()
        self._progressive_schedule_count = 0
        self._progressive_checkpoint_selection_count = 0

        # The parent attack method is retained only for its update/gradient
        # machinery. Its final-layer selector and view iterator are not used by
        # this subclass, but the phase-pair settings remain parent invariants.
        attack_method = kwargs.pop("attack_method", "original_score_postdrop_phase_pair")
        if attack_method != "original_score_postdrop_phase_pair":
            raise ValueError(
                "the progressive ViT attack requires "
                "attack_method='original_score_postdrop_phase_pair'"
            )
        views_per_group = kwargs.pop("input_diversity_views_per_group", 2)
        if views_per_group != 2:
            raise ValueError("the progressive ViT attack requires two views per group")
        kwargs["attack_method"] = attack_method
        kwargs["input_diversity_views_per_group"] = views_per_group
        kwargs["patch_selector"] = patch_selector
        # Progressive ratios are independent per-checkpoint budgets. Do not
        # feed their sum into the parent's unused single-dropout budget, whose
        # [0, 1] validation would reject otherwise valid schedules.
        kwargs["patch_dropout_ratio"] = 0.0
        kwargs.setdefault("patch_dropout_score_mode", "high")
        kwargs.setdefault("patch_dropout_sampling_mode", "random")
        kwargs.setdefault("post_dropout_feature_noise_type", "opponent_projected")
        kwargs.setdefault("post_dropout_feature_noise_strength", opponent_noise_strength)
        kwargs.setdefault("guide_aug_strength", opponent_noise_strength)
        # Score noise is generated explicitly once per checkpoint with a
        # distinct replay event.  The parent score helper is not used here.
        kwargs["token_score_cls_noise"] = False
        super().__init__(model, **kwargs)

    @staticmethod
    def _validate_progressive_config(
        checkpoints: tuple[int, ...],
        drop_ratios: tuple[float, ...],
    ) -> None:
        if len(checkpoints) != 3:
            raise ValueError("exactly three progressive checkpoints are required")
        if len(drop_ratios) != 3:
            raise ValueError("exactly three progressive drop ratios are required")
        if any(checkpoint < 1 or checkpoint > 11 for checkpoint in checkpoints):
            raise ValueError("progressive checkpoints must be in [1, 11]")
        if tuple(sorted(checkpoints)) != checkpoints or len(set(checkpoints)) != 3:
            raise ValueError("progressive checkpoints must be strictly increasing")
        if any(ratio <= 0.0 or ratio > 0.5 for ratio in drop_ratios):
            raise ValueError("each progressive drop ratio must satisfy 0 < ratio <= 0.5")

    @property
    def _vit(self):
        base = self._vit_base_model(self.model)
        if len(base.blocks) != 12:
            raise ValueError(f"expected a 12-block ViT-B/16, got {len(base.blocks)} blocks")
        return base

    @staticmethod
    def _apply_local_mask(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.size(1) < 2:
            raise ValueError(f"expected CLS + local tokens, got {tuple(tokens.shape)}")
        if mask.shape != (tokens.size(0), tokens.size(1) - 1):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match local tokens "
                f"{tuple(tokens[:, 1:].shape)}"
            )
        local = torch.where(mask.unsqueeze(-1), torch.zeros_like(tokens[:, 1:]), tokens[:, 1:])
        return torch.cat((tokens[:, :1], local), dim=1)

    def _score_at_checkpoint(
        self,
        tokens: torch.Tensor,
        *,
        checkpoint: int,
    ) -> torch.Tensor:
        local = tokens[:, 1:]
        cls = tokens[:, :1]
        if self.score_cls_noise_strength == 0:
            score_cls = cls
        else:
            token_rms = local.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            noise = self._randn_like(cls, f"progressive_score_cls_block{checkpoint}")
            score_cls = cls + self.score_cls_noise_strength * token_rms * noise
        return F.cosine_similarity(local, score_cls.expand_as(local), dim=-1)

    def _sample_high_mask(
        self,
        scores: torch.Tensor,
        ratio: float,
        *,
        checkpoint: int,
    ) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError(f"scores must have shape [B,N], got {tuple(scores.shape)}")
        batch_size, token_count = scores.shape
        candidate_count = max(1, token_count // 2)
        drop_count = max(1, int(round(token_count * ratio)))
        if drop_count > candidate_count:
            raise ValueError(
                f"drop ratio {ratio} selects {drop_count} tokens, but high half has "
                f"only {candidate_count} candidates"
            )
        candidates = torch.topk(scores, candidate_count, dim=1, largest=True).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for batch_index in range(batch_size):
            if self._gradient_replay is None:
                order = torch.randperm(candidate_count, device=scores.device)
            else:
                order = self._gradient_replay.randperm(
                    candidate_count,
                    f"progressive_drop_block{checkpoint}",
                    batch_index,
                    device=scores.device,
                )
            selected = candidates[batch_index, order[:drop_count]]
            mask[batch_index, selected] = True
        return mask.detach()

    def _sample_random_mask(
        self,
        *,
        batch_size: int,
        token_count: int,
        ratio: float,
        checkpoint: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample the checkpoint budget uniformly from all local patches."""
        drop_count = max(1, int(round(token_count * ratio)))
        mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
        for batch_index in range(batch_size):
            if self._gradient_replay is None:
                order = torch.randperm(token_count, device=device)
            else:
                order = self._gradient_replay.randperm(
                    token_count,
                    f"progressive_random_drop_block{checkpoint}",
                    batch_index,
                    device=device,
                )
            mask[batch_index, order[:drop_count]] = True
        return mask.detach()

    def _build_mask_schedule(self, pixels: torch.Tensor) -> ProgressiveMaskSchedule:
        """Run one no-grad progressive trajectory on the current attack iterate."""
        base = self._vit
        with torch.no_grad():
            tokens = self._embed_vit_tokens(base, pixels.detach())
            token_count = tokens.size(1) - 1
            grid_size = tuple(int(value) for value in base.patch_embed.grid_size)
            if grid_size[0] * grid_size[1] != token_count:
                raise ValueError(
                    f"ViT patch grid {grid_size} does not match token count {token_count}"
                )
            masks: list[torch.Tensor] = []
            counts: list[int] = []
            start = 0
            for checkpoint, ratio in zip(self.progressive_checkpoints, self.progressive_drop_ratios):
                tokens = self._run_vit_blocks(base.blocks, tokens, start, checkpoint)
                if self.progressive_patch_selector == "patch_score":
                    scores = self._score_at_checkpoint(tokens, checkpoint=checkpoint)
                    mask = self._sample_high_mask(scores, ratio, checkpoint=checkpoint)
                else:
                    mask = self._sample_random_mask(
                        batch_size=tokens.size(0),
                        token_count=token_count,
                        ratio=ratio,
                        checkpoint=checkpoint,
                        device=tokens.device,
                    )
                tokens = self._apply_local_mask(tokens, mask)
                masks.append(mask)
                count = int(mask.sum(dim=1)[0].item())
                counts.append(count)
                start = checkpoint
        schedule = ProgressiveMaskSchedule(
            checkpoints=self.progressive_checkpoints,
            masks=tuple(masks),
            counts=tuple(counts),
            grid_size=grid_size,
        )
        schedule.validate(batch_size=pixels.size(0), token_count=token_count)
        self._progressive_mask_counts = schedule.counts
        self._progressive_schedule_count += 1
        self._progressive_checkpoint_selection_count += len(schedule.masks)
        return schedule

    @staticmethod
    def _mask_to_image(
        mask: torch.Tensor,
        grid_size: tuple[int, int],
        height: int,
        width: int,
    ) -> torch.Tensor:
        return F.interpolate(
            mask[:, None].to(torch.float32).view(mask.size(0), 1, *grid_size),
            size=(height, width),
            mode="nearest",
        )

    def _phase_mask_schedule(
        self,
        schedule: ProgressiveMaskSchedule,
        phases: list[tuple[int, int]],
        *,
        height: int,
        width: int,
    ) -> ProgressiveMaskSchedule:
        """Shift each 14x14 mask in image space and restore its count."""
        shifted_masks: list[torch.Tensor] = []
        for checkpoint, mask, count in zip(
            schedule.checkpoints, schedule.masks, schedule.counts
        ):
            image_mask = self._mask_to_image(mask, schedule.grid_size, height, width)
            shifted_image = self._apply_samplewise_phase_shifts(image_mask, phases)
            occupancy = F.adaptive_avg_pool2d(shifted_image, schedule.grid_size).flatten(1)
            indices = torch.argsort(occupancy, dim=1, descending=True, stable=True)[:, :count]
            shifted = torch.zeros_like(mask)
            shifted.scatter_(1, indices, True)
            shifted_masks.append(shifted.detach())
        return ProgressiveMaskSchedule(
            checkpoints=schedule.checkpoints,
            masks=tuple(shifted_masks),
            counts=schedule.counts,
            grid_size=schedule.grid_size,
        )

    def _forward_with_schedule(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        schedule: ProgressiveMaskSchedule,
    ) -> torch.Tensor:
        base = self._vit
        state = self.model.prepare_attack_feature_state(self._normalize(pixels))
        state.validate()
        local_tokens = state.local_tokens
        union_mask = torch.stack(schedule.masks, dim=0).any(dim=0)
        if self.post_dropout_phase_token_noise:
            noise = self._strict_opponent_feature_noise(state)
            local_tokens = torch.where(
                (~union_mask).unsqueeze(-1),
                local_tokens + noise,
                local_tokens,
            )
        tokens = torch.cat((state.context["prefix_tokens"], local_tokens), dim=1)
        start = 0
        for checkpoint, mask in zip(schedule.checkpoints, schedule.masks):
            tokens = self._run_vit_blocks(base.blocks, tokens, start, checkpoint)
            tokens = self._apply_local_mask(tokens, mask)
            start = checkpoint
        tokens = self._run_vit_blocks(base.blocks, tokens, start, len(base.blocks))
        logits = base.forward_head(base.norm(tokens))
        return F.cross_entropy(logits, labels)

    def _iter_attack_losses(self, pixels: torch.Tensor, labels: torch.Tensor):
        for group_index in range(self.input_diversity_groups):
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(group=group_index, view=-1)
            schedule = self._build_mask_schedule(pixels.detach())
            phases = self._pick_input_diversity_phases(pixels.size(0), pixels.device)

            if self._gradient_replay is not None:
                self._gradient_replay.set_context(view=0)
            self._actual_forward_view_count += 1
            yield self._forward_with_schedule(pixels, labels, schedule)

            phase_schedule = self._phase_mask_schedule(
                schedule,
                phases,
                height=pixels.size(-2),
                width=pixels.size(-1),
            )
            shifted_pixels = self._apply_samplewise_phase_shifts(pixels, phases)
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(view=1)
            self._actual_forward_view_count += 1
            yield self._forward_with_schedule(shifted_pixels, labels, phase_schedule)

    def mainline_metadata(self) -> dict[str, object]:
        return {
            "attack_method": f"vit_progressive_{self.progressive_patch_selector}",
            "whitebox_model": MODEL_NAME,
            "patch_selector": self.progressive_patch_selector,
            "progressive_checkpoints": list(self.progressive_checkpoints),
            "progressive_drop_ratios": list(self.progressive_drop_ratios),
            "progressive_drop_counts": list(self._progressive_mask_counts),
            "progressive_repeated_positions": True,
            "score_reference": (
                "current_cls_plus_checkpoint_gaussian_noise"
                if self.progressive_patch_selector == "patch_score"
                else "none_uniform_all_local_tokens"
            ),
            "score_cls_noise_active": self.progressive_patch_selector == "patch_score",
            "score_cls_noise_strength": self.score_cls_noise_strength,
            "token_intervention": "local_patch_tokens_hard_zero_after_checkpoint",
            "mask_schedule_policy": "current_attack_iterate_per_step_group",
            "mask_schedule_count_per_image": self.steps * self.input_diversity_groups,
            "checkpoint_mask_selection_count_per_image": (
                self.steps * self.input_diversity_groups * len(self.progressive_checkpoints)
            ),
            "mask_pair_sharing": "same_schedule_with_phase_transformed_masks",
            "phase_mask_transform": "image_reflect_shift_then_14x14_occupancy_topk",
            "opponent_noise": "initial_rgb_projection_kept_union_only",
            "opponent_noise_strength": self.opponent_noise_strength,
            "feature_noise_cls": False,
            "asr_definition": "1 - adversarial accuracy over all evaluated samples",
            "model_mean": self.model_mean.flatten().tolist(),
            "model_std": self.model_std.flatten().tolist(),
            "gaussian_sigma": self.gaussian_sigma,
            "gaussian_alpha": self.gaussian_alpha,
        }


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def _parse_float_list(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc


def _parse_phase_shift(value: str) -> tuple[int, int]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != 2:
        raise argparse.ArgumentTypeError("phase shift must be two comma-separated integers")
    return values


def _parse_phase_shift_set(value: str) -> tuple[tuple[int, int], ...]:
    shifts = tuple(_parse_phase_shift(item) for item in value.split(";") if item.strip())
    if not shifts:
        raise argparse.ArgumentTypeError("phase shift set cannot be empty")
    return shifts


def _validate_output_dir(output_dir: str) -> Path:
    repo_root = Path(__file__).resolve().parent
    attack_root = (repo_root / "outputs" / "attack").resolve()
    resolved = Path(output_dir).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    resolved = resolved.resolve()
    try:
        resolved.relative_to(attack_root)
    except ValueError as exc:
        raise ValueError(f"output-dir must be under {attack_root}") from exc
    if resolved == attack_root:
        raise ValueError("output-dir must name a subdirectory under outputs/attack")
    return resolved


def _clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _attack_all_samples(
    dataloader,
    attacker: ViTProgressivePatchScoreAttacker,
    output_dir: Path,
    max_attacked_samples: int | None,
    sample_offset: int,
    replay: GradientReplay | None,
) -> list[str]:
    total = len(dataloader.dataset)
    if sample_offset < 0 or sample_offset >= total:
        raise ValueError(f"sample_offset must be in [0, {total}), got {sample_offset}")
    available = total - sample_offset
    limit = available if max_attacked_samples is None else min(available, max_attacked_samples)
    progress = tqdm(total=limit, desc="Attacking samples")
    attacked = 0
    saved_count = 0
    seen = 0
    sample_ids: list[str] = []
    for images, labels, indices in dataloader:
        if attacked >= limit:
            break
        batch_end = seen + images.size(0)
        if batch_end <= sample_offset:
            seen = batch_end
            continue
        if seen < sample_offset:
            start = sample_offset - seen
            images, labels, indices = images[start:], labels[start:], indices[start:]
        seen = batch_end
        remaining = limit - attacked
        images, labels, indices = images[:remaining], labels[:remaining], indices[:remaining]
        filenames = [
            str(dataloader.dataset.samples[index]["image_name"])
            for index in indices.tolist()
        ]
        sample_ids.extend(filenames)
        adversarial = attacker.attack_batch(
            images,
            labels,
            replay=replay,
            sample_ids=filenames if replay is not None else None,
        )
        saved = save_adversarial_images(
            images=adversarial,
            output_dir=str(output_dir),
            prefix="adv",
            start_index=saved_count,
            filenames=filenames,
        )
        attacked += images.size(0)
        saved_count += len(saved)
        progress.update(images.size(0))
    progress.close()
    print(f"Attacked {attacked} samples and saved {saved_count} images to {output_dir}")
    return sample_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone progressive ViT patch-score attack")
    parser.add_argument("--checkpoints", type=_parse_int_list, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--drop-ratios", type=_parse_float_list, default=DEFAULT_DROP_RATIOS)
    parser.add_argument(
        "--patch-selector",
        choices=PROGRESSIVE_PATCH_SELECTORS,
        default="patch_score",
        help="Select from the score-high half or uniformly from all local patches.",
    )
    parser.add_argument("--score-cls-noise-strength", type=float, default=0.2)
    parser.add_argument("--opponent-noise-strength", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mi", dest="mi", action="store_true")
    parser.add_argument("--no-mi", dest="mi", action="store_false")
    parser.set_defaults(mi=True)
    parser.add_argument("--mi-decay", type=float, default=1.0)
    parser.add_argument("--ni", action="store_true")
    parser.add_argument("--ti-sigma", type=float, default=0.0)
    parser.add_argument("--input-diversity-phase-shift-set", type=_parse_phase_shift_set, default=((4, 4), (8, 8), (12, 12)))
    parser.add_argument("--input-diversity-groups", type=int, default=10)
    parser.add_argument("--gaussian-sigma", type=float, default=4.0)
    parser.add_argument("--gaussian-alpha", type=float, default=0.75)
    parser.add_argument("--max-attacked-samples", type=int, default=1000)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/attack/vit_progressive_patch_score")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    output_dir = _validate_output_dir(args.output_dir)

    dataloader, num_classes = load_data(
        image_dir_arg=args.image_dir,
        annotations_path_arg=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    model = build_whitebox_model(
        num_classes=num_classes,
        model_name=MODEL_NAME,
        device=DEVICE,
    )
    attacker = ViTProgressivePatchScoreAttacker(
        model,
        checkpoints=args.checkpoints,
        drop_ratios=args.drop_ratios,
        patch_selector=args.patch_selector,
        score_cls_noise_strength=args.score_cls_noise_strength,
        opponent_noise_strength=args.opponent_noise_strength,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        use_momentum=args.mi,
        momentum_decay=args.mi_decay,
        nesterov=args.ni,
        ti_sigma=args.ti_sigma,
        input_diversity=False,
        input_diversity_groups=args.input_diversity_groups,
        input_diversity_views_per_group=2,
        input_diversity_phase_shift_set=args.input_diversity_phase_shift_set,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_alpha=args.gaussian_alpha,
        device=DEVICE,
    )
    _clear_directory_contents(output_dir)
    replay = GradientReplay(args.seed) if args.seed is not None else None
    sample_ids = _attack_all_samples(
        dataloader,
        attacker,
        output_dir,
        args.max_attacked_samples,
        sample_offset=args.sample_offset,
        replay=replay,
    )
    if replay is not None:
        (output_dir / "replay_manifest.json").write_text(
            json.dumps(replay.manifest(sample_ids), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    params = vars(args).copy()
    params.update(
        {
            "whitebox_model": MODEL_NAME,
            "input_diversity_views_per_group": 2,
            "input_diversity_total_views": args.input_diversity_groups * 2,
            "input_diversity_phase_shift_set": [list(shift) for shift in args.input_diversity_phase_shift_set],
            "step_size": args.step_size if args.step_size is not None else args.epsilon / args.steps,
        }
    )
    params.update(attacker.mainline_metadata())
    (output_dir / "attack_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Attacked samples saved to {output_dir}")
    print("Run transfer evaluation separately with transfer_eval.py.")


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    main(parse_args())
