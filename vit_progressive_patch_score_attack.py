"""Backward-compatible ViT CLI for the architecture-neutral mainline.

New experiments should invoke ``main.py``.  This file intentionally contains
no attack implementation and has no dependency on the legacy ``attack.py``.
"""

from __future__ import annotations

import argparse

from main import main as _run_mainline
from main import parse_args as _parse_mainline_args
from progressive_attack import (
    DEFAULT_DROP_RATIOS,
    PROGRESSIVE_PATCH_SELECTORS,
    ProgressiveMaskSchedule,
    ProgressivePatchScoreAttacker,
)
from utils import DEVICE


MODEL_NAME = "vit_base_patch16_224"
DEFAULT_CHECKPOINTS = (3, 7, 11)
ViTProgressivePatchScoreAttacker = ProgressivePatchScoreAttacker


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


def parse_args() -> argparse.Namespace:
    args = _parse_mainline_args()
    args.attack_method = "progressive"
    args.whitebox_model = MODEL_NAME
    return args


def main(args: argparse.Namespace) -> None:
    args.attack_method = "progressive"
    args.whitebox_model = MODEL_NAME
    _run_mainline(args)


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    main(parse_args())
