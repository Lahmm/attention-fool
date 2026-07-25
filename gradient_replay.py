from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


class GradientReplay:
    """Deterministic, sample-keyed random stream for gradient studies.

    Seeds depend only on the master seed, sample identifier, attack step,
    augmentation group, view, and event name. They therefore do not depend on
    batch position or on random calls made by experimental probes.
    """

    VERSION = 1

    def __init__(self, master_seed: int) -> None:
        self.master_seed = int(master_seed)
        self.sample_ids: tuple[str, ...] = ()
        self.step = -1
        self.group = -1
        self.view = -1
        self._event_hasher = hashlib.sha256()
        self._event_count = 0
        self._phase_events: list[dict[str, object]] = []

    def begin_batch(self, sample_ids: list[str] | tuple[str, ...]) -> None:
        if not sample_ids:
            raise ValueError("sample_ids cannot be empty for random replay.")
        self.sample_ids = tuple(str(value) for value in sample_ids)
        self.step = self.group = self.view = -1

    def set_context(self, *, step: int | None = None, group: int | None = None, view: int | None = None) -> None:
        if step is not None:
            self.step = int(step)
        if group is not None:
            self.group = int(group)
        if view is not None:
            self.view = int(view)

    def _seed(self, event: str, sample_id: str) -> int:
        key = (
            f"v{self.VERSION}|{self.master_seed}|{sample_id}|{self.step}|"
            f"{self.group}|{self.view}|{event}"
        )
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little") & ((1 << 63) - 1)
        self._event_hasher.update(f"{key}|{seed}\n".encode("utf-8"))
        self._event_count += 1
        return seed

    @staticmethod
    def _generator(device: torch.device, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def randn_like(self, tensor: torch.Tensor, event: str) -> torch.Tensor:
        if tensor.size(0) != len(self.sample_ids):
            raise ValueError("replay tensor batch does not match sample_ids.")
        values = []
        for index, sample_id in enumerate(self.sample_ids):
            generator = self._generator(tensor.device, self._seed(event, sample_id))
            values.append(
                torch.randn(
                    tensor[index].shape,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    generator=generator,
                )
            )
        return torch.stack(values, dim=0)

    def rand_scalar(self, event: str, sample_index: int, *, device: torch.device) -> torch.Tensor:
        sample_id = self.sample_ids[sample_index]
        generator = self._generator(device, self._seed(event, sample_id))
        return torch.rand((), device=device, generator=generator)

    def randint(self, high: int, event: str, sample_index: int, *, device: torch.device) -> int:
        sample_id = self.sample_ids[sample_index]
        generator = self._generator(device, self._seed(event, sample_id))
        return int(torch.randint(high, (), device=device, generator=generator).item())

    def randperm(self, count: int, event: str, sample_index: int, *, device: torch.device) -> torch.Tensor:
        sample_id = self.sample_ids[sample_index]
        generator = self._generator(device, self._seed(event, sample_id))
        return torch.randperm(count, device=device, generator=generator)

    def multinomial(
        self,
        weights: torch.Tensor,
        count: int,
        event: str,
        sample_index: int,
    ) -> torch.Tensor:
        sample_id = self.sample_ids[sample_index]
        generator = self._generator(weights.device, self._seed(event, sample_id))
        return torch.multinomial(weights, count, replacement=False, generator=generator)

    def record_phase(self, sample_index: int, phase: tuple[int, int]) -> None:
        self._phase_events.append(
            {
                "sample_id": self.sample_ids[sample_index],
                "step": self.step,
                "group": self.group,
                "phase": [int(phase[0]), int(phase[1])],
            }
        )

    def manifest(self, sample_ids: list[str]) -> dict[str, object]:
        canonical_sample_ids = json.dumps(
            list(sample_ids), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return {
            "version": self.VERSION,
            "master_seed": self.master_seed,
            "sample_ids": list(sample_ids),
            "sample_ids_sha256": hashlib.sha256(canonical_sample_ids).hexdigest(),
            "event_count": self._event_count,
            "event_digest": self._event_hasher.hexdigest(),
            "phase_events": self._phase_events,
        }

    def save_manifest(self, path: Path, sample_ids: list[str]) -> None:
        path.write_text(
            json.dumps(self.manifest(sample_ids), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
