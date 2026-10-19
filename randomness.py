from __future__ import annotations

import hashlib
import torch

from utils import SEED


class AttackRandomness:
    def __init__(self, sample_ids):
        if not sample_ids:
            raise ValueError("sample_ids cannot be empty.")
        self.sample_ids = tuple(str(value) for value in sample_ids)
        self.step = self.group = self.view = -1

    def set_context(self, *, step: int | None = None, group: int | None = None, view: int | None = None) -> None:
        if step is not None:
            self.step = int(step)
        if group is not None:
            self.group = int(group)
        if view is not None:
            self.view = int(view)

    def _seed(self, event, sample_id):
        key = f"v1|{SEED}|{sample_id}|{self.step}|{self.group}|{self.view}|{event}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") & ((1 << 63) - 1)

    @staticmethod
    def _generator(device: torch.device, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def randn_like(self, tensor: torch.Tensor, event: str) -> torch.Tensor:
        if tensor.size(0) != len(self.sample_ids):
            raise ValueError("random tensor batch does not match sample_ids.")
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

    def randint(self, high: int, event: str, sample_index: int, *, device: torch.device) -> int:
        sample_id = self.sample_ids[sample_index]
        generator = self._generator(device, self._seed(event, sample_id))
        return int(torch.randint(high, (), device=device, generator=generator).item())

    def randperm(self, count: int, event: str, sample_index: int, *, device: torch.device) -> torch.Tensor:
        sample_id = self.sample_ids[sample_index]
        generator = self._generator(device, self._seed(event, sample_id))
        return torch.randperm(count, device=device, generator=generator)
