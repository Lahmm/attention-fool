"""Non-invasive, per-sample gradient diagnostics for transfer studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class GradientObserver:
    """Collect per-sample/per-step statistics without consuming randomness."""

    def __init__(
        self,
        enabled: bool = True,
        sample_ids: list[str] | None = None,
        reference_signs: list[torch.Tensor] | None = None,
        capture_signs: bool = False,
    ) -> None:
        self.enabled = enabled
        self.sample_ids = list(sample_ids or [])
        self.reference_signs = reference_signs
        self.capture_signs = capture_signs
        self.captured_signs: list[torch.Tensor] = []
        self._step = 0
        self._records: list[dict[str, Any]] = []
        self._gradient_by_step: list[torch.Tensor] = []

    def _current(self) -> dict[str, Any]:
        while len(self._records) <= self._step:
            self._records.append({"step": self._step, "per_sample": []})
        return self._records[self._step]

    def _ids(self, batch_size: int) -> list[str]:
        if self.sample_ids:
            if len(self.sample_ids) != batch_size:
                raise ValueError("observer sample_ids do not match the gradient batch.")
            return self.sample_ids
        return [f"batch_sample_{index}" for index in range(batch_size)]

    @staticmethod
    def _ensure_sample_records(rec: dict[str, Any], sample_ids: list[str]) -> list[dict[str, Any]]:
        records = rec["per_sample"]
        if not records:
            records.extend({"sample_id": sample_id, "step": rec["step"]} for sample_id in sample_ids)
        return records

    def record_raw_views(self, view_grads: torch.Tensor) -> None:
        if not self.enabled:
            return
        with torch.no_grad():
            views = view_grads.detach()
            view_count, batch_size = views.shape[:2]
            sample_ids = self._ids(batch_size)
            rec = self._current()
            samples = self._ensure_sample_records(rec, sample_ids)

            flat = views.flatten(2).permute(1, 0, 2)  # [B,V,D]
            norms = flat.norm(p=2, dim=2)
            unit = flat / norms.unsqueeze(-1).clamp_min(1e-12)
            gram = torch.bmm(unit, unit.transpose(1, 2))
            off_diagonal = ~torch.eye(view_count, device=views.device, dtype=torch.bool)
            pairwise = gram[:, off_diagonal].mean(dim=1)

            mean_unit = unit.mean(dim=1)
            mean_unit = mean_unit / mean_unit.norm(dim=1, keepdim=True).clamp_min(1e-12)
            view_to_mean = (unit * mean_unit.unsqueeze(1)).sum(dim=2)
            mean_sign = flat.mean(dim=1).sign()
            sign_agreement = flat.sign().eq(mean_sign.unsqueeze(1)).float().mean(dim=2)

            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
            probabilities = eigenvalues / eigenvalues.sum(dim=1, keepdim=True).clamp_min(1e-12)
            effective_rank = torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            )

            if view_count % 2 == 0:
                group_count = view_count // 2
                grouped = flat.view(batch_size, group_count, 2, -1)
                group_unit = grouped / grouped.norm(dim=3, keepdim=True).clamp_min(1e-12)
                within = (group_unit[:, :, 0] * group_unit[:, :, 1]).sum(dim=2)
                group_means = grouped.mean(dim=2)
                group_norms = group_means.norm(dim=2)
                total_group_mean = group_means.mean(dim=1)
                group_to_rest = []
                loo_influence = []
                for group_index in range(group_count):
                    rest = (
                        group_means.sum(dim=1) - group_means[:, group_index]
                    ) / max(1, group_count - 1)
                    group_to_rest.append(
                        F.cosine_similarity(group_means[:, group_index], rest, dim=1)
                    )
                    loo_influence.append(
                        1.0 - F.cosine_similarity(total_group_mean, rest, dim=1)
                    )
                group_to_rest_tensor = torch.stack(group_to_rest, dim=1)
                loo_influence_tensor = torch.stack(loo_influence, dim=1)
            else:
                within = group_norms = group_to_rest_tensor = loo_influence_tensor = None

            for batch_index, sample in enumerate(samples):
                sample.update(
                    {
                        "view_pairwise_cosine": float(pairwise[batch_index].cpu()),
                        "view_to_mean_cosine": float(view_to_mean[batch_index].mean().cpu()),
                        "view_sign_agreement": float(sign_agreement[batch_index].mean().cpu()),
                        "view_effective_rank": float(effective_rank[batch_index].cpu()),
                        "view_norm_mean": float(norms[batch_index].mean().cpu()),
                        "view_norm_cv": float(
                            (norms[batch_index].std() / norms[batch_index].mean().clamp_min(1e-12)).cpu()
                        ),
                    }
                )
                if within is not None:
                    sample.update(
                        {
                            "group_within_cosine_mean": float(within[batch_index].mean().cpu()),
                            "group_within_cosine_min": float(within[batch_index].min().cpu()),
                            "group_to_rest_cosine_mean": float(
                                group_to_rest_tensor[batch_index].mean().cpu()
                            ),
                            "group_to_rest_cosine_min": float(
                                group_to_rest_tensor[batch_index].min().cpu()
                            ),
                            "group_norm_cv": float(
                                (
                                    group_norms[batch_index].std(unbiased=False)
                                    / group_norms[batch_index].mean().clamp_min(1e-12)
                                ).cpu()
                            ),
                            "group_loo_influence_max": float(
                                loo_influence_tensor[batch_index].max().cpu()
                            ),
                        }
                    )

            self._update_step_means(rec)

    def record_aggregated(self, grad: torch.Tensor) -> None:
        if not self.enabled:
            return
        with torch.no_grad():
            gradient = grad.detach()
            batch_size = gradient.size(0)
            rec = self._current()
            samples = self._ensure_sample_records(rec, self._ids(batch_size))
            flat = gradient.flatten(1)
            abs_flat = flat.abs()
            centered = flat - flat.mean(dim=1, keepdim=True)
            standardized = centered / centered.std(dim=1, keepdim=True).clamp_min(1e-12)
            fft_features = self._signed_frequency_features(gradient)
            spatial_features = self._spatial_features(gradient)

            for index, sample in enumerate(samples):
                sample.update(
                    {
                        "agg_norm": float(flat[index].norm().cpu()),
                        "agg_abs_mean": float(abs_flat[index].mean().cpu()),
                        "agg_kurtosis": float(standardized[index].pow(4).mean().cpu()),
                        **fft_features[index],
                        **spatial_features[index],
                    }
                )
            self._update_step_means(rec)

    def record_probe(
        self,
        baseline_gradient: torch.Tensor,
        probed_gradient: torch.Tensor,
        probe_name: str,
    ) -> None:
        if not self.enabled:
            return
        with torch.no_grad():
            baseline = baseline_gradient.detach().flatten(1)
            probed = probed_gradient.detach().flatten(1)
            delta = baseline - probed
            rec = self._current()
            samples = self._ensure_sample_records(rec, self._ids(baseline.size(0)))
            for index, sample in enumerate(samples):
                sample["probe_name"] = probe_name
                sample["probe_delta_l2_fraction"] = float(
                    (delta[index].norm() / baseline[index].norm().clamp_min(1e-12)).cpu()
                )
                sample["probe_to_baseline_cosine"] = float(
                    F.cosine_similarity(
                        probed[index : index + 1], baseline[index : index + 1], dim=1
                    )[0].cpu()
                )
            self._update_step_means(rec)

    def record_gradient(self, grad: torch.Tensor) -> None:
        if not self.enabled:
            return
        with torch.no_grad():
            gradient = grad.detach().flatten(1).cpu()
            self._gradient_by_step.append(gradient)
            rec = self._current()
            samples = self._ensure_sample_records(rec, self._ids(grad.size(0)))
            for index, sample in enumerate(samples):
                sample["norm_l2"] = float(gradient[index].norm())
                sample["norm_abs_mean"] = float(gradient[index].abs().mean())
                if self._step > 0:
                    sample["step_to_step_grad_cosine"] = float(
                        F.cosine_similarity(
                            gradient[index : index + 1],
                            self._gradient_by_step[self._step - 1][index : index + 1],
                            dim=1,
                        )[0]
                    )
            self._update_step_means(rec)

    def record_pre_momentum(self, momentum: torch.Tensor, gradient: torch.Tensor) -> None:
        if not self.enabled:
            return
        with torch.no_grad():
            rec = self._current()
            samples = self._ensure_sample_records(rec, self._ids(gradient.size(0)))
            momentum_flat = momentum.detach().flatten(1)
            gradient_flat = gradient.detach().flatten(1)
            cosines = F.cosine_similarity(momentum_flat, gradient_flat, dim=1)
            for index, sample in enumerate(samples):
                sample["pre_momentum_to_grad_cosine"] = (
                    None if self._step == 0 else float(cosines[index].cpu())
                )

    def record_momentum(self, momentum: torch.Tensor) -> None:
        if not self.enabled:
            return
        with torch.no_grad():
            rec = self._current()
            samples = self._ensure_sample_records(rec, self._ids(momentum.size(0)))
            momentum_flat = momentum.detach().flatten(1).cpu()
            gradient = self._gradient_by_step[self._step]
            cosines = F.cosine_similarity(momentum_flat, gradient, dim=1)
            for index, sample in enumerate(samples):
                sample["post_momentum_to_grad_cosine"] = float(cosines[index])
                sample["momentum_norm"] = float(momentum_flat[index].norm())
            self._update_step_means(rec)

    def record_sign_update(self, sign_update: torch.Tensor) -> None:
        if not self.enabled:
            return
        with torch.no_grad():
            signs = sign_update.detach().sign().cpu()
            rec = self._current()
            samples = self._ensure_sample_records(rec, self._ids(sign_update.size(0)))
            if self.capture_signs:
                self.captured_signs.append(signs.clone())
            reference = None
            if self.reference_signs is not None and self._step < len(self.reference_signs):
                reference = self.reference_signs[self._step]
            for index, sample in enumerate(samples):
                sample["sign_positive_fraction"] = float((signs[index] > 0).float().mean())
                if reference is not None:
                    sample["update_sign_flip_rate"] = float(
                        signs[index].ne(reference[index]).float().mean()
                    )
            self._update_step_means(rec)

    def close_step(self) -> None:
        if self.enabled:
            self._step += 1

    @staticmethod
    def _signed_frequency_features(gradient: torch.Tensor) -> list[dict[str, float]]:
        power = torch.fft.fftshift(torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1)).abs().pow(2)
        power = power.sum(dim=1)
        height, width = gradient.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(height, device=gradient.device, dtype=torch.float32),
            torch.arange(width, device=gradient.device, dtype=torch.float32),
            indexing="ij",
        )
        radius = ((yy - height / 2.0).square() + (xx - width / 2.0).square()).sqrt()
        maximum = radius.max()
        masks = (
            radius <= maximum * 0.25,
            (radius > maximum * 0.25) & (radius <= maximum * 0.50),
            radius > maximum * 0.50,
        )
        total = power.sum(dim=(1, 2)).clamp_min(1e-12)
        fractions = [(power * mask).sum(dim=(1, 2)) / total for mask in masks]
        return [
            {
                "freq_low": float(fractions[0][index].cpu()),
                "freq_mid": float(fractions[1][index].cpu()),
                "freq_high": float(fractions[2][index].cpu()),
            }
            for index in range(gradient.size(0))
        ]

    @staticmethod
    def _spatial_features(gradient: torch.Tensor) -> list[dict[str, float]]:
        energy = gradient.detach().pow(2).sum(dim=1).flatten(1)
        probabilities = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        sorted_values = energy.sort(dim=1).values
        count = energy.size(1)
        ranks = torch.arange(1, count + 1, device=energy.device, dtype=energy.dtype)
        gini = (
            2.0 * (sorted_values * ranks).sum(dim=1)
            / (count * sorted_values.sum(dim=1).clamp_min(1e-12))
            - (count + 1.0) / count
        )
        tail_count = max(1, int(round(count * 0.05)))
        tail_fraction = sorted_values[:, -tail_count:].sum(dim=1) / sorted_values.sum(dim=1).clamp_min(1e-12)
        return [
            {
                "spatial_entropy": float(entropy[index].cpu()),
                "spatial_gini": float(gini[index].cpu()),
                "spatial_top5_energy": float(tail_fraction[index].cpu()),
            }
            for index in range(gradient.size(0))
        ]

    @staticmethod
    def _update_step_means(rec: dict[str, Any]) -> None:
        samples = rec.get("per_sample", [])
        if not samples:
            return
        keys = {
            key
            for sample in samples
            for key, value in sample.items()
            if isinstance(value, (int, float)) and key != "step"
        }
        for key in keys:
            values = [float(sample[key]) for sample in samples if isinstance(sample.get(key), (int, float))]
            if values:
                rec[key] = _mean(values)

    @property
    def step_count(self) -> int:
        return self._step

    def per_sample_summary(self) -> list[dict[str, Any]]:
        by_sample: dict[str, dict[str, list[float]]] = {}
        for rec in self._records:
            for sample in rec.get("per_sample", []):
                sample_id = str(sample["sample_id"])
                feature_values = by_sample.setdefault(sample_id, {})
                for key, value in sample.items():
                    if isinstance(value, (int, float)) and key != "step":
                        feature_values.setdefault(key, []).append(float(value))
        output = []
        for sample_id, features in by_sample.items():
            output.append({"sample_id": sample_id, **{key: _mean(values) for key, values in features.items()}})
        return output

    def summarize(self) -> dict[str, Any]:
        samples = self.per_sample_summary()
        summary: dict[str, Any] = {"num_steps": len(self._records), "num_samples": len(samples)}
        keys = {key for sample in samples for key, value in sample.items() if isinstance(value, (int, float))}
        for key in sorted(keys):
            summary[key] = _mean([float(sample[key]) for sample in samples if key in sample])
        # Compatibility aliases for previous scripts.
        aliases = {
            "view_pairwise_cosine_mean": "view_pairwise_cosine",
            "view_to_mean_cosine_mean": "view_to_mean_cosine",
            "view_sign_agreement_mean": "view_sign_agreement",
            "view_effective_rank_mean": "view_effective_rank",
            "agg_norm_mean": "agg_norm",
            "mom_to_grad_cosine": "post_momentum_to_grad_cosine",
        }
        for alias, source in aliases.items():
            if source in summary:
                summary[alias] = summary[source]
        return summary

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "gradient_summary.json").write_text(
            json.dumps(self.summarize(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        serializable_records = []
        for rec in self._records:
            serializable_records.append(
                {
                    key: value
                    for key, value in rec.items()
                    if key != "per_sample" and isinstance(value, (int, float, str, bool))
                }
            )
        (path / "gradient_per_step.json").write_text(
            json.dumps(serializable_records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with (path / "gradient_per_sample_step.jsonl").open("w", encoding="utf-8") as handle:
            for rec in self._records:
                for sample in rec.get("per_sample", []):
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
