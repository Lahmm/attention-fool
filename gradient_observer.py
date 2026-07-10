"""Gradient observation and diagnostic infrastructure.

Hooks into PatchScoreAttacker to capture per-view gradients, aggregated
gradients, momentum, and sign updates at every step without changing the
attack logic.  Designed for hypothesis-driven exploration of what gradient
structures correlate with transferability.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _ensure_numpy(tensor: torch.Tensor) -> "np.ndarray":
    import numpy as np

    return tensor.detach().cpu().numpy()


class GradientObserver:
    """Non-invasive gradient capture and structural analysis.

    Usage inside attack_batch (pseudocode)::

        obs = GradientObserver()
        for step in range(steps):
            per_view_grads = _attack_grad(...)  # 20 x B x C x H x W
            aggregated = _aggregate_gradients(per_view_grads)
            obs.record_raw_views(per_view_grads)
            normalized = _normalize_grad(aggregated)
            obs.record_normalized(normalized)
            momentum = decay * momentum + normalized
            obs.record_momentum(momentum)
            sign_update = momentum.sign()
            obs.record_sign_update(sign_update)
            obs.close_step()
        obs.summarize()
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._step: int = 0
        # Per-step records
        self._records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Per-step capture
    # ------------------------------------------------------------------

    def record_raw_views(self, view_grads: torch.Tensor) -> None:
        """Capture the 20 x B x C x H x W per-view raw gradients."""
        if not self.enabled:
            return
        rec = self._current()
        # Keep a detached copy – expensive in memory for many steps, so
        # we compute summary statistics immediately and discard the tensor.
        v = view_grads.detach()  # [V, B, C, H, W]
        V, B = v.shape[0], v.shape[1]

        # Per-sample, per-view norm
        rec["view_norms"] = v.flatten(2).norm(p=2, dim=2).cpu().tolist()  # [V, B]

        # Pairwise cosine between views (batched: one matrix per sample)
        flat = v.flatten(2)  # [V, B, D]
        flat = flat / flat.norm(p=2, dim=2, keepdim=True).clamp_min(1e-12)
        pairwise_cos = torch.bmm(flat.transpose(0, 1), flat.transpose(0, 1).transpose(1, 2))
        rec["view_pairwise_cosine_mean"] = float(pairwise_cos.mean().cpu())  # scalar

        # Per-view cosine to the mean view
        mean_view = flat.mean(dim=0, keepdim=True)  # [1, B, D]
        mean_view = mean_view / mean_view.norm(p=2, dim=2, keepdim=True).clamp_min(1e-12)
        view_to_mean = (flat * mean_view).sum(dim=2)  # [V, B]
        rec["view_to_mean_cosine_mean"] = float(view_to_mean.mean().cpu())
        rec["view_to_mean_cosine_std"] = float(view_to_mean.std().cpu())

        # Sign agreement: per-coordinate sign of each view vs sign of mean view
        sign_mean = flat.mean(dim=0).sign()  # [B, D]
        view_signs = flat.sign()  # [V, B, D]
        agreement = (view_signs == sign_mean.unsqueeze(0)).float().mean(dim=2)  # [V, B]
        rec["view_sign_agreement_mean"] = float(agreement.mean().cpu())
        rec["view_sign_agreement_std"] = float(agreement.std().cpu())

        # Effective rank of the view set (spectral entropy of Gram matrix)
        gram = torch.bmm(flat.transpose(0, 1), flat.transpose(0, 1).transpose(1, 2))  # [B, V, V]
        try:
            eigvals = torch.linalg.eigvalsh(gram)  # [B, V]
            eigvals = eigvals.clamp_min(0)
            probs = eigvals / eigvals.sum(dim=1, keepdim=True).clamp_min(1e-12)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            effective_rank = torch.exp(entropy)  # [B]
            rec["view_effective_rank_mean"] = float(effective_rank.mean().cpu())
        except Exception:
            rec["view_effective_rank_mean"] = float(V)

        # Frequency decomposition: 2D DCT of spatial gradient magnitude
        rec["freq"] = self._freq_analysis(view_grads)

        # Channel statistics
        rec["channel"] = self._channel_analysis(view_grads)

        # Spatial concentration: Gini coefficient of per-pixel gradient energy
        rec["spatial"] = self._spatial_analysis(view_grads)

        del v, flat, pairwise_cos

    def record_aggregated(self, grad: torch.Tensor) -> None:
        """Capture post-aggregation gradient (B x C x H x W)."""
        if not self.enabled:
            return
        rec = self._current()
        g = grad.detach()
        rec["agg_norm_mean"] = float(g.flatten(1).norm(p=2, dim=1).mean().cpu())
        rec["agg_norm_std"] = float(g.flatten(1).norm(p=2, dim=1).std().cpu())
        rec["agg_abs_mean"] = float(g.abs().mean().cpu())
        rec["agg_abs_std"] = float(g.abs().std().cpu())

        # Skewness / kurtosis of gradient values
        flat = g.flatten().cpu()
        mean_val = flat.mean()
        std_val = flat.std().clamp_min(1e-12)
        rec["agg_skewness"] = float(((flat - mean_val) / std_val).pow(3).mean())
        rec["agg_kurtosis"] = float(((flat - mean_val) / std_val).pow(4).mean())

    def record_normalized(self, grad: torch.Tensor) -> None:
        """Capture post-normalization gradient (B x C x H x W)."""
        if not self.enabled:
            return
        rec = self._current()
        g = grad.detach()
        rec["norm_norm_mean"] = float(g.flatten(1).norm(p=2, dim=1).mean().cpu())
        rec["norm_abs_mean"] = float(g.abs().mean().cpu())
        # Fraction of near-zero entries after normalization
        rec["norm_zero_frac"] = float((g.abs() < 1e-8).float().mean().cpu())
        # Stash flattened normalized gradient for momentum-cosine later
        rec["normalized_flat"] = g.flatten(1).cpu()

    def record_momentum(self, momentum: torch.Tensor) -> None:
        """Capture the MI momentum accumulator (B x C x H x W)."""
        if not self.enabled:
            return
        rec = self._current()
        m = momentum.detach()
        rec["mom_norm_mean"] = float(m.flatten(1).norm(p=2, dim=1).mean().cpu())
        rec["mom_abs_mean"] = float(m.abs().mean().cpu())
        # Cosine between momentum direction and latest normalized gradient
        if "normalized_flat" in rec:
            n_flat = rec.pop("normalized_flat").to(m.device)
            m_flat = m.flatten(1)
            m_flat = m_flat / m_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            n_flat = n_flat / n_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            rec["mom_to_grad_cosine"] = float((m_flat * n_flat).sum(dim=1).mean().cpu())
        # Cosine between current momentum and current normalized gradient
        # (computed externally and stored here if available)

    def record_sign_update(self, sign_update: torch.Tensor) -> None:
        """Capture the final sign update applied to pixels (B x C x H x W)."""
        if not self.enabled:
            return
        rec = self._current()
        s = sign_update.detach()
        # Fraction of positive signs
        rec["sign_pos_frac"] = float((s > 0).float().mean().cpu())
        rec["sign_neg_frac"] = float((s < 0).float().mean().cpu())
        rec["sign_zero_frac"] = float((s == 0).float().mean().cpu())

    def close_step(self) -> None:
        if not self.enabled:
            return
        # Compute step-to-step gradient correlation if we have previous step data
        rec = self._current()
        if self._step > 0:
            prev = self._records[self._step - 1]
            if "normalized_flat" in rec and "normalized_flat" in prev:
                cur_flat = rec["normalized_flat"]
                prv_flat = prev["normalized_flat"]
                # Both on CPU from record_normalized
                cur_flat = cur_flat / cur_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                prv_flat = prv_flat / prv_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                rec["step_to_step_grad_cosine"] = float((cur_flat * prv_flat).sum(dim=1).mean())
            if "mom_to_grad_cosine" in prev:
                rec["prev_mom_to_grad_cosine"] = prev["mom_to_grad_cosine"]
        self._step += 1

    # ------------------------------------------------------------------
    # Structural analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _freq_analysis(view_grads: torch.Tensor) -> dict[str, float]:
        """Frequency energy distribution via 2D FFT radial binning.

        Uses the radially-averaged power spectrum of the mean-view gradient
        magnitude to estimate how gradient energy distributes across spatial
        frequencies.  Low frequencies correspond to broad image structures;
        high frequencies to fine texture / edge detail.
        """
        g = view_grads.detach()  # [V, B, C, H, W]
        # Mean gradient across views, averaged over batch
        mean_g = g.mean(dim=(0, 1))  # [C, H, W]
        # Spatial gradient magnitude
        mag = mean_g.abs().mean(dim=0)  # [H, W]
        H, W = int(mag.shape[-2]), int(mag.shape[-1])

        try:
            # 2D FFT and radial power spectrum
            fft = torch.fft.fft2(mag)
            fft_shift = torch.fft.fftshift(fft)
            power = (fft_shift.abs() ** 2).float()

            # Radial frequency bins
            cy, cx = H / 2.0, W / 2.0
            y_grid = torch.arange(H, device=g.device, dtype=torch.float32)
            x_grid = torch.arange(W, device=g.device, dtype=torch.float32)
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
            radii = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt()
            max_radius = float(radii.max().item())

            low_cut = max_radius * 0.25
            mid_cut = max_radius * 0.50

            low_mask = radii <= low_cut
            mid_mask = (radii > low_cut) & (radii <= mid_cut)
            high_mask = radii > mid_cut

            total = power.sum().clamp_min(1e-12)
            low_energy = float((power * low_mask).sum() / total)
            mid_energy = float((power * mid_mask).sum() / total)
            high_energy = float((power * high_mask).sum() / total)

            return {
                "low_freq_frac": low_energy,
                "mid_freq_frac": mid_energy,
                "high_freq_frac": high_energy,
            }
        except Exception:
            return {"low_freq_frac": 1.0 / 3, "mid_freq_frac": 1.0 / 3, "high_freq_frac": 1.0 / 3}

    @staticmethod
    def _channel_analysis(view_grads: torch.Tensor) -> dict[str, float]:
        """Per-channel statistics of the gradient tensor."""
        g = view_grads.detach()  # [V, B, C, H, W]
        # Mean view gradient
        mean_g = g.mean(dim=0)  # [B, C, H, W]
        ch_norms = mean_g.flatten(2).norm(p=2, dim=2)  # [B, C]
        ch_ratios = ch_norms / ch_norms.sum(dim=1, keepdim=True).clamp_min(1e-12)  # [B, C]
        return {
            f"ch{i}_norm_frac": float(ch_ratios[:, i].mean().cpu())
            for i in range(min(3, g.shape[2]))
        }

    @staticmethod
    def _spatial_analysis(view_grads: torch.Tensor) -> dict[str, float]:
        """Spatial concentration metrics."""
        g = view_grads.detach()  # [V, B, C, H, W]
        # Per-pixel gradient energy
        energy = g.pow(2).mean(dim=(0, 2))  # [B, H, W]
        B = energy.shape[0]
        flat = energy.flatten(1)  # [B, H*W]
        sorted_vals = flat.sort(dim=1).values
        n = flat.shape[1]
        ranks = torch.arange(1, n + 1, device=g.device, dtype=torch.float32)
        # Gini coefficient: 1 - 2 * sum((n+1-i)*sorted) / (n*sum(sorted))
        numerator = ((n + 1 - ranks) * sorted_vals).sum(dim=1)
        denominator = sorted_vals.sum(dim=1).clamp_min(1e-12)
        gini = 1.0 - 2.0 * numerator / (n * denominator)
        return {
            "spatial_gini_mean": float(gini.mean().cpu()),
            "spatial_entropy_mean": float(
                (-(flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()
                 * (flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)))
                .sum(dim=1)
                .mean()
                .cpu()
            ),
        }

    # ------------------------------------------------------------------
    # Internal records
    # ------------------------------------------------------------------

    def _current(self) -> dict[str, Any]:
        while len(self._records) <= self._step:
            self._records.append({})
        return self._records[self._step]

    @property
    def step_count(self) -> int:
        return self._step

    def summarize(self) -> dict[str, Any]:
        """Return averaged summary across all recorded steps."""
        if not self._records:
            return {}
        # Average all scalar fields across steps
        keys = set()
        for rec in self._records:
            keys.update(rec.keys())

        summary: dict[str, Any] = {"num_steps": len(self._records)}
        for key in sorted(keys):
            if key in ("freq", "channel", "spatial"):
                # Aggregate nested dicts
                sub_keys = set()
                for rec in self._records:
                    if key in rec:
                        sub_keys.update(rec[key].keys())
                for sk in sorted(sub_keys):
                    vals = [rec[key][sk] for rec in self._records if key in rec]
                    if vals:
                        summary[f"{key}_{sk}"] = sum(vals) / len(vals)
            elif key.startswith("view_norms"):
                continue  # skip large arrays
            else:
                vals = [rec[key] for rec in self._records if key in rec]
                if vals and isinstance(vals[0], (int, float)):
                    summary[key] = sum(vals) / len(vals)
        return summary

    def save(self, path: Path) -> None:
        """Persist summary and per-step records as JSON."""
        path.mkdir(parents=True, exist_ok=True)
        summary = self.summarize()
        (path / "gradient_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Save per-step records (excluding large arrays)
        step_records = []
        for rec in self._records:
            clean = {}
            for k, v in rec.items():
                if isinstance(v, (int, float, str, bool)):
                    clean[k] = v
                elif isinstance(v, dict):
                    clean[k] = {sk: sv for sk, sv in v.items() if isinstance(sv, (int, float, str, bool))}
            step_records.append(clean)
        (path / "gradient_per_step.json").write_text(
            json.dumps(step_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
