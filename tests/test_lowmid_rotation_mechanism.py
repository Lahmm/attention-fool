import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from attack import LazyAggregationAttacker
from experiments.lowmid_rotation_mechanism import (
    DEFAULT_ASR,
    aggregate_report,
    band_energy,
    build_conclusion_zh,
    source_rows_from_trace,
    spectrum_metrics,
    trace_attack,
    write_outputs,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 16 * 16, 4)

    def forward(self, x, return_attn=False):
        return self.linear(x.flatten(1))


def make_attacker(**kwargs):
    return LazyAggregationAttacker(
        TinyModel(),
        epsilon=0.1,
        steps=2,
        ti_sigma=0,
        layers=(-1,),
        guide_aug=False,
        device=torch.device("cpu"),
        **kwargs,
    )


def minimal_args(**kwargs):
    values = {
        "output_dir": "unused",
        "image_dir": "unused",
        "annotations_path": "unused",
        "img_size": 16,
        "batch_size": 2,
        "num_workers": 0,
        "max_samples": 2,
        "target_samples": 0,
        "trace_steps": (1, 2),
        "target_steps": (),
        "target_models": (),
        "seed": 0,
        "asr_csv": (),
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


class LowmidRotationMechanismTests(unittest.TestCase):
    def test_band_energy_sum_matches_total(self):
        attacker = make_attacker()
        grad = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        expected = sum(attacker._fft_project_grad(grad, band).square().flatten(1).sum(1) for band in range(8))
        observed = spectrum_metrics(grad)["total_energy"]
        self.assertTrue(torch.allclose(observed, expected, atol=1e-10, rtol=1e-10))

    def test_rotation_increases_lowmid_ratio(self):
        attacker = make_attacker(lowmid_grad_tuning=True, lowmid_grad_rotation_strength=0.5)
        grad = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        tuned = attacker._tune_lowmid_gradient(grad)
        raw = spectrum_metrics(grad)["lowmid_ratio"]
        rot = spectrum_metrics(tuned)["lowmid_ratio"]
        self.assertTrue(torch.all(rot > raw))

    def test_report_handles_empty_target_rows(self):
        rows = [
            {
                "scope": "source",
                "branch": "rotation/mi_on",
                "delta_lowmid_ratio": 0.1,
                "cos_raw_rot": 0.9,
                "sign_raw_rot_agree": 0.8,
                "cos_rot_momentum_after": 0.7,
                "sign_rot_momentum_after_agree": 0.75,
                "rotation_update_sign_change": 0.2,
            }
        ]
        report = aggregate_report(rows, minimal_args())
        self.assertEqual(report["known_asr"], DEFAULT_ASR)
        self.assertEqual(report["source_summary"]["rotation/mi_on"]["delta_lowmid_ratio"], 0.1)
        self.assertEqual(report["target_summary"]["rotation/mi_on"]["delta_target_cos"], None)
        self.assertIn("频谱比例", build_conclusion_zh(report))

    def test_trace_attack_tiny_smoke_writes_outputs(self):
        torch.manual_seed(2)
        attacker = make_attacker(lowmid_grad_tuning=True, lowmid_grad_rotation_strength=0.5, use_momentum=True)
        images = torch.randn(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        trace = trace_attack(attacker, images, labels, "rotation/mi_on", {1, 2})
        rows = source_rows_from_trace(trace, torch.tensor([10, 11]))
        report = aggregate_report(rows, minimal_args())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(rows, report, out)
            self.assertTrue((out / "lowmid_rotation_mechanism_metrics.csv").exists())
            self.assertTrue((out / "lowmid_rotation_mechanism_report.json").exists())
            self.assertTrue((out / "lowmid_rotation_mechanism_conclusion_zh.md").exists())


if __name__ == "__main__":
    unittest.main()
