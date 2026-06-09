import argparse
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from attack import LazyAggregationAttacker
from mi_gradient_accumulation_mechanism import (
    aggregate_report,
    build_conclusion_zh,
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
        lowmid_dss_filter=True,
        lowmid_dss_consistency="sign",
        lowmid_grad_tuning=True,
        lowmid_grad_rotation_strength=0.5,
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
        "trace_samples": 2,
        "target_samples": 0,
        "steps": 2,
        "trace_steps": (1, 2),
        "target_steps": (),
        "target_models": (),
        "seed": 0,
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


class MIGradientAccumulationMechanismTests(unittest.TestCase):
    def test_manual_accumulator_matches_mi_update_sign(self):
        torch.manual_seed(1)
        images = torch.randn(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        mi_attacker = make_attacker(use_momentum=True, momentum_decay=1.0)
        manual_attacker = make_attacker(use_momentum=False, momentum_decay=1.0)
        manual_attacker.model.load_state_dict(mi_attacker.model.state_dict())
        _mi_rows, mi_tensors, _ = trace_attack(mi_attacker, images, labels, "control_mi", keep_steps={1, 2})
        _manual_rows, manual_tensors, _ = trace_attack(
            manual_attacker,
            images,
            labels,
            "manual_accumulator_no_mi_flag",
            manual_accumulator=True,
            keep_steps={1, 2},
        )
        for mi_trace, manual_trace in zip(mi_tensors, manual_tensors):
            self.assertTrue(torch.equal(mi_trace["update_sign"], manual_trace["update_sign"]))

    def test_history_component_identity(self):
        torch.manual_seed(2)
        attacker = make_attacker(use_momentum=True, momentum_decay=1.0)
        images = torch.randn(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        _rows, tensors, _ = trace_attack(attacker, images, labels, "control_mi", keep_steps={1, 2})
        for trace in tensors:
            expected = trace["momentum_after"] - trace["grad_after_rotation"]
            self.assertTrue(torch.allclose(trace["history_component"], expected))

    def test_report_and_outputs_smoke(self):
        torch.manual_seed(3)
        attacker = make_attacker(use_momentum=True, momentum_decay=1.0)
        images = torch.randn(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        rows, _tensors, _ = trace_attack(attacker, images, labels, "control_mi", keep_steps={1, 2})
        report = aggregate_report(rows, minimal_args())
        self.assertIn("control_mi", report["source_summary"])
        self.assertIn("MI", build_conclusion_zh(report))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(rows, report, out)
            self.assertTrue((out / "mi_gradient_accumulation_metrics.csv").exists())
            self.assertTrue((out / "mi_gradient_accumulation_report.json").exists())
            self.assertTrue((out / "mi_gradient_accumulation_conclusion_zh.md").exists())


if __name__ == "__main__":
    unittest.main()
