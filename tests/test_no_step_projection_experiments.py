import argparse
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from attack import LazyAggregationAttacker
from lm_dss_no_step_projection_mi_ablation_s100 import BRANCHES, build_attack_cmd
from mi_no_step_projection_mechanism import (
    aggregate_report,
    make_attacker as make_full_attacker,
    trace_attack,
    write_outputs,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 16 * 16, 4)

    def forward(self, x, return_attn=False):
        return self.linear(x.flatten(1))


def make_trace_attacker(**kwargs):
    return LazyAggregationAttacker(
        TinyModel(),
        epsilon=0.1,
        step_size=0.1,
        steps=2,
        ti_sigma=0,
        layers=(-1,),
        guide_aug=False,
        lowmid_dss_filter=True,
        lowmid_dss_consistency="sign",
        lowmid_grad_tuning=True,
        lowmid_grad_rotation_strength=0.5,
        project_each_step=False,
        device=torch.device("cpu"),
        **kwargs,
    )


def args(**kwargs):
    values = {
        "max_samples": 100,
        "steps": 40,
        "batch_size": 8,
        "eval_batch_size": 128,
        "num_workers": 4,
        "prefetch_factor": 4,
        "output_dir": "unused",
        "image_dir": "unused",
        "annotations_path": "unused",
        "img_size": 16,
        "trace_samples": 2,
        "target_samples": 0,
        "trace_steps": (1, 2),
        "target_steps": (),
        "target_models": (),
        "seed": 0,
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


class NoStepProjectionExperimentTests(unittest.TestCase):
    def test_asr_dry_run_commands_force_no_step_projection_and_only_mi_differs(self):
        repo = Path.cwd()
        commands = {}
        for branch in BRANCHES:
            commands[branch.name] = build_attack_cmd(
                repo, args(), branch, repo / "outputs" / "attack" / "lazyagg" / "lm_dss_no_step_projection_s100" / branch.name
            )
        mi = commands["no_step_projection_aug_all_mi"]
        no_mi = commands["no_step_projection_aug_all_no_mi"]
        for cmd in (mi, no_mi):
            self.assertIn("--no-step-projection", cmd)
            self.assertEqual(cmd[cmd.index("--guide-aug-area") + 1], "all")
            self.assertIn("--lowmid-grad-tuning", cmd)
        self.assertIn("--mi", mi)
        self.assertIn("--mi-decay", mi)
        self.assertNotIn("--mi", no_mi)
        self.assertNotIn("--mi-decay", no_mi)

    def test_no_step_trace_update_omits_epsilon_projection(self):
        attacker = make_trace_attacker(use_momentum=False)
        images = torch.zeros(1, 3, 16, 16)
        labels = torch.tensor([1])
        attacker._attack_grad_terms = lambda pixels, _labels, _guide: (torch.ones_like(pixels), (torch.ones_like(pixels),))
        _rows, tensors, adv = trace_attack(attacker, images, labels, "no_step_projection_no_mi", keep_steps={1, 2})
        clean = attacker._denormalize(images)
        self.assertGreater((adv.cpu() - clean).abs().max().item(), attacker.epsilon)
        self.assertTrue(torch.allclose(tensors[-1]["x_next"] - tensors[-1]["clean"], torch.full_like(tensors[-1]["clean"], 0.2)))

    def test_manual_accumulator_matches_mi_update_sign_under_no_step_projection(self):
        torch.manual_seed(4)
        images = torch.randn(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        mi = make_trace_attacker(use_momentum=True, momentum_decay=1.0)
        manual = make_trace_attacker(use_momentum=False, momentum_decay=1.0)
        manual.model.load_state_dict(mi.model.state_dict())
        _mi_rows, mi_tensors, _ = trace_attack(mi, images, labels, "no_step_projection_mi", keep_steps={1, 2})
        _manual_rows, manual_tensors, _ = trace_attack(
            manual, images, labels, "manual_accumulator_no_mi_flag", manual_accumulator=True, keep_steps={1, 2}
        )
        for mi_trace, manual_trace in zip(mi_tensors, manual_tensors):
            self.assertTrue(torch.equal(mi_trace["update_sign"], manual_trace["update_sign"]))

    def test_report_outputs_handle_all_area_without_guide_map(self):
        rows = [
            {
                "scope": "source",
                "branch": "no_step_projection_mi",
                "step": 1,
                "history_norm": 0.0,
                "current_norm": 1.0,
                "update_norm": 1.0,
                "history_background_energy_ratio": float("nan"),
                "history_foreground_energy_ratio": float("nan"),
            }
        ]
        report = aggregate_report(rows, args())
        self.assertEqual(report["no_step_config"]["guide_aug_area"], "all")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(rows, report, out)
            self.assertTrue((out / "mi_no_step_projection_mechanism_metrics.csv").exists())
            self.assertTrue((out / "mi_no_step_projection_mechanism_report.json").exists())
            self.assertTrue((out / "mi_no_step_projection_mechanism_conclusion_zh.md").exists())

    def test_full_attacker_config_uses_all_area_and_no_projection(self):
        original = make_full_attacker.__globals__["build_vit_model"]
        try:
            make_full_attacker.__globals__["build_vit_model"] = lambda *a, **k: TinyModel()
            _source, attacker = make_full_attacker(4, mi=True, steps=2)
        finally:
            make_full_attacker.__globals__["build_vit_model"] = original
        self.assertEqual(attacker.guide_aug_area, "all")
        self.assertFalse(attacker.project_each_step)


if __name__ == "__main__":
    unittest.main()
