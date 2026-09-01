import importlib.util
import unittest


if importlib.util.find_spec("torch") is None:

    class ProgressiveViTDependencyTests(unittest.TestCase):
        @unittest.skip("PyTorch is not installed; install requirements.txt")
        def test_progressive_vit_dependencies(self):
            pass

else:
    from tests.vit_progressive_patch_score_attack_cases import *  # noqa: F401,F403
