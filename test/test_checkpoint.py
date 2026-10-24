"""Checkpoint mixing must not silently combine different token semantics."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

directory = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("calibrate_checkpoint", directory / "calibrate-checkpoint.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CheckpointCompatibility(unittest.TestCase):
    def test_same_shape_reordered_vocabulary_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b, out = [Path(folder) / s for s in ("a", "b", "out")]
            info = dict(config="{}", vocab='["甲", "乙"]', conditional="wime-ms-pinyin-v1")
            weights = {"tok.weight": torch.zeros(2, 4, dtype=torch.float16)}
            save_file(weights, str(a), metadata=info)
            save_file(weights, str(b), metadata=dict(info, vocab='["乙", "甲"]'))
            with self.assertRaises(ValueError):
                module.interpolate(str(a), str(b), .5, out)
            self.assertFalse(out.exists())

    def test_endpoints_reload_exactly_and_retain_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b, out = [Path(folder) / s for s in ("a", "b", "out")]
            info = dict(config="{}", vocab='["甲", "乙"]', conditional="wime-ms-pinyin-v1")
            for path, value in ((a, .125), (b, .25)):
                save_file({"tok.weight": torch.full((2, 4), value, dtype=torch.float16)}, str(path), metadata=info)
            for alpha, source in ((0, a), (1, b)):
                module.interpolate(str(a), str(b), alpha, out)
                with safe_open(str(out), framework="pt") as actual, safe_open(str(source), framework="pt") as expected:
                    self.assertTrue(torch.equal(actual.get_tensor("tok.weight"), expected.get_tensor("tok.weight")))
                    self.assertIn("parent_checkpoints", actual.metadata())
                    self.assertEqual(actual.metadata()["vocab"], info["vocab"])


if __name__ == "__main__":
    unittest.main()
