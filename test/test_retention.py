"""Reference retention must not train on context, input tokens or padding."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

directory = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("train_conditional", directory / "train-conditional.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class RetentionLoss(unittest.TestCase):
    def test_only_target_positions_have_a_gradient(self):
        student = torch.zeros(1, 4, 3, requires_grad=True)
        teacher = torch.tensor([[[0., 1., 2.]] * 4])
        labels = torch.tensor([[0, 1, 2, 0]])
        expected = training.reference_kl(student, teacher, labels)
        changed = teacher.clone()
        changed[:, [0, 3]] *= -100
        self.assertAlmostEqual(expected.item(), training.reference_kl(student, changed, labels).item())
        expected.backward()
        self.assertEqual(student.grad[:, [0, 3]].abs().sum().item(), 0)
        self.assertGreater(student.grad[:, [1, 2]].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
