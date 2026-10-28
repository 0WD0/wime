"""Packing and output-only projection must preserve the training objective."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from conditional import CharLM, Config, EXTRA, encode_example
from replay import Protection
from training_data import pack_examples, supervised_logits, training_characters


class TrainingData(unittest.TestCase):
    def test_projection_matches_loss_and_gradients(self):
        torch.manual_seed(3)
        model = CharLM(Config(vocab=13, context=10, n_embd=8, n_head=2, n_layer=2, dropout=0))
        other = copy.deepcopy(model)
        inputs = torch.randint(13, (3, 10))
        labels = torch.randint(1, 13, (3, 10))
        labels[:, :7] = 0
        logits, _ = model(inputs)
        full = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=0)
        picked, targets = supervised_logits(other, inputs, labels)
        sparse = F.cross_entropy(picked, targets)
        torch.testing.assert_close(full, sparse)
        full.backward()
        sparse.backward()
        for left, right in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)

    def test_file_backed_batches_protection_cache_and_train_only_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(context="上文", input="jayi", gold=["甲乙"], split="train", category="technical"),
                    dict(context="", input="jayi", gold=["甲乙"], split="train", category="chat")]
            source = root / "train.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            (root / "dev.jsonl").write_text(json.dumps(dict(rows[0], gold=["丙"], split="dev")) + "\n")
            self.assertEqual(training_characters([root]), ["乙", "甲"])
            vocab = ["<pad>", "<unk>", "<bos>"] + EXTRA + list("上文甲乙")
            index = {c: i for i, c in enumerate(vocab)}
            data = pack_examples(source, root / "cache", vocab, 32, Protection([]))
            self.assertIsInstance(data.data, np.memmap)
            x, y = data.batch(np.array([1, 0, 1]), "cpu")
            for i, j in enumerate([1, 0, 1]):
                a, b = encode_example(rows[j], index, 32)
                self.assertEqual(x[i, :len(a)].tolist(), a)
                self.assertEqual(y[i, :len(b)].tolist(), b)
            self.assertEqual(data.categories.tolist(), [2, 1])
            again = pack_examples(source, root / "cache", vocab, 32, Protection([]))
            self.assertEqual(again.data.filename, data.data.filename)
            heldout = root / "heldout.json"
            heldout.write_text(json.dumps([rows[0]]))
            protected = pack_examples(source, root / "cache", vocab, 32, Protection([heldout]))
            self.assertEqual(len(protected), 1)
            self.assertEqual(protected.info["exclusions"], {"protected_source": 1})
            source.write_text(json.dumps(rows[1]) + "\n")
            changed = pack_examples(source, root / "cache", vocab, 32, Protection([]))
            self.assertNotEqual(changed.data.filename, data.data.filename)


if __name__ == "__main__":
    unittest.main()
