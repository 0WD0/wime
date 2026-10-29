"""Packing and output-only projection must preserve the training objective."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from conditional import CharLM, Config, EXTRA, encode_example
from replay import Protection
from training_data import pack_examples, pack_trajectories, sentence_views, supervised_logits, training_characters


class TrainingData(unittest.TestCase):
    def test_sentence_controls_share_groups_and_rng_without_revealing_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(id=str(i),document=str(i),group=str(i),source='chat',split='train',
                         context='允许',input='bmyix;ui',gold=['便宜行事'],legacy_span=[2,4],
                         boundaries=[dict(end=2,reason='complete phrase')] if i<4 else []) for i in range(5)]
            path=root/'train.jsonl'
            rows[0]['category']='technical'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            vocab=['<pad>','<unk>','<bos>']+EXTRA+list('允许便宜行事')
            data=pack_trajectories(path,root/'cache',vocab,32,Protection([]),2,'trajectories')
            control=pack_trajectories(path,root/'cache',vocab,32,Protection([]),2,'legacy-spans')
            self.assertEqual(len(data),5)
            self.assertEqual(data.categories.tolist(),[2,1,1,1,1])
            self.assertEqual(data.info['expected_state_shares']['chat'],dict(whole=.6,prefix=.2,continuation=.2))
            self.assertEqual(data.data.filename,control.data.filename)
            self.assertEqual(sentence_views(rows[0])[2]['context'],'允许')
            self.assertEqual(sentence_views(rows[0])[3]['known_reading'],dict(text='便宜',input='bmyi'))
            with patch('torch.rand',return_value=torch.tensor([.1,.6,.9,.9])):
                x,y=data.batch(np.array([0,0,0,4]),'cpu')
            index={c:i for i,c in enumerate(vocab)}
            for actual,text in zip(y,['便宜行事','便宜','行事','便宜行事']):
                self.assertEqual(actual[actual!=0].tolist(),[index[c] for c in text]+[index['\ue2e2']])
            indices=np.arange(5)
            torch.manual_seed(5); data.batch(indices,'cpu'); after=torch.rand(3)
            torch.manual_seed(5); control.batch(indices,'cpu'); torch.testing.assert_close(after,torch.rand(3))
            anchored=pack_trajectories(path,root/'cache',vocab,32,Protection([]),0,'legacy-spans')
            self.assertNotEqual(anchored.data.filename,control.data.filename)
            held=root/'held.json';held.write_text(json.dumps([dict(rows[0],gold=['无关'])]))
            filtered=pack_trajectories(path,root/'cache',vocab,32,Protection([held]),2,'trajectories')
            self.assertEqual(len(filtered),4)
            dev_path=root/'dev.jsonl'
            dev_path.write_text(json.dumps(dict(rows[0],split='dev'))+'\n')
            held_dev=pack_trajectories(dev_path,root/'cache',vocab,32,Protection([held]),2,'trajectories')
            self.assertEqual(len(held_dev),1)

            for row in rows[1:]:row['boundaries']=[]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            sparse=pack_trajectories(path,root/'cache',vocab,32,Protection([]),2,'trajectories')
            with patch('torch.rand',return_value=torch.tensor([.1])):
                _,y=sparse.batch(np.array([0]),'cpu')
            self.assertEqual(y[y!=0].tolist(),[index[c] for c in '便宜行事']+[index['\ue2e2']])

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
            self.assertEqual(training_characters([root], Protection([heldout])), ["乙", "甲"])
            only_heldout = root / "all-heldout.json"
            only_heldout.write_text(json.dumps(rows))
            self.assertEqual(training_characters([root], Protection([only_heldout])), [])
            protected = pack_examples(source, root / "cache", vocab, 32, Protection([heldout]))
            self.assertEqual(len(protected), 1)
            self.assertEqual(protected.info["exclusions"], {"protected_source": 1})
            source.write_text(json.dumps(rows[1]) + "\n")
            changed = pack_examples(source, root / "cache", vocab, 32, Protection([]))
            self.assertNotEqual(changed.data.filename, data.data.filename)


if __name__ == "__main__":
    unittest.main()
