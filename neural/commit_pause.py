"""Choose commit boundaries where a general LM expects a pause.

Among grammatical candidates from sentence_sampling.boundary_candidates, keep
the one after which Qwen most expects pause punctuation, if that probability
reaches PAUSE_THRESHOLD. The prefix is encoded on its own so the boundary is
always a token end. Selected on dev-v2 review and manual training samples.
"""
import torch

from sentence_sampling import REASONS

PAUSE_THRESHOLD = 0.02
PUNCTUATION = "，、；：。！？,;:!?"


class PauseScorer:
    def __init__(self, path, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.pause = torch.tensor([i for t, i in self.tokenizer.get_vocab().items()
                                   if self.tokenizer.convert_tokens_to_string([t])[:1] in PUNCTUATION], device=device)

    @torch.inference_mode()
    def __call__(self, contexts, prefixes, batch=128):
        """p(next token is pause punctuation | context + prefix) for each pair."""
        out, pad = [], self.tokenizer.pad_token_id or 0
        for start in range(0, len(prefixes), batch):
            seqs = [self.tokenizer(c[-64:] or "\n")["input_ids"] + self.tokenizer(p, add_special_tokens=False)["input_ids"]
                    for c, p in zip(contexts[start:start + batch], prefixes[start:start + batch])]
            ids = torch.full((len(seqs), max(map(len, seqs))), pad)
            for i, seq in enumerate(seqs):
                ids[i, :len(seq)] = torch.tensor(seq)  # right padding follows every scored position
            last = torch.tensor([len(seq) - 1 for seq in seqs], device=self.device)
            logits = self.model(input_ids=ids.to(self.device)).logits[torch.arange(len(seqs), device=self.device), last]
            out += logits.float().softmax(-1)[:, self.pause].sum(-1).tolist()
        return out


def choose(rows, scorer):
    """Set row['boundaries'] from row['boundary_candidates'] in place."""
    flat = [(row, rule, end) for row in rows for rule, end in row["boundary_candidates"]]
    scores = scorer([row["context"] for row, _, _ in flat], [row["gold"][0][:end] for row, _, end in flat])
    best = {}
    for (row, rule, end), score in zip(flat, scores):
        if score >= PAUSE_THRESHOLD and score > best.get(id(row), (None, -1))[1]:
            best[id(row)] = ((rule, end), score)
    for row in rows:
        chosen = best.get(id(row))
        row["boundaries"] = [] if chosen is None else [
            dict(end=chosen[0][1], rule=chosen[0][0], reason=REASONS[chosen[0][0]], pause=round(chosen[1], 4))]
