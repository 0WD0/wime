#!/usr/bin/env python3
"""Ablate author history on frozen Japanese candidate pools.

Use prepare-japanese-context.py's contexts and *.genrank.json candidate pools.
Only Qwen scores change: z + a * ((1-b) * q_none + b * q_history). Select b
on dev NFKC accuracy (ties favor less history), then apply once to test.
This research evaluates the supplied candidates; it does not validate their
readings or make unconstrained generated candidates suitable for live input.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import unicodedata

MODES = ("none", "random", "retrieved")
STRENGTHS = (0, .25, .5, .75, 1)
NEUTRAL = "これは日本語の文章です。\n"


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def nfkc(text):
    return unicodedata.normalize("NFKC", text)


def paired_cases(contexts, pools, split):
    """Pair by explicit case index; repeated kana remain distinct examples."""
    cases = sorted((c for c in contexts["cases"] if c["split"] == split), key=lambda c: c["case_index"])
    if len(cases) != len(pools):
        raise ValueError(f"{split}: {len(cases)} contexts but {len(pools)} pool rows")
    for index, (case, pool) in enumerate(zip(cases, pools)):
        if case["case_index"] != index or case["kana"] != pool["kana"]:
            raise ValueError(f"{split} row {index}: context/pool alignment differs")
        # Older wime_sets.py caches called the kana string "id".
        if "id" in pool and str(pool["id"]) not in (str(case["id"]), pool["kana"]):
            raise ValueError(f"{split} row {index}: case IDs differ")
        cands, z = pool["cands"], pool["z"]
        if not cands or len(cands) != len(z) or not all(isinstance(t, str) and t for t in cands):
            raise ValueError(f"{split} row {index}: invalid candidate pool")
        if not all(math.isfinite(s) for s in z) or not pool["answers"]:
            raise ValueError(f"{split} row {index}: invalid scores or answers")
        yield case, pool


def score_context(history, original):
    """Keep up to 200 history characters, then the original 64-char context."""
    context = history[-200:] + original[-64:]
    japanese = any("ぁ" <= c <= "ヺ" or "㐀" <= c <= "鿿" or c in "々〆" for c in context)
    return context if japanese else NEUTRAL + context


class Scorer:
    """Causal sum-logprob, projecting only scored positions in small chunks."""

    def __init__(self, path, device="cuda", batch_size=16, head_chunk=128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, local_files_only=True).to(device).eval()
        self.device, self.batch_size, self.head_chunk = device, batch_size, head_chunk

    def score(self, context, texts):
        torch = self.torch
        with torch.inference_mode():
            prefix = self.tokenizer(context, add_special_tokens=False)["input_ids"]
            targets = [self.tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
            if not prefix or any(not t for t in targets):
                raise ValueError("context and candidates must tokenize to nonempty sequences")
            scores = []
            for start in range(0, len(targets), self.batch_size):
                batch = targets[start:start + self.batch_size]
                size = len(prefix) + max(map(len, batch)) - 1
                ids = torch.full((len(batch), size), self.tokenizer.pad_token_id or 0, dtype=torch.long)
                rows, positions, labels = [], [], []
                for i, target in enumerate(batch):
                    sequence = prefix + target[:-1]
                    ids[i, :len(sequence)] = torch.tensor(sequence)
                    rows.extend([i] * len(target))
                    positions.extend(range(len(prefix) - 1, len(prefix) - 1 + len(target)))
                    labels.extend(target)
                # Right padding follows every scored position, including in linear-attention layers.
                hidden = self.model.model(input_ids=ids.to(self.device), use_cache=False).last_hidden_state
                rows = torch.tensor(rows, device=self.device)
                positions = torch.tensor(positions, device=self.device)
                labels = torch.tensor(labels, device=self.device)
                total = torch.zeros(len(batch), device=self.device)
                for offset in range(0, len(labels), self.head_chunk):
                    sl = slice(offset, offset + self.head_chunk)
                    logits = self.model.lm_head(hidden[rows[sl], positions[sl]]).float()
                    logprob = -torch.nn.functional.cross_entropy(logits, labels[sl], reduction="none")
                    total.index_add_(0, rows[sl], logprob)
                scores.extend(total.tolist())
            return scores


def prediction(row, mode="none", strength=0, weight=.7):
    q = row["q"]
    index = max(range(len(row["cands"])),
                key=lambda i: row["z"][i] + weight * ((1 - strength) * q["none"][i] + strength * q[mode][i]))
    return row["cands"][index]


def metrics(rows, mode="none", strength=0, weight=.7):
    out = dict(total=len(rows), exact=0, nfkc=0, oracle_exact=0, oracle_nfkc=0,
               wins_exact=0, losses_exact=0, wins_nfkc=0, losses_nfkc=0)
    for row in rows:
        actual, baseline = prediction(row, mode, strength, weight), prediction(row, weight=weight)
        for label, normalize in (("exact", lambda s: s), ("nfkc", nfkc)):
            answers = set(map(normalize, row["answers"]))
            good, was_good = normalize(actual) in answers, normalize(baseline) in answers
            out[label] += good
            out["oracle_" + label] += any(normalize(c) in answers for c in row["cands"])
            out["wins_" + label] += good and not was_good
            out["losses_" + label] += was_good and not good
    return out


def summarize(rows, weight=.7):
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    if not dev or not test:
        raise ValueError("both complete dev and test results are required")
    report = dict(weight=weight, selection="dev NFKC accuracy, tie favors smaller history strength",
                  baseline={s: metrics(r, weight=weight) for s, r in (("dev", dev), ("test", test))}, modes={})
    for mode in MODES[1:]:
        sweep = {b: metrics(dev, mode, b, weight) for b in STRENGTHS}
        selected = max(STRENGTHS, key=lambda b: (sweep[b]["nfkc"], -b))
        report["modes"][mode] = dict(strength=selected, dev_sweep=sweep,
                                      dev=sweep[selected], test=metrics(test, mode, selected, weight))
    return report


def resume(path, metadata):
    """Read durable JSONL records, discarding only an interrupted final line."""
    if not path.exists():
        path.write_text(json.dumps(dict(type="metadata", **metadata), ensure_ascii=False) + "\n")
        return {}
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    lines = data[:end].splitlines()
    if not lines or json.loads(lines[0]) != dict(type="metadata", **metadata):
        raise ValueError("resume metadata differs; choose a new --out directory")
    results = {}
    for line in lines[1:]:
        row = json.loads(line)
        key = row["split"], row["case_index"]
        if row.get("type") != "result" or key in results:
            raise ValueError(f"invalid or repeated progress row {key}")
        results[key] = row
    if end < len(data):
        with path.open("r+b") as f:
            f.truncate(end)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("contexts", "dev-pool", "test-pool", "rescorer", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--weight", type=float, default=.7, help="Previously dev-selected Qwen weight; not tuned here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--head-chunk", type=int, default=128)
    args = ap.parse_args()
    if args.batch_size < 1 or args.head_chunk < 1 or not math.isfinite(args.weight):
        ap.error("batch sizes must be positive and weight must be finite")
    if not args.rescorer.is_dir():
        ap.error("--rescorer must be a local model directory")
    context = json.loads(args.contexts.read_text())
    tasks = []
    for split, path in (("dev", args.dev_pool), ("test", args.test_pool)):
        tasks.extend(paired_cases(context, json.loads(path.read_text()), split))
    model_files = sorted(p for p in args.rescorer.rglob("*")
                         if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".model", ".txt"))
    if not model_files:
        ap.error("--rescorer contains no model/tokenizer files")
    metadata = dict(version=1, inputs={str(p.resolve()): digest(p)
                                      for p in (args.contexts, args.dev_pool, args.test_pool, Path(__file__))},
                    model={str(p.relative_to(args.rescorer)): digest(p) for p in model_files},
                    runtime={name: importlib.metadata.version(name) for name in ("torch", "transformers")},
                    parameters=dict(weight=args.weight, device=args.device, batch_size=args.batch_size,
                                    head_chunk=args.head_chunk, history_chars=200, original_chars=64),
                    cases=len(tasks))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "progress.jsonl"
    done = resume(path, metadata)
    expected = {(c["split"], c["case_index"]): c for c, _ in tasks}
    for key, row in done.items():
        case = expected.get(key)
        if case is None or row["id"] != case["id"] or row["kana"] != case["kana"]:
            raise ValueError(f"resume row is not aligned: {key}")
    scorer = None
    with path.open("a") as f:
        for case, pool in tasks:
            key = case["split"], case["case_index"]
            if key in done:
                continue
            if scorer is None:
                scorer = Scorer(str(args.rescorer), args.device, args.batch_size, args.head_chunk)
            original = pool.get("context", "")
            q = {"none": scorer.score(score_context("", original), pool["cands"])}
            for mode in MODES[1:]:
                history = case["contexts"][mode]["context"]
                q[mode] = scorer.score(score_context(history, original), pool["cands"]) if history else q["none"]
            if not all(math.isfinite(s) for scores in q.values() for s in scores):
                raise ValueError(f"nonfinite model score at {key}")
            row = dict(type="result", split=case["split"], case_index=case["case_index"], id=case["id"],
                       kana=case["kana"], answers=pool["answers"], cands=pool["cands"], z=pool["z"], q=q,
                       has_history={m: bool(case["contexts"][m]["context"]) for m in MODES[1:]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[key] = row
            if len(done) % 25 == 0 or len(done) == len(tasks):
                print(json.dumps(dict(completed=len(done), total=len(tasks), split=key[0], case_index=key[1])), flush=True)
    report = summarize(list(done.values()), args.weight)
    report["input_digests"] = metadata["inputs"]
    report["scope"] = "Frozen candidate pools; same examples and candidates across modes; test is not used for strength selection."
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
