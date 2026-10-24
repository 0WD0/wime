#!/usr/bin/env python3
"""Create reproducible dialog-disjoint conditional train/dev/test JSONL files.

The held-out set is independent of this fine-tuning run, NOT certified unseen
by the upstream pretrained model or octagram. Pinyin labels are automatic.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

from pypinyin import lazy_pinyin, Style

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
from wime_gen import Converter, load_rules


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--algebra", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dialogs", type=int, default=500000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    converter = Converter(rules=load_rules(args.algebra))
    # Assign dialogs first; exact target duplicates across splits are removed
    # globally from every split, never moved to a different dialog's split.
    rows, counts, owners = [], Counter(), {}
    with gzip.open(args.source, "rt") as source:
        for number, line in enumerate(source):
            if number >= args.dialogs:
                break
            value = json.loads(line)
            dialog = value if isinstance(value, list) else value["dialog"]
            dialog = [re.sub(r"\s+", "", text) for text in dialog]
            identity = digest("\n".join(dialog))
            bucket = int(identity[:8], 16) % 100
            split = "test" if bucket == 0 else "dev" if bucket == 1 else "train"
            for turn, text in enumerate(dialog):
                for match in re.finditer(r"[\u4e00-\u9fff]{6,32}", text):
                    target = match.group()
                    if len(set(target)) < 4 or re.search(r"(.)\1{3}", target):
                        counts["repetitive"] += 1
                        continue
                    # Deterministic target grouping also prevents identical
                    # fragments from otherwise different dialogs crossing splits.
                    key = digest(target)
                    previous = owners.get(key)
                    if previous is not None:
                        if rows[previous]["split"] != split:
                            rows[previous]["excluded"] = True
                        counts["duplicate"] += 1
                        continue
                    pinyin = lazy_pinyin(target, style=Style.NORMAL, errors="default")
                    codes = [sorted(converter.codes(p)) for p in pinyin]
                    if len(codes) != len(target) or any(not c for c in codes):
                        counts["unencodable"] += 1
                        continue
                    code = "".join(c[0] for c in codes)
                    if len(code) != 2 * len(target):
                        counts["unencodable"] += 1
                        continue
                    # Prior dialogue is useful context; target itself never is.
                    context = ((dialog[turn - 1] + "。") if turn else "") + text[:match.start()]
                    owners[key] = len(rows)
                    rows.append(dict(id=key[:20], document=identity, split=split,
                                     context=context[-64:], input=code,
                                     pinyin=" ".join(pinyin), gold=[target]))
            if number and number % 50000 == 0:
                print(f"{number} dialogs, {len(rows)} examples", file=sys.stderr, flush=True)
    # A dialog may contain a duplicate moved to holdout. Exclude all remaining
    # training examples from any dialog represented in either held-out split.
    heldout_documents = {r["document"] for r in rows if r["split"] != "train"}
    handles = {s: (args.out / f"{s}.jsonl").open("w") for s in ("train", "dev", "test")}
    for row in rows:
        if row.get("excluded"):
            counts["overlap_target_removed"] += 1
            continue
        if row["split"] == "train" and row["document"] in heldout_documents:
            counts["overlap_dialog_removed"] += 1
            continue
        handles[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[row["split"]] += 1
        counts[row["split"] + "_chars"] += len(row["gold"][0])
    for h in handles.values():
        h.close()
    manifest = dict(source="https://huggingface.co/datasets/silver/lccc",
                    upstream="https://github.com/thu-coai/CDial-GPT", license="MIT",
                    source_sha256=hashlib.file_digest(args.source.open("rb"), "sha256").hexdigest(),
                    algebra_sha256=hashlib.file_digest(args.algebra.open("rb"), "sha256").hexdigest(),
                    dialogs=args.dialogs, counts=dict(counts), pypinyin="0.55.0",
                    limitation="Automatically derived pinyin; upstream pretraining overlap unknown.")
    manifest["files"] = {s: hashlib.file_digest((args.out / f"{s}.jsonl").open("rb"), "sha256").hexdigest()
                         for s in handles}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
