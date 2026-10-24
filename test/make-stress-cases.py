#!/usr/bin/env python3
"""Deterministic failure probes, never training data or a natural-usage score.

Suffixes reuse the original reading, including polyphonic words. Auxiliary
keys come from the actual character table. Beyond-window perturbations have
explicit equivalence controls; other variations are robustness probes only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from conditional import composition_span


def variants(row, db, full=True):
    target, code = row["gold"][0], row["input"]
    assert len(code) == 2 * len(target) and not any(c.isupper() for c in code)
    base = dict(row, parent=row["id"], family=row.get("family", row["id"]), variant="whole")
    yield base
    for start in range(1, len(target)):
        yield dict(composition_span(base, start, len(target)),
                   id=row["id"] + f"/split-{start}", variant="commit_suffix", start=start)
    # Isolate two/three-character fragments inside words, not only final suffixes.
    if not full:
        for length in (2, 3):
            for start in range(1, len(target) - length):
                yield dict(composition_span(base, start, start + length),
                           id=row["id"] + f"/span-{start}-{length}",
                           variant="commit_span", start=start)
        return
    context = row["context"]
    contexts = {
        "space": context + " ",
        "newline": context + "\n",
        "paragraph": context + "\n\n",
        "english_prefix": "Previous topic: compiler performance and benchmark scores. " + context,
        "topic_prefix": "上个话题是模型跑分、价格和网络争论。\n" + context,
        "emoji_prefix": "🙂👩‍💻🧪。" + context,
        "delimiter_prefix": "\ue2e0\ue2e1" + context,
        "window_control": context[-64:].rjust(64),
        "outside_window": "这段历史不应影响输入结果。" * 20 + context[-64:].rjust(64),
    }
    for variant, ctx in contexts.items():
        item = dict(base, id=row["id"] + "/" + variant, context=ctx, variant=variant)
        if variant == "outside_window":
            item["equivalent_to"] = row["id"] + "/window_control"
        yield item
    # One or two auxiliary keys at both ends; keys constrain the corresponding
    # syllable only, so no target pronunciation is reconstructed at inference.
    for position in sorted({0, len(target) - 1}):
        entries = db.execute("SELECT word, aux FROM entry WHERE code=? ORDER BY weight DESC LIMIT 64",
                             (code[2 * position:2 * position + 2],)).fetchall()
        aux = next((a.split(",")[0] for word, a in entries if word == target[position]), None)
        if aux:
            for length in range(1, min(2, len(aux)) + 1):
                typed = code[:2 * position + 2] + aux[:length].upper() + code[2 * position + 2:]
                yield dict(base, id=row["id"] + f"/aux-{position}-{length}",
                           input=typed, variant="auxiliary")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--seeds", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-domain", type=int, default=96)
    args = p.parse_args()
    selected, counts = [], Counter()
    # Stable input order, fixed per-domain quota, selected before any predictions.
    for row in json.loads(args.heldout.read_text()):
        category = row["category"]
        if category in ("chat", "kubernetes", "python") and counts[category] < args.per_domain:
            selected.append(dict(row, id="heldout/" + row["id"], category="split_" + category))
            counts[category] += 1
    seeds = json.loads(args.seeds.read_text())
    db = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    rows = [v for row in selected for v in variants(row, db, full=False)]
    rows += [v for row in seeds for v in variants(row, db)]
    db.close()
    assert len({r["id"] for r in rows}) == len(rows)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    manifest = dict(heldout=str(args.heldout), seeds=str(args.seeds),
                    heldout_sha256=hashlib.sha256(args.heldout.read_bytes()).hexdigest(),
                    seeds_sha256=hashlib.sha256(args.seeds.read_bytes()).hexdigest(),
                    cases_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                    sources=dict(counts), manual_seeds=len(seeds), cases=len(rows),
                    variants=dict(Counter(r["variant"] for r in rows)),
                    limitation="Correlated synthetic probes and known held-out documents; not independent human keystrokes or a natural-usage accuracy estimate.")
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
