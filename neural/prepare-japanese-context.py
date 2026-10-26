#!/usr/bin/env python3
"""Build a frozen pre-2025 author-history context ablation, without model calls."""
import argparse
import collections
import csv
import hashlib
import heapq
import importlib.util
import json
import random
import unicodedata
from pathlib import Path


def nfkc(s):
    return unicodedata.normalize("NFKC", s)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / ".cache/wime-japanese")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    parser_path = Path(__file__).with_name("prepare-japanese.py")
    spec = importlib.util.spec_from_file_location("prep", parser_path)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    cutoff = "2025-01-01 00:00:00"
    source = args.root / "sources/jpn_sentences_detailed.tsv"
    transcription = args.root / "sources/jpn_transcriptions.tsv"
    test_path, dev_path = [args.root / f"work/data/{name}.json" for name in ("test", "dev")]
    cases = json.loads(test_path.read_text())
    dev = json.loads(dev_path.read_text())
    # Gold is used only for this global contamination guard, never to rank history.
    protected = {nfkc(c["gold"]) for c in cases + dev}
    rows = {r[0]: dict(id=r[0], text=r[2], author=r[3], added=r[4], modified=r[5])
            for r in csv.reader(open(source), delimiter="\t", quoting=csv.QUOTE_NONE)}
    users = {rows[c["id"]]["author"] for c in cases + dev} - {"", "\\N"}
    stats = collections.Counter()
    candidates = {}
    for sid, r in rows.items():
        if r["author"] not in users:
            continue
        if r["added"] == "\\N" or r["modified"] == "\\N":
            stats["history_unknown_date_excluded"] += 1
            continue
        if r["added"] >= cutoff or r["modified"] >= cutoff:
            stats["history_not_frozen_before_cutoff_excluded"] += 1
            continue
        if nfkc(r["text"]) in protected:
            stats["history_heldout_normalized_text_excluded"] += 1
            continue
        candidates[sid] = r
    for r in csv.reader(open(transcription), delimiter="\t", quoting=csv.QUOTE_NONE):
        if r[2] != "Hrkt" or r[0] not in candidates:
            continue
        try:
            surface, kana, _ = prep.parse(r[4])
        except (ValueError, IndexError):
            stats["history_bad_transcription_excluded"] += 1
            continue
        if surface != candidates[r[0]]["text"] or not kana:
            stats["history_transcription_mismatch_excluded"] += 1
            continue
        candidates[r[0]]["kana"] = nfkc(kana)
    # Deduplicate normalized history text within author, deterministically oldest first.
    by_user = collections.defaultdict(list)
    seen = set()
    for r in sorted(candidates.values(), key=lambda r: (r["added"], r["id"])):
        if "kana" not in r:
            continue
        key = r["author"], nfkc(r["text"])
        if key in seen:
            stats["history_normalized_duplicates_excluded"] += 1
            continue
        seen.add(key)
        by_user[r["author"]].append(r)
    stats["eligible_history_sentences"] = sum(map(len, by_user.values()))
    grams = lambda s: collections.Counter(s[j:j + 2] for j in range(len(s) - 1))
    inverted = collections.defaultdict(lambda: collections.defaultdict(list))
    for user, history in by_user.items():
        for index, r in enumerate(history):
            for gram, count in grams(r["kana"]).items():
                inverted[user][gram].append((index, count))

    def selection(history, indices, score=None):
        picked = [dict(history[i], overlap_score=score.get(i, 0) if score is not None else None)
                  for i in indices]
        text = "\n".join(r["text"] for r in picked)
        # Preserve historical experiment's character budget, with exact context recorded.
        return dict(context=text[-200:] + ("\n" if text else ""), selected=picked)

    output = []
    for split, index, c in [(split, i, c) for split, group in (("test", cases), ("dev", dev))
                            for i, c in enumerate(group)]:
        target = rows[c["id"]]
        user, added, kana = target["author"], target["added"], nfkc(c["input"])
        history = by_user[user]
        eligible = {i for i, r in enumerate(history)
                    if added != "\\N" and r["added"] < added and r["modified"] < added and r["kana"] != kana}
        stats["case_same_reading_exclusions"] += sum(r["kana"] == kana for r in history)
        scores = collections.Counter()
        for g, n in grams(kana).items():
            for i, count in inverted[user].get(g, []):
                if i in eligible:
                    scores[i] += min(n, count)
        retrieved = heapq.nsmallest(3, eligible, key=lambda i: (-scores[i], i))
        random_ids = random.Random("wime-context-audit-v1:" + c["id"]).sample(sorted(eligible), min(3, len(eligible)))
        contexts = dict(none=dict(context="", selected=[]),
                        retrieved=selection(history, retrieved, scores),
                        random=selection(history, random_ids))
        if eligible:
            stats[f"{split}_cases_with_history"] += 1
        else:
            stats[f"{split}_cases_without_history"] += 1
        for mode in ("retrieved", "random"):
            # Audit annotation only; not used in retrieval or in score selection.
            if any(any(c["gold"][j:j+8] in r["text"] for j in range(max(0, len(c["gold"])-7)))
                   for r in contexts[mode]["selected"]):
                stats[f"{split}_{mode}_cases_sharing_gold_8gram"] += 1
        output.append(dict(split=split, case_index=index, id=c["id"], kana=c["input"], author=user, added=added,
                           eligible_history_count=len(eligible), contexts=contexts))
    result = dict(protocol=dict(version=1, cutoff=cutoff, history=3, max_characters=200,
                               score="kana bigram multiset overlap", random_seed="wime-context-audit-v1:ID",
                               excluded="unknown authors/dates, added/modified >= cutoff, normalized test/dev text, same input kana, mismatched transcription surface",
                               limitation="Sentence modification dates are checked; transcription edits have no history in this dump."),
                  provenance=[dict(path=str(p), sha256=digest(p))
                              for p in (source, transcription, test_path, dev_path, parser_path, Path(__file__))],
                  statistics=dict(stats), cases=output)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(path=str(args.out), cases=len(output), statistics=dict(stats)), ensure_ascii=False))


if __name__ == "__main__":
    main()
