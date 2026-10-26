#!/usr/bin/env python3
"""Kana-conditioned Japanese training rows: (context, kana typed, text).

Sources and their licenses are pinned in japanese-sources.json; run with a
Python providing fugashi, UniDic 3.1 and pyarrow.

Readings: Tatoeba transcriptions (bracket furigana) as given; Aozora (modern
orthography only), Wikipedia and Wikinews through MeCab + UniDic 3.1 'kana'
(orthographic reading: particle は stays は). Punctuation, digits and Latin
letters without a reading are typed literally. Held-out Tatoeba sentences
(added 2025-2026) and the dev sample are excluded by id, by exact text and by
any shared 8-character run.
"""
import argparse
import collections
import csv
import gzip
import json
import multiprocessing
import os
import random
import re
import zlib
from pathlib import Path

SOURCES = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "wime-japanese/sources"
MAXTEXT, MAXCONTEXT, GRAM = 40, 32, 8


def hira(text):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def kanji(c):
    return "㐀" <= c <= "鿿" or c in "々〆"


def is_hiragana(c):
    return "ぁ" <= c <= "ゖ" or c in "ゝゞ"


def parse(transcription):
    """(surface, kana typed, shift): shift is '1' for kana a user would type
    with Shift held (kanji groups and katakana), else '0'."""
    surface, reading, shift, i = [], [], [], 0
    while i < len(transcription):
        if transcription[i] == "[":
            j = transcription.index("]", i)
            text, *parts = transcription[i + 1:j].split("|")
            surface.append(text)
            if len(parts) == len(text):  # one reading per character
                for c, r in zip(text, parts):
                    reading.append(hira(r)); shift.append(("0" if is_hiragana(c) else "1") * len(r))
            else:
                r = hira("".join(parts)); reading.append(r); shift.append("1" * len(r))
            i = j + 1
        else:
            c = transcription[i]
            surface.append(c); reading.append(hira(c))
            shift.append("1" if "ァ" <= c <= "ヺ" or c == "ー" and shift and shift[-1].endswith("1") else "0")
            i += 1
    return "".join(surface), "".join(reading), "".join(shift)


def tatoeba():
    """(held-out rows, dev rows, training (text, kana) pairs), selected as ja_decode.py does."""
    dates = {}
    for row in csv.reader(open(SOURCES / "jpn_sentences_detailed.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE):
        dates[row[0]] = ((row[4] if row[4] != r"\N" else "")[:4], row[2])
    recent, older, train = [], [], []
    for row in csv.reader(open(SOURCES / "jpn_transcriptions.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row[0] not in dates:
            continue
        year, text = dates[row[0]]
        surface, reading, shift = parse(row[4])
        if surface != text or not reading:
            continue
        item = dict(id=row[0], year=year or "none", gold=text, input=reading, shift=shift)
        if row[3]:
            (recent if year in ("2025", "2026") else older).append(item)
        elif year not in ("2025", "2026"):
            train.append(item)
    random.Random(0).shuffle(older)
    # older[:500] is the dev sample the Qwen-only run reported; [500:1000] spare.
    held = recent + older[:1000]
    train += older[1000:]
    return recent, older[:500], held, train


class Reader:
    def __init__(self):
        import fugashi
        self.tagger = fugashi.Tagger()

    def tokens(self, sentence):
        """[(surface, kana)] or None if a word with kanji has no reading."""
        out = []
        for w in self.tagger(sentence):
            s = w.surface
            kana = getattr(w.feature, "kana", None)
            if any(kanji(c) for c in s):
                if not kana or kana == "*":
                    return None
                out.append((s, hira(kana)))
            elif kana and kana != "*" and any("ぁ" <= c <= "ヺ" for c in s):
                out.append((s, hira(s)))
            elif kana and kana != "*" and s.isdigit():
                out.append((s, hira(kana)))
            else:
                out.append((s, s))
        return out


SPLIT = re.compile(r"(?<=[。！？!?])")


def segments(tokens):
    """Cut a token list into rows of at most MAXTEXT characters at punctuation, else word boundaries."""
    rows, text, kana = [], "", ""
    for s, k in tokens:
        if len(text) + len(s) > MAXTEXT and text:
            rows.append((text, kana)); text, kana = "", ""
        text += s; kana += k
        if s in "、。！？!?" and len(text) >= 12:
            rows.append((text, kana)); text, kana = "", ""
    if text:
        rows.append((text, kana))
    return rows


def document_rows(args):
    source, doc_id, text = args
    reader = getattr(document_rows, "reader", None) or Reader()
    document_rows.reader = reader
    rows, context = [], ""
    for line in text.split("\n"):
        line = line.strip().replace("　", "")
        if len(line) < 6 or not any(kanji(c) or "ぁ" <= c <= "ん" for c in line):
            context = ""
            continue
        for sentence in SPLIT.split(line):
            if not sentence:
                continue
            tokens = reader.tokens(sentence)
            if tokens is None:
                context = ""
                continue
            for t, k in segments(tokens):
                rows.append(dict(source=source, doc=doc_id, context=context[-MAXCONTEXT:], text=t, kana=k))
                context += t
        context += "\n"
    return rows


def documents(limits):
    import pyarrow.parquet as pq
    with gzip.open(SOURCES / "aozorabunko-dedupe-clean.jsonl.gz", "rt") as f:
        n = 0
        for line in f:
            r = json.loads(line)
            if r["meta"].get("文字遣い種別") == "新字新仮名":
                yield "aozora", r["meta"]["作品ID"], r["text"]
                n += len(r["text"])
                if n > limits["aozora"]:
                    break
    for path, name in [(SOURCES / "wikinews-ja.parquet", "wikinews")] + \
            [(p, "wikipedia") for p in sorted(SOURCES.glob("wikipedia-ja-*.parquet"))]:
        table = pq.read_table(path, columns=["text"] + (["id"] if name == "wikipedia" else []))
        texts = table.column("text").to_pylist()
        ids = table.column("id").to_pylist() if name == "wikipedia" else range(len(texts))
        n = 0
        for i, t in zip(ids, texts):
            yield name, f"{name}-{i}", t
            n += len(t)
            if n > limits[name]:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--aozora-chars", type=int, default=60_000_000)
    ap.add_argument("--wikipedia-chars", type=int, default=150_000_000)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    recent, dev, held, tatoeba_train = tatoeba()
    (args.out / "test.json").write_text(json.dumps(recent, ensure_ascii=False))
    (args.out / "dev.json").write_text(json.dumps(dev, ensure_ascii=False))
    protected_text = {r["gold"] for r in held}
    grams = {r["gold"][i:i + GRAM] for r in held for i in range(len(r["gold"]) - GRAM + 1)}

    def clean(text):
        return text not in protected_text and not any(text[i:i + GRAM] in grams for i in range(len(text) - GRAM + 1))

    counts = collections.Counter()
    rng = random.Random(1)
    with open(args.out / "train.jsonl", "w") as train, open(args.out / "valid.jsonl", "w") as valid:
        def write(row):
            if not clean(row["text"]):
                counts[row["source"] + "/protected"] += 1
                return
            split = valid if zlib.crc32(row["doc"].encode()) % 100 == 0 else train
            split.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[row["source"] + ("/valid" if split is valid else "/train")] += 1
            counts[row["source"] + "/chars"] += len(row["text"])
        for r in tatoeba_train:
            write(dict(source="tatoeba", doc=r["id"], context="", text=r["gold"], kana=r["input"]))
        limits = dict(aozora=args.aozora_chars, wikipedia=args.wikipedia_chars // 3, wikinews=10 ** 9)
        with multiprocessing.Pool(args.workers) as pool:
            for rows in pool.imap_unordered(document_rows, documents(limits), chunksize=8):
                for row in rows:
                    write(row)
    (args.out / "counts.json").write_text(json.dumps(counts, indent=1, sort_keys=True))
    print(json.dumps(counts, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
