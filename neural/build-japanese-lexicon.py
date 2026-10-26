#!/usr/bin/env python3
"""Words for the Japanese reading table beyond JMdict's common entries.

Runs MeCab + UniDic 3.1 over the Aozora, Wikinews and Wikipedia sources of
prepare-japanese.py (never Tatoeba, which holds the test sentences) and keeps
  - UniDic words with kanji and their 'kana' readings (conjugated forms too),
  - JMdict and JMnedict writings with kanji, with the dictionary's readings,
that occur at least --min-count times, as {writing: {reading: count}};
build_table() keeps the most frequent few per reading. JMnedict (names, places, works) is
EDRDG's, CC BY-SA 4.0; see japanese-sources.json.
"""
import argparse
import collections
import gzip
import importlib.util
import json
import multiprocessing
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).with_name("prepare-japanese.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
hira, is_kanji = prepare.hira, prepare.kanji
LONGEST = 8  # UniDic words a dictionary writing may span


def jmdict(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    writings = collections.defaultdict(set)
    for writing, reading in db.execute(
            "SELECT kt.c0reading, nt.c0reading FROM kanji k JOIN kanjiText_content kt ON kt.docid = k.docid "
            "JOIN kana n ON n.id = k.id JOIN kanaText_content nt ON nt.docid = n.docid"):
        writings[writing].add(hira(reading))
    return writings


def jmnedict(path):
    writings = collections.defaultdict(set)
    with gzip.open(path, "rt") as f:
        for _, entry in ET.iterparse(f):
            if entry.tag != "entry":
                continue
            readings = {hira(r.text) for r in entry.iter("reb")}
            for k in entry.iter("keb"):
                if any(is_kanji(c) for c in k.text):
                    writings[k.text] |= readings
            entry.clear()
    return writings


def count(args):
    _, _, text = args
    reader = getattr(count, "reader", None) or prepare.Reader()
    count.reader = reader
    words, seen = collections.Counter(), collections.Counter()
    for line in text.split("\n"):
        for sentence in prepare.SPLIT.split(line.strip().replace("　", "")):
            tokens = reader.tokens(sentence) if sentence else None
            if not tokens:
                continue
            surfaces = [s for s, _ in tokens]
            for i, (s, k) in enumerate(tokens):
                if any(is_kanji(c) for c in s):
                    words[s, k] += 1
                    joined = ""
                    for t in surfaces[i:i + LONGEST]:
                        joined += t
                        if joined in count.known:
                            seen[joined] += 1
    return words, seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jmdict", default="/usr/share/tagainijisho/jmdict.db")
    ap.add_argument("--jmnedict", type=Path, default=prepare.SOURCES / "JMnedict.xml.gz")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    known = jmnedict(args.jmnedict)
    for w, readings in jmdict(args.jmdict).items():
        if any(is_kanji(c) for c in w):
            known[w] |= readings
    count.known = known  # inherited by the forked workers
    words, seen = collections.Counter(), collections.Counter()
    limits = dict(aozora=10 ** 10, wikipedia=10 ** 10, wikinews=10 ** 10)
    with multiprocessing.Pool(args.workers) as pool:
        for n, (w, s) in enumerate(pool.imap_unordered(count, prepare.documents(limits), chunksize=16)):
            words.update(w), seen.update(s)
            if n % 50000 == 0:
                print(n, "documents", len(words), "UniDic words", len(seen), "dictionary writings", flush=True)
    lexicon = collections.defaultdict(dict)
    for (s, k), n in words.items():
        if n >= args.min_count and re.fullmatch(r"[ぁ-ゖー]+", k):
            lexicon[s][k] = n
    for w, n in seen.items():
        if n >= args.min_count:
            for r in known[w]:
                lexicon[w][r] = max(n, lexicon[w].get(r, 0))
    args.out.write_text(json.dumps(dict(sorted(lexicon.items())), ensure_ascii=False))
    print(len(lexicon), "writings,", sum(map(len, lexicon.values())), "readings")


if __name__ == "__main__":
    main()
