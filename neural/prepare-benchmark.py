#!/usr/bin/env python3
"""Draw the public keystroke benchmark from Chinese Wikipedia articles no corpus used.

The cleaned corpus took the dump's first articles in order, so later articles
belong to no training, development or sealed set; the registry's evaluation
cases are excluded again by text. Each article gives one prose sentence of
6–24 Han characters ending in 。！？, with earlier text as left context.
Readings come from the evaluation sets' Mandarin frontend, automatic and
unreviewed, so accuracy on this set is a sanity check.
"""
import argparse
import bz2
import hashlib
import importlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from corpus_cleaning import clean_document, make_simplifier
from corpus_readings import Lexicon, Reader
from dataset_registry import Registry
from mandarin_frontend import MandarinFrontend
from replay import Protection, fingerprint

render_wiki = importlib.import_module("prepare-corpus").render_wiki
SOURCES = json.loads(Path(__file__).with_name("corpus-v2-sources.json").read_text())
BUCKETS = {"6-12": range(6, 13), "13-24": range(13, 25)}


def articles(dump):
    with bz2.open(dump) as stream:
        events = ET.iterparse(stream, events=("start", "end"))
        _, root = next(events)
        for event, element in events:
            if event != "end" or element.tag.rsplit("}", 1)[-1] != "page":
                continue
            page = dict(ns=element.find("{*}ns").text, redirect=element.find("{*}redirect") is not None,
                        title=element.find("{*}title").text, page=int(element.find("{*}id").text),
                        revision=int(element.find("{*}revision/{*}id").text),
                        text=element.find("{*}revision/{*}text").text or "")
            element.clear()
            root.clear()
            if page["ns"] == "0" and not page["redirect"] and not page["text"].lstrip().lower().startswith("#redirect"):
                yield page


def sentences(page, text):
    for match in re.finditer(r"[^。！？\n]+[。！？]", text):
        raw = match.group()
        sentence = raw[:-1].strip()
        start = match.start() + len(raw) - len(raw.lstrip())
        if start and re.fullmatch(r"[\u4e00-\u9fff]+", sentence) and len(set(sentence)) >= 4:
            yield dict(id=hashlib.sha256(f"benchmark-v1:{page['page']}:{start}".encode()).hexdigest()[:20],
                       title=page["title"], page=page["page"], revision=page["revision"], offset=start,
                       context=text[max(0, start - 64):start], gold=[sentence])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dump", "database", "g2pw", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=50)
    args = parser.parse_args()
    registry = Registry()
    cleaned = json.loads(registry.path("cleaned-multisource-v4", "manifest").read_text())
    used = cleaned["stats"]["zhwiki/raw_documents"]
    suites = [name for name, entry in registry.datasets.items() if entry["kind"] in ("evaluation", "diagnostic")]
    protection = Protection([registry.evaluation(name)[0] for name in suites])
    golds = {gold for _, gold in protection.exact}
    t2s, _ = make_simplifier()
    # Reading drops a few candidates, so draw spares per bucket.
    pool = {bucket: [] for bucket in BUCKETS}
    for i, page in enumerate(articles(args.dump)):
        if i < used:
            continue
        text, _, rejection = clean_document("zhwiki", render_wiki(page["text"]), t2s)
        if rejection:
            continue
        open_buckets = [b for b, rows in pool.items() if len(rows) < args.per_bucket + 8]
        if not open_buckets:
            break
        for row in sorted(sentences(page, text), key=lambda row: row["id"]):
            bucket = next((b for b in open_buckets if len(row["gold"][0]) in BUCKETS[b]), None)
            if bucket and row["gold"][0] not in golds and not protection.contains(row):
                pool[bucket].append(row)
                break
    reader = Reader(Lexicon(args.database), MandarinFrontend(args.g2pw))
    cases = []
    for bucket, rows in pool.items():
        readings = reader.readings([row["gold"][0] for row in rows])
        kept = [dict(row, input="".join(reading["keys"])) for row, reading in zip(rows, readings)
                if all(reading["keys"])][:args.per_bucket]
        cases += kept
    args.out.write_text(json.dumps(dict(
        format="wime-keystroke-benchmark-v1",
        source=dict(name="Chinese Wikipedia", dump=SOURCES["sources"]["zhwiki"]["url"],
                    dump_sha1=SOURCES["sources"]["zhwiki"]["sha1"], license="CC BY-SA 4.0",
                    license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                    attribution="Each case quotes the article revision https://zh.wikipedia.org/w/index.php"
                                "?oldid=<revision>; its history lists the authors. Converted to simplified "
                                "Chinese with OpenCC; wiki markup removed."),
        selection=dict(skipped_articles=used, policy=__doc__.strip(),
                       buckets={b: sum(len(c["gold"][0]) in BUCKETS[b] for c in cases) for b in BUCKETS},
                       excluded_suites=suites),
        reading=dict(pipeline=SOURCES["reading"]["pipeline"], model=SOURCES["reading"]["model"],
                     tokenizer=SOURCES["reading"]["tokenizer"], scheme="Microsoft double pinyin",
                     database_sha256=fingerprint(args.database)),
        cases=cases), ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(cases=len(cases), out=str(args.out)), ensure_ascii=False))


if __name__ == "__main__":
    main()
