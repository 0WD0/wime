#!/usr/bin/env python3
"""Prepare bounded, document-disjoint MNBVC/ChID composition corpora.

Only data files are downloaded; no upstream software is executed or vendored.
The pinned source manifest and hashes permit replay without dataset scripts.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import urllib.request

from opencc import OpenCC
from pypinyin import lazy_pinyin

from conditional import augment_composition, composition_span

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
from wime_gen import Converter, load_rules

HAN = re.compile(r"[\u4e00-\u9fff]+")
BOILERPLATE = re.compile(r"https?://|www\.|责任编辑|点击下载|登录注册|版权声明|编辑本段|参考文献|外部链接")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalized(text):
    return "".join(HAN.findall(text))


def restore_idioms(row):
    """Return restored prose and offsets; choices never enter the prose."""
    pieces = row["content"].split("#idiom#")
    gold = row["groundTruth"]
    if len(pieces) != len(gold) + 1 or row["realCount"] != len(gold):
        raise ValueError("misaligned ChID labels")
    text, spans = pieces[0], []
    for word, suffix in zip(gold, pieces[1:]):
        if not HAN.fullmatch(word):
            raise ValueError("non-Chinese idiom label")
        spans.append((len(text), len(text) + len(word)))
        text += word + suffix
    return text, spans


class BoundedRows:
    """Stable lowest-hash sampling, independent of file order and random state."""
    def __init__(self):
        self.heaps = defaultdict(list)
        self.ids = set()

    def add(self, row):
        key = row["split"], row["category"]
        limits = {"idiom": 180000, "chid_prose": 70000,
                  "encyclopedia": 200000, "entity": 90000, "news": 60000}
        limit = limits[key[1]] if key[0] == "train" else 1800
        identity = digest(row["context"] + "\0" + row["gold"][0])
        if identity in self.ids:
            return
        self.ids.add(identity)
        row["id"] = identity[:24]
        item = (-int(identity, 16), row)
        heap = self.heaps[key]
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


class Protection:
    """Exclude held-out context/answer overlaps, not common target words."""
    def __init__(self, paths):
        self.exact, self.fragments = set(), set()
        self.files = {}
        for path in paths:
            raw = path.read_text()
            rows = json.loads(raw) if raw.lstrip().startswith("[") else map(json.loads, raw.splitlines())
            self.files[str(path)] = digest(raw)
            for row in rows:
                for target in row["gold"]:
                    context = normalized(row.get("context", ""))
                    target = normalized(target)
                    self.exact.add((context[-64:], target))
                    # Avoid treating a common word/idiom alone as contamination.
                    combined = context[-24:] + target
                    if len(context) >= 8 and len(combined) >= 24:
                        self.fragments.add(combined[-24:])

    def overlaps(self, text):
        text = normalized(text)
        return any(text[i:i + 24] in self.fragments for i in range(len(text) - 23))

    def contains(self, row):
        return (normalized(row["context"])[-64:], normalized(row["gold"][0])) in self.exact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, default=Path(__file__).with_name("corpus-sources.json"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--algebra", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--protect", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fetch", action="store_true")
    args = p.parse_args()
    sources = json.loads(args.sources.read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    for source in sources:
        path = args.cache / source["local"]
        if not path.exists() and args.fetch:
            with urllib.request.urlopen(source["url"], timeout=120) as response, path.with_suffix(".part").open("wb") as out:
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
            path.with_suffix(".part").rename(path)
        if hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != source["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
    cc = OpenCC("t2s")
    protected, selected, counts = Protection(args.protect), BoundedRows(), Counter()
    seen_documents, seen_paragraphs = set(), set()

    def add(text, start, end, split, category, document, source):
        target = text[start:end]
        if not 1 <= len(target) <= 24 or not HAN.fullmatch(target):
            return
        context = text[max(0, start - 64):start]
        row = dict(context=context, gold=[target], split=split, category=category,
                   document=document, source=source)
        if protected.contains(row):
            counts["protected_pair"] += 1
        else:
            selected.add(row)

    def document_rows(text, split, category, document, source, anchors=()):
        if protected.overlaps(text):
            counts["protected_document"] += 1
            return
        identity = digest(normalized(text))
        if identity in seen_documents:
            counts["duplicate_document"] += 1
            return
        seen_documents.add(identity)
        counts[category + "/" + split + "/documents"] += 1
        for start, end in anchors:
            add(text, start, end, split, "idiom" if category == "chid_prose" else "entity", document, source)
        matches = []
        offset = 0
        for para in text.splitlines(keepends=True):
            phash = digest(normalized(para))
            if phash not in seen_paragraphs:
                seen_paragraphs.add(phash)
                for m in re.finditer(r"[\u4e00-\u9fff]{4,24}", para):
                    if len(set(m.group())) < 3 or re.search(r"(.)\1{3}", m.group()):
                        continue
                    start, end = offset + m.start(), offset + m.end()
                    matches.append((digest(text[max(0, start - 64):end]), start, end))
            else:
                counts["duplicate_paragraph"] += 1
            offset += len(para)
        # Limit prolific documents rather than let long articles dominate.
        for _, start, end in sorted(matches)[:4]:
            add(text, start, end, split, category, document, source)

    # Official ChID dev is reserved before training to remove copied passages.
    sources.sort(key=lambda s: ("dev_data" not in s["path"], s["repo"], s["path"]))
    for source in sources:
        path = args.cache / source["local"]
        chid = source["repo"] == "thu-coai/chid"
        with (path.open() if chid else gzip.open(path, "rt")) as f:
            for line in f:
                row = json.loads(line)
                if chid:
                    text, anchors = restore_idioms(row)
                    # Conversion must not move the annotated offsets.
                    converted = cc.convert(text)
                    if len(converted) != len(text):
                        counts["conversion_alignment"] += 1
                        continue
                    text = converted
                    document = digest(normalized(text))
                    split = ("dev" if int(document[:8], 16) % 2 else "test") if "dev_data" in path.name else "train"
                    if "�" in text or re.search(r"[\u4e00-\u9fff]\?[\u4e00-\u9fff]", text):
                        counts["corrupt_text"] += 1
                        continue
                    document_rows(text, split, "chid_prose", document, source["path"], anchors)
                else:
                    if row.get("是否待查文件") or row.get("是否重复文件"):
                        counts["flagged_document"] += 1
                        continue
                    wiki = source["path"].startswith("wiki/")
                    title = cc.convert(row["段落"][0]["内容"]).strip("【】 ") if wiki and row["段落"] else ""
                    paragraphs = []
                    for para in row["段落"]:
                        text = cc.convert(para["内容"]).strip()
                        if (para.get("是否重复") or para.get("是否跨文件重复")
                                or len(text) < 20 or not re.search(r"[。！？；]", text)
                                or len(normalized(text)) / len(text) < .65
                                or BOILERPLATE.search(text) or "�" in text):
                            counts["filtered_paragraph"] += 1
                            continue
                        paragraphs.append(text)
                    text = "\n".join(paragraphs)
                    if not text:
                        continue
                    document = digest("wiki:" + title) if wiki else digest(normalized(text))
                    bucket = int(document[:8], 16) % 100
                    split = "test" if bucket < 2 else "dev" if bucket < 4 else "train"
                    anchors = []
                    if wiki and HAN.fullmatch(title) and 2 <= len(title) <= 10:
                        anchors = [(m.start(), m.end()) for m in re.finditer(re.escape(title), text)][:2]
                    document_rows(text, split, "encyclopedia" if wiki else "news", document, source["path"], anchors)
        print(json.dumps(dict(source=path.name, counts=counts, selected={str(k):len(v) for k,v in selected.heaps.items()})), flush=True)

    rows = [row for heap in selected.heaps.values() for _, row in heap]
    # For whole dictionary words, use an unambiguous dictionary reading. This
    # especially helps names and polyphonic idioms; other readings are automatic.
    targets = {r["gold"][0] for r in rows}
    readings = defaultdict(set)
    with sqlite3.connect(f"file:{args.database}?mode=ro", uri=True) as db:
        for word, code in db.execute("SELECT word, code FROM entry"):
            if word in targets and len(code) == len(word) * 2:
                readings[word].add(code)
    converter = Converter(rules=load_rules(args.algebra))

    @lru_cache(maxsize=None)
    def syllable(py):
        codes = sorted(c for c in converter.codes(py) if len(c) == 2)
        return codes[0] if codes else ""

    @lru_cache(maxsize=None)
    def encode(target):
        choices = readings.get(target, set())
        if len(choices) == 1:
            return next(iter(choices)), "dictionary_unique"
        code = "".join(syllable(py) for py in lazy_pinyin(target))
        return code, "automatic"

    args.out.mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    for row in sorted(rows, key=lambda r:r["id"]):
        target = row["gold"][0]
        code, method = encode(target)
        if len(code) != 2 * len(target):
            counts["invalid_reading"] += 1
            continue
        row.update(input=code, reading=method)
        if row["split"] == "dev":
            row = augment_composition(row, random.Random(row["id"]), .3, .15, .1)
        if protected.contains(row):
            counts["protected_derived_pair"] += 1
            continue
        by_split[row["split"]].append(row)
        counts[row["split"] + "/" + row["category"] + "/rows"] += 1
        counts[row["split"] + "/reading/" + method] += 1
    for split, records in by_split.items():
        with (args.out / (split + ".jsonl")).open("w") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Independently sourced evaluation: full words, suffixes and single chars.
    evaluation = []
    for category in sorted({r["category"] for r in by_split["test"]}):
        for row in [r for r in by_split["test"] if r["category"] == category][:128]:
            n = len(row["gold"][0])
            for variant, start, end in (("whole", 0, n), ("suffix", n // 2, n), ("single", n - 1, n)):
                r = composition_span(row, start, end)
                r.update(id=row["id"] + "/" + variant, parent=row["id"], variant=variant)
                if not protected.contains(r):
                    evaluation.append(r)
    (args.out / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n")
    manifest = dict(sources=sources, counts=counts, protected_files=protected.files,
                    sampling="lowest SHA256; per-document maximum 4 general chunks and 2 entity mentions",
                    split="ChID official train vs dev; official dev hash halves for new dev/test; wiki title/news document SHA256 96/2/2",
                    limitations="Automatic pronunciation except unambiguous dictionary words; pretraining overlap unknown. Corpus words may occur in frozen tests; exact context/answer pairs and 24-character overlaps excluded.",
                    pypinyin="0.55.0", conversion="OpenCC t2s",
                    database_sha256=hashlib.file_digest(args.database.open("rb"), "sha256").hexdigest(),
                    files={f.name:hashlib.file_digest(f.open("rb"), "sha256").hexdigest() for f in args.out.glob("*.json*") if f.name != "manifest.json"})
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
