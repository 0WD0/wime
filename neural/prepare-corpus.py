#!/usr/bin/env python3
"""Build (left context, typed keys, committed text) rows from pinned corpora.

Spans follow word boundaries with a mix of single characters, words, phrases
and sentences; context is only earlier text of the same document, as an
editor sees it. Documents are split by their group (conversation, article,
question or thread). Readings come from corpus_readings; spans with any
character lacking a legal key are dropped, never re-read character by character.
"""
import argparse
import bz2
from collections import Counter
import glob
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

from corpus_readings import HAN, Lexicon, Reader
from replay import Protection

LENGTHS = ((0.15, 1, 1), (0.45, 2, 5), (0.25, 6, 12), (0.15, 13, 24))
SENTENCE = re.compile(r"[^。！？!?\n]*[。！？!?\n]?")
URL = re.compile(r"https?://\S+|www\.\S+")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def clean(text, t2s):
    text = URL.sub("", text.replace("\r", ""))
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t　]+", " ", text)
    return t2s.convert(text).strip()


def wildchat(root):
    import pyarrow.parquet as pq
    seen = set()
    for path in sorted(glob.glob(f"{root}/wildchat/data/*.parquet")):
        table = pq.read_table(path, columns=["conversation_hash", "language", "conversation"])
        for group, language, dialog in zip(*(table.column(c).to_pylist() for c in table.column_names)):
            if language != "Chinese":
                continue
            for turn in dialog:
                text = turn["content"]
                # Typed prompts, not pasted documents or code.
                if turn["role"] != "user" or len(text) > 300 or "```" in text or text.count("\n") > 5:
                    continue
                if (key := digest(text)) not in seen:
                    seen.add(key)
                    yield "wildchat:" + group, "prompt", text


def zhwiki(root):
    import mwparserfromhell
    variant = re.compile(r"-\{(?:[^{}]*?:)?([^{};]*)[^{}]*\}-")
    with bz2.open(f"{root}/zhwiki/zhwiki-20261001-pages-articles.xml.bz2") as stream:
        for _, element in ET.iterparse(stream):
            if not element.tag.endswith("}page"):
                continue
            ns = element.find("{*}ns").text
            redirect = element.find("{*}redirect") is not None
            title = element.find("{*}title").text
            text = element.find("{*}revision/{*}text").text or ""
            element.clear()
            if ns != "0" or redirect or text.lstrip().lower().startswith("#redirect"):
                continue
            plain = mwparserfromhell.parse(variant.sub(r"\1", text)).strip_code()
            plain = "\n".join(line for line in plain.splitlines()
                              if len(HAN.findall(line)) and not line.lstrip().startswith(("|", "!", "{", "参考", "外部链接")))
            yield "zhwiki:" + title, "wiki", plain


def fineweb(root):
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(f"{root}/fineweb-edu/4_5/*.parquet")):
        for i, text in enumerate(pq.read_table(path, columns=["text"]).column("text").to_pylist()):
            yield f"fineweb:{Path(path).stem}:{i}", "web", text


def zhihu(root):
    for path in sorted(glob.glob(f"{root}/mnbvc/qa/20230196/zhihu/*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                group = f"zhihu:{Path(path).name}:{row['id']}"
                yield group, "qa", row["问"]
                yield group, "qa", row["答"]


def forum(root):
    def threads():
        for path in sorted(glob.glob(f"{root}/mnbvc/forum/20230253/*/*.jsonl.gz")):
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    yield Path(path).name, json.loads(line)
        for path in sorted(glob.glob(f"{root}/mnbvc/forum/20260107/*/*.zip")):
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    for line in archive.open(name):
                        yield Path(path).name, json.loads(line)
    for source, thread in threads():
        group = f"forum:{source}:{thread.get('ID', digest(thread['主题']))}"
        # Every post was typed into its own box; replies never share context.
        yield group, "forum", thread["主题"]
        for reply in thread.get("回复", []):
            yield group, "forum", reply["回复"]


SOURCES = dict(wildchat=wildchat, zhwiki=zhwiki, fineweb=fineweb, zhihu=zhihu, forum=forum)


def split_of(group):
    bucket = int(digest(group)[:8], 16) % 100
    return "test" if bucket == 0 else "dev" if bucket == 1 else "train"


def spans(text, keys, lexicon, rng, density):
    """Sample word-aligned, non-overlapping spans whose every character has keys."""
    for match in HAN.finditer(text):
        run, base = match.group(), match.start()
        bounds = [i for i, _, _ in lexicon.segment(run)] + [len(run)]
        i = 0
        while i < len(bounds) - 1:
            if rng.random() > density:
                i += 1
                continue
            choice, total = rng.random(), 0
            for share, low, high in LENGTHS:
                total += share
                if choice <= total:
                    break
            want = rng.randint(low, high)
            start = bounds[i]
            j = next((k for k in range(i + 1, len(bounds)) if bounds[k] - start >= want), len(bounds) - 1)
            end = min(bounds[j], start + 24)
            if end - start >= 1 and all(keys[base + k] for k in range(start, end)):
                yield base + start, base + end
            i = j + rng.randint(0, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--g2pw", type=Path, required=True, help="directory with model/G2PWModel and bert-base-chinese")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--max-chars", type=int, default=0, help="per-source Han character budget; 0 is unlimited")
    parser.add_argument("--density", type=float, default=0.12, help="chance that a word boundary starts a span")
    parser.add_argument("--blank-share", type=float, default=0.15)
    parser.add_argument("--exclude-cases", type=Path, action="append", default=[])
    parser.add_argument("--seed", default="wime-corpus-v2")
    args = parser.parse_args()

    import onnxruntime
    import opencc
    from g2pw import G2PWConverter
    onnxruntime.preload_dlls()
    converter = G2PWConverter(model_dir=str(args.g2pw / "model/G2PWModel"), style="pinyin", num_workers=0,
                              model_source=str(args.g2pw / "bert-base-chinese"), enable_non_tradional_chinese=True)
    converter.session_g2pw = onnxruntime.InferenceSession(str(args.g2pw / "model/G2PWModel/g2pw.onnx"),
                                                         providers=["CUDAExecutionProvider"])
    lexicon = Lexicon(args.database)
    reader = Reader(lexicon, converter)
    t2s = opencc.OpenCC("t2s")
    protection = Protection(args.exclude_cases)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {s: (args.out / f"{s}.jsonl").open("w") for s in ("train", "dev", "test")}
    stats, started = Counter(), time.time()

    for name in args.sources.split(","):
        rng = random.Random(args.seed + name)
        budget, batch = args.max_chars, []

        def flush():
            sentences = [s for _, _, _, s, _ in batch]
            for (group, category, document, sentence, offset), keys in zip(batch, reader.keys(sentences)):
                for start, end in spans(sentence, keys, lexicon, rng, args.density):
                    context = (document[:offset] + sentence[:start])[-64:]
                    if rng.random() < args.blank_share:
                        context = ""
                    row = dict(id=digest(f"{group}:{offset + start}:{sentence[start:end]}")[:24], document=digest(group),
                               source=name, category=category, split=split_of(group), context=context,
                               input="".join(keys[start:end]), gold=[sentence[start:end]], reading="corpus-readings-v1")
                    if protection.contains(row):
                        stats[name + "/protected"] += 1
                        continue
                    outputs[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats[f"{name}/{row['split']}"] += 1
                    stats[f"length/{min(end - start, 13) if end - start < 13 else '13+'}"] += 1
            batch.clear()

        for group, category, text in SOURCES[name](args.raw):
            text = clean(text, t2s)
            han = sum(map(len, HAN.findall(text)))
            if han < 2:
                continue
            stats[name + "/documents"] += 1
            stats[name + "/han_chars"] += han
            offset = 0
            for sentence in SENTENCE.findall(text):
                # g2pW and the editor both see bounded windows.
                for piece in (sentence[i:i + 128] for i in range(0, len(sentence), 128)):
                    if HAN.search(piece):
                        batch.append((group, category, text, piece, offset))
                    offset += len(piece)
            if len(batch) >= 512:
                flush()
            if budget and stats[name + "/han_chars"] >= budget:
                break
            if stats[name + "/documents"] % 20000 == 0:
                print(json.dumps(dict(source=name, elapsed_s=round(time.time() - started), **{
                    k: v for k, v in stats.items() if k.startswith(name)})), flush=True)
        flush()
    for stream in outputs.values():
        stream.close()
    manifest = dict(stats=dict(stats), readings=dict(reader.methods), arguments={k: str(v) for k, v in vars(args).items()},
                    protection=protection.files, elapsed_s=time.time() - started)
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest["stats"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
