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
from corpus_integrity import UniqueRows, checked_row, empty_output, file_metadata, reading_assets, verify_sources

# Short commits dominate actual editor input; retain long spans for coverage.
LENGTHS = ((0.50, 1, 1), (0.25, 2, 3), (0.15, 4, 8), (0.10, 9, 24))
SENTENCE = re.compile(r"[^。！？!?\n]*[。！？!?\n]?")
URL = re.compile(r"https?://\S+|www\.\S+")
PARQUET_BATCH_SIZE = 256


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def clean(text, t2s):
    text = URL.sub("", text.replace("\r", ""))
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t　]+", " ", text)
    return t2s.convert(text).strip()


def parquet_rows(path, columns):
    """Yield stable file row numbers without materializing an entire shard."""
    import pyarrow.parquet as pq
    offset = 0
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=columns):
            for index, row in enumerate(batch.to_pylist(), offset):
                yield index, row
            offset += batch.num_rows


def wildchat(root):
    seen = set()
    for path in sorted(glob.glob(f"{root}/wildchat/data/*.parquet")):
        for _, row in parquet_rows(path, ["conversation_hash", "language", "conversation"]):
            if row["language"] != "Chinese":
                continue
            group = "wildchat:" + row["conversation_hash"]
            for turn_index, turn in enumerate(row["conversation"]):
                text = turn["content"]
                # Typed prompts, not pasted documents or code.
                if turn["role"] != "user" or len(text) > 300 or "```" in text or text.count("\n") > 5:
                    continue
                if (key := digest(text)) not in seen:
                    seen.add(key)
                    yield group, f"{group}:turn:{turn_index}", "prompt", text


def zhwiki(root):
    import mwparserfromhell
    variant = re.compile(r"-\{(?:[^{}]*?:)?([^{};]*)[^{}]*\}-")
    with bz2.open(f"{root}/zhwiki/zhwiki-20261001-pages-articles.xml.bz2") as stream:
        events = ET.iterparse(stream, events=("start", "end"))
        _, root_element = next(events)
        for event, element in events:
            if event != "end" or element.tag.rsplit("}", 1)[-1] != "page":
                continue
            ns = element.find("{*}ns").text
            redirect = element.find("{*}redirect") is not None
            title = element.find("{*}title").text
            text = element.find("{*}revision/{*}text").text or ""
            element.clear()
            # Clearing the page alone leaves one empty element per article
            # attached to the root for the lifetime of the entire dump.
            root_element.clear()
            if ns != "0" or redirect or text.lstrip().lower().startswith("#redirect"):
                continue
            plain = mwparserfromhell.parse(variant.sub(r"\1", text)).strip_code()
            plain = "\n".join(line for line in plain.splitlines()
                              if len(HAN.findall(line)) and not line.lstrip().startswith(("|", "!", "{", "参考", "外部链接")))
            group = "zhwiki:" + title
            yield group, group, "wiki", plain


def fineweb(root):
    for path in sorted(glob.glob(f"{root}/fineweb-edu/4_5/*.parquet")):
        for i, row in parquet_rows(path, ["text"]):
            # The released shards have no URL/document ID. Exact copies must
            # still stay in the same split even when stored in different files.
            yield "fineweb:" + digest(row["text"]), f"fineweb:{Path(path).stem}:{i}", "web", row["text"]


def source_id(value):
    """MNBVC metadata sometimes serializes integer source IDs as floats."""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def zhihu(root):
    seen_questions = set()
    for path in sorted(glob.glob(f"{root}/mnbvc/qa/20230196/zhihu/*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                metadata = row.get("元数据", {}).get("扩展字段", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                question = metadata.get("question_id")
                question = source_id(question) if question is not None else digest(row["问"])
                group = "zhihu:question:" + question
                if group not in seen_questions:
                    seen_questions.add(group)
                    yield group, group, "qa", row["问"]
                answer = metadata.get("answer_id")
                answer = source_id(answer) if answer is not None else f"{Path(path).name}:{row['id']}"
                yield group, "zhihu:answer:" + answer, "qa", row["答"]


def forum(root):
    def threads():
        for path in sorted(glob.glob(f"{root}/mnbvc/forum/20230253/*/*.jsonl.gz")):
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    yield Path(path).parent.name, json.loads(line)
        for path in sorted(glob.glob(f"{root}/mnbvc/forum/20260107/*/*.zip")):
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith("/"):
                        continue
                    with archive.open(name) as stream:
                        for line in stream:
                            yield Path(path).parent.name, json.loads(line)
    for source, thread in threads():
        thread_id = thread.get("ID")
        thread_id = source_id(thread_id) if thread_id is not None else digest(thread["主题"])
        group = f"forum:{source}:{thread_id}"
        # Every post was typed into its own box; replies never share context.
        yield group, group + ":topic", "forum", thread["主题"]
        for index, reply in enumerate(thread.get("回复", [])):
            yield group, f"{group}:reply:{index}", "forum", reply["回复"]


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
            # A hard character slice could bisect the final dictionary word.
            while bounds[j] - start > 24 and j > i:
                j -= 1
            end = bounds[j]
            if end - start >= 1 and all(keys[base + k] for k in range(start, end)):
                yield base + start, base + end
            i = max(i + 1, j + rng.randint(0, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--g2pw", type=Path, required=True, help="directory with model/G2PWModel and bert-base-chinese")
    parser.add_argument("--source-manifest", type=Path, default=Path(__file__).with_name("corpus-v2-sources.json"))
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--max-chars", type=int, default=0, help="per-source Han character budget; 0 is unlimited")
    parser.add_argument("--density", type=float, default=0.12, help="chance that a word boundary starts a span")
    parser.add_argument("--blank-share", type=float, default=0.30,
                        help="probability of dropping left context, in addition to naturally empty context")
    parser.add_argument("--exclude-cases", type=Path, action="append", default=[])
    parser.add_argument("--seed", default="wime-corpus-v2")
    args = parser.parse_args()
    names = args.sources.split(",")
    if len(set(names)) != len(names) or any(n not in SOURCES for n in names):
        parser.error("sources must be unique names from " + ",".join(SOURCES))
    if args.max_chars < 0 or not 0 < args.density <= 1 or not 0 <= args.blank_share <= 1:
        parser.error("invalid character budget, density or blank share")
    empty_output(args.out)
    started = time.time()
    provenance = verify_sources(args.raw, args.source_manifest, names)

    provenance["database"] = file_metadata(args.database)
    provenance["reading_assets"] = reading_assets(args.g2pw)
    provenance["code"] = {name: file_metadata(Path(__file__).with_name(name)) for name in
                          ("prepare-corpus.py", "corpus_readings.py", "corpus_integrity.py", "corpus_g2pw.py", "replay.py", "context-pinyin.json", "requirements-corpus.txt")}

    import importlib.metadata
    import onnxruntime
    import opencc
    from corpus_g2pw import CorpusG2PWConverter
    onnxruntime.preload_dlls()
    converter = CorpusG2PWConverter(model_dir=str(args.g2pw / "model/G2PWModel"), style="pinyin", num_workers=0,
                              model_source=str(args.g2pw / "bert-base-chinese"),
                              enable_non_tradional_chinese=True, use_cuda=True)
    # g2pw 0.1.2 uses `num_workers if num_workers else config.num_workers`.
    # Set this explicitly: forking two PyTorch workers on every batch costs seconds.
    converter.num_workers = 0
    providers = converter.session_g2pw.get_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDA provider unavailable: {providers}")
    lexicon = Lexicon(args.database)
    reader = Reader(lexicon, converter)
    t2s = opencc.OpenCC("t2s")
    protection = Protection(args.exclude_cases)
    unique = UniqueRows(args.out / "identities.sqlite3")
    splits = ("train", "dev", "test")
    outputs = {s: (args.out / f"{s}.jsonl.partial").open("w", encoding="utf-8") for s in splits}
    stats = Counter()
    last_report = time.time()

    try:
        for name in names:
            rng = random.Random(args.seed + name)
            batch = []

            def flush():
                sentences = [s for _, _, _, _, _, s, _ in batch]
                for (group, doc_id, content_id, category, document, sentence, offset), keys in zip(batch, reader.keys(sentences)):
                    for start, end in spans(sentence, keys, lexicon, rng, args.density):
                        context = document[max(0, offset + start - 64):offset + start]
                        gold = sentence[start:end]
                        row = dict(id=digest(f"{doc_id}:{offset + start}:{gold}")[:24], document=doc_id,
                                   group=digest(group), content=content_id, source=name, category=category, split=split_of(group),
                                   context=context, input="".join(keys[start:end]), gold=[gold],
                                   reading="corpus-readings-v2")
                        row, rejection = checked_row(row, protection, unique, rng, args.blank_share)
                        if rejection:
                            stats[f"{name}/{rejection}"] += 1
                            continue
                        outputs[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats[f"{name}/{row['split']}"] += 1
                        stats[f"length/{min(end - start, 13) if end - start < 13 else '13+'}"] += 1
                        stats["blank_context"] += not bool(row["context"])
                batch.clear()
                unique.commit()

            for group, document_id, category, text in SOURCES[name](args.raw):
                text = clean(text, t2s)
                han = sum(map(len, HAN.findall(text)))
                if han < 2:
                    continue
                content_id = digest(text)
                if not unique.add("document", content_id):
                    stats[name + "/duplicate_document"] += 1
                    continue
                # A held-out passage anywhere in a document excludes that document,
                # including rows whose context is later blanked for augmentation.
                doc_id = digest(document_id)
                if protection.contains(dict(document=doc_id, context="", gold=[text])):
                    stats[name + "/protected_document"] += 1
                    continue
                stats[name + "/documents"] += 1
                stats[name + "/han_chars"] += han
                offset = 0
                for sentence in SENTENCE.findall(text):
                    for piece in (sentence[i:i + 128] for i in range(0, len(sentence), 128)):
                        if HAN.search(piece):
                            batch.append((group, doc_id, content_id, category, text, piece, offset))
                        offset += len(piece)
                        if len(batch) >= 512:
                            flush()
                if args.max_chars and stats[name + "/han_chars"] >= args.max_chars:
                    break
                if time.time() - last_report >= 30:
                    print(json.dumps(dict(event="progress", source=name, elapsed_s=round(time.time() - started),
                                          stats={k: v for k, v in stats.items() if k.startswith(name)})), flush=True)
                    last_report = time.time()
            flush()
            print(json.dumps(dict(event="source_complete", source=name,
                                  stats={k: v for k, v in stats.items() if k.startswith(name)})), flush=True)
    finally:
        for stream in outputs.values():
            stream.close()
        unique.close()

    for split in splits:
        (args.out / f"{split}.jsonl.partial").rename(args.out / f"{split}.jsonl")
    manifest = dict(format="wime-corpus-v2", stats=dict(stats), readings=dict(reader.methods),
                    sampling=dict(lengths=LENGTHS, max_context=64, max_span=24),
                    arguments={k: [str(p) for p in v] if isinstance(v, list) else str(v)
                               for k, v in vars(args).items()},
                    protection=protection.files, provenance=provenance, providers=providers,
                    packages={name: importlib.metadata.version(name) for name in
                              ("onnxruntime-gpu", "g2pw", "pypinyin", "opencc", "pyarrow", "mwparserfromhell", "transformers", "torch")},
                    outputs={s: file_metadata(args.out / f"{s}.jsonl") for s in splits},
                    elapsed_s=time.time() - started)
    (args.out / "manifest.json.partial").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.out / "manifest.json.partial").rename(args.out / "manifest.json")
    print(json.dumps(manifest["stats"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
