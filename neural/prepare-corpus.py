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
import sqlite3
import sys
import time
from typing import NamedTuple
import xml.etree.ElementTree as ET
import zipfile

from corpus_readings import HAN, Lexicon, Reader
from replay import Protection
from corpus_integrity import UniqueRows, checked_row, empty_output, file_metadata, reading_assets, verify_sources
from corpus_tones import TONE_POLICY, validate_tones
from corpus_cleaning import clean_document, decode_entities

# Short commits dominate actual editor input; retain long spans for coverage.
LENGTHS = ((0.50, 1, 1), (0.25, 2, 3), (0.15, 4, 8), (0.10, 9, 24))
SENTENCE = re.compile(r"[^。！？!?\n]*[。！？!?\n]?")
PARQUET_BATCH_SIZE = 256
ANNOTATION_CORE = 128
ANNOTATION_HALO = 64
SOURCE_TRANSFORMS = Counter()


class AnnotationWindow(NamedTuple):
    text: str
    offset: int
    core_start: int
    core_end: int


def annotation_windows(document):
    """Give each character one core while retaining neighbouring context."""
    for match in SENTENCE.finditer(document):
        sentence = match.group()
        for core in range(0, len(sentence), ANNOTATION_CORE):
            core_end = min(len(sentence), core + ANNOTATION_CORE)
            left = max(0, core - ANNOTATION_HALO)
            right = min(len(sentence), core_end + ANNOTATION_HALO)
            yield AnnotationWindow(sentence[left:right], match.start() + left, core - left, core_end - left)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


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


def render_wiki(text):
    """Render article prose while retaining explicitly marked code literally."""
    import mwparserfromhell

    # Protect literal regions before interpreting Wiki links, variants or HTML.
    # In real articles malformed italics can hide later HTML from the Wiki AST,
    # so these source-level boundaries must precede mwparserfromhell.parse.
    attributes = r'''(?:[^>"']|"[^"]*"|'[^']*')*'''
    literal = re.compile(
        rf"<(?P<tag>nowiki|code|pre|syntaxhighlight|source)(?=[\s/>]){attributes}>"
        r"(?P<body>.*?)</(?P=tag)\s*>", re.IGNORECASE | re.DOTALL)
    placeholders = {}
    prefix = "WIMECORPUSLITERAL"
    while prefix in text:
        prefix += "X"

    def protect(match):
        body = match.group("body")
        # MediaWiki commonly marks inline examples as <code><nowiki>...</nowiki>
        # </code>. Remove that structural inner wrapper, not its literal HTML.
        nested = re.fullmatch(r"<nowiki\s*>(.*?)</nowiki\s*>", body, re.IGNORECASE | re.DOTALL)
        if nested:
            body = nested.group(1)
        # Decode explicit entities once, without interpreting the resulting tags
        # or Wiki syntax. Restoring after strip_code avoids a second decoding.
        body = decode_entities(body)
        key = f"{prefix}{len(placeholders)}END"
        placeholders[key] = body
        SOURCE_TRANSFORMS["zhwiki/literal_regions_preserved"] += 1
        return key

    text = literal.sub(protect, text)
    # References are ancillary text, not inline article prose. Handling their
    # actual source tags also works when a citation contains broken Wiki quotes.
    references = re.compile(
        rf"<(ref|references)(?=[\s/>]){attributes}/\s*>|"
        rf"<(?P<paired>ref|references)(?=[\s/>]){attributes}>.*?</(?P=paired)\s*>",
        re.IGNORECASE | re.DOTALL)
    text, count = references.subn("", text)
    SOURCE_TRANSFORMS["zhwiki/references_removed"] += count
    block = {"div", "p", "br", "hr", "table", "thead", "tbody", "tfoot", "tr", "td", "th",
             "caption", "ul", "ol", "li", "dl", "dt", "dd", "blockquote", "center",
             "h1", "h2", "h3", "h4", "h5", "h6"}
    inline = {"span", "b", "i", "em", "strong", "u", "s", "small", "big", "sup", "sub", "font"}
    layout = re.compile(rf"</?(?P<tag>{'|'.join(sorted(block | inline))})(?=[\s/>]){attributes}>", re.IGNORECASE)

    def boundary(match):
        SOURCE_TRANSFORMS["zhwiki/layout_tags_removed"] += 1
        return "\n" if match.group("tag").lower() in block else ""

    text = layout.sub(boundary, text)
    # Bold/italic delimiters carry no prose. Removing them before parsing keeps
    # an unmatched five-quote run from swallowing later link boundaries.
    text, count = re.subn(r"'{2,}", "", text)
    SOURCE_TRANSFORMS["zhwiki/style_markers_removed"] += count
    # A source typo puts link brackets inside a language-variant delimiter.
    # Recover only a list of explicit Chinese variant keys, never arbitrary
    # bracketed prose (or literal code, which is still protected above).
    entry = r"zh(?:-(?:hans|hant|cn|tw|hk|sg|mo|my))?\s*:[^{};\n]+"
    misplaced_link = re.compile(rf"-\[\[\{{(?P<variants>{entry}(?:;\s*{entry})*;?)\}}\]\]-", re.IGNORECASE)
    text, count = misplaced_link.subn(lambda match: "-{" + match.group("variants") + "}-", text)
    SOURCE_TRANSFORMS["zhwiki/malformed_link_variants_recovered"] += count
    variant = re.compile(r"-\{(?:[^{}]*?:)?([^{};]*)[^{}]*\}-")
    markup = mwparserfromhell.parse(variant.sub(r"\1", text))
    # strip_code alone renders image options as prose (right|thumb|...).
    for link in markup.filter_wikilinks(recursive=True):
        link_title = str(link.title).strip().lstrip(":")
        if re.match(r"(?i)^(?:file|image|文件|檔案|图像|圖像)\s*:", link_title):
            try:
                markup.remove(link, recursive=True)
                SOURCE_TRANSFORMS["zhwiki/media_links_removed"] += 1
            except ValueError:
                pass  # A parent media link may already have been removed.
    plain = markup.strip_code()
    plain = "\n".join(line for line in plain.splitlines()
                      # Layout boundaries can put years or foreign names on
                      # their own line. Keep them as neighbouring context.
                      if line.strip()
                      and not line.lstrip().startswith(("|", "!", "{", "参考", "外部链接")))
    for key, body in placeholders.items():
        plain = plain.replace(key, body)
    return plain


def zhwiki(root):
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
            plain = render_wiki(text)
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
        # Archives can repeat one thread ID with edited/reformatted contents.
        # Keep those versions in the same split, but distinguish documents so
        # one identity never refers to two different full texts.
        yield group, f"{group}:topic:{digest(thread['主题'])}", "forum", thread["主题"]
        for index, reply in enumerate(thread.get("回复", [])):
            yield group, f"{group}:reply:{index}:{digest(reply['回复'])}", "forum", reply["回复"]


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


def annotated_spans(document, window, reading, lexicon, rng, density):
    """Sample only the core, slicing every annotation from the same window."""
    core_start, core_end = window.core_start, window.core_end
    keys = reading["keys"]
    for start, end in spans(window.text[core_start:core_end], keys[core_start:core_end], lexicon, rng, density):
        start, end = start + core_start, end + core_start
        offset = window.offset + start
        yield dict(offset=offset, reading_window=[window.offset, window.offset + len(window.text)],
                   context=document[max(0, offset - 64):offset], gold=[window.text[start:end]],
                   input="".join(keys[start:end]), pinyin=reading["pinyin"][start:end],
                   tones=reading["tones"][start:end], tone_sources=reading["tone_sources"][start:end],
                   reading_rules=reading["reading_rules"][start:end],
                   reading_evidence={source: values[start:end]
                                     for source, values in reading["reading_evidence"].items()})


def clean_corpus(args, names):
    """Materialize reviewed text before any model is loaded or labels created."""
    import importlib.metadata
    import opencc
    empty_output(args.out)
    started = time.time()
    provenance = verify_sources(args.raw, args.source_manifest, names)
    provenance["database"] = file_metadata(args.database)
    provenance["code"] = {name: file_metadata(Path(__file__).with_name(name)) for name in
                          ("prepare-corpus.py", "corpus_cleaning.py", "corpus_integrity.py", "replay.py")}
    t2s = opencc.OpenCC("t2s")
    protection = Protection(args.exclude_cases)
    unique = UniqueRows(args.out / "identities.sqlite3")
    index = sqlite3.connect(args.out / "documents.sqlite3")
    index.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, group_id TEXT, source TEXT, category TEXT, "
                  "split TEXT, content TEXT, raw_content TEXT, text TEXT) WITHOUT ROWID")
    stats, examples = Counter(), {}
    SOURCE_TRANSFORMS.clear()
    last_report = time.time()

    def record(source, reason, doc_id, text):
        stats[f"{source}/{reason}"] += 1
        bucket = examples.setdefault(f"{source}/{reason}", [])
        if len(bucket) < 5:
            # Keep invalid surrogates out of the UTF-8 review artifact.
            snippet = text[:320].encode("utf-8", errors="backslashreplace").decode("utf-8")
            bucket.append(dict(document=doc_id, snippet=snippet))

    try:
        for source in names:
            with (args.out / f"{source}.jsonl.partial").open("w", encoding="utf-8") as output:
                for group, identity, category, raw in SOURCES[source](args.raw):
                    stats[f"{source}/raw_documents"] += 1
                    doc_id = digest(identity)
                    text, operations, rejection = clean_document(source, raw, t2s)
                    for operation in operations:
                        record(source, "changed/" + operation, doc_id, raw)
                    if rejection:
                        record(source, "rejected/" + rejection, doc_id, raw)
                        continue
                    han = sum(map(len, HAN.findall(text)))
                    if han < 2:
                        record(source, "rejected/insufficient_han", doc_id, text)
                        continue
                    if any(protection.contains(dict(document=doc_id, context="", gold=[candidate]))
                           for candidate in (raw, text)):
                        record(source, "rejected/protected_document", doc_id, text)
                        continue
                    content = digest(text)
                    if not unique.add("content", content):
                        record(source, "rejected/duplicate_document", doc_id, text)
                        continue
                    if not unique.add("document", doc_id):
                        raise ValueError(f"source document identity has different cleaned content: {doc_id}")
                    row = dict(document=doc_id, group=digest(group), source=source, category=category,
                               split=split_of(group), content=content, raw_content=digest(raw), text=text,
                               source_document=identity, cleaning=operations)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    index.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                  tuple(row[key] for key in ("document", "group", "source", "category",
                                                            "split", "content", "raw_content", "text")))
                    stats[f"{source}/documents"] += 1
                    stats[f"{source}/han_chars"] += han
                    if stats[f"{source}/documents"] % 10000 == 0:
                        index.commit()
                        unique.commit()
                    if args.max_chars and stats[f"{source}/han_chars"] >= args.max_chars:
                        break
                    if time.time() - last_report >= 30:
                        print(json.dumps(dict(event="cleaning_progress", source=source,
                                              elapsed_s=round(time.time() - started),
                                              stats={k: v for k, v in stats.items() if k.startswith(source + "/")})), flush=True)
                        last_report = time.time()
            index.commit()
            unique.commit()
            (args.out / f"{source}.jsonl.partial").rename(args.out / f"{source}.jsonl")
            print(json.dumps(dict(event="source_cleaned", source=source,
                                  stats={k: v for k, v in stats.items() if k.startswith(source + "/")})), flush=True)
    finally:
        index.close()
        unique.close()
    review = args.out / "cleaning-review.json"
    review.write_text(json.dumps(dict(stats=dict(stats), source_transforms=dict(SOURCE_TRANSFORMS),
                                     examples=examples), ensure_ascii=False, indent=2) + "\n")
    manifest = dict(format="wime-cleaned-corpus-v1", complete=True, stats=dict(stats), source_transforms=dict(SOURCE_TRANSFORMS),
                    policy="source-aware-v1; quarantine corrupted documents; preserve mixed-language and code context",
                    provenance=provenance, protection=protection.files,
                    arguments={k: [str(p) for p in v] if isinstance(v, list) else str(v)
                               for k, v in vars(args).items()},
                    packages={name: importlib.metadata.version(name) for name in
                              ("opencc", "pyarrow", "mwparserfromhell")},
                    outputs={name: file_metadata(args.out / f"{name}.jsonl") for name in names},
                    documents=file_metadata(args.out / "documents.sqlite3"), review=file_metadata(review),
                    elapsed_s=time.time() - started)
    (args.out / "manifest.json.partial").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.out / "manifest.json.partial").rename(args.out / "manifest.json")
    print(json.dumps(dict(event="cleaning_complete", stats=dict(stats), output=str(args.out))), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--raw", type=Path, help="raw sources, accepted only with --clean-only")
    inputs.add_argument("--cleaned", type=Path, help="completed, validated cleaning artifact")
    parser.add_argument("--clean-only", action="store_true", help="write cleaned documents without loading a model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--g2pw", type=Path, help="directory with model/G2PWModel and the pinned Mandarin tokenizer")
    parser.add_argument("--tokenizer", type=Path, help="default: G2PW root/chinese-roberta-wwm-ext-large")
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
    if args.clean_only:
        if args.raw is None:
            parser.error("--clean-only requires --raw")
        clean_corpus(args, names)
        return 0
    if args.cleaned is None or args.g2pw is None:
        parser.error("annotation requires --cleaned and --g2pw; first run --raw ... --clean-only")
    from corpus_cleaned import iter_cleaned, load_cleaned
    cleaned = load_cleaned(args.cleaned, names)
    empty_output(args.out)
    started = time.time()
    provenance = dict(cleaned["provenance"])
    provenance["cleaned_corpus"] = file_metadata(args.cleaned / "manifest.json")
    database = file_metadata(args.database)
    if database["sha256"] != provenance["database"]["sha256"]:
        raise ValueError("annotation dictionary differs from the cleaning artifact")
    provenance["database"] = database
    for path, expected in cleaned["protection"].items():
        if file_metadata(Path(path))["sha256"] != expected:
            raise ValueError(f"changed held-out protection input: {path}")
    tokenizer = args.tokenizer or args.g2pw / "chinese-roberta-wwm-ext-large"
    provenance["reading_assets"] = reading_assets(args.g2pw, tokenizer)
    provenance["code"] = {name: file_metadata(Path(__file__).with_name(name)) for name in
                          ("prepare-corpus.py", "corpus_readings.py", "corpus_integrity.py", "corpus_cleaning.py", "corpus_cleaned.py", "mandarin_frontend.py", "corpus_tones.py", "replay.py", "context-pinyin.json", "requirements-corpus.txt")}

    import importlib.metadata
    import onnxruntime
    from mandarin_frontend import MandarinFrontend
    onnxruntime.preload_dlls()
    converter = MandarinFrontend(model_dir=args.g2pw / "model/G2PWModel", tokenizer_dir=tokenizer)
    providers = converter.providers
    upstream = Path(__file__).resolve().parents[3] / "third_party/gpt-sovits/GPT_SoVITS/text"
    upstream_files = [upstream / "tone_sandhi.py", upstream / "zh_normalization/char_convert.py"]
    upstream_files += sorted((upstream / "g2pw").glob("*.py")) + sorted((upstream / "g2pw").glob("*.rep"))
    provenance["code"].update({"gpt-sovits/" + str(p.relative_to(upstream)): file_metadata(p) for p in upstream_files})
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDA provider unavailable: {providers}")
    lexicon = Lexicon(args.database)
    reader = Reader(lexicon, converter)
    protection = Protection(sorted(set(map(Path, cleaned["protection"])) | set(args.exclude_cases)))
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
                sentences = [window.text for _, window in batch]
                for (doc, window), reading in zip(batch, reader.readings(sentences)):
                    document, doc_id = doc["text"], doc["document"]
                    for sample in annotated_spans(document, window, reading, lexicon, rng, args.density):
                        gold = sample["gold"][0]
                        row = dict(id=digest(f"{doc_id}:{sample['offset']}:{gold}")[:24], document=doc_id,
                                   group=doc["group"], content=doc["content"], source=name, category=doc["category"], split=doc["split"],
                                   reading="gpt-sovits-mandarin-v1", **sample)
                        validate_tones(row)
                        row, rejection = checked_row(row, protection, unique, rng, args.blank_share)
                        if rejection:
                            stats[f"{name}/{rejection}"] += 1
                            continue
                        outputs[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats[f"{name}/{row['split']}"] += 1
                        stats[f"length/{len(gold) if len(gold) < 13 else '13+'}"] += 1
                        stats["blank_context"] += not bool(row["context"])
                        stats["tone/total"] += len(row["tones"])
                        stats["tone/neutral"] += row["tones"].count(5)
                batch.clear()
                unique.commit()

            for doc in iter_cleaned(args.cleaned, name):
                text = doc["text"]
                han = sum(map(len, HAN.findall(text)))
                if (doc["source"] != name or digest(text) != doc["content"]
                        or not unique.add("document", doc["document"])):
                    raise ValueError("cleaned source identity, content hash or document uniqueness is invalid")
                if protection.contains(dict(document=doc["document"], context="", gold=[text])):
                    stats[name + "/protected_document"] += 1
                    continue
                stats[name + "/documents"] += 1
                stats[name + "/han_chars"] += han
                for window in annotation_windows(text):
                    if HAN.search(window.text[window.core_start:window.core_end]):
                        batch.append((doc, window))
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
                    annotation_window=dict(core=ANNOTATION_CORE, left_halo=ANNOTATION_HALO, right_halo=ANNOTATION_HALO,
                                           max_characters=ANNOTATION_CORE + 2 * ANNOTATION_HALO,
                                           boundary="sentence", offsets="zero-based characters in cleaned document; end exclusive"),
                    tone_policy=TONE_POLICY,
                    arguments={k: [str(p) for p in v] if isinstance(v, list) else str(v)
                               for k, v in vars(args).items()},
                    protection=protection.files, provenance=provenance, providers=providers,
                    packages={name: importlib.metadata.version(name) for name in
                              ("onnxruntime-gpu", "pypinyin", "opencc", "pyarrow", "mwparserfromhell", "transformers", "jieba-fast", "requests", "setuptools")},
                    outputs={s: file_metadata(args.out / f"{s}.jsonl") for s in splits},
                    elapsed_s=time.time() - started)
    (args.out / "manifest.json.partial").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.out / "manifest.json.partial").rename(args.out / "manifest.json")
    print(json.dumps(manifest["stats"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
