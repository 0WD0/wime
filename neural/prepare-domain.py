#!/usr/bin/env python3
"""Prepare document-disjoint IME/development prose and its extra-corpus mix.

Local sources are tracked monorepo docs and, optionally, Claude Code assistant
replies (the user's own messages stay a test set). External sources are the
pinned Chinese doc subtrees in domain-sources.json, fetched as sparse partial
clones; nothing from them is executed. Rows sharing a 6+ character run with a
protected answer are dropped. Training rows with words the general corpus
barely covers are repeated, and the result is mixed with an existing extra
corpus at --share of its rows.
"""
import argparse
from collections import Counter
import glob
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

from pypinyin import lazy_pinyin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
from wime_gen import Converter, load_rules

ROOT = Path(__file__).resolve().parents[3]
HAN = re.compile(r"[一-鿿]+")


def clean(text):
    text = re.sub(r"(?ims)^[ \t]*#\+begin_(src|example).*?^[ \t]*#\+end_\1.*?$", " ", text)
    text = re.sub(r"(?s)```.*?```|<!--.*?-->", " ", text)
    text = re.sub(r"(?m)^[ \t]*#\+\w+:.*$|^[ \t]*:\w+:.*$", " ", text)
    text = re.sub(r"\[\[[^\]]*\]\[([^\]]*)\]\]|\[\[([^\]]*)\]\]", lambda m: m[1] or m[2], text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`[^`\n]*`|(?<!\w)[=~][^=~\n]+[=~](?!\w)", " ", text)
    text = re.sub(r"(?m)^\s*\|.*$", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Soft line wraps in Chinese prose should not split its meaning.
    return re.sub(r"(?<=[一-鿿])\n(?=[一-鿿])", "", text)


def clean_external(text):
    """Site generators add front matter, template macros and MDX/HTML tags."""
    text = re.sub(r"\A---\n.*?\n---\n", " ", text, flags=re.S)
    text = re.sub(r"\{\{[^{}]*\}\}|\{%.*?%\}", " ", text, flags=re.S)
    text = re.sub(r"(?m)^\s*(import|export) .*$", " ", text)
    return clean(re.sub(r"</?[A-Za-z][^>]*>", " ", text))


def fetch(sources, directory):
    """Sparse partial clone of each pinned revision; never runs hooks."""
    for name, source in sources.items():
        target = directory / name
        if target.exists():
            head = subprocess.run(["git", "-C", target, "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            if head != source["revision"]:
                raise ValueError(f"{target} is at {head}, expected {source['revision']}")
            continue
        git = ["git", "-c", "core.hooksPath=/dev/null", "-C", str(target)]
        target.mkdir(parents=True)
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["remote", "add", "origin", f"https://github.com/{source['repository']}.git"], check=True)
        subprocess.run(git + ["sparse-checkout", "set", "--no-cone", *source["paths"], "/LICENSE*"], check=True)
        subprocess.run(git + ["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", source["revision"]],
                       check=True)
        subprocess.run(git + ["checkout", "-q", "FETCH_HEAD"], check=True)


def documents(args, sources):
    """Yield (source, name, cleaned text, raw text)."""
    files = subprocess.run(["jjosh", "file", "list"], capture_output=True, text=True, cwd=ROOT,
                           check=True).stdout.split()
    for path in files:
        # essay.txt is librime's word-frequency list, not prose.
        # Concurrent working-copy edits can remove a listed file.
        if (path.endswith((".org", ".md", ".txt", ".rst")) and not path.endswith("/essay.txt")
                and (ROOT / path).is_file()):
            raw = (ROOT / path).read_text(errors="ignore")
            yield ("third_party" if path.startswith("third_party/") else "monorepo"), path, clean(raw), raw
    for transcript in sorted(glob.glob(args.transcripts)) if args.transcripts else []:
        for line in open(transcript, errors="ignore"):
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            content = (entry.get("message") or {}).get("content")
            if entry.get("type") == "assistant" and isinstance(content, list):
                raw = "\n".join(block.get("text", "") for block in content
                                if isinstance(block, dict) and block.get("type") == "text")
                yield "assistant", f"{Path(transcript).name}:{entry.get('uuid')}", clean(raw), raw
    for name in sources:
        for path in sorted((args.external / name).rglob("*")):
            if path.suffix in (".md", ".mdx", ".rst") and ".git" not in path.parts:
                raw = path.read_text(errors="ignore")
                yield name, str(path.relative_to(args.external / name)), clean_external(raw), raw


def general_counts(paths, words):
    counts = Counter()
    for path in paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                for run in HAN.findall(row["context"] + "|" + row["gold"][0]):
                    counts.update(run[i:i + n] for n in (2, 3, 4) for i in range(len(run) - n + 1)
                                  if run[i:i + n] in words)
    return counts


def features(text, words):
    return {text[i:i + n] for n in (2, 3, 4) for i in range(len(text) - n + 1)} & words


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algebra", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True, help="wanxiang.db, for homophone words")
    p.add_argument("--sources", type=Path, default=Path(__file__).with_name("domain-sources.json"))
    p.add_argument("--external", type=Path, required=True, help="checkout directory for pinned sources")
    p.add_argument("--fetch", action="store_true", help="sparse-clone missing pinned sources")
    p.add_argument("--transcripts", help="glob of Claude Code transcripts; assistant replies only")
    p.add_argument("--protected", type=Path, action="append", default=[],
                   help="answers whose 6+ character runs must not appear in any split")
    p.add_argument("--general", type=Path, action="append", default=[], help="general training jsonl")
    p.add_argument("--extra", type=Path, required=True, help="extra corpus to mix with, e.g. diverse")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--combined", type=Path, required=True)
    p.add_argument("--share", type=float, default=0.5)
    p.add_argument("--rare-general", type=int, default=5)
    p.add_argument("--rare-domain", type=int, default=3)
    p.add_argument("--rare-repeat", type=int, default=3)
    args = p.parse_args()
    sources = json.loads(args.sources.read_text())
    if args.fetch:
        fetch(sources, args.external)
    converter = Converter(rules=load_rules(args.algebra))
    protected, grams = set(), set()
    for path in args.protected:
        raw = path.read_text()
        for row in json.loads(raw) if raw.lstrip().startswith("[") else map(json.loads, raw.splitlines()):
            for gold in row["gold"]:
                protected.add(gold)
                grams.update(gold[i:i + 6] for i in range(len(gold) - 5))
    counts, rows, seen = Counter(), [], {}
    for source, name, text, raw in documents(args, sources):
        if not re.search(r"[一-鿿]{6}", raw):
            continue
        document = hashlib.sha256((source + name + text).encode()).hexdigest()
        bucket = int(document[:8], 16) % 100
        split = "test" if bucket < 10 else "dev" if bucket < 20 else "train"
        counts[f"{source}/{split}_documents"] += 1
        for match in re.finditer(r"[一-鿿]{4,32}", text):
            target = match.group()
            window = text[max(0, match.start() - 64):match.end()]
            if target in protected or any(window[i:i + 6] in grams for i in range(len(window) - 5)):
                counts["protected_overlap"] += 1
                continue
            if target in seen:
                prior = rows[seen[target]]
                if prior["split"] != split:
                    prior["excluded"] = True
                counts["duplicates"] += 1
                continue
            pinyin = lazy_pinyin(target)
            codes = [sorted(converter.codes(p)) for p in pinyin]
            if any(not c for c in codes) or len("".join(c[0] for c in codes)) != 2 * len(target):
                continue
            seen[target] = len(rows)
            rows.append(dict(id=hashlib.sha256(target.encode()).hexdigest()[:20], document=document,
                             source=source, split=split, input="".join(c[0] for c in codes),
                             pinyin=" ".join(pinyin), gold=[target],
                             context=text[max(0, match.start() - 64):match.start()], category="domain"))
    args.out.mkdir(parents=True, exist_ok=True)
    selected = {s: sorted((r for r in rows if r["split"] == s and not r.get("excluded")), key=lambda r: r["id"])
                for s in ("train", "dev", "test")}
    for split, items in selected.items():
        with (args.out / f"{split}.jsonl").open("w") as stream:
            stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in items)
        counts[split] = len(items)
        counts[split + "_chars"] = sum(len(r["gold"][0]) for r in items)
        for source, n in Counter(r["source"] for r in items).items():
            counts[f"{source}/{split}_rows"] = n
    # Words with same-key rivals that general training data barely covers.
    groups = {}
    for code, word in sqlite3.connect(args.database).execute(
            "SELECT code, word FROM entry WHERE length(word) BETWEEN 2 AND 4 GROUP BY code, word"):
        groups.setdefault(code, set()).add(word)
    words = {w for group in groups.values() if len(group) > 1 for w in group}
    general = general_counts(args.general, words)
    domain = Counter()
    for row in selected["train"]:
        domain.update(features(row["gold"][0], words))
    # As built for domain-v2: words absent from general data were not counted
    # as rare, although they are the least covered. Revisit before reuse.
    rare = {w for w, n in domain.items() if n >= args.rare_domain and 1 <= general[w] <= args.rare_general}
    lines = []
    for row in selected["train"]:
        lines += [json.dumps(row, ensure_ascii=False)] * (1 + args.rare_repeat * bool(features(row["gold"][0], rare)))
    counts["rare_words"] = len(rare)
    counts["rare_rows"] = sum(bool(features(r["gold"][0], rare)) for r in selected["train"])
    manifest = dict(local=dict(monorepo="tracked .org/.md/.txt/.rst, librime essay.txt excluded",
                               assistant=bool(args.transcripts)),
                    external={n: dict(s, repository=f"https://github.com/{s['repository']}")
                              for n, s in sources.items()},
                    protected={str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                               for p in args.protected},
                    counts=dict(counts), pypinyin="0.55.0",
                    coverage_upsampling=dict(general_range=[1, args.rare_general], domain_min=args.rare_domain,
                                             repeat=args.rare_repeat),
                    limitation="Automatic pinyin labels; assistant replies are model-written; each source's "
                               "attribution and share-alike terms apply to derived text.")
    manifest["files"] = {s: hashlib.file_digest((args.out / f"{s}.jsonl").open("rb"), "sha256").hexdigest()
                         for s in ("train", "dev", "test")}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # Only newline separates rows; JSON strings can hold U+2028 and similar.
    extra = (args.extra / "train.jsonl").read_text().rstrip("\n").split("\n")
    repeat = max(1, round(args.share / (1 - args.share) * len(extra) / len(lines)))
    args.combined.mkdir(parents=True, exist_ok=True)
    with (args.combined / "train.jsonl").open("w") as stream:
        stream.writelines(line + "\n" for line in extra)
        for _ in range(repeat):
            stream.writelines(line + "\n" for line in lines)
    with (args.combined / "dev.jsonl").open("w") as stream:
        stream.write((args.extra / "dev.jsonl").read_text())
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in selected["dev"])
    (args.combined / "manifest.json").write_text(json.dumps(dict(
        extra=json.loads((args.extra / "manifest.json").read_text()), domain=manifest,
        domain_repeat=repeat, domain_share=args.share), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(counts=dict(counts), repeat=repeat), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
