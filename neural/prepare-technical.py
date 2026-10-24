#!/usr/bin/env python3
"""Prepare document-disjoint Chinese Kubernetes prose from a pinned archive.

The archive is a text corpus, not a vendored software dependency. Never execute
archive contents or extract paths. Retain English/punctuation in the context.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

from pypinyin import lazy_pinyin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
from wime_gen import Converter, load_rules

REVISION = "fe59320b9e3858bc9d41b4ad8d86b1c77568ceb4"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--algebra", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude-data", type=Path, required=True,
                   help="Other corpus splits; exclude every shared target string")
    args = p.parse_args()
    converter = Converter(rules=load_rules(args.algebra))
    excluded = set()
    for split in ("train", "dev", "test"):
        for line in (args.exclude_data / f"{split}.jsonl").open():
            excluded.update(json.loads(line)["gold"])
    rows, seen, counts = [], {}, Counter()
    license_text = None
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if member.name.split("/", 1)[0] != "website-" + REVISION:
                raise ValueError("archive does not match the pinned revision")
            if member.name.endswith("/LICENSE") and member.name.count("/") == 1:
                license_text = archive.extractfile(member).read().decode()
            if (not member.isfile() or "/content/zh-cn/docs/" not in member.name
                    or not member.name.endswith(".md") or "/reference/" in member.name):
                continue
            text = archive.extractfile(member).read().decode()
            text = re.sub(r"\A---\n.*?\n---\n|<!--.*?-->|```.*?```|{{.*?}}", " ", text, flags=re.S)
            text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
            text = re.sub(r"`([^`]*)`", r"\1", text)
            text = re.sub(r"</?[^>]+>", " ", text)
            text = re.sub(r"\n\s*\|.*", " ", text)
            text = re.sub(r"[ \t]+", " ", text)
            # Soft line wraps in Chinese prose should not split its meaning.
            text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)
            document = hashlib.sha256(text.encode()).hexdigest()
            bucket = int(document[:8], 16) % 100
            split = "test" if bucket < 10 else "dev" if bucket < 20 else "train"
            counts[split + "_documents"] += 1
            for match in re.finditer(r"[\u4e00-\u9fff]{6,32}", text):
                target = match.group()
                if target in excluded:
                    counts["other_corpus_overlap"] += 1
                    continue
                if target in seen:
                    prior = rows[seen[target]]
                    if prior["split"] != split:
                        prior["excluded"] = True
                    counts["duplicates"] += 1
                    continue
                pinyin = lazy_pinyin(target)
                codes = [sorted(converter.codes(p)) for p in pinyin]
                if any(not c for c in codes):
                    continue
                code = "".join(c[0] for c in codes)
                if len(code) != 2 * len(target):
                    continue
                seen[target] = len(rows)
                rows.append(dict(id=hashlib.sha256(target.encode()).hexdigest()[:20],
                                 document=document, source=member.name, split=split,
                                 input=code, pinyin=" ".join(pinyin), gold=[target],
                                 context=text[max(0, match.start() - 64):match.start()]))
    args.out.mkdir(parents=True, exist_ok=True)
    if not license_text or "Creative Commons" not in license_text:
        raise ValueError("archive license is missing or unexpected")
    (args.out / "SOURCE-LICENSE").write_text(license_text)
    for split in ("train", "dev", "test"):
        selected = sorted((r for r in rows if r["split"] == split and not r.get("excluded")),
                          key=lambda r: r["id"])
        with (args.out / f"{split}.jsonl").open("w") as f:
            for row in selected:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(selected)
        counts[split + "_chars"] = sum(len(r["gold"][0]) for r in selected)
    manifest = dict(source="https://github.com/kubernetes/website", revision=REVISION,
                    archive_sha256=hashlib.file_digest(args.archive.open("rb"), "sha256").hexdigest(),
                    license="CC-BY-4.0", attribution="The Kubernetes Authors, Chinese documentation (modified into pinyin/text pairs).",
                    counts=dict(counts), pypinyin="0.55.0",
                    limitation="Automatic pinyin labels; upstream pretrained-model overlap unknown.")
    manifest["files"] = {s: hashlib.file_digest((args.out / f"{s}.jsonl").open("rb"), "sha256").hexdigest()
                         for s in ("train", "dev", "test")}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
