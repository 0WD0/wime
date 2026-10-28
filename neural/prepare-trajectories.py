#!/usr/bin/env python3
"""Select source sentences, review boundaries, then freeze a contextual IME diagnostic."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from replay import fingerprint
from trajectories import validate


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def candidates(root, out, per_bucket=16):
    pool, counts = defaultdict(list), Counter()
    db = sqlite3.connect((root / "documents.sqlite3").resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    for document in db.execute("SELECT * FROM documents WHERE split='dev'"):
        for match in re.finditer(r"[^。！？!?\n]+[。！？!?\n]?", document["text"]):
            raw = match.group()
            text = raw.rstrip("。！？!?\n").strip()
            start = match.start() + len(raw) - len(raw.lstrip())
            terminal = raw[-1] if raw[-1] in "。！？!?" else ""
            counts["segments"] += 1
            if not re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
                counts["excluded_mixed_or_internal_punctuation"] += 1
                continue
            if not 6 <= len(text) <= 48:
                counts["excluded_length"] += 1
                continue
            if len(set(text)) < 4:
                counts["excluded_repetition"] += 1
                continue
            if document["text"][start:start+len(text)+len(terminal)] != text + terminal:
                counts["excluded_terminal_spacing"] += 1
                continue
            bucket = "6-12" if len(text) <= 12 else "13-24" if len(text) <= 24 else "25-48"
            counts[document["source"] + "/" + bucket] += 1
            identity = hashlib.sha256(f"trajectory-v1:{document['id']}:{start}".encode()).hexdigest()
            pool[document["source"] + "/" + bucket].append(dict(
                id=identity[:20], document=document["id"], group=document["group_id"], source=document["source"],
                bucket=bucket, offset=start, text=text, terminal=terminal,
                context=document["text"][max(0, start - 64):start]))
    db.close()
    selected, groups = [], set()
    for key in sorted(pool):
        count = 0
        for row in sorted(pool[key], key=lambda row: row["id"]):
            if row["group"] in groups:
                continue
            if count >= per_bucket:
                break
            selected.append(row)
            groups.add(row["group"])
            count += 1
    write(out, selected)
    write(out.with_suffix(".manifest.json"), dict(
        cleaned_manifest_sha256=fingerprint(root / "manifest.json"), candidates_sha256=fingerprint(out),
        counts=dict(counts), selected=dict(Counter(row["source"] + "/" + row["bucket"] for row in selected)),
        policy="dev only; deterministic hash order; unique source group; no model results consulted"))


def build(args):
    from corpus_cleaned import CleanedDocuments
    from corpus_integrity import reading_assets
    from corpus_readings import Lexicon, Reader
    from corpus_tones import reading_notes, validate_tones, READING_SCHEMA, TONE_POLICY
    from mandarin_frontend import MandarinFrontend

    proposed = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    decisions = {row["id"]: row for row in review}
    if len(decisions) != len(review) or set(decisions) != {row["id"] for row in proposed}:
        raise ValueError("every candidate needs exactly one review decision")
    selection = json.loads(args.candidates.with_suffix(".manifest.json").read_text())
    if selection["candidates_sha256"] != fingerprint(args.candidates):
        raise ValueError("candidate selection changed after creation")
    if selection["cleaned_manifest_sha256"] != fingerprint(args.cleaned / "manifest.json"):
        raise ValueError("cleaned source version changed")
    accepted = [row for row in proposed if decisions[row["id"]]["accept"]]
    with CleanedDocuments(args.cleaned) as documents:
        records = []
        for row in accepted:
            document = documents._document(row["document"])
            if document["split"] != "dev" or document["group_id"] != row["group"]:
                raise ValueError("trajectory must remain in its original development group")
            start, end = row["offset"], row["offset"] + len(row["text"])
            if document["text"][start:end] != row["text"] or document["text"][max(0,start-64):start] != row["context"]:
                raise ValueError("review text differs from cleaned document")
            if document["text"][start:end+len(row["terminal"])] != row["text"] + row["terminal"]:
                raise ValueError("annotation window must preserve spacing before terminal punctuation")
            decision = decisions[row["id"]]
            if not row["text"].startswith(decision["prefix"]):
                raise ValueError("reviewed prefix is not a literal sentence prefix")
            records.append(dict(id=row["id"], source=row["source"], category=document["category"],
                                document=row["document"], group=row["group"], content=document["content"],
                                split="dev", offset=start, context=row["context"], context_dropped=False,
                                gold=[row["text"]], reading_window=[start,end+len(row["terminal"])],
                                boundaries=[dict(end=len(decision["prefix"]), reason=decision["reason"])],
                                terminal=row["terminal"]))
    frontend = MandarinFrontend(args.g2pw)
    reader = Reader(Lexicon(args.database), frontend)
    readings = reader.readings([row["gold"][0] + row["terminal"] for row in records])
    for row, reading in zip(records, readings, strict=True):
        size = len(row["gold"][0])
        if not all(reading["keys"][:size]):
            raise ValueError(f"unresolved or unreachable reading in {row['id']}")
        row.update(input="".join(reading["keys"][:size]), pinyin=reading["pinyin"][:size], tones=reading["tones"][:size])
        notes = reading_notes(reading, 0, size)
        if notes:
            row["reading_notes"] = notes
        validate_tones(row)
        validate(row)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "cases.json", records)
    write(args.out / "boundary-review.json", review)
    write(args.out / "candidates.json", proposed)
    write(args.out / "manifest.json", dict(
        format="wime-input-trajectories-v1", reading_schema=READING_SCHEMA, tone_policy=TONE_POLICY,
        provenance=dict(cleaned_corpus=dict(path=str((args.cleaned / "manifest.json").resolve()),
                                           sha256=fingerprint(args.cleaned / "manifest.json")),
                        reading_assets=reading_assets(args.g2pw, args.g2pw / "chinese-roberta-wwm-ext-large")),
        selection=selection, boundary_review="agent reviewed before inference; not human verified",
        cases=len(records), documents=len({row["document"] for row in records}),
        source_counts=dict(Counter(row["source"] for row in records)),
        database_sha256=fingerprint(args.database),
        code_sha256={name:fingerprint(Path(__file__).with_name(name)) for name in
                     ("prepare-trajectories.py", "trajectories.py", "mandarin_frontend.py", "corpus_tones.py")},
        scope="Complete 6–48 Han sentences without internal punctuation; terminal punctuation not typed. "
              "25+ is decoder capacity diagnosis beyond the current 24-syllable editor window.",
        limitations=["Development documents previously used for checkpoint selection; not an untouched test.",
                     "Source/length sampling and boundary review are diagnostic, not measured user behavior.",
                     "Mixed-script and internally punctuated sentences are excluded, not concatenated.",
                     "Correct committed prefix is an oracle condition; prefix selection success is measured separately."]
    ))
    print(json.dumps(dict(cases=len(records), out=str(args.out)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    pick = sub.add_parser("candidates")
    pick.add_argument("--cleaned", type=Path, required=True)
    pick.add_argument("--out", type=Path, required=True)
    pick.add_argument("--per-bucket", type=int, default=16)
    freeze = sub.add_parser("build")
    for name in ("cleaned", "candidates", "review", "database", "g2pw", "out"):
        freeze.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "candidates":
        candidates(args.cleaned, args.out, args.per_bucket)
    else:
        build(args)


if __name__ == "__main__":
    main()
