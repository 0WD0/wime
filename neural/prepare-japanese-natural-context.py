#!/usr/bin/env python3
"""Prepare unseen natural insertion contexts, without model inference.

Recent unapproved Tatoeba transcriptions are attributed automatic readings,
not human-certified labels. UniDic supplies word boundaries only; all readings
come from the original annotation and require independent linguistic review.
Full-source corpus overlap is checked before a deterministic review queue is
selected. Each approved source can later produce full/short/no-right variants.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unicodedata

_SPEC = importlib.util.spec_from_file_location(
    "evaluation_audit", Path(__file__).with_name("prepare-japanese-evaluation.py"))
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def normalize(text):
    return unicodedata.normalize("NFKC", text)


def kanji(c):
    return "㐀" <= c <= "鿿" or c in "々〆"


def hira(text):
    return "".join(chr(ord(c)-0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def units(transcription):
    """Indivisible bracket annotations plus literal characters, with offsets."""
    out, surface, reading, i = [], "", "", 0
    while i < len(transcription):
        if transcription[i] == "[":
            j = transcription.index("]", i)
            text, *parts = transcription[i+1:j].split("|")
            if not text or not parts or any(not part for part in parts):
                raise ValueError("empty annotated surface or reading")
            kana = hira("".join(parts))
            end = j+1
        else:
            text, kana, end = transcription[i], hira(transcription[i]), i+1
        if any(kanji(c) for c in kana):
            raise ValueError("transcription has an unread kanji")
        out.append(dict(text=text, kana=kana, surface_start=len(surface),
                        surface_end=len(surface)+len(text), kana_start=len(reading),
                        kana_end=len(reading)+len(kana), annotation_start=i, annotation_end=end))
        surface += text
        reading += kana
        i = end
    return surface, reading, out


def priority(identifier):
    return hashlib.sha256(("wime-natural-context-v3/"+str(identifier)).encode()).hexdigest()


def prior_material(paths):
    ids, texts, kana = set(), set(), set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        rows = data["cases"] if isinstance(data, dict) else data
        for c in rows:
            # AJIMEE case_index/case_id are not Tatoeba sentence identities.
            for key in ("source_id", "id"):
                value = str(c.get(key, ""))
                candidate = value.rsplit(":", 1)[-1]
                if candidate.isdigit():
                    ids.add(candidate)
            answers = c.get("answers", [c["gold"]] if "gold" in c else [])
            for answer in answers:
                texts.add(normalize(answer))
                texts.add(normalize(c.get("context", "")+answer+c.get("right_context", "")))
            if c.get("source_sentence"):
                texts.add(normalize(c["source_sentence"]))
            if c.get("kana", c.get("input")):
                kana.add(normalize(c.get("kana", c.get("input"))))
    return ids, texts, kana


def choose_span(sid, text, reading, annotations, words):
    """Intersect annotation and morphological boundaries; never infer readings.

    A fixed hash picks one eligible span, rather than choosing a model failure
    or an especially ambiguous word. The source's remaining text is preserved.
    """
    boundaries = {0: 0}
    for unit in annotations:
        boundaries[unit["surface_end"]] = unit["kana_end"]
    starts, ends, offset = [], set(), 0
    for surface, pos in words:
        if text[offset:offset+len(surface)] != surface:
            raise ValueError("word segmentation does not preserve source spacing")
        if pos not in ("助詞", "助動詞", "補助記号", "記号", "空白"):
            starts.append(offset)
        offset += len(surface)
        ends.add(offset)
    if offset != len(text):
        raise ValueError("word segmentation does not cover source")
    spans = []
    for start in starts:
        if start not in boundaries or start > 40:
            continue
        for end in ends:
            if end not in boundaries or not 3 <= end-start <= 20 or not 4 <= len(text)-end <= 40:
                continue
            target = text[start:end]
            target_reading = reading[boundaries[start]:boundaries[end]]
            if not any(kanji(c) for c in target) or not 3 <= len(target_reading) <= 40:
                continue
            if any(c in "。！？!?\n\r" for c in target):
                continue
            spans.append((start, end, boundaries[start], boundaries[end]))
    if not spans:
        return None
    # Prefer substantial phrase spans over 1-word fragments, but preserve broad
    # positional diversity via a fixed hash within that band.
    preferred = [s for s in spans if 6 <= s[1]-s[0] <= 16]
    options = preferred or spans
    start, end, ks, ke = min(options, key=lambda s: priority(f"{sid}:{s[0]}:{s[1]}"))
    return dict(context=text[:start], kana=reading[ks:ke], answers=[text[start:end]],
                right_context=text[end:], source_span=[start, end], source_kana_span=[ks, ke])


def extract(source_dir, prior_paths):
    import fugashi
    tagger = fugashi.Tagger()
    details = source_dir/"jpn_sentences_detailed.tsv"
    transcriptions = source_dir/"jpn_transcriptions.tsv"
    old_ids, old_text, old_kana = prior_material(prior_paths)
    dates = {r[0]: dict(year=r[4][:4], text=r[2], author=r[3], added=r[4], modified=r[5])
             for r in csv.reader(details.open(), delimiter="\t", quoting=csv.QUOTE_NONE)}
    counts, cases, seen_ids, seen_text = collections.Counter(), [], set(), set()
    for row in csv.reader(transcriptions.open(), delimiter="\t", quoting=csv.QUOTE_NONE):
        sid = row[0]
        source = dates.get(sid)
        if not source or source["year"] not in ("2025", "2026") or row[2] != "Hrkt" or row[3]:
            continue
        counts["recent_unapproved_rows"] += 1
        if sid in old_ids:
            counts["prior_id"] += 1
            continue
        text = source["text"]
        if not 12 <= len(text) <= 80 or any(c.isspace() for c in text):
            counts["length_or_whitespace"] += 1
            continue
        if normalize(text) in old_text:
            counts["prior_text"] += 1
            continue
        if sid in seen_ids or normalize(text) in seen_text:
            counts["duplicate_source"] += 1
            continue
        try:
            surface, reading, annotations = units(row[4])
            if surface != text:
                raise ValueError("surface mismatch")
            if normalize(reading) in old_kana:
                counts["prior_full_reading"] += 1
                continue
            words = [(w.surface, w.feature.pos1) for w in tagger(text)]
            span = choose_span(sid, text, reading, annotations, words)
        except ValueError:
            counts["transcription_or_segmentation_invalid"] += 1
            continue
        if span is None:
            counts["no_eligible_span"] += 1
            continue
        seen_ids.add(sid)
        seen_text.add(normalize(text))
        cases.append(dict(id="natural-source:"+sid, source_id=sid, source_sentence=text,
                          source_reading=reading, source_transcription=row[4],
                          source_units=annotations, source_words=words, **source, **span,
                          provenance="Tatoeba unapproved automatic transcription; linguistic review required"))
    cases.sort(key=lambda c: priority(c["source_id"]))
    counts["extracted"] = len(cases)
    return dict(version=3, policy="Unapproved 2025/2026 readings; no model calls; one deterministic annotation-aligned span per unseen source", counts=dict(counts),
                inputs={str(p.resolve()):audit.digest(p) for p in [details, transcriptions, Path(__file__), *prior_paths]},
                segmentation="fugashi/UniDic word boundaries only, never reading substitution", cases=cases)


def select_queue(pool, report, count):
    """Zero corpus shared8 and no source-shared8 among queued cases."""
    selected, used, excluded = [], set(), collections.Counter()
    for c in pool["cases"]:
        if c["id"] not in report["cases"]:
            raise ValueError("source absent from audit")
        if report["cases"][c["id"]]:
            excluded["corpus_shared8"] += 1
            continue
        grams = audit.grams(normalize(c["source_sentence"]))
        if grams & used:
            excluded["queue_shared8"] += 1
            continue
        selected.append(c)
        used.update(grams)
    if len(selected) < count:
        raise ValueError(f"need {count} eligible sources, have {len(selected)}")
    return dict(policy="Fixed hash order; all queued full sources have zero local corpus shared8 and zero shared8 with each other. Review labels before inference.",
                eligible=len(selected), requested=count, excluded=dict(excluded), cases=selected[:count])


def variants(source, split, review):
    if review.get("id") != source["id"] or review.get("original_kana") != source["kana"] or review.get("original_answers") != source["answers"]:
        raise ValueError("review no longer matches source")
    if review.get("status") != "approved":
        raise ValueError("source needs independent approval")
    answers = review.get("answers", source["answers"])
    kana = review.get("kana", source["kana"])
    if not answers or not all(isinstance(a, str) and a for a in answers) or not isinstance(kana, str) or not kana:
        raise ValueError("review must supply nonempty reading and answers")
    if split not in ("dev", "sealed"):
        raise ValueError("invalid split")
    return [dict(id=f"natural-v3:{source['source_id']}:{mode}", split=split,
                 topic_group="natural-v3:"+source["source_id"], domain="natural_tatoeba_insertion",
                 case_type="natural_"+mode, context=source["context"], kana=kana,
                 right_context=right, answers=answers)
            for mode, right in (("full", source["right_context"]),
                                ("short", source["right_context"][:2]), ("no_right", ""))]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--prior", action="append", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("audit")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--corpus", action="append", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("queue")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--count", type=int, default=400)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refuse to overwrite an existing artifact")
    if args.action == "extract":
        result = extract(args.source_dir, args.prior)
    elif args.action == "audit":
        data = json.loads(args.pool.read_text())
        cases = [dict(id=c["id"], answers=[c["source_sentence"]], context="", right_context="") for c in data["cases"]]
        result = audit.overlap(cases, args.corpus, lambda r: print(json.dumps(r), flush=True))
        result["inputs"] = {str(p.resolve()):audit.digest(p) for p in (args.pool, Path(__file__))}
    else:
        data, report = json.loads(args.pool.read_text()), json.loads(args.audit.read_text())
        if report["inputs"].get(str(args.pool.resolve())) != audit.digest(args.pool):
            raise ValueError("pool changed after corpus audit")
        result = select_queue(data, report, args.count)
        result["inputs"] = {str(p.resolve()):audit.digest(p) for p in (args.pool, args.audit, Path(__file__))}
    write(args.out, result)
    print(json.dumps(dict(out=str(args.out), counts=result.get("counts"), cases=len(result.get("cases", [])),
                          eligible=result.get("eligible")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
