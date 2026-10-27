#!/usr/bin/env python3
"""Audit corpus overlap and assemble reviewed, versioned Japanese evaluation sets.

The historical benchmark is never rewritten. Natural cases come from the
500 protected Tatoeba reserve rows, not the previously scored dev/test rows.
Controlled cases are explicitly assistant-authored. Labels are separated
from sealed inputs before model experiments. Public-model pretraining
membership cannot be established by this local-corpus audit.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import unicodedata


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def norm(text):
    return unicodedata.normalize("NFKC", text)


def read(path):
    data = json.loads(Path(path).read_text())
    return data["cases"] if isinstance(data, dict) else data


def canonical(row, source):
    identity = str(row.get("id", row.get("case_id", row.get("case_index"))))
    if identity == "None":
        raise ValueError("case needs a stable identity")
    answers = row.get("answers", [row["gold"]] if "gold" in row else None)
    if not answers or not all(isinstance(t, str) and t for t in answers):
        raise ValueError("case needs nonempty answers")
    return dict(id=source + ":" + identity, source_id=identity, source=source,
                context=row.get("context", ""), right_context=row.get("right_context", ""),
                kana=row.get("kana", row.get("input")), answers=answers)


def grams(text, n=8):
    return {text[i:i+n] for i in range(len(text)-n+1)}


def overlap(cases, corpora, progress=None):
    """Check target and available surrounding text, not just output segments.

    Shared 8-character strings are conservative screening evidence, not a
    declaration of contamination. Full matches are reported independently.
    Short targets (e.g. 橋) are not useful membership evidence; their surrounding
    text is still audited. At most two examples per case/corpus are retained.
    """
    index = collections.defaultdict(set)
    patterns = {}
    for case in cases:
        cid = case["id"]
        if cid in patterns:
            raise ValueError("duplicate case identity: " + cid)
        targets = {norm(t) for t in case["answers"]}
        spans = {norm(case["context"] + t + case["right_context"]) for t in case["answers"]}
        patterns[cid] = (targets, spans)
        for text in targets | spans:
            for g in grams(text):
                index[g].add(cid)
    hits = {c["id"]: {} for c in cases}
    stats = {}
    for path in corpora:
        path = Path(path)
        rows = 0
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                rows += 1
                body = norm(row.get("context", "") + row["text"] + row.get("right_context", ""))
                candidates = set()
                body_grams = grams(body)
                for g in body_grams:
                    candidates.update(index.get(g, ()))
                for cid in candidates:
                    targets, spans = patterns[cid]
                    entry = hits[cid].setdefault(str(path.resolve()), dict(shared8_rows=0, full_target_rows=0,
                                                                         full_surrounding_rows=0, examples=[]))
                    entry["shared8_rows"] += 1
                    full_target = any(len(t) >= 8 and t in body for t in targets)
                    full_span = any(len(t) >= 12 and t in body for t in spans)
                    entry["full_target_rows"] += full_target
                    entry["full_surrounding_rows"] += full_span
                    if len(entry["examples"]) < 2:
                        entry["examples"].append(dict(line=number, source=row.get("source"), doc=row.get("doc"),
                                                      context=row.get("context", ""), text=row["text"],
                                                      full_target=full_target, full_surrounding=full_span))
                if progress and rows % 250000 == 0:
                    progress(dict(corpus=str(path), rows=rows))
        stats[str(path.resolve())] = dict(rows=rows, sha256=digest(path))
    return dict(protocol="NFKC; contiguous 8-char screening plus exact target (>=8) and surrounding (>=12) matches; context+text scanned",
                limitation="Audits local training/validation files only; upstream model pretraining membership is unknown",
                corpora=stats, cases=hits)


def priority(row):
    return hashlib.sha256(("wime-japanese-eval-20261028/" + row["id"]).encode()).hexdigest()


def apply_review(row, review):
    if review.get("original_kana") != row["kana"] or review.get("original_answers") != row["answers"]:
        raise ValueError("review no longer matches source: " + row["id"])
    return dict(row, kana=review.get("kana", row["kana"]), answers=review.get("answers", row["answers"]))


def assemble(reserve, reviews, controlled, audit, historical, count):
    """Only pre-reviewed, locally disjoint natural rows enter a new split."""
    historical_answers = {norm(t) for c in historical for t in c["answers"]}
    historical_kana = {norm(c["kana"]) for c in historical}
    edits = {r["id"]: r for r in reviews}
    if len(edits) != len(reviews):
        raise ValueError("duplicate review identity")
    pool, excluded, selected_grams = [], [], set()
    for row in sorted(reserve, key=priority):
        review = edits.get(row["id"])
        if review:
            row = apply_review(row, review)
        reason = None
        if not review or review["status"] != "approved":
            reason = "not_approved"
        elif min(map(len, row["answers"])) < 8:
            reason = "too_short_for_membership_audit"
        elif row["id"] not in audit["cases"]:
            raise ValueError("case missing from overlap audit: " + row["id"])
        elif audit["cases"][row["id"]]:
            reason = "training_or_validation_shared8"
        elif any(norm(t) in historical_answers for t in row["answers"]) or norm(row["kana"]) in historical_kana:
            reason = "previous_evaluation_overlap"
        if reason:
            excluded.append(dict(id=row["id"], reason=reason))
            continue
        # Prevent duplicate readings/answers from crossing natural splits.
        if any(norm(row["kana"]) == norm(r["kana"]) or
               set(map(norm, row["answers"])) & set(map(norm, r["answers"])) for r in pool):
            excluded.append(dict(id=row["id"], reason="duplicate_natural_case"))
            continue
        row_grams = set().union(*(grams(norm(t)) for t in row["answers"]))
        if row_grams & selected_grams:
            excluded.append(dict(id=row["id"], reason="shared8_with_selected_natural"))
            continue
        pool.append(row)
        selected_grams.update(row_grams)
    if len(pool) < 2 * count:
        raise ValueError(f"need {2*count} reviewed/disjoint rows, have {len(pool)}")
    natural = []
    for i, row in enumerate(pool[:2*count]):
        natural.append(dict(row, split="dev" if i % 2 == 0 else "sealed", domain="natural_tatoeba",
                            topic_group=row["id"], provenance="Tatoeba protected reserve; independently reviewed before model inference"))
    cases = natural + controlled
    ids, groups = set(), {}
    for case in cases:
        if case["id"] in ids:
            raise ValueError("duplicate final ID")
        ids.add(case["id"])
        if case["split"] not in ("dev", "sealed"):
            raise ValueError("invalid split")
        old = groups.setdefault(case["topic_group"], case["split"])
        if old != case["split"]:
            raise ValueError("topic group crosses split")
    return cases, dict(eligible_natural=len(pool), selected_natural=len(natural), excluded=excluded)


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--cases", action="append", required=True, help="SOURCE=JSON")
    a.add_argument("--corpus", action="append", required=True, type=Path)
    a.add_argument("--reviews", type=Path, help="guarded natural-case edits to apply before overlap checking")
    a.add_argument("--out", required=True, type=Path)
    a = sub.add_parser("assemble")
    for flag in ("reserve", "reviews", "controlled", "overlap", "historical", "out"):
        a.add_argument("--"+flag, required=True, type=Path)
    a.add_argument("--natural-per-split", type=int, default=100)
    args = ap.parse_args()
    if args.action == "audit":
        paths, cases = {}, []
        for item in args.cases:
            name, p = item.split("=", 1)
            paths[str(Path(p).resolve())] = digest(p)
            cases.extend(canonical(row, name) for row in read(p))
        if args.reviews:
            reviews = {r["id"]:r for r in read(args.reviews)}
            cases = [apply_review(c, reviews[c["id"]]) if c["id"] in reviews else c for c in cases]
            paths[str(args.reviews.resolve())] = digest(args.reviews)
        report = overlap(cases, args.corpus, lambda r: print(json.dumps(r), flush=True))
        report["inputs"] = paths
        write(args.out, report)
        print(json.dumps(dict(cases=len(cases), shared8=sum(bool(v) for v in report["cases"].values())), flush=True))
    else:
        if args.out.exists():
            raise ValueError("evaluation output already exists; choose a new version directory")
        controlled_data = json.loads(args.controlled.read_text())
        if not controlled_data.get("metadata", {}).get("independent_review_complete"):
            raise ValueError("controlled cases require independent review before sealing")
        controlled = controlled_data["cases"]
        reserve = [canonical(row, "reserve") for row in read(args.reserve)]
        audit = json.loads(args.overlap.read_text())
        if audit["inputs"].get(str(args.reserve.resolve())) != digest(args.reserve):
            raise ValueError("reserve changed after overlap audit")
        if audit["inputs"].get(str(args.controlled.resolve())) != digest(args.controlled):
            raise ValueError("controlled cases changed after overlap audit")
        if audit["inputs"].get(str(args.reviews.resolve())) != digest(args.reviews):
            raise ValueError("reviewed answers were not covered by the overlap audit")
        # A common short target is expected; a full surrounding match excludes a controlled case.
        for case in controlled:
            key = "controlled:" + case["id"]
            if key not in audit["cases"]:
                raise ValueError("controlled case absent from overlap audit")
            if any(v["full_surrounding_rows"] for v in audit["cases"][key].values()):
                raise ValueError("controlled surrounding text overlaps local corpus: " + key)
        cases, selection = assemble(reserve, read(args.reviews), controlled, audit, read(args.historical), args.natural_per_split)
        args.out.mkdir(parents=True)
        dev = [r for r in cases if r["split"] == "dev"]
        sealed = [r for r in cases if r["split"] == "sealed"]
        write(args.out / "dev.json", dev)
        # Notes can reveal answers; omit them as well as the labels.
        write(args.out / "sealed.inputs.json", [{k:v for k,v in r.items() if k not in ("answers", "note")} for r in sealed])
        write(args.out / "sealed.labels.json", [dict(id=r["id"], answers=r["answers"]) for r in sealed])
        # Future training excludes complete surrounding text for short controlled
        # targets; excluding isolated words such as 橋 would remove useful data.
        protection = sorted({norm(r["context"] + a + r["right_context"])
                             for r in cases for a in r["answers"]
                             if len(norm(r["context"] + a + r["right_context"])) >= 8})
        write(args.out / "training-exclusions.json", protection)
        inputs = [args.reserve,args.reviews,args.controlled,args.overlap,args.historical,Path(__file__)]
        manifest = dict(version="20261028-v1", selection=selection,
                        counts=dict(collections.Counter((r["split"]+"/"+r["source"] if "source" in r else r["split"]+"/controlled") for r in cases)),
                        policy="Data reviewed and hashed before inference; dev selects ranking; sealed labels consumed only after frozen predictions",
                        limitation="Controlled cases are assistant-authored and agent-reviewed; natural Tatoeba pretraining overlap is unknown",
                        inputs={str(p.resolve()):digest(p) for p in inputs},
                        files={p.name:digest(p) for p in args.out.glob("*.json")})
        write(args.out / "manifest.json", manifest)
        print(json.dumps(dict(out=str(args.out), counts=manifest["counts"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
