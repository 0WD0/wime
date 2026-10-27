#!/usr/bin/env python3
"""Replay fixed Japanese cases against reading tables with identical models.

Cases contain suite, case_index, kana, context and answers (all acceptable
spellings). Rows are appended and flushed individually for resumable runs.
Latency measures fresh, complete compositions, not incremental keystrokes.
"""
import argparse
import hashlib
import json
import time
import unicodedata
from pathlib import Path

from japanese import Decoder, KanaModel, RERANK, Table, Zenz
from rescorer import Rescorer


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def outcome(answers, best, normalize=True):
    norm = (lambda s: unicodedata.normalize("NFKC", s)) if normalize else (lambda s: s)
    expected = {norm(s) for s in answers}
    return next((i + 1 for i, s in enumerate(best) if norm(s) in expected), None)


def summarize(rows):
    result = {}
    for suite in sorted({r["suite"] for r in rows}):
        result[suite] = {}
        for variant in sorted({r["variant"] for r in rows}):
            selected = [r for r in rows if r["suite"] == suite and r["variant"] == variant]
            if not selected:
                continue
            times = sorted(r["elapsed_ms"] for r in selected)
            result[suite][variant] = dict(
                total=len(selected), top1=sum(r["rank"] == 1 for r in selected),
                top5=sum(r["rank"] is not None and r["rank"] <= 5 for r in selected),
                spellable=sum(r["spellable"] for r in selected),
                ms_p50=times[len(times) // 2], ms_p95=times[int(.95 * len(times))])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--table", action="append", required=True, help="NAME=PATH; repeated for paired runs")
    ap.add_argument("--rescorer", type=Path, required=True)
    ap.add_argument("--kana-model", type=Path, required=True)
    ap.add_argument("--zenz", type=Path, required=True)
    ap.add_argument("--weight", type=float, default=0.6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tables = dict(pair.split("=", 1) for pair in args.table)
    if len(tables) != len(args.table):
        ap.error("table names must be unique")
    cases = json.loads(args.cases.read_text())
    keys = [(r["suite"], r["case_index"]) for r in cases]
    if len(set(keys)) != len(keys):
        ap.error("case identity must be unique within each suite")
    for r in cases:
        if not isinstance(r["answers"], list) or not r["answers"]:
            ap.error("each case must have acceptable answers")
    files = [args.cases, *map(Path, tables.values()), args.kana_model,
             *sorted(args.rescorer.glob("*.safetensors")), args.rescorer / "tokenizer.json",
             args.zenz / "model.safetensors", args.zenz / "tokenizer.json",
             Path(__file__), Path(__file__).with_name("japanese.py"), Path(__file__).with_name("rescorer.py")]
    manifest = dict(weight=args.weight, rerank_weight=RERANK, tables=tables,
                    normalization="NFKC full-string match; any supplied answer",
                    latency="warm models, fresh session per complete composition; alternating variant order",
                    cases=len(cases), sha256={str(p.resolve()): digest(p) for p in files})
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            ap.error("inputs changed; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    path = args.out / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r["variant"], r["suite"], r["case_index"]) for r in rows}
    if len(done) != len(rows) or any(v not in tables or (s, i) not in set(keys) for v, s, i in done):
        ap.error("invalid or duplicate saved rows")
    rescorer = Rescorer(args.rescorer, args.weight)
    kana_model, zenz = KanaModel(args.kana_model), Zenz(args.zenz)
    decoders = {name: Decoder(Table(path, rescorer.tokenizer), rescorer, kana_model, args.weight, zenz=zenz)
                for name, path in tables.items()}
    for decoder in decoders.values():
        decoder.decode("", "にほんご")
        decoder.session = None
    start = time.monotonic()
    with path.open("a", buffering=1) as stream:
        for n, case in enumerate(cases):
            order = list(decoders) if n % 2 == 0 else list(reversed(decoders))
            for name in order:
                if (name, case["suite"], case["case_index"]) in done:
                    continue
                decoder = decoders[name]
                decoder.session = None
                tick = time.perf_counter()
                ranked = decoder.decode(case.get("context", ""), case["kana"])
                elapsed = 1000 * (time.perf_counter() - tick)
                best = [text for text, _ in ranked]
                row = dict(case, variant=name, elapsed_ms=elapsed, best=best[:5],
                           rank=outcome(case["answers"], best), raw_rank=outcome(case["answers"], best, False),
                           spellable=any(decoder.table.spells(case["kana"], answer) for answer in case["answers"]))
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                done.add((name, case["suite"], case["case_index"]))
                decoder.session = None
            if (n + 1) % 25 == 0 or n + 1 == len(cases):
                print(json.dumps(dict(cases_done=n + 1, cases_total=len(cases),
                                      elapsed_s=round(time.monotonic() - start), results=summarize(rows))), flush=True)
    (args.out / "summary.json").write_text(json.dumps(dict(completed=True, results=summarize(rows)), indent=2) + "\n")


if __name__ == "__main__":
    main()
