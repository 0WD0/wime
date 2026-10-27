#!/usr/bin/env python3
"""Decode held-out kana (prepare-japanese.py's test.json/dev.json) as the worker does."""
import argparse
import collections
import json
import time
from pathlib import Path

from japanese import Decoder, KanaModel, Table, Zenz
from rescorer import Rescorer


def errors(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rescorer", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--kana-model", required=True)
    ap.add_argument("--weight", type=float, default=0.6)
    ap.add_argument("--zenz", help="convert-zenz.py directory: rerank finished readings")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--explicit", action="store_true", help="hold Shift as the cases' shift field says")
    args = ap.parse_args()
    cases = json.loads(args.cases.read_text())[:args.limit]
    rescorer = Rescorer(args.rescorer, args.weight)
    decoder = Decoder(Table(args.table, rescorer.tokenizer), rescorer, KanaModel(args.kana_model), args.weight,
                      zenz=Zenz(args.zenz) if args.zenz else None)
    stats, rows, times = collections.Counter(), [], []
    for case in cases:
        start = time.perf_counter()
        decoder.session = None  # every case is a fresh composition
        best = [t for t, _ in (decoder.decode("", case["input"], shift=case["shift"], explicit=True)
                               if args.explicit else decoder.decode("", case["input"]))]
        times.append(1000 * (time.perf_counter() - start))
        stats.update(total=1, correct=best[:1] == [case["gold"]], top5=case["gold"] in best[:5],
                     chars=len(case["gold"]), char_errors=errors(best[0] if best else "", case["gold"]))
        rows.append(dict(case, best=best[:5]))
    times.sort()
    summary = dict(stats, weight=args.weight, explicit=args.explicit, zenz=bool(args.zenz),
                   ms_p50=times[len(times) // 2], ms_p95=times[int(.95 * len(times))])
    print(json.dumps(summary))
    args.out.write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
