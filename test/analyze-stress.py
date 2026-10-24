#!/usr/bin/env python3
"""Summarize failure probes and independently check keyboard/aux constraints."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
import sqlite3


def summarize(rows):
    counts = Counter(total=len(rows))
    failure_kinds = Counter()
    for row in rows:
        good = row["constrained"] in row["gold"]
        baseline = row["beam"] in row["gold"]
        counts.update(baseline_correct=baseline, pipeline_correct=good,
                      rescue=good and not baseline, regression=baseline and not good,
                      neural_evaluated=row["calls"] > 0)
        if not good:
            kind = ("grammar_only" if row["calls"] == 0 else
                    "overrode_correct_baseline" if baseline else
                    "gold_in_pool_but_ranked_wrong" if row["gold_in_beam"] else
                    "search_missed_reachable_gold" if row["oracle_reachable"] else
                    "gold_unreachable")
            failure_kinds[kind] += 1
    return dict(counts, failures=dict(failure_kinds))


def check_constraints(rows, database):
    db = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)

    @lru_cache(maxsize=None)
    def entries(code):
        return db.execute("SELECT word, aux FROM entry WHERE code=? ORDER BY weight DESC LIMIT 64",
                          (code,)).fetchall()

    errors = []
    for row in rows:
        units = re.findall(r"([a-z;]{2})([A-Z]*)", row["input"])
        if "".join(a + b for a, b in units) != row["input"]:
            raise ValueError("audit requires complete typed input")
        text = row["constrained"]
        # Independent DP, using literal codes and auxiliary groups, not the
        # evaluator's gold-constrained oracle or its neural candidate pool.
        paths = defaultdict(set, {0: {0}})
        for start in range(len(units)):
            for end in range(start + 1, min(start + 6, len(units)) + 1):
                if not paths[start]:
                    continue
                code = "".join(c for c, _ in units[start:end])
                for word, aux in entries(code):
                    groups = aux.split("/")
                    if len(groups) != end - start or any(
                        not any(group.startswith(typed.lower()) for group in entry.split(","))
                        for (_, typed), entry in zip(units[start:end], groups)
                    ):
                        continue
                    for offset in paths[start]:
                        if text.startswith(word, offset):
                            paths[end].add(offset + len(word))
        coverage = [end for end, offsets in paths.items() if len(text) in offsets]
        if not coverage or (row["calls"] > 0 and len(units) not in coverage):
            errors.append(dict(id=row["id"], input=row["input"], result=text, coverage=coverage))
    db.close()
    return dict(checked=len(rows), violations=errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    fixed, probes, cases = [json.loads(path.read_text()) for path in (args.fixed, args.probes, args.cases)]
    by_id = {r["id"]: r for r in probes}
    equivalent = [(row, by_id[row["equivalent_to"]]) for row in cases if row.get("equivalent_to")]
    changed = [row["id"] for row, control in equivalent
               if any(by_id[row["id"]][k] != control[k] for k in ("beam", "constrained"))]
    segmentation = [r for r in probes if r.get("variant") in ("commit_suffix", "commit_span")
                    and by_id[r["parent"]]["constrained"] in by_id[r["parent"]]["gold"]]
    segment_failures = [r for r in segmentation if r["constrained"] not in r["gold"]]
    aux = [r for r in probes if r.get("variant") == "auxiliary"]
    aux_regressions = [r for r in aux if by_id[r["parent"]]["constrained"] in by_id[r["parent"]]["gold"]
                       and r["constrained"] not in r["gold"]]
    report = dict(fixed=summarize(fixed), probes=summarize(probes),
                  fixed_categories={c: summarize([r for r in fixed if r["category"] == c])
                                    for c in sorted({r["category"] for r in fixed})},
                  probe_categories={c: summarize([r for r in probes if r["category"] == c])
                                    for c in sorted({r["category"] for r in probes})},
                  variants={v: summarize([r for r in probes if r["variant"] == v])
                            for v in sorted({r["variant"] for r in probes})},
                  outside_window=dict(pairs=len(equivalent), changed=changed),
                  segmentation=dict(variants_of_correct_whole=len(segmentation),
                                    failed=len(segment_failures),
                                    model_failures=sum(r["calls"] > 0 for r in segment_failures),
                                    affected_parents=len({r["parent"] for r in segment_failures})),
                  auxiliary=dict(cases=len(aux), regressed_from_correct_whole=len(aux_regressions),
                                 ids=[r["id"] for r in aux_regressions]),
                  constraints=check_constraints(fixed + probes, args.database))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("fixed_categories", "probe_categories", "variants")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
