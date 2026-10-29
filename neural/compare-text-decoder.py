#!/usr/bin/env python3
"""Bounded development comparison and separate text-decoder invariants."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for filename, expected in plan["frozen_code_sha256"].items():
        assert digest(Path(plan["experimental_code"]) / filename) == expected
    sys.path.insert(0, plan["experimental_code"])
    import torch
    from decoder import readings
    from inference import ConditionalEvaluator
    from rescorer import Rescorer
    from text_decoder import TextDecoder

    settings = next(m for m in plan["models"] if m["label"] == args.model)
    output = args.plan.parent / args.model
    output.mkdir(exist_ok=True)
    if (output / "dev.jsonl").exists() or (output / "diagnostics.jsonl").exists():
        raise SystemExit("Refusing to overwrite results")
    with (Path.home() / ".local/share/wime/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evaluator = ConditionalEvaluator(plan["conditional"], device="cuda")
        lm = Rescorer(settings["path"], settings["weight"])
        table = readings(plan["database"])
        decoder = TextDecoder(evaluator, lm, table, **plan["decoder_parameters"])
        decoder.decode("", "nihk")
        summaries = {}
        for suite_name, suite in plan["suites"].items():
            assert digest(suite["path"]) == suite["sha256"]
            rows = json.loads(Path(suite["path"]).read_text())
            results = []
            with (output / (suite_name + ".jsonl")).open("x") as stream:
                for row in rows:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    candidates = decoder.decode(row["context"], row["input"],
                                                pins=row.get("pins", ()), aux=row.get("aux", ()))
                    torch.cuda.synchronize()
                    elapsed = 1000 * (time.perf_counter() - started)
                    n = (len(row["input"]) + 1) // 2
                    for text, score in candidates:
                        assert len(text) == n
                        for position, char in enumerate(text):
                            keys = row["input"][2*position:2*position+2]
                            compatible = [(c, groups) for c, groups in table.get(char, {}).items()
                                          if c == keys or len(keys) == 1 and c.startswith(keys)]
                            assert compatible
                            assert all(char == value for p, value in row.get("pins", ()) if p == position)
                            assert all(any(group.startswith(value) for _, gs in compatible for group in gs)
                                       for p, value in row.get("aux", ()) if p == position)
                    expectations = row.get("expectations", {})
                    if expectations.get("empty"):
                        assert not candidates, row["id"]
                    elif row.get("diagnostic_type") == "constraint_probe":
                        assert candidates, row["id"]
                    if "only_text" in expectations:
                        assert [t for t, _ in candidates] == [expectations["only_text"]], row["id"]
                    if "starts_with" in expectations:
                        assert all(t.startswith(expectations["starts_with"]) for t, _ in candidates)
                    components = decoder.last_components
                    fused = sorted(components, key=lambda c: c["fused"] + c["bonus"], reverse=True)
                    # Reference access follows all candidate generation/ranking.
                    gold = row.get("gold", [])
                    result = dict(id=row["id"], context=row["context"][-64:], input=row["input"],
                        gold=gold, candidates=candidates, components=components, audit=decoder.last_audit.copy(),
                        top1=bool(candidates and candidates[0][0] in gold),
                        fused_top1=bool(fused and fused[0]["text"] in gold),
                        fused_prediction=fused[0]["text"] if fused else "",
                        oracle=any(t in gold for t, _ in candidates), elapsed_ms=elapsed,
                        **{k: row[k] for k in ("document", "category", "variant", "diagnostic_type") if k in row})
                    if suite_name == "diagnostics" and row.get("diagnostic_type") == "ranking_probe":
                        previous = decoder.rewind_tokens
                        decoder.rewind_tokens = 0
                        try:
                            without = decoder.decode(row["context"], row["input"])
                            result["without_healing"] = dict(candidates=without, audit=decoder.last_audit.copy())
                        finally:
                            decoder.rewind_tokens = previous
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    results.append(result)
                    if len(results) % 10 == 0 or len(results) == len(rows):
                        print(json.dumps(dict(model=args.model, suite=suite_name, completed=len(results),
                            total=len(rows), top1=sum(r["top1"] for r in results),
                            fused_top1=sum(r["fused_top1"] for r in results),
                            oracle=sum(r["oracle"] for r in results))), flush=True)
            summaries[suite_name] = dict(rows=len(results), top1=sum(r["top1"] for r in results),
                fused_top1=sum(r["fused_top1"] for r in results), oracle=sum(r["oracle"] for r in results),
                empty=sum(not r["candidates"] for r in results),
                p50_ms=sorted(r["elapsed_ms"] for r in results)[len(results)//2],
                p95_ms=sorted(r["elapsed_ms"] for r in results)[int(.95*(len(results)-1))])
        (output / "summary.json").write_text(json.dumps(dict(model=settings, suites=summaries,
            plan_sha256=digest(args.plan), runner_sha256=digest(__file__)), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
