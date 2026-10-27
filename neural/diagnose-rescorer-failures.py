#!/usr/bin/env python3
"""Explain frozen comparison failures without tuning or changing test labels.

Replay each failed row, then pin its reference characters through the same
decoder. Record the actual search token path and its conditional/LM scores;
canonical tokenizer teacher forcing is not equivalent to this path. For a
missing reference, also try a four-times-wider search as a diagnostic only.
"""
import argparse
import collections
import fcntl
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    settings = next(m for m in plan["models"] if m["label"] == args.model)
    suite = plan["suites"]["expanded"]
    assert digest(suite["path"]) == suite["sha256"]
    rows = json.loads(Path(suite["path"]).read_text())
    prior_path = args.plan.parent / args.model / "expanded.jsonl"
    prior = [json.loads(line) for line in prior_path.read_text().splitlines()]
    assert len(rows) == len(prior)
    for row, result in zip(rows, prior):
        assert all(row[k] == result[k] for k in ("id", "input", "gold", "document"))
        assert result["context"] == row["context"][-64:]
    failures = [(r, p) for r, p in zip(rows, prior) if not p["top1"]]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / (args.model + ".jsonl")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    sys.path.insert(0, plan["code"])
    import decoder as search
    from inference import ConditionalEvaluator
    from rescorer import Rescorer

    with (Path.home() / ".local/share/wime/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evaluator = ConditionalEvaluator(plan["conditional"], device="cuda")
        rescorer = Rescorer(settings["path"], settings["weight"])
        units = search.Units(rescorer.tokenizer, search.readings(plan["database"]))
        decoder = search.Decoder(evaluator, rescorer, units)
        original_rescored = search.RESCORED

        def snapshot(row, pins=(), wide=False, context=None):
            decoder.session = None
            decoder.beam = search.BEAM * (4 if wide else 1)
            decoder.per_hypothesis = search.PER_HYPOTHESIS * (4 if wide else 1)
            search.RESCORED = original_rescored * (4 if wide else 1)
            try:
                decoder.decode(row["context"][-64:] if context is None else context,
                               row["input"], anchor=2, pins=pins)
                session = decoder.session
                final = session.kept[-1]
                n = len(row["input"]) // 2
                def record(h):
                    return dict(text=h.text, tokens=list(h.ids),
                                conditional=h.m / n, lm=h.q / n,
                                fused=session.rank(h) / n)
                return dict(candidates=[record(h) for h in final],
                            conditional_active=session.prompt(row["input"], n) is not None,
                            reference_prefixes=[
                                [h.text for h in frontier
                                 if any(g.startswith(h.text) for g in row["gold"])]
                                for frontier in session.kept])
            finally:
                search.RESCORED = original_rescored

        counts = collections.Counter()
        with output.open("x") as stream:
            for index, (row, previous) in enumerate(failures):
                baseline = snapshot(row)
                assert [h["text"] for h in baseline["candidates"]] == previous["candidates"], row["id"]
                winner = baseline["candidates"][0]
                retained = [h for h in baseline["candidates"] if h["text"] in row["gold"]]
                pinned = [snapshot(row, pins=tuple(enumerate(g))) for g in row["gold"]]
                forced = [h for run in pinned for h in run["candidates"] if h["text"] in row["gold"]]
                best = max(retained + forced, key=lambda h: h["fused"], default=None)
                wide = snapshot(row, wide=True) if not retained else None
                delta = ({key: best[key] - winner[key] for key in ("conditional", "lm", "fused")}
                         if best else None)
                if best and delta["fused"] > 0.001:
                    classification = "better_reference_path_pruned"
                elif retained:
                    classification = "reference_retained_but_lower_scored"
                elif best:
                    classification = "forced_reference_lower_than_winner"
                else:
                    classification = "reference_not_recovered_even_with_pins"
                counts[classification] += 1
                result = dict(id=row["id"], category=row["category"], variant=row["variant"],
                              context=row["context"][-64:], input=row["input"], gold=row["gold"],
                              original_rank=min([previous["candidates"].index(g) + 1
                                                 for g in row["gold"] if g in previous["candidates"]]
                                                or [0]),
                              baseline=baseline, pinned=pinned, best_reference=best,
                              reference_minus_winner=delta, classification=classification,
                              wider_search=wide)
                # All single-character failures: isolate the context sensitivity
                # of both models together. This is not a quality metric.
                if row["variant"] == "single":
                    result["empty_context"] = snapshot(row, context="")
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                if (index + 1) % 10 == 0 or index + 1 == len(failures):
                    print(json.dumps(dict(model=args.model, completed=index + 1,
                                          total=len(failures), counts=dict(counts))), flush=True)
        summary = dict(model=settings, rows=len(rows), failures=len(failures),
                       replay="All failed rows reproduce every candidate in its original order",
                       classifications=dict(counts), plan_sha256=digest(args.plan),
                       prior_sha256=digest(prior_path), script_sha256=digest(__file__),
                       decoder_sha256=digest(Path(plan["code"]) / "decoder.py"),
                       search=dict(beam=search.BEAM, per_hypothesis=search.PER_HYPOTHESIS,
                                   rescored=original_rescored, diagnostic_wide_multiplier=4),
                       limitations=[
                           "Pinned search supplies the reference; it is diagnostic, not test accuracy.",
                           "Pins can still hit rare-character gates and prefix deduplication.",
                           "Lower forced scores do not prove globally optimal search.",
                           "Wide search changes pruning only, not model weights or labels.",
                           "Scores use the actual decoder token path and its BF16 memo rows.",
                           "Context removal affects both the conditional model and general LM."])
        (args.output / (args.model + "-summary.json")).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
