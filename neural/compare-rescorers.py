#!/usr/bin/env python3
"""Compare causal LMs through Wime's production Chinese constrained decoder.

Deterministic category-balanced development/test subsets are frozen before
inference. Select the fusion weight on development only. Each model runs in
its own process, using the same conditional checkpoint and word database.
Timings exclude Emacs, IPC and debounce; typing runs retain the real session.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
import transformers

from decoder import Decoder, Units, readings
from inference import ConditionalEvaluator
from rescorer import Rescorer


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def select(path, per_category, seed):
    groups = collections.defaultdict(list)
    for row in json.loads(path.read_text()):
        groups[row["category"]].append(row)
    result = []
    for category, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: hashlib.sha256(
            f"{seed}:{row['id']}".encode()).hexdigest())
        result.extend(ordered[:per_category])
    assert len({row["id"] for row in result}) == len(result)
    return result


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def summarize(rows):
    def counts(items):
        times = [r["elapsed_ms"] for r in items]
        return dict(total=len(items), top1=sum(r["top1"] for r in items),
                    top5=sum(r["top5"] for r in items), empty=sum(not r["candidates"] for r in items),
                    p50_ms=percentile(times, .5), p95_ms=percentile(times, .95))
    return dict(overall=counts(rows), categories={
        category: counts([r for r in rows if r["category"] == category])
        for category in sorted({r["category"] for r in rows})})


def decode(decoder, row, code):
    torch.cuda.synchronize()
    started = time.perf_counter()
    candidates = decoder.decode(row["context"][-64:], code, anchor=2)
    torch.cuda.synchronize()
    return [text for text, _ in candidates], 1000 * (time.perf_counter() - started)


def evaluate(decoder, rows, path):
    results = []
    with path.open("w") as stream:
        for i, row in enumerate(rows):
            decoder.session = None
            candidates, elapsed = decode(decoder, row, row["input"])
            result = dict(id=row["id"], category=row["category"], input=row["input"],
                          context=row["context"][-64:], gold=row["gold"], candidates=candidates,
                          top1=bool(candidates and candidates[0] in row["gold"]),
                          top5=any(text in row["gold"] for text in candidates[:5]), elapsed_ms=elapsed)
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if (i + 1) % 12 == 0 or i + 1 == len(rows):
                print(json.dumps(dict(phase=path.stem, completed=i + 1, total=len(rows),
                                      top1=sum(r["top1"] for r in results)), ensure_ascii=False), flush=True)
    return summarize(results)


def typing(decoder, rows, path):
    results, selected = [], set()
    for row in rows:
        if row["category"] in selected:
            continue
        selected.add(row["category"])
        decoder.session, events = None, []
        for length in range(2, len(row["input"]) + 1):
            candidates, elapsed = decode(decoder, row, row["input"][:length])
            events.append(dict(keys=length, text=candidates[0] if candidates else None, elapsed_ms=elapsed))
        results.append(dict(id=row["id"], category=row["category"], gold=row["gold"], events=events,
                            top1=bool(events and events[-1]["text"] in row["gold"])))
    save(path, results)
    times = [event["elapsed_ms"] for row in results for event in row["events"]]
    return dict(compositions=len(results), events=len(times), top1=sum(r["top1"] for r in results),
                p50_ms=percentile(times, .5), p95_ms=percentile(times, .95))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "database", "dev", "test", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rescorer", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--dev-per-category", type=int, default=8)
    parser.add_argument("--test-per-category", type=int, default=32)
    parser.add_argument("--seed", default="wime-minicpm-20261028")
    parser.add_argument("--weights", type=float, nargs="+", default=[.75, 1.5, 2.25])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dev = select(args.dev, args.dev_per_category, args.seed)
    test = select(args.test, args.test_per_category, args.seed)
    assert not {r["document"] for r in dev} & {r["document"] for r in test}
    sources = {str(path.resolve()): digest(path) for path in (args.model, args.database, args.dev, args.test)}
    plan = dict(seed=args.seed, dev_ids=[r["id"] for r in dev], test_ids=[r["id"] for r in test],
                weights=args.weights, context_characters=64, context_anchor=2, sources=sources,
                selection="Best development top1; ties prefer the weight nearest production 1.5, then lower weight.",
                timing="Warm backend decode only; no Emacs/IPC/40ms debounce; fresh session per full composition.")
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing experiment plan differs; use a new output directory")
    else:
        save(plan_path, plan)
    if args.prepare_only:
        print(json.dumps(dict(dev=len(dev), test=len(test), plan=str(plan_path))))
        return
    if not args.rescorer or not args.label or Path(args.label).name != args.label:
        parser.error("--rescorer and a simple --label are required")
    output = args.output / args.label
    output.mkdir(exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("Completed model output already exists")
    lock_path = Path.home() / ".local/share/wime/gpu.lock"
    with lock_path.open("a") as lock:
        print("Waiting for the shared Wime GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.perf_counter()
        evaluator = ConditionalEvaluator(args.model, device="cuda")
        rescorer = Rescorer(args.rescorer, 1.5)
        decoder = Decoder(evaluator, rescorer, Units(rescorer.tokenizer, readings(args.database)))
        decode(decoder, {"context": ""}, "nihk")
        startup = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        development = {}
        for weight in args.weights:
            decoder.weight = weight
            development[str(weight)] = evaluate(decoder, dev, output / f"dev-{weight}.jsonl")
        chosen = max(args.weights, key=lambda w: (development[str(w)]["overall"]["top1"], -abs(w - 1.5), -w))
        save(output / "selection.json", dict(weight=chosen, development=development))
        decoder.weight = chosen
        evaluation = evaluate(decoder, test, output / "test.jsonl")
        typed = typing(decoder, test, output / "typing.json")
        summary = dict(label=args.label, selected_weight=chosen, development=development, test=evaluation,
                       typing=typed, startup_seconds=startup,
                       gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       gpu_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                       environment=dict(python=platform.python_version(), torch=torch.__version__,
                                        transformers=transformers.__version__, gpu=torch.cuda.get_device_name()),
                       parameters=sum(p.numel() for p in rescorer.model.parameters()),
                       model_class=type(rescorer.model).__name__,
                       rescorer=str(args.rescorer.resolve()),
                       files={str(p.resolve()): digest(p) for p in sorted(args.rescorer.iterdir())
                              if p.suffix in (".json", ".safetensors")},
                       code={p.name: digest(p) for p in [Path(__file__), *[
                           Path(__file__).with_name(name) for name in ("rescorer.py", "decoder.py", "inference.py")]]})
        save(output / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("label", "selected_weight", "test", "typing", "startup_seconds",
                                                    "gpu_peak_allocated_mib", "gpu_peak_reserved_mib")}), flush=True)


if __name__ == "__main__":
    main()
