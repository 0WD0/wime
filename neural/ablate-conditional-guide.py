#!/usr/bin/env python3
"""Frozen paired evaluation of conditional search guidance, with unchanged LM ranking."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--suite", choices=("accuracy", "timing"), default="accuracy")
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    arm = next(a for a in plan["arms"] if a["label"] == args.arm)
    setting = next(m for m in plan["models"] if m["label"] == arm["model"])
    assert args.repeat in range(1 if args.suite == "accuracy" else plan["timing_repeats"])
    assert digest(__file__) == plan["runner_sha256"]
    for name, expected in plan["code_sha256"].items():
        assert digest(Path(plan["code"]) / name) == expected, name
    assert digest(plan["fixtures"]["path"]) == plan["fixtures"]["sha256"]
    for filename in plan["model_assets"][setting["label"]]:
        assert digest(filename) == plan["asset_sha256"][filename], filename
    if arm["guide"]:
        assert digest(plan["conditional"]) == plan["asset_sha256"][plan["conditional"]]
    rows = json.loads(Path(plan["fixtures"]["path"]).read_text())
    if args.suite == "timing":
        lookup = {row["id"]: row for row in rows}
        rows = [lookup[identity] for identity in plan["timing_ids"]]
    assert len(rows) == (plan["rows"] if args.suite == "accuracy" else len(plan["timing_ids"]))
    out = Path(plan["output"]) / f"{args.suite}-{args.arm}-{args.repeat}"
    out.mkdir(exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    sys.path[:0] = [plan["code"], plan["python_dependencies"]]
    import torch
    from decoding_model import readings
    from inference import ConditionalEvaluator
    from rescorer import Rescorer
    from text_decoder import TextDecoder

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    with Path(plan["gpu_lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps(dict(arm=args.arm, suite=args.suite, repeat=args.repeat,
                              status="gpu_lock_acquired")), flush=True)
        # The off arm does not load or call the conditional model.
        evaluator = ConditionalEvaluator(plan["conditional"], device="cuda") if arm["guide"] else None
        if setting["backend"] == "native":
            from native_rescorer import NativeRescorer
            rescorer = NativeRescorer(setting["path"], setting["weight"], setting["library"], setting["gguf"])
            backend = dict(lm_backend=rescorer.backend)
        else:
            rescorer = Rescorer(setting["path"], setting["weight"])
            backend = {}
        table = readings(plan["database"])
        unreachable = [(row["id"], i, char) for row in rows
                       for i, char in enumerate(row["gold"][0])
                       if row["input"][2*i:2*i+2] not in table.get(char, {})]
        assert not unreachable, unreachable
        parameters = dict(plan["decoder_parameters"], guide_weight=1.0 if arm["guide"] else 0.0)
        assert parameters["final_mode"] == "lm"
        decoder = TextDecoder(evaluator, rescorer, table, **parameters, **backend)
        for warm in plan["warmup"]:
            decoder.decode(warm["context"], warm["input"], anchor=plan["anchor"])
        # Fresh Python caches after unrelated warmup. Native root changes on
        # the first evaluated request; no evaluated prompt is in warmup.
        del decoder
        decoder = TextDecoder(evaluator, rescorer, table, **parameters, **backend)
        write(out / "run.json", dict(arm=arm, model=setting, suite=args.suite, repeat=args.repeat,
            plan_sha256=digest(args.plan), runner_sha256=digest(__file__),
            code_sha256=plan["code_sha256"], decoder_parameters=parameters,
            conditional_loaded=evaluator is not None, torch=torch.__version__,
            torch_threads=torch.get_num_threads(), cuda_build=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), model_loading_timed=False))
        totals = Counter()
        started = time.time()
        with (out / "results.jsonl").open("x") as stream:
            for index, row in enumerate(rows, 1):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                candidates = decoder.decode(row["context"], row["input"], anchor=plan["anchor"])
                torch.cuda.synchronize()
                elapsed = 1000 * (time.perf_counter() - begin)
                audit = dict(decoder.last_audit)
                assert audit["conditional_active"] == arm["guide"], row["id"]
                assert not audit["request_cache_hit"], row["id"]
                if not arm["guide"]:
                    assert audit["guide_texts"] == 0 and decoder._guide_scores is None
                assert len({text for text, _ in candidates}) == len(candidates)
                for text, score in candidates:
                    assert len(text) == len(row["input"]) // 2 and math.isfinite(score)
                    assert all(row["input"][2*i:2*i+2] in table.get(char, {}) for i, char in enumerate(text))
                rank = next((i for i, (text, _) in enumerate(candidates, 1) if text in row["gold"]), None)
                result = {key: row[key] for key in
                          ("id", "document", "parent", "category", "variant", "context", "input", "gold")}
                result.update(candidates=candidates, gold_rank=rank, top1=rank == 1,
                    top5=rank is not None and rank <= 5, oracle=rank is not None,
                    mrr=1/rank if rank else 0.0, elapsed_ms=elapsed, audit=audit,
                    components=decoder.last_components, invariants_passed=True)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                totals["top1"] += result["top1"]
                totals["oracle"] += result["oracle"]
                if index % 30 == 0 or index == len(rows):
                    print(json.dumps(dict(arm=args.arm, suite=args.suite, repeat=args.repeat,
                        completed=index, total=len(rows), **totals,
                        elapsed_s=round(time.time()-started, 1))), flush=True)
        write(out / "completion.json", dict(status="complete", rows=len(rows), totals=totals,
            elapsed_s=time.time()-started, results_sha256=digest(out / "results.jsonl")))
        if setting["backend"] == "native":
            rescorer.close()


if __name__ == "__main__":
    main()
