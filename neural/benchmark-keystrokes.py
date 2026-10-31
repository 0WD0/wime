#!/usr/bin/env python3
"""Per-keystroke legacy decode latency, typing each case key by key.

Each case keeps one decoder session across its keystrokes, as the editor
does; every prefix with a complete first syllable is decoded, although the
editor's debounce skips some. Times include CUDA synchronization on a GPU.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from composition import decode as decode_composition
from dataset_registry import Registry
from decoder import Decoder, Units
from inference import ConditionalEvaluator
from replay import fingerprint
from rescorer import Rescorer


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cases", type=int, default=100, help="first N cases by id")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--rescorer", type=Path, required=True)
    parser.add_argument("--rescorer-weight", type=float, default=1.0)
    parser.add_argument("--beam", type=int, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path, snapshot = Registry().evaluation(args.dataset)
    cases = sorted(json.loads(path.read_text()), key=lambda c: c["id"])[:args.cases]
    evaluator = ConditionalEvaluator(args.model, device=args.device)
    rescorer = Rescorer(str(args.rescorer), args.rescorer_weight, device=args.device)
    torch.set_num_threads(args.threads)
    sync = torch.cuda.synchronize if args.device.startswith("cuda") else lambda: None
    units = Units.from_database(rescorer.tokenizer, args.database)
    report = dict(dataset=snapshot, cases=len(cases), model_sha256=fingerprint(args.model),
                  rescorer_weight=args.rescorer_weight, device=args.device, threads=args.threads,
                  semantics=__doc__, beams={})
    for beam in args.beam:
        decoder = Decoder(evaluator, rescorer, units, beam=beam)
        decoder.decode("评测预热", "nihk")
        times = []
        for case in cases:
            decoder.session = None
            for k in range(2, len(case["input"]) + 1):
                sync()
                start = time.perf_counter()
                decode_composition(decoder, case["context"], case["input"][:k], window=evaluator.model.cfg.context)
                sync()
                times.append(1000 * (time.perf_counter() - start))
        report["beams"][beam] = dict(keystrokes=len(times), **{f"p{int(f * 100)}": percentile(times, f)
                                                               for f in (.5, .9, .95, .99)}, max=max(times))
        print(json.dumps({beam: report["beams"][beam]}), flush=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
