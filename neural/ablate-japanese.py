#!/usr/bin/env python3
"""Controlled Japanese backend ablation; experimental, never a default worker.

Batch mode scores one shared zenz candidate pool with/without Qwen. Serve
mode loads only the requested Japanese backend and speaks the normal worker
protocol, so the real Emacs scheduler can measure typing and settling.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import time
import unicodedata
from pathlib import Path

import torch

from japanese import Decoder, KanaModel, RERANK, Table, Zenz
from rescorer import Rescorer


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def katakana(text):
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def displayed(kana, ranked):
    # Same literal fallbacks and deduplication as the Japanese front end.
    return list(dict.fromkeys([t for t, _ in ranked] + [kana, katakana(kana)]))[:5]


def rank(answers, best):
    acceptable = {unicodedata.normalize("NFKC", text) for text in answers}
    return next((i + 1 for i, text in enumerate(best)
                 if unicodedata.normalize("NFKC", text) in acceptable), None)


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, 1000 * (time.perf_counter() - start)


def memory():
    return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


class GeneratedDecoder(Decoder):
    """Use generated candidates only, with the production reading/span checks.

Pending romaji does not speculate a next kana: leave the front end's literal
preedit visible until the syllable is complete. Report this difference in
typing results rather than presenting cheap skipped requests as model speed.
"""

    def inspect(self, context, kana, shift=None, explicit=False):
        constraints = tuple("c" if f == "1" else "k" for f in (shift or "0" * len(kana))) if explicit else None
        generated, gen_ms = timed(lambda: self.zenz.generate(context, kana))
        generated = list(dict.fromkeys(t for t in generated if t))
        accepted, check_ms = timed(lambda: [t for t in generated if self.table.spells(kana, t, constraints)])
        z, z_ms = timed(lambda: self.zenz.score(context, kana, accepted)) if accepted else ([], 0.0)
        q, q_ms = ([], 0.0)
        if accepted and self.rescorer is not None:
            q, q_ms = timed(lambda: self.lm_tails(self.lm_context(context), [()] * len(accepted),
                                                 [self.table.tokenizer(t)["input_ids"] for t in accepted]))
        return dict(generated=generated, accepted=accepted, z=z, q=q,
                    generate_ms=gen_ms, validation_ms=check_ms, z_score_ms=z_ms, q_score_ms=q_ms)

    @staticmethod
    def ranked(details, with_qwen):
        q = details["q"] if with_qwen else [0.0] * len(details["accepted"])
        return sorted(zip(details["accepted"], [z + RERANK * v for z, v in zip(details["z"], q)]),
                      key=lambda item: -item[1])

    def decode(self, context, kana, pending=(), shift=None, explicit=False, pending_shift=False):
        self.last = None
        if pending or not kana:
            return []
        self.last = self.inspect(context, kana, shift, explicit)
        return self.ranked(self.last, self.rescorer is not None)


def load_backend(args, mode):
    from transformers import AutoTokenizer
    rescorer = Rescorer(args.rescorer, 0.6) if mode != "zenz" else None
    tokenizer = rescorer.tokenizer if rescorer else AutoTokenizer.from_pretrained(args.rescorer)
    table, zenz = Table(args.japanese_table, tokenizer), Zenz(args.japanese_zenz)
    if mode == "full":
        return Decoder(table, rescorer, KanaModel(args.japanese_model), 0.6, zenz=zenz)
    return GeneratedDecoder(table, rescorer, None, 0.6, zenz=zenz)


def summary(rows):
    out = {}
    for suite in sorted({r["suite"] for r in rows}):
        out[suite] = {}
        for mode in sorted({r["variant"] for r in rows}):
            chosen = [r for r in rows if r["suite"] == suite and r["variant"] == mode]
            if chosen:
                out[suite][mode] = dict(total=len(chosen), top1=sum(r["rank"] == 1 for r in chosen),
                                       top5=sum(r["rank"] is not None for r in chosen),
                                       no_valid_generated=sum(not r["details"]["accepted"] for r in chosen))
    return out


def batch(args):
    cases = json.loads(args.cases.read_text())
    reference = json.loads((args.full_reference / "manifest.json").read_text())
    # Pin the exact current production reference; never silently reuse other models or tables.
    for path, expected in reference["sha256"].items():
        if digest(path) != expected:
            raise ValueError("full-reference input changed: " + path)
    fixed = reference["sha256"]
    for path in [args.cases, args.japanese_table, args.japanese_model,
                 args.japanese_zenz / "model.safetensors", args.japanese_zenz / "tokenizer.json",
                 *sorted(args.rescorer.glob("*.safetensors")), args.rescorer / "tokenizer.json"]:
        if str(path.resolve()) not in fixed or digest(path) != fixed[str(path.resolve())]:
            raise ValueError("ablation and full reference must share inputs: " + str(path))
    baseline = [json.loads(line) for line in (args.full_reference / "rows.jsonl").read_text().splitlines()]
    baseline = [r for r in baseline if r["variant"] == "current"]
    assert len(baseline) == len(cases) == len({(r["suite"], r["case_index"]) for r in cases})
    baseline = {(r["suite"], r["case_index"]): r for r in baseline}
    for case in cases:
        row = baseline[(case["suite"], case["case_index"])]
        assert all(row[k] == case[k] for k in ("kana", "context", "answers", "group"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol="Shared generated pool, fixed z + 0.7*q; NFKC full-string matching; literal UI fallbacks",
                    full_reference=str(args.full_reference), full_rows_sha256=digest(args.full_reference / "rows.jsonl"),
                    reference=reference, script_sha256=digest(__file__), cases=len(cases),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    timing="Batch components share resident models; standalone resources measured separately with --serve")
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Inputs changed; use a fresh output directory"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output = args.out / "rows.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    done = {(r["suite"], r["case_index"], r["variant"]) for r in rows}
    assert len(done) == len(rows)
    decoder = load_backend(args, "zenz-qwen")
    decoder.decode("", "にほんご")
    with output.open("a", buffering=1) as stream:
        for index, case in enumerate(cases):
            if all((case["suite"], case["case_index"], mode) in done for mode in ("zenz", "zenz-qwen")):
                continue
            details = decoder.inspect(case["context"], case["kana"])
            for mode in ("zenz", "zenz-qwen"):
                key = (case["suite"], case["case_index"], mode)
                if key in done:
                    continue
                best = displayed(case["kana"], decoder.ranked(details, mode == "zenz-qwen"))
                row = dict(case, variant=mode, best=best, rank=rank(case["answers"], best), details=details)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                done.add(key)
            if (index + 1) % 50 == 0 or index + 1 == len(cases):
                print(json.dumps(dict(cases_done=index + 1, results=summary(rows))), flush=True)
    assert len(rows) == 2 * len(cases)
    (args.out / "summary.json").write_text(json.dumps(dict(completed=True, results=summary(rows)), indent=2) + "\n")


def serve(args):
    spec = importlib.util.spec_from_file_location("gpu_worker", Path(__file__).with_name("gpu-worker.py"))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    decoder = load_backend(args, args.mode)
    decoder.decode("", "にほんご")
    decoder.session = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trace = args.trace.open("w", buffering=1) if args.trace else None
    if trace:
        trace.write(json.dumps(dict(event="ready", mode=args.mode, memory=memory())) + "\n")
    import sys
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("operation") != "japanese":
            raise ValueError("Ablation worker accepts Japanese requests only")
        response = worker.respond(None, request, japanese=decoder)
        if trace:
            trace.write(json.dumps(dict(event="request", id=request["id"], kana=request["kana"],
                                        pending=request.get("pending", []), explicit=request.get("explicit", False),
                                        candidate_count=len(response["candidates"]), elapsed_ms=response["elapsed_ms"],
                                        memory=memory()), ensure_ascii=False) + "\n")
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", help="unused Chinese model argument from the editor protocol")
    ap.add_argument("--rescorer", type=Path, required=True)
    ap.add_argument("--rescorer-weight", type=float, default=0.6, help="accepted for editor compatibility; experiment fixes 0.6/0.7")
    ap.add_argument("--japanese-table", type=Path, required=True)
    ap.add_argument("--japanese-model", type=Path, required=True)
    ap.add_argument("--japanese-zenz", type=Path, required=True)
    ap.add_argument("--mode", choices=["full", "zenz", "zenz-qwen"], default=os.environ.get("WIME_ABLATION_MODE", "zenz-qwen"))
    ap.add_argument("--serve", action="store_true", default=bool(os.environ.get("WIME_ABLATION_SERVE")))
    ap.add_argument("--trace", type=Path, default=os.environ.get("WIME_ABLATION_TRACE"))
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--full-reference", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.serve:
        serve(args)
    elif args.cases and args.full_reference and args.out:
        batch(args)
    else:
        ap.error("batch requires --cases, --full-reference and --out")


if __name__ == "__main__":
    main()
