#!/usr/bin/env python3
"""Evaluate paired sentence, meaningful-prefix and correctly committed continuation states."""
import argparse
import json
from pathlib import Path
import time

from dataset_registry import Registry
from replay import fingerprint
from trajectories import states, validate
from composition import ENCODING, decode as decode_composition
from trajectory_metrics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("conditional", "legacy", "qwen-hf"), required=True)
    parser.add_argument("--rescorer")
    parser.add_argument("--rescorer-weight", type=float, default=1.5)
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.backend != "conditional" and not args.rescorer:
        parser.error("a full backend requires --rescorer")
    registry = Registry()
    path, snapshot = registry.evaluation(args.dataset)
    cases = json.loads(path.read_text())
    for row in cases:
        validate(row)
    args.out.mkdir(parents=True, exist_ok=False)

    import torch
    from conditional import load_model
    from conditional_decode import ConditionalDecoder
    from decoding_model import readings

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.backend == "conditional":
        model, vocab, metadata = load_model(args.model)
        decoder = ConditionalDecoder.from_database(model.to(args.device), vocab, args.database, args.beam)
    else:
        from inference import ConditionalEvaluator
        from backends import make_decoder
        from rescorer import Rescorer
        evaluator = ConditionalEvaluator(args.model, device=args.device)
        metadata, vocab = evaluator.metadata, evaluator.vocab
        rescorer = Rescorer(args.rescorer, args.rescorer_weight, device=args.device)
        decoder = make_decoder(args, evaluator, rescorer)
        decoder.beam = args.beam
    legal = readings(args.database)

    def predict(state):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        error, candidates = None, []
        try:
            if args.backend == "conditional":
                candidates = decoder.decode(state["context"], state["input"], state.get("committed"))
            else:
                candidates = decode_composition(decoder, state["context"], state["input"], state.get("committed"), window=evaluator.model.cfg.context)
        except ValueError as failure:
            error = str(failure)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        output = dict(prediction=candidates[0][0] if candidates else "",
                      candidates=[text for text, score in candidates[:5]],
                      ranked_candidates=[dict(text=text, score=score) for text, score in candidates],
                      committed=state.get("committed"),
                      milliseconds=1000 * (time.perf_counter() - start),
                      error=error or ("no_candidates" if not candidates else None))
        if args.backend == "legacy" and candidates:
            # Preserve the actual search paths: retokenizing their surface text
            # later can change the LM score and misattribute a ranking error.
            output["score_components"] = [dict(text=h.text, token_ids=list(h.ids),
                                               conditional=h.m, language_model=h.q)
                                          for h in decoder.session.kept[-1]]
        if hasattr(decoder, "last_audit"):
            output["audit"] = decoder.last_audit.copy()
        return output

    # Initialization latency is excluded; all measured calls still use real input.
    predict(dict(context="评测预热", input="nihk"))
    results = []
    started = time.monotonic()
    with (args.out / "results.jsonl").open("w") as stream:
        for row in cases:
            full, pairs = states(row)
            # Prefix is evaluated before the extra keystrokes are supplied.
            prefixes = [dict(end=pair["end"], reason=pair["reason"], prefix=predict(pair["prefix"]))
                        for pair in pairs]
            full_result = predict(full)
            for result, pair in zip(prefixes, pairs):
                result["continuation"] = predict(pair["continuation"])
            text, code = row["gold"][0], row["input"]
            result = dict(id=row["id"], document=row["document"], group=row["group"], source=row["source"],
                          context=row["context"], input=code, gold=row["gold"], full=full_result, prefixes=prefixes,
                          gold_key_reachable=all(code[2*i:2*i+2] in legal.get(char, {}) for i,char in enumerate(text)),
                          gold_in_conditional_vocabulary=all(char in vocab for char in text))
            for field in ("test_partition", "unit_kind"):
                if field in row:
                    result[field] = row[field]
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if len(results) % 10 == 0:
                print(json.dumps(dict(completed=len(results), total=len(cases), seconds=time.monotonic()-started)), flush=True)
    report = dict(encoding=ENCODING, dataset=snapshot, model=str(args.model.resolve()), model_sha256=fingerprint(args.model),
                  database_sha256=fingerprint(args.database), cases_sha256=fingerprint(path),
                  backend=args.backend, rescorer=args.rescorer, rescorer_weight=args.rescorer_weight,
                  rescorer_files_sha256={file.name:fingerprint(file) for file in sorted(Path(args.rescorer).iterdir())
                                        if file.suffix in (".json", ".safetensors")} if args.rescorer else None,
                  beam=args.beam,
                  training_steps=metadata.get("training_steps"), seconds=time.monotonic()-started,
                  precision="conditional FP32/TF32 disabled; Qwen uses existing BF16 runtime",
                  semantics="Fresh decode at reviewed boundaries; committed prefix and its actual keys supplied only after commit. "
                            "No auxiliary codes, learned user preferences or within-syllable states.",
                  code_sha256={file:fingerprint(Path(__file__).with_name(file)) for file in
                               ("evaluate-trajectories.py", "trajectories.py", "trajectory_metrics.py",
                                "composition.py", "conditional_decode.py", "conditional.py", "inference.py", "decoder.py", "text_decoder.py",
                                "decoding_model.py", "rescorer.py", "backends.py")},
                  metrics=summarize(results))
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(out=str(args.out), metrics=report["metrics"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
