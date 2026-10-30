#!/usr/bin/env python3
"""Full-backend regression controls sharing one baseline and identical requests."""
import argparse
import json
from lab_tracking import add_arguments, track

from pathlib import Path
import time

from composition import ENCODING, decode as decode_composition
from dataset_registry import Registry
from regression import compare, requests, request_key
from replay import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=/path; first checkpoint is the baseline")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--rescorer", type=Path, required=True)
    parser.add_argument("--weight", action="append", default=[], help="name=LM fusion weight; default 1.5")
    parser.add_argument("--out", type=Path, required=True)
    add_arguments(parser)
    args = parser.parse_args()
    with track(args, "evaluate") as checkpoint:
        return execute(args, parser, checkpoint)


def execute(args, parser, checkpoint):
    if (args.out / "summary.json").exists() or (args.out / "results.jsonl").exists():
        raise ValueError("Evaluation output already exists; choose a new directory")
    models = dict(value.split("=",1) for value in args.model)
    if len(models) < 2 or len(models) != len(args.model):
        parser.error("at least two distinct model names required")
    weights = {name: 1.5 for name in models}
    for value in args.weight:
        name, weight = value.split("=", 1)
        if name not in models:
            parser.error(f"--weight names an unknown model: {name}")
        weights[name] = float(weight)
    registry, suites = Registry(), []
    for dataset in args.dataset:
        path, snapshot = registry.evaluation(dataset)
        cases = json.loads(path.read_text())
        suites.append((dataset,snapshot,cases,list(requests(cases))))
    args.out.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()

    import torch
    from decoder import Decoder, Units
    from decoding_model import readings
    from inference import ConditionalEvaluator
    from rescorer import Rescorer
    rescorer = Rescorer(str(args.rescorer),1.5)
    units = Units(rescorer.tokenizer,readings(args.database))
    decoders = {name:Decoder(ConditionalEvaluator(path),rescorer,units) for name,path in models.items()}
    for name, decoder in decoders.items():
        decoder.weight = weights[name]
    for decoder in decoders.values():
        decoder.decode("评测预热","nihk")
    cache, results = {}, []
    with (args.out / "results.jsonl").open("w") as stream:
        for dataset,snapshot,cases,prepared in suites:
            print(json.dumps(dict(dataset=dataset,cases=len(cases))),flush=True)
            for row,request in zip(cases,prepared,strict=True):
                identity = request_key(request)
                if identity not in cache:
                    predictions = {}
                    for name,decoder in decoders.items():
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        candidates = decode_composition(decoder, request["context"], request["input"],
                                                        request["committed"], aux=request["aux"])
                        torch.cuda.synchronize()
                        predictions[name] = dict(prediction=candidates[0][0] if candidates else "",
                                                 candidates=[text for text,score in candidates],
                                                 scores=[score for text,score in candidates],
                                                 milliseconds=1000*(time.perf_counter()-start))
                    cache[identity] = predictions
                result = dict(dataset=dataset,id=row["id"],gold=row["gold"],typed_input=row["input"],
                              request=request,models=cache[identity],
                              **{key:row[key] for key in ("document","parent","category","variant") if key in row})
                results.append(result)
                stream.write(json.dumps(result,ensure_ascii=False)+"\n")
                stream.flush()
                if len(results)%50 == 0:
                    print(json.dumps(dict(completed=len(results),unique_requests=len(cache),seconds=time.monotonic()-started)),flush=True)
    report = dict(models={name:dict(path=path,sha256=fingerprint(Path(path)),weight=weights[name]) for name,path in models.items()},
                  datasets={name:snapshot for name,snapshot,_,_ in suites},
                  database_sha256=fingerprint(args.database),
                  rescorer_files_sha256={file.name:fingerprint(file) for file in sorted(args.rescorer.iterdir())
                                        if file.suffix in (".json",".safetensors")},
                  profile=dict(backend="legacy",encoding=ENCODING,beam=16,rescorer_weight=(next(iter(weights.values())) if len(set(weights.values())) == 1 else weights),
                               context_characters=64,personalization=False),
                  code_sha256={name:fingerprint(Path(__file__).with_name(name)) for name in
                               ("evaluate-regression.py","regression.py","decoder.py","decoding_model.py",
                                "inference.py","conditional.py","composition.py","rescorer.py")},
                  unique_requests=len(cache),cases=len(results),seconds=time.monotonic()-started,
                  metrics=(compare(results,list(models)) if len(models)==2 else
                           {name:compare(results,[next(iter(models)),name]) for name in list(models)[1:]}),
                  interpretation="Archived regression counts, not estimated user traffic. "
                                 "Auxiliary keys retained; explicit or reconstructible committed keystrokes supplied. "
                                 "Predictions shared only for identical model-facing requests.")
    (args.out / "summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(cases=len(results),unique_requests=len(cache),
                          summary=str(args.out / "summary.json"))),flush=True)


if __name__ == "__main__":
    main()
