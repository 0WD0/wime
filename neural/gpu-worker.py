#!/usr/bin/env -S python3 -B
"""Persistent GPU worker for wime's JSON-lines protocol (wime-neural-program).

It loads a syllable transducer and a general language model for Chinese
beam decoding. Stdout is exclusively the protocol; typed text is omitted
from errors. Optional Japanese models share the worker and language model.
"""
import argparse
import json
import math
import sys
import time

from composition import CODE_KEYS, ENCODING, validate_committed, decode as decode_composition
from fusion import add_fusion_arguments, make_rescorer, make_decoder
from transducer import Evaluator

LETTERS = set("abcdefghijklmnopqrstuvwxyz")
KEYS = set(CODE_KEYS)


def validate(request):
    """Reject malformed envelopes before dispatch."""
    if not isinstance(request, dict):
        raise ValueError("invalid request")
    for key in ("id", "syllables"):
        if type(request.get(key)) is not int or not 0 <= request[key] < 2**64:
            raise ValueError("invalid request")
    if (not isinstance(request.get("context"), str)
            or request.get("operation") not in ("decode", "japanese")):
        raise ValueError("invalid request")
    if request.get("encoding", ENCODING) != ENCODING:
        raise ValueError("obsolete composition encoding")
    validate_committed(request.get("committed"))
    if request.get("operation") == "decode":
        weight = request.get("memory_weight", 0)
        if (not isinstance(request.get("input"), str)
                or any(c not in KEYS for c in request["input"])
                or not isinstance(request.get("pins", []), list)
                or not isinstance(request.get("learned", []), list)
                or not isinstance(request.get("aux", []), list)
                or not isinstance(request.get("literals", []), list)
                or not isinstance(request.get("tones", []), list)
                or type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0):
            raise ValueError("invalid request")
        for pair in (request.get("pins", []) + request.get("learned", []) + request.get("aux", [])
                     + request.get("literals", [])):
            if (not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not int
                    or not 0 <= pair[0] < 256 or not isinstance(pair[1], str) or not pair[1]):
                raise ValueError("invalid request")
        for pair in request.get("tones", []):
            if (not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not int or not 0 <= pair[0] < 256
                    or type(pair[1]) is not int or not 1 <= pair[1] <= 5):
                raise ValueError("invalid request")
        if (any(len(text) != 1 for _, text in request.get("pins", []))
                or any(position < 1 or any(0xE000 <= ord(c) <= 0xF8FF for c in text)
                       for position, text in request.get("literals", []))
                or any(set(keys) - LETTERS for _, keys in request.get("aux", []))):
            raise ValueError("invalid request")
    if request.get("operation") == "japanese":
        pending = request.get("pending", [])
        shift = request.get("shift", "")
        if (not isinstance(request.get("kana"), str) or not isinstance(pending, list)
                or not all(isinstance(p, str) and len(p) == 1 for p in pending)
                or not isinstance(shift, str) or set(shift) - set("01")
                or (shift and len(shift) != len(request["kana"]))
                or type(request.get("explicit", False)) is not bool
                or type(request.get("pending_shift", False)) is not bool):
            raise ValueError("invalid request")


def respond(evaluator, request, decoder=None, japanese=None):
    validate(request)
    start = time.perf_counter()
    result = dict(id=request["id"], elapsed_ms=0.0)
    if request.get("operation") == "japanese":
        result["candidates"] = [] if japanese is None else [
            [text, score] for text, score in japanese.decode(
                request["context"], request["kana"], request.get("pending", []), request.get("shift"),
                request.get("explicit", False), request.get("pending_shift", False))]
        result["elapsed_ms"] = 1000 * (time.perf_counter() - start)
        return result
    if request.get("operation") == "decode":
        # Without a decoder the editor keeps its lattice candidates.
        result["candidates"] = [] if decoder is None else [
            [text, score] for text, score in decode_composition(decoder,
                request["context"], request.get("input", ""), request.get("committed"),
                window=evaluator.model.cfg.context,
                pins=request.get("pins", []), learned=request.get("learned", []),
                weight=request.get("memory_weight", 0), aux=request.get("aux", []),
                literals=request.get("literals", []), tones=request.get("tones", []))]
        result["elapsed_ms"] = 1000 * (time.perf_counter() - start)
        return result
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    add_fusion_arguments(parser)
    parser.add_argument("--database", help="wime word database, whose readings enable decode")
    parser.add_argument("--beam", type=int, default=32, help="decoder beam width")
    parser.add_argument("--japanese-model", help="kana-conditioned model (train-japanese.py)")
    parser.add_argument("--japanese-table", help="reading table for the rescorer (build-japanese-table.py)")
    parser.add_argument("--japanese-zenz", help="zenz directory (convert-zenz.py) for the final Japanese ranking")
    args = parser.parse_args(argv)
    if not args.rescorer or not args.database:
        parser.error("Chinese decoding requires --rescorer and --database")
    return args


def main():
    args = parse_args()
    evaluator = Evaluator(args.model, args.device)
    rescorer = make_rescorer(args)
    decoder = make_decoder(args, evaluator, rescorer)
    decoder.beam = args.beam
    # Pay model and device initialization before the first interactive request.
    japanese = None
    if rescorer and args.japanese_model and args.japanese_table:
        import japanese as ja
        japanese = ja.Decoder(ja.Table(args.japanese_table, rescorer.tokenizer), rescorer,
                              ja.KanaModel(args.japanese_model, args.device),
                              zenz=ja.Zenz(args.japanese_zenz, args.device) if args.japanese_zenz else None)
        japanese.decode("", "にほんご")
    decoder.decode("", "nihk")
    for line in sys.stdin:
        response = respond(evaluator, json.loads(line), decoder, japanese)
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"wime worker failed ({type(error).__name__})", file=sys.stderr)
        sys.exit(1)
