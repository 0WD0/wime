#!/usr/bin/env -S python3 -B
"""Persistent CUDA worker for wime's JSON-lines protocol (wime-neural-program).

It loads a syllable transducer (transducer.py). Stdout is exclusively the
protocol; no typed text is included in errors. The verify operation scores
lattice drafts and proposes prefixes for the lattice to search. An optional
--rescorer adds a weighted general-LM score to every verified candidate,
including the seed pool, and interleaves its token proposals with the
model's per-character proposals. With --database as well, the decode
operation replaces the dictionary lattice with the fused decoder.py. With
--japanese-model and --japanese-table as well, the japanese operation
decodes typed kana, and --japanese-zenz reranks it: see japanese.py.
"""
import argparse
import json
import math
import sys
import time

from composition import ENCODING, validate_committed, decode as decode_composition
from rescorer import interleave
from fusion import add_fusion_arguments, make_rescorer, make_decoder
from transducer import Evaluator

LETTERS = set("abcdefghijklmnopqrstuvwxyz")
KEYS = LETTERS | {";"}


def effective_prior_weight(scores, syllables, weight, confidence=1.0):
    """Use the model alone when its full-candidate probability is high enough.

    Select once from the initial complete pool, then keep it fixed for all
    verification rounds. One disables the gate.
    """
    if confidence < 1 and max(scores) * syllables >= math.log(confidence):
        return 0.0
    return weight


def validate(request):
    """Reject malformed envelopes before dispatch."""
    if not isinstance(request, dict):
        raise ValueError("invalid request")
    for key in ("id", "syllables"):
        if type(request.get(key)) is not int or not 0 <= request[key] < 2**64:
            raise ValueError("invalid request")
    if not isinstance(request.get("context"), str) or not isinstance(request.get("candidates"), list):
        raise ValueError("invalid request")
    for candidate in request["candidates"]:
        if (not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str)
                or type(candidate.get("end")) is not int or not 0 <= candidate["end"] < 2**64
                or type(candidate.get("dictionary")) is not bool):
            raise ValueError("invalid request")
        bonus = candidate.get("bonus", 0)
        if type(bonus) not in (int, float) or not math.isfinite(bonus):
            raise ValueError("invalid request")
    for key in ("operation", "input", "draft"):
        if not isinstance(request.get(key, ""), str):
            raise ValueError("invalid request")
    if type(request.get("seed_candidates", False)) is not bool:
        raise ValueError("invalid request")
    confidence = request.get("prior_confidence", 1.0)
    if type(confidence) not in (int, float) or not 0 < confidence <= 1:
        raise ValueError("invalid request")
    if request.get("encoding", ENCODING) != ENCODING or any(
            key in request for key in ("context_anchor", "context_reading")):
        raise ValueError("obsolete composition encoding")
    validate_committed(request.get("committed"))
    if request.get("operation") == "decode":
        weight = request.get("memory_weight", 0)
        if (any(c not in KEYS for c in request.get("input", ""))
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
    if request.get("seed_candidates"):
        if (type(request.get("prior_weight")) not in (int, float)
                or not math.isfinite(request["prior_weight"]) or request["prior_weight"] < 0):
            raise ValueError("invalid request")
        for candidate in request["candidates"]:
            if (type(candidate.get("score")) not in (int, float)
                    or not math.isfinite(candidate["score"])):
                raise ValueError("invalid request")


def respond(evaluator, request, rescorer=None, decoder=None, japanese=None):
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
    if request.get("operation") != "verify":
        return result
    result["prefixes"] = []
    code, draft, syllables = request.get("input", ""), request.get("draft", ""), request["syllables"]
    if not syllables or len(draft) != syllables or len(code) != 2 * syllables:
        return result
    candidates = [(i, c) for i, c in enumerate(request["candidates"])
                  if c["end"] == syllables and len(c["text"]) == syllables]
    seeding = request.get("seed_candidates", False) and bool(candidates)
    texts = [c["text"] for _, c in candidates] if seeding else [draft]
    try:
        scores, distributions = evaluator.evaluate(request["context"], code, texts,
                                                     committed=request.get("committed"))
    except ValueError:
        return result  # Invalid keys or an overlong composition retain traditional candidates.
    prior_weight = (effective_prior_weight(scores, syllables, request["prior_weight"],
                                           request.get("prior_confidence", 1.0)) if seeding else 0.0)
    if rescorer:
        # The confidence gate above stays a model-probability rule.
        scores = rescorer.combine(request["context"] + (request.get("committed") or {}).get("text", ""), texts, scores)
    # The editor's learned-reading bonus joins the choice but never the scores
    # returned, to which the editor adds it again.
    selected = (max(range(len(candidates)), key=lambda j: scores[j] +
                    prior_weight * candidates[j][1]["score"] / syllables +
                    candidates[j][1].get("bonus", 0))
                if seeding else 0)
    if seeding:
        # Return an index into the original, aux-filtered dictionary pool. The
        # editor never accepts free-form model text as a seed candidate.
        result["verified_index"] = candidates[selected][0]
        result["effective_prior_weight"] = prior_weight
        # Model scores for the whole pool let the editor estimate confidence.
        result["candidate_scores"] = [[i, score] for (i, _), score in zip(candidates, scores)]
    prefixes = evaluator.proposals(texts[selected], distributions[selected], 0.5)
    if rescorer:
        prefixes = interleave(rescorer.proposals(request["context"] + (request.get("committed") or {}).get("text", ""), texts[selected]), prefixes)
    result.update(score=scores[selected], prefixes=prefixes,
                  evaluated=len(texts),
                  elapsed_ms=1000 * (time.perf_counter() - start))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    add_fusion_arguments(parser)
    parser.add_argument("--database", help="wime word database, whose readings enable decode")
    parser.add_argument("--beam", type=int, default=16, help="decoder beam width")
    parser.add_argument("--japanese-model", help="kana-conditioned model (train-japanese.py)")
    parser.add_argument("--japanese-table", help="reading table for the rescorer (build-japanese-table.py)")
    parser.add_argument("--japanese-zenz", help="zenz directory (convert-zenz.py) for the final Japanese ranking")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    evaluator = Evaluator(args.model)
    rescorer = make_rescorer(args)
    decoder = make_decoder(args, evaluator, rescorer)
    if decoder:
        decoder.beam = args.beam
    # Pay CUDA/kernel initialization once, before the first interactive request;
    # the decoder's warm-up below covers its model too.
    if decoder is None:
        evaluator.evaluate("", "aaaa", ["中文"])
    if rescorer:
        rescorer.score("", ["中文"])
    japanese = None
    if rescorer and args.japanese_model and args.japanese_table:
        import japanese as ja
        japanese = ja.Decoder(ja.Table(args.japanese_table, rescorer.tokenizer), rescorer,
                              ja.KanaModel(args.japanese_model),
                              zenz=ja.Zenz(args.japanese_zenz) if args.japanese_zenz else None)
        japanese.decode("", "にほんご")
    if decoder:
        decoder.decode("", "nihk")
    for line in sys.stdin:
        response = respond(evaluator, json.loads(line), rescorer, decoder, japanese)
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"wime CUDA worker failed ({type(error).__name__})", file=sys.stderr)
        sys.exit(1)
