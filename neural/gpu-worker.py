#!/usr/bin/env -S python3 -B
"""Optional persistent CUDA worker using the existing wime JSON-lines protocol.

Select this executable as wime-neural-program, with a conditional model and
verify search. Requires the local Python environment to provide PyTorch with
CUDA and safetensors. Stdout is exclusively the protocol; no typed text is
included in errors. The model stays on the GPU until the process is stopped.
An optional --rescorer adds a weighted general-LM score to every verified
candidate, including the seed pool, and interleaves its token proposals with
the model's per-character proposals. With --database as well, the decode
operation replaces the dictionary lattice: see decoder.py. With
--japanese-model and --japanese-table as well, the japanese operation
decodes typed kana: see japanese.py.
"""
import argparse
import json
import math
import sys
import time

from inference import ConditionalEvaluator, effective_prior_weight
from rescorer import Rescorer, interleave

LETTERS = set("abcdefghijklmnopqrstuvwxyz")
KEYS = LETTERS | {";"}


def validate(request):
    """Reject malformed envelopes before dispatch, as the Rust worker does."""
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
    anchor = request.get("context_anchor", 0)
    if type(anchor) is not int or not 0 <= anchor <= 8:
        raise ValueError("invalid request")
    known = request.get("context_reading")
    if known is not None:
        if (not isinstance(known, dict) or not isinstance(known.get("text"), str)
                or not isinstance(known.get("input"), str)
                or len(known["input"]) != 2 * len(known["text"])
                or any(c not in "abcdefghijklmnopqrstuvwxyz;" for c in known["input"])
                or not request["context"].endswith(known["text"])):
            raise ValueError("invalid request")
    if request.get("operation") == "decode":
        weight = request.get("memory_weight", 0)
        if (any(c not in KEYS for c in request.get("input", ""))
                or not isinstance(request.get("pins", []), list)
                or not isinstance(request.get("learned", []), list)
                or not isinstance(request.get("aux", []), list)
                or type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0):
            raise ValueError("invalid request")
        for pair in request.get("pins", []) + request.get("learned", []) + request.get("aux", []):
            if (not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not int
                    or not 0 <= pair[0] < 256 or not isinstance(pair[1], str) or not pair[1]):
                raise ValueError("invalid request")
        if (any(len(text) != 1 for _, text in request.get("pins", []))
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
    """Keep keyboard/coverage validation identical to conditional Rust verify."""
    validate(request)
    start = time.perf_counter()
    result = dict(id=request["id"], promoted=None, elapsed_ms=0.0)
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
            [text, score] for text, score in decoder.decode(
                request["context"], request.get("input", ""), request.get("context_anchor", 0),
                request.get("context_reading"), [tuple(p) for p in request.get("pins", [])],
                [tuple(p) for p in request.get("learned", [])], request.get("memory_weight", 0),
                [tuple(p) for p in request.get("aux", [])])]
        result["elapsed_ms"] = 1000 * (time.perf_counter() - start)
        return result
    if request.get("operation") != "verify":
        return result  # A conditional model must never perform unconditioned reranking.
    result["prefixes"] = []
    code, draft, syllables = request.get("input", ""), request.get("draft", ""), request["syllables"]
    if not syllables or len(draft) != syllables or len(code) != 2 * syllables:
        return result
    candidates = [(i, c) for i, c in enumerate(request["candidates"])
                  if c["end"] == syllables and len(c["text"]) == syllables]
    seeding = request.get("seed_candidates", False) and bool(candidates)
    texts = [c["text"] for _, c in candidates] if seeding else [draft]
    try:
        options = (dict(context_anchor=request["context_anchor"],
                        known_reading=request.get("context_reading"))
                   if request.get("context_anchor") else {})
        scores, distributions = evaluator.evaluate(request["context"], code, texts, **options)
    except ValueError:
        return result  # Invalid keys or an overlong composition retain traditional candidates.
    prior_weight = (effective_prior_weight(scores, syllables, request["prior_weight"],
                                           request.get("prior_confidence", 1.0)) if seeding else 0.0)
    if rescorer:
        # The confidence gate above stays a model-probability rule.
        scores = rescorer.combine(request["context"], texts, scores)
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
        prefixes = interleave(rescorer.proposals(request["context"], texts[selected]), prefixes)
    result.update(score=scores[selected], prefixes=prefixes,
                  evaluated=len(texts),
                  elapsed_ms=1000 * (time.perf_counter() - start))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--rescorer", help="directory of a transformers causal LM")
    parser.add_argument("--rescorer-weight", type=float, default=1.5)
    parser.add_argument("--database", help="wime word database, whose readings enable decode")
    parser.add_argument("--japanese-model", help="kana-conditioned model (train-japanese.py)")
    parser.add_argument("--japanese-table", help="reading table for the rescorer (build-japanese-table.py)")
    args = parser.parse_args()
    evaluator = ConditionalEvaluator(args.model, device="cuda")
    rescorer = Rescorer(args.rescorer, args.rescorer_weight) if args.rescorer else None
    decoder = None
    if rescorer and args.database:
        from decoder import Decoder, Units, readings
        decoder = Decoder(evaluator, rescorer, Units(rescorer.tokenizer, readings(args.database)))
    # Pay CUDA/kernel initialization once, before the first interactive request.
    evaluator.evaluate("", "aaaa", ["中文"])
    if rescorer:
        rescorer.score("", ["中文"])
    japanese = None
    if rescorer and args.japanese_model and args.japanese_table:
        import japanese as ja
        japanese = ja.Decoder(ja.Table(args.japanese_table, rescorer.tokenizer), rescorer,
                              ja.KanaModel(args.japanese_model))
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
