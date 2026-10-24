#!/usr/bin/env python3
"""Optional persistent CUDA worker using the existing wime JSON-lines protocol.

Select this executable as wime-neural-program, with a conditional model and
verify search. Requires the local Python environment to provide PyTorch with
CUDA and safetensors. Stdout is exclusively the protocol; no typed text is
included in errors. The model stays on the GPU until the process is stopped.
"""
import argparse
import json
import math
import sys
import time

from inference import ConditionalEvaluator


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
    for key in ("operation", "input", "draft"):
        if not isinstance(request.get(key, ""), str):
            raise ValueError("invalid request")
    if type(request.get("seed_candidates", False)) is not bool:
        raise ValueError("invalid request")
    if request.get("seed_candidates"):
        if (type(request.get("prior_weight")) not in (int, float)
                or not math.isfinite(request["prior_weight"]) or request["prior_weight"] < 0):
            raise ValueError("invalid request")
        for candidate in request["candidates"]:
            if (type(candidate.get("score")) not in (int, float)
                    or not math.isfinite(candidate["score"])):
                raise ValueError("invalid request")


def respond(evaluator, request):
    """Keep keyboard/coverage validation identical to conditional Rust verify."""
    validate(request)
    start = time.perf_counter()
    result = dict(id=request["id"], promoted=None, elapsed_ms=0.0)
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
        scores, distributions = evaluator.evaluate(request["context"], code, texts)
    except ValueError:
        return result  # Invalid keys or an overlong composition retain traditional candidates.
    selected = (max(range(len(candidates)), key=lambda j: scores[j] +
                    request["prior_weight"] * candidates[j][1]["score"] / syllables)
                if seeding else 0)
    if seeding:
        # Return an index into the original, aux-filtered dictionary pool. The
        # editor never accepts free-form model text as a seed candidate.
        result["verified_index"] = candidates[selected][0]
    result.update(score=scores[selected],
                  prefixes=evaluator.proposals(texts[selected], distributions[selected], 0.5),
                  evaluated=len(texts),
                  elapsed_ms=1000 * (time.perf_counter() - start))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    args = parser.parse_args()
    evaluator = ConditionalEvaluator(args.model, device="cuda")
    # Pay CUDA/kernel initialization once, before the first interactive request.
    evaluator.evaluate("", "aaaa", ["中文"])
    for line in sys.stdin:
        response = respond(evaluator, json.loads(line))
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"wime CUDA worker failed ({type(error).__name__})", file=sys.stderr)
        sys.exit(1)
