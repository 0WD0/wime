#!/usr/bin/env python3
"""Frozen right-aware ranking experiment; leaves the live backend unchanged.

Each candidate gets q = log P(candidate | left) and r = log P(right | left,
 candidate), using explicit token boundaries. Ranking is z + w*q + v*r.
The candidate pool and all three component scores are shared by every setting.
No-right cases always retain the original z + .7*q ranking. Select w/v on dev,
freeze, predict unseen sealed inputs, and only then parse sealed labels.
Previously disclosed evaluation rows may be reused as dev via --dev-pools.
"""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import time

_MODULE_PATH = Path(__file__).with_name("experiment-japanese.py")
_SPEC = importlib.util.spec_from_file_location("japanese_context_experiment", _MODULE_PATH)
previous = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(previous)

GRID = [(w, v) for w in (0., .35, .7) for v in (0., .175, .35, .7)]
BASELINE = (.7, 0.)
CONTROLS = {"baseline": BASELINE, "gated_zenz": (0., 0.), "gated_half": (.35, 0.)}
CONDITIONS = previous.CONDITIONS


def suffix_score(decoder, context, candidates, right):
    """Only suffix tokens are scored; the cached candidate score is not repeated.

    Separate segment tokenization deliberately imposes candidate/right token
    boundaries, just as production imposes left/candidate boundaries. This is
    not canonical tokenization of the concatenated full string. The same suffix
    token IDs are used for every candidate, without BOS/EOS insertion.
    """
    clipped = right[:40]
    if not clipped or not candidates:
        return [0.] * len(candidates), 0
    prefix = decoder.lm_context(context)
    tokens = decoder.table.tokenizer
    candidate_ids = [tokens(text)["input_ids"] for text in candidates]
    suffix_ids = tokens(clipped, add_special_tokens=False)["input_ids"]
    if not prefix or not suffix_ids or any(not ids for ids in candidate_ids):
        raise ValueError("context, candidates and nonempty suffix must tokenize to nonempty sequences")
    values = decoder.lm_tails(prefix, candidate_ids, [suffix_ids] * len(candidates))
    if len(values) != len(candidates) or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid suffix score")
    return values, len(suffix_ids)


def inspect(case, decoder, source=None):
    """Score the sorted union once, while preserving cached candidate q exactly."""
    cached = source is not None
    if source is None:
        source = previous.inspect(case, decoder)
    if source["input"] != previous.public_input(case):
        raise ValueError("cached pool inputs differ")
    # Copy only the known prediction fields, never case labels or review notes.
    row = dict(input=previous.public_input(case), pools=copy.deepcopy(source["pools"]),
               q_union=source["q_union"], q_union_ms=source["q_union_ms"],
               right_applied=bool(case.get("right_context")),
               pool_source="cached" if cached else "generated")
    union = sorted(set(t for p in row["pools"].values() for t in p["cands"]))
    if union != row["q_union"]:
        raise ValueError("cached Qwen union differs from candidate pools")
    right = case.get("right_context", "")
    if right and union:
        (scores, count), elapsed = previous.timed(
            lambda: suffix_score(decoder, case.get("context", ""), union, right))
    else:
        scores, count, elapsed = [0.] * len(union), 0, 0.
    by_text = dict(zip(union, scores))
    for pool in row["pools"].values():
        pool["r"] = [by_text[t] for t in pool["cands"]]
    row.update(q_right_union_ms=elapsed, right_chars_scored=len(right[:40]), right_tokens_scored=count)
    return row


def ranking(case, pool, setting):
    # Exact delegation guarantees the unchanged no-right baseline, even if a
    # selected right-aware setting drops q or gives the suffix a nonzero weight.
    if not case.get("right_context"):
        return previous.ranking(pool, previous.BASELINE)
    w, v = setting
    n = len(pool["cands"])
    if any(len(pool[k]) != n for k in ("z", "q", "r")):
        raise ValueError("candidate scores are not aligned")
    if not all(math.isfinite(x) for k in ("z", "q", "r") for x in pool[k]):
        raise ValueError("non-finite candidate score")
    scores = [z + w*q + v*r for z, q, r in zip(pool["z"], pool["q"], pool["r"])]
    return [pool["cands"][i] for i in sorted(range(n), key=lambda i: (-scores[i], i))]


def displayed(case, pool, setting):
    kana = case["kana"]
    katakana = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in kana)
    return list(dict.fromkeys(ranking(case, pool, setting) + [kana, katakana]))[:5]


def align_rows(cases, rows):
    ordered = previous.align_rows(cases, rows)
    for case, row in zip(cases, ordered):
        union = sorted(set(t for p in row["pools"].values() for t in p["cands"]))
        if row.get("q_union") != union:
            raise ValueError("Qwen union differs from candidate pools")
        q_by_text, r_by_text = {}, {}
        for pool in row["pools"].values():
            if len(pool.get("r", [])) != len(pool["cands"]):
                raise ValueError("suffix scores are not aligned")
            if not all(math.isfinite(r) for r in pool["r"]):
                raise ValueError("non-finite suffix score")
            for t, q, r in zip(pool["cands"], pool["q"], pool["r"]):
                if t in q_by_text and (q_by_text[t] != q or r_by_text[t] != r):
                    raise ValueError("shared candidates must have identical Qwen scores")
                q_by_text[t], r_by_text[t] = q, r
            ranking(case, pool, BASELINE)
        none, rescored = row["pools"]["none"], row["pools"]["right_rescore"]
        if any(none[k] != rescored[k] for k in ("cands", "q", "r")):
            raise ValueError("right-rescore must retain the no-right candidate pool")
        if not case.get("right_context"):
            if any(r_by_text.values()) or row.get("q_right_union_ms") != 0:
                raise ValueError("absent right context must not be scored")
            if any(pool[k] != none[k] for pool in row["pools"].values()
                   for k in ("generated", "cands", "z", "q")):
                raise ValueError("absent right context must retain identical baseline pools")
    return ordered


def select_setting(dev, rows):
    if any(c.get("split") != "dev" for c in dev):
        raise ValueError("only dev labels may select settings")
    rows = align_rows(dev, rows)
    if not any(c.get("right_context") for c in dev):
        raise ValueError("right-aware selection requires dev cases with right context")
    sweep = []
    for w, v in GRID:
        score = sum(previous.hit(c["answers"], displayed(c, r["pools"]["right"], (w, v))) == 1
                    for c, r in zip(dev, rows))
        sweep.append(dict(q_weight=w, right_weight=v, top1=score))
    # Prefer the existing baseline if tied; otherwise the smallest added suffix
    # weight, then the q weight nearest the original .7, then the lower q weight.
    def choose(options):
        return max(options, key=lambda s: (s["top1"], (s["q_weight"], s["right_weight"]) == BASELINE,
                                           -s["right_weight"], -abs(s["q_weight"] - .7), -s["q_weight"]))
    best = choose(sweep)
    gate = choose([s for s in sweep if s["right_weight"] == 0.])
    suffix = choose([s for s in sweep if s["right_weight"] > 0.])
    return dict(q_weight=best["q_weight"], right_weight=best["right_weight"],
                selected_gate=gate, selected_suffix=suffix, dev_sweep=sweep,
                selection="dev right-pool NFKC top1; ties baseline, smaller suffix weight, q nearer .7",
                absent_right="always original .7 q weight, zero suffix weight")


def settings(selection):
    result = dict(CONTROLS, selected=(selection["q_weight"], selection["right_weight"]))
    for name in ("selected_gate", "selected_suffix"):
        chosen = selection[name]
        result[name] = (chosen["q_weight"], chosen["right_weight"])
    return result


def metrics(cases, rows, condition, setting):
    values = dict(total=len(cases), top1=0, top5=0, exact_top1=0, raw_pool=0,
                  valid_pool=0, no_valid_pool=0)
    for c, r in zip(cases, rows):
        pool = r["pools"][condition]
        cands = displayed(c, pool, setting)
        rank = previous.hit(c["answers"], cands)
        values["top1"] += rank == 1
        values["top5"] += rank is not None
        values["exact_top1"] += previous.hit(c["answers"], cands, normalize=False) == 1
        values["raw_pool"] += previous.hit(c["answers"], pool["generated"]) is not None
        values["valid_pool"] += previous.hit(c["answers"], pool["cands"]) is not None
        values["no_valid_pool"] += not pool["cands"]
    return values


def groups(cases):
    result = {"all": list(range(len(cases))),
              "right_present": [i for i,c in enumerate(cases) if c.get("right_context")],
              "right_absent": [i for i,c in enumerate(cases) if not c.get("right_context")]}
    for field in ("domain", "case_type"):
        for name in sorted({c.get(field, "") for c in cases}):
            result[field + "/" + name] = [i for i,c in enumerate(cases) if c.get(field, "") == name]
    return {name: indices for name, indices in result.items() if indices}


def summary(cases, rows, selection):
    rows = align_rows(cases, rows)
    return {name: {condition: {label: metrics([cases[i] for i in indices], [rows[i] for i in indices],
                                            condition, setting)
                              for label, setting in settings(selection).items()}
                   for condition in CONDITIONS} for name, indices in groups(cases).items()}


def compare(cases, rows, left_condition, left_setting, right_condition, right_setting):
    clusters = {}
    wins = losses = 0
    for c, r in zip(cases, rows):
        left = previous.hit(c["answers"], displayed(c, r["pools"][left_condition], left_setting)) == 1
        right = previous.hit(c["answers"], displayed(c, r["pools"][right_condition], right_setting)) == 1
        delta = int(left) - int(right)
        wins += delta > 0
        losses += delta < 0
        group = clusters.setdefault(c.get("topic_group") or c["id"], [0, 0])
        group[0] += delta
        group[1] += 1
    if not cases:
        return dict(total=0)
    values = list(clusters.values())
    rng = random.Random(20261029)
    draws = []
    for _ in range(2000):
        sample = [rng.choice(values) for _ in values]
        draws.append(100 * sum(v[0] for v in sample) / sum(v[1] for v in sample))
    draws.sort()
    return dict(total=len(cases), wins=wins, losses=losses, delta_pp=100*(wins-losses)/len(cases),
                topic_groups=len(values), cluster_bootstrap95_delta_pp=[draws[50], draws[1949]],
                interpretation="Exploratory paired contrast; 2000 topic-group bootstrap draws; no multiple-comparison correction")


def contrasts(cases, rows, selection):
    rows = align_rows(cases, rows)
    chosen = (selection["q_weight"], selection["right_weight"])
    same_q_without_suffix = (chosen[0], 0.)
    gate, suffix = (settings(selection)[name] for name in ("selected_gate", "selected_suffix"))
    pairs = {"right_generation_at_baseline": ("right", BASELINE, "none", BASELINE),
             "selected_vs_baseline_right_pool": ("right", chosen, "right", BASELINE),
             "suffix_at_selected_q_right_pool": ("right", chosen, "right", same_q_without_suffix),
             "selected_suffix_vs_selected_gate": ("right", suffix, "right", gate),
             "selected_suffix_vs_same_q_without_suffix": ("right", suffix, "right", (suffix[0], 0.)),
             "gated_half_vs_baseline_right_pool": ("right", (.35, 0.), "right", BASELINE),
             "gated_zenz_vs_baseline_right_pool": ("right", (0., 0.), "right", BASELINE),
             "selected_right_rescore_vs_none": ("right_rescore", chosen, "none", BASELINE),
             "selected_right_generation_vs_rescore": ("right", chosen, "right_rescore", chosen)}
    return {name: {label: compare([cases[i] for i in indices], [rows[i] for i in indices], *pair)
                   for label, pair in pairs.items()} for name, indices in groups(cases).items()}


def outcomes(cases, rows, selection):
    """After unsealing, expose mistakes and fixed-control outcomes for inspection."""
    rows = align_rows(cases, rows)
    return [dict(id=c["id"], topic_group=c.get("topic_group"), case_type=c.get("case_type"),
                 answers=c["answers"],
                 predictions={condition: {name: dict(
                     candidates=displayed(c, r["pools"][condition], setting),
                     rank=previous.hit(c["answers"], displayed(c, r["pools"][condition], setting)))
                     for name, setting in settings(selection).items()} for condition in CONDITIONS})
            for c, r in zip(cases, rows)]


def fingerprint(args):
    manifest = previous.fingerprint(args)
    manifest["version"] = 2
    for path in (Path(__file__), args.dev_pools, args.sealed_pools):
        if path is not None:
            manifest["files"][str(path.resolve())] = previous.digest(path)
    manifest["protocol"] = dict(
        beams=8, zenz_left_chars=40, qwen_left_chars=64, right_chars=40,
        grid=GRID, baseline=BASELINE, conditions=CONDITIONS, fixed_controls=CONTROLS,
        condition_scope="conditions identify zenz pools/scores; a positive suffix weight also uses right in the none pool",
        selection_pool="right", selection_labels="dev only",
        separate_family_selections="best v=0 gate and best v>0 suffix, selected separately on dev",
        qwen_context="production last 64 chars and Japanese neutral prefix",
        qwen_candidate="cached or production summed logprob, no EOS",
        qwen_right="suffix-only summed logprob at explicit segment token boundary, same suffix tokens for all candidates, no BOS/EOS",
        score="z + q_weight*q + right_weight*r; no-right always original z+.7*q",
        timing="paired candidate pools and pooled Qwen suffix batch; not interactive latency",
        validation="production Table.spells; no unvalidated model candidates",
        disclosed_previous_data="may become dev only; reused pool inputs must match current dev inputs exactly")
    return manifest


def score_cases(args, cases, split):
    path = args.out / f"{split}.rows.jsonl"
    rows = previous.read_rows(path, repair=True)
    expected = {c["id"]: c for c in cases}
    for row in rows:
        case = expected.get(row["input"]["id"])
        if case is None:
            raise ValueError("saved prediction IDs differ")
        align_rows([case], [row])
    done = {r["input"]["id"] for r in rows}
    if len(done) == len(cases):
        return
    cached_path = args.dev_pools if split == "dev" else args.sealed_pools
    cached = {}
    if cached_path is not None:
        cached_rows = previous.read_rows(cached_path)
        previous.align_rows(cases, cached_rows)
        cached = {r["input"]["id"]: r for r in cached_rows}
    rescorer = previous.Rescorer(args.rescorer, .6)
    decoder = previous.Decoder(previous.Table(args.table, rescorer.tokenizer), rescorer, None,
                               zenz=None if cached else previous.RightZenz(args.zenz, beams=8))
    # No label enters warmup or model scoring.
    suffix_score(decoder, "", ["日本語"], "を学ぶ。")
    start = time.monotonic()
    with path.open("a", buffering=1) as stream:
        for c in cases:
            if c["id"] in done:
                continue
            row = inspect(previous.public_input(c), decoder, cached.get(c["id"]))
            align_rows([c], [row])
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            rows.append(row)
            done.add(c["id"])
            if len(done) % 10 == 0 or len(done) == len(cases):
                print(json.dumps(dict(split=split, done=len(done), total=len(cases),
                                      elapsed_s=round(time.monotonic() - start))), flush=True)
    align_rows(cases, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("score-dev", "freeze", "score-sealed", "report"))
    for name in ("evaluation-manifest", "dev-cases", "sealed-inputs", "sealed-labels", "table", "rescorer", "zenz", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    for name in ("dev-pools", "sealed-pools"):
        ap.add_argument("--" + name, type=Path)
    args = ap.parse_args()
    dev = previous.read_cases(args.dev_cases, "dev", labels=True)
    sealed = previous.read_cases(args.sealed_inputs, "sealed", labels=False)
    previous.check_partitions(dev, sealed)
    manifest = fingerprint(args)
    args.out.mkdir(parents=True, exist_ok=True)
    previous.write_once(args.out / "manifest.json", json.loads(json.dumps(manifest)))
    if args.phase == "score-dev":
        if (args.out / "selection.json").exists():
            previous.require_frozen(args.out, manifest)
        score_cases(args, dev, "dev")
    elif args.phase == "freeze":
        rows = previous.read_rows(args.out / "dev.rows.jsonl")
        selection = dict(manifest=manifest, dev_rows_sha256=previous.digest(args.out / "dev.rows.jsonl"),
                         selected=select_setting(dev, rows), label_access="sealed labels hashed, not parsed")
        previous.write_once(args.out / "selection.json", json.loads(json.dumps(selection)))
        previous.write_once(args.out / "dev.summary.json", summary(dev, rows, selection["selected"]))
        print(json.dumps(selection["selected"], ensure_ascii=False))
    elif args.phase == "score-sealed":
        previous.require_frozen(args.out, manifest)
        previous.write_once(args.out / "sealed.run.json", dict(selection_sha256=previous.digest(args.out / "selection.json")))
        score_cases(args, sealed, "sealed")
    else:
        selection = previous.require_frozen(args.out, manifest)
        expected = dict(selection_sha256=previous.digest(args.out / "selection.json"))
        if not (args.out / "sealed.run.json").exists() or json.loads(
                (args.out / "sealed.run.json").read_text()) != expected:
            raise ValueError("sealed predictions are not bound to this frozen selection")
        rows = previous.read_rows(args.out / "sealed.rows.jsonl")
        align_rows(sealed, rows)  # Complete blind prediction before any label parse.
        labeled = previous.read_labels(args.sealed_labels, sealed)
        report = dict(completed=True, selection_sha256=previous.digest(args.out / "selection.json"),
                      sealed_predictions_sha256=previous.digest(args.out / "sealed.rows.jsonl"),
                      selected=selection["selected"],
                      caution="fixed controls and dev-selected setting only; no sealed setting selection",
                      dev=summary(dev, previous.read_rows(args.out / "dev.rows.jsonl"), selection["selected"]),
                      sealed=summary(labeled, rows, selection["selected"]),
                      sealed_contrasts=contrasts(labeled, rows, selection["selected"]),
                      sealed_outcomes=outcomes(labeled, rows, selection["selected"]))
        previous.write_once(args.out / "report.json", report)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
