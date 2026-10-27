#!/usr/bin/env python3
"""Retain candidate plausibility while adding right-context likelihood.

Offline, four-phase protocol: score-dev, freeze, score-sealed, report. Previous
revealed cases may be development only. Fresh sealed labels are not parsed
until every sealed prediction exists. All settings share the same z/q/r scores.
The guarded family falls back to the ORIGINAL no-right candidate pool for
short/nonlexical right text. This is an explicit policy/pool change, not merely
another set of weights on the right-conditioned pool. Live defaults are untouched.
"""

import argparse
import copy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import random
import time
import unicodedata

_RIGHT_PATH = Path(__file__).with_name("experiment-japanese-right-ranking.py")
_SPEC = importlib.util.spec_from_file_location("japanese_right_ranking_v2", _RIGHT_PATH)
right = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(right)
previous = right.previous
CONDITIONS = previous.CONDITIONS


def setting(family, q=0.7, suffix=0.0, evidence_ratio=0.0):
    return dict(
        family=family, q_weight=q, right_weight=suffix, evidence_ratio=evidence_ratio
    )


BASELINE = setting("baseline")
PREVIOUS = setting("previous", q=0.0, suffix=0.7)
JOINT = setting("additive", suffix=0.7)
ADDITIVE = [
    setting("additive", q=w, suffix=v)
    for w in (0.35, 0.7)
    for v in (0.35, 0.7, 1.05, 1.4)
]
GUARDED = [
    setting("guarded", suffix=v, evidence_ratio=k)
    for v in (0.7, 1.05, 1.4)
    for k in (1.0, 2.0)
]
GRID = [BASELINE] + ADDITIVE + GUARDED
MIN_RIGHT_LEXICAL_CHARS = 3
MAX_NONE_Z_DRIFT = 1e-3


def lexical_chars(text):
    """Count letters/numbers, excluding spaces/punctuation; same 40-char budget."""
    return sum(unicodedata.category(c)[0] in "LN" for c in text[:40])


def decision(case, row, condition, config):
    """Complete ordering plus the pool and evidence used for its first candidate.

    A guarded replacement must have suffix gain exceeding the lost candidate
    likelihood (times a fixed ratio). It may overcome a large candidate deficit
    if actual right evidence supports it; no word-specific or gold-driven rule.
    The legacy winner always remains eligible, and rejected candidates remain
    in the returned ordering. Pool coverage is therefore unaffected by guards
    except for the declared short-right fallback to the original none pool.
    """
    right_text = case.get("right_context", "")
    short = (
        config["family"] == "guarded"
        and lexical_chars(right_text) < MIN_RIGHT_LEXICAL_CHARS
    )
    effective = "none" if not right_text or short else condition
    pool = row["pools"][effective]
    legacy = previous.ranking(pool, previous.BASELINE)
    reason = (
        "absent_right"
        if not right_text
        else "short_right_original_pool"
        if short
        else None
    )
    if reason or config["family"] == "baseline":
        return dict(
            candidates=legacy,
            effective_pool=effective,
            reason=reason or "legacy",
            baseline_top=legacy[0] if legacy else None,
        )
    n = len(pool["cands"])
    if any(len(pool[k]) != n for k in ("z", "q", "r")):
        raise ValueError("candidate scores are not aligned")
    if not all(math.isfinite(x) for k in ("z", "q", "r") for x in pool[k]):
        raise ValueError("non-finite component score")
    if not n:
        return dict(
            candidates=[],
            effective_pool=effective,
            reason="empty_pool",
            baseline_top=None,
        )
    scores = [
        z + config["q_weight"] * q + config["right_weight"] * r
        for z, q, r in zip(pool["z"], pool["q"], pool["r"])
    ]
    anchor = pool["cands"].index(legacy[0])
    guarded = config["family"] == "guarded"
    order = sorted(
        range(n), key=lambda i: (-scores[i], i != anchor if guarded else False, i)
    )
    raw_top = order[0]
    if not guarded:
        return dict(
            candidates=[pool["cands"][i] for i in order],
            effective_pool=effective,
            reason="additive"
            if config["family"] == "additive"
            else "previous_suffix_only",
            baseline_top=legacy[0],
        )

    def supported(i):
        return i == anchor or pool["r"][i] - pool["r"][anchor] > config[
            "evidence_ratio"
        ] * max(0.0, pool["q"][anchor] - pool["q"][i])

    chosen = next(i for i in order if supported(i))
    ordered = [chosen] + [i for i in order if i != chosen]
    reason = (
        "guard_rejected_additive_top"
        if chosen != raw_top
        else "legacy_winner_retained"
        if chosen == anchor
        else "supported_replacement"
    )
    return dict(
        candidates=[pool["cands"][i] for i in ordered],
        effective_pool=effective,
        reason=reason,
        baseline_top=legacy[0],
        additive_top=pool["cands"][raw_top],
        raw_q_deficit=pool["q"][anchor] - pool["q"][raw_top],
        raw_suffix_gain=pool["r"][raw_top] - pool["r"][anchor],
        evidence_ratio=config["evidence_ratio"],
    )


def displayed(case, row, condition, config):
    kana = case["kana"]
    katakana = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in kana)
    return list(
        dict.fromkeys(
            decision(case, row, condition, config)["candidates"] + [kana, katakana]
        )
    )[:5]


def read_safety(path, dev):
    if path is None:
        return {}
    data = json.loads(path.read_text())
    data = data["cases"] if isinstance(data, dict) else data
    known = {c["id"] for c in dev}
    result = {}
    for c in data:
        cid, answers = c.get("id"), c.get("answers")
        if cid not in known or cid in result:
            raise ValueError("safety cases must be unique development IDs")
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(a, str) and a for a in answers)
        ):
            raise ValueError("safety cases need reviewed nonempty answers")
        result[cid] = answers
    return result


def safety_metrics(cases, rows, config, safety):
    out = dict(
        total=len(safety), baseline_correct=0, regressions=0, fixes=0, outcomes=[]
    )
    for c, r in zip(cases, rows):
        if c["id"] not in safety:
            continue
        before = displayed(c, r, "right", BASELINE)
        after = displayed(c, r, "right", config)
        was = previous.hit(safety[c["id"]], before) == 1
        good = previous.hit(safety[c["id"]], after) == 1
        out["baseline_correct"] += was
        out["regressions"] += was and not good
        out["fixes"] += good and not was
        out["outcomes"].append(
            dict(
                id=c["id"],
                baseline=before[0],
                selected=after[0],
                baseline_correct=was,
                selected_correct=good,
            )
        )
    return out


def performance(cases, rows, config, safety):
    grouped = {}
    total = changed = 0
    for c, r in zip(cases, rows):
        answer = displayed(c, r, "right", config)
        baseline = displayed(c, r, "right", BASELINE)
        good = previous.hit(c["answers"], answer) == 1
        group = grouped.setdefault(c.get("topic_group") or c["id"], [0, 0])
        group[0] += good
        group[1] += 1
        total += good
        changed += answer[0] != baseline[0]
    macro = sum(Fraction(a, b) for a, b in grouped.values()) / len(grouped)
    safety_result = safety_metrics(cases, rows, config, safety)
    return dict(
        config=config,
        top1=total,
        total=len(cases),
        topic_groups=len(grouped),
        macro_accuracy=float(macro),
        macro_fraction=[macro.numerator, macro.denominator],
        changed_top1=changed,
        safety_regressions=safety_result["regressions"],
        safety_baseline_correct=safety_result["baseline_correct"],
        eligible=safety_result["regressions"] == 0,
    )


def select_setting(dev, rows, safety=None):
    if any(c.get("split") != "dev" for c in dev):
        raise ValueError("only dev labels may select settings")
    rows = right.align_rows(dev, rows)
    safety = safety or {}
    if set(safety) - {c["id"] for c in dev}:
        raise ValueError("safety cases must be development cases")
    sweep = [performance(dev, rows, config, safety) for config in GRID]
    baseline = sweep[0]

    def key(s):
        c = s["config"]
        return (
            Fraction(*s["macro_fraction"]),
            s["top1"],
            -s["changed_top1"],
            c["family"] == "baseline",
            c["family"] == "additive",
            c["q_weight"],
            -c["right_weight"],
            c["evidence_ratio"],
        )

    def choose(family=None):
        options = [
            s
            for s in sweep
            if s["eligible"] and (family is None or s["config"]["family"] == family)
        ]
        return (
            max(options, key=key)
            if options
            else dict(baseline, family_fallback="no safety-eligible setting")
        )

    best = choose()
    return dict(
        config=best["config"],
        selected_performance=best,
        selected_additive=choose("additive"),
        selected_guarded=choose("guarded"),
        best_unconstrained=max(sweep, key=key),
        dev_sweep=sweep,
        safety=safety_metrics(dev, rows, best["config"], safety),
        selection="zero reviewed safety regressions, then macro accuracy per topic/source group, raw top1, fewer changed first candidates, baseline/simple family, retained q, lower suffix weight, stronger evidence ratio",
        no_right="exact original none-pool z+.7q",
        short_guard="fewer than3 Unicode letters/numbers: original none-pool z+.7q",
    )


def settings(selection):
    return dict(
        baseline=BASELINE,
        previous_suffix_only=PREVIOUS,
        joint_qr=JOINT,
        selected_additive=selection["selected_additive"]["config"],
        selected_guarded=selection["selected_guarded"]["config"],
        selected=selection["config"],
    )


def metrics(cases, rows, condition, config):
    out = dict(
        total=len(cases),
        top1=0,
        top5=0,
        exact_top1=0,
        raw_pool=0,
        valid_pool=0,
        no_valid_pool=0,
        effective_pools={},
        guard_reasons={},
    )
    for c, r in zip(cases, rows):
        selected = decision(c, r, condition, config)
        pool = r["pools"][selected["effective_pool"]]
        candidates = displayed(c, r, condition, config)
        rank = previous.hit(c["answers"], candidates)
        out["top1"] += rank == 1
        out["top5"] += rank is not None
        out["exact_top1"] += (
            previous.hit(c["answers"], candidates, normalize=False) == 1
        )
        out["raw_pool"] += previous.hit(c["answers"], pool["generated"]) is not None
        out["valid_pool"] += previous.hit(c["answers"], pool["cands"]) is not None
        out["no_valid_pool"] += not pool["cands"]
        for field, value in (
            ("effective_pools", selected["effective_pool"]),
            ("guard_reasons", selected["reason"]),
        ):
            out[field][value] = out[field].get(value, 0) + 1
    return out


def groups(cases):
    result = right.groups(cases)
    result["short_right"] = [
        i
        for i, c in enumerate(cases)
        if c.get("right_context") and lexical_chars(c["right_context"]) < 3
    ]
    result["substantive_right"] = [
        i for i, c in enumerate(cases) if lexical_chars(c.get("right_context", "")) >= 3
    ]
    return {k: v for k, v in result.items() if v}


def summary(cases, rows, selection):
    rows = right.align_rows(cases, rows)
    return {
        group: {
            condition: {
                name: metrics(
                    [cases[i] for i in indices],
                    [rows[i] for i in indices],
                    condition,
                    config,
                )
                for name, config in settings(selection).items()
            }
            for condition in CONDITIONS
        }
        for group, indices in groups(cases).items()
    }


def compare(cases, rows, left_condition, left_setting, right_condition, right_setting):
    clusters = {}
    wins = losses = 0
    for c, r in zip(cases, rows):
        a = (
            previous.hit(c["answers"], displayed(c, r, left_condition, left_setting))
            == 1
        )
        b = (
            previous.hit(c["answers"], displayed(c, r, right_condition, right_setting))
            == 1
        )
        delta = int(a) - int(b)
        wins += delta > 0
        losses += delta < 0
        group = clusters.setdefault(c.get("topic_group") or c["id"], [0, 0])
        group[0] += delta
        group[1] += 1
    if not cases:
        return dict(total=0)
    values = list(clusters.values())
    rng = random.Random(20261030)
    draws = []
    for _ in range(2000):
        sample = [rng.choice(values) for v in values]
        draws.append(100 * sum(v[0] for v in sample) / sum(v[1] for v in sample))
    draws.sort()
    return dict(
        total=len(cases),
        wins=wins,
        losses=losses,
        delta_pp=100 * (wins - losses) / len(cases),
        topic_groups=len(values),
        cluster_bootstrap95_delta_pp=[draws[50], draws[1949]],
        interpretation="Exploratory2000 source/topic-group bootstrap draws; context variants clustered; no multiplicity correction",
    )


def contrasts(cases, rows, selection):
    rows = right.align_rows(cases, rows)
    chosen = settings(selection)
    pairs = dict(
        selected_vs_baseline=("right", chosen["selected"], "right", BASELINE),
        selected_vs_previous=("right", chosen["selected"], "right", PREVIOUS),
        additive_vs_baseline=("right", chosen["selected_additive"], "right", BASELINE),
        guarded_vs_additive=(
            "right",
            chosen["selected_guarded"],
            "right",
            chosen["selected_additive"],
        ),
        guarded_vs_baseline=("right", chosen["selected_guarded"], "right", BASELINE),
        selected_vs_original_no_right=("right", chosen["selected"], "none", BASELINE),
    )
    return {
        group: {
            name: compare([cases[i] for i in ix], [rows[i] for i in ix], *pair)
            for name, pair in pairs.items()
        }
        for group, ix in groups(cases).items()
    }


def outcomes(cases, rows, selection):
    rows = right.align_rows(cases, rows)
    return [
        dict(
            id=c["id"],
            topic_group=c.get("topic_group"),
            case_type=c.get("case_type"),
            answers=c["answers"],
            predictions={
                condition: {
                    name: dict(
                        decision(c, r, condition, config),
                        candidates=displayed(c, r, condition, config),
                        rank=previous.hit(
                            c["answers"], displayed(c, r, condition, config)
                        ),
                    )
                    for name, config in settings(selection).items()
                }
                for condition in CONDITIONS
            },
        )
        for c, r in zip(cases, rows)
    ]


def fingerprint(args):
    manifest = previous.fingerprint(args)
    manifest["version"] = 3
    extra = [
        Path(__file__),
        _RIGHT_PATH,
        args.dev_pools,
        args.sealed_pools,
        args.pool_manifest,
        args.safety_cases,
    ]
    for path in extra:
        if path is not None:
            manifest["files"][str(path.resolve())] = previous.digest(path)
    if args.dev_pools is not None or args.sealed_pools is not None:
        if args.pool_manifest is None:
            raise ValueError(
                "cached pools require a pool manifest to verify model/scoring provenance"
            )
        prior = json.loads(args.pool_manifest.read_text())
        prior_files = {str(Path(p).resolve()): v for p, v in prior["files"].items()}
        required = [
            args.table,
            Path(__file__).with_name("japanese.py"),
            Path(__file__).with_name("rescorer.py"),
            _RIGHT_PATH,
            _RIGHT_PATH.with_name("experiment-japanese.py"),
        ]
        for model in (args.zenz, args.rescorer):
            required.extend(
                p
                for p in model.rglob("*")
                if p.is_file()
                and p.suffix in (".json", ".safetensors", ".model", ".txt")
            )
        for path in required:
            key = str(path.resolve())
            if prior_files.get(key) != manifest["files"].get(key):
                raise ValueError("cached model/scoring dependency differs: " + key)
    manifest["protocol"] = dict(
        grid=GRID,
        controls=dict(baseline=BASELINE, previous_suffix_only=PREVIOUS, joint_qr=JOINT),
        beams=8,
        zenz_left_chars=40,
        qwen_left_chars=64,
        right_chars=40,
        selection="development macro source/topic-group accuracy subject to zero reviewed safety regressions",
        guard="Suffix gain must exceed evidence_ratio*max(0,candidate q deficit), relative to same-pool legacy winner; winner always eligible; no candidates deleted",
        short_guard="<3 lexical Unicode letters/numbers in first40 right chars: original none pool and exact legacy z+.7q",
        no_right="original none pool exact legacy z+.7q for all settings",
        score="z + retained q_weight*q + right_weight*r; explicit same suffix token boundaries as v2",
        safety="separate reviewed development-only answers; never model input",
        cache="partial caches permitted; inputs exact; cached q/r preserved; model and scoring provenance verified",
        fresh_variant_q="first observed q per candidate reused within topic_group/context/kana; computed q and origins recorded; original none generated/cands exact; none z pinned to first within declared numeric tolerance; resume reconstructs/verifies",
        max_none_z_drift=MAX_NONE_Z_DRIFT,
        condition_scope="none describes zenz no-right pool; substantive right may still be used by Qwen suffix scoring",
        timing="offline pooled union scoring; not live interaction latency",
    )
    return manifest


def share_fresh_q(case, row, shared, replay=False):
    """Pin candidate q across fresh full/short/no-right variants of one source.

    A different union batch can perturb BF16 candidate scores although the left
    context is identical. Retain the first observed q per candidate and record
    the freshly computed values so this adjustment is auditable. Suffix r stays
    specific to each right context. Original none candidates must reproduce
    exactly; tiny repeated sum-logprob drift is pinned to the first z, with
    computed values retained and drift above MAX_NONE_Z_DRIFT rejected.
    Cached previous-round rows bypass this function and stay unchanged.
    """
    right.align_rows([case], [row])
    key = (case.get("topic_group") or case["id"], case.get("context", ""), case["kana"])
    none = {name: row["pools"]["none"][name] for name in ("generated", "cands", "z")}
    group = shared.setdefault(
        key, dict(first_id=case["id"], none=copy.deepcopy(none), q={}, origins={})
    )
    if any(none[name] != group["none"][name] for name in ("generated", "cands")):
        raise ValueError(
            "fresh context variants changed original none pool/generated/cands"
        )
    union = row["q_union"]
    if replay:
        saved = row.get("shared_q")
        if not isinstance(saved, dict):
            raise ValueError("fresh saved row lacks shared-q provenance")
        computed = saved.get("computed_q")
        computed_none_z = saved.get("computed_none_z")
        if (
            not isinstance(computed_none_z, list)
            or len(computed_none_z) != len(none["z"])
            or not all(math.isfinite(v) for v in computed_none_z)
        ):
            raise ValueError("invalid saved computed-none-z provenance")
        if (
            not isinstance(computed, list)
            or len(computed) != len(union)
            or not all(math.isfinite(v) for v in computed)
        ):
            raise ValueError("invalid saved computed-q provenance")
    else:
        by_text = {
            t: q
            for pool in row["pools"].values()
            for t, q in zip(pool["cands"], pool["q"])
        }
        computed = [by_text[t] for t in union]
        computed_none_z = list(none["z"])
    z_drift = max(
        (abs(a - b) for a, b in zip(computed_none_z, group["none"]["z"])), default=0.0
    )
    if z_drift > MAX_NONE_Z_DRIFT:
        raise ValueError(
            f"fresh original none z drift {z_drift} exceeds {MAX_NONE_Z_DRIFT}"
        )
    z_pools = ("none",) if case.get("right_context") else CONDITIONS
    for name in z_pools:
        if replay:
            if row["pools"][name]["z"] != group["none"]["z"]:
                raise ValueError(
                    "saved fresh variants do not share identical original none z scores"
                )
        else:
            row["pools"][name]["z"] = list(group["none"]["z"])
    reused = 0
    for text, value in zip(union, computed):
        if text in group["q"]:
            reused += 1
        else:
            group["q"][text], group["origins"][text] = value, case["id"]
    expected = dict(
        group_key=list(key),
        none_reference_id=group["first_id"],
        computed_q=computed,
        computed_none_z=computed_none_z,
        max_abs_none_z_adjustment=z_drift,
        q_origins=[group["origins"][t] for t in union],
        reused_candidates=reused,
        new_candidates=len(union) - reused,
        max_abs_q_adjustment=max(
            (abs(q - group["q"][t]) for t, q in zip(union, computed)), default=0.0
        ),
    )
    if replay and saved != expected:
        raise ValueError("saved shared-q provenance differs from prior fresh rows")
    for pool in row["pools"].values():
        values = [group["q"][t] for t in pool["cands"]]
        if replay:
            if pool["q"] != values:
                raise ValueError("saved fresh variants do not share identical q scores")
        else:
            pool["q"] = values
    if not replay:
        row["shared_q"] = expected
    return row


def verify_shared_rows(cases, rows):
    """Require complete predictions and replay fresh sharing before selection/unsealing."""
    right.align_rows(cases, rows)
    by_id = {c["id"]: c for c in cases}
    shared = {}
    # Physical row order is the first-seen scoring order, including resumes.
    for row in rows:
        if row.get("pool_source") != "cached_q_and_suffix":
            share_fresh_q(by_id[row["input"]["id"]], row, shared, replay=True)
    return shared


def score_cases(args, cases, split):
    path = args.out / f"{split}.rows.jsonl"
    rows = previous.read_rows(path, repair=True)
    expected = {c["id"]: c for c in cases}
    shared = {}
    for row in rows:
        c = expected.get(row["input"]["id"])
        if c is None:
            raise ValueError("saved prediction IDs differ")
        right.align_rows([c], [row])
        if row.get("pool_source") != "cached_q_and_suffix":
            share_fresh_q(c, row, shared, replay=True)
    done = {r["input"]["id"] for r in rows}
    cached_path = args.dev_pools if split == "dev" else args.sealed_pools
    cached = {}
    if cached_path is not None:
        for row in previous.read_rows(cached_path):
            cid = row["input"]["id"]
            if cid not in expected:
                raise ValueError("cached pool ID absent from requested cases")
            right.align_rows([expected[cid]], [row])
            cached[cid] = row
    decoder = None
    start = time.monotonic()
    with path.open("a", buffering=1) as stream:
        for c in cases:
            if c["id"] in done:
                continue
            if c["id"] in cached:
                source = cached[c["id"]]
                row = dict(
                    input=previous.public_input(c),
                    pools=copy.deepcopy(source["pools"]),
                    q_union=source["q_union"],
                    q_union_ms=source["q_union_ms"],
                    q_right_union_ms=source["q_right_union_ms"],
                    right_applied=bool(c.get("right_context")),
                    pool_source="cached_q_and_suffix",
                    new_suffix_inference_ms=0.0,
                )
                for key in ("right_chars_scored", "right_tokens_scored"):
                    if key in source:
                        row[key] = source[key]
            else:
                if decoder is None:
                    scorer = previous.Rescorer(args.rescorer, 0.6)
                    decoder = previous.Decoder(
                        previous.Table(args.table, scorer.tokenizer),
                        scorer,
                        None,
                        zenz=previous.RightZenz(args.zenz, beams=8),
                    )
                    right.suffix_score(decoder, "", ["日本語"], "を学ぶ。")
                row = right.inspect(previous.public_input(c), decoder)
                row["new_suffix_inference_ms"] = row["q_right_union_ms"]
                share_fresh_q(c, row, shared)
            right.align_rows([c], [row])
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            rows.append(row)
            done.add(c["id"])
            if len(done) % 10 == 0 or len(done) == len(cases):
                print(
                    json.dumps(
                        dict(
                            split=split,
                            done=len(done),
                            total=len(cases),
                            elapsed_s=round(time.monotonic() - start),
                        )
                    ),
                    flush=True,
                )
    right.align_rows(cases, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("score-dev", "freeze", "score-sealed", "report"))
    for name in (
        "evaluation-manifest",
        "dev-cases",
        "sealed-inputs",
        "sealed-labels",
        "table",
        "rescorer",
        "zenz",
        "out",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    for name in ("dev-pools", "sealed-pools", "pool-manifest", "safety-cases"):
        ap.add_argument("--" + name, type=Path)
    args = ap.parse_args()
    dev = previous.read_cases(args.dev_cases, "dev", labels=True)
    sealed = previous.read_cases(args.sealed_inputs, "sealed", labels=False)
    previous.check_partitions(dev, sealed)
    safety = read_safety(args.safety_cases, dev)
    manifest = fingerprint(args)
    args.out.mkdir(parents=True, exist_ok=True)
    previous.write_once(args.out / "manifest.json", json.loads(json.dumps(manifest)))
    if args.phase == "score-dev":
        if (args.out / "selection.json").exists():
            previous.require_frozen(args.out, manifest)
        score_cases(args, dev, "dev")
    elif args.phase == "freeze":
        rows = previous.read_rows(args.out / "dev.rows.jsonl")
        verify_shared_rows(dev, rows)
        selected = select_setting(dev, rows, safety)
        selection = dict(
            manifest=manifest,
            dev_rows_sha256=previous.digest(args.out / "dev.rows.jsonl"),
            selected=selected,
            label_access="sealed labels hashed,not parsed",
        )
        previous.write_once(
            args.out / "selection.json", json.loads(json.dumps(selection))
        )
        previous.write_once(args.out / "dev.summary.json", summary(dev, rows, selected))
        print(json.dumps(selected, ensure_ascii=False))
    elif args.phase == "score-sealed":
        previous.require_frozen(args.out, manifest)
        previous.write_once(
            args.out / "sealed.run.json",
            dict(selection_sha256=previous.digest(args.out / "selection.json")),
        )
        score_cases(args, sealed, "sealed")
    else:
        selection = previous.require_frozen(args.out, manifest)
        expected = dict(selection_sha256=previous.digest(args.out / "selection.json"))
        if (
            not (args.out / "sealed.run.json").exists()
            or json.loads((args.out / "sealed.run.json").read_text()) != expected
        ):
            raise ValueError("sealed predictions are not bound to frozen selection")
        rows = previous.read_rows(args.out / "sealed.rows.jsonl")
        verify_shared_rows(sealed, rows)
        labeled = previous.read_labels(args.sealed_labels, sealed)
        report = dict(
            completed=True,
            selection_sha256=previous.digest(args.out / "selection.json"),
            sealed_predictions_sha256=previous.digest(args.out / "sealed.rows.jsonl"),
            selected=selection["selected"],
            caution="fixed controls and dev-only selection; guarded short right switches to original pool; no sealed tuning",
            dev=summary(
                dev,
                previous.read_rows(args.out / "dev.rows.jsonl"),
                selection["selected"],
            ),
            sealed=summary(labeled, rows, selection["selected"]),
            sealed_contrasts=contrasts(labeled, rows, selection["selected"]),
            sealed_outcomes=outcomes(labeled, rows, selection["selected"]),
        )
        previous.write_once(args.out / "report.json", report)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
