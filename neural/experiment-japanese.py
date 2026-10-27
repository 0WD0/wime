#!/usr/bin/env python3
"""Frozen right-context × candidate-ranking experiment (not a live backend).

Run score-dev, freeze, score-sealed, then report with the same arguments.
Sealed inputs must omit answers. Only report parses --sealed-labels; earlier
phases hash that file to commit to the labels before choosing a setting.
The ranking grid is selected on the no-right-context dev pool, then applied
unchanged to all conditions. This keeps the two experimental factors separate.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time
import unicodedata

from japanese import Decoder, Table, Zenz
from rescorer import Rescorer


GRID = [(w, p) for w in (0., .35, .7, 1.05) for p in (0, 1)]
BASELINE = (.7, 0)
CONDITIONS = ("none", "right", "right_rescore")
INPUT_FIELDS = ("id", "split", "context", "kana", "right_context", "topic_group", "domain", "case_type")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nfkc(text):
    return unicodedata.normalize("NFKC", text)


def public_input(case):
    return {k: case.get(k, "") for k in INPUT_FIELDS}


def read_cases(path, split, labels):
    data = json.loads(Path(path).read_text())
    cases = data["cases"] if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    ids = set()
    for c in cases:
        if not isinstance(c.get("id"), str) or not c["id"] or c["id"] in ids:
            raise ValueError("case IDs must be nonempty and unique")
        ids.add(c["id"])
        if c.get("split") != split or not isinstance(c.get("kana"), str) or not c["kana"]:
            raise ValueError("case split/kana differs from the requested partition")
        for field in ("context", "right_context", "topic_group", "domain"):
            if field in c and not isinstance(c[field], str):
                raise ValueError("input fields must be strings")
        if labels:
            if not isinstance(c.get("answers"), list) or not c["answers"] or not all(
                    isinstance(a, str) and a for a in c["answers"]):
                raise ValueError("dev cases need nonempty answers")
        elif any(k in c for k in ("answers", "gold", "target", "output")):
            raise ValueError("sealed inputs must not contain labels")
    return cases


def check_partitions(dev, sealed):
    if {c["id"] for c in dev} & {c["id"] for c in sealed}:
        raise ValueError("dev and sealed IDs overlap")
    groups = lambda rows: {c["topic_group"] for c in rows if c.get("topic_group")}
    if groups(dev) & groups(sealed):
        raise ValueError("dev and sealed topic groups overlap")


class RightZenz(Zenz):
    """Reuse production generation/scoring with v3.2's native right context."""
    right_context = ""

    def prompt(self, context, kana):
        if not self.right_context:
            return super().prompt(context, kana)
        katakana = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in kana)
        left, right = context[-40:], self.right_context[:40]
        return self.encode(("\uee02" + left if left else "") + "\uee07" + right
                           + "\uee00" + katakana + "\uee01")


def timed(function):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, 1000 * (time.perf_counter() - start)


def inspect(case, decoder):
    """Generate two pools; every ranking of a pool shares identical candidates.

    Qwen scores the sorted union once, so overlapping candidates have exactly
    the same q score in both conditions (including BF16 batching effects).
    No answer is used here or copied into prediction records.
    """
    context, kana, right = case.get("context", ""), case["kana"], case.get("right_context", "")
    pools = {}
    for condition in ("none", "right"):
        if condition == "right" and not right:
            pools[condition] = dict(pools["none"], reused_from="none")
            continue
        decoder.zenz.right_context = right if condition == "right" else ""
        generated, generation_ms = timed(lambda: decoder.zenz.generate(context, kana))
        generated = list(dict.fromkeys(t for t in generated if t))
        accepted, validation_ms = timed(lambda: [t for t in generated if decoder.table.spells(kana, t)])
        z, z_ms = timed(lambda: decoder.zenz.score(context, kana, accepted)) if accepted else ([], 0.)
        pools[condition] = dict(generated=generated, cands=accepted, z=z,
                                generation_ms=generation_ms, validation_ms=validation_ms, z_ms=z_ms)
    union = sorted(set(pools["none"]["cands"] + pools["right"]["cands"]))
    q, q_ms = timed(lambda: decoder.lm_tails(decoder.lm_context(context), [()] * len(union),
                    [decoder.table.tokenizer(t)["input_ids"] for t in union])) if union else ([], 0.)
    q_by_text = dict(zip(union, q))
    for pool in pools.values():
        pool["q"] = [q_by_text[t] for t in pool["cands"]]
    # Isolate changing only the conditional score, with the no-right pool fixed.
    base = pools["none"]
    decoder.zenz.right_context = right
    if right and base["cands"]:
        z, z_ms = timed(lambda: decoder.zenz.score(context, kana, base["cands"]))
    else:
        z, z_ms = base["z"], 0.
    pools["right_rescore"] = dict(base, z=z, z_ms=z_ms, pool_source="none")
    decoder.zenz.right_context = ""
    return dict(input=public_input(case), right_applied=bool(right), pools=pools,
                q_union=union, q_union_ms=q_ms)


def ranking(pool, setting):
    weight, power = setting
    n = len(pool["cands"])
    if len(pool["z"]) != n or len(pool["q"]) != n or any(not t for t in pool["cands"]):
        raise ValueError("candidate scores are not aligned")
    if not all(math.isfinite(v) for v in pool["z"] + pool["q"]):
        raise ValueError("non-finite candidate score")
    scores = [(z + weight * q) / len(t) ** power
              for t, z, q in zip(pool["cands"], pool["z"], pool["q"])]
    return [pool["cands"][i] for i in sorted(range(n), key=lambda i: (-scores[i], i))]


def displayed(case, pool, setting):
    kana = case["kana"]
    katakana = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in kana)
    return list(dict.fromkeys(ranking(pool, setting) + [kana, katakana]))[:5]


def hit(answers, candidates, normalize=True):
    norm = nfkc if normalize else lambda x: x
    acceptable = {norm(a) for a in answers}
    return next((i + 1 for i, t in enumerate(candidates) if norm(t) in acceptable), None)


def select_setting(dev, rows):
    if any(c.get("split") != "dev" for c in dev):
        raise ValueError("only dev labels may select settings")
    ordered = align_rows(dev, rows)
    sweep = []
    for setting in GRID:
        score = sum(hit(c["answers"], displayed(c, r["pools"]["none"], setting)) == 1
                    for c, r in zip(dev, ordered))
        sweep.append(dict(weight=setting[0], length_power=setting[1], top1=score))
    # An exactly tied existing baseline wins; otherwise no length normalization,
    # then the weight nearest the established baseline, then the smaller weight.
    best = max(sweep, key=lambda s: (s["top1"], (s["weight"], s["length_power"]) == BASELINE,
                                     -s["length_power"], -abs(s["weight"] - .7), -s["weight"]))
    return dict(weight=best["weight"], length_power=best["length_power"], dev_sweep=sweep,
                selection="dev none-pool NFKC top1; ties baseline, no length norm, nearest baseline weight")


def align_rows(cases, rows):
    by_id = {r["input"]["id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {c["id"] for c in cases}:
        raise ValueError("prediction rows are incomplete, duplicated or unexpected")
    ordered = [by_id[c["id"]] for c in cases]
    if any(r["input"] != public_input(c) for c, r in zip(cases, ordered)):
        raise ValueError("prediction inputs differ")
    for r in rows:
        if set(r["pools"]) != set(CONDITIONS):
            raise ValueError("prediction conditions differ")
        for pool in r["pools"].values():
            ranking(pool, BASELINE)
    return ordered


def read_rows(path, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    if end != len(data):
        if not repair:
            raise ValueError("interrupted prediction tail; resume scoring first")
        with path.open("r+b") as stream:
            stream.truncate(end)
    rows = [json.loads(line) for line in data[:end].splitlines()]
    if len({r["input"]["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate prediction rows")
    return rows


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(args):
    seal = json.loads(args.evaluation_manifest.read_text())
    for path in (args.dev_cases, args.sealed_inputs, args.sealed_labels):
        if seal.get("files", {}).get(path.name) != digest(path):
            raise ValueError("evaluation data changed after sealing: " + str(path))
    files = [args.evaluation_manifest, args.dev_cases, args.sealed_inputs, args.sealed_labels, args.table,
             Path(__file__), Path(__file__).with_name("japanese.py"), Path(__file__).with_name("rescorer.py")]
    for root in (args.zenz, args.rescorer):
        files.extend(sorted(p for p in root.rglob("*") if p.is_file()
                            and p.suffix in (".json", ".safetensors", ".model", ".txt")))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("no model weights found")
    return dict(version=1, files={str(p.resolve()): digest(p) for p in files},
                protocol=dict(beams=8, left_chars=40, right_chars=40, grid=GRID, baseline=BASELINE,
                              qwen_context="production last 64 chars and Japanese neutral prefix",
                              score="zenz summed logprob including EOS; Qwen summed logprob without EOS",
                              timing="paired generation/scoring, pooled Qwen batch; not interactive latency",
                              validation="production Table.spells; no unvalidated model candidates",
                              conditions=CONDITIONS),
                runtime={name: importlib.metadata.version(name) for name in ("torch", "transformers")})


def require_frozen(out, manifest):
    path = out / "selection.json"
    if not path.exists():
        raise ValueError("freeze dev settings before accessing sealed predictions")
    selection = json.loads(path.read_text())
    # JSON roundtrip normalizes tuple-valued protocol fields.
    if selection["manifest"] != json.loads(json.dumps(manifest)):
        raise ValueError("inputs, model, protocol or source changed after freeze")
    if selection["dev_rows_sha256"] != digest(out / "dev.rows.jsonl"):
        raise ValueError("dev predictions changed after freeze")
    return selection


def score_cases(args, cases, split):
    path = args.out / f"{split}.rows.jsonl"
    rows = read_rows(path, repair=True)
    expected = {c["id"]: c for c in cases}
    for r in rows:
        c = expected.get(r["input"]["id"])
        if c is None or r["input"] != public_input(c):
            raise ValueError("saved prediction inputs differ")
    done = {r["input"]["id"] for r in rows}
    if len(done) == len(cases):
        align_rows(cases, rows)
        return
    rescorer = Rescorer(args.rescorer, .6)
    decoder = Decoder(Table(args.table, rescorer.tokenizer), rescorer, None,
                      zenz=RightZenz(args.zenz, beams=8))
    inspect(dict(id="warmup", split=split, context="", kana="にほんご", right_context=""), decoder)
    start = time.monotonic()
    with path.open("a", buffering=1) as stream:
        for c in cases:
            if c["id"] in done:
                continue
            row = inspect(public_input(c), decoder)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            rows.append(row)
            done.add(c["id"])
            if len(done) % 10 == 0 or len(done) == len(cases):
                print(json.dumps(dict(split=split, done=len(done), total=len(cases),
                                      elapsed_s=round(time.monotonic() - start))), flush=True)
    align_rows(cases, rows)


def paired_p(wins, losses):
    n = wins + losses
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.


def metrics(cases, rows, condition, setting):
    values = dict(total=len(cases), top1=0, top5=0, exact_top1=0, raw_pool=0,
                  valid_pool=0, no_valid_pool=0, wins=0, losses=0)
    for c, r in zip(cases, rows):
        pool = r["pools"][condition]
        candidates = displayed(c, pool, setting)
        rank = hit(c["answers"], candidates)
        good = rank == 1
        baseline = hit(c["answers"], displayed(c, r["pools"]["none"], BASELINE)) == 1
        values["top1"] += good
        values["top5"] += rank is not None
        values["exact_top1"] += hit(c["answers"], candidates, False) == 1
        values["raw_pool"] += hit(c["answers"], pool["generated"]) is not None
        values["valid_pool"] += hit(c["answers"], pool["cands"]) is not None
        values["no_valid_pool"] += not pool["cands"]
        values["wins"] += good and not baseline
        values["losses"] += baseline and not good
    values["paired_p_uncorrected"] = paired_p(values["wins"], values["losses"])
    return values


def summary(cases, rows, selection):
    rows = align_rows(cases, rows)
    settings = dict(zenz=(0., 0), baseline=BASELINE,
                    dev_selected=(selection["weight"], selection["length_power"]))
    groups = {"all": list(range(len(cases)))}
    for field in ("domain", "case_type"):
        for name in sorted({c.get(field, "") for c in cases}):
            groups[field + "/" + name] = [i for i, c in enumerate(cases) if c.get(field, "") == name]
    groups["right_present"] = [i for i, c in enumerate(cases) if c.get("right_context")]
    groups["right_absent"] = [i for i, c in enumerate(cases) if not c.get("right_context")]
    return {group: {condition: {name: metrics([cases[i] for i in indices], [rows[i] for i in indices],
                                             condition, setting) for name, setting in settings.items()}
                    for condition in CONDITIONS} for group, indices in groups.items() if indices}


def compare(cases, rows, left_condition, left_setting, right_condition, right_setting):
    """Paired factor contrast; bootstrap whole topic groups, not paired sentences."""
    clusters = {}
    wins = losses = 0
    for case, row in zip(cases, rows):
        left = hit(case["answers"], displayed(case, row["pools"][left_condition], left_setting)) == 1
        right = hit(case["answers"], displayed(case, row["pools"][right_condition], right_setting)) == 1
        delta = int(left) - int(right)
        wins += delta > 0
        losses += delta < 0
        key = case.get("topic_group") or case["id"]
        group = clusters.setdefault(key, [0, 0])
        group[0] += delta
        group[1] += 1
    if not cases:
        return dict(total=0)
    values = list(clusters.values())
    rng = random.Random(20261028)
    draws = []
    for _ in range(2000):
        sample = [rng.choice(values) for _ in values]
        draws.append(100 * sum(x[0] for x in sample) / sum(x[1] for x in sample))
    draws.sort()
    return dict(total=len(cases), wins=wins, losses=losses, delta_pp=100*(wins-losses)/len(cases),
                topic_groups=len(values), cluster_bootstrap95_delta_pp=[draws[50], draws[1949]],
                interpretation="Exploratory paired contrast; 2000 topic-group bootstrap draws, fixed seed; intervals not adjusted for multiple comparisons")


def contrasts(cases, rows, selection):
    rows = align_rows(cases, rows)
    chosen = (selection["weight"], selection["length_power"])
    pairs = {"right_at_baseline": ("right", BASELINE, "none", BASELINE),
             "right_at_selected": ("right", chosen, "none", chosen),
             "right_rescore_at_baseline": ("right_rescore", BASELINE, "none", BASELINE),
             "ranking_without_right": ("none", chosen, "none", BASELINE),
             "ranking_with_right": ("right", chosen, "right", BASELINE)}
    subsets = {"all": list(range(len(cases))),
               "right_present": [i for i,c in enumerate(cases) if c.get("right_context")],
               "right_absent": [i for i,c in enumerate(cases) if not c.get("right_context")]}
    for kind in ("disambiguation", "control"):
        subsets[kind] = [i for i,c in enumerate(cases) if c.get("case_type") == kind]
    return {group: {name: compare([cases[i] for i in indices], [rows[i] for i in indices], *pair)
                    for name,pair in pairs.items()} for group,indices in subsets.items() if indices}


def read_labels(path, cases):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    if isinstance(data, list):
        ids = [r["id"] for r in data]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sealed labels")
        data = {r["id"]: r["answers"] for r in data}
    if set(data) != {c["id"] for c in cases}:
        raise ValueError("sealed labels differ from sealed input IDs")
    if any(not isinstance(a, list) or not a or not all(isinstance(t, str) and t for t in a)
           for a in data.values()):
        raise ValueError("sealed answers must be nonempty lists")
    return [dict(c, answers=data[c["id"]]) for c in cases]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("score-dev", "freeze", "score-sealed", "report"))
    for name in ("evaluation-manifest", "dev-cases", "sealed-inputs", "sealed-labels", "table", "rescorer", "zenz", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    dev = read_cases(args.dev_cases, "dev", labels=True)
    sealed = read_cases(args.sealed_inputs, "sealed", labels=False)
    check_partitions(dev, sealed)
    manifest = fingerprint(args)
    args.out.mkdir(parents=True, exist_ok=True)
    write_once(args.out / "manifest.json", json.loads(json.dumps(manifest)))
    if args.phase == "score-dev":
        if (args.out / "selection.json").exists():
            require_frozen(args.out, manifest)
        score_cases(args, dev, "dev")
    elif args.phase == "freeze":
        rows = read_rows(args.out / "dev.rows.jsonl")
        selection = dict(manifest=manifest, dev_rows_sha256=digest(args.out / "dev.rows.jsonl"),
                         selected=select_setting(dev, rows), label_access="sealed labels hashed, not parsed")
        write_once(args.out / "selection.json", json.loads(json.dumps(selection)))
        print(json.dumps(selection["selected"], ensure_ascii=False))
        write_once(args.out / "dev.summary.json", summary(dev, rows, selection["selected"]))
    elif args.phase == "score-sealed":
        require_frozen(args.out, manifest)
        write_once(args.out / "sealed.run.json", dict(selection_sha256=digest(args.out / "selection.json")))
        score_cases(args, sealed, "sealed")
    else:
        selection = require_frozen(args.out, manifest)
        expected_seal = dict(selection_sha256=digest(args.out / "selection.json"))
        if not (args.out / "sealed.run.json").exists() or json.loads(
                (args.out / "sealed.run.json").read_text()) != expected_seal:
            raise ValueError("sealed predictions are not bound to this frozen selection")
        rows = read_rows(args.out / "sealed.rows.jsonl")
        align_rows(sealed, rows)  # Require completed blind predictions before parsing labels.
        labeled = read_labels(args.sealed_labels, sealed)
        report = dict(completed=True, selection_sha256=digest(args.out / "selection.json"),
                      sealed_predictions_sha256=digest(args.out / "sealed.rows.jsonl"),
                      selected=selection["selected"],
                      caution="paired p values are uncorrected descriptive comparisons; no test-set setting selection",
                      dev=summary(dev, read_rows(args.out / "dev.rows.jsonl"), selection["selected"]),
                      sealed=summary(labeled, rows, selection["selected"]),
                      sealed_contrasts=contrasts(labeled, rows, selection["selected"]))
        write_once(args.out / "report.json", report)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
