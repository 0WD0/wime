"""Replay archived IME cases with their actual auxiliary and committed keys."""
from collections import Counter, defaultdict
import json
from statistics import median
from composition import validate_committed


def typed_keys(raw):
    keys, auxiliary = [], {}
    for char in raw:
        if "A" <= char <= "Z":
            if not keys:
                raise ValueError("auxiliary key precedes the first syllable")
            position = (len(keys) - 1) // 2
            auxiliary[position] = auxiliary.get(position, "") + char.lower()
        elif char in "abcdefghijklmnopqrstuvwxyz;":
            keys.append(char)
        else:
            raise ValueError(f"unsupported typed key: {char!r}")
    return "".join(keys), list(auxiliary.items())


def requests(cases):
    """Adapt old archives explicitly; new rows already separate history/commit."""
    controls = {row.get("parent", row["id"]): row for row in cases if row.get("variant") == "whole"}
    for row in cases:
        context, committed = row["context"], row.get("committed")
        code, aux = typed_keys(row["input"])
        if row.get("committed_text"):
            committed = dict(text=row["committed_text"], input=row["committed_input"])
        elif row.get("context_reading"):
            committed = row["context_reading"]
            if not context.endswith(committed["text"]):
                raise ValueError("archived reading does not match visible context")
            context = context[:-len(committed["text"])]
        control = row.get("whole_control") or controls.get(row.get("parent"))
        if committed is None and control is not None and control is not row:
            # Recover only the prefix already visible in the archived state.
            full_code, _ = typed_keys(control["input"])
            for end in range(1, len(control["gold"][0])):
                prefix = control["gold"][0][:end]
                if (context == (control["context"] + prefix)[-64:]
                        and full_code[2 * end:].startswith(code)):
                    committed = dict(text=prefix, input=full_code[:2*end])
                    context = control["context"]
                    break
        validate_committed(committed)
        yield dict(context=context, input=code, committed=committed, aux=aux)


def request_key(request):
    return json.dumps(request, sort_keys=True, ensure_ascii=False)


def compare(rows, names):
    left, right = names
    def scores(group):
        result = dict(cases=len(group),
                      units=len({row.get("document") or row.get("parent") or row["id"] for row in group}),
                      models={name:dict(top1=sum(row["models"][name]["prediction"] in row["gold"] for row in group),
                                        top5=sum(any(text in row["gold"] for text in row["models"][name]["candidates"][:5]) for row in group),
                                        no_candidates=sum(not row["models"][name]["candidates"] for row in group),
                                        median_ms=median(row["models"][name]["milliseconds"] for row in group)) for name in names})
        changes = Counter()
        for row in group:
            a, b = (row["models"][name]["prediction"] in row["gold"] for name in names)
            changes["both_correct" if a and b else "both_wrong" if not a and not b else "improved" if b else "regressed"] += 1
        result["paired"] = {key:changes[key] for key in ("improved", "regressed", "both_correct", "both_wrong")}
        return result
    datasets, variants, categories = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        datasets[row["dataset"]].append(row)
        variants[row["dataset"] + "/" + row.get("variant", "original")].append(row)
        categories[row["dataset"] + "/" + row.get("category", "unspecified")].append(row)
    return dict(datasets={key:scores(group) for key,group in datasets.items()},
                variants={key:scores(group) for key,group in variants.items()},
                categories={key:scores(group) for key,group in categories.items()})
