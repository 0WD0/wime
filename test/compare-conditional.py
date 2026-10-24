#!/usr/bin/env python3
"""Paired before/after results; never feed gold or diagnostics into prediction."""
import argparse
import json
from pathlib import Path


def compare(before, after, single_policy="model"):
    previous = {r["id"]: r for r in before}
    if (set(previous) != {r["id"] for r in after} or len(previous) != len(before)
            or len(previous) != len(after)):
        raise ValueError("case sets must be identical and unique")
    pairs = []
    for row in after:
        old = previous[row["id"]]
        if any(old[k] != row[k] for k in ("input", "gold")):
            raise ValueError("changed input or label")
        if "context" in old and old["context"] != row["context"]:
            raise ValueError("changed context")
        if "beam" in old and old["beam"] != row["beam"]:
            raise ValueError("changed traditional search; not a model-only comparison")
        old_text = old.get("constrained", old.get("result"))
        # An explicitly labelled projection of the existing client switch:
        # skipping a single-character job preserves the initial beam leader.
        prediction = (row["beam"] if single_policy == "grammar" and row.get("syllables") == 1
                      else row["constrained"])
        pairs.append((old_text in row["gold"], prediction in row["gold"], row))

    def counts(items):
        return dict(total=len(items), before=sum(a for a,_,_ in items), after=sum(b for _,b,_ in items),
                    fixed=sum(b and not a for a,b,_ in items), regressed=sum(a and not b for a,b,_ in items),
                    same_width_octagram=sum(r["beam"] in r["gold"] for _,_,r in items))

    groups = {}
    for key in ("category", "variant"):
        groups[key] = {str(value): counts([p for p in pairs if p[2].get(key) == value])
                       for value in sorted({r.get(key) for r in after}, key=str)}
    return dict(single_character_policy=single_policy, overall=counts(pairs), groups=groups,
                fixed_ids=[r["id"] for a,b,r in pairs if b and not a],
                regressed_ids=[r["id"] for a,b,r in pairs if a and not b])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--single-policy", choices=("model", "grammar"), default="model",
                        help="grammar projects the default client gate onto audited experimental results")
    args = parser.parse_args()
    result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()), args.single_policy)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("fixed_ids", "regressed_ids")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
