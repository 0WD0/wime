"""Training-only coverage selection and held-out overlap guards."""
from collections import Counter
import hashlib
import json
import re


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def phrases(row):
    """Keep the actual reading in each 2–4 character coverage feature."""
    text, code = row["gold"][0], row["input"]
    if len(code) != 2 * len(text):
        raise ValueError("unaligned training reading")
    return {(text[i:i+n], code[2*i:2*(i+n)])
            for n in (2, 3, 4) for i in range(len(text)-n+1)}


def coverage_additions(path, selected, limit, min_support=3):
    """Add rows covering supported phrases missed by uniform replay.

    Only the original training split determines priorities. Support counts
    are per row, and source order breaks ties deterministically. The minimum
    avoids spending the budget on one-off typos and random strings.
    """
    support = Counter()
    for row in rows(path):
        if row["split"] != "train":
            raise ValueError("coverage selection accepts training rows only")
        support.update(phrases(row))
    eligible = {word for word, count in support.items() if count >= min_support}
    covered = set().union(*(phrases(row) for row in selected)) if selected else set()
    missing = eligible - covered
    before = len(missing)
    identities = {row["id"] for row in selected}
    added = []
    for row in rows(path):
        if len(added) >= limit:
            break
        if row["id"] in identities:
            continue
        features = phrases(row)
        if missing.intersection(features):
            added.append(row)
            identities.add(row["id"])
            missing.difference_update(features)
    return added, dict(supported_phrases=len(eligible), missing_before=before,
                       missing_after=len(missing), additional_rows=len(added),
                       min_support=min_support)


class Protection:
    """Exclude held-out documents, context/target pairs and long overlaps."""
    def __init__(self, paths):
        self.documents, self.exact, self.fragments = set(), set(), set()
        self.files = {}
        for path in paths:
            self.files[str(path)] = fingerprint(path)
            raw = path.read_text()
            examples = json.loads(raw) if raw.lstrip().startswith("[") else map(json.loads, raw.splitlines())
            for row in examples:
                if row.get("document"):
                    self.documents.add(row["document"])
                context = self.normalize(row.get("context", ""))
                for gold in row["gold"]:
                    gold = self.normalize(gold)
                    self.exact.add((context[-64:], gold))
                    combined = context[-24:] + gold
                    if len(context) >= 8 and len(combined) >= 24:
                        self.fragments.add(combined[-24:])

    @staticmethod
    def normalize(text):
        return re.sub(r"[^\u4e00-\u9fff]", "", text)

    def contains(self, row):
        if row.get("document") in self.documents:
            return True
        context = self.normalize(row.get("context", ""))
        for gold in row["gold"]:
            gold = self.normalize(gold)
            if (context[-64:], gold) in self.exact:
                return True
            combined = context + gold
            if any(combined[i:i+24] in self.fragments for i in range(len(combined)-23)):
                return True
        return False
