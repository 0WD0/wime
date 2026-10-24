#!/usr/bin/env python3
"""Post-hoc training coverage audit; never use these labels to sample training."""
import argparse
from collections import Counter, deque
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path


class PhraseMatcher:
    """Find all overlapping phrases, counting each at most once per row."""
    def __init__(self, patterns):
        self.edges, self.failure, self.outputs = [{}], [0], [set()]
        for i, word in enumerate(patterns):
            if not word:
                raise ValueError("empty phrase")
            state = 0
            for char in word:
                if char not in self.edges[state]:
                    self.edges[state][char] = len(self.edges)
                    self.edges.append({})
                    self.failure.append(0)
                    self.outputs.append(set())
                state = self.edges[state][char]
            self.outputs[state].add(i)
        queue = deque(self.edges[0].values())
        while queue:
            state = queue.popleft()
            for char, nxt in self.edges[state].items():
                queue.append(nxt)
                fallback = self.failure[state]
                while fallback and char not in self.edges[fallback]:
                    fallback = self.failure[fallback]
                self.failure[nxt] = self.edges[fallback].get(char, 0)
                self.outputs[nxt].update(self.outputs[self.failure[nxt]])

    def matches(self, text):
        state, seen = 0, set()
        for char in text:
            while state and char not in self.edges[state]:
                state = self.failure[state]
            state = self.edges[state].get(char, 0)
            seen.update(self.outputs[state])
        return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    patterns = set()
    for row in cases:
        a, b, spans = row["gold"][0], row["after"], []
        for op, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            # Contextual contrasts avoid counting only ubiquitous single Han
            # characters. Also retain changed spans of at least two characters.
            pair = [a[max(0, i - 1):j + 1], b[max(0, k - 1):l + 1]]
            spans.append(pair)
            patterns.update(pair)
            raw = [a[i:j], b[k:l]]
            if min(map(len, raw)) >= 2 and raw not in spans:
                spans.append(raw)
                patterns.update(raw)
        row["contrasts"] = spans
    patterns = sorted(patterns)
    matcher = PhraseMatcher(patterns)
    counts, metadata = {}, {}
    sources = dict(original_chat="data/train.jsonl", original_technical="programming/train.jsonl",
                   replay="replay/train.jsonl", new="diverse/train.jsonl")
    for name, relative in sources.items():
        path, counter, total, characters = args.root / relative, Counter(), 0, 0
        with path.open() as source:
            for line in source:
                text = json.loads(line)["gold"][0]
                counter.update(matcher.matches(text))
                total += 1
                characters += len(text)
        counts[name] = {patterns[i]: v for i, v in counter.items()}
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        metadata[name] = dict(rows=total, characters=characters, sha256=digest)
        print(name, total, flush=True)
    for row in cases:
        row["contrast_counts"] = [dict(gold=a, prediction=b,
            counts={name: dict(gold=c.get(a, 0), prediction=c.get(b, 0)) for name, c in counts.items()})
            for a, b in row.pop("contrasts")]
        row.pop("candidates")
    summary = dict(sources=metadata, cases=cases,
        note="Post-hoc target-row occurrence counts of multi-character differing spans and contextual spans "
             "with one character on each side; raw prepared rows before augmentation. Frequency is "
             "evidence of coverage, not proof of causation.")
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
