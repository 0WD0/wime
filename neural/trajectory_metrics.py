"""Sentence-weighted metrics for complete input and paired commit trajectories."""
from collections import Counter, defaultdict
import random
from statistics import mean

BOOTSTRAP_SEED = 20261029
BOOTSTRAP_SAMPLES = 4000


def _average(values):
    return mean(values) if values else None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _outcome(state, gold):
    prediction = state["prediction"]
    error = state.get("error")
    return dict(correct=not error and prediction == gold,
                top5=not error and gold in state["candidates"][:5],
                characters=(sum(a == b for a, b in zip(prediction, gold)) /
                            max(len(prediction), len(gold))) if not error else 0.0,
                milliseconds=state["milliseconds"], error=error)


def _stage(groups):
    present = [group for group in groups if group]
    outcomes = [value for group in present for value in group]
    times = [value["milliseconds"] for value in outcomes if value["milliseconds"] is not None]
    return dict(sentences=len(present), requests=len(outcomes),
                exact_correct=sum(value["correct"] for value in outcomes),
                top5_correct=sum(value["top5"] for value in outcomes),
                exact_accuracy=_average([mean(value["correct"] for value in group) for group in present]),
                top5_accuracy=_average([mean(value["top5"] for value in group) for group in present]),
                character_accuracy=_average([mean(value["characters"] for value in group) for group in present]),
                errors=sum(bool(value["error"]) for value in outcomes),
                error_counts=dict(Counter(value["error"] for value in outcomes if value["error"])),
                request_latency_ms=dict(measured=len(times), median=_percentile(times, .5),
                                        p95=_percentile(times, .95)))


def _bootstrap(differences):
    if not differences:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    means = [mean(rng.choices(differences, k=len(differences))) for _ in range(BOOTSTRAP_SAMPLES)]
    return [_percentile(means, .025), _percentile(means, .975)]


def _paired(groups, before_name, after_name, gain="helped", loss="harmed"):
    present = [group for group in groups if group]
    labels = (gain, loss, "both_correct", "both_wrong")

    def category(before, after):
        return "both_correct" if before and after else "both_wrong" if not before and not after else gain if after else loss

    classified = [[category(before, after) for before, after in group] for group in present]
    counts = Counter(label for group in classified for label in group)
    differences = [mean(int(after) - int(before) for before, after in group) for group in present]
    return dict(sentences=len(present), comparisons=sum(map(len, present)),
                **{before_name: _average([mean(before for before, _ in group) for group in present]),
                   after_name: _average([mean(after for _, after in group) for group in present])},
                counts={label: counts[label] for label in labels},
                rates={label: _average([group.count(label) / len(group) for group in classified]) for label in labels},
                delta_accuracy=_average(differences), delta_95_ci=_bootstrap(differences))


def _aggregate(rows):
    full, prefixes, continuations, suffix_pairs, recoveries, selectable = [], [], [], [], [], []
    for row in rows:
        gold = row["gold"][0]
        whole = row["full"]
        full.append([_outcome(whole, gold)])
        prefix_group, continuation_group, suffix_group, recovery_group, selectable_group = [], [], [], [], []
        for boundary in row["prefixes"]:
            end = boundary["end"]
            prefix = _outcome(boundary["prefix"], gold[:end])
            continuation = _outcome(boundary["continuation"], gold[end:])
            # Errors must fail even if a partial/stale prediction matches a slice.
            full_prefix = not whole.get("error") and whole["prediction"][:end] == gold[:end]
            full_suffix = not whole.get("error") and whole["prediction"][end:] == gold[end:]
            pair = (full_suffix, continuation["correct"])
            prefix_group.append(prefix)
            continuation_group.append(continuation)
            suffix_group.append(pair)
            recovery_group.append((prefix["correct"], full_prefix))
            if prefix["correct"]:
                selectable_group.append(pair)
        prefixes.append(prefix_group)
        continuations.append(continuation_group)
        suffix_pairs.append(suffix_group)
        recoveries.append(recovery_group)
        selectable.append(selectable_group)
    whole, prefix, continuation = _stage(full), _stage(prefixes), _stage(continuations)
    return dict(
        coverage=dict(sentences=len(rows), documents=len({row["document"] for row in rows}),
                      groups=len({row["group"] for row in rows}),
                      sentences_with_prefixes=prefix["sentences"], sentences_without_prefixes=len(rows)-prefix["sentences"],
                      boundaries=prefix["requests"], requests=sum(stage["requests"] for stage in (whole, prefix, continuation)),
                      errors=sum(stage["errors"] for stage in (whole, prefix, continuation))),
        whole=whole, prefix=prefix, continuation=continuation,
        continuation_pair=_paired(suffix_pairs, "uncommitted_suffix_accuracy", "committed_suffix_accuracy"),
        prefix_recovery=_paired(recoveries, "early_prefix_accuracy", "full_input_prefix_accuracy", "recovered", "regressed"),
        correct_prefix_only=_paired(selectable, "uncommitted_suffix_accuracy", "committed_suffix_accuracy"))


def summarize(results):
    """Summarize one result per original sentence, retaining failed requests.

    Every accuracy/rate averages within each sentence before averaging across
    sentences. Counts and latency percentiles describe individual requests.
    A sentence without a boundary remains in whole-sentence coverage; it has no
    prefix/continuation observations. Bootstrap resamples paired sentence means.
    """
    rows = list(results)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("trajectory results must contain each sentence once")
    sources, lengths = defaultdict(list), defaultdict(list)
    for row in rows:
        size = len(row["gold"][0])
        ends = [boundary["end"] for boundary in row["prefixes"]]
        if not size or len(set(ends)) != len(ends) or any(not 0 < end < size for end in ends):
            raise ValueError("prefix boundaries must be distinct positions inside a nonempty sentence")
        sources[row["source"]].append(row)
        lengths["1-5" if size < 6 else "6-12" if size <= 12 else "13-24" if size <= 24 else "25+"].append(row)
    return dict(
        definitions=dict(
            weighting="Accuracy and paired rates: mean within each sentence, then equal mean across eligible sentences. Counts are unweighted.",
            character_accuracy="Position-aligned matching characters divided by max(gold length, prediction length); errors score zero.",
            continuation_pair="Same gold suffix: full uncommitted prediction after the boundary versus continuation after an oracle-correct committed prefix.",
            prefix_recovery="The same prefix before and after typing the full sentence; recovered means wrong early, correct with full input.",
            correct_prefix_only="Continuation pairs restricted to boundaries whose early top-1 prefix was correct; sentences with none are counted outside this subset.",
            errors="Unsupported inputs and other request errors count as incorrect, including top-5 and character accuracy.",
            confidence_interval=dict(method="sentence-level paired percentile bootstrap", seed=BOOTSTRAP_SEED,
                                     samples=BOOTSTRAP_SAMPLES, level=.95)),
        overall=_aggregate(rows),
        by_source={name: _aggregate(group) for name, group in sorted(sources.items())},
        by_length={name: _aggregate(group) for name, group in lengths.items()})
