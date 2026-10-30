"""Reviewed overlays on frozen evaluation sets: accepted variants, dropped boundaries.

Problems live in the person-facing review store (dataset_browser.human_problems);
this module adds what that store cannot express. A prediction is correct when it
matches the reference or any accepted variant at the same span.
"""
import json
from pathlib import Path

REVIEW = Path(__file__).resolve().parent.parent / "datasets/artifacts/evaluation-review-v1/decisions.json"


def overlay(dataset, path=REVIEW):
    """{case id: decision} for DATASET; empty without a review."""
    return json.loads(path.read_text())["datasets"].get(dataset, {}) if path.exists() else {}


def golds(decision, gold):
    """The reference first, then every accepted variant."""
    return list(gold) + (decision or {}).get("alternatives", [])


def accepted(prediction, references, start=0, end=None):
    """Whether PREDICTION equals some reference's [start:end] span."""
    return any(prediction == text[start:end] for text in references)


def keeps_boundaries(decision):
    return (decision or {}).get("status") != "remove_boundary"
