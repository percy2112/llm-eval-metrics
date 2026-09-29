"""Lightweight text-generation evaluation metrics.

These are compact, well-understood reference implementations with no third-party
dependencies, suitable for unit tests, CI gates, and quick offline evaluation of
LLM outputs. They intentionally favour clarity and determinism over squeezing out
the last fraction of a point versus reference toolkits.

Included:
    * ``exact_match``  - normalized string equality (SQuAD-style normalization)
    * ``token_f1``     - unigram overlap F1
    * ``rouge_l``      - longest-common-subsequence F1
    * ``bleu``         - sentence-level BLEU up to 4-grams with smoothing
    * ``is_valid_json``/``json_keys_match`` - structured-output checks
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from typing import Callable, Dict, Iterable, List, Sequence

__all__ = [
    "normalize_text",
    "exact_match",
    "token_f1",
    "rouge_l",
    "bleu",
    "is_valid_json",
    "json_keys_match",
    "evaluate",
]

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation/articles, collapse space."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _tokens(text: str) -> List[str]:
    return normalize_text(text).split()


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 if normalized strings are identical, else 0.0."""
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Unigram-overlap F1 over normalized tokens (SQuAD F1)."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 based on longest common subsequence of tokens."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Sentence-level BLEU with add-1 smoothing and a brevity penalty.

    Returns a score in ``[0, 1]``. Uses clipped n-gram precision for ``n`` up to
    ``max_n`` and the standard brevity penalty.
    """
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    log_precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(pred_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = max(sum(pred_ngrams.values()), 0)
        # Add-1 smoothing keeps the geometric mean defined for short sentences.
        precision = (overlap + 1) / (total + 1)
        log_precisions.append(math.log(precision))

    geo_mean = math.exp(sum(log_precisions) / max_n)
    brevity = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))
    return brevity * geo_mean


def is_valid_json(text: str) -> float:
    """Return 1.0 if ``text`` parses as JSON, else 0.0."""
    try:
        json.loads(text)
        return 1.0
    except (ValueError, TypeError):
        return 0.0


def json_keys_match(text: str, required_keys: Iterable[str]) -> float:
    """Return 1.0 if ``text`` is a JSON object containing all ``required_keys``."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return 0.0
    if not isinstance(obj, dict):
        return 0.0
    return float(all(key in obj for key in required_keys))


_METRICS: Dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
    "bleu": bleu,
}


def evaluate(
    predictions: Sequence[str],
    references: Sequence[str],
    metrics: Iterable[str] = ("exact_match", "token_f1", "rouge_l"),
) -> Dict[str, float]:
    """Average the requested metrics over aligned prediction/reference pairs."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not predictions:
        return {name: 0.0 for name in metrics}

    results: Dict[str, float] = {}
    for name in metrics:
        fn = _METRICS[name]
        results[name] = sum(fn(p, r) for p, r in zip(predictions, references)) / len(
            predictions
        )
    return results
