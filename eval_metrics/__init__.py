"""llm-eval-metrics: lightweight, dependency-free evaluation metrics for LLM output."""

from .metrics import (
    normalize_text,
    exact_match,
    token_f1,
    rouge_l,
    bleu,
    is_valid_json,
    json_keys_match,
    evaluate,
)

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

__version__ = "0.1.0"
