# llm-eval-metrics

Lightweight, **dependency-free** evaluation metrics for LLM and NLG outputs.

Reference toolkits (NLTK, `rouge-score`, `sacrebleu`, `evaluate`) are great, but
pulling them into a small service or CI job is often overkill. `llm-eval-metrics`
gives you compact, deterministic, well-understood implementations in pure Python.

## Metrics

| Function | What it measures |
|----------|------------------|
| `exact_match` | Normalized string equality (SQuAD-style) |
| `token_f1` | Unigram overlap F1 |
| `rouge_l` | Longest-common-subsequence F1 |
| `bleu` | Sentence BLEU (up to 4-grams, smoothed, brevity penalty) |
| `is_valid_json` | Whether output parses as JSON |
| `json_keys_match` | Whether a JSON object has required keys |

## Install

```bash
pip install llm-eval-metrics      # once published
```

## Usage

```python
from eval_metrics import exact_match, token_f1, rouge_l, bleu, evaluate

exact_match("The Answer.", "answer")     # 1.0
token_f1("a b c", "a b d")               # 0.666...
rouge_l("the cat sat", "the cat ran")    # LCS-based F1
bleu("the cat sat on the mat", "the cat sat on the mat")  # ~0.9+

# Aggregate over a dataset
preds = ["paris", "the eiffel tower"]
refs  = ["Paris", "Eiffel Tower"]
evaluate(preds, refs, metrics=["exact_match", "token_f1", "rouge_l"])
# -> {'exact_match': 1.0, 'token_f1': 1.0, 'rouge_l': 1.0}
```

### Structured-output checks

```python
from eval_metrics import is_valid_json, json_keys_match

is_valid_json('{"answer": 42}')                       # 1.0
json_keys_match('{"answer": 42, "unit": "m"}', ["answer", "unit"])  # 1.0
```

## Notes

These are practical reference implementations optimized for clarity and
determinism. For competition-grade corpus BLEU, use `sacrebleu`.

## Development

```bash
python -m unittest discover -s tests -v
```

## License

MIT
