import unittest

from eval_metrics import (
    normalize_text,
    exact_match,
    token_f1,
    rouge_l,
    bleu,
    is_valid_json,
    json_keys_match,
    evaluate,
)


class TestNormalize(unittest.TestCase):
    def test_removes_articles_punct_case(self):
        self.assertEqual(normalize_text("The Cat, sat!"), "cat sat")


class TestExactMatch(unittest.TestCase):
    def test_match_after_normalization(self):
        self.assertEqual(exact_match("The Answer.", "answer"), 1.0)

    def test_mismatch(self):
        self.assertEqual(exact_match("cat", "dog"), 0.0)


class TestTokenF1(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(token_f1("hello world", "hello world"), 1.0)

    def test_partial(self):
        # "x y z" vs "x y w": 2 shared of 3 each -> F1 = 2/3.
        self.assertAlmostEqual(token_f1("x y z", "x y w"), 2 / 3)

    def test_disjoint(self):
        self.assertEqual(token_f1("cat", "dog"), 0.0)

    def test_both_empty(self):
        self.assertEqual(token_f1("", ""), 1.0)


class TestRougeL(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(rouge_l("a b c d", "a b c d"), 1.0)

    def test_subsequence(self):
        # LCS of "a c" within both -> lcs=2
        score = rouge_l("a b c", "a x c")
        self.assertTrue(0 < score < 1)

    def test_disjoint(self):
        self.assertEqual(rouge_l("a b", "c d"), 0.0)


class TestBleu(unittest.TestCase):
    def test_identical_high(self):
        self.assertGreater(bleu("the cat sat on the mat", "the cat sat on the mat"), 0.9)

    def test_bounded(self):
        score = bleu("a b c", "a b d")
        self.assertTrue(0.0 <= score <= 1.0)

    def test_empty(self):
        self.assertEqual(bleu("", "something"), 0.0)


class TestJson(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(is_valid_json('{"a": 1}'), 1.0)

    def test_invalid(self):
        self.assertEqual(is_valid_json("{not json}"), 0.0)

    def test_keys_present(self):
        self.assertEqual(json_keys_match('{"a": 1, "b": 2}', ["a", "b"]), 1.0)

    def test_keys_missing(self):
        self.assertEqual(json_keys_match('{"a": 1}', ["a", "b"]), 0.0)

    def test_non_object(self):
        self.assertEqual(json_keys_match("[1, 2]", ["a"]), 0.0)


class TestEvaluate(unittest.TestCase):
    def test_average_over_pairs(self):
        preds = ["hello world", "foo"]
        refs = ["hello world", "bar"]
        result = evaluate(preds, refs, metrics=["exact_match"])
        self.assertEqual(result["exact_match"], 0.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            evaluate(["a"], ["a", "b"])

    def test_empty(self):
        self.assertEqual(evaluate([], [], metrics=["token_f1"]), {"token_f1": 0.0})


if __name__ == "__main__":
    unittest.main()
