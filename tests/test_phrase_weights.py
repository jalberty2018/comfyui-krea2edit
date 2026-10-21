"""Model-free mapping and additive-mask checks (NumPy, no GPU required)."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

p = load("phrase_weights")
parser = load("phrase_parser")

class Tensor(np.ndarray):
    def new_zeros(self, shape):
        return np.zeros(shape).view(Tensor)
    def clone(self):
        return self.copy()

class PhraseTests(unittest.TestCase):
    def fixture(self, images):
        template = "SYSTEM USER " + "I" * images + "{} END"
        clean, spans = parser.parse_phrases("make (red:2)")
        prefix = template.index("{}")
        ids = [151644, 9, 151644, 872, 198] + [151655] * images + [11, 12, 13]
        offsets = [(0, 0)] * (5 + images) + [(prefix, prefix + 5),
            (prefix + 5, prefix + 8), (prefix + 8, prefix + 12)]
        pairs = [(i, 1) for i in ids]
        for i in range(images):
            pairs[5 + i] = ({"type": "image"}, 1)
        tokens = {"qwen3vl_4b": [pairs]}
        tokenizer = lambda *a, **kw: {"input_ids": ids, "offset_mapping": offsets}
        return clean, spans, template, tokens, tokenizer

    def test_zero_one_two_images_and_expansion(self):
        for images in (0, 1, 2):
            layout = p.map_phrases(*self.fixture(images))
            count = 3 + images * 64
            biases, report = p.finish_mapping(layout, count)
            self.assertEqual(report[0]["rows"], [count - 2])
            self.assertAlmostEqual(biases[0][1], np.log(2))

    def test_tokenizer_mismatch(self):
        args = self.fixture(1)
        args[3]["qwen3vl_4b"][0][-1] = (999, 1)
        with self.assertRaises(ValueError):
            p.map_phrases(*args)

    def test_wrong_length(self):
        layout = p.map_phrases(*self.fixture(0))
        with self.assertRaises(ValueError):
            p.finish_mapping(layout, 4)

    def test_neutral_and_zero(self):
        layout = p.map_phrases(*self.fixture(2))
        layout["phrases"][0]["weight"] = 1
        self.assertEqual(p.finish_mapping(layout, 131)[0], ())
        layout["phrases"][0]["weight"] = 0
        self.assertEqual(p.finish_mapping(layout, 131)[0][0][1], -80)

    def test_bias_preserves_reference_boost_and_non_target_rows(self):
        original = np.zeros((1, 1, 7, 7)).view(Tensor)
        original[:, :, 5:, 3:5] = np.log(3)
        combined = np.zeros((1, 7, 8)).view(Tensor)
        result = p.apply_phrase_bias(original, {"count": 3, "biases": [(1, np.log(2))]},
                                     3, 2, 2, combined)
        np.testing.assert_array_equal(result[:, :, :5], original[:, :, :5])
        np.testing.assert_array_equal(result[:, :, :, 3:], original[:, :, :, 3:])
        self.assertEqual(original[0, 0, 5, 1], 0)
        self.assertAlmostEqual(np.exp(result[0, 0, 5, 1]), 2)

    def test_missing_hook_noop(self):
        self.assertIsNone(p.apply_phrase_bias(None, None, 3, 2, 2, None))

    def test_changed_rows_rejected(self):
        with self.assertRaises(ValueError):
            p.apply_phrase_bias(None, {"count": 4, "biases": [(1, 1)]}, 3, 2, 2, None)

if __name__ == "__main__":
    unittest.main()
