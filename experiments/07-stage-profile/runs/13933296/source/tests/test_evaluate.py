"""Check parser behavior and hand-computed scores, including invalid predictions."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from evaluate import INVALID, LABELS, parse_prediction, score_predictions, select_sample


class EvaluationTests(unittest.TestCase):
    def test_punctuation_fix_does_not_guess_from_explanations(self):
        for text in ['Glioma.', ' glioma ', '\nGLIOMA.\n']:
            self.assertEqual(parse_prediction(text), 'glioma')
        for text in ['', 'This is glioma.', 'glioma or meningioma', 'not glioma',
                     'glioma..', 'glioma.\nmeningioma', 'notumorish']:
            self.assertEqual(parse_prediction(text), INVALID)

    def test_previous_eight_image_example(self):
        truths = [label for label in LABELS for _ in range(2)]
        predictions = ['glioma'] * 4 + ['notumor'] + ['glioma'] * 3
        metrics = score_predictions([dict(true_label=t, prediction=p) for t, p in zip(truths, predictions)])
        self.assertEqual(metrics['accuracy'], 3 / 8)
        self.assertTrue(math.isclose(metrics['macro_f1'], 5 / 18))
        self.assertEqual(metrics['invalid_count'], 0)
        self.assertEqual([metrics['per_class'][x]['recall'] for x in LABELS], [1, 0, .5, 0])

    def test_invalid_is_a_false_negative_and_not_dropped(self):
        results = [dict(true_label=label, prediction=label) for label in LABELS]
        results.append(dict(true_label='glioma', prediction=INVALID))
        metrics = score_predictions(results)
        self.assertEqual(metrics['n'], 5)
        self.assertEqual(metrics['accuracy'], .8)
        self.assertEqual(metrics['invalid_rate'], .2)
        self.assertEqual(metrics['per_class']['glioma']['recall'], .5)
        self.assertTrue(math.isclose(metrics['macro_f1'], 11 / 12))
        self.assertEqual(metrics['confusion_matrix']['counts'][0], [1, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            score_predictions([])

    def test_sample_is_balanced_reproducible_and_input_order_independent(self):
        rows = [dict(path=f'{label}/{i}', label=label) for label in LABELS for i in range(10)]
        selected = select_sample(rows, 2, 42)
        self.assertEqual(selected, select_sample(list(reversed(rows)), 2, 42))
        self.assertEqual([sum(r['label'] == label for r in selected) for label in LABELS], [2] * 4)
        self.assertEqual(len(select_sample(rows, 0, 42)), 40)


if __name__ == '__main__':
    unittest.main()
