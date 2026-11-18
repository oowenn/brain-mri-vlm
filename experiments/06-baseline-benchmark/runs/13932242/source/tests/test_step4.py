"""Guard full-split coverage and independent verification of saved results."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from evaluate import evaluation_protocol, score_predictions
from verify_step4 import LABELS, SUPPORTS, verify_predictions


class Step4Tests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(path=f'{label}/{i:03}', label=label, split='validation',
                          sha256='0' * 64, group_id=f'{label}/{i:03}')
                     for label in LABELS for i in range(SUPPORTS[label])]
        self.selected = [{k: row[k] for k in ('path', 'label', 'sha256', 'group_id')} for row in self.rows]
        self.predictions = [dict(path=row['path'], true_label=row['label'],
                                 prediction=row['label'], raw_response=row['label'].title() + '.')
                            for row in self.rows]
        self.predictions[-1].update(prediction='INVALID', raw_response='uncertain')
        self.metrics = score_predictions(self.predictions)

    def test_config_preserves_prior_inference_protocol(self):
        repo = Path(__file__).parents[1]
        previous = json.loads((repo / 'configs/step3.json').read_text())
        current = json.loads((repo / 'configs/step4.json').read_text())
        self.assertEqual(current, {**previous, 'stage': 'step4', 'per_class': 0})
        self.assertEqual(evaluation_protocol(current, 'fp16', None), ('step4', 0, 'zero-shot-validation'))
        self.assertEqual(evaluation_protocol(previous, 'int4', None), ('step3', 2, 'step3-compatibility'))
        for precision, count in [('int8', None), ('int4', None), ('fp16', 2)]:
            with self.assertRaises(ValueError):
                evaluation_protocol(current, precision, count)

    def test_independent_metrics_keep_invalid_response(self):
        result = verify_predictions(self.rows, self.predictions, self.selected, self.metrics)
        self.assertEqual(result['n'], 579)
        self.assertEqual(result['accuracy'], 578 / 579)
        self.assertEqual(result['invalid_rate'], 1 / 579)

    def test_missing_duplicate_or_wrong_split_rejected(self):
        for bad in [self.predictions[:-1], self.predictions[:-1] + [self.predictions[0]]]:
            with self.assertRaises(ValueError):
                verify_predictions(self.rows, bad, self.selected, self.metrics)
        rows = copy.deepcopy(self.rows)
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            verify_predictions(rows, self.predictions, self.selected, self.metrics)

    def test_raw_answer_truth_or_manifest_tampering_rejected(self):
        for field, value in [('raw_response', 'It could be glioma.'), ('true_label', 'pituitary')]:
            bad = copy.deepcopy(self.predictions)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                verify_predictions(self.rows, bad, self.selected, self.metrics)
        selected = copy.deepcopy(self.selected)
        selected[0]['sha256'] = '1' * 64
        with self.assertRaises(ValueError):
            verify_predictions(self.rows, self.predictions, selected, self.metrics)

    def test_altered_metrics_rejected(self):
        for value in [1.0, float('nan')]:
            metrics = copy.deepcopy(self.metrics)
            metrics['accuracy'] = value
            with self.assertRaises(ValueError):
                verify_predictions(self.rows, self.predictions, self.selected, metrics)


if __name__ == '__main__':
    unittest.main()
