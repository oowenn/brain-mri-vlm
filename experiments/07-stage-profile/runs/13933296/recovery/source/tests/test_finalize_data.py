"""Policy tests do not require images or third-party packages."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('finalize_data', Path(__file__).parents[1] / 'scripts/finalize_data.py')
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)
CONFIG = json.loads((Path(__file__).parents[1] / 'configs/data_policy.json').read_text())


class FinalizationTests(unittest.TestCase):
    def fixture(self):
        rows, mapping = [], []
        for label_id, label in enumerate(CONFIG['labels']):
            for index in range(5):
                path = f'{label}/{index}'
                rows.append(dict(path=path, label=label, label_id=str(label_id),
                                 original_split='test' if index == 4 else 'train',
                                 split='test' if index == 4 else 'train',
                                 group_id=path, sha256=path, exclusion_reason=''))
                mapping.append(dict(path=path, label=label, original_split=rows[-1]['original_split'],
                                    status='unmatched', patient_key='', source_label='', label_disagreement='',
                                    correlation_64='', correlation_256='', other_patient_correlation=''))
        return rows, mapping

    def confirm(self, mapping, path, key, source=None):
        row = next(r for r in mapping if r['path'] == path)
        source = source or row['label']
        row.update(status='confirmed', patient_key=key * 64, source_label=source,
                   label_disagreement=str(source != row['label']), correlation_64='0.9999',
                   correlation_256='0.9998', other_patient_correlation='0.85')

    def test_patient_test_constraint_annotation_quarantine_and_unknown_identity(self):
        rows, mapping = self.fixture()
        self.confirm(mapping, 'glioma/0', 'a')
        self.confirm(mapping, 'glioma/4', 'a')
        # Transitive image constraint excludes the unknown image, without assigning an ID.
        rows[1]['group_id'] = rows[0]['group_id']
        mapping[1]['status'] = 'needs_review'
        self.confirm(mapping, 'pituitary/0', 'b', 'meningioma')
        self.confirm(mapping, 'pituitary/4', 'b', 'meningioma')
        result = data.apply_policy(rows, mapping, CONFIG)
        by_path = {r['path']: r for r in result}
        self.assertEqual(by_path['glioma/0']['exclusion_reason'], 'connected_to_test')
        self.assertEqual(by_path['glioma/1']['split'], 'excluded')
        self.assertEqual(by_path['glioma/1']['patient_key'], '')
        self.assertEqual(by_path['glioma/1']['patient_status'], 'needs_review')
        self.assertEqual(by_path['pituitary/0']['exclusion_reason'], 'source_label_disagreement')
        self.assertEqual(by_path['pituitary/4']['split'], 'test')
        self.assertEqual(by_path['pituitary/4']['label'], 'pituitary')
        self.assertEqual(by_path['pituitary/4']['source_label_disagreement'], 'True')
        self.assertEqual(result, data.apply_policy(list(reversed(rows)), list(reversed(mapping)), CONFIG))
        self.assertEqual(sum(r['split'] == 'test' for r in result), 4)

    def test_same_patient_stays_together_and_mixed_labels_are_quarantined(self):
        rows, mapping = self.fixture()
        self.confirm(mapping, 'glioma/0', 'a')
        self.confirm(mapping, 'glioma/1', 'a')
        rows[10]['group_id'] = rows[5]['group_id']  # notumor/0 and meningioma/0
        result = {r['path']: r for r in data.apply_policy(rows, mapping, CONFIG)}
        self.assertEqual(result['glioma/0']['split'], result['glioma/1']['split'])
        for path in ['notumor/0', 'meningioma/0']:
            self.assertEqual(result[path]['exclusion_reason'], 'conflicting_train_labels')

    def test_reject_incomplete_or_unsafe_mapping(self):
        rows, mapping = self.fixture()
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            data.apply_policy(rows, mapping[:-1], CONFIG)
        mapping[0]['patient_key'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'remain unknown'):
            data.apply_policy(rows, mapping, CONFIG)
        self.confirm(mapping, 'glioma/0', 'a')
        mapping[0]['correlation_256'] = '0.9949'
        with self.assertRaisesRegex(ValueError, 'Invalid confirmed'):
            data.apply_policy(rows, mapping, CONFIG)

    def test_pinned_inputs_immutable_outputs_and_reproducibility(self):
        rows, mapping = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / 'audit'
            prepared.mkdir()
            data.write_csv(prepared / 'inventory.csv', rows, list(rows[0]))
            data.write_csv(root / 'mapping.csv', mapping, list(mapping[0]))
            config = copy.deepcopy(CONFIG)
            config['input_inventory_sha256'] = data.digest((prepared / 'inventory.csv').read_bytes())
            config['patient_mapping_sha256'] = data.digest((root / 'mapping.csv').read_bytes())
            summary = dict(source_inventory_sha256=config['source_inventory_sha256'],
                           output_sha256={'inventory.csv': config['input_inventory_sha256']})
            (prepared / 'summary.json').write_text(json.dumps(summary))
            first = data.finalize(prepared, root / 'mapping.csv', root / 'first', config)
            second = data.finalize(prepared, root / 'mapping.csv', root / 'second', config)
            self.assertEqual(first, second)
            self.assertEqual(first['known_patient_cross_split_count'], 0)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                data.finalize(prepared, root / 'mapping.csv', root / 'first', config)
            with (root / 'mapping.csv').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'Pinned input checksum'):
                data.finalize(prepared, root / 'mapping.csv', root / 'third', config)
            self.assertFalse((root / 'third').exists())
