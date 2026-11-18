"""Behavior checks for leakage prevention and portable, deterministic manifests."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest

from PIL import Image

SPEC = importlib.util.spec_from_file_location('prepare_data', Path(__file__).parents[1] / 'scripts/prepare_data.py')
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)
CONFIG = data.read_config(Path(__file__).parents[1] / 'configs/data.json')
CONFIG['expected_source_sha256'] = None  # Synthetic fixtures have their own source identity.


class DataPreparationTests(unittest.TestCase):
    def fixture(self, root):
        rng = random.Random(123)
        for split, count in [('Training', 6), ('Testing', 2)]:
            for label in CONFIG['labels']:
                folder = root / split / label
                folder.mkdir(parents=True)
                for index in range(count):
                    image = Image.frombytes('RGB', (48, 48), rng.randbytes(48 * 48 * 3))
                    image.save(folder / f'{index}.png')

    def test_transitive_links_and_inclusive_threshold(self):
        rows = [{'path': str(i), 'sha256': str(i), 'pixel_sha256': str(i),
                 'phash': f'{value:016x}', 'original_split': split, 'label': label,
                 '_signature': data.np.array([1.0, 0.0])}
                for i, (value, split, label) in enumerate([
                    (0, 'train', 'glioma'), (15, 'train', 'meningioma'), (31, 'test', 'pituitary')])]
        groups, pairs = data.group_duplicates(rows, 4, 0.98)
        self.assertEqual(groups, [[0, 1, 2]])  # 0 to 31 is distance 5 but linked through 15.
        self.assertEqual(len(pairs), 2)
        self.assertTrue(any(p['hamming_distance'] == 4 for p in pairs))
        self.assertFalse(any(p['path_a'] == '0' and p['path_b'] == '2' for p in pairs))

    def test_phash_collision_requires_similarity_confirmation(self):
        rows = [{'path': str(i), 'sha256': str(i), 'pixel_sha256': str(i),
                 'phash': '0' * 16, 'original_split': 'train', 'label': 'glioma',
                 '_signature': signature}
                for i, signature in enumerate([data.np.array([1.0, 0.0]), data.np.array([0.0, 1.0])])]
        groups, pairs = data.group_duplicates(rows, 4, 0.98)
        self.assertEqual(groups, [[0], [1]])
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0]['accepted'])
        self.assertEqual(pairs[0]['correlation'], 0.0)

    def test_brightness_adjusted_copy_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            rng = data.np.random.default_rng(19)
            pixels = rng.integers(20, 180, (48, 48, 3), dtype=data.np.uint8)
            Image.fromarray(pixels).save(root / 'Training/glioma/0.png')
            Image.fromarray(pixels + 20).save(root / 'Testing/glioma/0.png')
            rows = data.scan_source(root, CONFIG, 1)
            _, pairs = data.group_duplicates(rows, CONFIG['max_hamming_distance'], CONFIG['min_correlation'])
            target = next(p for p in pairs if {p['path_a'], p['path_b']} ==
                          {'Training/glioma/0.png', 'Testing/glioma/0.png'})
            self.assertEqual(target['match_kind'], 'phash')
            self.assertTrue(target['accepted'])

    def test_portable_manifests_no_leakage_and_unchanged_test(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / 'source'
            self.fixture(root)
            shutil.copyfile(root / 'Testing/pituitary/0.png', root / 'Training/glioma/leak.png')
            shutil.copyfile(root / 'Training/meningioma/1.png', root / 'Training/meningioma/copy.png')
            shutil.copyfile(root / 'Training/glioma/1.png', root / 'Training/meningioma/conflict.png')
            config = copy.deepcopy(CONFIG)
            config['max_hamming_distance'] = 0
            first = data.prepare(root, base / 'first', config, 1)
            self.assertEqual(first, json.loads(json.dumps(first)))  # W&B requires string keys.
            relocated = base / 'relocated'
            shutil.copytree(root, relocated)
            second = data.prepare(relocated, base / 'second', config, 2)
            self.assertEqual(first, second)
            for name in first['output_sha256']:
                self.assertEqual((base / 'first' / name).read_bytes(), (base / 'second' / name).read_bytes())
            with (base / 'first/inventory.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            by_path = {r['path']: r for r in rows}
            self.assertEqual(by_path['Training/glioma/leak.png']['exclusion_reason'], 'connected_to_test')
            self.assertEqual(by_path['Training/glioma/1.png']['exclusion_reason'], 'conflicting_train_labels')
            self.assertEqual(by_path['Training/meningioma/conflict.png']['split'], 'excluded')
            self.assertEqual(by_path['Training/meningioma/1.png']['split'], by_path['Training/meningioma/copy.png']['split'])
            self.assertEqual(sum(r['split'] == 'test' for r in rows), 8)
            self.assertEqual(sum(r['original_split'] == 'test' for r in rows), 8)
            for split in ['train', 'validation']:
                self.assertEqual({r['label'] for r in rows if r['split'] == split}, set(CONFIG['labels']))
            active = [r for r in rows if r['split'] != 'excluded']
            for i, a in enumerate(active):
                for b in active[:i]:
                    if a['split'] != b['split']:
                        self.assertNotEqual(a['sha256'], b['sha256'])
                        self.assertNotEqual(a['pixel_sha256'], b['pixel_sha256'])
                        self.assertNotEqual(a['phash'], b['phash'])
            self.assertEqual(len(list(root.rglob('*.png'))), len(rows))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                data.prepare(root, base / 'first', config, 1)
            config['expected_source_sha256'] = first['source_inventory_sha256']
            (root / 'Training/glioma/0.png').write_bytes((root / 'Training/glioma/2.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                data.prepare(root, base / 'changed', config, 1)
            self.assertFalse((base / 'changed').exists())

    def test_corrupt_images_fail_instead_of_silent_skipping(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.fixture(base / 'source')
            (base / 'source/Training/glioma/broken.jpg').write_bytes(b'not an image')
            with self.assertRaisesRegex(ValueError, 'Cannot read image'):
                data.prepare(base / 'source', base / 'output', CONFIG, 1)
            self.assertFalse((base / 'output').exists())

    def test_leakage_filter_cannot_silently_empty_a_class(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.fixture(base / 'source')
            for path in (base / 'source/Training/glioma').glob('*.png'):
                shutil.copyfile(base / 'source/Testing/glioma/0.png', path)
            with self.assertRaisesRegex(ValueError, 'independent training groups'):
                data.prepare(base / 'source', base / 'output', CONFIG, 1)
            self.assertFalse((base / 'output').exists())


if __name__ == '__main__':
    unittest.main()
