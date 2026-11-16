#!/usr/bin/env python3
"""Reproduce MRI manifests without modifying/copying the source image files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import imagehash
import numpy as np
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
FIELDS = ['path', 'label', 'label_id', 'original_split', 'split', 'group_id',
          'sha256', 'pixel_sha256', 'phash', 'width', 'height', 'exclusion_reason']
PAIR_FIELDS = ['path_a', 'path_b', 'hamming_distance', 'match_kind',
               'correlation', 'accepted', 'cross_source_split', 'label_conflict']


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if not isinstance(config['version'], int) or config['version'] < 1:
        raise ValueError('A positive dataset version is required; latest is not reproducible.')
    if config['labels'] != ['glioma', 'meningioma', 'notumor', 'pituitary']:
        raise ValueError('Unexpected class order; keep the agreed four-class label mapping.')
    if config['phash_size'] != 8 or config['phash_highfreq_factor'] != 4:
        raise ValueError('This protocol uses 64-bit pHash with highfreq_factor=4.')
    if not 0 <= config['max_hamming_distance'] <= 64:
        raise ValueError('Hamming threshold must be between 0 and 64.')
    if not 0 < config['validation_fraction'] < 1:
        raise ValueError('Validation fraction must be between 0 and 1.')
    if config['similarity_size'] != 128 or not 0 < config['min_correlation'] <= 1:
        raise ValueError('Use 128-pixel similarity signatures and a correlation in (0, 1].')
    if not isinstance(config['seed'], int):
        raise ValueError('Seed must be an integer.')
    return config


def scan_source(root: Path, config: dict, workers: int) -> list[dict]:
    """Fail on unexpected layout/unreadable files instead of silently losing samples."""
    tasks = []
    for directory, split in [('Training', 'train'), ('Testing', 'test')]:
        base = root / directory
        if not base.is_dir():
            raise ValueError(f'Missing {base}; source root must contain Training/ and Testing/.')
        classes = {p.name for p in base.iterdir() if p.is_dir()}
        if classes != set(config['labels']):
            raise ValueError(f'Unexpected class directories in {base}: {sorted(classes)}')
        if any(p.is_file() and not p.name.startswith('.') for p in base.iterdir()):
            raise ValueError(f'Unexpected files directly under {base}')
        for label_id, label in enumerate(config['labels']):
            files = sorted(p for p in (base / label).rglob('*') if p.is_file())
            files = [p for p in files if not any(part.startswith('.') for part in p.relative_to(base).parts)]
            if not files:
                raise ValueError(f'No images for {directory}/{label}')
            for path in files:
                if path.suffix.lower() not in EXTENSIONS:
                    raise ValueError(f'Unexpected non-image file: {path}')
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f'Symlinks outside the source are not supported: {path}')
                tasks.append((path, split, label, label_id))

    def inspect(task):
        path, split, label, label_id = task
        try:
            raw_hash = digest(path.read_bytes())
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                width, height = image.size
                pixel_hash = digest(canonical([width, height, 'RGB']) + image.tobytes())
                phash = str(imagehash.phash(image, hash_size=config['phash_size'],
                                           highfreq_factor=config['phash_highfreq_factor']))
                gray = image.convert('L').resize((config['similarity_size'],) * 2,
                                                Image.Resampling.LANCZOS)
                signature = np.asarray(gray, dtype=np.float64).reshape(-1)
                signature = signature - signature.mean()
                norm = np.linalg.norm(signature)
                signature = signature / norm if norm else signature
        except Exception as exc:
            raise ValueError(f'Cannot read image {path}: {exc}') from exc
        return {'path': path.relative_to(root).as_posix(), 'label': label,
                'label_id': label_id, 'original_split': split, 'sha256': raw_hash,
                'pixel_sha256': pixel_hash, 'phash': phash, 'width': width, 'height': height,
                '_signature': signature}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sorted(pool.map(inspect, tasks), key=lambda row: row['path'])


def group_duplicates(rows: list[dict], threshold: int, min_correlation: float) -> tuple[list[list[int]], list[dict]]:
    """Compare across all labels/splits; connected components include transitive links."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [int(row['phash'], 16) for row in rows]
    pairs = []
    # About 26M integer comparisons for 7,200 images; bounded CPU/RAM, no GPU needed.
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            distance = (hashes[i] ^ hashes[j]).bit_count()
            byte_equal = a['sha256'] == b['sha256']
            pixel_equal = a['pixel_sha256'] == b['pixel_sha256']
            if not (byte_equal or pixel_equal or distance <= threshold):
                continue
            correlation = round(float(np.clip(np.dot(a['_signature'], b['_signature']), -1, 1)), 10)
            accepted = byte_equal or pixel_equal or correlation >= min_correlation
            if accepted:
                parent[find(i)] = find(j)
            pairs.append({'path_a': b['path'], 'path_b': a['path'],
                          'hamming_distance': distance,
                          'match_kind': 'bytes' if byte_equal else 'pixels' if pixel_equal else 'phash',
                          'correlation': correlation, 'accepted': accepted,
                          'cross_source_split': a['original_split'] != b['original_split'],
                          'label_conflict': a['label'] != b['label']})
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda group: rows[group[0]]['path']), pairs


def assign_splits(rows: list[dict], groups: list[list[int]], config: dict) -> dict:
    """Test stays intact; quarantine conflicting train groups; stratify by whole group."""
    eligible = defaultdict(list)
    test_conflicts = 0
    for members in groups:
        group_id = digest(canonical([rows[i]['path'] for i in members]))[:16]
        has_test = any(rows[i]['original_split'] == 'test' for i in members)
        conflicting = len({rows[i]['label'] for i in members}) > 1
        if has_test and conflicting:
            test_conflicts += 1
        for i in members:
            row = rows[i]
            row.update(group_id=group_id, exclusion_reason='')
            if row['original_split'] == 'test':
                row['split'] = 'test'
            elif has_test:
                row.update(split='excluded', exclusion_reason='connected_to_test')
            elif conflicting:
                row.update(split='excluded', exclusion_reason='conflicting_train_labels')
            else:
                row['split'] = 'train'
        if not has_test and not conflicting:
            eligible[rows[members[0]]['label']].append(members)

    for label in config['labels']:
        candidates = eligible[label]
        if len(candidates) < 2:
            raise ValueError(f'Need at least two independent training groups for {label}; '
                             'review the duplicate threshold/source before splitting.')
        # Stable hash order avoids dependence on Python RNG/library versions or absolute paths.
        candidates.sort(key=lambda group: digest(canonical(
            [config['seed'], label, rows[group[0]]['group_id']])))
        target = sum(map(len, candidates)) * config['validation_fraction']
        prefix_sizes = []
        count = 0
        for group in candidates[:-1]:
            count += len(group)
            prefix_sizes.append(count)
        n_validation = min(range(1, len(candidates)),
                           key=lambda n: (abs(prefix_sizes[n - 1] - target), n))
        for group in candidates[:n_validation]:
            for i in group:
                rows[i]['split'] = 'validation'

    for members in groups:
        active = {rows[i]['split'] for i in members if rows[i]['split'] != 'excluded'}
        if len(active) > 1:
            raise AssertionError(f'Duplicate group crosses final splits: {active}')
    return {'test_connected_label_conflict_groups': test_conflicts}


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def prepare(root: Path, output: Path, config: dict, workers: int) -> dict:
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Use a new directory; do not overwrite frozen manifests.')
    print('Hashing source images...', flush=True)
    rows = scan_source(root, config, workers)
    source_hash = digest(canonical([[r['path'], r['sha256']] for r in rows]))
    expected = config.get('expected_source_sha256')
    if expected and expected != source_hash:
        raise ValueError(f'Source checksum mismatch: expected {expected}, got {source_hash}')
    print(f'Comparing {len(rows)} images for duplicates...', flush=True)
    groups, pairs = group_duplicates(rows, config['max_hamming_distance'], config['min_correlation'])
    for row in rows:
        del row['_signature']
    accepted_pairs = [p for p in pairs if p['accepted']]
    diagnostics = assign_splits(rows, groups, config)
    protocol = {k: v for k, v in config.items() if k != 'expected_source_sha256'}
    versions = {name: importlib.metadata.version(name)
                for name in ['Pillow', 'ImageHash', 'numpy', 'scipy']}
    report = {
        'schema_version': 1, 'protocol': protocol,
        'source_inventory_sha256': source_hash,
        'protocol_sha256': digest(canonical(protocol)),
        'pipeline_sha256': digest(Path(__file__).read_bytes()),
        'python_version': platform.python_version(), 'packages': versions,
        'counts': {split: dict(sorted(Counter(r['label'] for r in rows if r['split'] == split).items()))
                   for split in ['train', 'validation', 'test', 'excluded']},
        'source_counts': {split: dict(sorted(Counter(r['label'] for r in rows if r['original_split'] == split).items()))
                          for split in ['train', 'test']},
        'exclusions_by_reason': dict(Counter(r['exclusion_reason'] for r in rows if r['split'] == 'excluded')),
        'candidate_pairs': len(pairs),
        'duplicate_pairs': len(accepted_pairs),
        'rejected_candidate_pairs': len(pairs) - len(accepted_pairs),
        'cross_source_split_pairs': sum(p['cross_source_split'] for p in accepted_pairs),
        'cross_source_distance_histogram': {str(k): v for k, v in sorted(Counter(
            p['hamming_distance'] for p in accepted_pairs if p['cross_source_split']).items())},
        'label_conflict_pairs': sum(p['label_conflict'] for p in accepted_pairs),
        'duplicate_groups': sum(len(group) > 1 for group in groups),
        'largest_group': max(map(len, groups)),
        'review_required': True,
        'review_note': 'pHash similarity is a candidate match, not proof of identical patients/images. '
                       'Review pairs and freeze the protocol before training; test set is unchanged.',
        **diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output.parent) as temp:
        stage = Path(temp) / 'result'
        stage.mkdir()
        write_csv(stage / 'inventory.csv', rows, FIELDS)
        write_csv(stage / 'duplicate_pairs.csv', pairs, PAIR_FIELDS)
        for split in ['train', 'validation', 'test', 'excluded']:
            write_csv(stage / f'{split}.csv', [r for r in rows if r['split'] == split], FIELDS)
        report['output_sha256'] = {p.name: digest(p.read_bytes()) for p in sorted(stage.glob('*.csv'))}
        (stage / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        os.rename(stage, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/data.json')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--download', action='store_true', help='Download the pinned Kaggle version to its cache.')
    source.add_argument('--source', type=Path, help='Existing source root containing Training/ and Testing/.')
    parser.add_argument('--output', type=Path, required=True, help='New directory for manifests/audit; images stay in place.')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--log-wandb', action='store_true', help='Upload manifests/audit only, never images or credentials.')
    args = parser.parse_args()
    if args.workers < 1 or args.workers > int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)):
        parser.error('--workers must be positive and fit the allocated CPU count.')
    # Insomnia Slurm clients need a compute allocation. Local laptop usage is supported.
    if platform.node().startswith(('2402-login-', 'insomnia-login')) and not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run data preparation on a Slurm compute node, not an Insomnia login node.')
    try:
        config = read_config(args.config)
        if args.download:
            import kagglehub
            root = Path(kagglehub.dataset_download(f"{config['dataset']}/versions/{config['version']}"))
        else:
            root = args.source.expanduser().resolve()
        print(f'Source root: {root}', flush=True)
        report = prepare(root, args.output.expanduser().resolve(), config, args.workers)
        print(json.dumps({'source_inventory_sha256': report['source_inventory_sha256'],
                          'counts': report['counts'], 'duplicate_pairs': report['duplicate_pairs']}, indent=2))
        if args.log_wandb:
            import wandb
            with wandb.init(project=os.environ.get('WANDB_PROJECT', 'brain-mri-vlm'),
                            entity=os.environ.get('WANDB_ENTITY'), job_type='data-preparation',
                            name=f"data-v{config['version']}-phash{config['max_hamming_distance']}",
                            config=report['protocol']) as run:
                run.summary.update(report)
                artifact = wandb.Artifact('brain-mri-manifests', type='dataset-manifests',
                                          metadata={'source_inventory_sha256': report['source_inventory_sha256']})
                artifact.add_dir(str(args.output.expanduser().resolve()))
                run.log_artifact(artifact)
                print(f'W&B run: {run.url}')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Data preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
