#!/usr/bin/env python3
"""Finalize immutable manifests using reviewed image groups and recovered patient IDs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

REPO = Path(__file__).resolve().parents[1]
EXTRA = ['image_group_id', 'patient_key', 'patient_status', 'source_label',
         'source_label_disagreement']


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def apply_policy(inventory, mapping, config):
    """Union constraints, quarantine annotations, then stratify eligible whole groups."""
    if (config['exclude_training_source_label_disagreements'] is not True
            or config['preserve_test_labels'] is not True
            or config['infer_unknown_patient_ids'] is not False):
        raise ValueError('Unsupported policy: use the approved conservative settings.')
    if not 0 < config['validation_fraction'] < 1 or not isinstance(config['seed'], int):
        raise ValueError('Invalid validation fraction or seed.')
    rows = sorted((dict(row) for row in inventory), key=lambda r: r['path'])
    paths = [r['path'] for r in rows]
    lookup = {r['path']: r for r in mapping}
    if len(set(paths)) != len(paths) or len(lookup) != len(mapping) or set(paths) != set(lookup):
        raise ValueError('Mapping must cover each inventory path exactly once.')
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        match = lookup[row['path']]
        if any(row[k] != match[k] for k in ['label', 'original_split']):
            raise ValueError(f'Mapping metadata disagrees: {row["path"]}')
        if row['original_split'] not in {'train', 'test'}:
            raise ValueError('Unexpected original split.')
        if row['label'] not in config['labels'] or str(config['labels'].index(row['label'])) != str(row['label_id']):
            raise ValueError('Unexpected class or label ID.')
        status, key = match['status'], match['patient_key']
        if status not in {'confirmed', 'needs_review', 'unmatched'}:
            raise ValueError('Unexpected patient mapping status.')
        if status == 'confirmed':
            if (not re.fullmatch('[0-9a-f]{64}', key)
                    or not 0.995 <= float(match['correlation_64']) <= 1
                    or not 0.995 <= float(match['correlation_256']) <= 1
                    or not -1 <= float(match['other_patient_correlation']) < 0.98
                    or match['source_label'] not in config['labels']
                    or match['label_disagreement'] != str(row['label'] != match['source_label'])):
                raise ValueError(f'Invalid confirmed mapping: {row["path"]}')
        elif key or match['label_disagreement'] not in {'', 'False'}:
            raise ValueError('Unconfirmed identities must remain unknown and unflagged.')
        row.update(image_group_id=row['group_id'], patient_key=key, patient_status=status,
                   source_label=match['source_label'] if status == 'confirmed' else '',
                   source_label_disagreement=str(status == 'confirmed' and match['label_disagreement'] == 'True'))
        constraints = [('image', row['image_group_id'])]
        if key:
            constraints.append(('patient', key))
        for constraint in constraints:
            if constraint in seen:
                parent[find(i)] = find(seen[constraint])
            else:
                seen[constraint] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(rows[i])
    eligible = defaultdict(list)
    for group in groups.values():
        group_id = digest(canonical([r['path'] for r in group]))[:16]
        has_test = any(r['original_split'] == 'test' for r in group)
        remaining = [r for r in group if r['original_split'] == 'train'
                     and r['source_label_disagreement'] != 'True']
        conflict = len({r['label'] for r in remaining}) > 1
        for row in group:
            reason = ''
            if row['original_split'] == 'test':
                split = 'test'
            elif row['source_label_disagreement'] == 'True':
                split, reason = 'excluded', 'source_label_disagreement'
            elif has_test:
                split, reason = 'excluded', 'connected_to_test'
            elif conflict:
                split, reason = 'excluded', 'conflicting_train_labels'
            else:
                split = 'train'
            row.update(group_id=group_id, split=split, exclusion_reason=reason)
        if remaining and not has_test and not conflict:
            eligible[remaining[0]['label']].append(remaining)
    for label in config['labels']:
        candidates = eligible[label]
        if len(candidates) < 2:
            raise ValueError(f'Need at least two independent training groups for {label}.')
        candidates.sort(key=lambda g: digest(canonical([config['seed'], label, g[0]['group_id']])))
        target = sum(map(len, candidates)) * config['validation_fraction']
        prefix, count = [], 0
        for group in candidates[:-1]:
            count += len(group)
            prefix.append(count)
        n = min(range(1, len(candidates)), key=lambda n: (abs(prefix[n - 1] - target), n))
        for group in candidates[:n]:
            for row in group:
                row['split'] = 'validation'
    for group in groups.values():
        if len({r['split'] for r in group if r['split'] != 'excluded'}) > 1:
            raise AssertionError('Image/patient constraint crosses final splits.')
    return rows


def finalize(prepared, mapping, output, config):
    if output.exists():
        raise ValueError(f'Output already exists: {output}; use a new directory.')
    summary = json.loads((prepared / 'summary.json').read_text())
    if summary['source_inventory_sha256'] != config['source_inventory_sha256']:
        raise ValueError('Source inventory checksum mismatch.')
    for name, expected in summary['output_sha256'].items():
        if Path(name).name != name or digest((prepared / name).read_bytes()) != expected:
            raise ValueError(f'Input audit checksum mismatch: {name}')
    for path, expected in [(prepared / 'inventory.csv', config['input_inventory_sha256']),
                           (mapping, config['patient_mapping_sha256'])]:
        if digest(path.read_bytes()) != expected:
            raise ValueError(f'Pinned input checksum mismatch: {path.name}')
    original = read_csv(prepared / 'inventory.csv')
    rows = apply_policy(original, read_csv(mapping), config)
    patient_splits = defaultdict(set)
    for row in rows:
        if row['patient_key'] and row['split'] != 'excluded':
            patient_splits[row['patient_key']].add(row['split'])
    cross_split_count = sum(len(splits) > 1 for splits in patient_splits.values())
    if cross_split_count:
        raise AssertionError('Known patient crosses final splits.')
    # No test row or clinical label is changed by this stage.
    identity = lambda r: (r['path'], r['label'], str(r['label_id']), r['sha256'])
    if {identity(r) for r in original if r['original_split'] == 'test'} != {
            identity(r) for r in rows if r['split'] == 'test'}:
        raise AssertionError('Official test identity or label changed.')
    report = {
        'schema_version': 2, 'policy': config,
        'policy_sha256': digest(canonical(config)),
        'pipeline_sha256': digest(Path(__file__).read_bytes()),
        'input_summary_sha256': digest((prepared / 'summary.json').read_bytes()),
        'counts': {s: dict(sorted(Counter(r['label'] for r in rows if r['split'] == s).items()))
                   for s in ['train', 'validation', 'test', 'excluded']},
        'exclusions_by_reason': dict(sorted(Counter(r['exclusion_reason'] for r in rows if r['split'] == 'excluded').items())),
        'patient_status_counts': dict(sorted(Counter(r['patient_status'] for r in rows).items())),
        'confirmed_patients': len({r['patient_key'] for r in rows if r['patient_key']}),
        'source_label_disagreements': dict(Counter(r['original_split'] for r in rows if r['source_label_disagreement'] == 'True')),
        'known_patient_cross_split_count': cross_split_count,
        'limitation': 'Disjoint for recovered identities and accepted image groups only. Unknown patients may still overlap. Official test labels retain one source annotation disagreement.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(original[0]) + EXTRA
    with tempfile.TemporaryDirectory(prefix='.finalize-', dir=output.parent) as temp:
        stage = Path(temp) / 'result'
        stage.mkdir()
        write_csv(stage / 'inventory.csv', rows, fields)
        for split in ['train', 'validation', 'test', 'excluded']:
            write_csv(stage / f'{split}.csv', [r for r in rows if r['split'] == split], fields)
        write_csv(stage / 'label_disagreements.csv', [r for r in rows if r['source_label_disagreement'] == 'True'], fields)
        write_csv(stage / 'withheld_identities.csv', [r for r in rows if r['patient_status'] == 'needs_review'], fields)
        report['output_sha256'] = {p.name: digest(p.read_bytes()) for p in sorted(stage.glob('*.csv'))}
        (stage / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        os.rename(stage, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/data_policy.json')
    parser.add_argument('--log-wandb', action='store_true')
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        report = finalize(args.prepared, args.mapping, args.output, config)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.log_wandb:
            import wandb
            with wandb.init(project=os.environ.get('WANDB_PROJECT', 'brain-mri-vlm'),
                            entity=os.environ.get('WANDB_ENTITY'), job_type='data-finalization',
                            name=config['version'], config=config) as run:
                run.use_artifact(config['mapping_artifact'])
                artifact = wandb.Artifact('brain-mri-manifests', type='dataset-manifests', metadata=report)
                artifact.add_dir(str(args.output.resolve()))
                run.log_artifact(artifact)
                run.summary.update(report)
                print(f'W&B run: {run.url}')
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Data finalization failed: {exc}\n')


if __name__ == '__main__':
    main()
