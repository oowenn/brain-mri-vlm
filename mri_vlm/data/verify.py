#!/usr/bin/env python3
"""Verify the frozen split manifests and every source image's bytes without regenerating the splits."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from mri_vlm.common import PREPARED, REPO, canonical, read_csv, sha

POLICY = REPO / 'experiments/02-data-split/policy.json'
INVENTORY_SHA256 = 'ac8bc157c04710da31110c741fe8afa9984e78d9befaa18726d3d1727d3c824f'
EXPECTED = {'train': 3279, 'validation': 579, 'test': 1600, 'excluded': 1742}


def validate_manifests(prepared, policy_path=POLICY):
    policy = json.loads(policy_path.read_text())
    summary = json.loads((prepared / 'summary.json').read_text())
    if summary['policy'] != policy or summary['policy_sha256'] != sha(canonical(policy)):
        raise ValueError('Frozen data policy mismatch')
    if sha((prepared / 'inventory.csv').read_bytes()) != INVENTORY_SHA256:
        raise ValueError('Final inventory checksum mismatch')
    for name, expected in summary['output_sha256'].items():
        if Path(name).name != name or sha((prepared / name).read_bytes()) != expected:
            raise ValueError(f'Manifest checksum mismatch: {name}')
    rows = read_csv(prepared / 'inventory.csv')
    if len(rows) != 7200 or len({r['path'] for r in rows}) != 7200:
        raise ValueError('Expected 7200 unique inventory paths')
    counts = dict(Counter(r['split'] for r in rows))
    if counts != EXPECTED:
        raise ValueError(f'Unexpected split counts: {counts}')
    for split in EXPECTED:
        expected = [r for r in rows if r['split'] == split]
        if read_csv(prepared / f'{split}.csv') != expected:
            raise ValueError(f'{split}.csv does not match the pinned inventory')
    constraints = defaultdict(set)
    for row in rows:
        if row['split'] != 'excluded':
            constraints[('group', row['group_id'])].add(row['split'])
            if row['patient_key']:
                constraints[('patient', row['patient_key'])].add(row['split'])
    if any(len(splits) != 1 for splits in constraints.values()):
        raise ValueError('Known image/patient constraint crosses retained splits')
    return rows, policy, summary


def verify_source(source, rows):
    source = source.resolve()
    fingerprint = []
    for index, row in enumerate(sorted(rows, key=lambda r: r['path']), 1):
        path = (source / row['path']).resolve()
        if not path.is_relative_to(source) or sha(path.read_bytes()) != row['sha256']:
            raise ValueError(f'Source image mismatch: {row["path"]}')
        fingerprint.append([row['path'], row['sha256']])
        if index % 1000 == 0:
            print(f'Verified {index}/{len(rows)} source images', flush=True)
    return sha(canonical(fingerprint))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=PREPARED)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, policy, summary = validate_manifests(args.prepared)
    source_hash = verify_source(args.source, rows)
    if source_hash != policy['source_inventory_sha256']:
        raise ValueError('Source fingerprint mismatch')
    report = dict(status='passed', images_verified=len(rows),
                  counts=EXPECTED, final_inventory_sha256=INVENTORY_SHA256,
                  source_inventory_sha256=source_hash,
                  output_sha256=summary['output_sha256'], limitation=summary['limitation'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print('PASS: frozen manifests and all 7200 source images verified')


if __name__ == '__main__':
    main()
