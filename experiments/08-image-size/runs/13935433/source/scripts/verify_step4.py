#!/usr/bin/env python3
"""Independently verify full validation coverage, raw-answer parsing, and metrics."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from verify_data import REPO, read_csv, sha, validate_manifests

LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
SUPPORTS = dict(zip(LABELS, [181, 157, 121, 120]))


def recompute_metrics(predictions):
    """Count label pairs independently of evaluate.score_predictions."""
    pairs = Counter((row['true_label'], row['prediction']) for row in predictions)
    actual = Counter(row['true_label'] for row in predictions)
    predicted = Counter(row['prediction'] for row in predictions)
    classes = {}
    for label in LABELS:
        tp = pairs[label, label]
        precision = tp / predicted[label] if predicted[label] else 0.0
        recall = tp / actual[label] if actual[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes[label] = dict(precision=precision, recall=recall, f1=f1, support=actual[label])
    n = len(predictions)
    return dict(n=n, accuracy=sum(pairs[label, label] for label in LABELS) / n,
                macro_f1=sum(item['f1'] for item in classes.values()) / len(LABELS),
                per_class=classes, invalid_count=predicted['INVALID'],
                invalid_rate=predicted['INVALID'] / n,
                confusion_matrix=dict(rows=LABELS, columns=LABELS + ['INVALID'],
                    counts=[[pairs[truth, guess] for guess in LABELS + ['INVALID']] for truth in LABELS]))


def assert_matches(actual, expected, path='metrics'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'{path}: keys differ')
        for key in expected:
            assert_matches(actual[key], expected[key], f'{path}/{key}')
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'{path}: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'{path}: values differ')


def verify_predictions(rows, predictions, selected, metrics):
    expected = sorted(rows, key=lambda row: row['path'])
    if (len(expected) != 579 or len({r['path'] for r in expected}) != 579
            or Counter(r['label'] for r in expected) != SUPPORTS
            or any(r['split'] != 'validation' for r in expected)):
        raise ValueError('Expected the frozen 579-image validation split')
    if [r['path'] for r in predictions] != [r['path'] for r in expected]:
        raise ValueError('Predictions must cover each validation image exactly once in saved order')
    if selected != [{k: row[k] for k in ('path', 'label', 'sha256', 'group_id')} for row in expected]:
        raise ValueError('Selected-image evidence differs from the frozen validation manifest')
    for row, truth in zip(predictions, expected):
        match = re.fullmatch(r'(glioma|meningioma|notumor|pituitary)\.?', row['raw_response'].strip().lower())
        parsed = match.group(1) if match else 'INVALID'
        if row['prediction'] != parsed or row['true_label'] != truth['label']:
            raise ValueError(f'Label or raw-answer parse mismatch: {truth["path"]}')
    independently_scored = recompute_metrics(predictions)
    assert_matches(metrics, independently_scored)
    return independently_scored


def verify_run(root, prepared):
    validate_manifests(prepared)
    folder = root / 'fp16'
    record = json.loads((folder / 'run.json').read_text())
    config = json.loads((REPO / 'configs/step4.json').read_text())
    prior_config = json.loads((REPO / 'configs/step3.json').read_text())
    if config != {**prior_config, 'stage': 'step4', 'per_class': 0}:
        raise ValueError('Step 4 changed the accepted Step 3 inference protocol')
    if (record['config'] != config or record['status'] != 'passed'
            or record['stage'] != 'step4' or record['scope'] != 'zero-shot-validation'
            or record['precision'] != 'fp16' or record['per_class'] != 0
            or record['sample_size'] != 579):
        raise ValueError('Run is not a complete Step 4 FP16 evaluation')
    if record['validation_manifest_sha256'] != sha((prepared / 'validation.csv').read_bytes()):
        raise ValueError('Run manifest hash differs from the verified validation manifest')
    for name, expected in record['code_sha256'].items():
        if sha((root / 'source' / name).read_bytes()) != expected:
            raise ValueError(f'Executed source snapshot mismatch: {name}')
    predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
    selected = json.loads((folder / 'selected-images.json').read_text())
    metrics = json.loads((folder / 'metrics.json').read_text())
    verified = verify_predictions(read_csv(prepared / 'validation.csv'), predictions, selected, metrics)
    runtime = json.loads((folder / 'model-runtime.json').read_text())
    if runtime['quantized_module_count'] != 0 or runtime['parameter_devices'] != ['cuda:0']:
        raise ValueError('Expected unquantized FP16 model entirely on cuda:0')
    data = json.loads((root / 'data-verification.json').read_text())
    if data['status'] != 'passed' or data['images_verified'] != 7200:
        raise ValueError('Source verification did not complete')
    return dict(status='passed', unique_validation_images=579, expected_supports=SUPPORTS,
                metrics=verified, predictions_sha256=sha((folder / 'predictions.jsonl').read_bytes()),
                selected_images_sha256=sha((folder / 'selected-images.json').read_bytes()),
                raw_response_counts=dict(Counter(r['raw_response'] for r in predictions)),
                predicted_label_counts=dict(Counter(r['prediction'] for r in predictions)),
                scoring_method='Independent Counter-based calculation; invalid outputs retained as errors',
                test_set_inference=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=REPO / 'data/prepared-v3-conservative-patients')
    args = parser.parse_args()
    report = verify_run(args.run_dir, args.prepared)
    (args.run_dir / 'independent-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: all 579 validation predictions, raw-answer parsing, and metrics independently verified')


if __name__ == '__main__':
    main()
