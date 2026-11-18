"""Independent checks that saved validation predictions cover the frozen split and score correctly."""
from collections import Counter
import re

from mri_vlm.common import assert_matches

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
