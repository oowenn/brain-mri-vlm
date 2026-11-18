#!/usr/bin/env python3
"""Independently verify full validation coverage, raw-answer parsing, and metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path

from mri_vlm import naming
from mri_vlm.common import PREPARED, REPO, read_csv, sha
from mri_vlm.data.verify import validate_manifests
from mri_vlm.inference.prediction_check import SUPPORTS, verify_predictions


def verify_run(root, prepared):
    validate_manifests(prepared)
    folder = root / 'fp16'
    record = json.loads((folder / 'run.json').read_text())
    config = json.loads((REPO / naming.config_path('step4')).read_text())
    prior_config = json.loads((REPO / naming.config_path('step3')).read_text())
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
    parser.add_argument('--prepared', type=Path, default=PREPARED)
    args = parser.parse_args()
    report = verify_run(args.run_dir, args.prepared)
    (args.run_dir / 'independent-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: all 579 validation predictions, raw-answer parsing, and metrics independently verified')


if __name__ == '__main__':
    main()
