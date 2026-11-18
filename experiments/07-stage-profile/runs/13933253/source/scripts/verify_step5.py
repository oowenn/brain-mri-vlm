#!/usr/bin/env python3
"""Verify Step 5 training coverage, selection, validation scores, and merge evidence."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from lora_utils import choose_checkpoint, epoch_order, merge_passes
from evaluate import select_sample
from verify_data import REPO, read_csv, sha, validate_manifests
from verify_step4 import assert_matches, verify_predictions


def verify_run(root, prepared, checkpoint_root=None):
    validate_manifests(prepared)
    config = json.loads((root / 'run.json').read_text())
    if config['status'] != 'passed' or config['test_set_inference']:
        raise ValueError('Training/merge job did not pass')
    cfg = config['config']
    if cfg != json.loads((root / 'source/configs/step5.json').read_text()):
        raise ValueError('Executed config differs from the source snapshot')
    prior = json.loads((root / 'source/configs/step4.json').read_text())
    if {k: cfg[k] for k in prior} != {**prior, 'stage': 'step5'}:
        raise ValueError('Inference protocol changed')
    train = sorted(read_csv(prepared / 'train.csv'), key=lambda row: row['path'])
    validation = sorted(read_csv(prepared / 'validation.csv'), key=lambda row: row['path'])
    if len(train) != 3279 or any(row['split'] != 'train' for row in train):
        raise ValueError('Unexpected training split')
    for split in ['train', 'validation']:
        if config[f'{split}_manifest_sha256'] != sha((prepared / f'{split}.csv').read_bytes()):
            raise ValueError(f'{split} manifest hash changed')
    expected_train = [{k: r[k] for k in ('path', 'label', 'sha256', 'group_id')} for r in train]
    if json.loads((root / 'training-images.json').read_text()) != expected_train:
        raise ValueError('Training image record differs from frozen manifest')
    epochs = json.loads((root / 'epochs.json').read_text())
    tc = cfg['training']
    if [r['epoch'] for r in epochs] != list(range(1, tc['epochs'] + 1)):
        raise ValueError('Missing training epochs')
    history = [json.loads(line) for line in (root / 'training-history.jsonl').read_text().splitlines()]
    per_epoch = math.ceil(len(train) / (tc['micro_batch_size'] * tc['gradient_accumulation_steps']))
    if [r['step'] for r in history] != list(range(1, per_epoch * tc['epochs'] + 1)):
        raise ValueError('Optimizer-update evidence is incomplete')
    if config['optimizer_updates'] != len(history):
        raise ValueError('Optimizer step count mismatch')
    for result in epochs:
        epoch = result['epoch']
        expected = [train[i]['path'] for i in epoch_order(len(train), tc['seed'], epoch)]
        if json.loads((root / f'epoch-{epoch:03}-order.json').read_text()) != expected:
            raise ValueError('Training order or coverage differs')
        updates = [r for r in history if r['epoch'] == epoch]
        if len(updates) != per_epoch or sum(r['images'] for r in updates) != 3279:
            raise ValueError('Training examples were skipped or duplicated')
        if any(not math.isfinite(r['loss']) or not math.isfinite(r['preclip_gradient_norm']) for r in updates):
            raise ValueError('Nonfinite optimization evidence')
        mean_loss = sum(r['loss'] * r['supervised_tokens'] for r in updates) / sum(r['supervised_tokens'] for r in updates)
        assert_matches(result['training_loss'], mean_loss, f'epoch-{epoch}/training-loss')
    verified = {}
    for folder in sorted((root / 'validation').iterdir()):
        predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        metrics = json.loads((folder / 'metrics.json').read_text())
        selected = json.loads((folder / 'selected-images.json').read_text())
        verify_predictions(validation, predictions, selected, metrics)
        loss = json.loads((folder / 'loss.json').read_text())
        if loss['images'] != 579 or not math.isfinite(loss['token_weighted_loss']):
            raise ValueError('Validation loss evidence is incomplete')
        verified[folder.name] = dict(metrics=metrics, loss=loss['token_weighted_loss'],
                                     predictions_sha256=sha((folder / 'predictions.jsonl').read_bytes()))
    expected_folders = {f'epoch-{i:03}' for i in range(1, tc['epochs'] + 1)} | {'merged-fp16'}
    if set(verified) != expected_folders:
        raise ValueError('Missing or unexpected validation runs')
    for item in epochs:
        saved = verified[f'epoch-{item["epoch"]:03}']
        assert_matches(item['metrics'], saved['metrics'])
        assert_matches(item['validation_loss'], saved['loss'])
    selection = json.loads((root / 'selection.json').read_text())
    best = choose_checkpoint(epochs)
    if selection['selected'] != best or selection['candidates'] != epochs or config['selected_epoch'] != best['epoch']:
        raise ValueError('Checkpoint selection differs from the fixed rule')
    merged = json.loads((root / 'merge-verification.json').read_text())
    if merged['status'] != 'passed' or not merge_passes(cfg['merge_verification'], merged):
        raise ValueError('Merge checks failed')
    selected_predictions = [json.loads(x) for x in (root / 'validation' / f'epoch-{best["epoch"]:03}' / 'predictions.jsonl').read_text().splitlines()]
    merged_predictions = [json.loads(x) for x in (root / 'validation/merged-fp16/predictions.jsonl').read_text().splitlines()]
    agreement = sum(a['prediction'] == b['prediction'] for a, b in zip(selected_predictions, merged_predictions)) / 579
    assert_matches(merged['full_validation_label_agreement'], agreement)
    assert_matches(merged['absolute_macro_f1_difference'], abs(best['metrics']['macro_f1'] - verified['merged-fp16']['metrics']['macro_f1']))
    assert_matches(config['merged_metrics'], verified['merged-fp16']['metrics'])
    probes = select_sample(validation, cfg['merge_verification']['probe_per_class'], cfg['merge_verification']['seed'])
    probe_paths = [row['path'] for row in probes]
    adapter_probe = [json.loads(x) for x in (root / 'merge-probe/adapter/predictions.jsonl').read_text().splitlines()]
    merged_probe = [json.loads(x) for x in (root / 'merge-probe/merged/predictions.jsonl').read_text().splitlines()]
    selected_by_path = {r['path']: r for r in selected_predictions}
    merged_by_path = {r['path']: r for r in merged_predictions}
    if [r['path'] for r in adapter_probe] != probe_paths or [r['path'] for r in merged_probe] != probe_paths:
        raise ValueError('Merge probes differ from the fixed sample')
    for adapter_row, merged_row in zip(adapter_probe, merged_probe):
        if (adapter_row['generated_token_ids'] != selected_by_path[adapter_row['path']]['generated_token_ids']
                or merged_row['generated_token_ids'] != merged_by_path[merged_row['path']]['generated_token_ids']):
            raise ValueError('Probe output differs from its complete validation run')
    actual_matches = sum(a['generated_token_ids'] == b['generated_token_ids'] for a, b in zip(adapter_probe, merged_probe))
    assert_matches(merged['probe_token_matches'], actual_matches)
    parameters = json.loads((root / 'trainable-parameters.json').read_text())
    if (len(parameters['modules']) != tc['expected_lora_modules']
            or any('text_model.layers.' not in name for name in parameters['modules'])
            or any('lora_' not in name or info['dtype'] != 'torch.float32' for name, info in parameters['parameters'].items())):
        raise ValueError('Trainable parameter evidence differs from decoder-only FP32 LoRA')
    for name in ['label-mask-audit.json', 'loss-audit.json', 'gradient-audit.json']:
        if json.loads((root / name).read_text())['status'] != 'passed':
            raise ValueError(f'Preflight audit failed: {name}')
    files = json.loads((root / 'checkpoint-files.json').read_text())
    weights_verified = False
    if checkpoint_root is not None:
        import numpy as np
        if {str(p.relative_to(checkpoint_root)) for p in checkpoint_root.rglob('*') if p.is_file()} != set(files):
            raise ValueError('Checkpoint file inventory differs')
        for name, info in files.items():
            path = checkpoint_root / name
            if path.stat().st_size != info['bytes'] or sha(path.read_bytes()) != info['sha256']:
                raise ValueError(f'Checkpoint checksum mismatch: {name}')
        for path in (checkpoint_root / 'merged-fp16').glob('*.safetensors'):
            with path.open('rb') as stream:
                header = json.loads(stream.read(int.from_bytes(stream.read(8), 'little')))
            if any('lora_' in name or info['dtype'] != 'F16' for name, info in header.items() if name != '__metadata__'):
                raise ValueError('Saved merged weights are not plain FP16 tensors')
        with np.load(checkpoint_root / 'merge-logits.npz', allow_pickle=False) as logits:
            delta = np.abs(logits['adapter'].astype('float64') - logits['merged'].astype('float64'))
            assert_matches(merged['max_absolute_logit_difference'], float(delta.max()))
            # Reduction rounding differs between NumPy float64 and PyTorch float32.
            if not math.isclose(merged['mean_absolute_logit_difference'], float(delta.mean()), rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError('Saved merge logits differ from the reported mean error')
        weights_verified = True
    return dict(status='passed', training_images_per_epoch=3279, epochs=len(epochs), optimizer_updates=len(history),
                validation_images=579, validation_runs=verified, selected_epoch=best['epoch'],
                merge_verified=True, checkpoint_files_verified=weights_verified, checkpoint_file_count=len(files),
                test_set_inference=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=REPO / 'data/prepared-v3-conservative-patients')
    parser.add_argument('--checkpoints', type=Path)
    args = parser.parse_args()
    report = verify_run(args.run_dir, args.prepared, args.checkpoints)
    (args.run_dir / 'independent-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: training coverage, validation metrics, selection, and merge evidence independently verified')


if __name__ == '__main__':
    main()
