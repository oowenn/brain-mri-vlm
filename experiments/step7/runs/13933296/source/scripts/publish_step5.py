#!/usr/bin/env python3
"""Verify Step 5 tracking and back up adapters, optimizer states, and merged weights."""
import argparse
import csv
import json
from pathlib import Path
import re
import tempfile

from verify_data import REPO, sha
from verify_step4 import assert_matches
from verify_step5 import verify_run


def main():
    import wandb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=REPO / 'data/prepared-v3-conservative-patients')
    args = parser.parse_args()
    root, checkpoints = args.run_dir.resolve(), args.checkpoints.resolve()
    verified = verify_run(root, args.prepared, checkpoints)
    record = json.loads((root / 'run.json').read_text())
    cfg = record['config']
    with (root / 'slurm-accounting.tsv').open() as stream:
        row = next(row for row in csv.DictReader(stream, delimiter='|') if row['JobIDRaw'] == root.name)
    if row['State'] != 'COMPLETED' or row['ExitCode'] != '0:0':
        raise ValueError('GPU job did not complete successfully')
    tests = (root / 'tests.txt').read_text()
    count = re.search(r'Ran (\d+) tests', tests)
    if not count or not tests.rstrip().endswith('OK'):
        raise ValueError('Test evidence is incomplete')
    api = wandb.Api()
    project = f'{cfg["wandb_entity"]}/{cfg["wandb_project"]}'
    remote = api.run(f'{project}/{record["wandb_run_id"]}')
    if remote.state != 'finished' or remote.summary['status'] != 'passed':
        raise ValueError('Training run has not finished successfully in W&B')
    for key, value in cfg.items():
        assert_matches(remote.config[key], value, f'config/{key}')
    for remote_key, value in dict(selected_epoch=record['selected_epoch'], optimizer_updates=record['optimizer_updates'],
                                  merged_macro_f1=record['merged_metrics']['macro_f1'],
                                  merged_accuracy=record['merged_metrics']['accuracy'],
                                  merged_invalid_rate=record['merged_metrics']['invalid_rate'], merge_verified=True).items():
        assert_matches(remote.summary[remote_key], value, f'remote/{remote_key}')
    artifacts = [a for a in remote.logged_artifacts() if a.type == 'runtime-evidence'
                 and a.name.split(':')[0] == f'step5-{root.name}-training-evidence']
    if len(artifacts) != 1:
        raise ValueError('Expected exactly one completed training evidence artifact')
    training_artifact = artifacts[0]
    expected = json.loads((root / 'training-artifact-files.json').read_text())
    expected['training-artifact-files.json'] = sha((root / 'training-artifact-files.json').read_bytes())
    with tempfile.TemporaryDirectory(prefix='step5-evidence-') as temporary:
        downloaded = Path(training_artifact.download(root=temporary))
        actual = {str(p.relative_to(downloaded)): sha(p.read_bytes()) for p in downloaded.rglob('*') if p.is_file()}
        if actual != expected:
            raise ValueError('Downloaded training evidence differs from saved hashes')
    print('PASS: training W&B state, configuration, metrics, and downloaded evidence match', flush=True)
    with wandb.init(entity=cfg['wandb_entity'], project=cfg['wandb_project'],
                    name=f'step5-{root.name}-evidence', group=f'step5-{root.name}', job_type='step5-evidence',
                    config=cfg, tags=['step5', 'evidence', 'model-checkpoint'],
                    settings=wandb.Settings(disable_git=True, disable_code=True)) as run:
        run.use_artifact(f'{project}/{training_artifact.name}')
        models = wandb.Artifact(f'step5-{root.name}-checkpoints', type='model',
                                metadata=dict(training_run=record['wandb_run_id'], base_revision=cfg['revision'],
                                              selected_epoch=record['selected_epoch'], merged_path='merged-fp16'))
        models.add_dir(str(checkpoints))
        run.log_artifact(models).wait()
        expected_weights = json.loads((root / 'checkpoint-files.json').read_text())
        with tempfile.TemporaryDirectory(prefix='step5-weights-') as temporary:
            restored = Path(api.artifact(f'{project}/{models.name}').download(root=temporary))
            if {str(p.relative_to(restored)) for p in restored.rglob('*') if p.is_file()} != set(expected_weights):
                raise ValueError('Downloaded model file set differs')
            for name, info in expected_weights.items():
                if sha((restored / name).read_bytes()) != info['sha256']:
                    raise ValueError(f'Downloaded model checksum mismatch: {name}')
        model_receipt = dict(artifact_name=models.name, artifact_digest=models.digest,
                             files_verified=len(expected_weights), bytes_verified=sum(v['bytes'] for v in expected_weights.values()),
                             selected_adapter=f'epoch-{record["selected_epoch"]:03}/adapter', merged_model='merged-fp16',
                             restore_command=f'wandb artifact get {project}/{models.name} --root checkpoints/step5/{root.name}')
        (root / 'model-artifacts.json').write_text(json.dumps(model_receipt, indent=2) + '\n')
        verification = dict(status='passed', training_run_state=remote.state, metrics_match=True,
                            training_evidence_artifact=training_artifact.name, training_evidence_digest=training_artifact.digest,
                            training_evidence_files_verified=len(expected), model_artifact=model_receipt)
        (root / 'wandb-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
        (root / 'independent-verification.json').write_text(json.dumps(verified, indent=2) + '\n')
        print('PASS: all uploaded model/checkpoint files downloaded and verified by SHA-256', flush=True)
        bundle = wandb.Artifact(f'step5-checkpoint-{root.name}', type='runtime-checkpoint')
        bundle.add_dir(str(root), name=f'runs/{root.name}')
        run.log_artifact(bundle).wait()
        run.summary.update(dict(status='verified', unit_tests_passed=int(count.group(1)),
                                training_images_per_epoch=3279, epochs=cfg['training']['epochs'], validation_images=579,
                                selected_epoch=record['selected_epoch'], merged_macro_f1=record['merged_metrics']['macro_f1'],
                                merged_accuracy=record['merged_metrics']['accuracy'], merge_verified=True,
                                checkpoint_files_verified=len(expected_weights)))
        receipt = dict(wandb_run_id=run.id, wandb_url=run.url, artifact_name=bundle.name, artifact_digest=bundle.digest,
                       model_artifact=model_receipt, training_run_url=record['wandb_url'])
    (root / 'evidence-upload.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    print('PASS: Step 5 evidence and model artifacts published and verified')


if __name__ == '__main__':
    main()
