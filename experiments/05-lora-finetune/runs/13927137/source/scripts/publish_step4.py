#!/usr/bin/env python3
"""Verify the finished Step 4 W&B run and archive the complete evidence bundle."""
import argparse
import csv
import json
from pathlib import Path
import re
import tempfile

from verify_data import REPO, sha
from verify_step4 import LABELS, assert_matches, verify_run


def main():
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=REPO / 'data/prepared-v3-conservative-patients')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    report = verify_run(root, args.prepared)
    test_log = (root / 'tests.txt').read_text()
    test_count = re.search(r'Ran (\d+) tests', test_log)
    if not test_count or not test_log.rstrip().endswith('OK'):
        raise ValueError('Unit test evidence is incomplete')
    with (root / 'slurm-accounting.tsv').open() as stream:
        accounting = list(csv.DictReader(stream, delimiter='|'))
    allocation = [row for row in accounting if row['JobIDRaw'] == root.name]
    if len(allocation) != 1 or allocation[0]['State'] != 'COMPLETED' or allocation[0]['ExitCode'] != '0:0':
        raise ValueError('GPU job did not complete successfully')
    if (root / 'exit-status.tsv').read_text().strip() != 'fp16\t0':
        raise ValueError('FP16 process did not complete successfully')
    folder = root / 'fp16'
    record = json.loads((folder / 'run.json').read_text())
    config = record['config']
    metrics = report['metrics']
    api = wandb.Api()
    remote = api.run(f'{config["wandb_entity"]}/{config["wandb_project"]}/{record["wandb_run_id"]}')
    if remote.state != 'finished' or remote.summary['status'] != 'passed':
        raise ValueError('W&B inference run has not finished successfully')
    for key, expected in config.items():
        assert_matches(remote.config[key], expected, f'remote/config/{key}')
    if remote.config['precision'] != 'fp16':
        raise ValueError('Remote precision differs')
    for key in ['n', 'accuracy', 'macro_f1', 'invalid_count', 'invalid_rate']:
        assert_matches(remote.summary[key], metrics[key], f'remote/{key}')
    for label in LABELS:
        assert_matches(remote.summary[f'recall/{label}'], metrics['per_class'][label]['recall'], f'recall/{label}')
    artifacts = [a for a in remote.logged_artifacts()
                 if a.type == 'runtime-evidence' and a.name.split(':')[0] == f'step4-{root.name}-fp16']
    if len(artifacts) != 1:
        raise ValueError('Expected exactly one Step 4 inference artifact')
    artifact = artifacts[0]
    local_files = {str(p.relative_to(folder)): sha(p.read_bytes()) for p in folder.rglob('*') if p.is_file()}
    with tempfile.TemporaryDirectory(prefix='step4-verify-') as temporary:
        downloaded = Path(artifact.download(root=temporary))
        remote_files = {str(p.relative_to(downloaded)): sha(p.read_bytes())
                        for p in downloaded.rglob('*') if p.is_file()}
        if local_files != remote_files:
            raise ValueError('Downloaded W&B evidence differs from local inference files')
    verification = dict(status='passed', state=remote.state, url=record['wandb_url'],
                        metrics_match=True, per_class_recalls_match=True,
                        artifact_name=artifact.name, artifact_digest=artifact.digest,
                        downloaded_files_sha256=remote_files)
    (root / 'independent-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    (root / 'wandb-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print('PASS: W&B finished, all metrics match, and downloaded artifact bytes match', flush=True)
    with wandb.init(entity=config['wandb_entity'], project=config['wandb_project'],
                    job_type='step4-evidence', group=f'step4-{root.name}',
                    name=f'step4-{root.name}-evidence', config=config,
                    tags=['step4', 'evidence', 'zero-shot-validation'],
                    settings=wandb.Settings(disable_git=True, disable_code=True)) as run:
        bundle = wandb.Artifact(f'step4-checkpoint-{root.name}', type='runtime-checkpoint')
        bundle.add_dir(str(root), name=f'runs/{root.name}')
        run.log_artifact(bundle).wait()
        run.summary.update(dict(status='verified', validation_images_verified=579,
                                source_images_verified=7200, unit_tests_passed=int(test_count.group(1)),
                                macro_f1=metrics['macro_f1'], accuracy=metrics['accuracy'], scope='Step 4 only'))
        receipt = dict(wandb_run_id=run.id, wandb_url=run.url,
                       artifact_name=bundle.name, artifact_digest=bundle.digest,
                       inference_verification=verification)
    (root / 'evidence-upload.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    print('PASS: Step 4 evidence bundle uploaded')


if __name__ == '__main__':
    main()
