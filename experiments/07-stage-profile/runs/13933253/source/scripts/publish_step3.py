#!/usr/bin/env python3
"""Verify finished W&B runs, then archive the complete Step 3 evidence bundle."""
import argparse
import json
import math
from pathlib import Path
import re

from evaluate import parse_prediction, score_predictions
from verify_data import REPO, sha


def main():
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    config = json.loads((REPO / 'configs/step3.json').read_text())
    data_check = json.loads((root / 'data-verification.json').read_text())
    test_log = (root / 'tests.txt').read_text()
    test_count = re.search(r'Ran (\d+) tests', test_log)
    if data_check['status'] != 'passed' or not test_count or not test_log.rstrip().endswith('OK'):
        raise ValueError('Data verification or test evidence is incomplete')
    api = wandb.Api()
    verification = {}
    selected = None
    for precision in ['fp16', 'int8', 'int4']:
        folder = root / precision
        record = json.loads((folder / 'run.json').read_text())
        metrics = json.loads((folder / 'metrics.json').read_text())
        predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        sample = json.loads((folder / 'selected-images.json').read_text())
        if (record['status'] != 'passed' or record['config'] != config
                or len(predictions) != 4 * config['per_class']
                or metrics != score_predictions(predictions)
                or any(r['prediction'] != parse_prediction(r['raw_response']) for r in predictions)):
            raise ValueError(f'Local result verification failed: {precision}')
        if selected is not None and sample != selected:
            raise ValueError('Precision checks used different samples')
        selected = sample
        for name, expected in record['code_sha256'].items():
            if sha((REPO / name).read_bytes()) != expected:
                raise ValueError(f'Executed source changed: {name}')
        remote = api.run(f'{config["wandb_entity"]}/{config["wandb_project"]}/{record["wandb_run_id"]}')
        if remote.state != 'finished' or remote.summary['status'] != 'passed':
            raise ValueError(f'W&B run has not finished successfully: {precision}')
        if remote.config['revision'] != config['revision']:
            raise ValueError('Remote model revision mismatch')
        for key in ['n', 'accuracy', 'macro_f1', 'invalid_count', 'invalid_rate']:
            if not math.isclose(remote.summary[key], metrics[key], abs_tol=1e-8):
                raise ValueError(f'Remote metric mismatch: {precision}/{key}')
        artifacts = [{'name': a.name, 'digest': a.digest} for a in remote.logged_artifacts()
                     if a.type == 'runtime-evidence']
        if not artifacts:
            raise ValueError(f'No uploaded runtime evidence: {precision}')
        verification[precision] = dict(state=remote.state, url=record['wandb_url'],
                                       metrics_match=True, artifacts=artifacts)
        print(f'PASS: W&B {precision} finished, metrics match, evidence artifact exists', flush=True)
    (root / 'wandb-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    with wandb.init(entity=config['wandb_entity'], project=config['wandb_project'],
                    job_type='step3-evidence', group=f'step3-{root.name}',
                    name=f'step3-{root.name}-evidence', config=config,
                    tags=['step3', 'evidence'],
                    settings=wandb.Settings(disable_git=True, disable_code=True)) as run:
        artifact = wandb.Artifact(f'step3-checkpoint-{root.name}', type='runtime-checkpoint')
        artifact.add_dir(str(root), name=f'runs/{root.name}')
        for name in ['configs/step3.json', 'configs/data_policy.json', 'requirements-cluster.txt',
                     'scripts/evaluate.py', 'scripts/verify_data.py', 'scripts/publish_step3.py',
                     'tests/test_evaluate.py', 'cluster/step3.sbatch', 'cluster/publish_step3.sbatch',
                     'experiments/step3/README.md',
                     'experiments/step3/summary.json', 'experiments/step3/evidence/parser-recheck.json']:
            artifact.add_file(str(REPO / name), name=name)
        run.log_artifact(artifact).wait()
        run.summary.update(dict(status='verified', model_runs_verified=len(verification),
                                source_images_verified=data_check['images_verified'],
                                unit_tests_passed=int(test_count.group(1)), scope='Step 3 only'))
        receipt = dict(wandb_run_id=run.id, wandb_url=run.url,
                       artifact_name=artifact.name, artifact_digest=artifact.digest,
                       model_runs_verified=verification)
    (root / 'evidence-upload.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    print('PASS: complete Step 3 evidence bundle uploaded')


if __name__ == '__main__':
    main()
