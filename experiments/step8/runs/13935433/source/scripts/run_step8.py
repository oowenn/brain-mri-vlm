#!/usr/bin/env python3
"""Matched 2048/1024 inference experiment on the saved merged checkpoint."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

from evaluate import write_json
from run_step6 import verify_checkpoint
from step8_utils import inference_config
from verify_data import REPO, sha, validate_manifests


def remote_metrics(report):
    values = dict(status='passed', quality_gate_passed=report['comparison']['quality_gate_passed'],
                  preliminary_success=report['comparison']['preliminary_success'],
                  speedup=report['comparison']['speedup'],
                  macro_f1_change=report['comparison']['macro_f1_change'])
    for name in ('baseline', 'image1024'):
        for key in ('mean_batch_ms','p50_batch_ms','p95_batch_ms','throughput_images_s','peak_allocated_bytes','peak_reserved_bytes'):
            values[f'{name}/{key}'] = report['timing'][name][key]
        for key in ('accuracy','macro_f1','invalid_count'):
            values[f'{name}/{key}'] = report['quality'][name][key]
        for label, item in report['quality'][name]['per_class'].items():
            values[f'{name}/recall/{label}'] = item['recall']
        for stage, item in report['stages'][name]['stages'].items():
            values[f'{name}/stage_ms/{stage}'] = item['mean_ms']
    return values


def main():
    import wandb
    from verify_step8 import verify_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO/'configs/step8.json')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=REPO/'data/prepared-v3-conservative-patients')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.output.resolve()
    cfg = json.loads(args.config.read_text()); bc = json.loads((REPO/cfg['baseline_config']).read_text())
    frozen = json.loads((REPO/cfg['reference_run']/'frozen-baseline.json').read_text())
    if frozen['config'] != bc or frozen['batch_size'] != cfg['batch_size']:
        raise ValueError('Step 6 frozen baseline changed')
    validate_manifests(args.prepared)
    write_json(root/'checkpoint-verification.json', verify_checkpoint(args.checkpoint, REPO/bc['reference_run']))
    (root/'configs').mkdir(); (root/'worker-logs').mkdir()
    record = dict(status='running', config=cfg, started_utc=datetime.now(timezone.utc).isoformat(),
                  base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  validation_manifest_sha256=sha((args.prepared/'validation.csv').read_bytes()),
                  test_set_inference=False, worker_order=[])
    run = None

    def worker(name, mode, config, profile=False):
        config_path = root/'configs'/f'{name}.json'; write_json(config_path, config)
        command = [sys.executable,'-u',str(REPO/'scripts'/('profile_baseline.py' if profile else 'benchmark.py')),
                   '--mode',mode,'--config',str(config_path),'--source',str(args.source),
                   '--checkpoint',str(args.checkpoint),'--prepared',str(args.prepared),'--output',str(root/name)]
        if not profile:
            command += ['--batch-size',str(cfg['batch_size'])]
        write_json(root/'active-worker.json', dict(name=name, command=command))
        record['worker_order'].append(name); write_json(root/'run.json', record)
        print('Starting', name, flush=True)
        with (root/'worker-logs'/f'{name}.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Finished', name, flush=True)

    try:
        run = wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],name=f'step8-{root.name}-image-budget',
                         group=f'step8-{root.name}',job_type='matched-optimization',config=cfg,
                         tags=['step8','step9-preliminary','image-budget','fp16'],
                         settings=wandb.Settings(disable_git=True,disable_code=True))
        model = run.use_artifact(bc['checkpoint_artifact'])
        if model.digest != bc['checkpoint_artifact_digest']:
            raise ValueError('Checkpoint artifact changed')
        run.use_artifact(bc['data_artifact'])
        record.update(wandb_run_id=run.id,wandb_url=run.url);write_json(root/'run.json',record)
        for seed, order in zip(cfg['seeds'],cfg['trial_orders']):
            for name in order:
                worker(f'{name}-seed{seed}', 'baseline', inference_config(bc,cfg,name,seed))
                summary = json.loads((root/f'{name}-seed{seed}'/'summary.json').read_text())
                run.log(dict(seed=seed,variant=name,mean_batch_ms=summary['mean_batch_ms'],throughput_images_s=summary['throughput_images_s']))
        for name in cfg['longest_edges']:
            worker(f'{name}-quality','quality',inference_config(bc,cfg,name))
        for name in cfg['longest_edges']:
            pc = json.loads((REPO/'configs/step7.json').read_text())
            pc['baseline_config'] = str(root/'configs'/f'{name}-quality.json')
            pc['performance_boundary'] = 'Diagnostic only; performance comes from matched unprofiled Step 8 workers'
            worker(f'{name}-stages','stages',pc,profile=True)
        report = verify_run(root,args.prepared)
        write_json(root/'independent-verification.json',report)
        record.update(status='passed',finished_utc=datetime.now(timezone.utc).isoformat(),comparison=report['comparison'])
        write_json(root/'run.json',record);run.summary.update(remote_metrics(report))
        hashes = {str(p.relative_to(root)):sha(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='console.log'}
        write_json(root/'artifact-files.json',hashes)
        artifact = wandb.Artifact(f'step8-{root.name}-image-budget',type='runtime-evidence')
        for name in hashes:
            artifact.add_file(str(root/name),name=name)
        artifact.add_file(str(root/'artifact-files.json'),name='artifact-files.json')
        run.log_artifact(artifact).wait();run.finish();run=None
        print('PASS: matched experiment completed; quality gate:',report['comparison']['quality_gate_passed'],flush=True)
    except Exception:
        record.update(status='failed',error=traceback.format_exc());write_json(root/'run.json',record)
        if run is not None:
            run.summary.update(dict(status='failed',error=record['error']));run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
