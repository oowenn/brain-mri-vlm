#!/usr/bin/env python3
"""Verify and publish intact Step 7 GPU output after the known annotation-count failure."""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile

from evaluate import write_json
from run_step7 import remote_metrics
from verify_data import REPO,sha
from verify_step4 import assert_matches
from verify_step7 import verify_run


def main():
    import wandb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=REPO/'data/prepared-v3-conservative-patients')
    args=p.parse_args();root=args.run_dir.resolve();recovery=root/'recovery'
    record=json.loads((root/'run.json').read_text());cfg=record['config']
    with (root/'slurm-accounting.tsv').open() as f:
        job=next(r for r in csv.DictReader(f,delimiter='|') if r['JobIDRaw']==root.name)
    if job['State']!='FAILED' or job['ExitCode']!='1:0':
        raise ValueError('Recovery requires the preserved failed postprocessing job')
    report=verify_run(root,args.prepared,recover_trace_annotations=True)
    write_json(root/'recovered-verification.json',report)
    for t in report['traces']:
        write_json(recovery/f'trace-{t["seed"]}-summary.json',t)
    original_commit=(root/'base-commit.txt').read_text().strip()
    sources={}
    for path in sorted((root/'source').rglob('*')):
        if path.is_file():
            name=str(path.relative_to(root/'source'))
            expected=subprocess.check_output(['git','show',f'{original_commit}:{name}'])
            if path.read_bytes()!=expected:
                raise ValueError('Original source provenance changed: '+name)
            sources[name]=sha(expected)
    write_json(root/'source-provenance.json',dict(status='passed',commit=original_commit,files=sources))
    new_commit=(recovery/'base-commit.txt').read_text().strip()
    sources={}
    for path in sorted((recovery/'source').rglob('*')):
        if path.is_file():
            name=str(path.relative_to(recovery/'source'))
            expected=subprocess.check_output(['git','show',f'{new_commit}:{name}'])
            if path.read_bytes()!=expected:
                raise ValueError('Recovery source provenance changed: '+name)
            sources[name]=sha(expected)
    write_json(recovery/'source-provenance.json',dict(status='passed',commit=new_commit,files=sources))
    tests=(recovery/'tests.txt').read_text()
    count=re.search(r'Ran (\d+) tests',tests)
    if not count or not tests.rstrip().endswith('OK'):
        raise ValueError('Recovery tests failed')
    bc=json.loads((root/'source/configs/step6.json').read_text())
    env=json.loads((root/'control/environment.json').read_text())
    tracking=dict(cfg,inference_config=bc,precision='FP16',gpu=env['gpu'],gpu_uuid=env['gpu_uuid'],
                  cpu_threads=env['torch_threads'],measurement_commit=original_commit,
                  recovery_commit=new_commit,measurement_job=root.name,recovery_job=os.environ['SLURM_JOB_ID'])
    values=remote_metrics(report)
    with wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],
        name=f'step7-{root.name}-verified-profile',group=f'step7-{root.name}',
        job_type='profile-evidence-recovery',config=tracking,tags=['step7','profiling','verified-recovery'],
        settings=wandb.Settings(disable_git=True,disable_code=True)) as run:
        model=run.use_artifact(bc['checkpoint_artifact'])
        if model.digest!=bc['checkpoint_artifact_digest']:
            raise ValueError('Checkpoint artifact digest changed')
        run.use_artifact(bc['data_artifact'])
        run.use_artifact(f'{cfg["wandb_entity"]}/{cfg["wandb_project"]}/step6-13932242-baseline:v0')
        recovery_record=dict(status='passed',gpu_job_status='FAILED during final postprocessing',
            all_gpu_workers='passed',original_wandb_run_id=record['wandb_run_id'],
            original_commit=original_commit,recovery_commit=new_commit,
            recovery_job=os.environ['SLURM_JOB_ID'],recovered_utc=datetime.now(timezone.utc).isoformat(),
            correction='Count only CPU user_annotation batch ranges; mirrored gpu_user_annotation ranges are excluded. All other original trace-summary values match.',
            new_inference=False,test_set_inference=False,wandb_run_id=run.id,wandb_url=run.url,metrics=values)
        write_json(recovery/'run.json',recovery_record)
        run.summary.update(dict(status='passed',unit_tests_passed=int(count.group(1)),**values))
        rows=[json.loads(x) for x in (root/'stages/samples.jsonl').read_text().splitlines()]
        for trial in range(3):
            subset=[x for x in rows if x['phase']=='measured' and x['trial']==trial]
            run.log(dict(trial=trial,**{'stage_mean_ms/'+k:sum(r['stages']['wall_ns'][k] for r in subset)/len(subset)/1e6 for k in report['stage_profile']['stages']}))
        stable={str(p.relative_to(root)):sha(p.read_bytes()) for p in sorted(root.rglob('*'))
                if p.is_file() and p!=recovery/'console.log'}
        write_json(root/'artifact-files.json',stable)
        artifact=wandb.Artifact(f'step7-{root.name}-profile',type='runtime-evidence')
        for name in stable:
            artifact.add_file(str(root/name),name=name)
        artifact.add_file(str(root/'artifact-files.json'),name='artifact-files.json')
        run.log_artifact(artifact).wait()
        run_id=run.id
    api=wandb.Api();project=f'{cfg["wandb_entity"]}/{cfg["wandb_project"]}'
    remote=api.run(f'{project}/{run_id}')
    if remote.state!='finished' or remote.summary['status']!='passed':
        raise ValueError('Recovery W&B run did not finish')
    for k,v in tracking.items():
        assert_matches(remote.config[k],v,'config/'+k)
    for k,v in values.items():
        assert_matches(remote.summary[k],v,'summary/'+k)
    restored_artifact=api.artifact(f'{project}/{artifact.name}')
    hashes=dict(stable,**{'artifact-files.json':sha((root/'artifact-files.json').read_bytes())})
    with tempfile.TemporaryDirectory(prefix='step7-recovery-verify-') as temp:
        restored=Path(restored_artifact.download(root=temp))
        actual={str(p.relative_to(restored)):sha(p.read_bytes()) for p in restored.rglob('*') if p.is_file()}
        if actual!=hashes:
            raise ValueError('Downloaded profile evidence differs')
    write_json(root/'wandb-verification.json',dict(status='passed',run_state=remote.state,
        config_and_metrics_match=True,artifact_name=restored_artifact.name,
        artifact_digest=restored_artifact.digest,downloaded_files_verified=len(hashes)))
    with wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],
        name=f'step7-{root.name}-evidence',group=f'step7-{root.name}',job_type='step7-evidence',
        config=tracking,tags=['step7','evidence'],settings=wandb.Settings(disable_git=True,disable_code=True)) as run:
        run.use_artifact(f'{project}/{restored_artifact.name}')
        bundle=wandb.Artifact(f'step7-checkpoint-{root.name}',type='runtime-checkpoint')
        for path in sorted(root.rglob('*')):
            if path.is_file() and path!=recovery/'console.log':
                bundle.add_file(str(path),name=f'runs/{root.name}/{path.relative_to(root)}')
        run.log_artifact(bundle).wait()
        run.summary.update(dict(status='verified',unit_tests_passed=int(count.group(1)),**values))
        receipt=dict(wandb_run_id=run.id,wandb_url=run.url,artifact_name=bundle.name,
                     artifact_digest=bundle.digest,profile_run_url=remote.url,recovery_job=os.environ['SLURM_JOB_ID'])
    write_json(root/'evidence-upload.json',receipt)
    print(json.dumps(receipt,indent=2))
    print('PASS: preserved GPU results recovered, verified, and published without new inference')


if __name__=='__main__':
    main()
