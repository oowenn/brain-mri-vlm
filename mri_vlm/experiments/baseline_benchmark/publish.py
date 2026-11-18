#!/usr/bin/env python3
"""Verify remote Step 6 results and archive completed scheduler/run evidence."""
import argparse
import csv
import json
from pathlib import Path
import re
import tempfile

from mri_vlm import naming
from mri_vlm.common import assert_matches, passed_tests, PREPARED, REPO, sha
from mri_vlm.experiments.baseline_benchmark.verify import verify_run


def main():
    import wandb
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,default=PREPARED)
    args=parser.parse_args(); root=args.run_dir.resolve()
    report=verify_run(root,args.prepared)
    record=json.loads((root/'run.json').read_text()); cfg=record['config']
    with (root/'slurm-accounting.tsv').open() as stream:
        job=next(r for r in csv.DictReader(stream,delimiter='|') if r['JobIDRaw']==root.name)
    if job['State']!='COMPLETED' or job['ExitCode']!='0:0':
        raise ValueError('GPU job did not complete')
    tests_passed=passed_tests(root/'tests.txt')
    api=wandb.Api(); project=f'{cfg["wandb_entity"]}/{cfg["wandb_project"]}'
    remote=api.run(f'{project}/{record["wandb_run_id"]}')
    if remote.state!='finished' or remote.summary['status']!='passed':
        raise ValueError('W&B benchmark run is not finished/passed')
    for key,value in cfg.items():
        assert_matches(remote.config[key],value,f'config/{key}')
    expected=dict(selected_batch_size=record['selected_batch_size'],macro_f1=report['quality_metrics']['macro_f1'],
                  accuracy=report['quality_metrics']['accuracy'],invalid_count=report['quality_metrics']['invalid_count'],
                  **{k:report['baseline'][k] for k in ['mean_batch_ms','p50_batch_ms','p95_batch_ms','mean_amortized_image_ms',
                                                     'throughput_images_s','peak_allocated_bytes','peak_reserved_bytes']})
    expected.update({f'recall/{label}':v['recall'] for label,v in report['quality_metrics']['per_class'].items()})
    for key,value in expected.items():
        assert_matches(remote.summary[key],value,f'remote/{key}')
    artifacts=[a for a in remote.logged_artifacts() if a.type=='runtime-evidence' and a.name.split(':')[0]==naming.artifact('step6',root.name,'evidence')]
    if len(artifacts)!=1:
        raise ValueError('Expected one complete baseline evidence artifact')
    artifact=artifacts[0]
    hashes=json.loads((root/'artifact-files.json').read_text())
    hashes['artifact-files.json']=sha((root/'artifact-files.json').read_bytes())
    with tempfile.TemporaryDirectory(prefix='baseline-benchmark-verify-') as temporary:
        restored=Path(artifact.download(root=temporary))
        actual={str(p.relative_to(restored)):sha(p.read_bytes()) for p in restored.rglob('*') if p.is_file()}
        if actual!=hashes:
            raise ValueError('Restored baseline artifact differs from recorded hashes')
    verification=dict(status='passed',run_state=remote.state,config_and_metrics_match=True,
                      artifact_name=artifact.name,artifact_digest=artifact.digest,downloaded_files_verified=len(hashes))
    (root/'wandb-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (root/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    with wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],name=naming.run_name('step6','evidence',root.name),
                    group=naming.group('step6',root.name),job_type=f'{naming.slug("step6")}-evidence',config=cfg,
                    tags=[naming.slug('step6'),'evidence'],
                    settings=wandb.Settings(disable_git=True,disable_code=True)) as run:
        run.use_artifact(f'{project}/{artifact.name}')
        bundle=wandb.Artifact(naming.artifact('step6',root.name,'archive'),type='runtime-checkpoint')
        bundle.add_dir(str(root),name=f'runs/{root.name}')
        run.log_artifact(bundle).wait()
        run.summary.update(dict(status='verified',unit_tests_passed=tests_passed,**expected))
        receipt=dict(wandb_run_id=run.id,wandb_url=run.url,artifact_name=bundle.name,artifact_digest=bundle.digest,
                     benchmark_run_url=record['wandb_url'])
    (root/'evidence-upload.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    print('PASS: Step 6 W&B results, downloaded evidence, and complete checkpoint verified')


if __name__=='__main__':
    main()
