#!/usr/bin/env python3
"""Run isolated batch-size candidates, freeze a baseline, and track Step 6."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import traceback

from benchmark_utils import choose_batch
from evaluate import write_json
from verify_data import REPO, sha, validate_manifests


def verify_checkpoint(checkpoint, reference):
    inventory = json.loads((reference/'checkpoint-files.json').read_text())
    expected = {k.removeprefix('merged-fp16/'):v for k,v in inventory.items() if k.startswith('merged-fp16/')}
    if {str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()} != set(expected):
        raise ValueError('Merged checkpoint file set differs from Step 5')
    for name, info in expected.items():
        p = checkpoint/name
        if p.stat().st_size != info['bytes'] or sha(p.read_bytes()) != info['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    return dict(status='passed', reference=str(reference), files=expected, files_verified=len(expected))


def main():
    import wandb
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'configs/step6.json')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,default=REPO/'data/prepared-v3-conservative-patients')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    validate_manifests(args.prepared)
    reference=REPO/cfg['reference_run']
    write_json(args.output/'checkpoint-verification.json',verify_checkpoint(args.checkpoint,reference))
    record=dict(status='running',config=cfg,started_utc=datetime.now(timezone.utc).isoformat(),
                validation_manifest_sha256=sha((args.prepared/'validation.csv').read_bytes()),
                base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                checkpoint=str(args.checkpoint.resolve()),test_set_inference=False)
    run=None
    def worker(mode,batch,folder):
        target=args.output/folder
        log=args.output/'worker-logs'/f'{folder}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(REPO/'scripts/benchmark.py'),'--mode',mode,'--batch-size',str(batch),
                 '--config',str(args.config),'--checkpoint',str(args.checkpoint),'--source',str(args.source),
                 '--prepared',str(args.prepared),'--output',str(target)]
        write_json(args.output/'active-worker.json',dict(mode=mode,batch_size=batch,folder=folder,command=command))
        print(f'Starting {mode} B={batch}; log {log}',flush=True)
        with log.open('x') as stream:
            status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
        if not (target/'run.json').exists():
            raise RuntimeError(f'{mode} exited {status} before writing a run record; inspect {log}')
        result=json.loads((target/'run.json').read_text())
        result['process_exit_code']=status
        if status != 0 and not (mode=='sweep' and status==42 and result['status']=='oom'):
            raise RuntimeError(f'{mode} failed with exit {status}; inspect {log}')
        if status==0 and result['status']!='passed':
            raise RuntimeError('Worker did not pass')
        print(f'Finished {mode} B={batch}: {result["status"]}',flush=True)
        return result
    try:
        run=wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],name=f'step6-{args.output.name}-baseline',
                        group=f'step6-{args.output.name}',job_type='fp16-baseline-benchmark',config=cfg,
                        tags=['step6','fp16','benchmark'],settings=wandb.Settings(disable_git=True,disable_code=True))
        model_artifact=run.use_artifact(cfg['checkpoint_artifact'])
        if model_artifact.digest!=cfg['checkpoint_artifact_digest']:
            raise ValueError('W&B model artifact digest differs from Step 5')
        run.use_artifact(cfg['data_artifact'])
        record.update(wandb_run_id=run.id,wandb_url=run.url)
        write_json(args.output/'run.json',record)
        worker('audit',1,'audit')
        order=list(cfg['benchmark']['candidate_batch_sizes'])
        random.Random(cfg['benchmark']['candidate_order_seed']).shuffle(order)
        candidates=[]
        write_json(args.output/'sweep-order.json',order)
        for batch in order:
            result=worker('sweep',batch,f'sweep-b{batch:03}')
            candidates.append(result)
            write_json(args.output/'sweep.json',candidates)
            values=dict(batch_size=batch,status=result['status'])
            if result['status']=='passed':
                values.update({k:result['summary'][k] for k in ['mean_batch_ms','p95_batch_ms','throughput_images_s','peak_allocated_bytes','peak_reserved_bytes']})
            run.log({f'sweep/{k}':v for k,v in values.items()})
        bc=cfg['benchmark']
        selected=choose_batch(candidates,bc['selection_memory_fraction'],bc['selection_throughput_tolerance'])
        frozen=dict(status='selected-before-final-measurements',batch_size=selected,rule=bc['selection'],
                    candidate_order=order,selection_source='sweep.json',config=cfg,
                    checkpoint_files=json.loads((args.output/'checkpoint-verification.json').read_text())['files'])
        write_json(args.output/'frozen-baseline.json',frozen)
        worker('quality',selected,'quality')
        quality=json.loads((args.output/'quality/metrics.json').read_text())
        prior=json.loads((reference/'validation/merged-fp16/metrics.json').read_text())
        gate=dict(reference_macro_f1=prior['macro_f1'],measured_macro_f1=quality['macro_f1'],
                  allowed_drop=bc['quality_max_macro_f1_drop'],invalid_count=quality['invalid_count'],
                  status='passed' if quality['macro_f1']>=prior['macro_f1']-bc['quality_max_macro_f1_drop']
                         and quality['invalid_count']<=bc['quality_max_invalid_count'] else 'failed')
        write_json(args.output/'quality-gate.json',gate)
        if gate['status']!='passed':
            raise ValueError('Selected batch failed full-validation quality gate; investigate without silently changing batch size')
        baseline=worker('baseline',selected,'baseline')
        record.update(status='passed',finished_utc=datetime.now(timezone.utc).isoformat(),selected_batch_size=selected,
                      baseline_summary=baseline['summary'],quality_metrics=quality)
        write_json(args.output/'run.json',record)
        run.summary.update(dict(status='passed',selected_batch_size=selected,macro_f1=quality['macro_f1'],
                                accuracy=quality['accuracy'],invalid_count=quality['invalid_count'],
                                **{k:baseline['summary'][k] for k in ['mean_batch_ms','p50_batch_ms','p95_batch_ms',
                                     'mean_amortized_image_ms','throughput_images_s','peak_allocated_bytes','peak_reserved_bytes']}))
        for label,values in quality['per_class'].items():
            run.summary[f'recall/{label}']=values['recall']
        stable={str(p.relative_to(args.output)):sha(p.read_bytes()) for p in sorted(args.output.rglob('*'))
                if p.is_file() and p.name!='console.log'}
        write_json(args.output/'artifact-files.json',stable)
        artifact=wandb.Artifact(f'step6-{args.output.name}-baseline',type='runtime-evidence')
        for name in stable:
            artifact.add_file(str(args.output/name),name=name)
        artifact.add_file(str(args.output/'artifact-files.json'),name='artifact-files.json')
        run.log_artifact(artifact).wait()
        run.finish(); run=None
        print('PASS: Step 6 sweep, frozen baseline, quality gate, and repeated timings completed',flush=True)
    except Exception:
        record.update(status='failed',error=traceback.format_exc())
        write_json(args.output/'run.json',record)
        if run is not None:
            run.summary.update(dict(status='failed',error=record['error']))
            run.finish(exit_code=1)
        raise


if __name__=='__main__':
    main()
