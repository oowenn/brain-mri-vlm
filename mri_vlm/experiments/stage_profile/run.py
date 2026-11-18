#!/usr/bin/env python3
"""Run an unchanged control, stage diagnostics, and representative CUDA traces."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

from mri_vlm import naming
from mri_vlm.common import PREPARED, REPO, sha
from mri_vlm.inference.evaluate import write_json
from mri_vlm.experiments.baseline_benchmark.run import verify_checkpoint
from mri_vlm.experiments.stage_profile.verify import verify_run


def remote_metrics(report):
    values=dict(control_mean_batch_ms=report['control']['mean_batch_ms'],
                control_throughput_images_s=report['control']['throughput_images_s'],
                control_drift_fraction=report['control_drift_fraction'],
                stage_overhead_fraction=report['synchronized_stage_overhead_fraction'],
                trace_gpu_activity_fraction=report['trace_gpu_activity_fraction'],
                reference_macro_f1=report['quality_reference']['macro_f1'],
                dominant_stage=report['stage_profile']['dominant_stage'],
                exact_control_batches=report['control_step6_exact_batches'],
                exact_instrumented_batches=report['instrumented_exact_batches'])
    for name,s in report['stage_profile']['stages'].items():
        values['stage_ms/'+name]=s['mean_ms']
        values['stage_fraction/'+name]=s['fraction']
    return values


def main():
    import wandb
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=REPO/naming.config_path('step7'))
    parser.add_argument('--prepared',type=Path,default=PREPARED)
    args=parser.parse_args();root=args.output
    cfg=json.loads(args.config.read_text());bc=json.loads((REPO/naming.current_path(cfg['baseline_config'])).read_text())
    ref=REPO/naming.current_path(cfg['reference_run'])
    frozen=json.loads((ref/'frozen-baseline.json').read_text())
    if frozen['batch_size']!=cfg['batch_size'] or naming.upgrade_config(frozen['config'])!=naming.upgrade_config(bc):
        raise ValueError('Frozen baseline differs')
    write_json(root/'checkpoint-verification.json',verify_checkpoint(args.checkpoint,REPO/naming.current_path(bc['reference_run'])))
    record=dict(status='running',config=cfg,started_utc=datetime.now(timezone.utc).isoformat(),
                base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                test_set_inference=False,optimizations_implemented=False,
                reference_sha256={n:sha((ref/n).read_bytes()) for n in
                    ('frozen-baseline.json','baseline/timings.jsonl','baseline/summary.json','quality/metrics.json')})
    run=None
    try:
        run=wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],name=naming.run_name('step7','run',root.name),
                       group=naming.group('step7',root.name),job_type=naming.slug('step7'),config=cfg,
                       tags=[naming.slug('step7'),'profiling','fp16'],settings=wandb.Settings(disable_git=True,disable_code=True))
        artifact=run.use_artifact(bc['checkpoint_artifact'])
        if artifact.digest!=bc['checkpoint_artifact_digest']:
            raise ValueError('Checkpoint artifact digest changed')
        run.use_artifact(bc['data_artifact'])
        run.use_artifact(f'{cfg["wandb_entity"]}/{cfg["wandb_project"]}/{naming.artifact("step6","13932242","evidence")}:v0')
        record.update(wandb_run_id=run.id,wandb_url=run.url)
        write_json(root/'run.json',record)
        for mode in ('control','stages','traces'):
            if mode=='control':
                command=[sys.executable,'-u','-m','mri_vlm.inference.benchmark','--mode','baseline',
                         '--batch-size',str(cfg['batch_size']),'--config',str(REPO/naming.current_path(cfg['baseline_config']))]
            else:
                command=[sys.executable,'-u','-m','mri_vlm.inference.stage_profiler','--mode',mode,'--config',str(args.config)]
            command+=['--source',str(args.source),'--checkpoint',str(args.checkpoint),'--prepared',str(args.prepared),'--output',str(root/mode)]
            write_json(root/'active-worker.json',dict(mode=mode,command=command))
            print('Starting',mode,flush=True)
            (root/'worker-logs').mkdir(exist_ok=True)
            with (root/'worker-logs'/f'{mode}.log').open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)
            print('Finished',mode,flush=True)
        report=verify_run(root,args.prepared)
        write_json(root/'independent-verification.json',report)
        record.update(status='passed',finished_utc=datetime.now(timezone.utc).isoformat(),metrics=remote_metrics(report))
        write_json(root/'run.json',record)
        run.summary.update(dict(status='passed',**record['metrics']))
        for t in range(3):
            subset=[json.loads(x) for x in (root/'stages/samples.jsonl').read_text().splitlines()]
            subset=[x for x in subset if x['phase']=='measured' and x['trial']==t]
            run.log(dict(trial=t,**{'stage_mean_ms/'+k:sum(r['stages']['wall_ns'][k] for r in subset)/len(subset)/1e6 for k in report['stage_profile']['stages']}))
        stable={str(p.relative_to(root)):sha(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='console.log'}
        write_json(root/'artifact-files.json',stable)
        bundle=wandb.Artifact(naming.artifact('step7',root.name,'evidence'),type='runtime-evidence')
        for name in stable:
            bundle.add_file(str(root/name),name=name)
        bundle.add_file(str(root/'artifact-files.json'),name='artifact-files.json')
        run.log_artifact(bundle).wait()
        run.finish();run=None
        print('PASS: Step 7 control, stage diagnostics, CUDA traces, and W&B evidence',flush=True)
    except Exception:
        record.update(status='failed',error=traceback.format_exc())
        write_json(root/'run.json',record)
        if run is not None:
            run.summary.update(dict(status='failed',error=record['error']));run.finish(exit_code=1)
        raise


if __name__=='__main__':
    main()
