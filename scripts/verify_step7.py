#!/usr/bin/env python3
"""Recompute Step 7 timing accounting, trace activity, and unchanged-output evidence."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import statistics

from benchmark_utils import scheduled_batch,workload
from profiling_utils import PRIMARY,trace_summary,nvidia_uuid
from verify_data import REPO,read_csv,sha,validate_manifests
from verify_step4 import assert_matches
from verify_step6 import verify_timings


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def output_equal(a,b):
    for key in ('paths','input_tokens','generated_token_counts'):
        if a[key]!=b[key]:
            raise ValueError('Input/output metadata changed: '+key)
    if len(a['predictions'])!=len(b['predictions']):
        raise ValueError('Prediction coverage changed')
    for x,y in zip(a['predictions'],b['predictions']):
        for key in ('raw_response','prediction','generated_token_ids'):
            if x[key]!=y[key]:
                raise ValueError('Generated answer changed: '+key)


def verify_run(root,prepared):
    validate_manifests(prepared)
    record=read(root/'run.json');cfg=record['config']
    if record['status'] not in ('running','passed') or record['test_set_inference']:
        raise ValueError('Run failed or used the test set')
    if cfg!=read(root/'source/configs/step7.json'):
        raise ValueError('Profile config differs from execution snapshot')
    bc=read(root/'source'/cfg['baseline_config'])
    ref=REPO/cfg['reference_run']; frozen=read(ref/'frozen-baseline.json')
    if bc!=frozen['config'] or cfg['batch_size']!=frozen['batch_size'] or cfg['seeds']!=bc['benchmark']['baseline_seeds']:
        raise ValueError('Frozen baseline changed')
    if record['reference_sha256']!={n:sha((ref/n).read_bytes()) for n in record['reference_sha256']}:
        raise ValueError('Reference evidence changed')
    expected=read(ref/'checkpoint-verification.json')['files']
    if read(root/'checkpoint-verification.json')['files']!=expected:
        raise ValueError('Checkpoint differs from Step 6')
    data=read(root/'data-verification.json')
    if data['status']!='passed' or data['images_verified']!=7200:
        raise ValueError('Source hashes not verified')
    rows=read_csv(prepared/'validation.csv')
    worker=read(root/'control/run.json')
    if worker['status']!='passed' or worker['config']!=bc or worker['batch_size']!=cfg['batch_size']:
        raise ValueError('Control did not preserve the frozen configuration')
    control=verify_timings(root/'control',bc,rows,'baseline')
    controls=lines(root/'control/timings.jsonl')
    old=lines(ref/'baseline/timings.jsonl')
    if len(controls)!=len(old):
        raise ValueError('Control coverage changed')
    for a,b in zip(controls,old):
        output_equal(a,b)
    lookup={(r['trial'],r['phase'],r['iteration']):r for r in controls}
    reports={};profiles={};traces=[]
    for mode in ('stages','traces'):
        folder=root/mode;worker=read(folder/'run.json')
        if worker['status']!='passed' or worker['config']!=cfg or worker['inference_config']!=bc:
            raise ValueError('Profile worker did not pass with frozen config')
        samples=lines(folder/'samples.jsonl');profiles[mode]=samples
        phases=[('warmup',cfg['warmup_batches'])]
        phases+= [('measured',cfg['stage_batches_per_trial'])] if mode=='stages' else [('profiler_warmup',1),('trace',cfg['trace_batches_per_trial'])]
        wanted=[(t,p,i) for t in range(3) for p,n in phases for i in range(n)]
        if [(r['trial'],r['phase'],r['iteration']) for r in samples]!=wanted:
            raise ValueError('Missing/reordered profile samples')
        for t,seed in enumerate(cfg['seeds']):
            pool=workload(rows,bc['benchmark']['workload_per_class'],seed)
            if read(folder/f'workload-{seed}.json')!=[{k:r[k] for k in ('path','label','sha256','group_id')} for r in pool]:
                raise ValueError('Profile workload differs')
            for r in (r for r in samples if r['trial']==t):
                batch=scheduled_batch(pool,cfg['batch_size'],r['iteration'])
                if r['seed']!=seed or r['paths']!=[x['path'] for x in batch]:
                    raise ValueError('Profile input identities differ')
                for x,y in zip(r['predictions'],batch):
                    if x['path']!=y['path'] or x['true_label']!=y['label']:
                        raise ValueError('Prediction/image association changed')
                phase='warmup' if r['phase']=='warmup' else 'measured'
                output_equal(r,lookup[t,phase,r['iteration']])
                c=r['forward_calls'];ns=Counter()
                for s in r['spans']:
                    if s['wall_ns']<0 or s['device_ms']<0:
                        raise ValueError('Negative span')
                    ns[s['name']]+=s['wall_ns']
                if len(c)!=max(r['generated_token_counts']) or c[0]['sequence_length']!=r['padded_input_tokens'] or any(x['sequence_length']!=1 or x['pixels'] for x in c[1:]):
                    raise ValueError('Native cached decoding not preserved')
                if sum(s['name']=='vision' for s in r['spans'])!=1:
                    raise ValueError('Vision encoding repeated')
                exclusive={k:ns[k] for k in PRIMARY}
                exclusive['prefill']=ns['first_forward']-ns['vision']
                exclusive['other']=r['elapsed_ns']-sum(exclusive.values())
                if any(v<0 for v in exclusive.values()) or sum(exclusive.values())!=r['elapsed_ns']:
                    raise ValueError('Invalid exclusive stage accounting')
                assert_matches(r['stages']['wall_ns'],exclusive)
                if not 0<r['peak_allocated_bytes']<=r['peak_reserved_bytes']:
                    raise ValueError('Invalid CUDA memory accounting')
            if mode=='traces':
                with gzip.open(folder/f'trace-{seed}.json.gz','rt') as stream:
                    trace=json.load(stream)
                report=trace_summary(trace)
                if report['batch_windows']!=cfg['trace_batches_per_trial'] or not 0<report['gpu_activity_fraction']<=1:
                    raise ValueError('Invalid trace windows/GPU activity')
                required={'mri/batch','mri/preprocessing','mri/transfer','mri/vision','mri/first_forward','mri/decode'}
                if not required.issubset(report['annotation_names']):
                    raise ValueError('Missing stage annotations')
                assert_matches(read(folder/f'trace-{seed}-summary.json'),report)
                traces.append(dict(seed=seed,**report))
        with (folder/'gpu-samples.csv').open() as stream:
            gpu=list(csv.DictReader(stream,skipinitialspace=True))
        if not gpu:
            raise ValueError('Missing GPU telemetry')
        uuid=read(folder/'environment.json')['gpu_uuid']
        stats={}
        for key in ('gpu_utilization_percent','memory_controller_utilization_percent','memory_used_MiB','power_W'):
            values=[float(r[key]) for r in gpu]
            stats[key]=dict(mean=statistics.mean(values),min=min(values),max=max(values))
        if any(r['uuid']!=nvidia_uuid(uuid) for r in gpu):
            raise ValueError('Telemetry GPU differs')
        reports[mode]=dict(samples=len(gpu),scope='entire worker sampling period, including warmups and output writes',values=stats)
        runtime=read(folder/'model-runtime.json')
        if not runtime['all_parameters_fp16_cuda'] or not runtime['plain_merged_model']:
            raise ValueError('Model changed')
        if runtime['source_sha256']!=sha((folder/'installed-modeling-idefics3.py').read_bytes()):
            raise ValueError('Installed source hash mismatch')
    envs=[read(root/m/'environment.json') for m in ('control','stages','traces')]
    if any(e['gpu']['name']!=bc['gpu'] or e['torch_threads']!=4 for e in envs):
        raise ValueError('Expected the frozen A6000 and four CPU threads')
    for key in ('gpu_uuid','torch_threads','matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark'):
        if any(e[key]!=envs[0][key] for e in envs):
            raise ValueError('Worker environment changed: '+key)
    measured=[r for r in profiles['stages'] if r['phase']=='measured']
    total=sum(r['elapsed_ns'] for r in measured)
    stage={}
    for key in PRIMARY+['other']:
        values=[r['stages']['wall_ns'][key]/1e6 for r in measured]
        stage[key]=dict(mean_ms=statistics.mean(values),p50_ms=statistics.median(values),
                        p95_ms=statistics.quantiles(values,n=100,method='inclusive')[94],
                        fraction=sum(values)/(total/1e6))
    summary=dict(measurements=len(measured),mean_batch_ms=total/len(measured)/1e6,stages=stage,
                 dominant_stage=max(PRIMARY,key=lambda k:stage[k]['mean_ms']),
                 peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in measured),
                 peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in measured))
    assert_matches(read(root/'stages/summary.json'),summary)
    matched=[lookup[r['trial'],'measured',r['iteration']]['elapsed_ns'] for r in measured]
    sub={}
    for name in ('image_read','processor','postprocessing'):
        sub[name]=statistics.mean(sum(s['wall_ns'] for s in r['spans'] if s['name']==name)/1e6 for r in measured)
    return dict(status='passed',control=control,stage_profile=summary,cpu_substage_mean_ms=sub,
                control_step6_exact_batches=len(controls),
                instrumented_exact_batches=sum(len(x) for x in profiles.values()),
                matched_control_mean_ms=statistics.mean(matched)/1e6,
                synchronized_stage_overhead_fraction=total/sum(matched)-1,
                control_drift_fraction=control['mean_batch_ms']/read(ref/'baseline/summary.json')['mean_batch_ms']-1,
                trace_gpu_activity_fraction=sum(t['gpu_activity_busy_us'] for t in traces)/sum(t['batch_window_us'] for t in traces),
                traces=traces,gpu_telemetry=reports,
                quality_reference=read(ref/'quality/metrics.json'),
                quality_scope='Inherited Step 6 full validation; no new full quality run; exact matched token equivalence checked',
                test_set_inference=False,optimizations_implemented=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,default=REPO/'data/prepared-v3-conservative-patients')
    args=parser.parse_args()
    report=verify_run(args.run_dir,args.prepared)
    (args.run_dir/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: Step 7 stage accounting, native cached outputs, traces, and matched control independently verified')


if __name__=='__main__':
    main()
