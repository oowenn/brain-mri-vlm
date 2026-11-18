#!/usr/bin/env python3
"""Independently verify Step 6 workload coverage, timing arithmetic, selection, and quality."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import statistics

from mri_vlm import naming
from mri_vlm.common import PREPARED, read_csv, REPO, sha, assert_matches
from mri_vlm.data.verify import validate_manifests
from mri_vlm.inference.evaluate import select_sample
from mri_vlm.inference.prediction_check import verify_predictions
from mri_vlm.inference.workload import scheduled_batch, workload


def read(path):
    return json.loads(path.read_text())


def assert_summary(actual, expected):
    assert_matches({k:v for k,v in actual.items() if k!='trials'},
                   {k:v for k,v in expected.items() if k!='trials'})
    if len(actual['trials'])!=len(expected['trials']):
        raise ValueError('Trial summary coverage differs')
    for a, e in zip(actual['trials'], expected['trials']):
        assert_matches(a, e, 'trial')


def verify_timings(folder, cfg, rows, mode):
    result=read(folder/'run.json')
    records=[json.loads(line) for line in (folder/'timings.jsonl').read_text().splitlines()]
    b=result['batch_size']; bc=cfg['benchmark']
    seeds=bc['sweep_seeds'] if mode=='sweep' else bc['baseline_seeds']
    count=bc['sweep_measured_batches'] if mode=='sweep' else bc['baseline_measured_batches']
    expected=[(trial,phase,i) for trial in range(len(seeds))
              for phase,n in [('warmup',bc['warmup_batches']),('measured',count)] for i in range(n)]
    if [(r['trial'],r['phase'],r['iteration']) for r in records]!=expected:
        raise ValueError('Missing, duplicated, or reordered timing samples')
    for trial,seed in enumerate(seeds):
        pool=workload(rows,bc['workload_per_class'],seed)
        if read(folder/f'workload-{seed}.json') != [{k:r[k] for k in ('path','label','sha256','group_id')} for r in pool]:
            raise ValueError('Saved workload differs from the fixed validation sample')
        if Counter(r['label'] for r in pool)!=dict.fromkeys(cfg['labels'],bc['workload_per_class']):
            raise ValueError('Unbalanced timing workload')
        for r in [x for x in records if x['trial']==trial]:
            if r['seed']!=seed or r['images']!=b or r['paths']!=[x['path'] for x in scheduled_batch(pool,b,r['iteration'])]:
                raise ValueError('Timing sample input identities differ')
            if not isinstance(r['elapsed_ns'],int) or r['elapsed_ns']<=0:
                raise ValueError('Nonpositive/noninteger elapsed time')
            if not 0 < r['peak_allocated_bytes'] <= r['peak_reserved_bytes'] <= result['total_memory_bytes']:
                raise ValueError('Invalid CUDA memory accounting')
            if len(r['input_tokens'])!=b or len(r['predictions'])!=b or len(r['generated_token_counts'])!=b:
                raise ValueError('Incomplete timed output evidence')
            for prediction,n in zip(r['predictions'],r['generated_token_counts']):
                match=re.fullmatch(r'(glioma|meningioma|notumor|pituitary)\.?',prediction['raw_response'].strip().lower())
                if prediction['prediction']!=(match.group(1) if match else 'INVALID') or len(prediction['generated_token_ids'])!=n or not 1<=n<=cfg['max_new_tokens']:
                    raise ValueError('Timed response or token-count mismatch')
    measured=[r for r in records if r['phase']=='measured']
    ms=[r['elapsed_ns']/1_000_000 for r in measured]
    mean=sum(ms)/len(ms)
    trial_stats=[]
    for trial in range(len(seeds)):
        items=[r for r in measured if r['trial']==trial]
        elapsed=sum(r['elapsed_ns'] for r in items)
        trial_stats.append(dict(trial=trial,samples=len(items),mean_batch_ms=elapsed/len(items)/1_000_000,
                                throughput_images_s=len(items)*b/(elapsed/1_000_000_000)))
    p50=statistics.median(ms)
    p95=statistics.quantiles(ms,n=100,method='inclusive')[94]
    independent=dict(batch_size=b,measurements=len(measured),images=len(measured)*b,
                     mean_batch_ms=mean,p50_batch_ms=p50,p95_batch_ms=p95,
                     mean_amortized_image_ms=mean/b,p50_amortized_image_ms=p50/b,p95_amortized_image_ms=p95/b,
                     throughput_images_s=len(measured)*b/(sum(r['elapsed_ns'] for r in measured)/1_000_000_000),
                     peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in measured),
                     peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in measured),trials=trial_stats,
                     trial_mean_batch_ms_std=statistics.stdev(t['mean_batch_ms'] for t in trial_stats) if len(trial_stats)>1 else 0.0)
    for saved in (read(folder/'summary.json'), result['summary']):
        assert_summary(saved, independent)
    return independent


def verify_run(root, prepared):
    validate_manifests(prepared)
    record=read(root/'run.json'); cfg=record['config']; bc=cfg['benchmark']
    if record['status']!='passed' or record['test_set_inference']:
        raise ValueError('Step 6 did not pass')
    if cfg!=read(naming.snapshot(root,naming.config_path('step6'))):
        raise ValueError('Run config differs from frozen source')
    prior=read(naming.snapshot(root,naming.config_path('step4')))
    if any(cfg[k]!=v for k,v in prior.items() if k not in ('stage','batch_size')):
        raise ValueError('Prompt, model, parser, or generation policy changed')
    if cfg['reference_batch_size']!=1 or cfg['padding_side']!='left':
        raise ValueError('Unexpected batched-generation policy')
    if record['validation_manifest_sha256']!=sha((prepared/'validation.csv').read_bytes()):
        raise ValueError('Frozen validation manifest changed')
    rows=sorted(read_csv(prepared/'validation.csv'),key=lambda r:r['path'])
    data=read(root/'data-verification.json')
    if data['status']!='passed' or data['images_verified']!=7200:
        raise ValueError('Source hashes not verified')
    ref=REPO/naming.current_path(cfg['reference_run'])
    expected={k.removeprefix('merged-fp16/'):v for k,v in read(ref/'checkpoint-files.json').items() if k.startswith('merged-fp16/')}
    checkpoint=read(root/'checkpoint-verification.json')
    if checkpoint['status']!='passed' or checkpoint['files']!=expected:
        raise ValueError('Merged checkpoint provenance differs')
    audit=read(root/'audit/reference-check.json')
    probes=select_sample(rows,2,cfg['seed'])
    old={r['path']:r for r in [json.loads(x) for x in (ref/'validation/merged-fp16/predictions.jsonl').read_text().splitlines()]}
    if audit['status']!='passed' or audit['exact_batch_one_matches']!=8 or [r['path'] for r in audit['singles']]!=[r['path'] for r in probes]:
        raise ValueError('Batch-one checkpoint equivalence missing')
    for r in audit['singles']:
        for key in old[r['path']]:
            if r[key]!=old[r['path']][key]:
                raise ValueError('Batch-one output/input metadata differs from Step 5')
    if read(root/'audit/input-audit.json')['status']!='passed':
        raise ValueError('Batched token/image association audit failed')
    candidates=read(root/'sweep.json')
    order=list(bc['candidate_batch_sizes']); random.Random(bc['candidate_order_seed']).shuffle(order)
    if [r['batch_size'] for r in candidates]!=order or read(root/'sweep-order.json')!=order:
        raise ValueError('Sweep coverage or order differs')
    checked={}
    for result in candidates:
        b=result['batch_size']; folder=root/f'sweep-b{b:03}'
        saved=read(folder/'run.json')
        if {k:v for k,v in result.items() if k!='process_exit_code'}!=saved:
            raise ValueError('Sweep summary differs from worker result')
        if result['status']=='oom':
            if result['process_exit_code']!=42 or 'OutOfMemoryError' not in result['error']:
                raise ValueError('Candidate failure is not a documented GPU OOM')
        elif result['status']=='passed' and result['process_exit_code']==0:
            checked[str(b)]=verify_timings(folder,cfg,rows,'sweep')
        else:
            raise ValueError('Unexpected candidate failure')
    eligible=[c for c in candidates if c['status']=='passed' and c['summary']['peak_allocated_bytes']<=bc['selection_memory_fraction']*c['total_memory_bytes']]
    best=max(c['summary']['throughput_images_s'] for c in eligible)
    selected=min(c['batch_size'] for c in eligible if c['summary']['throughput_images_s']>=best*(1-bc['selection_throughput_tolerance']))
    frozen=read(root/'frozen-baseline.json')
    if record['selected_batch_size']!=selected or frozen['batch_size']!=selected or frozen['config']!=cfg or frozen['checkpoint_files']!=expected:
        raise ValueError('Frozen batch-size selection differs from the predeclared rule')
    for folder in [root/'audit',root/'quality',root/'baseline']+[root/f'sweep-b{b:03}' for b in order if str(b) in checked]:
        runtime=read(folder/'environment.json'); worker=read(folder/'run.json')
        if runtime['gpu']['name']!=cfg['gpu'] or worker['config']!=cfg or worker['status']!='passed':
            raise ValueError('Worker runtime/configuration differs')
        model=read(folder/'model-runtime.json')
        if not model['all_parameters_fp16'] or not model['all_parameters_cuda0'] or model['remaining_lora_modules']:
            raise ValueError('Worker did not load a plain merged FP16 model')
    environments=[read(root/f'sweep-b{b:03}/environment.json') for b in order if str(b) in checked]+[read(root/'baseline/environment.json')]
    for key in ('gpu_uuid','torch_threads','cuda_visible_devices','matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark'):
        if any(e[key]!=environments[0][key] for e in environments):
            raise ValueError(f'Benchmark environment changed: {key}')
    metrics=read(root/'quality/metrics.json')
    predictions=[json.loads(x) for x in (root/'quality/predictions.jsonl').read_text().splitlines()]
    verify_predictions(rows,predictions,read(root/'quality/selected-images.json'),metrics)
    if read(root/'quality/run.json')['batch_size']!=selected or read(root/'baseline/run.json')['batch_size']!=selected:
        raise ValueError('Quality/timing batch differs from the frozen selection')
    gate=read(root/'quality-gate.json'); prior_metrics=read(ref/'validation/merged-fp16/metrics.json')
    if gate['status']!='passed' or metrics['macro_f1']<prior_metrics['macro_f1']-bc['quality_max_macro_f1_drop'] or metrics['invalid_count']>bc['quality_max_invalid_count']:
        raise ValueError('Full-validation quality gate failed')
    assert_matches(record['quality_metrics'],metrics)
    baseline=verify_timings(root/'baseline',cfg,rows,'baseline')
    assert_summary(record['baseline_summary'],baseline)
    return dict(status='passed',selected_batch_size=selected,checkpoint_files_verified=len(expected),
                sweep=checked,baseline=baseline,quality_metrics=metrics,validation_images=579,
                exact_batch_one_reference_matches=8,test_set_inference=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,default=PREPARED)
    args=parser.parse_args()
    report=verify_run(args.run_dir,args.prepared)
    (args.run_dir/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: Step 6 timing arithmetic, workloads, selection, checkpoint provenance, and quality independently verified')


if __name__=='__main__':
    main()
