#!/usr/bin/env python3
"""Verify matched inputs, one-factor settings, quality, timings and stage accounting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from benchmark_utils import summarize
from profiling_utils import PRIMARY
from step8_utils import comparison, inference_config
from verify_data import REPO, read_csv, sha, validate_manifests
from verify_step4 import assert_matches, verify_predictions
from verify_step6 import verify_timings
from verify_step7 import output_equal


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def verify_run(root, prepared):
    validate_manifests(prepared)
    record = read(root/'run.json'); cfg = record['config']
    if record['status'] not in ('running','passed') or record['test_set_inference']:
        raise ValueError('Experiment failed or used test data')
    if cfg != read(root/'source/configs/step8.json'):
        raise ValueError('Config differs from execution source')
    bc = read(root/'source'/cfg['baseline_config']); ref = REPO/cfg['reference_run']
    if read(ref/'frozen-baseline.json')['config'] != bc:
        raise ValueError('Historical baseline configuration changed')
    if record['validation_manifest_sha256'] != sha((prepared/'validation.csv').read_bytes()):
        raise ValueError('Validation manifest changed')
    data = read(root/'data-verification.json'); checkpoint = read(root/'checkpoint-verification.json')
    if data['status'] != 'passed' or data['images_verified'] != 7200:
        raise ValueError('Source verification missing')
    if checkpoint['status'] != 'passed' or checkpoint['files'] != read(ref/'checkpoint-verification.json')['files']:
        raise ValueError('Merged checkpoint changed')
    expected_order = [f'{v}-seed{s}' for s,order in zip(cfg['seeds'],cfg['trial_orders']) for v in order]
    expected_order += [f'{v}-quality' for v in cfg['longest_edges']]+[f'{v}-stages' for v in cfg['longest_edges']]
    if record['worker_order'] != expected_order:
        raise ValueError('Paired order or worker coverage changed')
    rows = sorted(read_csv(prepared/'validation.csv'),key=lambda r:r['path'])
    old = {(r['seed'],r['phase'],r['iteration']):r for r in lines(ref/'baseline/timings.jsonl')}
    original_processor = read(REPO/'experiments/step7/runs/13933296/stages/model-runtime.json')['processor']
    timing = {}; quality = {}; stages = {}; context = {}; all_records = {}; environments = []

    def check_worker(folder, expected, profile=False):
        worker = read(folder/'run.json'); env = read(folder/'environment.json')
        if worker['status'] != 'passed' or worker['test_set_inference']:
            raise ValueError('Worker failed or used test data')
        if worker['inference_config' if profile else 'config'] != expected:
            raise ValueError('Unplanned inference change')
        if not profile and worker['batch_size'] != cfg['batch_size']:
            raise ValueError('Batch size changed')
        model = read(folder/'model-runtime.json')
        if profile:
            if not model['all_parameters_fp16_cuda'] or not model['plain_merged_model']:
                raise ValueError('Profile model changed')
            actual = model['processor']
        else:
            if not model['all_parameters_fp16'] or not model['all_parameters_cuda0'] or model['remaining_lora_modules']:
                raise ValueError('Benchmark model changed')
            actual = read(folder/'processor.json')
            audit = read(folder/'input-audit.json')
            if audit['status'] != 'passed' or audit['images'] != 8 or not audit['exact_effective_token_and_pixel_match']:
                raise ValueError('Processor batch association audit missing')
        wanted = dict(original_processor,size={'longest_edge':expected['processor_longest_edge']})
        assert_matches(actual,wanted,'processor')
        if env['gpu']['name'] != bc['gpu'] or env['torch_threads'] != 4:
            raise ValueError('Expected A6000 and four CPU threads')
        environments.append(env)
        return worker

    for name in cfg['longest_edges']:
        combined = []
        for trial, seed in enumerate(cfg['seeds']):
            folder = root/f'{name}-seed{seed}'; expected = inference_config(bc,cfg,name,seed)
            check_worker(folder,expected)
            verify_timings(folder,expected,rows,'baseline')
            records = lines(folder/'timings.jsonl')
            if name == 'baseline':
                for r in records:
                    output_equal(r,old[seed,r['phase'],r['iteration']])
            combined.extend(dict(r,trial=trial) for r in records)
        all_records[name] = combined; timing[name] = summarize(combined)
        folder = root/f'{name}-quality'; expected = inference_config(bc,cfg,name)
        check_worker(folder,expected)
        quality[name] = read(folder/'metrics.json'); predictions = lines(folder/'predictions.jsonl')
        verify_predictions(rows,predictions,read(folder/'selected-images.json'),quality[name])
        if name == 'baseline':
            previous = lines(ref/'quality/predictions.jsonl')
            if predictions != previous:
                raise ValueError('Fresh baseline full-validation outputs differ from Step 6')
        context[name] = dict(validation_input_token_counts=dict(Counter(p['input_tokens'] for p in predictions)),
                            validation_pixel_shapes=[list(x) for x in sorted({tuple(p['batch_pixel_values_shape']) for p in predictions})],
                            generated_token_range=[min(len(p['generated_token_ids']) for p in predictions),max(len(p['generated_token_ids']) for p in predictions)])
        folder = root/f'{name}-stages'; worker = check_worker(folder,expected,profile=True)
        pc = worker['config']
        frozen_profile = read(root/'source/configs/step7.json')
        for key in frozen_profile:
            if key not in ('baseline_config','performance_boundary') and pc[key] != frozen_profile[key]:
                raise ValueError('Stage diagnostic protocol changed')
        samples = lines(folder/'samples.jsonl')
        expected_sequence = [(t,p,i) for t in range(3) for p in ('warmup','measured') for i in range(10)]
        if [(r['trial'],r['phase'],r['iteration']) for r in samples] != expected_sequence:
            raise ValueError('Missing or duplicated stage samples')
        lookup = {(r['trial'],r['phase'],r['iteration']):r for r in combined}
        for r in samples:
            output_equal(r,lookup[r['trial'],r['phase'],r['iteration']])
            if r['seed'] != cfg['seeds'][r['trial']]:
                raise ValueError('Stage seed changed')
            counts = Counter()
            for span in r['spans']:
                if span['wall_ns'] < 0 or span['device_ms'] < 0:
                    raise ValueError('Negative stage time')
                counts[span['name']] += span['wall_ns']
            exclusive = {k:counts[k] for k in PRIMARY}
            exclusive['prefill'] = counts['first_forward']-counts['vision']
            exclusive['other'] = r['elapsed_ns']-sum(exclusive.values())
            if min(exclusive.values()) < 0:
                raise ValueError('Stage accounting double counts time')
            assert_matches(r['stages']['wall_ns'],exclusive)
            calls = r['forward_calls']
            if len(calls) != max(r['generated_token_counts']) or calls[0]['sequence_length'] != r['padded_input_tokens'] or any(c['sequence_length']!=1 or c['pixels'] for c in calls[1:]):
                raise ValueError('Cached decoding changed')
            if sum(s['name']=='vision' for s in r['spans']) != 1:
                raise ValueError('Vision must run once')
        measured = [r for r in samples if r['phase']=='measured']; total = sum(r['elapsed_ns'] for r in measured)
        detail = {}
        for key in PRIMARY+['other']:
            values = [r['stages']['wall_ns'][key]/1e6 for r in measured]
            detail[key] = dict(mean_ms=statistics.mean(values),p50_ms=statistics.median(values),
                               p95_ms=statistics.quantiles(values,n=100,method='inclusive')[94],fraction=sum(values)/(total/1e6))
        stages[name] = dict(measurements=len(measured),mean_batch_ms=total/len(measured)/1e6,stages=detail,
                            dominant_stage=max(PRIMARY,key=lambda k:detail[k]['mean_ms']),
                            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in measured),
                            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in measured))
        assert_matches(read(folder/'summary.json'),stages[name])
        context[name]['stage_pixel_shapes'] = [list(x) for x in sorted({tuple(r['pixel_values_shape']) for r in measured})]
        context[name]['stage_mean_input_tensor_bytes'] = statistics.mean(r['input_tensor_bytes'] for r in measured)
    for a,b in zip(all_records['baseline'],all_records['image1024']):
        if any(a[k]!=b[k] for k in ('trial','phase','iteration','seed','paths','images')):
            raise ValueError('Baseline and variant workloads are not paired')
    for key in ('hostname','gpu_uuid','torch_threads','slurm_job_id','packages','torch_cuda_build','matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark'):
        if any(e[key] != environments[0][key] for e in environments):
            raise ValueError('Workers changed hardware/environment: '+key)
    result = comparison(timing['baseline'],timing['image1024'],read(ref/'quality/metrics.json')['macro_f1'],quality['baseline'],quality['image1024'],cfg)
    paired = [dict(seed=s,order=cfg['trial_orders'][t],baseline_mean_ms=timing['baseline']['trials'][t]['mean_batch_ms'],
                   variant_mean_ms=timing['image1024']['trials'][t]['mean_batch_ms'],
                   speedup=timing['baseline']['trials'][t]['mean_batch_ms']/timing['image1024']['trials'][t]['mean_batch_ms']) for t,s in enumerate(cfg['seeds'])]
    return dict(status='passed',timing=timing,quality=quality,stages=stages,context=context,comparison=result,paired_trials=paired,
                baseline_exact_timing_batches=180,baseline_exact_validation_images=579,instrumented_exact_batches=120,
                gpu_uuid=environments[0]['gpu_uuid'],hostname=environments[0]['hostname'],test_set_inference=False,
                baseline_drift_fraction=timing['baseline']['mean_batch_ms']/read(ref/'baseline/summary.json')['mean_batch_ms']-1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,default=REPO/'data/prepared-v3-conservative-patients')
    args=parser.parse_args(); report=verify_run(args.run_dir,args.prepared)
    (args.run_dir/'independent-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('PASS: paired experiment verified; quality gate:',report['comparison']['quality_gate_passed'])
