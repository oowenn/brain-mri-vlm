#!/usr/bin/env python3
"""One isolated GPU worker for Step 6 auditing, timing, or full validation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from benchmark_utils import scheduled_batch, summarize, trim_after_eos, workload
from evaluate import parse_prediction, runtime_snapshot, score_predictions, select_sample, write_json
from verify_data import REPO, read_csv
from verify_step4 import verify_predictions


def load_images(rows, source):
    from PIL import Image, ImageOps
    images = []
    for row in rows:
        path = (source / row['path']).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Image path escapes source root')
        with Image.open(path) as original:
            images.append(ImageOps.exif_transpose(original).convert('RGB'))
    return images


def cpu_inputs(processor, prompt, images):
    return processor(text=[prompt] * len(images), images=[[im] for im in images],
                     padding=True, return_tensors='pt')


def infer_batch(model, processor, cfg, rows, source, prompt):
    import torch
    images = load_images(rows, source)
    inputs = cpu_inputs(processor, prompt, images)
    lengths = inputs['attention_mask'].sum(dim=1).tolist()
    width = inputs['input_ids'].shape[1]
    pixel_shape = list(inputs['pixel_values'].shape)
    inputs = inputs.to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=cfg['max_new_tokens'],
                                do_sample=cfg['do_sample'], num_beams=cfg['num_beams'])
    ids = [trim_after_eos(tokens, model.generation_config.eos_token_id)
           for tokens in output[:, width:].tolist()]
    answers = processor.batch_decode(ids, skip_special_tokens=True)
    return [dict(path=row['path'], true_label=row['label'], raw_response=answer.strip(),
                 prediction=parse_prediction(answer), generated_token_ids=tokens,
                 image_size=list(image.size), input_tokens=length, padded_input_tokens=width,
                 pixel_values_shape=[1, *pixel_shape[1:]], batch_pixel_values_shape=pixel_shape)
            for row, image, length, tokens, answer in zip(rows, images, lengths, ids, answers)]


def input_audit(processor, prompt, rows, source):
    """Check image association and effective text tokens against independent singles."""
    import torch
    images = load_images(rows, source)
    batched = cpu_inputs(processor, prompt, images)
    evidence = []
    for i, image in enumerate(images):
        single = processor(text=prompt, images=[image], return_tensors='pt')
        effective = batched['input_ids'][i][batched['attention_mask'][i].bool()]
        if not torch.equal(effective, single['input_ids'][0]):
            raise ValueError('Batched prompt tokens differ from single-image processing')
        if int(batched['attention_mask'][i, -1]) != 1:
            raise ValueError('Generation requires the last prompt position to be non-padding')
        for key in ('pixel_values', 'pixel_attention_mask'):
            a, b = batched[key][i], single[key][0]
            if a.shape[1:] != b.shape[1:] or a.shape[0] < b.shape[0] or not torch.equal(a[:b.shape[0]], b):
                raise ValueError(f'Batched image processing changed {key}')
            if a.shape[0] > b.shape[0] and bool(torch.count_nonzero(a[b.shape[0]:])):
                raise ValueError('Image-padding entries must be zero')
        evidence.append(dict(path=rows[i]['path'], effective_tokens=len(effective),
                             image_size=list(image.size), single_pixel_shape=list(single['pixel_values'].shape)))
    return dict(status='passed', padding_side=processor.tokenizer.padding_side,
                images=len(rows), exact_effective_token_and_pixel_match=True, examples=evidence)


def telemetry():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem',
        '--format=csv'], text=True).strip()


def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['audit','sweep','baseline','quality'], required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--config', type=Path, default=REPO/'configs/step6.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=REPO/'data/prepared-v3-conservative-patients')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    args.source = args.source.resolve()
    record = dict(status='running', mode=args.mode, batch_size=args.batch_size, config=cfg,
                  started_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(args.checkpoint.resolve()),
                  test_set_inference=False)
    write_json(args.output/'run.json', record)
    try:
        if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
            raise RuntimeError('Run inside a Slurm GPU allocation')
        if torch.cuda.get_device_name(0) != cfg['gpu']:
            raise ValueError('GPU type differs from the fixed A6000 protocol')
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
        torch.manual_seed(cfg['seed'])
        runtime = runtime_snapshot(torch)
        runtime.update(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                       torch_threads=torch.get_num_threads(),
                       gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
                       cudnn_benchmark=torch.backends.cudnn.benchmark,
                       matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        write_json(args.output/'environment.json', runtime)
        (args.output/'gpu-before.csv').write_text(telemetry()+'\n')
        processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
        processor.tokenizer.padding_side = cfg['padding_side']
        model = AutoModelForImageTextToText.from_pretrained(args.checkpoint, local_files_only=True,
                    dtype=torch.float16, attn_implementation=cfg['attention'], use_safetensors=True).to('cuda').eval()
        if any(p.dtype != torch.float16 or str(p.device) != 'cuda:0' for p in model.parameters()):
            raise ValueError('Expected plain FP16 weights entirely on cuda:0')
        if any(hasattr(m, 'lora_A') or type(m).__name__ in ('Linear4bit','Linear8bitLt') for m in model.modules()):
            raise ValueError('Expected the merged, unquantized model')
        write_json(args.output/'model-runtime.json', dict(model_class=type(model).__name__,
                   parameters=sum(p.numel() for p in model.parameters()), all_parameters_fp16=True,
                   all_parameters_cuda0=True, remaining_lora_modules=0))
        rows = sorted(read_csv(args.prepared/'validation.csv'), key=lambda r:r['path'])
        messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':cfg['prompt']}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        (args.output/'rendered-prompt.txt').write_text(prompt)
        (args.output/'generation-config.json').write_text(model.generation_config.to_json_string())
        if args.mode == 'audit':
            probes = select_sample(rows, 2, cfg['seed'])
            write_json(args.output/'input-audit.json', input_audit(processor, prompt, probes, args.source))
            reference = {r['path']:r for r in [json.loads(x) for x in
                         (REPO/cfg['reference_run']/'validation/merged-fp16/predictions.jsonl').read_text().splitlines()]}
            singles = [infer_batch(model, processor, cfg, [row], args.source, prompt)[0] for row in probes]
            for row in singles:
                for key in ('raw_response','prediction','generated_token_ids','image_size','input_tokens','pixel_values_shape'):
                    if row[key] != reference[row['path']][key]:
                        raise ValueError(f'New batch-one loader differs from Step 5: {key}, {row["path"]}')
            batched = infer_batch(model, processor, cfg, probes, args.source, prompt)
            write_json(args.output/'reference-check.json', dict(status='passed', exact_batch_one_matches=len(singles),
                       batched_probe_label_matches=sum(a['prediction']==b['prediction'] for a,b in zip(singles,batched)),
                       singles=singles, batched=batched))
        elif args.mode == 'quality':
            selected = [{k:r[k] for k in ('path','label','sha256','group_id')} for r in rows]
            write_json(args.output/'selected-images.json', selected)
            predictions = []
            with (args.output/'predictions.jsonl').open('x') as stream:
                for start in range(0,len(rows),args.batch_size):
                    batch = infer_batch(model, processor, cfg, rows[start:start+args.batch_size], args.source, prompt)
                    predictions.extend(batch)
                    for row in batch:
                        stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    print(f'Quality: {len(predictions)}/579 images', flush=True)
            metrics = score_predictions(predictions)
            verify_predictions(rows, predictions, selected, metrics)
            write_json(args.output/'metrics.json', metrics)
        else:
            bc = cfg['benchmark']
            seeds = bc['sweep_seeds'] if args.mode == 'sweep' else bc['baseline_seeds']
            measured = bc['sweep_measured_batches'] if args.mode == 'sweep' else bc['baseline_measured_batches']
            samples = []
            with (args.output/'timings.jsonl').open('x') as stream:
                for trial, seed in enumerate(seeds):
                    torch.manual_seed(seed)
                    pool = workload(rows, bc['workload_per_class'], seed)
                    write_json(args.output/f'workload-{seed}.json', [{k:r[k] for k in ('path','label','sha256','group_id')} for r in pool])
                    for phase, count in [('warmup',bc['warmup_batches']),('measured',measured)]:
                        for iteration in range(count):
                            batch = scheduled_batch(pool,args.batch_size,iteration)
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            started = time.perf_counter_ns()
                            predictions = infer_batch(model, processor, cfg, batch, args.source, prompt)
                            torch.cuda.synchronize()
                            elapsed = time.perf_counter_ns()-started
                            row = dict(phase=phase,trial=trial,seed=seed,iteration=iteration,
                                       images=len(batch),elapsed_ns=elapsed,
                                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                       paths=[r['path'] for r in batch],
                                       input_tokens=[r['input_tokens'] for r in predictions],
                                       generated_token_counts=[len(r['generated_token_ids']) for r in predictions],
                                       predictions=[{k:r[k] for k in ('raw_response','prediction','generated_token_ids')} for r in predictions])
                            samples.append(row)
                            stream.write(json.dumps(row)+'\n')
                            stream.flush()
                            if iteration % 10 == 0 or iteration+1 == count:
                                print(f'{args.mode} B={args.batch_size} trial={trial} {phase} {iteration+1}/{count}: {elapsed/1e6:.2f} ms', flush=True)
            summary = summarize(samples)
            write_json(args.output/'summary.json', summary)
            record['summary'] = summary
        record.update(status='passed',finished_utc=datetime.now(timezone.utc).isoformat(),
                      total_memory_bytes=runtime['gpu']['total_memory_bytes'])
        (args.output/'gpu-after.csv').write_text(telemetry()+'\n')
        write_json(args.output/'run.json',record)
        print(f'PASS: {args.mode} batch {args.batch_size}', flush=True)
    except torch.OutOfMemoryError:
        record.update(status='oom',error=traceback.format_exc())
        write_json(args.output/'run.json',record)
        raise SystemExit(42)
    except Exception:
        record.update(status='failed',error=traceback.format_exc())
        write_json(args.output/'run.json',record)
        raise


if __name__ == '__main__':
    main()
