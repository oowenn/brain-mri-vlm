#!/usr/bin/env python3
"""Evaluate a fixed validation sample or the full zero-shot validation split."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import socket
import subprocess
import traceback

from verify_data import REPO, canonical, read_csv, sha, validate_manifests

LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
INVALID = 'INVALID'


def parse_prediction(text):
    """Accept case/whitespace and one final period; never guess from prose."""
    match = re.fullmatch(r'(glioma|meningioma|notumor|pituitary)\.?', text.strip().lower())
    return match.group(1) if match else INVALID


def score_predictions(results):
    if not results:
        raise ValueError('Cannot score an empty prediction set')
    columns = LABELS + [INVALID]
    matrix = [[0] * len(columns) for _ in LABELS]
    for row in results:
        matrix[LABELS.index(row['true_label'])][columns.index(row['prediction'])] += 1
    per_class = {}
    for index, label in enumerate(LABELS):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    invalid = sum(row[-1] for row in matrix)
    return dict(n=len(results), accuracy=sum(matrix[i][i] for i in range(4)) / len(results),
                macro_f1=sum(row['f1'] for row in per_class.values()) / 4,
                per_class=per_class, invalid_count=invalid, invalid_rate=invalid / len(results),
                confusion_matrix=dict(rows=LABELS, columns=columns, counts=matrix))


def select_sample(rows, per_class, seed):
    if per_class < 0:
        raise ValueError('per_class must be nonnegative; zero selects all validation images')
    if not per_class:
        return sorted(rows, key=lambda r: r['path'])
    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        candidates = sorted((r for r in rows if r['label'] == label), key=lambda r: r['path'])
        selected.extend(rng.sample(candidates, per_class))
    return selected


def evaluation_protocol(config, precision, per_class_override):
    stage = config.get('stage', 'step3')
    per_class = config['per_class'] if per_class_override is None else per_class_override
    if stage not in ('step3', 'step4'):
        raise ValueError(f'Unsupported evaluation stage: {stage}')
    if stage == 'step4' and (precision != 'fp16' or per_class != 0):
        raise ValueError('Step 4 requires FP16 on the full validation split')
    return stage, per_class, 'step3-compatibility' if per_class else 'zero-shot-validation'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def runtime_snapshot(torch):
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    props = torch.cuda.get_device_properties(0)
    return dict(python=platform.python_version(), executable=os.sys.executable,
                platform=platform.platform(), hostname=socket.gethostname(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                packages=dict(sorted(versions.items(), key=lambda item: item[0].lower())),
                torch_cuda_build=torch.version.cuda,
                gpu=dict(name=props.name, total_memory_bytes=props.total_memory,
                         compute_capability=[props.major, props.minor]),
                nvidia_smi=subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
                    text=True).strip())


def evaluate(args):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    config = json.loads(args.config.read_text())
    if config['labels'] != LABELS or config['batch_size'] != 1:
        raise ValueError('This evaluator implements the fixed four-class, batch-one protocol')
    if config['parser'] != 'exact-label-optional-period-v2':
        raise ValueError('Unsupported parser protocol')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run inside a Slurm GPU allocation')
    if torch.cuda.get_device_name(0) != config['gpu']:
        raise RuntimeError(f'Expected fixed GPU type {config["gpu"]}')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    torch.manual_seed(config['seed'])
    validate_manifests(args.prepared)
    manifest = args.prepared / 'validation.csv'
    rows = read_csv(manifest)
    stage, per_class, scope = evaluation_protocol(config, args.precision, args.per_class)
    selected = select_sample(rows, per_class, config['seed'])
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = runtime_snapshot(torch)
    write_json(args.output / 'environment.json', runtime)
    write_json(args.output / 'selected-images.json', [
        {k: row[k] for k in ('path', 'label', 'sha256', 'group_id')} for row in selected])
    code_paths = ['scripts/evaluate.py', 'scripts/verify_data.py',
                  str(args.config.resolve().relative_to(REPO)), 'configs/data_policy.json']
    record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  stage=stage, scope=scope,
                  precision=args.precision, config=config, per_class=per_class,
                  sample_size=len(selected), validation_manifest_sha256=sha(manifest.read_bytes()),
                  code_sha256={name: sha((REPO / name).read_bytes()) for name in code_paths},
                  base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  limitation=('Small validation sample checks execution and scoring; not a quality or performance benchmark.'
                              if per_class else 'Full validation quality only; no final-test inference, training, or performance benchmark.'))
    write_json(args.output / 'run.json', record)
    run = None
    try:
        if args.log_wandb:
            import wandb
            run = wandb.init(entity=config['wandb_entity'], project=config['wandb_project'],
                             job_type=record['scope'], group=f'{stage}-{runtime["slurm_job_id"]}',
                             name=f'{stage}-{runtime["slurm_job_id"]}-{args.precision}',
                             config={**config, 'precision': args.precision,
                                     'per_class': per_class, 'validation_manifest_sha256': record['validation_manifest_sha256']},
                             tags=[stage, 'compatibility' if per_class else scope, args.precision],
                             settings=wandb.Settings(disable_git=True, disable_code=True))
            run.use_artifact(config['data_artifact'])
            record.update(wandb_run_id=run.id, wandb_url=run.url)
            write_json(args.output / 'run.json', record)
        processor = AutoProcessor.from_pretrained(config['model_id'], revision=config['revision'],
                                                  local_files_only=True)
        kwargs = dict(revision=config['revision'], dtype=torch.float16,
                      attn_implementation=config['attention'], use_safetensors=True,
                      local_files_only=True)
        quantization = None
        if args.precision == 'int8':
            quantization = BitsAndBytesConfig(load_in_8bit=True)
        elif args.precision == 'int4':
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                             bnb_4bit_compute_dtype=torch.float16,
                                             bnb_4bit_use_double_quant=True)
        if quantization is not None:
            kwargs.update(quantization_config=quantization, device_map={'': 0})
        model = AutoModelForImageTextToText.from_pretrained(config['model_id'], **kwargs)
        if args.precision == 'fp16':
            model.to('cuda')
        model.eval()
        quantized = {name: type(module).__name__ for name, module in model.named_modules()
                     if type(module).__name__ in {'Linear8bitLt', 'Linear4bit'}}
        expected_type = {'int8': 'Linear8bitLt', 'int4': 'Linear4bit'}.get(args.precision)
        if expected_type and expected_type not in quantized.values():
            raise RuntimeError('Requested quantization did not replace any linear layers')
        devices = sorted({str(parameter.device) for parameter in model.parameters()})
        if devices != ['cuda:0']:
            raise RuntimeError(f'Unexpected parameter devices: {devices}')
        write_json(args.output / 'model-runtime.json', dict(
            model_class=type(model).__name__, parameter_devices=devices,
            quantized_module_count=len(quantized), quantized_modules=quantized,
            quantization_config=quantization.to_dict() if quantization is not None else None))
        (args.output / 'model-config.json').write_text(model.config.to_json_string())
        (args.output / 'image-processor.json').write_text(processor.image_processor.to_json_string())
        messages = [{'role': 'user', 'content': [{'type': 'image'},
                    {'type': 'text', 'text': config['prompt']}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        (args.output / 'rendered-prompt.txt').write_text(prompt)
        (args.output / 'generation-config.json').write_text(model.generation_config.to_json_string())
        results = []
        source = args.source.resolve()
        with (args.output / 'predictions.jsonl').open('x') as stream:
            for index, row in enumerate(selected, 1):
                path = (source / row['path']).resolve()
                if not path.is_relative_to(source):
                    raise ValueError('Source path escapes dataset root')
                raw_image = path.read_bytes()
                if sha(raw_image) != row['sha256']:
                    raise ValueError(f'Image checksum mismatch: {row["path"]}')
                with Image.open(io.BytesIO(raw_image)) as original:
                    image = ImageOps.exif_transpose(original).convert('RGB')
                inputs = processor(text=prompt, images=[image], return_tensors='pt').to('cuda')
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=config['max_new_tokens'],
                                               do_sample=config['do_sample'], num_beams=config['num_beams'])
                token_ids = generated[0, inputs['input_ids'].shape[1]:].tolist()
                raw = processor.decode(token_ids, skip_special_tokens=True).strip()
                result = dict(path=row['path'], true_label=row['label'], raw_response=raw,
                              prediction=parse_prediction(raw), generated_token_ids=token_ids,
                              image_size=list(image.size), input_tokens=inputs['input_ids'].shape[1],
                              pixel_values_shape=list(inputs['pixel_values'].shape))
                results.append(result)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                print(f'{index}/{len(selected)} true={row["label"]} predicted={result["prediction"]} raw={raw!r}', flush=True)
        metrics = score_predictions(results)
        write_json(args.output / 'metrics.json', metrics)
        record.update(status='passed', finished_utc=datetime.now(timezone.utc).isoformat(),
                      quantized_module_count=len(quantized))
        write_json(args.output / 'run.json', record)
        if run is not None:
            flat = {k: metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'invalid_count', 'invalid_rate')}
            flat.update({f'recall/{label}': metrics['per_class'][label]['recall'] for label in LABELS})
            run.log(flat)
            run.summary.update({**flat, 'status': 'passed', 'quantized_module_count': len(quantized),
                                'gpu': runtime['gpu']['name'], 'code_sha256': record['code_sha256']})
            run.log({'predictions': wandb.Table(columns=['path', 'true_label', 'prediction', 'raw_response'],
                     data=[[r[k] for k in ('path', 'true_label', 'prediction', 'raw_response')] for r in results])})
            artifact = wandb.Artifact(f'{stage}-{runtime["slurm_job_id"]}-{args.precision}', type='runtime-evidence')
            artifact.add_dir(str(args.output))
            run.log_artifact(artifact).wait()
            run.finish()
            run = None
        print(json.dumps(metrics, indent=2))
        print(f'PASS: {args.precision} MRI inference and scoring completed')
    except Exception:
        record.update(status='failed', error=traceback.format_exc())
        write_json(args.output / 'run.json', record)
        if run is not None:
            run.summary['status'] = 'failed'
            run.summary['error'] = record['error']
            artifact = wandb.Artifact(f'{stage}-{runtime["slurm_job_id"]}-{args.precision}-failed', type='runtime-evidence')
            artifact.add_dir(str(args.output))
            run.log_artifact(artifact).wait()
            run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/step3.json')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=REPO / 'data/prepared-v3-conservative-patients')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=['fp16', 'int8', 'int4'], default='fp16')
    parser.add_argument('--per-class', type=int, help='Default from config; 0 evaluates all validation images')
    parser.add_argument('--log-wandb', action='store_true')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
