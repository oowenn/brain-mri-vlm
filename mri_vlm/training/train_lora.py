#!/usr/bin/env python3
"""One fixed LoRA training run, validation selection, and a verified FP16 merge."""
import argparse
from datetime import datetime, timezone
import gc
import io
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import traceback

from mri_vlm import naming
from mri_vlm.common import PREPARED, read_csv, REPO, sha
from mri_vlm.data.verify import validate_manifests
from mri_vlm.inference.evaluate import LABELS, generate_predictions, runtime_snapshot, score_predictions, select_sample, write_json
from mri_vlm.inference.prediction_check import verify_predictions
from mri_vlm.training.lora import choose_checkpoint, epoch_order, merge_passes, supervised_sequences


def image_for_row(source, row):
    from PIL import Image, ImageOps
    path = (source / row['path']).resolve()
    if not path.is_relative_to(source):
        raise ValueError('Image path escapes the source root')
    data = path.read_bytes()
    if sha(data) != row['sha256']:
        raise ValueError(f'Image checksum mismatch: {row["path"]}')
    with Image.open(io.BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


class CompletionBatch:
    def __init__(self, processor, config, source):
        self.processor, self.source = processor, source
        self.messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': config['prompt']}]}]
        self.prompt = processor.apply_chat_template(self.messages, add_generation_prompt=True)
        processor.tokenizer.padding_side = 'right'
        self.targets, self.texts = {}, {}
        end_id = processor.tokenizer.convert_tokens_to_ids('<end_of_utterance>')
        for label in LABELS:
            messages = self.messages + [{'role': 'assistant', 'content': [{'type': 'text', 'text': label}]}]
            full = processor.apply_chat_template(messages, add_generation_prompt=False).rstrip('\n')
            if not full.startswith(self.prompt):
                raise ValueError('Assistant completion does not extend the inference prompt')
            suffix = full[len(self.prompt):]
            ids = processor.tokenizer(suffix, add_special_tokens=False)['input_ids']
            if ids[-1] != end_id or processor.tokenizer.decode(ids, skip_special_tokens=True).strip() != label:
                raise ValueError('Target must be exactly the category plus end-of-answer token')
            self.targets[label], self.texts[label] = ids, full

    def __call__(self, rows, check_full_template=False):
        import torch
        images = [image_for_row(self.source, row) for row in rows]
        inputs = self.processor(text=[self.prompt] * len(rows), images=[[im] for im in images],
                                padding=True, return_tensors='pt')
        prefixes = [ids[mask.bool()].tolist() for ids, mask in zip(inputs['input_ids'], inputs['attention_mask'])]
        targets = [self.targets[row['label']] for row in rows]
        ids, masks, labels = supervised_sequences(prefixes, targets, self.processor.tokenizer.pad_token_id)
        if check_full_template:
            complete = self.processor(text=[self.texts[row['label']] for row in rows],
                                      images=[[im] for im in images], padding=True, return_tensors='pt')
            for i in range(len(rows)):
                actual = complete['input_ids'][i][complete['attention_mask'][i].bool()].tolist()
                if actual != prefixes[i] + targets[i]:
                    raise ValueError('Completion concatenation differs from canonical chat tokenization')
        inputs['input_ids'] = torch.tensor(ids, dtype=torch.long)
        inputs['attention_mask'] = torch.tensor(masks, dtype=torch.long)
        inputs['labels'] = torch.tensor(labels, dtype=torch.long)
        evidence = [dict(path=row['path'], true_label=row['label'], prefix_tokens=len(prefix),
                         supervised_token_ids=target,
                         supervised_text=self.processor.tokenizer.decode(target, skip_special_tokens=False))
                    for row, prefix, target in zip(rows, prefixes, targets)]
        return inputs, evidence


def to_gpu(inputs):
    import torch
    result = {name: value.to('cuda') for name, value in inputs.items()}
    result['pixel_values'] = result['pixel_values'].to(torch.float16)
    return result


def load_base(config, location=None):
    import torch
    from transformers import AutoModelForImageTextToText
    kwargs = dict(dtype=torch.float16, attn_implementation=config['attention'],
                  local_files_only=True, use_safetensors=True)
    if location is None:
        kwargs['revision'] = config['revision']
    return AutoModelForImageTextToText.from_pretrained(location or config['model_id'], **kwargs).to('cuda')


def validation_loss(model, batcher, rows, batch_size):
    import torch
    model.eval()
    loss_sum, tokens = 0.0, 0
    for start in range(0, len(rows), batch_size):
        inputs, _ = batcher(rows[start:start + batch_size])
        count = int((inputs['labels'] != -100).sum())
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            loss = model(**to_gpu(inputs), use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        loss_sum += float(loss) * count
        tokens += count
    return loss_sum / tokens, tokens


def validate_model(model, processor, config, rows, source, folder, batcher):
    rows = sorted(rows, key=lambda r: r['path'])
    folder.mkdir(parents=True, exist_ok=False)
    selected = [{k: row[k] for k in ('path', 'label', 'sha256', 'group_id')} for row in rows]
    write_json(folder / 'selected-images.json', selected)
    predictions = generate_predictions(model, processor, config, rows, source, folder)
    metrics = score_predictions(predictions)
    write_json(folder / 'metrics.json', metrics)
    independent = verify_predictions(rows, predictions, selected, metrics)
    loss, tokens = validation_loss(model, batcher, rows, config['training']['micro_batch_size'])
    write_json(folder / 'loss.json', dict(token_weighted_loss=loss, supervised_tokens=tokens, images=len(rows)))
    write_json(folder / 'verification.json', dict(status='passed', images=len(rows), independent_metrics=independent))
    return metrics, loss, predictions


def probe_logits(model, processor, config, rows, source):
    import torch
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': config['prompt']}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    outputs = []
    model.eval()
    for row in rows:
        inputs = processor(text=prompt, images=[image_for_row(source, row)], return_tensors='pt')
        with torch.inference_mode():
            logits = model(**to_gpu(inputs), use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
        outputs.append(logits)
    return torch.stack(outputs)


def file_inventory(root):
    return {str(path.relative_to(root)): dict(sha256=sha(path.read_bytes()), bytes=path.stat().st_size)
            for path in sorted(root.rglob('*')) if path.is_file()}


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import wandb
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=PREPARED)
    parser.add_argument('--config', type=Path, default=REPO/naming.config_path('step5'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    tc = cfg['training']
    prior = json.loads((REPO/naming.config_path('step4')).read_text())
    if {key: cfg[key] for key in prior} != {**prior, 'stage': 'step5'}:
        raise ValueError('Step 5 must preserve the accepted inference protocol')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Training requires a Slurm GPU allocation')
    if torch.cuda.get_device_name(0) != cfg['gpu']:
        raise ValueError('GPU does not match the fixed protocol')
    args.source = args.source.resolve()
    args.checkpoints.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=True)
    validate_manifests(args.prepared)
    train = sorted(read_csv(args.prepared / 'train.csv'), key=lambda r: r['path'])
    validation = sorted(read_csv(args.prepared / 'validation.csv'), key=lambda r: r['path'])
    assert len(train) == 3279 and len(validation) == 579
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    random.seed(tc['seed'])
    np.random.seed(tc['seed'])
    torch.manual_seed(tc['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write_json(args.output / 'environment.json', runtime_snapshot(torch))
    write_json(args.output / 'training-images.json', [{k: r[k] for k in ('path', 'label', 'sha256', 'group_id')} for r in train])
    record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), config=cfg,
                  base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  train_manifest_sha256=sha((args.prepared / 'train.csv').read_bytes()),
                  validation_manifest_sha256=sha((args.prepared / 'validation.csv').read_bytes()),
                  checkpoint_root=str(args.checkpoints.resolve()), test_set_inference=False)
    run = wandb.init(entity=cfg['wandb_entity'], project=cfg['wandb_project'],
                     name=naming.run_name('step5', 'run', os.environ['SLURM_JOB_ID']), group=naming.group('step5', os.environ['SLURM_JOB_ID']),
                     job_type=naming.slug('step5'), config=cfg, tags=[naming.slug('step5'), 'lora', 'fp16'],
                     settings=wandb.Settings(disable_git=True, disable_code=True))
    run.use_artifact(cfg['data_artifact'])
    record.update(wandb_run_id=run.id, wandb_url=run.url)
    write_json(args.output / 'run.json', record)
    try:
        processor = AutoProcessor.from_pretrained(cfg['model_id'], revision=cfg['revision'], local_files_only=True)
        batcher = CompletionBatch(processor, cfg, args.source)
        model = load_base(cfg)
        lora = LoraConfig(r=tc['lora_rank'], lora_alpha=tc['lora_alpha'], lora_dropout=tc['lora_dropout'],
                          target_modules=tc['target_modules'], bias='none', init_lora_weights=True,
                          base_model_name_or_path=cfg['model_id'], revision=cfg['revision'])
        model = get_peft_model(model, lora, revision=cfg['revision'])
        modules = [name for name, module in model.named_modules() if hasattr(module, 'lora_A')]
        parameters = {name: dict(shape=list(p.shape), count=p.numel(), dtype=str(p.dtype))
                      for name, p in model.named_parameters() if p.requires_grad}
        if len(modules) != tc['expected_lora_modules'] or any('text_model.layers.' not in name for name in modules):
            raise ValueError('LoRA must target only the configured language decoder modules')
        if any('lora_' not in name or value['dtype'] != 'torch.float32' for name, value in parameters.items()):
            raise ValueError('Only FP32 LoRA parameters may be trained')
        write_json(args.output / 'trainable-parameters.json', dict(modules=modules, parameters=parameters,
                   trainable_count=sum(v['count'] for v in parameters.values()), total_count=sum(p.numel() for p in model.parameters())))
        # Reproduce the original eight-image reference before any optimizer update.
        probes = select_sample(validation, cfg['merge_verification']['probe_per_class'], cfg['merge_verification']['seed'])
        initial = generate_predictions(model, processor, cfg, probes, args.source, args.output / 'preflight-base')
        original = [json.loads(line) for line in (REPO / 'experiments/03-precision-check/runs/13924504/fp16/predictions.jsonl').read_text().splitlines()]
        if [(r['path'], r['generated_token_ids']) for r in initial] != [(r['path'], r['generated_token_ids']) for r in original]:
            raise ValueError('Untrained LoRA model does not reproduce the pinned FP16 reference')
        # Audit all four target strings and exact prompt/completion token boundaries.
        audit_rows = [next(row for row in train if row['label'] == label) for label in LABELS]
        _, audit = batcher(audit_rows, check_full_template=True)
        write_json(args.output / 'label-mask-audit.json', dict(status='passed', examples=audit,
                   prompt_and_image_and_padding_labels=-100, completion_includes_end_of_utterance=True,
                   untrained_reference_predictions_match=8))
        effective = tc['micro_batch_size'] * tc['gradient_accumulation_steps']
        updates_per_epoch = math.ceil(len(train) / effective)
        total_updates = tc['epochs'] * updates_per_epoch
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc['learning_rate'],
                                     betas=tuple(tc['adam_betas']), eps=tc['adam_eps'], weight_decay=tc['weight_decay'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_updates * tc['warmup_ratio']), total_updates)
        scaler = torch.amp.GradScaler('cuda', init_scale=tc['fp16_grad_scaler_initial_scale'])
        epochs, global_step = [], 0
        history = (args.output / 'training-history.jsonl').open('x')
        for epoch in range(1, tc['epochs'] + 1):
            order = epoch_order(len(train), tc['seed'], epoch)
            write_json(args.output / f'epoch-{epoch:03}-order.json', [train[i]['path'] for i in order])
            model.train()
            model.get_base_model().model.vision_model.eval()
            model.get_base_model().model.connector.eval()
            epoch_loss_sum, epoch_tokens = 0.0, 0
            for start in range(0, len(order), effective):
                rows = [train[i] for i in order[start:start + effective]]
                group_tokens = sum(len(batcher.targets[row['label']]) for row in rows)
                optimizer.zero_grad(set_to_none=True)
                group_loss_sum = 0.0
                for offset in range(0, len(rows), tc['micro_batch_size']):
                    batch, _ = batcher(rows[offset:offset + tc['micro_batch_size']])
                    count = int((batch['labels'] != -100).sum())
                    batch = to_gpu(batch)
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = model(**batch, use_cache=False)
                        loss = output.loss
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    if global_step == 0 and offset == 0:
                        shifted = batch['labels'][:, 1:]
                        active = shifted != -100
                        manual = F.cross_entropy(output.logits[:, :-1][active].float(), shifted[active])
                        if not torch.allclose(loss.float(), manual, atol=1e-6, rtol=1e-6):
                            raise ValueError('Native model loss differs from completion-only cross entropy')
                        write_json(args.output / 'loss-audit.json', dict(status='passed', native_loss=float(loss.detach()),
                                   manual_completion_loss=float(manual.detach()), supervised_tokens=int(active.sum())))
                        del manual
                    scaler.scale(loss * count / group_tokens).backward()
                    group_loss_sum += float(loss.detach()) * count
                    del output, loss, batch
                scaler.unscale_(optimizer)
                grads = [p for p in model.parameters() if p.requires_grad]
                norm = torch.nn.utils.clip_grad_norm_(grads, tc['max_grad_norm'], error_if_nonfinite=True)
                if global_step == 0:
                    nonzero = sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in grads)
                    frozen_grads = [name for name, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
                    if not nonzero or frozen_grads:
                        raise ValueError('LoRA gradients missing or frozen model received gradients')
                    write_json(args.output / 'gradient-audit.json', dict(status='passed', nonzero_trainable_gradients=nonzero,
                               frozen_parameters_with_gradients=frozen_grads, preclip_gradient_norm=float(norm)))
                lr, scale = optimizer.param_groups[0]['lr'], scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale:
                    raise ValueError('An optimizer update was skipped due to FP16 overflow')
                scheduler.step()
                global_step += 1
                epoch_loss_sum += group_loss_sum
                epoch_tokens += group_tokens
                entry = dict(step=global_step, epoch=epoch, images=len(rows), supervised_tokens=group_tokens,
                             loss=group_loss_sum / group_tokens, learning_rate=lr, preclip_gradient_norm=float(norm),
                             scaler_scale=scaler.get_scale())
                history.write(json.dumps(entry) + '\n')
                history.flush()
                run.log({f'train/{k}': v for k, v in entry.items()})
                if global_step % 10 == 0 or start + effective >= len(order):
                    print(f'Epoch {epoch}/{tc["epochs"]} update {global_step}/{total_updates}: '
                          f'images {min(start + effective, len(order))}/{len(order)}, loss={entry["loss"]:.6f}', flush=True)
            checkpoint = args.checkpoints / f'epoch-{epoch:03}'
            adapter_dir = checkpoint / 'adapter'
            model.save_pretrained(adapter_dir, safe_serialization=True)
            processor.save_pretrained(adapter_dir)
            torch.save(dict(epoch=epoch, global_step=global_step, optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                            python_rng=random.getstate(), numpy_rng=np.random.get_state(), config=cfg), checkpoint / 'training-state.pt')
            metrics, loss, _ = validate_model(model, processor, cfg, validation, args.source,
                                              args.output / 'validation' / f'epoch-{epoch:03}', batcher)
            item = dict(epoch=epoch, global_step=global_step, training_loss=epoch_loss_sum / epoch_tokens,
                        training_images=len(order), training_supervised_tokens=epoch_tokens,
                        validation_loss=loss, metrics=metrics, adapter_path=str(adapter_dir.resolve()))
            epochs.append(item)
            write_json(args.output / 'epochs.json', epochs)
            run.log({'epoch': epoch, 'validation/loss': loss, 'validation/macro_f1': metrics['macro_f1'],
                     'validation/accuracy': metrics['accuracy'], 'validation/invalid_rate': metrics['invalid_rate'],
                     **{f'validation/recall/{label}': metrics['per_class'][label]['recall'] for label in LABELS}})
            print(f'EPOCH {epoch} COMPLETE: validation loss={loss:.6f}, macro-F1={metrics["macro_f1"]:.6f}', flush=True)
        history.close()
        selected = choose_checkpoint(epochs)
        write_json(args.output / 'selection.json', dict(rule=tc['selection'], selected=selected, candidates=epochs))
        # Reload the saved adapter before merging; no training state is reused for inference.
        del model, optimizer, scheduler, scaler, grads
        gc.collect()
        torch.cuda.empty_cache()
        adapter = PeftModel.from_pretrained(load_base(cfg), selected['adapter_path'], is_trainable=False)
        adapter.eval()
        adapter_probe = generate_predictions(adapter, processor, cfg, probes, args.source, args.output / 'merge-probe' / 'adapter')
        selected_predictions = [json.loads(line) for line in (args.output / 'validation' / f'epoch-{selected["epoch"]:03}' / 'predictions.jsonl').read_text().splitlines()]
        by_path = {r['path']: r for r in selected_predictions}
        if any(r['generated_token_ids'] != by_path[r['path']]['generated_token_ids'] for r in adapter_probe):
            raise ValueError('Reloaded adapter does not reproduce selected checkpoint probe answers')
        before = probe_logits(adapter, processor, cfg, probes, args.source)
        merged = adapter.float().merge_and_unload(safe_merge=True).half()
        merged_dir = args.checkpoints / 'merged-fp16'
        merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size='2GB')
        processor.save_pretrained(merged_dir)
        del adapter, merged
        gc.collect()
        torch.cuda.empty_cache()
        merged = load_base(cfg, merged_dir)
        merged.eval()
        after = probe_logits(merged, processor, cfg, probes, args.source)
        np.savez_compressed(args.checkpoints / 'merge-logits.npz', adapter=before.numpy(), merged=after.numpy())
        merged_probe = generate_predictions(merged, processor, cfg, probes, args.source, args.output / 'merge-probe' / 'merged')
        merged_metrics, merged_loss, merged_predictions = validate_model(merged, processor, cfg, validation, args.source,
                                                                        args.output / 'validation' / 'merged-fp16', batcher)
        differences = (before - after).abs()
        report = dict(probe_count=len(probes), probe_paths=[r['path'] for r in probes],
                      probe_token_matches=sum(a['generated_token_ids'] == b['generated_token_ids'] for a, b in zip(adapter_probe, merged_probe)),
                      max_absolute_logit_difference=float(differences.max()), mean_absolute_logit_difference=float(differences.mean()),
                      full_validation_label_agreement=sum(a['prediction'] == b['prediction'] for a, b in zip(selected_predictions, merged_predictions)) / len(validation),
                      absolute_macro_f1_difference=abs(selected['metrics']['macro_f1'] - merged_metrics['macro_f1']),
                      all_parameters_fp16=all(p.dtype == torch.float16 for p in merged.parameters()),
                      remaining_lora_modules=sum(hasattr(m, 'lora_A') for m in merged.modules()),
                      adapter_reloaded_probe_matches_selected=True, merged_reloaded_from_disk=True,
                      thresholds=cfg['merge_verification'], selected_epoch=selected['epoch'])
        report['status'] = 'passed' if merge_passes(cfg['merge_verification'], report) else 'failed'
        write_json(args.output / 'merge-verification.json', report)
        if report['status'] != 'passed':
            raise ValueError('FP16 merge failed the predeclared numerical or generation checks')
        write_json(args.output / 'checkpoint-files.json', file_inventory(args.checkpoints))
        record.update(status='passed', finished_utc=datetime.now(timezone.utc).isoformat(), optimizer_updates=global_step,
                      selected_epoch=selected['epoch'], merged_checkpoint=str(merged_dir.resolve()),
                      merged_metrics=merged_metrics, merged_validation_loss=merged_loss)
        write_json(args.output / 'run.json', record)
        run.summary.update(dict(status='passed', selected_epoch=selected['epoch'], optimizer_updates=global_step,
                                merged_macro_f1=merged_metrics['macro_f1'], merged_accuracy=merged_metrics['accuracy'],
                                merged_invalid_rate=merged_metrics['invalid_rate'], merge_verified=True))
        artifact = wandb.Artifact(naming.artifact('step5', os.environ['SLURM_JOB_ID'], 'evidence'), type='runtime-evidence')
        stable_files = {str(p.relative_to(args.output)): sha(p.read_bytes())
                        for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'console.log'}
        write_json(args.output / 'training-artifact-files.json', stable_files)
        for name in stable_files:
            artifact.add_file(str(args.output / name), name=name)
        artifact.add_file(str(args.output / 'training-artifact-files.json'), name='training-artifact-files.json')
        run.log_artifact(artifact).wait()
        run.finish()
        run = None
        print('PASS: LoRA training, validation selection, and saved/reloaded FP16 merge completed', flush=True)
    except Exception:
        record.update(status='failed', error=traceback.format_exc())
        write_json(args.output / 'run.json', record)
        if run is not None:
            run.summary.update(dict(status='failed', error=record['error']))
            run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
