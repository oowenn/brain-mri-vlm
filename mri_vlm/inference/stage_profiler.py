#!/usr/bin/env python3
"""Profile the native Step 6 generation path, retaining an explicit overhead boundary."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
from functools import wraps
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from mri_vlm import naming
from mri_vlm.common import PREPARED, read_csv, REPO
from mri_vlm.inference.benchmark import load_images, cpu_inputs, telemetry, configure_processor
from mri_vlm.inference.evaluate import parse_prediction, runtime_snapshot, write_json
from mri_vlm.inference.stage_timing import stage_times, summarize_stages, trace_summary, nvidia_uuid
from mri_vlm.inference.workload import scheduled_batch, workload, trim_after_eos


class Recorder:
    def __init__(self,model,synchronized=True):
        import torch
        self.torch=torch
        self.synchronized=synchronized
        self.spans=[]
        self.calls=[]
        self.original_forward=model.forward
        self.original_vision=model.model.get_image_features
        self.model=model
        @wraps(self.original_forward)
        def forward(*args,**kwargs):
            ids=kwargs.get('input_ids')
            embeds=kwargs.get('inputs_embeds')
            seq=ids.shape[1] if ids is not None else embeds.shape[1]
            name='first_forward' if not self.calls else 'decode'
            self.calls.append(dict(name=name,sequence_length=seq,pixels=kwargs.get('pixel_values') is not None))
            with self.span(name,True):
                return self.original_forward(*args,**kwargs)
        @wraps(self.original_vision)
        def vision(*args,**kwargs):
            with self.span('vision',True):
                return self.original_vision(*args,**kwargs)
        model.forward=forward
        model.model.get_image_features=vision

    def reset(self):
        self.spans=[]
        self.calls=[]

    def close(self):
        self.model.forward=self.original_forward
        self.model.model.get_image_features=self.original_vision

    @contextmanager
    def span(self,name,gpu=False):
        torch=self.torch
        gpu=gpu and self.synchronized
        if gpu:
            torch.cuda.synchronize()
            begin=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
        started=time.perf_counter_ns()
        with torch.profiler.record_function('mri/'+name):
            if gpu:
                begin.record()
            yield
            if gpu:
                end.record()
                end.synchronize()
        self.spans.append(dict(name=name,wall_ns=time.perf_counter_ns()-started,
                               device_ms=begin.elapsed_time(end) if gpu else 0))


def infer(model,processor,cfg,rows,source,prompt,recorder):
    import torch
    recorder.reset()
    with recorder.span('preprocessing'):
        with recorder.span('image_read'):
            images=load_images(rows,source)
        with recorder.span('processor'):
            inputs=cpu_inputs(processor,prompt,images)
        lengths=inputs['attention_mask'].sum(dim=1).tolist()
        width=inputs['input_ids'].shape[1]
        pixel_shape=list(inputs['pixel_values'].shape)
        input_bytes=sum(v.numel()*v.element_size() for v in inputs.values() if torch.is_tensor(v))
    with recorder.span('transfer',True):
        inputs=inputs.to('cuda')
        inputs['pixel_values']=inputs['pixel_values'].to(torch.float16)
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],
                              do_sample=cfg['do_sample'],num_beams=cfg['num_beams'])
    with recorder.span('postprocessing'):
        ids=[trim_after_eos(tokens,model.generation_config.eos_token_id) for tokens in output[:,width:].tolist()]
        answers=processor.batch_decode(ids,skip_special_tokens=True)
        predictions=[dict(path=row['path'],true_label=row['label'],raw_response=answer.strip(),
                          prediction=parse_prediction(answer),generated_token_ids=tokens)
                     for row,tokens,answer in zip(rows,ids,answers)]
    if len(recorder.calls)!=max(map(len,ids)) or sum(x['name']=='vision' for x in recorder.spans)!=1:
        raise ValueError('Unexpected generation/cache/vision call count')
    if recorder.calls[0]['sequence_length']!=width or any(c['sequence_length']!=1 or c['pixels'] for c in recorder.calls[1:]):
        raise ValueError('Expected one prompt prefill and cached one-token decoding')
    return dict(predictions=predictions,input_tokens=lengths,padded_input_tokens=width,
                pixel_values_shape=pixel_shape,input_tensor_bytes=input_bytes,
                generated_token_counts=list(map(len,ids)),forward_calls=recorder.calls,spans=recorder.spans)


def main():
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['stages','traces'],required=True)
    parser.add_argument('--config',type=Path,default=REPO/naming.config_path('step7'))
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,default=PREPARED)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.config.read_text())
    baseline=json.loads((REPO/naming.current_path(cfg['baseline_config'])).read_text())
    record=dict(status='running',mode=args.mode,config=cfg,inference_config=baseline,
                started_utc=datetime.now(timezone.utc).isoformat(),test_set_inference=False)
    write_json(args.output/'run.json',record)
    sampler=None
    try:
        if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
            raise RuntimeError('Use a Slurm GPU allocation')
        if torch.cuda.get_device_name(0)!=baseline['gpu']:
            raise ValueError('GPU differs from frozen baseline')
        torch.set_num_threads(4)
        if int(os.environ['SLURM_CPUS_PER_TASK'])<4:
            raise ValueError('Need four allocated CPU threads')
        torch.manual_seed(baseline['seed'])
        env=runtime_snapshot(torch)
        env.update(torch_threads=torch.get_num_threads(),
                   gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark)
        write_json(args.output/'environment.json',env)
        (args.output/'gpu-before.csv').write_text(telemetry()+'\n')
        processor=AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
        processor.tokenizer.padding_side=baseline['padding_side']
        configure_processor(processor,baseline)
        model=AutoModelForImageTextToText.from_pretrained(args.checkpoint,local_files_only=True,
            dtype=torch.float16,attn_implementation=baseline['attention'],use_safetensors=True).to('cuda').eval()
        if any(p.dtype!=torch.float16 or str(p.device)!='cuda:0' for p in model.parameters()):
            raise ValueError('Expected plain FP16 CUDA checkpoint')
        if any(hasattr(m,'lora_A') or type(m).__name__ in ('Linear4bit','Linear8bitLt') for m in model.modules()):
            raise ValueError('Unexpected adapter or quantization')
        storages={p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in model.parameters()}
        src=Path(inspect.getfile(type(model)))
        (args.output/'installed-modeling-idefics3.py').write_bytes(src.read_bytes())
        write_json(args.output/'model-runtime.json',dict(parameter_storage_bytes=sum(storages.values()),
            resident_allocated_bytes=torch.cuda.memory_allocated(),model_class=type(model).__name__,
            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            all_parameters_fp16_cuda=True,plain_merged_model=True,
            processor=json.loads(processor.image_processor.to_json_string())))
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':baseline['prompt']}]}]
        prompt=processor.apply_chat_template(messages,add_generation_prompt=True)
        (args.output/'rendered-prompt.txt').write_text(prompt)
        rows=read_csv(args.prepared/'validation.csv')
        recorder=Recorder(model,synchronized=args.mode=='stages')
        samples=[]
        with (args.output/'gpu-samples.csv').open('w') as gpu_log:
            gpu_log.write('timestamp,uuid,gpu_utilization_percent,memory_controller_utilization_percent,memory_used_MiB,power_W\n')
            gpu_log.flush()
            device=nvidia_uuid(env['gpu_uuid'])
            observed=subprocess.check_output(['nvidia-smi','--id='+device,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            if observed!=device:
                raise ValueError('GPU sampler identity differs')
            sampler=subprocess.Popen(['nvidia-smi','--id='+device,
                '--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,power.draw',
                '--format=csv,noheader,nounits','--loop-ms='+str(cfg['gpu_sample_interval_ms'])],
                stdout=gpu_log,stderr=subprocess.STDOUT)
            with (args.output/'samples.jsonl').open('x') as stream:
                for trial,seed in enumerate(cfg['seeds']):
                    torch.manual_seed(seed)
                    pool=workload(rows,baseline['benchmark']['workload_per_class'],seed)
                    write_json(args.output/f'workload-{seed}.json',[{k:r[k] for k in ('path','label','sha256','group_id')} for r in pool])
                    def call(phase,i):
                        batch=scheduled_batch(pool,cfg['batch_size'],i)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        started=time.perf_counter_ns()
                        with torch.profiler.record_function('mri/batch'):
                            data=infer(model,processor,baseline,batch,args.source.resolve(),prompt,recorder)
                            torch.cuda.synchronize()
                        elapsed=time.perf_counter_ns()-started
                        item=dict(phase=phase,trial=trial,seed=seed,iteration=i,elapsed_ns=elapsed,
                                  paths=[r['path'] for r in batch],**data,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
                        item['stages']=stage_times(item['spans'],elapsed)
                        samples.append(item);stream.write(json.dumps(item)+'\n');stream.flush()
                        print(f'{args.mode} trial={trial} {phase} {i+1}: {elapsed/1e6:.2f} ms',flush=True)
                    for i in range(cfg['warmup_batches']):
                        call('warmup',i)
                    if args.mode=='stages':
                        for i in range(cfg['stage_batches_per_trial']):
                            call('measured',i)
                    else:
                        # One profiler warm-up iteration, then two recorded batches.
                        def save_trace(prof):
                            raw=args.output/f'trace-{seed}.json'
                            prof.export_chrome_trace(str(raw))
                            trace=json.loads(raw.read_text())
                            write_json(args.output/f'trace-{seed}-summary.json',trace_summary(trace))
                            with raw.open('rb') as inp,gzip.GzipFile(filename=str(raw)+'.gz',mode='wb',mtime=0) as out:
                                import shutil
                                shutil.copyfileobj(inp,out)
                            raw.unlink()
                            operators=[]
                            for e in prof.key_averages(group_by_input_shape=True):
                                operators.append(dict(name=e.key,count=e.count,input_shapes=e.input_shapes,
                                    self_cpu_us=e.self_cpu_time_total,cpu_total_us=e.cpu_time_total,
                                    self_device_us=getattr(e,'self_device_time_total',0),
                                    device_total_us=getattr(e,'device_time_total',0),
                                    cpu_memory_bytes=e.cpu_memory_usage,
                                    device_memory_bytes=getattr(e,'device_memory_usage',0)))
                            write_json(args.output/f'operators-{seed}.json',operators)
                            (args.output/f'operators-{seed}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                schedule=torch.profiler.schedule(wait=0,warmup=1,active=cfg['trace_batches_per_trial'],repeat=1),
                                on_trace_ready=save_trace,record_shapes=True,profile_memory=True,with_stack=False) as prof:
                            call('profiler_warmup',0);prof.step()
                            for i in range(cfg['trace_batches_per_trial']):
                                call('trace',i);prof.step()
            if sampler.poll() is not None:
                raise RuntimeError('GPU sampler exited before profiling finished')
            sampler.terminate();sampler.wait();sampler=None
        recorder.close()
        if args.mode=='stages':
            record['summary']=summarize_stages(samples)
            write_json(args.output/'summary.json',record['summary'])
        record.update(status='passed',finished_utc=datetime.now(timezone.utc).isoformat())
        (args.output/'gpu-after.csv').write_text(telemetry()+'\n')
        write_json(args.output/'run.json',record)
        print('PASS:',args.mode,flush=True)
    except Exception:
        record.update(status='failed',error=traceback.format_exc());write_json(args.output/'run.json',record)
        raise
    finally:
        if sampler is not None:
            sampler.terminate();sampler.wait()


if __name__=='__main__':
    main()
