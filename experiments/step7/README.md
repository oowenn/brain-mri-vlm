# Step 7: profile the frozen FP16 baseline

Status: rerunning after fixing GPU UUID normalization; attempt 13933253 and publisher 13933254 were cancelled. Their records will remain as diagnostic evidence.

Step 6 remains the canonical performance and full-validation reference. This step explains where its runtime goes and proposes two or three changes for Step 8. It does not implement those changes or evaluate the final test set.

## Protocol and why

Use the saved Step 5 merged model, frozen Step 6 batch size 4, A6000, four CPU threads, prompt, processor, SDPA, greedy generation, and 16-token limit. Verify all 7,200 source hashes and 12 checkpoint files first.

1. Repeat the unchanged benchmark in a fresh process: three seeds, ten warmups and fifty measured batches each. This control detects runtime drift and provides matched answers/timings.
2. In another process, collect ten measured stage batches after ten warmups per seed. Explicit synchronization makes stage wall times interpretable. Record preprocessing, transfer, vision, prefill, decode, and an explicit residual. First forward includes vision; subtract nested image-feature time to avoid counting it twice. Record CPU image-read/processor subtimings and CUDA events.
3. In a third process, record three CPU/CUDA traces, each with one profiler warmup and two active batches. These traces keep asynchronous stage execution and the original whole-batch synchronization. Record tensor shapes, memory events, operators, and actual GPU kernels. Trace instrumentation is diagnostic and changes timing.
4. Check every control answer against Step 6 and every instrumented answer against the matching control. Verify exact token IDs and raw answers, deterministic inputs, one vision call, one prefill, and cached single-token decode calls.
5. Recompute accounting, workload coverage, GPU activity intervals, and control statistics; verify W&B metrics and downloaded artifact hashes before marking complete.

Stage timings include synchronization overhead. Traces use CPU annotations to identify launch regions; GPU execution may extend past an annotation. GPU activity is the union of kernel/copy intervals inside whole-batch windows, not SM occupancy. Separate 200 ms nvidia-smi samples cover the worker including warmups and file writes. Profiler shape and memory collection can add overhead: [PyTorch profiler](https://docs.pytorch.org/docs/2.9/profiler.html), [CUDA events](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html).

## Files and their purpose

| File | Purpose |
| --- | --- |
| ../../configs/step7.json | Fixed reference, sample counts, seeds, and measurement boundaries. |
| ../../scripts/run_step7.py | Runs isolated control/stage/trace workers and records W&B evidence. |
| ../../scripts/profile_baseline.py | Measures native model stages and saves compressed Chrome traces and GPU samples. |
| ../../scripts/profiling_utils.py | Exclusive timing and overlapping GPU interval accounting. |
| ../../scripts/verify_step7.py | Recomputes results and rejects changed inputs, answers, cache behavior, or invalid traces. |
| ../../scripts/publish_step7.py | Checks completed Slurm/W&B results and downloads artifact files for hash verification. |
| ../../tests/test_profiling.py | Tests double-counting, interval clipping, invalid evidence, warmup exclusion, and answer changes. |
| ../../cluster/step7.sbatch | GPU launcher; records source, packages, tests, hashes, and worker output. |
| ../../cluster/publish_step7.sbatch | Dependent CPU job for finished-job accounting, source provenance, and W&B publishing. |

## Reproduce on Insomnia or a teammate's server

Follow the environment, checkpoint restoration, caches, and source-root instructions in [IMPLEMENTATION.md](../../IMPLEMENTATION.md). No additional packages are needed. Sean's work root is /insomnia001/depts/edu/users/ct3013; the environment is envs/smolmri and the merged checkpoint is brain-mri-vlm/checkpoints/step5/13927137/merged-fp16 beneath it. Use your own paths/account and credentials on another server.

From the repository on the login node, submit the prepared jobs:

~~~bash
mkdir -p logs
gpu_job=$(sbatch --parsable cluster/step7.sbatch)
sbatch --dependency=afterok:"$gpu_job" \
  --export=ALL,MRI_RUN_DIR="experiments/step7/runs/$gpu_job" \
  cluster/publish_step7.sbatch
~~~

The GPU job requests one A6000, four CPUs, 16 GiB RAM, and a 45-minute maximum. The CPU publisher requests one CPU and 4 GiB; Slurm may allocate extra CPUs for memory. These limits are not expected inference times. Step 8 remains deferred until these measurements support a concrete selection.

## Run evidence layout

The run directory will retain these files so teammates can inspect the result without access to Sean's server:

| File or directory under runs/<job-id>/ | What it records and why |
| --- | --- |
| run.json, base-commit.txt | Job outcome, W&B run, execution commit, frozen reference hashes, and absence of test inference. |
| source/, source-provenance.json | Exact scripts, configs, tests, launcher, and package pins checked against the execution commit. |
| tests.txt, pip-check.txt, requirements-installed.txt, pip-freeze-original.txt, cuda-check.txt | Runtime dependency/test/GPU checks and installed versions. |
| data-verification.json, checkpoint-verification.json | Frozen manifest/source bytes and merged model/processor hashes. |
| active-worker.json, worker-logs/, console.log | Worker commands, model warnings, progress, failures, and overall execution output. |
| control/run.json, summary.json, timings.jsonl | Uninstrumented repeated benchmark with raw per-batch answers/tokens and timing/memory evidence. |
| control/workload-*.json, environment.json, model-runtime.json, rendered-prompt.txt, gpu-*.csv | Fixed sample identities, actual GPU/environment, model properties, exact prompt, and hardware snapshots. |
| stages/samples.jsonl, summary.json | Raw spans, exclusive stage accounting, native generation-call audit, and aggregated diagnostic timings. |
| stages/workload-*.json, environment.json, model-runtime.json | Stage-workload identities, thread/GPU settings, resident parameter storage and image-processor configuration. |
| stages/installed-modeling-idefics3.py | Exact installed Transformers source that defines the nested vision/prefill measurement boundary; original license retained. |
| stages/gpu-samples.csv, gpu-*.csv, rendered-prompt.txt, run.json | Sampled utilization/memory/power, before/after snapshots, prompt, and worker outcome. |
| traces/ | Same sample/environment/model evidence as stages, with asynchronous launch annotations. Trace sample wall spans are CPU launch regions, not synchronized GPU stage times. |
| traces/trace-*.json.gz | Complete losslessly compressed Chrome/Perfetto CPU/CUDA traces, two active batches per seed. |
| traces/trace-*-summary.json | Kernel counts, whole-batch windows, overlapping GPU activity union, and top kernel durations. |
| traces/operators-*.json, operators-*.txt | Operator counts, input shapes, CPU/device time, memory-event deltas, and a readable top-operator table. Memory-event deltas are not tensor lifetimes or peak allocator values. |
| independent-verification.json, local-verification.json | Recomputed coverage, unchanged outputs, stage accounting, control drift/overhead, and trace GPU activity. |
| artifact-files.json, wandb-verification.json, evidence-upload.json | Uploaded file hashes, downloaded-artifact verification, and final W&B evidence receipt. |
| slurm-accounting.tsv, slurm-output.log, publisher-accounting.tsv, publisher-output.log, exit-status.tsv | Completed-job resources, exit codes, elapsed times, and publisher output; allocation duration is not inference latency. |

The final W&B bundle precedes its own receipt and publisher-completion log; those final files are preserved in Git. Raw datasets, weights, installed environments, and credentials remain outside Git, with pinned sources and setup paths documented in IMPLEMENTATION.md.

To inspect a trace, decompress its .json.gz copy and load the resulting JSON in a compatible Chrome-trace viewer such as Perfetto. CPU annotations identify preprocessing, transfer, nested image features, the first forward, and later cached forwards; use the GPU tracks to inspect actual kernel execution.
