# Step 7: profile the frozen FP16 baseline

Status: implementation ready for cluster execution; measurements pending.

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
