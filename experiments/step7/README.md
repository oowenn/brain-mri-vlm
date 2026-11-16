# Step 7: profile the frozen FP16 baseline

Status: all GPU workers finished; CPU-only verification/publishing recovery is pending. The original GPU job failed in postprocessing because mirrored GPU annotations were counted as extra CPU batches. Measurements are retained unchanged.

Step 6 remains the canonical performance and full-validation reference. This step explains where its runtime goes and proposes two or three changes for Step 8. It does not implement those changes or evaluate the final test set.


## Measured breakdown

The completed stage worker measured 30 batches of four images after warmup, ten per seed. All three trials identify preprocessing as the largest cost.

| Exclusive stage | Mean ms per batch | Share of diagnostic elapsed time |
| --- | ---: | ---: |
| CPU preprocessing | 1348.567 | 74.17% |
| CPU-to-GPU transfer and FP16 cast | 70.967 | 3.90% |
| Vision features, connector, and image-mask preparation | 250.158 | 13.76% |
| First forward excluding nested vision features | 75.921 | 4.18% |
| Later cached one-token forwards | 67.895 | 3.73% |
| Generation loop, postprocessing, and instrumentation residual | 4.604 | 0.25% |

The synchronized diagnostic total is **1818.111 ms per batch**. Inside preprocessing, image reading/EXIF/RGB conversion averages **4.695 ms**, while the image/text processor averages **1343.775 ms**. The processor accounts for about 99.6% of preprocessing time. This is evidence of a host preprocessing bottleneck; fast image reads make storage reads a small measured component for this warmed workload.

The separate uninstrumented control measured **1830.319 ms mean, 1831.635 ms p50, 1859.369 ms p95, and 2.185411 images/s**, with the same **2.776 GiB allocated / 3.248 GiB reserved** peaks as Step 6. Step 6 remains the canonical 1428.367 ms / 2.800400 images/s result. Physical node/GPU and runtime conditions differ; this control drift is not an optimization result.

Model parameter storage is **0.945 GiB**, or **34.05%** of the control allocator peak. The remaining peak includes input tensors, activations, caches, buffers, and temporary workspaces. Weights do not dominate the measured peak, and the model fits comfortably on the allocated A6000.

## Proposed Step 8 experiments: three techniques

These are proposals only. No optimized model path or final-test inference has been run.

1. **Parallel CPU preprocessing — first priority.** Process the four images concurrently while retaining the same resize/crop/normalization and token construction. Keep the four-core allocation, cap per-worker library threads, and measure worker handoff/assembly inside end-to-end timing. Start with two workers and then four. Check exact processed tensors and token/image associations before full validation. This targets the 74.2% preprocessing share. Process startup, serialization, host-memory traffic, or contention may erase the benefit; warmup may exclude one-time startup but must not hide per-batch work. Proposal lead: Owen, with Sean's matched benchmark.
2. **Smaller image/crop budget — second priority.** Change only the processor longest edge from 2048 to 1024, retaining 512-pixel tiles and image splitting. The supported processor option reduces the maximum square-image views from 17 to 5 and should reduce processing, transferred data, vision work, and prefill length. Record actual crop/token counts and peak memory. Smaller visual detail can reduce tumor-class recall, so select on all 579 validation images. Keep the checkpoint, prompt, batch size, GPU, and generation policy fixed. Proposal lead: Sean. The option is supported by the [pinned Idefics3 processor source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/idefics3/image_processing_idefics3.py); this run has not measured its benefit.
3. **Weight quantization — lower-priority hypothesis test.** Evaluate INT8 and NF4 as two levels of one technique, each loaded from the same merged checkpoint. Keep the original 2048-pixel processor, FP16 computation for NF4, batch 4, and all weights on the GPU; record which modules convert. Its purpose is testing the proposal's memory/latency claim. With weights only about one third of the current peak and CPU preprocessing dominant, reduced weight storage alone cannot remove the main cost. Quantization may change predictions or add kernel/dequantization overhead. Step 3's pretrained eight-image smoke tests do not establish quality or speed for this fine-tuned checkpoint. Proposal lead: Ai. See the [pinned Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes).

Defer separate transfer-only, prompt-shortening, and decoding techniques: transfer, prefill, and decode each account for roughly 4% of the current diagnostic time. Generation already emits short labels using a KV cache. Retain these as possible follow-ups if the profile changes after the larger costs are reduced.

For every isolated variant, compare against a fresh baseline in the same allocation, at batch 4 with the same three seeds and workloads. Use ten warmups and fifty unprofiled measurements per trial; retain failures/slowdowns. Re-profile separately and evaluate all 579 validation images. The Step 6 batch-4 macro-F1 is **0.8562927507**, so the fixed absolute 0.02 allowance gives **0.8362927507** as the validation floor; require zero invalid outputs and report every class recall. Do not tune on the final 1,600-image test set. Combine techniques only after their isolated effects are understood. These outcomes are hypotheses until Steps 8–9 run.

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

## Interpretation boundaries

- The control, synchronized diagnostics, and traces run sequentially in fresh processes. Their timing difference includes instrumentation plus phase-order and runtime variation; it is not a randomized estimate of profiler overhead.
- CUDA-event stage durations are elapsed stream intervals, including possible launch gaps. They are not sums of GPU kernel durations. The trace separately measures the union of actual kernel/copy intervals.
- The job uses the same A6000 GPU type and thread count as Step 6, but another physical GPU/node. Any control drift is reported rather than called an optimization.
- GPU activity/utilization and kernel names do not by themselves prove arithmetic-throughput versus device-memory-bandwidth saturation. A strict compute-bound/memory-bound classification would need hardware-counter evidence.
- Parameter storage, allocator peak, allocator reservation, and nvidia-smi process/device usage measure different things. Shape/memory trace collection can retain extra tensors; use the unprofiled control for baseline memory.
- Exact output agreement on the profiling workloads preserves their behavior; full-validation quality remains the separately recorded Step 6 result. No new claim about unseen/test images follows.
