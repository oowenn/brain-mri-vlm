# Step 6: fixed-batch FP16 performance baseline

Status: **GPU execution pending**. The initial six-hour/64 GiB reservation (13932216) was canceled before execution after Slurm estimated a next-day start. A two-hour/32 GiB request is queued as job **13932242**; the inference and measurement protocol is unchanged. No batch size or final performance result is available yet.

Seven new benchmark tests pass locally. The complete local suite reports 26 passing tests and one import error: the current macOS Python lacks Pillow, so the six existing data-preparation tests cannot run. The complete 32-test suite and all GPU checks must run in the existing pinned cluster environment. No packages or model weights have been changed.

## What this step does and why

1. **Verify the exact merged model.** Check every model/processor file against Step 5's SHA-256 inventory and link the pinned W&B model artifact. Reproduce eight saved batch-one outputs so the benchmark starts from the verified trained checkpoint.
2. **Check batching.** Use left padding for decoder-only generation. Independently compare each example's effective tokens, image tensors, and image masks against single-image processing. This catches image/prompt association errors before measuring speed.
3. **Sweep batch sizes.** Test 1, 2, 4, 8, 16, and 32, with each candidate in a fresh process. Keep failed/OOM attempts. This is a bounded sweep, not a claim to find the maximum batch that fits the GPU.
4. **Freeze one batch size.** Among successful candidates whose peak allocated memory is at most 80% of GPU memory, choose the smallest batch within 5% of the best measured throughput. This avoids choosing a much larger batch for a negligible throughput gain and leaves memory headroom. The rule is fixed in advance.
5. **Check full validation quality.** Evaluate all 579 images at the selected batch size. Require macro-F1 at least Step 5's 0.8620186316 minus 0.02 and zero invalid outputs. A failure stops the workflow for investigation; the code does not silently choose another batch.
6. **Measure the frozen baseline.** Run three trials, each with 10 warm-up and 50 measured batches. Preserve every sample, compute latency/throughput/memory summaries, and report variation between trial means.
7. **Verify and publish evidence.** Independently recalculate metrics, timings, workloads, and selection; check the completed W&B run; restore its artifact and compare hashes. Update this README and `IMPLEMENTATION.md` with actual results before declaring completion.

Step 7 profiling and optimization implementation remain later work. This step does not train a model or perform final-test inference.

## Measurement contract

The fixed configuration is [`configs/step6.json`](../../configs/step6.json). The model is `step5-13927137-checkpoints:v0`, artifact digest `c5ebce54956dc062af2323368c87cfd0`, subdirectory `merged-fp16`. The A6000 GPU type, processor settings, prompt, parser, FP16 precision, SDPA attention, greedy generation, and 16-new-token limit remain unchanged.

| Item | Fixed rule |
| --- | --- |
| Sweep | Batch sizes 1, 2, 4, 8, 16, 32; order shuffled once with seed 2026; two trials with seeds 42 and 43; 10 warm-ups and 10 measured batches per trial. Sweep statistics are exploratory selection evidence. |
| Final baseline | Selected batch size; three trials with seeds 42, 43, 44; 10 warm-ups and 50 measured batches per trial, producing 150 measured batches. |
| Timing images | For each seed, sample 32 validation images per class and interleave classes. Cycle through this 128-image stream. All candidates use the same seeded stream; larger batches process longer prefixes. Warm-up and measured streams each start at index zero. |
| Timing scope | Synchronized wall time from opening images through JPEG decoding, EXIF/RGB conversion, processor/tokenizer, blocking transfer to GPU, generation, output transfer, text decoding, and parsing. Each call reads and processes its images again. |
| Exclusions | Model loading, integrity hashing, W&B/JSON writes, profiling, request queueing, and waiting to form a batch. Input files have been read during integrity checks; results describe a warm filesystem-cache workload, not cold-storage latency. |
| Batch latency | Time until the complete batch returns. Report mean, p50, and p95. |
| Amortized image time | Batch elapsed time divided by batch size. This measures cost per image; an individual request in the batch waits for the batch result. Queueing and batch-formation delay are unmeasured. |
| Throughput | Total measured images divided by total measured seconds, rather than averaging per-batch reciprocal latencies. |
| GPU memory | Reset PyTorch peak counters before every call; report maximum allocated and reserved bytes over measured calls, including the resident model. Retain the allocator cache between calls. These counters exclude memory owned by the CUDA context or other processes. |
| Trial variability | Save each trial's mean/throughput and sample standard deviation of the three trial means. All trials run in the same allocation; this does not establish variability across nodes or days. |
| Full validation | All 579 original validation images, once each at the selected batch size, retaining the smaller final batch. No test inference. |

Warm-up times are retained but excluded from final statistics. Source/checkpoint hashing happens before model work. Separate worker processes release all candidate allocations, including after GPU OOM. W&B logging and file serialization happen outside the timed interval. The raw generation records include actual token counts; EOS-completed sequences are trimmed only after their first EOS to remove batch-output padding.

Reference documentation: [PyTorch 2.9 CUDA synchronization and memory semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html), [Transformers 4.57.1 Idefics3 processor source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/idefics3/processing_idefics3.py), and the repository's proposal/roadmap. Installed-cluster behavior still needs the recorded GPU audit.

## Files and responsibilities

| File | Purpose |
| --- | --- |
| `configs/step6.json` | Predeclared checkpoint, workload, sweep, selection, quality, and measurement rules. |
| `scripts/benchmark.py` | Isolated GPU worker: checkpoint loading, single/batched input audit, full validation, synchronized timing, output tokens, memory peaks, and runtime telemetry. |
| `scripts/benchmark_utils.py` | Balanced deterministic workload, cyclic batches, EOS trimming, percentile/statistic calculations, and selection rule. |
| `scripts/run_step6.py` | Verify checkpoint bytes, orchestrate fresh workers, preserve OOM results, freeze the batch before final measurements, apply quality gate, and track/upload the run in W&B. |
| `scripts/verify_step6.py` | Independently recompute timing arithmetic and quality, verify exact schedules and selection, and check model/runtime provenance. |
| `scripts/publish_step6.py` | Confirm completed scheduler/W&B status, compare remote metrics and downloaded artifact hashes, then publish the completed evidence bundle. |
| `tests/test_benchmark.py` | Seven tests for workload balance/reproducibility, EOS handling, units/percentiles, warm-up exclusion, ratio-of-totals throughput, memory-aware selection, tampering detection, and frozen protocol. |
| `cluster/step6.sbatch` | Request one A6000, 4 CPUs, 32 GiB host RAM, and up to two hours; run the full test suite and data checks, then benchmark and independently verify. Actual allocated resources come from Slurm accounting. |
| `cluster/preflight_step6.sbatch` | Optional small CPU job on `edu1`: run the complete suite, verify saved model hashes, and audit batched processing while waiting for GPU allocation. It does not measure GPU performance. |
| `cluster/publish_step6.sbatch` | Request 1 CPU, 8 GiB RAM, and 45 minutes after GPU release; collect scheduler output, verify every source snapshot against the execution commit, then verify/publish remote evidence. Supports an `afterok` Slurm dependency. |
| `preflight/` | CPU preflight logs, executed launcher, commit, tests, checkpoint hashes, and input audit. GPU preflight repeats these checks in its own recorded runtime. |
| `scheduling/` | Scheduler evidence for an initial reservation canceled before execution; this is not a benchmark run. |
| `local-checks/` | Current macOS CPU-test logs, source hashes, environment identity, and the explicit boundary that no GPU/timing result has been measured. |

The existing Step 3–5 source and immutable run evidence are preserved.

## Cluster and teammate setup

Use the pinned environment and per-user layout in [IMPLEMENTATION.md](../../IMPLEMENTATION.md). Sean's reference locations are:

```text
/insomnia001/depts/edu/users/ct3013/envs/smolmri
/insomnia001/depts/edu/users/ct3013/brain-mri-vlm
/insomnia001/depts/edu/users/ct3013/brain-mri-vlm/checkpoints/step5/13927137/merged-fp16
/insomnia001/depts/edu/users/ct3013/brain-mri-vlm/data/prepared-v3-conservative-patients
/insomnia001/depts/edu/users/ct3013/cache/kaggle/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/2
```

Each teammate uses their own root and credentials. Restore the exact model with the [Step 5 restore command](../step5/README.md#restore-the-completed-checkpoint); no retraining is required. The new override `MRI_MERGED_MODEL` points to the restored `merged-fp16` directory. `MRI_SOURCE` can override `logs/source-root.txt`. Existing Conda, Hugging Face, Kaggle, temporary, and W&B cache variables remain unchanged; no new environment packages are required by Step 6.

From the repository on the Insomnia login node, after networking and the existing setup are ready:

```bash
export MRI_WORK="/insomnia001/depts/edu/users/$USER"
export MRI_CONDA_ENV="$MRI_WORK/envs/smolmri"
export MRI_MERGED_MODEL="$MRI_WORK/brain-mri-vlm/checkpoints/step5/13927137/merged-fp16"
export WANDB_BASE_URL="https://api.forge.coreweave.com"
mkdir -p logs
sbatch cluster/step6.sbatch
```

A different server must adapt its Slurm account/partition/GPU names and Conda initialization, while retaining the model, data, processor, prompt, and measurement settings. This portability setup has not been executed on another teammate's server.

After the GPU job completes successfully, substitute its job ID and run:

```bash
export MRI_RUN_DIR="experiments/step6/runs/JOB_ID"
sacct -j JOB_ID --parsable2 \
  --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,Start,End,NodeList,AllocCPUS,ReqMem,AllocTRES,MaxRSS \
  > "$MRI_RUN_DIR/slurm-accounting.tsv"
cp logs/step6-JOB_ID.out "$MRI_RUN_DIR/slurm-output.log"
sbatch cluster/publish_step6.sbatch
```

To queue publication in advance, after submitting the GPU job and setting `MRI_RUN_DIR` to that job's run directory:

```bash
sbatch --dependency=afterok:JOB_ID cluster/publish_step6.sbatch
```

The publisher collects GPU accounting/logs automatically and refuses to publish a failed or incomplete benchmark. If the GPU job fails, the dependency will not run; preserve the failure and cancel or replace the dependent publisher after investigation.

Preserve the publisher's final output/accounting and copy the completed compact run into GitHub. Retain weights and caches outside Git. The GPU batch job survives an SSH disconnect; its own time limit still applies.

## Generated evidence guide

These are intended outputs under `runs/<job-id>/`; their existence and verification, not this guide, establish execution.

| File or directory | What it proves or records |
| --- | --- |
| `base-commit.txt`, `source/`, `source-provenance.json` | Execution commit, frozen copies of code/tests/configs/batch scripts/environment pins, and byte-for-byte verification against that commit. |
| `tests.txt`, `pip-check.txt`, `cuda-check.txt`, package listings | Full-suite status, dependency consistency, GPU execution, and installed environment. |
| `data-verification.json`, `checkpoint-verification.json` | Frozen manifests/all 7,200 images and every merged model/processor file match their recorded hashes. |
| `run.json` | Overall status, effective config, selected batch, quality/timing results, and W&B identity. |
| `active-worker.json`, `worker-logs/` | Active subprocess command and complete logs for every candidate/phase, including failures. |
| `audit/` | Exact eight-image batch-one comparison to Step 5, batched token/image/mask checks, batched probe outputs, runtime and generation settings. |
| `sweep-order.json`, `sweep.json`, `sweep-bNNN/` | All candidates in fixed randomized order, successful timing results, and explicit OOM/failure evidence. |
| `frozen-baseline.json` | Selected batch and complete checkpoint/config identity, saved before full quality and final timing. |
| `quality/`, `quality-gate.json` | All 579 raw answers/labels/tokens, input metadata, confusion matrix, class recalls, and comparison with Step 5 quality. |
| `baseline/` | Three final repeated trials at the frozen batch size. |
| Each timing worker's `workload-<seed>.json` | Exact image identities/hashes and order used to form every batch. |
| Each timing worker's `timings.jsonl` | Every warm-up/measured sample: nanosecond elapsed time, images, input/output lengths, raw answers, memory peaks, seed/trial/iteration. |
| Each timing worker's `summary.json` | Batch and amortized mean/p50/p95, ratio-of-totals throughput, peak allocated/reserved memory, and per-trial variation. |
| Worker `environment.json`, `model-runtime.json`, `gpu-before.csv`, `gpu-after.csv` | Python/packages, hardware UUID, CPU threads, CUDA settings, plain FP16 checkpoint checks, and GPU clocks/power/temperature telemetry. |
| Worker `generation-config.json`, `rendered-prompt.txt` | Model defaults and exact prompt; generation overrides remain in the frozen config. |
| `independent-verification.json` | Recomputed timing/quality metrics, workloads, selection, and checkpoint/runtime checks. |
| `artifact-files.json`, `wandb-verification.json`, `evidence-upload.json` | Stable artifact file hashes, verified remote state/metrics/downloads, and final bundle receipt. |
| Console, scheduler logs, `exit-status.tsv`, accounting files | Job identity, requested/allocated resources, process outcome, and elapsed job time. Job duration is not a latency measurement. |

The running console log is excluded from the first W&B artifact. The completed GPU console/accounting go in the later evidence bundle. That bundle's own receipt and final publisher logs/accounting are collected afterward in GitHub.
