# Step 8 and preliminary Step 9: image-budget experiment

Status: **verified complete for this one technique and its preliminary Step 9 comparison.** GPU job **13935433** and CPU publisher **13935434** completed 0:0. All **46 tests**, **46 source snapshots**, **180 baseline timing-batch comparisons**, **579 baseline validation-output comparisons**, **120 instrumented-batch comparisons**, and **201 downloaded W&B artifact files** passed. The midpoint report needs one measured optimization attempt. The user selected the supported processor longest-edge change from 2048 to 1024. This supersedes the earlier deferral for this single technique; parallel preprocessing and quantization remain pending.

## Verified result

The 1024 image budget achieved **2.9916x speedup**, **66.57% lower mean batch latency**, and **45.80% lower peak allocated GPU memory**, while passing the predeclared validation gate. This is a preliminary result from one allocation and one fine-tuned checkpoint.

| Matched metric | Baseline 2048 | Image budget 1024 |
| --- | ---: | ---: |
| Mean batch latency | 1437.526 ms | 480.519 ms |
| p50 batch latency | 1434.985 ms | 479.946 ms |
| p95 batch latency | 1496.005 ms | 493.469 ms |
| Mean amortized image cost | 359.382 ms | 120.130 ms |
| Throughput | 2.782558 images/s | 8.324325 images/s |
| Peak allocated GPU memory | 2.776 GiB | 1.504 GiB |
| Peak reserved GPU memory | 3.248 GiB | 1.709 GiB |
| Validation correct / 579 | 494 | 496 |
| Validation accuracy | 85.32% | 85.66% |
| Validation macro-F1 | 0.856293 | 0.859061 |
| Invalid responses | 0 | 0 |

Both settings use A6000, merged FP16, SDPA, batch 4 and four CPU threads. Each has three trials of 10 warmups and 50 measured batches, using seeds 42/43/44 and the same cyclic workloads: 600 measured image executions per setting, not 600 distinct images. Mean-time speedup is the fresh baseline divided by the variant, not a comparison against another node. Trial speedups are **2.968x, 3.049x and 2.959x**. Mean-trial standard deviations are 19.749 ms and 4.914 ms. The fresh baseline differs from Step 6 mean latency by only 0.64%, but Step 6 remains a separate historical result.

Full-validation macro-F1 increases by 0.002768 in this run. The fixed floor is 0.8362927507 and both invalid counts are zero. The increase is not evidence of statistically established generalization improvement. There are **22 changed labels**: 11 wrong-to-correct, 9 correct-to-wrong, and 2 wrong-to-different-wrong; every changed image is listed in prediction-changes.json.

| Class recall | Baseline | 1024 |
| --- | ---: | ---: |
| glioma | 85.64% | 84.53% |
| meningioma | 76.43% | 77.71% |
| notumor | 95.87% | 95.87% |
| pituitary | 85.83% | 87.50% |

### Diagnostic explanation

Separate stage workers use three seeds, 10 warmups and 10 measured batches per seed, excluded from speedup timing. All their warmup/measured outputs agree with the matching unprofiled worker.

| Exclusive diagnostic stage | Baseline ms | 1024 ms |
| --- | ---: | ---: |
| Preprocessing | 1024.784 | 310.939 |
| Transfer and FP16 cast | 26.611 | 8.503 |
| Vision features and connector | 249.391 | 72.768 |
| Prefill excluding vision | 75.870 | 28.572 |
| Cached decoding | 64.059 | 61.371 |
| Other | 4.240 | 4.053 |

The largest absolute savings are preprocessing and vision work, while the short decoder changes little. Measured padded pixels change from **[4,17,3,512,512]** to **[4,5,3,512,512]**. CPU input tensors fall from 356,591,424 to 104,882,368 bytes per stage batch. Baseline validation contexts are 1181 tokens for 576 images, 916 for two and 917 for one; the variant uses **387 tokens for all 579 images**. Labels remain 2–4 generated tokens including EOS. Preprocessing remains the largest diagnostic stage after this change, leaving a reason to test parallel preprocessing next.

### Execution and evidence

- Execution source: c07cc8af2b3581a6d21b4f3af850d6ee4e418e2c. The original submission referenced aad54be; the publisher-status correction was pulled before GPU execution. base-commit.txt and all source snapshots establish the actual execution source.
- GPU 13935433: ins094, A6000, AMD EPYC 7543P host, four allocated CPUs, 8 GiB RAM, burst partition/QoS, 15m58s, COMPLETED 0:0. Actual maximum batch-job host RSS was about 2.11 GiB. Two short metadata commands (process status during setup and lscpu during validation) are visible in accounting; neither was a benchmark.
- CPU publisher 13935434: ins085, two allocated CPUs (one requested), 2 GiB RAM, burst, 1m00s, COMPLETED 0:0. Scheduling dry-runs and changes are retained under scheduling/. The original 16 GiB short reservation would have waited longer; only resource reservation and queue changed, not the inference protocol.
- [W&B experiment imqjpch9](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/imqjpch9). Runtime artifact step8-13935433-image-budget:v0, digest e6f4973d083bfc88fdee8d13d5cb00cf; all 201 downloaded files match (200 payload files plus the hash index).
- [W&B evidence zwfrasg8](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/zwfrasg8). Final bundle step8-checkpoint-13935433:v0, digest 694950cc226e987948d00e253655444f.
- Local transfer checks reproduce all 200 payload hashes and compare all 46 source snapshots to the execution commit. Final receipts/accounting, prediction-change analysis, local verification, summary and documentation are recorded in Git after the W&B bundle; they are not retroactively claimed as part of its immutable upload.

## What changes and why

Step 7 attributed 74.17% of diagnostic batch time to CPU preprocessing. Reducing the image budget should reduce image preparation, transferred tensors, vision work and prefill, but may discard useful MRI detail. This experiment changes only `processor.image_processor.size['longest_edge']`. Tiles remain 512 pixels and image splitting stays enabled. It uses the same saved merged FP16 checkpoint, SDPA, prompt, parser, greedy generation, 16-token output limit, batch 4 and four CPU threads. It does not retrain the model or evaluate the final test set.

The native 2048 processor remains the default for older configurations. The new override checks the saved processor before changing it, and all workers record the effective settings. Batched pixels and tokens are checked against independent single-image processing on eight probes.

## Fixed experiment before execution

- One physical RTX A6000 in one allocation, fresh process for each timing setting/seed. Seeds 42, 43 and 44 use baseline/variant, variant/baseline, baseline/variant order. Alternation reduces order effects but does not eliminate within-allocation variation.
- Each setting gets 10 warmup and 50 measured batches per seed: 150 measured batches and 600 image executions. Workloads and timing boundaries are unchanged from Step 6. Image executions repeat; they are not 600 distinct images.
- Evaluate all 579 validation images for **both** settings. Require exact baseline answers and tokens against Step 6 on all 180 timing batches and 579 validation images.
- The variant gate is macro-F1 at least 0.8362927507, a drop no greater than 0.02 against the fresh baseline, and zero invalid answers. Report accuracy, macro-F1, per-class recall and changed predictions. A failed quality gate is a completed negative experiment, not an accepted optimization.
- Collect separate synchronized stage diagnostics: 10 warmups and 10 measured batches per seed for each setting. These are explanatory measurements, excluded from unprofiled speedup calculations. Compare instrumented answers with matching unprofiled outputs. No new CUDA traces are required for this preliminary experiment; Step 7 traces remain available.
- Retain source/checkpoint/data hashes, actual GPU identity, environment, raw timing/output records, W&B runs and downloaded-artifact verification. The unchanged 1,600-image test set remains held out.

## Files and their purpose

| File | Purpose |
| --- | --- |
| `configs/step8.json` | Predeclared settings, paired order, repetitions and quality gate. |
| `scripts/step8_utils.py` | Derives configurations with one changed inference factor and distinguishes negative experiments from successful optimizations. |
| `scripts/benchmark.py` | Existing inference/timing worker with an opt-in resize override and saved processor/audit evidence. Historical configurations keep their native processor. |
| `scripts/profile_baseline.py` | Existing stage instrumentation, now using the same optional processor override. |
| `scripts/run_step8.py` | Launches six paired timing workers, two full-validation workers and two diagnostic workers, then records W&B results and evidence. |
| `scripts/verify_step8.py` | Independently checks workload pairing, settings, baseline equivalence, metrics, stage accounting, hardware identity and quality gates. |
| `scripts/publish_step8.py` | Checks completed Slurm/W&B records, downloads and verifies evidence, and publishes a final checkpoint bundle. |
| `experiments/step8/analyze_predictions.py` | Derives the 22 changed validation labels and error transitions from the verified raw predictions. |
| `experiments/step8/summarize_run.py` | Builds summary.json only when execution, source provenance, Slurm completion and W&B verification have passed. |
| `tests/test_step8.py` | Guards the one-factor comparison, unchanged old processor behavior, and negative-result interpretation. |
| `cluster/step8.sbatch` | A6000 launcher: four CPUs, 16 GiB RAM, 40-minute limit; tests and source/model/data checks precede measurements. |
| `cluster/publish_step8.sbatch` | Dependent CPU publisher: one requested CPU, 2 GiB RAM, ten-minute limit. |

## Run layout

Under `runs/<job-id>/`, `run.json` and `base-commit.txt` identify execution; `source/` and `source-provenance.json` identify exact code. `configs/` holds generated per-worker configurations and `worker-logs/` holds their output. `baseline-seed42/` through `image1024-seed44/` contain raw timings, summaries, workloads, input audits, processor settings, prompt, generation configuration and actual hardware/environment snapshots. `baseline-quality/` and `image1024-quality/` contain every validation prediction and recomputable metrics. `baseline-stages/` and `image1024-stages/` contain raw spans, exclusive stage summaries, shapes, tensor sizes and sampled GPU telemetry. Each worker has its own status record.

`cpu-hardware.json` records the paired host CPU model; `prediction-changes.json` lists every changed label; `local-verification.json` records the downloaded Git-workspace file/source checks. `summary.json` at the step root is the report-facing aggregate, and `checksums.sha256` covers preserved run/scheduling files. `independent-verification.json` collects paired results, validation gates, shape/token changes and diagnostic comparisons. `tests.txt`, package snapshots, CUDA checks, `data-verification.json` and `checkpoint-verification.json` establish prerequisites. `artifact-files.json`, `wandb-verification.json` and `evidence-upload.json` record uploaded/downloaded files and W&B identities. Scheduler accounting and console/Slurm logs preserve resource requests, actual allocations, exit codes and failures. Allocation elapsed time is not inference latency. Final publisher receipts/accounting and this README are added to Git after publication.

## Reproduce

Use the environment and directory instructions in [IMPLEMENTATION.md](../../IMPLEMENTATION.md). No new dependencies are needed. Reference work root: `/insomnia001/depts/edu/users/ct3013`; environment: `envs/smolmri`; repository: `brain-mri-vlm`; merged checkpoint beneath the repository: `checkpoints/step5/13927137/merged-fp16`. Each teammate substitutes their own paths/account and restores the pinned Step 5 model artifact. The source data root is read from `logs/source-root.txt` or `MRI_SOURCE`.

~~~bash
mkdir -p logs
gpu_job=$(sbatch --parsable --partition=burst --qos=burst --mem=8G --time=00:30:00 cluster/step8.sbatch)
sbatch --partition=burst --qos=burst --dependency=afterok:"$gpu_job" \
  --export=ALL,MRI_RUN_DIR="experiments/step8/runs/$gpu_job" \
  cluster/publish_step8.sbatch
~~~

The measured result is ready for the midpoint report. This preliminary experiment does not complete all techniques or all of Step 9; repeat paired timing on another allocation and evaluate other techniques before final configuration selection.
