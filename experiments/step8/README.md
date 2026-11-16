# Step 8 and preliminary Step 9: image-budget experiment

Status: **implemented; cluster measurements pending.** The midpoint report needs one measured optimization attempt. The user selected the supported processor longest-edge change from 2048 to 1024. This supersedes the earlier deferral for this single technique; parallel preprocessing and quantization remain pending.

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
| `tests/test_step8.py` | Guards the one-factor comparison, unchanged old processor behavior, and negative-result interpretation. |
| `cluster/step8.sbatch` | A6000 launcher: four CPUs, 16 GiB RAM, 40-minute limit; tests and source/model/data checks precede measurements. |
| `cluster/publish_step8.sbatch` | Dependent CPU publisher: one requested CPU, 2 GiB RAM, ten-minute limit. |

## Run layout

Under `runs/<job-id>/`, `run.json` and `base-commit.txt` identify execution; `source/` and `source-provenance.json` identify exact code. `configs/` holds generated per-worker configurations and `worker-logs/` holds their output. `baseline-seed42/` through `image1024-seed44/` contain raw timings, summaries, workloads, input audits, processor settings, prompt, generation configuration and actual hardware/environment snapshots. `baseline-quality/` and `image1024-quality/` contain every validation prediction and recomputable metrics. `baseline-stages/` and `image1024-stages/` contain raw spans, exclusive stage summaries, shapes, tensor sizes and sampled GPU telemetry. Each worker has its own status record.

`independent-verification.json` collects paired results, validation gates, shape/token changes and diagnostic comparisons. `tests.txt`, package snapshots, CUDA checks, `data-verification.json` and `checkpoint-verification.json` establish prerequisites. `artifact-files.json`, `wandb-verification.json` and `evidence-upload.json` record uploaded/downloaded files and W&B identities. Scheduler accounting and console/Slurm logs preserve resource requests, actual allocations, exit codes and failures. Allocation elapsed time is not inference latency. Final publisher receipts/accounting and this README are added to Git after publication.

## Reproduce

Use the environment and directory instructions in [IMPLEMENTATION.md](../../IMPLEMENTATION.md). No new dependencies are needed. Reference work root: `/insomnia001/depts/edu/users/ct3013`; environment: `envs/smolmri`; repository: `brain-mri-vlm`; merged checkpoint beneath the repository: `checkpoints/step5/13927137/merged-fp16`. Each teammate substitutes their own paths/account and restores the pinned Step 5 model artifact. The source data root is read from `logs/source-root.txt` or `MRI_SOURCE`.

~~~bash
mkdir -p logs
gpu_job=$(sbatch --parsable cluster/step8.sbatch)
sbatch --dependency=afterok:"$gpu_job" \
  --export=ALL,MRI_RUN_DIR="experiments/step8/runs/$gpu_job" \
  cluster/publish_step8.sbatch
~~~

Results and report conclusions will be added after the actual jobs and artifact checks finish. This preliminary experiment does not complete all techniques or all of Step 9.
