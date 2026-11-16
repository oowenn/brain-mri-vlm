# Step 4: full zero-shot validation

**Status: complete and verified on November 15, 2026 (America/New_York).** GPU job `13925021` and CPU evidence job `13925075` both finished with exit code `0:0`. Execution commit: `492ab709cdf48d8416f6087050860b6254b8b5e8`.

Evaluate all 579 frozen validation images with pretrained `HuggingFaceTB/SmolVLM-500M-Instruct`, revision `753c1752602b5e739af7a7fa6947a635051897a3`, FP16 on RTX A6000. The Step 3 prompt, parser, image handling, seed 42, batch size 1, SDPA, and greedy generation with at most 16 new tokens are unchanged. No training or final-test inference belongs in this step.

| File | Purpose |
| --- | --- |
| `configs/step4.json` | Same inference protocol as Step 3; selects all validation images and Step 4 tracking labels. |
| `scripts/evaluate.py` | Shared image loading, generation, parsing, metric calculation, and W&B tracking. |
| `scripts/verify_step4.py` | Checks all 579 unique validation paths, frozen labels, raw answers, source snapshots, and independently recomputed metrics. |
| `scripts/publish_step4.py` | Verifies scheduler success and remote metrics; downloads the inference artifact to compare every file before publishing the complete checkpoint. |
| `cluster/step4.sbatch` | Runs package checks, tests, CUDA and data checks, FP16 inference, and independent verification inside one GPU allocation. |
| `cluster/publish_step4.sbatch` | Runs the evidence publisher using a small CPU allocation after the GPU job completes. |
| `tests/test_step4.py` | Guards protocol consistency, full-split coverage, invalid-output handling, and detection of altered labels/metrics. |

Reproduction uses the environment and frozen inputs described in [IMPLEMENTATION.md](../../IMPLEMENTATION.md#teammate-setup-environment-and-directories). From the repository on Insomnia, set `MRI_WORK`, `MRI_CONDA_ENV`, and the W&B login as documented, then submit `sbatch cluster/step4.sbatch`. The job has a one-hour limit; this is an allocation limit, not an inference benchmark. Every execution creates a new directory under `runs/<job-id>/`.

## Full validation result

| Measure | Result |
| --- | ---: |
| Validation images | 579 / 579 |
| Correct predictions | 240 / 579 |
| Accuracy | 41.4508% |
| Macro-F1 | 0.26622028 |
| Invalid responses | 0 / 579 (0%) |

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| glioma | 181 | 0.412346 | 0.922652 | 0.569966 |
| meningioma | 157 | 0.000000 | 0.000000 | 0.000000 |
| notumor | 121 | 0.419540 | 0.603306 | 0.494915 |
| pituitary | 120 | 0.000000 | 0.000000 | 0.000000 |

Confusion-matrix rows are true labels; columns are predictions. Invalid predictions have their own column and would remain in the metric denominators.

| True label | glioma | meningioma | notumor | pituitary | INVALID |
| --- | ---: | ---: | ---: | ---: | ---: |
| glioma | 167 | 0 | 14 | 0 | 0 |
| meningioma | 113 | 0 | 44 | 0 | 0 |
| notumor | 48 | 0 | 73 | 0 | 0 |
| pituitary | 77 | 0 | 43 | 0 | 0 |

The model returned valid category names for every image, but used only two categories: **405 glioma and 174 notumor predictions**. It never predicted meningioma or pituitary, so both classes have zero recall. The execution and scoring pipeline works; the pretrained model is a weak four-class classifier under this fixed prompt. The next roadmap step is LoRA adaptation, followed by measured validation and merge checks. These results do not establish that future training or optimization will improve quality.

The GPU allocation completed with exit code `0:0` in **11 minutes**, including startup, tests, source verification, model loading, inference, and uploads. This elapsed time is not an inference-latency measurement. All **19 tests** and all **7,200 source-image checks** passed. A separate local verification reproduced every metric, checked all **19 source snapshots** against commit `492ab70`, and confirmed that the **8 overlapping Step 3 FP16 images** retained identical raw answers, predictions, generated token IDs, and recorded input metadata.

[W&B inference run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/cpawhwk5) · [machine-readable summary](summary.json) · [raw predictions](runs/13925021/fp16/predictions.jsonl) · [independent verification](runs/13925021/independent-verification.json)

The CPU evidence job completed with exit code `0:0` in **1 minute 11 seconds**. It verified the finished W&B inference run, matched all aggregate metrics and class recalls, and downloaded all **10 inference-artifact files** to confirm their SHA-256 hashes match local files. The complete checkpoint is **`step4-checkpoint-13925021:v0`**, digest `74dfbd1877d3efc93db0bcef00be8edf`, recorded in the [evidence publishing run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/sbgxu7p7).

## Run evidence inventory

`summary.json` gives the compact result and verification status. [`checksums.sha256`](checksums.sha256) lists the collected Step 4 evidence files; verify it from the repository root with `sha256sum -c experiments/step4/checksums.sha256` (Linux) or `shasum -a 256 -c experiments/step4/checksums.sha256` (macOS).

Each execution has its own `runs/<job-id>/` directory. For this run, use `runs/13925021/`. Failed or partial executions must retain their own directories.

| File inside the run directory | What it records and why |
| --- | --- |
| `base-commit.txt` | Git commit checked out when the GPU job started. |
| `source/` | Copies of the Python scripts, tests, configurations, Step 4 batch scripts, and package pins present at execution; these preserve the source after the working repository changes. Each file keeps its repository-relative path. |
| `console.log` | Complete job stdout/stderr, including checks and one progress line per image. |
| `slurm-output.log` | Scheduler-captured copy of GPU job output, collected after completion. |
| `slurm-accounting.tsv` | Scheduler state, exit code, node, requested/allocated resources, and elapsed time. Elapsed allocation time is not inference latency. |
| `tests.txt` | The cluster's complete unit-test output, including the test count and final result. |
| `pip-check.txt` | Dependency consistency check. |
| `pip-freeze-original.txt` | Original package snapshot, retaining installed package provenance. |
| `requirements-installed.txt` | Installed package versions in a simple comparison format. |
| `cuda-check.txt` | Python/package versions, allocated GPU, and FP16 CUDA calculation result. |
| `data-verification.json` | Manifest hashes, split counts, source inventory hash, and verification of all 7,200 input files. |
| `exit-status.tsv` | Exit status of the FP16 evaluation process. |
| `independent-verification.json` | Full validation coverage, class supports, independently recomputed scores, response counts, and prediction hashes. |
| `provenance-verification.json` | Local verification matches the cluster; source snapshots match the execution commit; the eight overlapping prior FP16 examples retain the same outputs and input metadata. |
| `wandb-verification.json` | Finished remote run, matched metrics/recalls, and SHA-256 hashes of downloaded artifact files checked against local files. |
| `evidence-upload.json` | W&B checkpoint run URL, artifact name/digest, and inference-verification receipt. Created after the checkpoint upload. |
| `publisher-output.log` | CPU publisher's verification and upload log, collected after its completion. |
| `publisher-accounting.tsv` | CPU publisher's scheduler state, exit code, resources, and elapsed time. |

The `fp16/` subdirectory contains the actual model evaluation:

| File inside `fp16/` | What it records and why |
| --- | --- |
| `run.json` | Status, timestamps, effective configuration, input hash, source hashes, commit, and W&B run identity. |
| `environment.json` | Interpreter, packages, operating system, GPU, CUDA build, driver, node, and Slurm job identity. |
| `selected-images.json` | Ordered validation-image paths with frozen labels, image hashes, and constraint group IDs. |
| `predictions.jsonl` | One record per image: true label, raw generated answer, parsed prediction, generated token IDs, image dimensions, and input tensor/token sizes. |
| `metrics.json` | Accuracy, macro-F1, invalid-response count/rate, per-class precision/recall/F1/support, and a confusion matrix with an explicit INVALID column. |
| `model-runtime.json` | Loaded model class, CUDA parameter placement, and confirmation that no quantized modules were used. |
| `model-config.json` | Loaded model architecture/configuration. |
| `image-processor.json` | Model-native image processing settings. |
| `rendered-prompt.txt` | Exact chat-formatted prompt presented to the model. |
| `generation-config.json` | Loaded generation defaults; the effective greedy/beam/token overrides are recorded in `run.json`. |

The checkpoint artifact contains the run evidence and source snapshot available at upload time. Its own final upload receipt and the publisher's completed log/accounting are saved afterward in GitHub. MRI pixels, model weights, and credentials are not included in these evidence files.

## Collect and publish a new run

After the GPU job has completed successfully, collect the scheduler evidence and submit the CPU publisher. Replace the example job ID for a new execution:

```bash
step4_job=13925021
step4_dir="experiments/step4/runs/$step4_job"
sacct -j "$step4_job" --parsable2 \
  --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,Start,End,NodeList,AllocCPUS,ReqMem,AllocTRES,MaxRSS \
  > "$step4_dir/slurm-accounting.tsv"
cp "logs/step4-$step4_job.out" "$step4_dir/slurm-output.log"
export MRI_RUN_DIR="$step4_dir"
sbatch cluster/publish_step4.sbatch
```

After the publisher finishes, save its `sacct` output as `publisher-accounting.tsv` and its scheduler log as `publisher-output.log` in the same run directory. Confirm both jobs show `COMPLETED` and `0:0`. Keep the new run files, summary, and checksums in GitHub and update [IMPLEMENTATION.md](../../IMPLEMENTATION.md).

## Interpretation and provenance

This is the pretrained zero-shot validation reference, before LoRA training. The optimization baseline later in the roadmap will be the merged, fine-tuned FP16 model. Invalid responses count as errors and false negatives rather than disappearing from the denominator. No prompt/parser tuning is performed after seeing this full-validation result.

The dataset still has unknown patient identities and the documented source-label limitations; the frozen split is not guaranteed fully patient-disjoint. Integrity checks may read test-file bytes, but the evaluator only generates predictions for validation images. There is no final-test quality result or performance benchmark in Step 4.

Historical Step 3 checksums refer to the source at commit `0243f26`; the current shared evaluator has been extended for Step 4, while historical Step 3 run files remain unchanged. Use the recorded commit for old source hashes, and the source snapshot plus commit `492ab70` for this run.
