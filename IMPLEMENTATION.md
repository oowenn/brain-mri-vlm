# Implementation status and handoff

Last updated: **2026-11-16 (America/New_York)**

Latest verified implementation commit: [1697704b4e12910322f61b5259c9483184cafad2](https://github.com/oowenn/brain-mri-vlm/commit/1697704b4e12910322f61b5259c9483184cafad2) (Step 6 execution source; final result documentation and launcher resource defaults are subsequent changes).

## Current position

**Step 6 is complete. Step 7 GPU measurements are complete; CPU-only evidence recovery is pending after a trace annotation-count check failed.** The frozen FP16 baseline uses **batch size 4 on RTX A6000 with four CPU threads**: **2.8004 images/s, 1428.367 ms mean / 1430.270 ms p50 / 1473.171 ms p95 batch latency, and 2.776 GiB peak allocated GPU memory**. Three trials produced 150 measured batches (600 image executions). Full batch-4 validation gave **494/579 correct (85.32%), macro-F1 0.856293, and zero invalid responses**.

Batching changed four predictions from the Step 5 batch-one result (498 correct, macro-F1 0.862019). The absolute macro-F1 drop of 0.005726 passes the fixed 0.02 allowance and is preserved in the evidence. All 32 tests, source/checkpoint hashes, source snapshots, independent timing/quality checks, and W&B runtime-artifact downloads passed. Step 7 instrumentation is being prepared; no measured profiling result, optimization speedup claim, or final-test inference is available yet.

This is the living implementation record. It follows the numbering in the [project roadmap](output/pdf/brain-mri-vlm-roadmap.pdf). The [Step 3 README](experiments/step3/README.md), [Step 4 README](experiments/step4/README.md), [Step 5 README](experiments/step5/README.md), and [Step 6 README](experiments/step6/README.md) contain detailed file inventories and reproduction commands; historical run files remain the evidence for what actually happened.

The latest team discussion proposes working through profiling and selecting two or three optimizations, with implementation deferred for now. Step 6 is complete with the same evidence, W&B, repository, and teammate-documentation requirements. The next work is stage profiling and evidence-based selection of techniques; implementing them remains a later step.

## Roadmap status

| Step | Work | Status | Completion evidence or required output |
| --- | --- | --- | --- |
| 1 | Define the experiment and comparison rules | Defined | Primary model, four labels, quality/performance metrics, matched GPU/batch rules, and 0.02 absolute macro-F1 allowance are documented. Step 6 selected and froze batch size 4. |
| 2 | Audit data and freeze splits | Complete | Conservative manifests, source and output hashes, recovered patient constraints, and documented identity limitations. |
| 3 | Verify GPU, model, and evaluator | Complete | All 7,200 source images verified; 14 tests passed; FP16, INT8, and INT4 completed the same eight-image validation sample; evidence saved in GitHub and W&B. |
| 4 | Run zero-shot evaluation | Complete | All 579 validation images; accuracy 0.414508, macro-F1 0.266220, invalid rate 0; independently verified predictions and W&B artifacts. |
| 5 | Fine-tune once with LoRA and merge into FP16 | Complete | Three epochs, 615 updates, epoch 3 selected; merged FP16 validation accuracy 0.860104, macro-F1 0.862019; merge and downloaded checkpoint hashes verified. |
| 6 | Select batch size and measure the FP16 baseline | Complete | Six-size sweep; frozen batch 4; three trials / 150 measurements; 2.8004 images/s; 2.776 GiB peak allocated; full validation and downloaded W&B evidence verified. |
| 7 | Profile the five inference stages | Measurements complete; CPU verification recovery pending | Stage breakdown, representative traces, bottleneck explanation, and evidence-based proposals for two or three techniques. |
| 8 | Implement selected techniques | Deferred under current team discussion | Isolated implementations/configurations based on the measured bottleneck. Step 3 quantization loading is only a compatibility check. |
| 9 | Re-measure, re-profile, and validate variants | Not started; follows Step 8 | Matched before/after measurements and validation quality checks, including unsuccessful variants. |
| 10 | Freeze settings and evaluate the test set | Not started | Final results on the unchanged 1,600-image test set, without using test outcomes to tune settings. |
| 11 | Explain results and deliver the project | Not started as a final deliverable | Reproduction instructions, report/presentation, bottleneck explanation, and demonstration. Existing notes and evidence provide a foundation. |

## Completed work

### Step 1: protocol

- Primary model: `HuggingFaceTB/SmolVLM-500M-Instruct`.
- Labels, in order: `glioma`, `meningioma`, `notumor`, `pituitary`.
- Record macro-F1, per-class recall, invalid-response rate, latency, throughput, and peak GPU memory at the appropriate stages.
- Compare optimizations against the **merged, fine-tuned FP16 model**, using the same GPU type and a batch size selected on that baseline. The pretrained Step 3 model is not that optimization baseline.
- The proposed quality allowance is a maximum **0.02 absolute macro-F1 drop** relative to the baseline. Quantization's memory/latency effect remains a hypothesis to test.
- The proposal's timing protocol calls for 10 warm-up runs, 50 timed measurements, and repeated trials. Step 3 did not perform this benchmark.

### Step 2: data checkpoint

- Source: Kaggle `masoudnickparvar/brain-tumor-mri-dataset`, version **2**.
- Frozen policy: [`configs/data_policy.json`](configs/data_policy.json), version `v3-conservative-patients`.
- Shared manifests: `coms6998-team32-fall2026/brain-mri-vlm/brain-mri-manifests:v2`. The artifact's version number and the data-policy version number are separate.
- Split counts: **3,279 train / 579 validation / 1,600 test / 1,742 excluded**, accounting for all 7,200 images.
- Accepted image matches and 1,778 confirmed source correspondences covering 212 recovered patients constrain splitting. Unknown identities remain; the dataset is **not fully patient-disjoint**.
- Official test images and labels were preserved, including one flagged source-label disagreement. Twelve conflicting original training annotations were excluded.
- Step 3 independently rechecked the frozen manifests and every expected source image's bytes. It did not regenerate the splits or rerun patient recovery.

Evidence: [data finalization record](experiments/data_finalization.md), [similarity calibration](experiments/similarity_calibration.md), and [fresh data verification](experiments/step3/runs/13924504/data-verification.json).

### Step 3: runtime and evaluator

- Verified RTX A6000 execution on Insomnia, including an FP16 CUDA calculation and actual MRI generation.
- Recorded Python **3.11.18**, PyTorch **2.9.1+cu126**, torchvision **0.24.1+cu126**, Transformers **4.57.1**, and bitsandbytes **0.48.2**. Full installed-version pins are in [`requirements-cluster.txt`](requirements-cluster.txt).
- Pinned model revision: `753c1752602b5e739af7a7fa6947a635051897a3`.
- Built a reusable evaluator with manifest/image checks, original-image loading, EXIF orientation, RGB conversion, the model's own processor, fixed prompts, response parsing, and metric calculation.
- Corrected the parser to accept case differences, surrounding whitespace, and one optional final period. Explanations, ambiguous responses, and unknown labels remain invalid. Invalid outputs count as errors rather than being dropped.
- Used the same deterministic sample of two validation images per class, seed 42, batch size 1, SDPA attention, greedy generation, and at most 16 new tokens in all three modes.
- INT8 and INT4 each replaced **297 linear modules** and completed inference with all model parameters on `cuda:0`.
- Preserved original interactive logs, the parser correction, complete batch-run evidence, W&B verification, package snapshots, and checksums.
- Published the implementation and evidence to GitHub `main`; the local and cluster checkouts were synchronized to `0243f26` at the end of Step 3. This tracker and its maintenance instructions are subsequent documentation; the recorded experiment still refers to that implementation commit.

| Mode | Valid responses | Correct on eight-image sample | Sample macro-F1 | W&B run |
| --- | --- | --- | --- | --- |
| FP16 | 8/8 | 3/8 | 0.2778 | [7b74ixt5](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/7b74ixt5) |
| INT8 | 8/8 | 3/8 | 0.2778 | [3n3x3b29](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/3n3x3b29) |
| INT4 / NF4 | 8/8 | 0/8 | 0.0000 | [kgqava8s](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/kgqava8s) |

These establish execution and scoring compatibility. They do not establish full-dataset quality, quality preservation under quantization, or any speed/memory advantage. INT4's poor sample result must remain visible in subsequent analysis.

GPU job **13924504** completed with exit code `0:0` in **7 minutes 31 seconds**, including setup, data checking, model loading, and uploads. CPU publishing job **13924530** also completed with exit code `0:0`. Neither elapsed job time is an inference-latency measurement.

The publisher independently verified all three finished W&B runs and matched their remote metrics to the local results. The [complete evidence bundle](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/6bdx7var) is `step3-checkpoint-13924504:v0`. See the [summary](experiments/step3/summary.json), [run files](experiments/step3/runs/13924504/), and [upload receipt](experiments/step3/runs/13924504/evidence-upload.json).

### Step 4: full zero-shot validation

- Evaluated all **579 unique validation images** with pretrained FP16 SmolVLM on RTX A6000, preserving the Step 3 prompt, parser, model revision, preprocessing, seed, and generation settings. No training or final-test inference was performed.
- **240/579 correct; accuracy 0.4145077720; macro-F1 0.2662202811; invalid responses 0/579.**
- Per-class recall: glioma **0.922652** (167/181), meningioma **0** (0/157), notumor **0.603306** (73/121), pituitary **0** (0/120). The model predicted only glioma (405) and notumor (174); valid formatting does not imply good four-class classification.
- All **19 tests** passed and all **7,200 source images** matched their frozen hashes. Independent cluster and local checks reproduced the metrics from the saved raw answers and confirmed complete validation coverage.
- All **19 saved source files** match execution commit `492ab70`. All **8 overlapping Step 3 FP16 examples** retain identical raw responses, predicted labels, generated token IDs, and recorded input metadata.
- GPU job **13925021** on `ins087` completed with exit `0:0` in **11 minutes**. CPU evidence job **13925075** on `ins093` completed with exit `0:0` in **1 minute 11 seconds**. These are allocation times, not performance measurements.
- Verified the finished W&B inference run, all aggregate metrics and class recalls, and byte-for-byte agreement for all **10 downloaded inference-artifact files**. Published checkpoint `step4-checkpoint-13925021:v0`, digest `74dfbd1877d3efc93db0bcef00be8edf`.

Evidence: [Step 4 README and confusion matrix](experiments/step4/README.md), [summary](experiments/step4/summary.json), [raw predictions](experiments/step4/runs/13925021/fp16/predictions.jsonl), [independent verification](experiments/step4/runs/13925021/independent-verification.json), [W&B inference run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/cpawhwk5), and [W&B checkpoint run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/sbgxu7p7).

### Step 5: LoRA training and saved FP16 checkpoint

- Completed one fixed three-epoch run over all **3,279 training images per epoch**, with **615 optimizer updates**. Rank-16/alpha-32 LoRA trained **8,683,520 parameters** in 224 language-decoder modules; base, vision, and connector weights remained frozen.
- Verified the response-only label mask, native/manual loss agreement, LoRA-only gradients, and all eight unchanged pretraining reference answers. All **25 tests** and **7,200 source-image checks** passed.
- Evaluated all **579 validation images after every epoch**, with loss and quality logged to W&B. Selected **epoch 3** using highest macro-F1, then lowest validation loss, then earliest epoch; the rule was fixed before training.
- Reloaded the selected saved adapter, merged in FP32, cast to FP16, saved, and reloaded the standalone checkpoint. All 8 probe token sequences match; maximum/mean absolute logit differences are **0.12109375/0.01579220**. Full-validation label agreement is **99.4819%**, with macro-F1 difference **0.00445901**. All predeclared merge tolerances passed.
- Final merged validation result: **498/579 correct; accuracy 0.8601036269; macro-F1 0.8620186316; invalid responses 0/579**. These are validation-selection results, not final held-out test results.
- Independent checks reproduced training coverage/order, all validation metrics, selection, and saved merge-logit differences. All **25 execution-source snapshots** match commit `873598e`.
- GPU job **13927137** on `ins093` completed `0:0` in **01:50:40**. CPU publishing job **13927707** completed `0:0` in **00:01:54**. Slurm allocated 6 CPUs for the GPU job (4 requested) and 2 for publishing (1 requested); training used 4 CPU threads. Allocation duration is not an inference benchmark.
- Verified the finished training run and downloaded evidence, then backed up and downloaded all **52 checkpoint files (1,350,270,604 bytes)** with matching SHA-256 hashes. Model artifact: `step5-13927137-checkpoints:v0`. Evidence artifact: `step5-checkpoint-13927137:v0`.

Evidence: [Step 5 README and restore command](experiments/step5/README.md), [summary](experiments/step5/summary.json), [run files](experiments/step5/runs/13927137/), [W&B training run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/0povsppx), and [W&B evidence run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/qi5s4b7s).

### Step 6: selected batch and measured FP16 baseline

- Swept **1/2/4/8/16/32** on one A6000 with two trials per candidate; every candidate completed without OOM. Batch 4 was the smallest within 5% of the best eligible throughput: 2.820566 versus 2.961316 images/s at batch 32. It narrowly clears the 2.813250 cutoff; this is a bounded single-allocation selection, not a universal optimum.
- Froze batch 4 before full validation and final timing. Ran **three trials, each with 10 warm-ups and 50 measured batches**: 150 measured batches, 600 image executions, 305 distinct timing images.
- Final batch latency: **mean 1428.367 ms, p50 1430.270 ms, p95 1473.171 ms**. Throughput **2.800400 images/s**; peak allocated/reserved GPU memory **2.776 / 3.248 GiB**. Trial-mean standard deviation **10.678 ms**. Amortized image cost **357.092 ms** is not per-request latency.
- Full batch-4 validation: **494/579 correct; accuracy 0.853195; macro-F1 0.856293; invalid 0/579**. Four glioma predictions changed to meningioma relative to Step 5 batch one; the macro-F1 drop **0.005726** passed the 0.02 gate. Future matched variants compare to the batch-4 baseline.
- All **32 tests**, **7,200 image hashes**, **12 model/processor file hashes**, **32 execution-source snapshots**, and eight exact batch-one reference outputs passed. Independent local/cluster checks reproduced every sweep, selection, final timing result, and validation metric.
- GPU **13932242**, ins094, **4 CPUs / 16 GiB / one A6000**, completed **0:0 in 25m12s**. Publisher **13932262**, ins021, **2 allocated CPUs / 4 GiB**, completed **0:0 in 53s**. The reduced host-memory reservation enabled the GPU allocation; original scheduling records and launcher snapshots remain preserved.
- W&B benchmark **apjpl38l** is finished with matching metrics/config. All **153 runtime-artifact files** were downloaded and hash-verified. Evidence run **z3njw1g9** published **step6-checkpoint-13932242:v0**; subsequent publisher receipt/logs/accounting are preserved in GitHub.

Evidence: [Step 6 README](experiments/step6/README.md), [summary](experiments/step6/summary.json), [frozen baseline](experiments/step6/runs/13932242/frozen-baseline.json), [raw timings](experiments/step6/runs/13932242/baseline/timings.jsonl), and [verification](experiments/step6/runs/13932242/local-verification.json).

## Implementation map

| Component | Files | Current purpose |
| --- | --- | --- |
| Image audit | [`scripts/prepare_data.py`](scripts/prepare_data.py), [`configs/data.json`](configs/data.json) | Inventory, duplicate constraints, initial split preparation, and reproducibility checks. |
| Patient-aware finalization | [`scripts/finalize_data.py`](scripts/finalize_data.py), [`configs/data_policy.json`](configs/data_policy.json) | Apply the approved recovered-patient and annotation policy to produce the frozen manifests. |
| Runtime checks | [`scripts/check_environment.py`](scripts/check_environment.py), [`requirements-cluster.txt`](requirements-cluster.txt) | Check package/GPU readiness and preserve the tested installed versions. |
| Frozen-input verification | [`scripts/verify_data.py`](scripts/verify_data.py) | Verify manifests, accepted constraints, and source bytes before model work. |
| Shared evaluator | [`scripts/evaluate.py`](scripts/evaluate.py), [`configs/step3.json`](configs/step3.json), [`configs/step4.json`](configs/step4.json) | Load images, build prompts, generate/parse labels, calculate metrics, and track stage-specific runs. |
| Batch execution | [`cluster/step3.sbatch`](cluster/step3.sbatch), [`cluster/step4.sbatch`](cluster/step4.sbatch) | Reproduce compatibility checks or full FP16 validation inside a GPU allocation. |
| W&B evidence verification | [`scripts/publish_step3.py`](scripts/publish_step3.py), [`cluster/publish_step3.sbatch`](cluster/publish_step3.sbatch) | Check remote run completion/metrics and archive the complete checkpoint using a CPU allocation. |
| Full-validation verification | [`scripts/verify_step4.py`](scripts/verify_step4.py), [`scripts/publish_step4.py`](scripts/publish_step4.py), [`cluster/publish_step4.sbatch`](cluster/publish_step4.sbatch) | Independently verify coverage, raw answers, metrics, scheduler success, and downloaded W&B evidence before publishing the Step 4 checkpoint. |
| LoRA training and checkpoint verification | [`scripts/train_lora.py`](scripts/train_lora.py), [`scripts/lora_utils.py`](scripts/lora_utils.py), [`configs/step5.json`](configs/step5.json), [`scripts/verify_step5.py`](scripts/verify_step5.py), [`cluster/step5.sbatch`](cluster/step5.sbatch) | Response-only training, per-epoch validation, checkpoint selection, saved/reloaded FP16 merge, and independent checks. |
| Model backup | [`scripts/publish_step5.py`](scripts/publish_step5.py), [`cluster/publish_step5.sbatch`](cluster/publish_step5.sbatch) | Verify remote tracking and download-check every uploaded adapter, optimizer state, processor, and merged model file. |
| Baseline benchmark | [`scripts/benchmark.py`](scripts/benchmark.py), [`scripts/benchmark_utils.py`](scripts/benchmark_utils.py), [`scripts/run_step6.py`](scripts/run_step6.py), [`configs/step6.json`](configs/step6.json), [`cluster/step6.sbatch`](cluster/step6.sbatch) | Audit batching, sweep candidates, freeze the baseline, and retain raw repeated timings and quality. Verified six-size sweep and batch-4 baseline. |
| Baseline evidence verification | [`scripts/verify_step6.py`](scripts/verify_step6.py), [`scripts/publish_step6.py`](scripts/publish_step6.py), [`cluster/publish_step6.sbatch`](cluster/publish_step6.sbatch) | Independently recompute measurements, check W&B metrics and artifact downloads, and publish the evidence bundle. |
| Tests | [`tests/`](tests/) | Ten data tests, four evaluator tests, five Step 4 verification tests, and six LoRA tests passed at Step 5 (25 total); seven new benchmark tests bring the full suite to 32, all passed in cluster CPU preflight `13932248`. |
| Evidence documentation | [`experiments/step3/README.md`](experiments/step3/README.md), [`experiments/step4/README.md`](experiments/step4/README.md), [`experiments/step5/README.md`](experiments/step5/README.md) | Explain step outputs, reproduction commands, results, and limitations. |
| Living progress record | [`IMPLEMENTATION.md`](IMPLEMENTATION.md), [`AGENTS.md`](AGENTS.md) | Maintain the current implementation status and carry the update instruction into later repository sessions. |

## Step 4 completion checklist

- [x] Added Step 4 configuration, tracking names, GPU batch job, and CPU evidence publisher.
- [x] Preserved the frozen inputs and accepted inference protocol.
- [x] Executed full FP16 validation and retained every raw answer and prediction.
- [x] Verified 579 unique images with supports 181/157/121/120, with no dropped images or final-test predictions.
- [x] Independently recomputed accuracy, macro-F1, class recalls, the confusion matrix, and invalid-response rate.
- [x] Verified finished W&B runs, matched remote metrics, compared downloaded inference artifact bytes, and saved scheduler/source/environment evidence.
- [x] Documented results and every evidence-file category in the Step 4 README; preserved compact runs and logs in the repository.

## Step 5 completion checklist

- [x] Fixed LoRA settings, seeds, inference protocol, selection rules, and merge tolerances before training.
- [x] Verified response-only loss and LoRA-only gradients; preserved every frozen training image.
- [x] Completed all three epochs with W&B tracking, losses, checkpoints, and complete validation predictions.
- [x] Selected epoch 3 using validation only and preserved all candidate checkpoints.
- [x] Saved/reloaded the merged FP16 model and passed numerical, generation, full-validation, and weight-format checks.
- [x] Downloaded and verified every model artifact file; saved exact restore instructions.
- [x] Added Step 5 results, file explanations, source/environment/scheduler evidence, and this updated handoff.

## Step 6 completion checklist

- [x] Fix checkpoint, workloads, selection/quality gates, and timing boundaries before execution.
- [x] Pass all 32 tests, verify 7,200 source images and 12 model files, and reproduce eight Step 5 GPU outputs.
- [x] Complete all six A6000 sweep candidates and freeze batch 4 before final quality/timing.
- [x] Evaluate every validation image and preserve the four changed predictions.
- [x] Complete all 150 measured baseline batches and independently reproduce workloads, memory, latency, and throughput.
- [x] Verify completed GPU/publisher jobs, W&B metrics/configuration, and 153 downloaded runtime-artifact files.
- [x] Preserve code snapshots, raw timing/prediction records, logs, receipts, checksums, per-file explanations, and teammate setup.

## Immediate next work: Step 7

- [ ] Use the saved merged checkpoint, frozen batch 4, RTX A6000, and four CPU threads.
- [ ] Profile preprocessing, host-to-device transfer, vision encoding, prefill, and decode.
- [ ] Save representative traces/stage statistics and explain the dominant cost; keep profiling overhead separate from benchmark timings.
- [ ] Propose two or three techniques tied to measured bottlenecks, with expected tradeoffs and quality checks.
- [ ] Save the Step 7 README, W&B evidence, and tracker update before implementing optimizations.

## Remaining work after Step 6

| Step | What to implement or run | Why / completion gate |
| --- | --- | --- |
| 7 | Profile the five stages on the frozen batch-4 baseline; save traces and propose two or three techniques. | Explain the measured bottleneck; do not infer it from overall timing alone. |
| 8 | If included in the next team scope, implement selected changes one at a time. | Attribute effects to specific changes. Earlier pretrained quantization checks are not optimization evidence for this merged model. |
| 9 | Repeat matched timings and full validation for each variant; re-profile as needed. | Compare to the Step 6 batch-4 FP16 quality/performance baseline, retaining regressions and failed variants. |
| 10 | Freeze final settings and evaluate the unchanged test set. | Obtain held-out results without tuning against test outcomes. |
| 11 | Assemble the report, presentation, reproduction guide, and demonstration. | Connect each claim to saved measurements and explain successes and failures. |

Parallel proposal work still outstanding: fine-tune **ResNet-18** on the same splits for one accuracy-reference row. A larger SmolVLM2-2.2B comparison is optional and conditional on the proposal's midpoint criterion; a Gradio demo is optional. None of these has been implemented in this checkout.

## Teammate setup: environment and directories

Each teammate creates their own environment and downloads the same pinned inputs on their own server. The paths below describe Sean's working installation and show which values to change. The repository contains the setup instructions and package pins; installed environments and download caches are created on each server.

### Verified reference environment

These values come from the [recorded Step 3 environment](experiments/step3/runs/13924504/fp16/environment.json) and the [Step 5 runtime record](experiments/step5/runs/13927137/environment.json); the cluster layout is unchanged apart from the documented checkpoint/artifact-cache paths. The complete Python package list is [`requirements-cluster.txt`](requirements-cluster.txt).

| Component | Verified value | Purpose / teammate action |
| --- | --- | --- |
| Operating system | Linux x86_64, glibc 2.34 | Use a compatible Linux GPU server for this environment. These are not macOS installation pins. |
| Conda on Insomnia | Module `anaconda/2023.09` | Supplies Conda; another server can use its own Conda installation. |
| Python | `3.11.18` | Create an isolated environment with this version. |
| GPU | NVIDIA RTX A6000, 48 GB class | Canonical GPU for the current evaluator and later matched comparisons. |
| NVIDIA driver | `575.57.08` | Reference driver; another server needs a driver compatible with the CUDA build below. |
| PyTorch / torchvision | `2.9.1+cu126` / `0.24.1+cu126` | CUDA 12.6 wheels, pinned in the requirements file. `nvidia-smi` showing CUDA 12.9 describes driver capability, not the PyTorch build. |
| Transformers / Accelerate | `4.57.1` / `1.10.1` | Model loading, processing, generation, and device placement. |
| PEFT / bitsandbytes | `0.17.1` / `0.48.2` | PEFT was used in Step 5 LoRA training; bitsandbytes was exercised in Step 3 INT8/INT4 checks. |
| W&B | `0.31.0` | Records runs and artifacts under the shared team project. |
| Interactive allocation | Account `edu`, partition `short`, 1 A6000, 4 CPUs, 16 GiB RAM, 2 hours | Sean's setup allocation. Teammates must use resources permitted by their own account. |
| Step 3 batch allocation | Same resource request, 30-minute time limit | Defined in [`cluster/step3.sbatch`](cluster/step3.sbatch); installation/downloads happen before this job. |
| Step 4 batch allocation | Same resource request, one-hour time limit | Defined in [`cluster/step4.sbatch`](cluster/step4.sbatch); actual verified job completed in 11 minutes. |
| Step 5 batch request | 1 A6000, 4 CPUs, 32 GiB RAM, six-hour maximum | Defined in [`cluster/step5.sbatch`](cluster/step5.sbatch); Slurm actually allocated 6 CPUs. The publisher requested 1 CPU, 8 GiB RAM, and 45 minutes; Slurm allocated 2 CPUs. Actual reservations are preserved in the accounting files. |
| Step 6 batch request | 1 A6000, 4 CPUs, 16 GiB RAM, two-hour maximum | [`cluster/step6.sbatch`](cluster/step6.sbatch); job 13932242 actually used 4 CPUs / 16 GiB. Current defaults match that successful reservation. CPU publisher requests 1 CPU / 4 GiB / 45 minutes; the recorded publisher received 2 CPUs. |

The existing environment passed the recorded checks. **A fresh installation from the pins has not yet been independently tested.** A teammate should complete the verification below before treating a new installation as reproduced.

### Storage map

Sean's `MRI_WORK` is **`/insomnia001/depts/edu/users/ct3013`**. Paths beginning with `$MRI_WORK` below expand from that directory. Sean's home is `/insomnia001/home/ct3013`; it is separate from the project work root.

| Directory or file | Contents and reason | Configuration |
| --- | --- | --- |
| `$MRI_WORK/brain-mri-vlm/` | Git checkout: code, configuration, roadmap, and compact experiment evidence. Run project commands here. | Working directory |
| `$MRI_WORK/envs/smolmri/` | Installed Python interpreter and packages. Keep this outside the Git checkout. | `MRI_CONDA_ENV` |
| `$MRI_WORK/cache/conda/` | Conda package downloads/extractions used to build environments. | `CONDA_PKGS_DIRS` |
| `$MRI_WORK/cache/pip/` | pip download cache. | `PIP_CACHE_DIR` |
| `$MRI_WORK/cache/huggingface/` | Pinned SmolVLM model weights, tokenizer, and processor files. | `HF_HOME` |
| `$MRI_WORK/cache/kaggle/` | Downloaded/extracted Kaggle source images. | `KAGGLEHUB_CACHE` |
| `$MRI_WORK/cache/wandb/artifacts/` | W&B artifact cache, including downloaded checkpoint verification copies. | `WANDB_CACHE_DIR` |
| `$MRI_WORK/cache/wandb/staging/` | W&B artifact upload staging, kept out of the home directory. | `WANDB_DATA_DIR` |
| `$MRI_WORK/tmp/` | Temporary working files. | `TMPDIR` |
| `brain-mri-vlm/data/prepared-v3-conservative-patients/` | Ten files from the frozen manifest artifact, including split CSVs and audit metadata. | Evaluator's default prepared-data directory |
| `brain-mri-vlm/logs/source-root.txt` | One absolute path to the downloaded source directory containing `Training/` and `Testing/`. | Read by the batch job; `MRI_SOURCE` can override it |
| `brain-mri-vlm/logs/` | Scheduler output and local setup logs. Preserve useful completed-run logs in the step evidence directory. | Batch output path |
| `brain-mri-vlm/experiments/step3/runs/<job-id>/` | Saved predictions, metrics, environment, configuration, console output, and verification evidence for one execution. | Created by the Step 3 batch script |
| `brain-mri-vlm/experiments/step4/runs/<job-id>/` | Full validation predictions, metrics, independent verification, source snapshots, and scheduler/W&B evidence. | Created by the Step 4 batch job; publisher receipts collected afterward |
| `brain-mri-vlm/experiments/step5/runs/<job-id>/` | LoRA training history, loss/gradient audits, epoch predictions, selection, merge checks, source snapshots, and publishing receipts. | Created by Step 5 jobs |
| `brain-mri-vlm/checkpoints/step5/<job-id>/` | Epoch adapter weights, optimizer/RNG states, saved merge probe logits, and merged FP16 weights. Restored through the recorded W&B model artifact. | `MRI_CHECKPOINT_DIR` when publishing; kept outside Git |
| `brain-mri-vlm/experiments/step6/runs/<job-id>/` | Input audit, batch sweep, frozen selection, full quality, raw timing trials, GPU telemetry, and verification/publishing evidence. | Created by Step 6 jobs; `MRI_MERGED_MODEL` overrides the saved Step 5 merged-model path |
| `brain-mri-vlm/wandb/` | Local W&B working files; remote runs/artifacts provide the shared tracking record. | `WANDB_DIR` |

The current source root is `/insomnia001/depts/edu/users/ct3013/cache/kaggle/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/2`. CSV image paths are relative to this source root, not to the directory containing the CSVs. A teammate's absolute source root can differ because it is recorded separately in `logs/source-root.txt`.

These shared-filesystem directories persist when an SSH session or GPU allocation ends. The allocated compute node and its GPU are temporary. Scratch storage is not backed up, so preserve important results in the repository and W&B. Each teammate uses their own login and credentials; do not copy another person's home directory or `.netrc`.

### A. Choose your work root and allocate compute

On Insomnia, log in using your own UNI. The following is the resource request that worked for Sean; account/partition access must be checked for each teammate. Run installations, downloads, and model processing inside the resulting compute allocation.

```bash
srun --account=edu --partition=short --gres=gpu:A6000:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
  --time=02:00:00 --pty bash -l
```

Inside that allocation, set the work root to your own storage. On a different server, replace the first line with its writable persistent/scratch project path; the subdirectory layout can stay the same.

```bash
export MRI_WORK="/insomnia001/depts/edu/users/$USER"
export MRI_CONDA_ENV="$MRI_WORK/envs/smolmri"
export CONDA_PKGS_DIRS="$MRI_WORK/cache/conda"
export PIP_CACHE_DIR="$MRI_WORK/cache/pip"
export HF_HOME="$MRI_WORK/cache/huggingface"
export KAGGLEHUB_CACHE="$MRI_WORK/cache/kaggle"
export TMPDIR="$MRI_WORK/tmp"
export WANDB_CACHE_DIR="$MRI_WORK/cache/wandb/artifacts"
export WANDB_DATA_DIR="$MRI_WORK/cache/wandb/staging"
export WANDB_BASE_URL="https://api.forge.coreweave.com"
mkdir -p "$MRI_WORK/envs" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" \
  "$HF_HOME" "$KAGGLEHUB_CACHE" "$TMPDIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
```

Setting cache paths before installing keeps downloads beside the project environment. Do not use Sean's `ct3013` directory as your own work root.

### B. Create the environment once

For a new checkout/environment, run:

```bash
git clone https://github.com/oowenn/brain-mri-vlm.git "$MRI_WORK/brain-mri-vlm"
cd "$MRI_WORK/brain-mri-vlm"
module load anaconda/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --yes --override-channels --channel conda-forge \
  --prefix "$MRI_CONDA_ENV" python=3.11.18 pip
conda activate "$MRI_CONDA_ENV"
python -m pip install -r requirements-cluster.txt
python -m pip check
mkdir -p logs
export WANDB_DIR="$PWD/wandb"
mkdir -p "$WANDB_DIR"
```

If the checkout/environment already exists, enter and activate it instead of repeating `git clone` or `conda create`. On another server, initialize that server's Conda installation in place of the Insomnia module command. No administrator installation is part of this recipe. Do not replace failed version pins with newer versions silently; record the difference and rerun verification.

### C. Download the fixed data and model

Authenticate privately using your own W&B account, with access to team `coms6998-team32-fall2026`. The API host above matches the team's existing Forge-hosted W&B runs. Credentials must stay out of the repository.

```bash
wandb login
wandb artifact get \
  coms6998-team32-fall2026/brain-mri-vlm/brain-mri-manifests:v2 \
  --root data/prepared-v3-conservative-patients
python - <<'PY'
from pathlib import Path
import kagglehub
from huggingface_hub import snapshot_download

source = Path(kagglehub.dataset_download(
    "masoudnickparvar/brain-tumor-mri-dataset/versions/2"
)).resolve()
assert (source / "Training").is_dir()
assert (source / "Testing").is_dir()
Path("logs/source-root.txt").write_text(str(source) + "\n")
snapshot_download(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    revision="753c1752602b5e739af7a7fa6947a635051897a3",
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
)
print("Source root:", source)
PY
```

Use these frozen manifests rather than regenerating splits for each teammate. Model prefetch is required because the evaluator loads from the cache with `local_files_only=True`. The artifact version, Kaggle version, and model revision stay the same across servers.

### D. Verify the new setup and preserve its evidence

On Insomnia, after setup finishes, `exit` the interactive allocation to release its GPU. From the login node, submit the existing Step 3 batch job from your checkout:

```bash
export MRI_WORK="/insomnia001/depts/edu/users/$USER"
export MRI_CONDA_ENV="$MRI_WORK/envs/smolmri"
export WANDB_BASE_URL="https://api.forge.coreweave.com"
cd "$MRI_WORK/brain-mri-vlm"
mkdir -p logs
sbatch cluster/step3.sbatch
```

The batch script activates the environment and cache paths itself. It records package checks, the current unit-test suite (32 tests at Step 6), GPU execution, verification of all 7,200 source images, and FP16/INT8/INT4 inference on the same eight validation images. Outputs go into a new `experiments/step3/runs/<job-id>/` directory and W&B. Check completion and exit status, then follow the [Step 3 evidence handoff](experiments/step3/README.md#reproduce-on-insomnia) to collect scheduler records and verify/upload the bundle. The eight-image check establishes runtime readiness. To reproduce full validation after setup, use `sbatch cluster/step4.sbatch` and follow the [Step 4 evidence instructions](experiments/step4/README.md).

**Portability boundary:** the current batch scripts use Insomnia's module/account/partition settings, and `scripts/evaluate.py` explicitly requires a Slurm allocation and the GPU named in `configs/step3.json` (RTX A6000). Another Slurm server needs a site-specific batch wrapper with its own resource names and Conda initialization. A different GPU requires a separately recorded configuration and hardware label; its timing results cannot be treated as the matched A6000 comparison. A server without Slurm needs an explicit evaluator change to support its execution environment before this workflow can run there; do not invent `SLURM_JOB_ID` to bypass the guard. These adaptations have not been implemented or verified yet.

### E. Return after SSH disconnects or a time limit

Do not reinstall the environment or redownload cached inputs. For interactive model work, request a new allocation, repeat the environment-variable block in A, then run:

```bash
module load anaconda/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MRI_CONDA_ENV"
cd "$MRI_WORK/brain-mri-vlm"
export WANDB_DIR="$PWD/wandb"
```

Activation and shell exports are session state; files under `MRI_WORK` are persistent storage. Check `python scripts/check_environment.py` inside the new GPU allocation before new model work. For batch work, the submission block in D is enough once the environment, inputs, and W&B login are prepared. Keep the path map and these commands current when the setup changes.

## Limitations and operational handoff

- The tested environment is pinned, but a clean-room reinstall has not been independently repeated.
- The dataset's accepted image constraints and recovered patient IDs do not establish full patient independence. Keep the existing annotation limitations explicit.
- Step 6 establishes a measured batch-4 baseline, not an optimization speedup. No final-test model results exist yet. The three timing trials share one allocation; the training used one seed/run. These do not establish variation across training seeds, nodes, or days.
- The [teammate setup section](#teammate-setup-environment-and-directories) records the actual cluster paths, environment pins, cache variables, reproduction commands, and portability limitations.
- Recorded resource choice: account `edu`, partition `short`, RTX A6000. Recheck live job/allocation state before new work; a completed job ID is not a reusable allocation.
- Use Slurm for model work and extended processing. Batch jobs survive SSH disconnects but still have their own time limits. Cluster scratch storage is not backed up; preserve important evidence outside it.
- Keep compact code/configuration, metrics, predictions, and useful logs in GitHub and W&B. Keep MRI image files, weights, large caches/traces, and all credentials outside Git. Do not copy passwords, API keys, or temporary SSH connection details into this tracker.
- Preserve historical run files and their checksum records. When code or documentation changes, distinguish the current version from the older snapshot that produced a recorded result; create a new run directory for new executions.

## Updating this record

Update this file after a meaningful implementation change, a completed or failed experiment, a scope decision, or a handoff. A run submission is **in progress**, not complete; code that has only been written is **implemented but unverified** until the required checks pass.

Each update should refresh the date, current step, completion evidence, remaining checklist, limitations/blockers, and next concrete action. Add an entry below with the relevant commit, Slurm job, W&B run/artifact, and local evidence paths when available. Keep the per-step README as the detailed explanation of that step's files.

This record is maintained as part of project work; it is not a background monitor that observes cluster or GitHub changes automatically.

## Progress history

| Date (America/New_York) | Milestone | Evidence / handoff |
| --- | --- | --- |
| 2026-11-12 | Conservative data checkpoint finalized. | [Finalization record](experiments/data_finalization.md); manifest artifact `brain-mri-manifests:v2`. |
| 2026-11-15 | Step 3 completed and published. | Commit `0243f26`; GPU job `13924504`; publisher job `13924530`; [Step 3 evidence](experiments/step3/README.md). Next: full zero-shot validation. |
| 2026-11-15 | Created the implementation tracker and repository maintenance instructions. | Documentation-only update after the Step 3 checkpoint. No Step 4 run started. |
| 2026-11-15 | Added the teammate environment and directory handoff. | Verified the existing cluster layout and installed environment; documented per-user setup, fixed inputs, verification, and cross-server limitations. User clarified that setup documentation is required, rather than a binary environment/cache backup. No fresh reinstall or Step 4 run performed. |
| 2026-11-15 | Started Step 4 implementation. | Added full-validation configuration, stage-aware run labels, Slurm entry points, and independent coverage/metric verification. Implementation commit `492ab70`; GPU job `13925021` submitted for full validation. |
| 2026-11-15 | Step 4 completed and independently verified. | Execution commit `492ab70`; GPU job `13925021`; evidence job `13925075`; W&B inference `cpawhwk5`; checkpoint `step4-checkpoint-13925021:v0`. Accuracy 41.45%, macro-F1 0.2662, invalid rate 0. Next: Step 5 LoRA training and merge verification. |
| 2026-11-15 | Started Step 5 implementation. | Fixed three epochs, rank-16 decoder LoRA, response-only loss, effective batch 16, validation checkpoint selection, and explicit FP16 merge tolerances before training. Commit `873598e`; GPU job `13927137` submitted. |
| 2026-11-15 | Step 5 preflight passed; training started. | 25 tests; 7,200 source files; 8 unchanged base outputs; response-only loss and frozen-gradient checks. W&B `0povsppx`; 8,683,520 trainable parameters in 224 decoder LoRA modules. Full training result pending. |
| 2026-11-16 | Step 5 completed and independently verified. | Execution commit `873598e`; GPU job `13927137`; publisher `13927707`; W&B `0povsppx`; model `step5-13927137-checkpoints:v0`. Epoch 3 selected; merged accuracy 86.01%, macro-F1 0.8620; merge and artifact downloads verified. Next: Step 6 baseline benchmarking. |
| 2026-11-16 | Implemented Step 6; cluster execution pending. | Seven new benchmark tests pass locally. Fixed the sweep, selection rule, quality gate, timing boundaries, and repeated baseline protocol. Full tests, GPU measurements, and W&B evidence verification remain pending; see [Step 6](experiments/step6/README.md). |
| 2026-11-16 | Submitted Step 6 GPU benchmark. | Execution commit `9c79baa`; Slurm job `13932216`. VPN/SSH access restored. Full tests, input audit, sweep, quality, and repeated timings are pending allocation; no completion or performance claim yet. |
| 2026-11-16 | Resized the unstarted Step 6 reservation. | Job `13932216` canceled before execution: six-hour/64 GiB reservation estimated a next-day start. Slurm dry-run estimated a same-day start with two hours/32 GiB; one hour gave no further improvement. Model and benchmark protocol unchanged; scheduling evidence retained. |
| 2026-11-16 | Queued the smaller Step 6 job and added CPU preflight. | GPU job `13932242`; optional `cluster/preflight_step6.sbatch` checks tests, checkpoint hashes, and processor batching while waiting. GPU execution and performance remain pending. |
| 2026-11-16 | Queued automatic Step 6 evidence publication. | GPU job `13932242`; CPU preflight `13932248` started on `ins084`; publisher `13932262` depends on `afterok:13932242`. The publisher collects scheduler logs and verifies source/W&B artifacts. VPN/SSH disconnected before further results were collected; submitted jobs continue independently. See [submission record](experiments/step6/scheduling/submitted-jobs.json). |
| 2026-11-16 | Step 6 CPU preflight passed. | Job `13932248`, `ins084`, `0:0`, 3m15s, execution commit `393ff0b`: all 32 tests, all 12 saved model/processor file hashes, and eight-image batching audit. Local checks reproduce input metadata and launcher provenance. GPU job `13932242` and dependent publisher `13932262` remain queued. |
| 2026-11-16 | Reduced the pending Step 6 host-memory request. | `scontrol update JobId=13932242 MinMemoryNode=16384` retains the original job/queue age and all benchmark settings. Some A6000 nodes have free GPUs but less than 32 GiB unreserved host RAM; reservation is now 16 GiB and 4 CPUs. Scheduling evidence retained; no inference result yet. |
| 2026-11-16 | Step 6 GPU execution started. | Job `13932242` started on `ins094` at 13:19:34 EST with 4 CPUs/16 GiB after the memory reduction. Execution source is commit `1697704`; benchmark protocol unchanged. Full GPU results still pending. |
| 2026-11-16 | Step 6 GPU preflight passed; sweep started. | All 32 tests, 7,200 source-image hashes, FP16 CUDA execution, and eight exact Step 5 batch-one outputs passed. W&B baseline run `apjpl38l`; first sweep candidate is batch 16. Final selection, quality, and timing results still pending. |
| 2026-11-16 | Step 6 completed and independently verified. | Execution commit 1697704; GPU 13932242 (25m12s), publisher 13932262 (53s), both 0:0. Batch 4: 2.8004 images/s, 1428.367 ms mean / 1473.171 ms p95, 2.776 GiB allocated; accuracy 85.32%, F1 0.856293. W&B apjpl38l/z3njw1g9; 153 downloaded runtime files and 32 source snapshots verified. Next: Step 7 profiling. |

### 2026-11-16: Step 7 execution preparation

Added a frozen batch-4 control, synchronized stage diagnostics, asynchronous CPU/CUDA traces, exact-output/accounting verification, and W&B publishing. GPU job 13933253 started on ins090; publisher 13933254 depends on its success. Execution source is e38b35718eede86b89df4f3d2c5ad0b16010dd95. The existing environment and checkpoint are reused; no packages or model optimizations are changed. Step 7 requests one A6000, four CPUs, 16 GiB RAM, and 45 minutes. Run files will be under `$MRI_WORK/brain-mri-vlm/experiments/step7/runs/<job-id>/`. See [Step 7 protocol](experiments/step7/README.md).

Step 7 attempt 13933253 was cancelled before stage profiling because nvidia-smi rejected the unprefixed UUID returned by PyTorch. The GPU- prefix is now normalized and checked before sampling; a regression test covers both formats. This attempt is diagnostic evidence, not a completed benchmark. Its dependent publisher 13933254 was cancelled too.

Replacement GPU job 13933296 is running on ins092 from commit 563898d6e2dcac9525c9b4d306fe29c7caaf19b7. W&B run: wvqg8vuq. Dependent evidence job: 13933297. Completion and results remain pending.

GPU job 13933296 finished all control/stage/trace workers but exited 1 during final verification: PyTorch exports both CPU and mirrored GPU annotations, and the original checker counted four batch ranges instead of two. Original logs/traces and failed status are preserved. The corrected checker selects CPU user_annotation ranges; local recovery passed all 180 control and 99 instrumented output comparisons. CPU-only recovery will publish the intact measurements, with no new inference. Preprocessing is 74.17% of diagnostic time.
