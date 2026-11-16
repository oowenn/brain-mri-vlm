# Implementation status and handoff

Last updated: **2026-11-15 (America/New_York)**

Latest verified implementation commit: [`492ab709cdf48d8416f6087050860b6254b8b5e8`](https://github.com/oowenn/brain-mri-vlm/commit/492ab709cdf48d8416f6087050860b6254b8b5e8) (Step 4 execution source; result documentation is a subsequent commit).

## Current position

**Step 4 is complete. Step 5 is being implemented; no completed training result is claimed yet.** The project has frozen data, a verified GPU environment, an audited evaluator, and a full pretrained FP16 validation result: **240/579 correct (41.45% accuracy), macro-F1 0.2662, and 0 invalid responses**. It does not yet have a fine-tuned model or performance/profile results.

This is the living implementation record. It follows the numbering in the [project roadmap](output/pdf/brain-mri-vlm-roadmap.pdf). The [Step 3 README](experiments/step3/README.md) and [Step 4 README](experiments/step4/README.md) contain detailed file inventories and reproduction commands; historical run files remain the evidence for what actually happened.

The latest team discussion proposes working through profiling and selecting two or three optimizations, with implementation of those optimizations deferred for now. The full proposal roadmap is retained below, with that distinction explicit. The user has authorized Step 5 with the same evidence, W&B, repository, and teammate-documentation requirements. Step 6 benchmarking remains subsequent work.

## Roadmap status

| Step | Work | Status | Completion evidence or required output |
| --- | --- | --- | --- |
| 1 | Define the experiment and comparison rules | Defined | Primary model, four labels, quality/performance metrics, matched GPU/batch rules, and 0.02 absolute macro-F1 allowance are documented. The eventual benchmark batch size is still to be selected in Step 6. |
| 2 | Audit data and freeze splits | Complete | Conservative manifests, source and output hashes, recovered patient constraints, and documented identity limitations. |
| 3 | Verify GPU, model, and evaluator | Complete | All 7,200 source images verified; 14 tests passed; FP16, INT8, and INT4 completed the same eight-image validation sample; evidence saved in GitHub and W&B. |
| 4 | Run zero-shot evaluation | Complete | All 579 validation images; accuracy 0.414508, macro-F1 0.266220, invalid rate 0; independently verified predictions and W&B artifacts. |
| 5 | Fine-tune once with LoRA and merge into FP16 | In progress | Reproducible training, selected adapter checkpoint, merged FP16 weights, and merge-equivalence checks. |
| 6 | Select batch size and measure the FP16 baseline | Not started | Batch-size sweep, selected fixed batch size, repeated timing results, throughput, peak memory, and validation quality. |
| 7 | Profile the five inference stages | Not started; within proposed near-term team scope | Stage breakdown, representative traces, bottleneck explanation, and evidence-based proposals for two or three techniques. |
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
| Tests | [`tests/`](tests/) | Ten data tests, four evaluator tests, and five Step 4 verification tests passed (19 total). |
| Evidence documentation | [`experiments/step3/README.md`](experiments/step3/README.md), [`experiments/step4/README.md`](experiments/step4/README.md) | Explain step outputs, reproduction commands, results, and limitations. |
| Living progress record | [`IMPLEMENTATION.md`](IMPLEMENTATION.md), [`AGENTS.md`](AGENTS.md) | Maintain the current implementation status and carry the update instruction into later repository sessions. |

## Step 4 completion checklist

- [x] Added Step 4 configuration, tracking names, GPU batch job, and CPU evidence publisher.
- [x] Preserved the frozen inputs and accepted inference protocol.
- [x] Executed full FP16 validation and retained every raw answer and prediction.
- [x] Verified 579 unique images with supports 181/157/121/120, with no dropped images or final-test predictions.
- [x] Independently recomputed accuracy, macro-F1, class recalls, the confusion matrix, and invalid-response rate.
- [x] Verified finished W&B runs, matched remote metrics, compared downloaded inference artifact bytes, and saved scheduler/source/environment evidence.
- [x] Documented results and every evidence-file category in the Step 4 README; preserved compact runs and logs in the repository.

## Immediate next work: Step 5

The goal is **one reproducible LoRA training workflow and a verified merged FP16 checkpoint**. The Step 4 pretrained model remains the zero-shot reference; later optimization comparisons use the merged fine-tuned model.

- [ ] Define the LoRA target modules, training configuration, seed, and validation/checkpoint-selection rule before training. Preserve the frozen 3,279-image training split and 579-image validation split.
- [ ] Implement training input/label construction and verify that loss is applied to the intended target tokens. Keep validation generation aligned with the accepted inference protocol; record any deliberate change.
- [ ] Run training on the allocated GPU with W&B tracking, retaining configuration, losses, validation metrics, failures, and checkpoint identities.
- [ ] Select the adapter checkpoint using validation only; save it as a reproducible artifact.
- [ ] Merge the adapter into FP16 and check adapter-loaded versus merged behavior with an explicit numerical/generation comparison and recorded tolerances.
- [ ] Evaluate and preserve the selected checkpoint's validation quality. Do not infer improvement from successful training alone or use the test set for selection.
- [ ] Add the Step 5 README, evidence, artifact links, and actual outcome to this tracker before advancing to baseline timing.

Completion gate: a reproducible trained adapter, selected with validation evidence, and a verified merged FP16 model. Step 5 implementation is in progress; the full training run is pending.

## Remaining work after Step 4

| Step | What to implement or run | Why / completion gate |
| --- | --- | --- |
| 5 | Implement one LoRA training workflow on 3,279 training images; track validation, select a checkpoint, merge into FP16, and check merged versus adapter-loaded behavior. | Produces task-adapted weights that remain fixed during later optimization comparisons. Save training settings, checkpoint identities, loss/quality records, and merge checks. |
| 6 | Build the timing harness, sweep batch sizes on the merged model, choose the baseline batch size, then measure repeated latency/throughput/memory and quality. | Establishes a fair measured reference. Define request/batch/amortized latency, use correct CUDA synchronization, and apply the proposal's warm-up and repeat protocol. |
| 7 | Instrument preprocessing, host-to-device transfer, vision encoding, prefill, and decode; collect representative traces and utilization evidence. | Explain the dominant cost and propose two or three changes tied to that evidence. Collect benchmark timings separately from profiler overhead. |
| 8 | If included in the next team scope, implement the selected changes one at a time on the same merged checkpoint. | Makes the effects attributable to specific changes. Early pretrained INT8/INT4 compatibility does not validate later merged-model quantization or an optimization claim. |
| 9 | Repeat matched measurements and validation-quality evaluation for each implemented variant; re-profile when needed. | Test both improvements and regressions, retaining failed variants. Apply the 0.02 absolute macro-F1 allowance to the appropriate baseline. |
| 10 | Freeze chosen settings and evaluate the unchanged test set, including the planned quality-reference rows. | Obtain the final held-out comparison without tuning against test outcomes. |
| 11 | Assemble the report, presentation, reproduction guide, and demonstration around the measured results. | Explain why changes helped or failed, and connect claims to saved evidence. |

Parallel proposal work still outstanding: fine-tune **ResNet-18** on the same splits for one accuracy-reference row. A larger SmolVLM2-2.2B comparison is optional and conditional on the proposal's midpoint criterion; a Gradio demo is optional. None of these has been implemented in this checkout.

## Teammate setup: environment and directories

Each teammate creates their own environment and downloads the same pinned inputs on their own server. The paths below describe Sean's working installation and show which values to change. The repository contains the setup instructions and package pins; installed environments and download caches are created on each server.

### Verified reference environment

These values come from the [recorded Step 3 environment](experiments/step3/runs/13924504/fp16/environment.json) and the cluster layout checked on November 15. The complete Python package list is [`requirements-cluster.txt`](requirements-cluster.txt).

| Component | Verified value | Purpose / teammate action |
| --- | --- | --- |
| Operating system | Linux x86_64, glibc 2.34 | Use a compatible Linux GPU server for this environment. These are not macOS installation pins. |
| Conda on Insomnia | Module `anaconda/2023.09` | Supplies Conda; another server can use its own Conda installation. |
| Python | `3.11.18` | Create an isolated environment with this version. |
| GPU | NVIDIA RTX A6000, 48 GB class | Canonical GPU for the current evaluator and later matched comparisons. |
| NVIDIA driver | `575.57.08` | Reference driver; another server needs a driver compatible with the CUDA build below. |
| PyTorch / torchvision | `2.9.1+cu126` / `0.24.1+cu126` | CUDA 12.6 wheels, pinned in the requirements file. `nvidia-smi` showing CUDA 12.9 describes driver capability, not the PyTorch build. |
| Transformers / Accelerate | `4.57.1` / `1.10.1` | Model loading, processing, generation, and device placement. |
| PEFT / bitsandbytes | `0.17.1` / `0.48.2` | PEFT is installed for future LoRA work; bitsandbytes was exercised in Step 3 INT8/INT4 checks. |
| W&B | `0.31.0` | Records runs and artifacts under the shared team project. |
| Interactive allocation | Account `edu`, partition `short`, 1 A6000, 4 CPUs, 16 GiB RAM, 2 hours | Sean's setup allocation. Teammates must use resources permitted by their own account. |
| Step 3 batch allocation | Same resource request, 30-minute time limit | Defined in [`cluster/step3.sbatch`](cluster/step3.sbatch); installation/downloads happen before this job. |
| Step 4 batch allocation | Same resource request, one-hour time limit | Defined in [`cluster/step4.sbatch`](cluster/step4.sbatch); actual verified job completed in 11 minutes. |

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
| `$MRI_WORK/tmp/` | Temporary working files. | `TMPDIR` |
| `brain-mri-vlm/data/prepared-v3-conservative-patients/` | Ten files from the frozen manifest artifact, including split CSVs and audit metadata. | Evaluator's default prepared-data directory |
| `brain-mri-vlm/logs/source-root.txt` | One absolute path to the downloaded source directory containing `Training/` and `Testing/`. | Read by the batch job; `MRI_SOURCE` can override it |
| `brain-mri-vlm/logs/` | Scheduler output and local setup logs. Preserve useful completed-run logs in the step evidence directory. | Batch output path |
| `brain-mri-vlm/experiments/step3/runs/<job-id>/` | Saved predictions, metrics, environment, configuration, console output, and verification evidence for one execution. | Created by the Step 3 batch script |
| `brain-mri-vlm/experiments/step4/runs/<job-id>/` | Full validation predictions, metrics, independent verification, source snapshots, and scheduler/W&B evidence. | Created by the Step 4 batch job; publisher receipts collected afterward |
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
export WANDB_BASE_URL="https://api.forge.coreweave.com"
mkdir -p "$MRI_WORK/envs" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" \
  "$HF_HOME" "$KAGGLEHUB_CACHE" "$TMPDIR"
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

The batch script activates the environment and cache paths itself. It records package checks, the current unit-test suite (19 tests at Step 4), GPU execution, verification of all 7,200 source images, and FP16/INT8/INT4 inference on the same eight validation images. Outputs go into a new `experiments/step3/runs/<job-id>/` directory and W&B. Check completion and exit status, then follow the [Step 3 evidence handoff](experiments/step3/README.md#reproduce-on-insomnia) to collect scheduler records and verify/upload the bundle. The eight-image check establishes runtime readiness. To reproduce full validation after setup, use `sbatch cluster/step4.sbatch` and follow the [Step 4 evidence instructions](experiments/step4/README.md).

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
- There are no measured optimization speedups, no benchmark batch-size selection, no LoRA checkpoint/merge validation, and no final-test model results yet.
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
| 2026-11-15 | Started Step 5 implementation. | Fixed three epochs, rank-16 decoder LoRA, response-only loss, effective batch 16, validation checkpoint selection, and explicit FP16 merge tolerances before training. |
