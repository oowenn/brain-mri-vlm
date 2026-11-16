# Step 3: verify the GPU, model, and evaluator

This implements Step 3 of [the project roadmap](../../output/pdf/brain-mri-vlm-roadmap.pdf): establish a working runtime and evaluator, and check quantization support early. The output is a pinned environment and an evaluator that works on a small validation sample.

The scope is eight fixed validation images (two per class), with FP16 inference and INT8/INT4 compatibility checks on the same GPU type. Full zero-shot validation is Step 4. LoRA training and merging are Step 5; performance benchmarking, profiling, and optimization selection come later. These smoke results are too small to estimate classification quality, and they do not measure an optimization speedup or the proposal's memory/latency hypothesis.

## Completion evidence

**Completed November 15, 2026 (America/New_York).** Slurm job **13924504** finished on `ins093` with state `COMPLETED`, exit code `0:0`, and elapsed allocation time **7 minutes 31 seconds**. This elapsed time includes environment imports, data verification, W&B uploads, and three model loads; it is not inference latency.

| Check | Recorded outcome |
| --- | --- |
| Dependencies | `pip check`: no broken requirements; Python 3.11.18, PyTorch 2.9.1+cu126, torchvision 0.24.1+cu126, Transformers 4.57.1, bitsandbytes 0.48.2. |
| GPU execution | RTX A6000, CUDA available, FP16 matrix calculation passed. |
| Source and splits | All 7,200 source images match; train 3,279, validation 579, test 1,600, excluded 1,742. |
| Tests | All 14 tests passed, including four evaluator tests. |
| Evaluator | Eight images, two per class, same sample and prompt across all precisions. Raw answers, parsed labels, class metrics, and confusion matrices saved. |
| Quantization | INT8 and INT4 each replaced 297 linear modules and completed generation with all model parameters on `cuda:0`. |
| Provenance | Saved code hashes match the published source; selected images match across modes; saved scores were recomputed from raw predictions. |

| Mode | Runtime check | Valid responses | Correct | Sample macro-F1 | W&B run |
| --- | --- | --- | --- | --- | --- |
| FP16 | Passed | 8/8 | 3/8 | 0.2778 | [7b74ixt5](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/7b74ixt5) |
| INT8 | Passed | 8/8 | 3/8 | 0.2778 | [3n3x3b29](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/3n3x3b29) |
| INT4 (NF4) | Passed | 8/8 | 0/8 | 0.0000 | [kgqava8s](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/kgqava8s) |

FP16 and INT8 produced the same eight answers. Their recalls were glioma 1.0, meningioma 0.0, notumor 0.5, and pituitary 0.0. INT4 changed five answers and had no correct predictions on this sample. This establishes that the quantized execution paths work, while showing that INT4 must not be assumed to preserve quality. Eight images cannot establish full-validation performance or the quality allowance; these checks do not select an optimization.

Each linked W&B run records metrics and a raw-prediction table and uploads its files as a `runtime-evidence` artifact. Every run directory retains its raw predictions, environment, configuration, and scores. The machine-readable overview is [`summary.json`](summary.json).

The CPU-only publishing job **13924530** independently verified the finished W&B runs, matched the remote metrics to the local files, and uploaded **55 artifact files** as `step3-checkpoint-13924504:v0`. The [complete evidence-bundle run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/6bdx7var) links that artifact. Its digest and verification receipt are in [`runs/13924504/evidence-upload.json`](runs/13924504/evidence-upload.json). Publisher logs, receipts, and this final documentation update were produced after the bundle upload and are additionally preserved in GitHub.

## Files and why they exist

| File | What it does and why |
| --- | --- |
| [`../../configs/step3.json`](../../configs/step3.json) | Freezes the model revision, labels, GPU type, prompt, seed, sample size, generation settings, parser version, and W&B destination so the same check can be repeated. |
| [`../../scripts/verify_data.py`](../../scripts/verify_data.py) | Checks the pinned inventory, policy, split CSVs, accepted identity constraints, and the bytes of all 7,200 source images. It verifies the existing checkpoint without generating new splits. |
| [`../../scripts/evaluate.py`](../../scripts/evaluate.py) | Loads the validation manifest, selects a deterministic balanced sample, checks image hashes, applies EXIF orientation and RGB conversion, calls the pinned model processor, generates answers, parses them, and computes quality metrics. Optional quantized loading exercises the actual CUDA path. |
| [`../../scripts/publish_step3.py`](../../scripts/publish_step3.py) | Recomputes local results, verifies all three finished W&B runs and their uploaded artifacts through the API, and archives a complete evidence bundle including source, tests, data verification, and logs. Run inside a small compute allocation with W&B authenticated. |
| [`../../tests/test_evaluate.py`](../../tests/test_evaluate.py) | Tests the punctuation correction, rejects ambiguous prose, checks manually calculated metrics including invalid outputs, and verifies stable balanced sampling. These tests run without a GPU. |
| [`../../cluster/step3.sbatch`](../../cluster/step3.sbatch) | Requests one RTX A6000, activates the existing Conda environment, checks dependencies and CUDA, verifies the data, and runs all three precision checks with W&B tracking. A batch job can continue after SSH disconnects, until its own time limit. |
| [`../../cluster/publish_step3.sbatch`](../../cluster/publish_step3.sbatch) | Uses one CPU core and no GPU to verify finished W&B records and upload the complete evidence bundle after the GPU job. |
| [`../../requirements-cluster.txt`](../../requirements-cluster.txt) | Pins all installed package versions from the checked GPU environment, including PyTorch's CUDA 12.6 build. The original `packaging @ file://...` build reference is represented as the installed `packaging==26.3` version. |
| [`evidence/prior/`](evidence/prior/) | Preserves earlier interactive JSON evidence: synthetic blue-image inference, complete data verification, and the original and corrected eight-image MRI smoke results. These were saved before the interactive allocation expired. |
| [`evidence/parser-recheck.json`](evidence/parser-recheck.json) | Reprocesses the original eight raw answers using the reusable evaluator's parser and scorer. It records source/code hashes and the before/after validity counts without rerunning inference. |
| `runs/<job-id>/console.log` | Captures the verification commands' output, warnings, test results, prediction lines, and W&B links. |
| `runs/<job-id>/base-commit.txt` | Records the repository commit before the new scripts were published. Each model run also hashes the executed code, avoiding a false claim that an older commit already contains these scripts. |
| `runs/<job-id>/pip-freeze-original.txt` | Preserves the environment's original pip snapshot, including any machine-specific package references. |
| `runs/<job-id>/requirements-installed.txt` | Records installed package names and versions without machine-specific build URLs. This is the source for the portable dependency pins. |
| `runs/<job-id>/pip-check.txt` | Records dependency consistency validation. |
| `runs/<job-id>/tests.txt` | Records the existing data tests and the new evaluator tests. |
| `runs/<job-id>/cuda-check.txt` | Records package availability, GPU properties, and the FP16 matrix computation check. Model execution is checked separately by the three evaluator runs. |
| `runs/<job-id>/data-verification.json` | Records the full source/manifest checksums, split counts, and remaining unknown-patient limitation. |
| `runs/<job-id>/exit-status.tsv` | Records the exit code for each precision, including failures rather than silently dropping them. |
| `runs/<job-id>/slurm-accounting.tsv` | Records Slurm's final states, exit codes, allocated CPUs, memory request, node, and start/end times. The zero-duration `.0`-`.2` steps were read-only process/GPU progress checks. |
| `runs/<job-id>/slurm-output.log` | Preserves the scheduler's original stdout/stderr file alongside the script's console capture. |
| `runs/<job-id>/wandb-verification.json` | Records API confirmation that each model run finished, its remote scalar metrics match local results, and its evidence artifact exists. |
| `runs/<job-id>/evidence-upload.json` | Records the completed evidence-bundle run and artifact digest. This receipt is written after upload, so it is stored in GitHub rather than inside its own artifact. |
| `runs/<job-id>/publisher-accounting.tsv`, `publisher-output.log` | Preserve the separate CPU publisher job's final Slurm state and complete output, including W&B's confirmation of uploaded files. |
| [`summary.json`](summary.json) | Provides a compact machine-readable result table and the checks performed on the downloaded evidence. |
| `checksums.sha256` | Pins the final published scripts, configuration, documentation, and evidence files. From the repository root, use `sha256sum -c experiments/step3/checksums.sha256` on Linux or `shasum -a 256 -c experiments/step3/checksums.sha256` on macOS. |
| `runs/<job-id>/<precision>/run.json` | Records the run status, exact protocol, code hashes, validation-manifest hash, timestamps, and W&B run identity. A failed run also records its traceback. |
| `runs/<job-id>/<precision>/environment.json` | Records Python/packages, CUDA build, driver, GPU, host, and Slurm allocation. |
| `runs/<job-id>/<precision>/selected-images.json` | Identifies exactly which eight images were evaluated, including their labels, file hashes, and group IDs. |
| `runs/<job-id>/<precision>/model-runtime.json` | Records actual model class, parameter devices, quantization settings, and the names/types of replaced quantized layers. Importing bitsandbytes alone is insufficient proof of compatibility. |
| `runs/<job-id>/<precision>/model-config.json` | Records the resolved model architecture/configuration without weights. |
| `runs/<job-id>/<precision>/image-processor.json` | Records the processor's actual resizing, normalization, and crop settings. The data audit's 128-pixel similarity representation is not the model input. |
| `runs/<job-id>/<precision>/rendered-prompt.txt` | Saves the full templated prompt. Ground-truth labels and image filenames are never inserted into the model prompt. |
| `runs/<job-id>/<precision>/generation-config.json` | Saves upstream generation defaults; explicit overrides are in `run.json` and `configs/step3.json`. |
| `runs/<job-id>/<precision>/predictions.jsonl` | Saves each original answer, parsed label, truth, generated token IDs, image size, input-token count, and processed image shape. Rows are flushed as they finish, retaining partial evidence after interruption. |
| `runs/<job-id>/<precision>/metrics.json` | Saves macro-F1, accuracy, per-class precision/recall/F1/support, invalid-output rate, and a confusion matrix with an explicit invalid column. |

## Parsing and scoring contract

The first interactive parser classified `Glioma.` as invalid because it expected the bare label. The corrected parser lowercases, strips surrounding whitespace, and accepts one optional final period. It accepts `Glioma.` as `glioma`; it rejects explanations, multiple labels, negations, and unknown strings. The original raw answers are preserved. The earlier eight-image result changes from 0/8 valid responses to 8/8 valid responses, with 3/8 correct; no inference was rerun to make that correction.

Invalid outputs remain in the denominator. They count as false negatives for their true class. Macro-F1 averages all four named classes, with zero for undefined class scores. Confusion-matrix rows are true classes; columns are the four predictions plus `INVALID`. The sample includes two images per class but does not imply two independent patients per class.

## Reproduce on Insomnia

Use the already prepared Python environment and cached pinned model. The source directory must contain `Training/` and `Testing/`; the frozen manifests must be under `data/prepared-v3-conservative-patients/`.

The verified interpreter is Python 3.11.18. For a new environment inside a compute allocation, load `anaconda/2023.09`, create/activate a Conda environment with `python=3.11.18 pip`, then run `python -m pip install -r requirements-cluster.txt` and `python -m pip check`. The pins describe the environment actually used for this checkpoint; a fresh installation from scratch has not been independently repeated. Keep the original tested environment available while reproducing it.

If the inputs are not cached yet, prepare them inside a compute allocation with the environment activated:

```bash
export MRI_WORK=/insomnia001/depts/edu/users/"$USER"
export HF_HOME="$MRI_WORK/cache/huggingface"
export KAGGLEHUB_CACHE="$MRI_WORK/cache/kaggle"
mkdir -p logs
wandb artifact get coms6998-team32-fall2026/brain-mri-vlm/brain-mri-manifests:v2 \
  --root data/prepared-v3-conservative-patients
python - <<'PY'
from pathlib import Path
import kagglehub
from huggingface_hub import snapshot_download

source = kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset/versions/2')
Path('logs/source-root.txt').write_text(str(Path(source).resolve()) + '\n')
snapshot_download('HuggingFaceTB/SmolVLM-500M-Instruct',
                  revision='753c1752602b5e739af7a7fa6947a635051897a3',
                  allow_patterns=['*.json', '*.txt', '*.safetensors'])
PY
```

The evaluator uses `local_files_only=True` so a later run cannot silently fetch a different revision or depend on an unexpected model download.

From the repository on the login node:

```bash
mkdir -p logs
export MRI_WORK=/insomnia001/depts/edu/users/"$USER"
export MRI_CONDA_ENV="$MRI_WORK/envs/smolmri"
# Optional if logs/source-root.txt already exists:
export MRI_SOURCE="$MRI_WORK/cache/kaggle/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/2"
sbatch cluster/step3.sbatch
```

The job requests account `edu`, partition `short`, one `gpu:A6000:1`, four CPU cores, 16 GiB RAM, and 30 minutes. Overrides can be passed to `sbatch` if required, while retaining the fixed GPU type. W&B authentication must already exist for the shared team project; enter credentials privately with `wandb login` when preparing the environment, never in a tracked file.

For a single check inside an existing GPU allocation, with the environment and cache paths activated:

```bash
python scripts/evaluate.py --source "$MRI_SOURCE" --precision fp16 \
  --output experiments/step3/runs/manual-fp16 --log-wandb
```

Use a new output directory each time. `--precision int8` selects bitsandbytes LLM.int8; `--precision int4` selects NF4 with nested quantization and FP16 compute. Both require actual quantized linear modules and CUDA-resident parameters. None of these smoke modes uses a fine-tuned checkpoint.

After collecting the completed run's scheduler logs and summary, archive and independently verify the W&B evidence:

```bash
export MRI_RUN_DIR=experiments/step3/runs/13924504
sbatch cluster/publish_step3.sbatch
```

Replace the job directory for a new run. The publisher checks the recorded code hashes and same-image requirement before uploading. Its final receipt and API verification are saved back into that run directory.

## Evidence storage and handoff

At the user's request, this directory tracks compact run evidence and logs in GitHub, in addition to W&B. Source MRI images, model weights, credentials, and the W&B local cache remain outside Git. W&B runs use the frozen data artifact as an input and upload the corresponding run directory as a `runtime-evidence` artifact.

After Step 3 is complete, Step 4 can use this evaluator on all 579 validation images with `--per-class 0`. Do not use the test set to tune the prompt or parser. The latest team direction is to work toward profiling and selecting two or three optimizations; implementing those optimizations is a later decision. This checkpoint completes only Step 3.

Quantization compatibility follows the [Transformers 4.57.1 bitsandbytes API](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes). Compatibility on eight samples does not guarantee quality, speed, or compatibility of the later merged LoRA checkpoint.
