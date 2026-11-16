# brain-mri-vlm

Profile-driven inference optimization of SmolVLM-500M for four-class brain MRI classification. COMS 6998 HPML, Fall 2026, Team 32.

Status: **Step 7 GPU measurements and local verification are complete; W&B publishing is pending in the CPU queue.** CPU preprocessing is the main measured cost (**74.17%**), followed by vision encoding (**13.76%**). Three saved CUDA traces show **18.83% GPU activity** within batch windows. The [Step 7 README](experiments/step7/README.md) explains the evidence, preserved postprocessing failure/recovery, and three proposed techniques. The canonical [Step 6 baseline](experiments/step6/README.md) remains **2.8004 images/s, 1428.367 ms mean batch latency, 2.776 GiB peak allocated memory, and validation macro-F1 0.856293**. [IMPLEMENTATION.md](IMPLEMENTATION.md) tracks progress and teammate setup. Optimization implementation and final-test inference remain deferred.

## Study

Zero-shot evaluation → one LoRA fine-tune → merged FP16 baseline → profile → 2–3 targeted optimizations → matched performance and quality evaluation.

Use one fixed NVIDIA GPU type and one batch size chosen on the baseline. Measure preprocessing, transfer, vision encoding, prefill, and decode. Report mean/p50/p95 latency, throughput, peak GPU memory, macro-F1, and per-class recall. The quality allowance is 0.02 absolute macro-F1 relative to baseline. The quantization claim is a hypothesis to test, not a promised speedup.

## Reproduction status

The verified environment is pinned in [requirements-cluster.txt](requirements-cluster.txt). [Step 3 configuration](configs/step3.json) selects the eight-image compatibility sample; [Step 4 configuration](configs/step4.json) keeps the same model revision, GPU, prompt, parser, and generation protocol while selecting all validation images. The [Step 3 evidence](experiments/step3/runs/13924504/) and [Step 4 evidence](experiments/step4/runs/13925021/) record the actual environment, executed source, predictions, verification, and W&B links. The [Step 5 configuration](configs/step5.json) fixes the LoRA/selection/merge protocol; its [evidence](experiments/step5/runs/13927137/) also includes training losses, every epoch's predictions, merge checks, and downloadable checkpoint identities. Steps 4 and 5 measure validation quality. The [Step 6 evidence](experiments/step6/runs/13932242/) adds the verified batch sweep, frozen batch-4 timing/quality baseline, and downloaded W&B evidence. A clean-room reinstall of the dependency snapshot has not been repeated. `requirements.in` remains the broader dependency inventory.

Keep source images, weights, credentials, large profiler traces, and W&B caches outside Git. At the user's request, compact run evidence and logs are tracked under `experiments/step3/`, `experiments/step4/`, `experiments/step5/`, `experiments/step6/`, and `experiments/step7/`; runs and checkpoint artifacts are also published through the shared W&B project. Planning notes and reference PDFs are not required to run the project.

## Experiment notes

- [Step 7 profiling](experiments/step7/README.md): stage costs, compressed CUDA traces, exact-output checks, bottleneck explanation, and three proposed changes.
- [Step 6 baseline benchmark](experiments/step6/README.md): completed sweep and repeated-trial results, quality comparison, raw evidence, file explanations, and teammate launch instructions.

- [Step 5 training and merged checkpoint](experiments/step5/README.md): full epoch results, merge verification, W&B model restoration, file explanations, and reproduction.
- [Step 4 zero-shot validation](experiments/step4/README.md): the complete pretrained quality reference and independent evidence.
- [Step 3 runtime and evaluator](experiments/step3/README.md): completed checks, file-by-file explanations, reproduction commands, W&B runs, and limitations.
- [Conservative data checkpoint](experiments/data_finalization.md): finalized counts, approved annotation policy, reproducibility checks and remaining identity limitations.
- [Similarity-threshold calibration](experiments/similarity_calibration.md): patient-held-out data-quality experiment, results and decision to retain the current thresholds. Full code and generated artifacts are linked in W&B.

## Reproduce the data

Training should use **`data/prepared-v3-conservative-patients/`**, produced by the finalization step below. The version-2 image audit remains available for pair review.

Source: Masoud Nickparvar's [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), Kaggle version **2**, listed as CC BY 4.0. The source combines figshare, SARTAJ and Br35H images. Cite the dataset and retain its attribution when using it. Version 2 reports upstream deduplication; we independently audit it.

The shared protocol is [configs/data.json](configs/data.json): 64-bit pHash candidates at Hamming distance **<= 4**, confirmed automatically only when **normalized grayscale correlation >= 0.98** after resizing to 128 x 128; byte-identical or RGB-pixel-identical images also match. Seed **42**, approximately **15% validation per class**. The similarity thresholds are provisional pending pair review. These heuristics are not proof of a duplicate or shared patient; do not choose thresholds using test classification results.

From a **Slurm compute allocation**, after activating the project environment:

```bash
python -m pip install -r requirements-data.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export KAGGLEHUB_CACHE="$PWD/data/cache"
python scripts/prepare_data.py --download --output data/prepared-v2-phash4-ncc98 --workers 2
```

Use no more workers than allocated CPUs. Dataset download uses a pinned Kaggle handle; if Kaggle requires authentication, complete `kagglehub.login()` privately in your terminal. An already downloaded dataset can be used with `--source /path/containing/Training-and-Testing` instead of `--download`; the actual root must contain `Training/` and `Testing/` directories.

The script compares all images across all labels using file SHA-256, normalized RGB pixel SHA-256 and pHash plus correlation (EXIF orientation applied). Correlation uses zero-mean, unit-length grayscale vectors; unlike pHash alone, it rejects many anatomically different images with similar coarse structure. Constant images are matched only by file/pixel equality. Connected components include transitive accepted matches. It preserves **every provided test image**, excludes training images connected to any test image, quarantines conflicting-label training groups, and assigns remaining whole groups to train/validation. Duplicates within a final split are retained as a group; no source file is deleted, rewritten or resized on disk. This prevents leakage under the specified matching rule; without patient IDs it does not establish patient-disjoint evaluation.

Outputs (under ignored `data/`):

- `train.csv`, `validation.csv`, `test.csv`: relative source paths, labels, group IDs and hashes. Resolve paths against the source root printed by the script, not against the manifest directory.
- `excluded.csv`: excluded training images and reasons.
- `inventory.csv`, `duplicate_pairs.csv`: complete inventory and candidate-pair audit, including correlation, accepted/rejected decisions, label conflicts and source-split crossings.
- `summary.json`: counts, source/protocol/code checksums, dependency versions and CSV checksums.

The source fingerprint in the shared config guards against changed or incomplete copies. Outputs are immutable: use a new output directory for another run. Same source, code, config and pinned environment should produce byte-identical CSVs regardless of download directory or worker count. Team members should compare `source_inventory_sha256` and `output_sha256` in the summary. Review the audit before freezing manifests for training. Future changes require a new data version and rerunning downstream experiments.

Verified version-2 audit ([W&B run and manifest artifact](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/40ajx1cx)):

| Class | Train | Validation | Test | Excluded from original training |
| --- | ---: | ---: | ---: | ---: |
| glioma | 1,187 | 209 | 400 | 4 |
| meningioma | 1,100 | 195 | 400 | 105 |
| notumor | 684 | 121 | 400 | 595 |
| pituitary | 1,190 | 210 | 400 | 0 |
| Total | 4,161 | 735 | 1,600 | 704 |

These are provisional manifests for pair review, not final training results. The exclusions make the retained training data imbalanced; report that distribution rather than rebalancing the test set. The strict correlation guard can miss transformed copies (cropping, rotation or large contrast changes), so inspect rejected candidates as well as accepted matches before freezing. The audit keeps both, making matching-policy changes visible.

For W&B, set `WANDB_ENTITY=coms6998-team32-fall2026` and `WANDB_PROJECT=brain-mri-vlm`, then append `--log-wandb`. This uploads manifests/audits as a `brain-mri-manifests` artifact, **not images**. W&B is optional for data generation; install/authenticate it separately if needed.

For a standalone CPU job, first prepare the environment inside an allocation, then submit from the repo:

```bash
mkdir -p logs
sbatch cluster/prepare_data.sbatch
```

That script uses the verified `edu` account and `edu1` partition, activates `~/.conda/envs/coms6998`, and requires no GPU. Override `MRI_CONDA_ENV` or `MRI_DATA_OUTPUT` if needed. Within an existing Jupyter session, run the Python command in its terminal directly; do not request an additional job just to prepare data.

Verify the leakage and reproducibility behavior inside a compute allocation:

```bash
python -m unittest discover -s tests -v
```

For visual pair review, open the tracked [notebooks/data_preprocessing.ipynb](notebooks/data_preprocessing.ipynb) in JupyterLab with the **Python (coms6998)** kernel and run its cells in order. It verifies the existing audit, displays accepted/rejected pairs side by side, and provides class/conflict/borderline filters. Set `SOURCE` or `AUDIT_DIR` in the first code cell if your paths differ. Review notes save under ignored `data/reviews/` and do not change manifests. Keep the notebook's outputs cleared when committing; the notebook includes the cleanup command.

### Finalize patient constraints and annotation policy

The [approved policy](configs/data_policy.json) pins the original inventory and recovered mapping by SHA-256. The recovery used all four original [Figshare archives](https://doi.org/10.6084/m9.figshare.1512427.v8): 1,778 image correspondences identify 212 patients. Mapping requires correlation >= 0.995 at both 64 and 256 pixels and other-patient correlation < 0.98 at 64 pixels. These thresholds recover correspondence to a source slice with an existing ID; they do not infer patient identity from anatomical similarity. See the [full recovery run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/fniput3c) and the [threshold experiment](experiments/similarity_calibration.md).

Download the small, pinned mapping artifact once; teammates do not need to repeat the original archive downloads. Run inside a compute allocation with W&B authenticated:

```bash
wandb artifact get coms6998-team32-fall2026/brain-mri-vlm/brain-mri-figshare-recovery:v0 --root data/figshare-recovery/results
```

Then finalize into a new directory:

```bash
python scripts/finalize_data.py \
  --prepared data/prepared-v2-phash4-ncc98 \
  --mapping data/figshare-recovery/results/patient_mapping.csv \
  --output data/prepared-v3-conservative-patients --log-wandb
```

This stage reads manifests only; no images are copied or modified. It unions accepted image groups and confirmed patient IDs, then keeps each combined constraint group in one retained split. Training members connected to any test image/patient are excluded. It excludes all 12 training images with source-label disagreements, keeps the official test set and labels unchanged, and flags its one source-label disagreement. Remaining mixed-label training groups are quarantined after explicit annotation exclusions; the clean remainder can stay eligible. Whole eligible groups are assigned deterministically to train/validation using seed 42 and an approximately 15% validation target per class.

The 11 below-threshold source correspondences retain blank patient IDs. They are not automatically excluded just for having unknown identity; image-group constraints still apply. The other 5,411 unmapped images also retain blank IDs. No threshold-based patient inference was adopted. **This is not a fully patient-disjoint dataset:** the guarantees cover recovered IDs and accepted image groups, while unknown patients can still overlap.

| Class | Train | Validation | Test | Excluded |
| --- | ---: | ---: | ---: | ---: |
| glioma | 1,023 | 181 | 400 | 196 |
| meningioma | 890 | 157 | 400 | 353 |
| notumor | 684 | 121 | 400 | 595 |
| pituitary | 682 | 120 | 400 | 598 |
| Total | 3,279 | 579 | 1,600 | 1,742 |

Exclusion reasons are 1,730 connected to test and 12 source-label disagreements (reasons are mutually exclusive). All 7,200 original images are accounted for. No recovered patient or accepted image constraint crosses retained splits. `inventory.csv` preserves original image group IDs and adds patient status, patient key, source label and disagreement flags; `label_disagreements.csv` and `withheld_identities.csv` provide explicit review lists. `summary.json` records policy, input/code/output checksums and limitations. Every CSV is deterministic; existing output directories are never overwritten. The original audit/notebook remain unchanged.

Published checkpoint: **`brain-mri-manifests:v2`**, [W&B finalization run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/lnqg1zfw). This artifact version contains the **v3 data policy**; artifact version numbers and data-policy versions are separate. You can download the shared manifests instead of regenerating them:

```bash
wandb artifact get coms6998-team32-fall2026/brain-mri-vlm/brain-mri-manifests:v2 --root data/prepared-v3-conservative-patients
```

`cluster/finalize_data.sbatch` runs this metadata-only stage in a separate small CPU allocation if no interactive allocation is active. It expects the audit and mapping to exist; override `MRI_AUDIT_DIR`, `MRI_MAPPING_PATH` and `MRI_DATA_OUTPUT` as needed. `cluster/prepare_data.sbatch` still produces the initial image audit. All data and generated manifests remain ignored by Git; W&B stores the shared artifacts.

## Cluster readiness

The verified Step 3 setup uses Slurm account `edu`, partition `short`, one RTX A6000, Python 3.11.18, PyTorch 2.9.1+cu126, Transformers 4.57.1, and bitsandbytes 0.48.2. The complete [Step 3 instructions](experiments/step3/README.md) cover dependencies, cached inputs, W&B authentication, and all saved evidence.

From the repository on the login node, after preparing the environment and inputs:

```bash
mkdir -p logs
sbatch cluster/step3.sbatch
```

This verifies the source files, runs the tests, and evaluates the fixed eight-image sample in FP16, INT8, and INT4. It does not run training, full validation, final testing, or performance benchmarks. LoRA checkpoint merging remains part of Step 5.

The earlier CUDA-only smoke script is retained for a lightweight environment diagnostic using a Python venv:

From the repository on Insomnia, with the chosen modules loaded and environment prepared:

```bash
mkdir -p logs
export SMOLMRI_VENV=/absolute/path/to/venv
sbatch --account=COURSE_ACCOUNT --partition=COURSE_PARTITION --gres=gpu:VERIFIED_TYPE:1 cluster/smoke.sbatch
```

Replace the placeholders with verified allocation values; the GPU request syntax must match the cluster. This older smoke job runs `scripts/check_environment.py` to report package/GPU information and check FP16 CUDA execution. The full Step 3 job adds model inference and quantization compatibility checks.
