# brain-mri-vlm

Profile-driven inference optimization of SmolVLM-500M for four-class brain MRI classification. COMS 6998 HPML, Fall 2026, Team 32.

Status: CPU environment and W&B verified on Insomnia. Data preparation is implemented; model/GPU baseline work is next.

## Study

Zero-shot evaluation → one LoRA fine-tune → merged FP16 baseline → profile → 2–3 targeted optimizations → matched performance and quality evaluation.

Use one fixed NVIDIA GPU type and one batch size chosen on the baseline. Measure preprocessing, transfer, vision encoding, prefill, and decode. Report mean/p50/p95 latency, throughput, peak GPU memory, macro-F1, and per-class recall. The quality allowance is 0.02 absolute macro-F1 relative to baseline. The quantization claim is a hypothesis to test, not a promised speedup.

## Reproduction status

Environment versions, exact GPU, dataset manifests, model revision, and commands will be frozen after the cluster smoke test. `requirements.in` is a dependency inventory, not a tested lockfile. Do not compare benchmark results across unpinned environments.

Keep data, weights, credentials, profiler traces, and generated run outputs outside Git; publish selected artifacts through the shared W&B project. Planning notes and reference PDFs are kept locally and are not required to run the project.

## Reproduce the data

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

## Cluster readiness

Confirm the course Slurm account, partition, and GPU type before submitting work.

`requirements.in` lists candidate dependencies. Install a compatible CUDA PyTorch/torchvision build and resolve the remaining packages in a cluster Python environment. Package pins and complete installation commands will be added after verification.

From the repository on Insomnia, with the chosen modules loaded and environment prepared:

```bash
mkdir -p logs
export SMOLMRI_VENV=/absolute/path/to/venv
sbatch --account=COURSE_ACCOUNT --partition=COURSE_PARTITION --gres=gpu:VERIFIED_TYPE:1 cluster/smoke.sbatch
```

Replace the placeholders with verified allocation values; the GPU request syntax must match the cluster. The smoke job runs `scripts/check_environment.py` to report package/GPU information and check FP16 CUDA execution. Model loading, LoRA merging, and INT8/INT4 compatibility still need separate checks.
