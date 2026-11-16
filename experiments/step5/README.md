# Step 5: LoRA training and FP16 merge

Status: **complete and verified**. GPU job `13927137` and CPU publishing job `13927707` completed with exit `0:0`. Execution commit: `873598e3a0f1fe8cecb3b2da5bda5962f63de7fa`. [W&B training run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/0povsppx); [W&B evidence run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/qi5s4b7s).

## Results

| Checkpoint | Training loss | Validation loss | Validation accuracy | Macro-F1 | Invalid answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step 4 pretrained FP16 | — | Not measured | 41.45% | 0.266220 | 0/579 |
| Epoch 1 | 0.610942 | 0.149584 | 80.83% | 0.815385 | 0/579 |
| Epoch 2 | 0.113512 | 0.131799 | 83.25% | 0.833350 | 0/579 |
| Epoch 3 (selected) | 0.071642 | 0.107852 | 85.49% | 0.857560 | 0/579 |
| Saved/reloaded merged FP16 | — | 0.107863 | 86.01% | 0.862019 | 0/579 |

The merged model classified **498/579** images correctly. Relative to Step 4, accuracy increased by **44.56 percentage points** and macro-F1 by **0.595798**. Epoch 3 was selected by the rule fixed before training. These validation results were used for selection; they are not final test results or an estimate of training-seed variability. Exact checkpoint restoration is verified; bitwise-identical retraining has not been established.

Rows are true labels; columns are predicted labels:

| True label | glioma | meningioma | notumor | pituitary | INVALID | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| glioma | 159 | 16 | 0 | 6 | 0 | 0.878453 |
| meningioma | 19 | 120 | 3 | 15 | 0 | 0.764331 |
| notumor | 3 | 2 | 116 | 0 | 0 | 0.958678 |
| pituitary | 2 | 7 | 8 | 103 | 0 | 0.858333 |

### Verification and saved evidence

- All **25 tests**, all **7,200 source-image hashes**, response-only loss/gradient audits, and the eight unchanged pretraining reference outputs passed.
- All **615 optimizer updates** are recorded. Each of the three epochs used every one of the **3,279 training images exactly once**; all four full validation passes cover the same **579 images**.
- Merge checks passed: **8/8 identical probe token sequences**, maximum absolute next-token logit difference **0.12109375**, mean absolute difference **0.01579220**. Full-validation label agreement: **99.4819%**; absolute macro-F1 difference: **0.00445901**.
- The merge changed **3 of 579 labels**; [the exact differences](runs/13927137/merge-prediction-differences.json) are retained. Image identities, input token counts, and preprocessing metadata match across all 579 pairs. The full-validation changes are included in the reported agreement and F1 checks.
- The selected adapter and merged model were both reloaded from disk. The merged checkpoint contains FP16 tensors and no remaining LoRA modules.
- All **25 source snapshots** match the execution commit. Independent checks reproduce validation metrics, image coverage, training order, checkpoint selection, saved-logit differences, and checkpoint hashes.
- Downloaded and verified **85 training-evidence files** and all **52 checkpoint files (1,350,270,604 bytes)** from W&B. Model artifact: `step5-13927137-checkpoints:v0`; complete compact evidence artifact: `step5-checkpoint-13927137:v0`.
- GPU job elapsed time: **01:50:40**; CPU publishing job: **00:01:54**. These include setup, validation, verification, and uploads; they are **not inference benchmarks**.
- The GPU job requested 4 CPUs and Slurm allocated 6; the publisher requested 1 CPU and Slurm allocated 2. The saved accounting files record the actual reservations; training used the requested 4 CPU threads.

See [summary.json](summary.json) for exact values, [checksums.sha256](checksums.sha256) for the repository evidence inventory, and the run files below for raw evidence. Unknown patient identities still prevent a fully patient-disjoint guarantee. Step 6 timing, Step 7 profiling, and final-test inference have not run.

## Fixed protocol

The fixed protocol is in `configs/step5.json`: one three-epoch run, rank-16/alpha-32 LoRA on all 224 language-decoder attention/MLP projection modules; original weights, vision encoder, and connector frozen. Verified trainable count: **8,683,520 of 516,165,824 total parameters**. Microbatch 4 with 4 accumulation steps gives effective batch 16 (last update includes all 15 remaining images). AdamW learning rate 0.0001, weight decay 0.01, cosine decay, 5% warm-up, gradient clipping 1.0, LoRA dropout 0.05. FP16 model/compute and FP32 trainable adapters use gradient scaling. Training supervises only the exact category completion and end-of-answer token, with prompt, image, and padding positions ignored. No augmentation, resizing-policy change, or truncation is introduced.

Every epoch uses all 3,279 frozen training images exactly once, shuffled by `random.Random(42 + epoch)`. Accumulated gradients are weighted by supervised token counts, including the final incomplete batch. The same 579 validation images are used for generation metrics and token-weighted validation loss. Select highest macro-F1, then lowest validation loss, then earliest epoch. No final-test inference is allowed.

Before the first optimizer update, reproduce the eight original FP16 reference outputs, verify canonical chat token boundaries for every class, compare native loss with manually masked cross entropy, and verify that gradients reach only LoRA parameters. Epoch adapters and optimizer/scheduler/scaler/RNG states are saved outside Git. These are recovery records; automated resume is not implemented.

Reload the selected adapter, merge in FP32 with PEFT safe merging, cast to FP16, save, and reload the merged model. Predeclared checks require all 8 probe token sequences to match; maximum absolute next-token logit error <= 0.15 and mean absolute error <= 0.02; >= 99% label agreement across all validation images and absolute macro-F1 difference <= 0.005. Require no remaining LoRA modules and all model parameters in FP16. These are numerical merge tolerances, not the later optimization quality allowance.

## Why these stages are required

| Stage | What it does and why |
| --- | --- |
| Verify inputs and the unchanged base model | Hashes the frozen data and reproduces the original eight FP16 answers, so later changes can be attributed to training. |
| Audit labels, loss, and gradients | Confirms that only category/end-of-answer tokens contribute to loss and only decoder LoRA weights learn. This catches a training run that executes successfully but optimizes the wrong target. |
| Train once and retain every epoch | Uses the predeclared settings and every training image, preserving losses and checkpoint identities without a hyperparameter search. |
| Select using full validation | Macro-F1 gives each of the four classes equal weight. Validation loss breaks ties; the test set remains reserved for the final evaluation. |
| Merge, save, and reload | Produces a standalone FP16 model for later comparisons and checks the actual saved weights, including numerical and generated-answer agreement. |
| Independently verify and back up | Recomputes scores and coverage, then downloads the W&B artifacts and checks their hashes. Teammates can restore the same model even if cluster scratch storage is lost. |

## Files

| File | Purpose |
| --- | --- |
| `configs/step5.json` | Frozen training, inference, selection, and merge-check settings. |
| `scripts/lora_utils.py` | Testable response-only label masks, epoch ordering, checkpoint selection, and merge gates. |
| `scripts/train_lora.py` | Training, audits, validation, checkpoint selection, reload, merging, and W&B logging. |
| `scripts/evaluate.py` | The existing generation/parser/scoring path, now reusable by adapted validation. |
| `scripts/verify_step5.py` | Independent coverage, metric, selection, merge, and checkpoint integrity checks. |
| `scripts/publish_step5.py` | Verify W&B results, upload checkpoint weights, download and verify them, and publish the complete evidence bundle. |
| `tests/test_lora.py` | Checks loss masks, full training coverage, tie-break rules, and merge failure gates. |
| `cluster/step5.sbatch` | Requests a six-hour maximum, 4 CPUs, 32 GiB RAM, and one A6000; runs tests/data checks, training, and verification. |
| `cluster/publish_step5.sbatch` | CPU publication job, 8 GiB RAM and a 45-minute maximum, to back up and verify model artifacts after the GPU is released. |

Use the existing setup in [IMPLEMENTATION.md](../../IMPLEMENTATION.md). New checkpoints live at `$MRI_WORK/brain-mri-vlm/checkpoints/step5/<job-id>/`; compact evidence lives at `experiments/step5/runs/<job-id>/`. W&B staging and artifact caches use `$MRI_WORK/cache/wandb/staging` and `$MRI_WORK/cache/wandb/artifacts`. Weights and optimizer states are published as verified W&B artifacts with checksums and restoration commands in this repository, rather than as Git blobs.

## Implementation references

The model's [official fine-tuning example](https://github.com/huggingface/smollm/blob/main/vision/finetuning/Smol_VLM_FT.ipynb) supplies the architecture context; this project uses unquantized LoRA and response-only supervision. [PEFT 0.17 documentation](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model) describes adapter loading, FP32 adapter casting, and safe merging. The installed PEFT version is 0.17.1; actual behavior is checked on the cluster. The repository proposal and roadmap define the one-run, validation-selected, merged-FP16 milestone.

## Restore the completed checkpoint

Inside a compute allocation on Insomnia (CPU is sufficient for downloading), activate the documented environment, authenticate to the team's W&B project, and run from your repository:

```bash
export WANDB_BASE_URL="https://api.forge.coreweave.com"
wandb artifact get coms6998-team32-fall2026/brain-mri-vlm/step5-13927137-checkpoints:v0 --root checkpoints/step5/13927137
```

The selected adapter is `checkpoints/step5/13927137/epoch-003/adapter`. The standalone model and processor are in `checkpoints/step5/13927137/merged-fp16`. Other servers use these relative paths under their own checkout; Sean's absolute root is `/insomnia001/depts/edu/users/ct3013/brain-mri-vlm`.

The next benchmark must load this **merged-fp16 directory**, preserving its processor and generation protocol. The current standalone `scripts/evaluate.py` CLI still loads the original Hugging Face model; Step 6 needs an explicit checkpoint-aware benchmark loader. Downloading the artifact does not run model inference or require a GPU; loading and benchmarking the model does require the documented allocation.

## Reproduction and evidence collection

After preparing the pinned environment and data as documented in `IMPLEMENTATION.md`, run from the repository on the Insomnia login node:

```bash
export MRI_WORK="/insomnia001/depts/edu/users/$USER"
export MRI_CONDA_ENV="$MRI_WORK/envs/smolmri"
export WANDB_BASE_URL="https://api.forge.coreweave.com"
mkdir -p logs
sbatch cluster/step5.sbatch
```

The batch job sets the remaining cache variables and creates fresh run/checkpoint directories keyed by its job ID. Do not reuse a previous output directory. After successful GPU completion, collect accounting and scheduler output, then submit the CPU publisher (replace the example job ID for a new run):

```bash
step5_job=13927137
export MRI_RUN_DIR="experiments/step5/runs/$step5_job"
export MRI_CHECKPOINT_DIR="checkpoints/step5/$step5_job"
sacct -j "$step5_job" --parsable2 \
  --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,Start,End,NodeList,AllocCPUS,ReqMem,AllocTRES,MaxRSS \
  > "$MRI_RUN_DIR/slurm-accounting.tsv"
cp "logs/step5-$step5_job.out" "$MRI_RUN_DIR/slurm-output.log"
sbatch cluster/publish_step5.sbatch
```

Collect the publisher's completed scheduler log/accounting into `publisher-output.log` and `publisher-accounting.tsv`. It must verify the finished training run, download and compare the stable training-evidence files, publish the model artifact, download and compare every checkpoint file, and publish the final evidence bundle. The resulting `model-artifacts.json` contains the exact artifact version and restore command for teammates; use its relative selected-adapter and merged-model paths on your own server.

The training artifact deliberately excludes the actively written console log. The final evidence bundle includes the completed GPU console/scheduler log. Its own upload receipt and the publisher's final log/accounting are collected afterward in GitHub.

## Evidence file guide

Each run uses `runs/<job-id>/`; each validation directory contains predictions for all 579 images. The completed run is `runs/13927137/`. The following explains its preserved evidence; `summary.json` consolidates results and `checksums.sha256` protects the entire Step 5 evidence tree.

| File or directory inside the run | Purpose |
| --- | --- |
| `base-commit.txt`, `source/` | Execution commit and frozen copies of scripts, tests, configurations, batch jobs, and package pins. |
| `console.log`, `slurm-output.log`, `slurm-accounting.tsv` | GPU job output and scheduler identity, resources, elapsed time, state, and exit code. |
| `tests.txt`, `pip-check.txt`, `cuda-check.txt` | Unit tests, installed-dependency consistency, and FP16 GPU execution checks. |
| `pip-freeze-original.txt`, `requirements-installed.txt`, `environment.json` | Installed environment and hardware/runtime identity. |
| `data-verification.json` | Frozen manifest hashes and verification of all 7,200 source files. |
| `run.json` | Effective configuration, input hashes, code commit, checkpoint paths, W&B identity, status, and final merged result. |
| `training-images.json` | Frozen training image identities, labels, SHA-256 hashes, and group IDs. |
| `trainable-parameters.json` | Every LoRA target module and trainable tensor, dtype, shape, and parameter count. |
| `preflight-base/` | Untrained LoRA generation outputs checked against the original eight FP16 examples before optimization. |
| `label-mask-audit.json` | Canonical completion tokenization for every class and the prompt/image/padding masking policy. |
| `loss-audit.json` | Native model loss compared with manually calculated response-only cross entropy. |
| `gradient-audit.json` | Nonzero LoRA gradients, finite gradient norm, and absence of gradients on frozen parameters. |
| `epoch-001-order.json` through `epoch-003-order.json` | Full deterministic per-epoch image order, proving each training image is included once. |
| `training-history.jsonl` | One row per optimizer update: epoch, sample/token counts, token-weighted loss, learning rate, gradient norm, and gradient-scaler value. |
| `epochs.json` | Per-epoch training loss, validation loss/metrics, optimizer step, and adapter checkpoint identity. |
| `selection.json` | Frozen selection rule, all candidates, and the selected epoch. |
| `validation/epoch-001/` through `validation/epoch-003/` | Full validation generation and loss after each epoch. |
| `validation/merged-fp16/` | Full validation after saving and reloading the merged FP16 model. |
| `merge-probe/adapter/`, `merge-probe/merged/` | Fixed eight-image generation checks before and after merging. |
| `merge-verification.json` | Numerical logit differences, token/label agreement, F1 change, dtype/module checks, and the predeclared thresholds. |
| `merge-prediction-differences.json` | Every changed validation label, raw answer, and token sequence between the selected adapter and merged model; input metadata match is independently checked. |
| `checkpoint-files.json` | SHA-256 and byte size for every adapter, optimizer-state, processor, merged-weight, and saved-logit file outside Git. |
| `training-artifact-files.json` | Exact stable files captured in the training run's W&B evidence artifact. |
| `exit-status.tsv`, `independent-verification.json` | Training process exit status and independent coverage/metric/selection/merge/checkpoint checks. |
| `model-artifacts.json` | Model-artifact version/digest, downloaded-file verification, relative adapter/merged paths, and the teammate restore command. |
| `wandb-verification.json`, `evidence-upload.json` | Verified remote training state/metrics, artifact identities, checkpoint upload receipt, and publishing-run link. |
| `provenance-verification.json` | Local independent checks and SHA-256 comparisons linking every source snapshot to the execution commit. |
| `publisher-output.log`, `publisher-accounting.tsv` | CPU publishing job's final output and scheduler evidence. |

Every full validation subdirectory contains `selected-images.json` (ordered frozen identities), `predictions.jsonl` (raw/parsed answers, true labels, generated tokens, and input metadata), `metrics.json` (including confusion matrix and invalid-response rate), `loss.json` (token-weighted target loss), `verification.json` (independently recomputed metrics), `rendered-prompt.txt` (exact inference prompt), and `generation-config.json` (loaded defaults; overrides are fixed in the run config). Probe directories contain the prediction, prompt, and generation-default files.

The external checkpoint tree contains `epoch-001/adapter/` through `epoch-003/adapter/` (PEFT adapter plus processor/tokenizer files), each epoch's `training-state.pt` (optimizer, scheduler, scaler, and RNG snapshot), `merged-fp16/` (standalone FP16 model plus processor/tokenizer), and `merge-logits.npz` (full saved vocabulary logits for independent numerical verification). No test predictions or MRI pixel files belong in this checkpoint.
