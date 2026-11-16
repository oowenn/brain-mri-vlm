# Step 5: LoRA training and FP16 merge

Status: GPU job `13927137` is training on `ins093` from commit `873598e3a0f1fe8cecb3b2da5bda5962f63de7fa`. All 25 tests, 7,200 source checks, eight untrained reference outputs, label masks, native/manual loss agreement, and LoRA-only gradients passed. Full validation, selection, and merge verification are pending. [W&B training run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/0povsppx).

The fixed protocol is in `configs/step5.json`: one three-epoch run, rank-16/alpha-32 LoRA on all 224 language-decoder attention/MLP projection modules; original weights, vision encoder, and connector frozen. Verified trainable count: **8,683,520 of 516,165,824 total parameters**. Microbatch 4 with 4 accumulation steps gives effective batch 16 (last update includes all 15 remaining images). AdamW learning rate 0.0001, weight decay 0.01, cosine decay, 5% warm-up, gradient clipping 1.0, LoRA dropout 0.05. FP16 model/compute and FP32 trainable adapters use gradient scaling. Training supervises only the exact category completion and end-of-answer token, with prompt, image, and padding positions ignored. No augmentation, resizing-policy change, or truncation is introduced.

Every epoch uses all 3,279 frozen training images exactly once, shuffled by `random.Random(42 + epoch)`. Accumulated gradients are weighted by supervised token counts, including the final incomplete batch. The same 579 validation images are used for generation metrics and token-weighted validation loss. Select highest macro-F1, then lowest validation loss, then earliest epoch. No final-test inference is allowed.

Before the first optimizer update, reproduce the eight original FP16 reference outputs, verify canonical chat token boundaries for every class, compare native loss with manually masked cross entropy, and verify that gradients reach only LoRA parameters. Epoch adapters and optimizer/scheduler/scaler/RNG states are saved outside Git. These are recovery records; automated resume is not implemented.

Reload the selected adapter, merge in FP32 with PEFT safe merging, cast to FP16, save, and reload the merged model. Predeclared checks require all 8 probe token sequences to match; maximum absolute next-token logit error <= 0.15 and mean absolute error <= 0.02; >= 99% label agreement across all validation images and absolute macro-F1 difference <= 0.005. Require no remaining LoRA modules and all model parameters in FP16. These are numerical merge tolerances, not the later optimization quality allowance.

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
| `cluster/step5.sbatch` | Six-hour maximum GPU allocation, 4 CPUs, 32 GiB RAM, one A6000; tests/data checks, training, and verification. |
| `cluster/publish_step5.sbatch` | CPU publication job, 8 GiB RAM and a 45-minute maximum, to back up and verify model artifacts after the GPU is released. |

Use the existing setup in [IMPLEMENTATION.md](../../IMPLEMENTATION.md). New checkpoints live at `$MRI_WORK/brain-mri-vlm/checkpoints/step5/<job-id>/`; compact evidence lives at `experiments/step5/runs/<job-id>/`. W&B staging and artifact caches use `$MRI_WORK/cache/wandb/staging` and `$MRI_WORK/cache/wandb/artifacts`. Weights and optimizer states will be published as W&B artifacts with checksums and restoration commands in this repository, rather than as Git blobs.

## Implementation references

The model's [official fine-tuning example](https://github.com/huggingface/smollm/blob/main/vision/finetuning/Smol_VLM_FT.ipynb) supplies the architecture context; this project uses unquantized LoRA and response-only supervision. [PEFT 0.17 documentation](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model) describes adapter loading, FP32 adapter casting, and safe merging. The installed PEFT version is 0.17.1; actual behavior is checked on the cluster. The repository proposal and roadmap define the one-run, validation-selected, merged-FP16 milestone.

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

Each run uses `runs/<job-id>/`; each validation directory contains predictions for all 579 images. The following explains the evidence generated by this implementation. File existence and successful verification determine completion; this guide alone is not a claim that the run has finished.

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
| `checkpoint-files.json` | SHA-256 and byte size for every adapter, optimizer-state, processor, merged-weight, and saved-logit file outside Git. |
| `training-artifact-files.json` | Exact stable files captured in the training run's W&B evidence artifact. |
| `exit-status.tsv`, `independent-verification.json` | Training process exit status and independent coverage/metric/selection/merge/checkpoint checks. |
| `model-artifacts.json` | Model-artifact version/digest, downloaded-file verification, relative adapter/merged paths, and the teammate restore command. |
| `wandb-verification.json`, `evidence-upload.json` | Verified remote training state/metrics, artifact identities, checkpoint upload receipt, and publishing-run link. |
| `publisher-output.log`, `publisher-accounting.tsv` | CPU publishing job's final output and scheduler evidence. |

Every full validation subdirectory contains `selected-images.json` (ordered frozen identities), `predictions.jsonl` (raw/parsed answers, true labels, generated tokens, and input metadata), `metrics.json` (including confusion matrix and invalid-response rate), `loss.json` (token-weighted target loss), `verification.json` (independently recomputed metrics), `rendered-prompt.txt` (exact inference prompt), and `generation-config.json` (loaded defaults; overrides are fixed in the run config). Probe directories contain the prediction, prompt, and generation-default files.

The external checkpoint tree contains `epoch-001/adapter/` through `epoch-003/adapter/` (PEFT adapter plus processor/tokenizer files), each epoch's `training-state.pt` (optimizer, scheduler, scaler, and RNG snapshot), `merged-fp16/` (standalone FP16 model plus processor/tokenizer), and `merge-logits.npz` (full saved vocabulary logits for independent numerical verification). No test predictions or MRI pixel files belong in this checkpoint.
