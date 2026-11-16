# Step 5: LoRA training and FP16 merge

Status: implementation prepared; training and GPU verification are pending. No trained result is claimed yet.

The fixed protocol is in `configs/step5.json`: one three-epoch run, rank-16/alpha-32 LoRA on all 224 language-decoder attention/MLP projection modules; original weights, vision encoder, and connector frozen. Microbatch 4 with 4 accumulation steps gives effective batch 16 (last update includes all 15 remaining images). AdamW learning rate 0.0001, weight decay 0.01, cosine decay, 5% warm-up, gradient clipping 1.0, LoRA dropout 0.05. FP16 model/compute and FP32 trainable adapters use gradient scaling. Training supervises only the exact category completion and end-of-answer token, with prompt, image, and padding positions ignored. No augmentation, resizing-policy change, or truncation is introduced.

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
