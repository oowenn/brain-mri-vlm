# Step 4: full zero-shot validation

Status: implementation prepared; the full cluster run is pending. No result is claimed yet.

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

The final evidence inventory and measured results will be added after verification. Historical Step 3 checksums refer to the source at commit `0243f26`; the shared evaluator is now being extended, while historical run files remain unchanged.
