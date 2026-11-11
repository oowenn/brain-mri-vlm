# brain-mri-vlm

Profile-driven inference optimization of SmolVLM-500M for four-class brain MRI classification. COMS 6998 HPML, Fall 2026, Team 32.

Status: project preparation; no baseline results yet. Environment and cluster allocation verification are next.

## Study

Zero-shot evaluation → one LoRA fine-tune → merged FP16 baseline → profile → 2–3 targeted optimizations → matched performance and quality evaluation.

Use one fixed NVIDIA GPU type and one batch size chosen on the baseline. Measure preprocessing, transfer, vision encoding, prefill, and decode. Report mean/p50/p95 latency, throughput, peak GPU memory, macro-F1, and per-class recall. The quality allowance is 0.02 absolute macro-F1 relative to baseline. The quantization claim is a hypothesis to test, not a promised speedup.

## Reproduction status

Environment versions, exact GPU, dataset manifests, model revision, and commands will be frozen after the cluster smoke test. `requirements.in` is a dependency inventory, not a tested lockfile. Do not compare benchmark results across unpinned environments.

Keep data, weights, credentials, profiler traces, and generated run outputs outside Git; publish selected artifacts through the shared W&B project. Planning notes and reference PDFs are kept locally and are not required to run the project.

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
