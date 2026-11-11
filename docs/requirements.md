# Project requirements and readiness

Source of scope: the accepted `COMS 6998 Proposal.pdf`; course deliverables and dates: `HPML-Final-Project-Fall2026.pdf`. This document separates commitments from provisional setup choices. Readiness checked November 11, 2026.

## Required scientific scope

- Primary model: `HuggingFaceTB/SmolVLM-500M-Instruct`; pin its Hub commit and processor revision.
- Dataset: Nickparvar Brain Tumor MRI, four classes: glioma, meningioma, pituitary, no tumor. Record source version, license/terms, checksums, label mapping, and counts.
- Retain the provided test partition; remove matching training images for cross-split exact/near duplicates before training. Define the perceptual-hash algorithm and threshold, audit removals, and group duplicates when making train/validation splits. Patient-disjoint splitting cannot be claimed without patient identifiers. Use test images only for the predefined leakage audit until final evaluation.
- Zero-shot evaluation, one LoRA fine-tune selected on validation, then merge adapters into FP16 weights. Record adapter targets/rank, seed, optimizer, epochs, validation loss, and checkpoint hashes. Verify merged-model predictions against the adapter model.
- One trained ResNet-18 accuracy reference row; do not turn this into another optimization study.
- Sweep baseline batch sizes, select one, then hold it fixed with the GPU type for all comparisons.
- Profile preprocessing, transfer, vision encoder, prefill, and decoding; explain the dominant resource. Define non-overlapping stage boundaries so encoder work is not counted again inside prefill. Measure end-to-end time independently; GPU events alone omit CPU work.
- Choose at most 2–3 optimizations from measured bottlenecks. Quantization is part of the stated hypothesis: attempt INT8 and INT4 compatibility early, document unsupported variants, and discuss any scope adjustment with the instructor. Do not assume either improves latency.
- Use the same merged checkpoint, input workload, prompt/output rules, and generation settings except for the explicit factor under study. Record image/crop/token budgets and any invalid label outputs.
- Choose configurations on validation, freeze them before final testing. Keep the test set out of tuning.
- Quality: macro-F1 degradation at most 0.02 absolute (two percentage points), with per-class recall reported. Verify this interpretation with the team before freezing the protocol.

## Measurement contract

- 10 warm-up runs, 50 timed measurements per trial, repeated trials, recorded seeds. Decide and freeze the trial count before benchmarking.
- Record whether each measurement is a batch; distinguish batch latency, batch latency divided by batch size, and request latency. Compute throughput from images processed and wall time.
- Mean/p50/p95 latency, images/second, peak GPU allocated and reserved memory; record total device usage separately if sampled. GPU utilization and profiler traces support bottleneck explanations.
- Synchronize CUDA at timing boundaries, reset peak memory statistics for each measurement window, and keep model loading/compilation separate from steady-state inference. Report cold-start costs separately if relevant.
- Run profiler collection separately from performance trials. Avoid CPU/disk caches, shared GPU contention, logging overhead, or changed prompts silently changing the comparison. Report the cache policy and GPU sharing status.
- Every run: Git commit/dirty status, model and dataset revisions, split manifest hash, baseline run ID, GPU model/UUID, driver/CUDA/PyTorch/package versions, Slurm job ID, dtype/quantization scheme, CPU count, batch/input/output sizes, timing boundaries, warm-up/repetition/trial settings, seeds, quality and performance metrics.
- Log configurations/results to one shared W&B project, with selected traces and checkpoint/split references. Ensure teaching staff can open the final record. Offline runs may be synced later.

## Infrastructure and dependencies

One consistently available NVIDIA GPU type; start with one GPU, four CPUs, 32 GB host RAM for a smoke test (provisional requests, not measured training requirements). Measure actual memory and runtime before requesting training resources. The model's FP16 weights alone are roughly 1 GB; training activations, optimizer states, images, and caches add to this. Do not assume that estimate is total VRAM.

Core software: Python, compatible CUDA PyTorch/torchvision, Transformers, Accelerate, PEFT, bitsandbytes, Pillow, ImageHash, NumPy, scikit-learn, YAML, W&B, and a dataset download route. Python 3.11 is a provisional starting point; use a version supported by the chosen cluster packages. Pin versions only after FP16, LoRA merge, INT8 and INT4 smoke tests. Nsight Systems is optional; PyTorch Profiler is required.

Accounts/access: GitHub team write access; approved Insomnia account for each member; Duo and campus/VPN access; authorized Slurm account/partition/GPU type; shared storage and quota; W&B team/project plus reviewer access; dataset download access; Hugging Face download access (token only if required).

## First milestones

Course midpoint ends **November 13, 2026**: baseline running, profile available, first optimization underway. This is two days after the readiness check; prioritize the smallest end-to-end baseline. The larger SmolVLM2 model and Gradio are stretch work and should be deferred.

1. Verify login and allocation; execute GPU smoke job; record GPU/driver/modules/storage/quota.
2. Resolve a compatible environment; pin it; verify model load, inference, LoRA/merge and quantization support on a small sample.
3. Acquire data; audit duplicates; freeze manifests, labels, and validation split.
4. Evaluate zero-shot; run the one LoRA training workflow; create merged FP16 baseline and batch-size sweep.
5. Collect baseline quality/performance/profile; select the first bottleneck-driven optimization.
6. Finish matched experiments and ResNet reference; assemble reproducible commands, W&B comparisons, and demo.

Final report and code: **December 11, 2026**. IEEE conference LaTeX report, GitHub code with setup/reproduction instructions, tracking record, and individual contribution statement. Presentations: **December 17–23, 2026**, 15 minutes plus questions, class template, each member presents.

## Open decisions

- Course-specific allocation, partition, GPU request syntax/type, limits and shared storage path.
- W&B entity/project and artifact retention.
- Dataset license/version, duplicate threshold, split seed/fraction, patient metadata availability.
- Prompt/label parser, maximum generation length, baseline batch size, repeated trial count.
- Compatible package pins, model revision and LoRA configuration.
- GitHub URL and collaborator handles (Owen will create the remote).
