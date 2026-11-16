# Brain MRI VLM Midpoint Report

Profile Driven Inference Optimization of a Compact Vision Language Model for Brain MRI Classification

COMS 6998 HPML · Fall 2026 · Team 32 · 16 November 2026

## Project summary

We have completed data preparation, GPU compatibility checks, zero-shot evaluation, one LoRA fine-tune, a reproducible FP16 benchmark, and stage profiling. We also tested the first inference change: reducing the processor image budget from a 2048-pixel longest edge to 1024 while keeping the merged checkpoint and batch size fixed. The paired run reduced mean batch latency by 66.57% (2.99x speedup) while passing the validation quality gate.

The zero-shot model achieved validation macro-F1 0.2662. LoRA followed by an FP16 merge improved it to 0.8620 at batch 1. The frozen batch-4 reference achieved macro-F1 0.8563, 2.8004 images/s and 2.776 GiB peak allocated GPU memory. Profiling then identified CPU preprocessing as the largest measured cost, at 74.17% of diagnostic batch time. These are validation and systems results; the final 1,600-image test set has not been used for model inference. [E2–E6]

## Team and responsibilities

The responsibilities below follow the latest proposal. They describe planned ownership, rather than attributing every recorded run to a particular teammate. [E0]

| Team member | UNI | Proposal responsibility |
| --- | --- | --- |
| Owen Shi | as8254 | Data audit and splits; preprocessing and transfer; quality validation |
| Chengshun Tan | ct3013 | Benchmarking and profiling; vision and prefill experiments |
| Ai Hsiao | ah4617 | LoRA and checkpoint merge; quantization and memory analysis |
| Trung Kien Giang | tg3049 | W&B, repository and Slurm reproducibility; decode and report assembly |

## Goal and background

Our research question is whether a profile-guided change can reduce inference cost for four-class brain MRI classification while keeping macro-F1 within 0.02 absolute of the merged FP16 baseline. A second hypothesis is that INT8/NF4 weight quantization reduces memory but may not reduce latency. The quantization hypothesis remains untested on the merged model.

We use HuggingFaceTB/SmolVLM-500M-Instruct, a compact general-purpose image-text model, on the Nickparvar Brain Tumor MRI Dataset. The task is to return one of four labels: glioma, meningioma, notumor or pituitary. SmolVLM supplies the model architecture; LoRA supplies the parameter-efficient adaptation method. Our contribution is the controlled MRI evaluation and performance experiment, not a new model architecture. [1–4]

Project repository: [oowenn/brain-mri-vlm](https://github.com/oowenn/brain-mri-vlm). Experiment dashboard: [Team 32 brain-mri-vlm](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm). Specific run links and immutable evidence references appear in the run ledger and references.

<!-- pagebreak -->

## Data and experimental setup

We froze Kaggle dataset version 2 and policy v3-conservative-patients before training. The source contains 7,200 images. The W&B artifact is brain-mri-manifests:v2; its artifact version is separate from the policy version. The pipeline retains every original test image and excludes original training images connected to test images under the accepted similarity/patient constraints. Twelve conflicting training annotations were excluded. One flagged original test-label disagreement was preserved. [1, E1]

| Frozen split | Images | Use in this checkpoint |
| --- | ---: | --- |
| Training | 3,279 | One three-epoch LoRA run |
| Validation | 579 | Checkpoint selection, quality checks and timing workloads |
| Test | 1,600 | Held out; no model inference |
| Excluded | 1,742 | Kept in audit inventory; not used for training or evaluation |

The audit recovered 1,778 confirmed source correspondences spanning 212 patients and uses those constraints when splitting. Patient identities remain incomplete, so we cannot claim a fully patient-disjoint evaluation. Exact source hashes and manifest checksums were rechecked before GPU experiments; no run silently regenerated the splits. [E1]

### Hardware and software

GPU work ran through Slurm on Columbia Insomnia using one NVIDIA RTX A6000, 48 GB class, compute capability 8.6. The recorded driver is 575.57.08; PyTorch reports 50,897,289,216 usable device bytes. Benchmark/profile workers used four PyTorch CPU threads. Step 6 ran on ins094, Step 7 on ins092, and the paired image-budget experiment on ins094. The paired host has an AMD EPYC 7543P 32-Core Processor; that job used four allocated CPUs and 8 GiB host RAM. Exact GPU UUIDs, hostnames, allocations and package snapshots are retained with each run. [E2–E6]

The reference environment is Python 3.11.18, PyTorch 2.9.1+cu126, torchvision 0.24.1+cu126, Transformers 4.57.1, PEFT 0.17.1, bitsandbytes 0.48.2 and W&B 0.31.0. PyTorch's CUDA build is 12.6; the driver's advertised CUDA ceiling is not the wheel runtime. Inference uses FP16 and PyTorch SDPA. Complete pins are in requirements-cluster.txt. A separate clean-room reinstall has not been verified.

### Inputs and measurement rules

The model and processor are pinned to revision 753c1752602b5e739af7a7fa6947a635051897a3. The baseline processor uses longest edge 2048, 512-pixel tiles and image splitting. The prompt asks for exactly one of the four class names. We use greedy decoding, one beam, a 16-new-token cap, and left padding for batched generation. Parsing accepts case/whitespace differences and one final period; other outputs are invalid and count as errors. [E2–E6]

On full validation, baseline effective input lengths are 1,181 tokens for 576 images, 916 for two images and 917 for one image. Zero-shot answers used five generated tokens; merged-model labels used two to four, including EOS. Quality passes visit all 579 images once per stated checkpoint/configuration; benchmark warmups and repetitions do not apply to these accuracy passes. Variable input lengths and image shapes are saved per example.

Performance measurements use CUDA synchronization before and after a complete batch: image read/decode, EXIF/RGB conversion, processor, blocking host-to-device copy and FP16 cast, generation, device-to-host output, text decode and parsing. They use warmed filesystem caches and exclude model loading, hashing, W&B/file writes, profiling, request queueing and batch formation. Batch time divided by four is amortized cost, not single-request latency. Peak allocated and reserved memory include the resident model and are distinct from nvidia-smi device usage. [E4]

<!-- pagebreak -->

## Baseline implementation and LoRA results

Step 3 established GPU execution and evaluator compatibility. On the same eight validation images, two per class, pretrained FP16 and INT8 each classified 3/8 correctly (macro-F1 0.2778); NF4 classified 0/8 correctly. All three produced eight valid label responses. These were single compatibility passes at batch 1 on A6000 with SDPA, the fixed prompt and 16-token cap; NF4 used FP16 computation and double quantization. They had no timed repetitions or memory benchmark and do not establish quantization quality preservation. [E2]

Step 4 evaluated all 579 validation images at batch 1. The pretrained FP16 model predicted only glioma or notumor, with zero recall for meningioma and pituitary. Valid output formatting therefore did not imply useful four-class classification. [E2]

### One fixed training run

Step 5 trained LoRA rank 16, alpha 32 and dropout 0.05 in 224 language-decoder attention/MLP projection modules. The 8,683,520 trainable parameters are the adapters; the base model, vision encoder and connector remained frozen. We used an FP16 base with FP32 adapters and gradient scaling, AdamW at 1e-4, weight decay 0.01, cosine decay, 5% learning-rate warmup, gradient clipping 1.0 and seed 42. Training microbatch 4 with four accumulation steps gives effective batch 16, except the tail of each epoch. [3, E3]

All 3,279 training images were visited per epoch, for three epochs and 615 optimizer updates. Loss was applied only to the class label and EOS. There was no augmentation or token truncation. Validation selection was fixed in advance: highest macro-F1, then lowest token-weighted validation loss, then earliest epoch. Learning-rate warmup is separate from benchmark warmup.

| Epoch | Training loss | Validation loss | Validation accuracy | Validation macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.610942 | 0.149584 | 80.83% | 0.815385 |
| 2 | 0.113512 | 0.131799 | 83.25% | 0.833350 |
| 3 | 0.071642 | 0.107852 | 85.49% | 0.857560 |

Epoch results are full 579-image validation passes at batch 1, A6000, SDPA, with the baseline processor/context described in the experimental setup. This is one training seed, not three independent training runs.

### Saved checkpoint quality

The selected epoch-3 adapter was reloaded, merged in FP32, cast to FP16, saved and reloaded again. All eight probe token sequences agreed. Full-validation label agreement between adapter and merged model was 99.4819%; three labels changed and the absolute macro-F1 difference was 0.004459, within the predeclared 0.005 merge tolerance. The merged result is the actual inference checkpoint used later. [E3]

| Evaluated model | Batch | Correct of 579 | Accuracy | Macro-F1 | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained FP16 | 1 | 240 | 41.45% | 0.266220 | 0 |
| Selected adapter before merge | 1 | 495 | 85.49% | 0.857560 | 0 |
| Saved merged FP16 | 1 | 498 | 86.01% | 0.862019 | 0 |
| Frozen benchmark FP16 | 4 | 494 | 85.32% | 0.856293 | 0 |

The merged batch-1 improvement is 44.56 percentage points of accuracy and 0.595798 macro-F1 over zero-shot. It is a quality improvement, not an inference speedup. Moving to batch 4 changed four predictions and lowered macro-F1 by 0.005726; the batch-4 result is the quality reference for optimizations. The cause of those individual changes was not separately isolated. [E3–E4]

<!-- pagebreak -->

## Canonical FP16 performance baseline

Step 6 swept batches 1, 2, 4, 8, 16 and 32 on the saved merged checkpoint. Every candidate completed without an out-of-memory error. Each used two workload seeds, 42 and 43, with 10 warmup and 10 measured batches per seed. Candidate order was shuffled with seed 2026. All used the same A6000 on ins094, four CPU threads, FP16/SDPA, baseline processor and generation policy. [E4]

| Batch | Mean batch ms | p95 batch ms | Images per second | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 398.009 | 431.284 | 2.512504 | 1.427 |
| 2 | 752.254 | 774.432 | 2.658677 | 1.880 |
| 4 | 1418.155 | 1447.047 | 2.820566 | 2.776 |
| 8 | 2763.551 | 2817.161 | 2.894826 | 4.577 |
| 16 | 5437.726 | 5541.576 | 2.942406 | 8.181 |
| 32 | 10806.008 | 11049.214 | 2.961316 | 15.387 |

The rule selected the smallest batch within 5% of the best eligible throughput while using no more than 80% of device memory. Batch 4 narrowly qualified: 2.820566 images/s against a 2.813250 cutoff, or 95.25% of batch 32 throughput. This is a bounded single-allocation choice, not a universal optimum. We froze batch 4 before the final quality and timing passes.

### Repeated measurements at frozen batch 4

The final benchmark used seeds 42, 43 and 44, each with 10 warmup and 50 measured batches. The timing stream draws 32 validation images per class per seed and cycles through them. Thus, 150 measured batches represent 600 image executions and 305 distinct images. Of the measured executions, 596 had 1,181 input tokens and four had 916; generated labels contained two to four tokens. [E4]

| Metric | Step 6 result |
| --- | ---: |
| Mean batch latency | 1428.367 ms |
| Median batch latency | 1430.270 ms |
| p95 batch latency | 1473.171 ms |
| Mean amortized image cost | 357.092 ms |
| Throughput | 2.800400 images/s |
| Peak allocated GPU memory | 2.776 GiB |
| Peak reserved GPU memory | 3.248 GiB |

Trial mean batch times were 1438.949, 1417.595 and 1428.557 ms; their sample standard deviation was 10.678 ms. The p95 above is the pooled batch percentile, not a confidence interval. The separate full-validation pass achieved 494/579 correct, macro-F1 0.856293 and zero invalid responses.

The implementation includes deterministic workload selection, recorded raw answers/token IDs, CUDA-synchronized timing, allocator peak resets, package/hardware snapshots and independent recomputation. W&B metrics and 153 downloaded runtime-artifact files were checked against local evidence. This makes the benchmark a completed core module for the checkpoint. [E4]

<!-- pagebreak -->

## Profiling and bottleneck analysis

Step 7 kept the checkpoint, batch 4, FP16/SDPA, four CPU threads, prompt and processor fixed, but ran on another A6000 node, ins092. A separate uninstrumented control repeated 10 warmups and 50 measured batches for each of seeds 42/43/44. It measured 1830.319 ms per batch and 2.185411 images/s: 28.14% slower than Step 6, with unchanged memory and exact token/answer agreement on all 180 batches including warmups. This difference is runtime variation, not an optimization result. It motivates same-allocation comparisons. [E5]

### Stage measurements

The diagnostic worker used 10 warmups and 10 measured batches per seed, giving 30 measured batches. Synchronized spans isolate CPU preprocessing, transfer/cast, vision features, prefill and cached decode. Prefill excludes nested vision time to avoid double counting. The diagnostic total was 1818.111 ms per batch. [E5]

| Exclusive stage | Mean ms per batch | Share |
| --- | ---: | ---: |
| CPU preprocessing | 1348.567 | 74.17% |
| Host-to-device transfer and FP16 cast | 70.967 | 3.90% |
| Vision features and connector | 250.158 | 13.76% |
| First forward excluding vision | 75.921 | 4.18% |
| Cached decoding forwards | 67.895 | 3.73% |
| Generation and instrumentation residual | 4.604 | 0.25% |

Within preprocessing, image reading/EXIF/RGB averaged 4.695 ms and the image-text processor averaged 1343.775 ms. The processor span includes transforms, padding, tensor creation and tokenization; we did not isolate each internal operation. Warmed image reads were a small measured component. All measured stage batches had padded pixels [4, 17, 3, 512, 512] and 356,591,424 bytes of CPU input tensors. Of 120 image executions, 119 had 1,181 input tokens and one had 916. Short two-to-four-token outputs make further decode shortening a lower priority.

### Traces and memory

PyTorch Profiler recorded CPU/CUDA activity, tensor shapes and memory events. Three traces each used 10 model warmups, one profiler warmup and two active batches. The union of GPU kernel/copy intervals occupied 18.83% of whole-batch windows across the six active batches. This is GPU activity time, not SM occupancy or proof of compute/bandwidth saturation. Separate 200 ms nvidia-smi samples support the host-side bottleneck but include worker warmups and file writes. CUDA events and custom spans support stage timing; Nsight was not used. [5–6, E5]

Model parameter storage was 0.945 GiB, about 34.05% of the unprofiled allocator peak. Weights therefore did not dominate that peak. Reducing the image budget is a direct experiment on the larger input/preprocessing cost; quantization remains a separate memory/latency hypothesis.

All GPU workers completed, but the aggregate job failed when a checker counted both CPU and mirrored GPU annotations. CPU-only recovery corrected that counting, retained the original failed record and traces, passed 42 tests and verified 153 downloaded W&B files. It did not rerun or invent GPU measurements. Instrumented outputs matched their controls on all 99 batches, including warmups. [E5]

<!-- pagebreak -->

## First optimization and implementation status

We changed only the supported Idefics3 processor longest edge from 2048 to 1024. Tiles remain 512 pixels, with splitting enabled. Checkpoint, FP16/SDPA, batch 4, prompt and generation remain fixed. Smaller inputs target the profiled costs but may remove useful MRI detail. [7, E6]

GPU job 13935433 used one A6000 on ins094 and four CPU threads. Fresh timing processes ran seeds 42/43/44 in baseline-first, variant-first and baseline-first order. Each setting used 10 warmups and 50 measured batches per seed: 150 batches and 600 image executions. Workloads and end-to-end timing boundaries follow the experimental setup.

| Matched metric | Baseline 2048 | Variant 1024 |
| --- | ---: | ---: |
| Mean batch latency ms | 1437.526 | 480.519 |
| Median batch latency ms | 1434.985 | 479.946 |
| p95 batch latency ms | 1496.005 | 493.469 |
| Images per second | 2.782558 | 8.324325 |
| Peak allocated GPU GiB | 2.776 | 1.504 |
| Peak reserved GPU GiB | 3.248 | 1.709 |
| Validation accuracy | 85.32% | 85.66% |
| Validation macro-F1 | 0.856293 | 0.859061 |
| Invalid responses | 0 | 0 |

The variant was 2.992x faster by mean batch time and reduced peak allocated memory by 45.80%. Its macro-F1 changed by +0.002768; the predefined validation gate passed. This closes the first-optimization evidence gap with a measured preliminary improvement.

Both settings evaluated all 579 validation images once at batch 4. The gate requires macro-F1 at least 0.8362927507, no more than 0.02 loss against the fresh baseline, and zero invalid answers. The fresh baseline exactly reproduced all 579 Step 6 validation outputs and all 180 timing-batch outputs. Failed gates remain reported.

| Class recall on 579 validation images | Baseline 2048 | Variant 1024 |
| --- | ---: | ---: |
| glioma | 85.64% | 84.53% |
| meningioma | 76.43% | 77.71% |
| notumor | 95.87% | 95.87% |
| pituitary | 85.83% | 87.50% |

### Why the result changed

Separate diagnostics used 10 warmups and 10 measured batches per seed per setting; all 120 instrumented batches matched unprofiled outputs. Pixel shapes changed from [[4, 17, 3, 512, 512]] to [[4, 5, 3, 512, 512]]. Variant validation context: 387 tokens for 579 images. Generated labels used 2–4 tokens including EOS; baseline contexts are listed in the experimental setup.

Diagnostic mean stage changes were: preprocessing 1024.784 to 310.939 ms; transfer 26.611 to 8.503 ms; vision 249.391 to 72.768 ms; prefill 75.870 to 28.572 ms; decode 64.059 to 61.371 ms. The largest absolute savings came from preprocessing and vision work; cached decoding changed little.

Diagnostics are excluded from speedup calculations. Paired trial speedups: 2.968x, 3.049x, 2.959x. These are preliminary results from one allocation and one trained checkpoint. [E6]

<!-- pagebreak -->

## Remaining work and limitations

### Roadmap position

Steps 1–7 are complete. Step 8 now has one implemented image-budget technique, and Step 9 has its matched timing, full-validation and diagnostic results. Other proposed techniques and the full Step 9 comparison matrix are still outstanding. This report records midpoint progress; Steps 10–11 are not complete.

### Next experiments

First, retain the 1024 image budget as a validation-qualified candidate and confirm its benefit in another paired allocation before combining it with another technique. Keep configuration selection on validation data. Any new comparison should retain batch 4, the same checkpoint, GPU type, prompt and timing boundary, with a fresh baseline in the same allocation.

Second, implement parallel CPU preprocessing with two workers, then four, under the same four-core allocation. Check exact processed tensors and image/token association before quality evaluation. Include worker handoff and batch assembly inside timing; process overhead or memory traffic could cancel the benefit.

Third, test INT8 and NF4 as two levels of the quantization technique on the saved merged checkpoint. Keep the original image processor fixed to isolate this effect. Record converted modules, actual GPU placement, latency and allocator peaks. LLM.int8 and NF4 provide the method background, but their published benefits do not establish a benefit for this MRI workload. The project used ordinary LoRA training, not QLoRA training. [8–10]

After isolated experiments, combine only promising settings, re-profile and run full validation. Freeze the chosen configuration before evaluating the unchanged 1,600-image test set. Report class recall, macro-F1, invalid outputs, timing and memory, including regressions. The proposal's ResNet-18 accuracy reference remains outstanding. A larger SmolVLM2 comparison and Gradio demo remain optional; neither has been implemented.

### Blockers and methodological limits

Patient identities are incomplete. Our constraints reduce known leakage, but the results do not establish fully patient-disjoint generalization. The retained data are imbalanced, and source labels contain documented disagreements. We preserve the audit rather than claiming that all label or identity issues are solved.

Validation is reused for epoch selection and system-configuration decisions. The reported quality can be optimistic relative to unseen data. One training seed does not measure training variability, and repeated timing batches are not independent clinical cases. No patient-level or clinical-deployment claim follows from this work.

Runtime differed by 28.14% between Step 6 and the Step 7 control on different nodes. Same-allocation pairing improves attribution but does not remove thermal, CPU-contention, cache or order effects. Stage synchronization and profiler collection add overhead; trace GPU activity is not hardware-counter evidence of arithmetic or bandwidth saturation.

SSH/VPN/Duo interruptions and allocation time limits affected interactive setup. Batch jobs, saved environment paths and W&B/Git evidence make subsequent runs independent of an open terminal. Slurm scheduling estimates can change; job elapsed time includes setup, hashing and uploads and must not be used as inference latency. The Step 7 trace-check failure and CPU recovery are retained explicitly.

Cluster storage is scratch and not backed up by the cluster. We retain compact evidence and source in Git and verified model/evidence artifacts in W&B. Teammates still need to restore the pinned data/model and adapt paths on their own servers. Package pins describe the verified reference environment; a fresh install on every teammate's server has not been tested.

<!-- pagebreak -->

## Run ledger and reproducibility

The ledger lists actual GPU work and verification jobs. Durations are allocation elapsed times, not inference benchmarks. Exact source commits, runtime details, raw predictions and checksums are linked in the evidence references. [E2–E6]

| Step | GPU job and node | Outcome | W&B result run |
| --- | --- | --- | --- |
| 3 compatibility | 13924504; A6000 | Completed; 14 tests; 7m31s | [FP16 7b74ixt5](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/7b74ixt5); [INT8 3n3x3b29](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/3n3x3b29); [NF4 kgqava8s](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/kgqava8s) |
| 4 zero-shot | 13925021; ins087 | Completed; 19 tests; 11m | [cpawhwk5](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/cpawhwk5) |
| 5 LoRA | 13927137; ins093 | Completed; 25 tests; 1h50m40s | [0povsppx](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/0povsppx) |
| 6 benchmark | 13932242; ins094 | Completed; 32 tests; 25m12s | [apjpl38l](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/apjpl38l) |
| 7 profile | 13933296; ins092 | Workers passed; final checker failed; 13m36s; CPU recovery verified | [rvufjfp3](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/rvufjfp3) |
| 8 first variant | 13935433; ins094 | Completed; 46 tests; 00:15:58; quality gate passed | [imqjpch9](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/imqjpch9) |

CPU evidence jobs were 13924530, 13925075, 13927707 and 13932262 for Steps 3–6. Step 7 recovery job 13933535 completed in 1m31s and passed 42 tests. Step 8 evidence job 13935434 completed and verified 201 downloaded runtime-artifact files. The earlier Step 7 attempt 13933253 was canceled after a telemetry UUID error; it is preserved separately and is not included in final results.

### What teammates can reproduce

Each experiments/stepN/README.md explains the files and launch commands. Configurations live in configs/; Slurm launchers in cluster/; scripts/ contains preparation, evaluation, training, benchmarking, profiling and verification. In runs/<job-id>/, source/ plus base-commit.txt identify executed code. Raw prediction/timing files support independent recomputation. Environment, package, scheduler and verification records establish execution context. W&B receipts identify the uploaded artifacts and downloaded-file checks.

The model backup is step5-13927137-checkpoints:v0, digest c5ebce54956dc062af2323368c87cfd0. All 52 model/checkpoint files were downloaded and SHA-256 verified. Step 6, Step 7 and Step 8 use the saved merged-fp16 subdirectory, whose 12 files are checked before execution. Raw datasets and model weights are restored from pinned sources/artifacts rather than committed as Git binaries. [E3–E6]

Reference work root: /insomnia001/depts/edu/users/ct3013. The repository is brain-mri-vlm and the environment is envs/smolmri under that root. The source images are under cache/kaggle/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/2. The merged checkpoint is checkpoints/step5/13927137/merged-fp16 under the repository. Conda, pip, Hugging Face, Kaggle and W&B caches live under the work root's cache directory. IMPLEMENTATION.md documents activation, cache variables, restoration and server-specific substitutions.

Repository evidence snapshot: 21ac283379a30af2fd72cd6d5167baa538423fe6. The source commits that actually executed the historical experiments remain recorded separately; this snapshot collects their final evidence and documentation. GitHub and W&B links identify the project evidence. Authentication may be required for W&B; reviewer access should be checked before submission.

<!-- pagebreak -->

## References

External sources were checked on 16 November 2026. Model/data versions and library documentation versions are given where relevant. Project measurements are supported by the evidence entries below, not inferred from published model benchmarks.

[1] Masoud Nickparvar. Brain Tumor MRI Dataset. Kaggle, version 2. [Dataset version page](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/2).

[2] Hugging Face Smol Models Research. SmolVLM-500M-Instruct model card. Model revision used: 753c1752602b5e739af7a7fa6947a635051897a3. [Official model card](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct).

[3] Edward J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685, 2021. [Original paper](https://arxiv.org/abs/2106.09685).

[4] Andrés Marafioti et al. SmolVLM: Redefining small and efficient multimodal models. arXiv:2504.05299, 2025. [Original paper](https://arxiv.org/abs/2504.05299).

[5] PyTorch. torch.profiler, version 2.9 documentation. [Profiler API and overhead notes](https://docs.pytorch.org/docs/2.9/profiler.html).

[6] PyTorch. torch.cuda.Event, version 2.9 documentation. [CUDA event API](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html).

[7] Hugging Face. Idefics3 image processor implementation, Transformers v4.57.1. [Versioned processor source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/idefics3/image_processing_idefics3.py).

[8] Tim Dettmers, Mike Lewis, Younes Belkada and Luke Zettlemoyer. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. arXiv:2208.07339, 2022. [Original paper](https://arxiv.org/abs/2208.07339).

[9] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman and Luke Zettlemoyer. QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314, 2023. Cited for NF4 background, not as our training procedure. [Original paper](https://arxiv.org/abs/2305.14314).

[10] Hugging Face. Bitsandbytes quantization, Transformers v4.57.1 documentation. [Versioned implementation guide](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes).

### Project evidence

[E0] Team 32. Latest project proposal, COMS 6998 Proposal (5).pdf. [Repository copy](https://github.com/oowenn/brain-mri-vlm/blob/21ac283379a30af2fd72cd6d5167baa538423fe6/COMS%206998%20Proposal%20%285%29.pdf).

[E1] Frozen data policy and finalization record. Counts, recovered identity constraints and remaining limitations. [Data finalization](https://github.com/oowenn/brain-mri-vlm/blob/21ac283379a30af2fd72cd6d5167baa538423fe6/experiments/data_finalization.md).

[E2] Runtime compatibility and full zero-shot evaluation. [Step 3 evidence](https://github.com/oowenn/brain-mri-vlm/blob/21ac283379a30af2fd72cd6d5167baa538423fe6/experiments/step3/README.md); [Step 4 summary and run files](https://github.com/oowenn/brain-mri-vlm/blob/21ac283379a30af2fd72cd6d5167baa538423fe6/experiments/step4/README.md).

[E3] LoRA settings, epoch/merge results and checkpoint verification. [Step 5 evidence](https://github.com/oowenn/brain-mri-vlm/blob/21ac283379a30af2fd72cd6d5167baa538423fe6/experiments/step5/README.md).

[E4] Batch sweep, benchmark protocol, timing records and full validation. [Step 6 evidence](https://github.com/oowenn/brain-mri-vlm/blob/21ac283379a30af2fd72cd6d5167baa538423fe6/experiments/step6/README.md).

[E5] Stage statistics, CUDA traces, fresh control, original failure and verified CPU recovery. [Step 7 evidence](https://github.com/oowenn/brain-mri-vlm/blob/21ac283379a30af2fd72cd6d5167baa538423fe6/experiments/step7/README.md).

[E6] Matched image-budget implementation, raw paired timings, both validation passes, stage diagnostics and downloaded-artifact checks. [Step 8 evidence](https://github.com/oowenn/brain-mri-vlm/blob/21ac283379a30af2fd72cd6d5167baa538423fe6/experiments/step8/README.md).
