# smolmri

Profile-driven inference optimization of SmolVLM-500M for four-class brain MRI classification. COMS 6998 HPML, Fall 2026, Team 32.

Status: project preparation; no baseline results yet. `smolmri` is the suggested repository name. The GitHub repository will be created by Owen.

## Start here

- [Requirements and milestones](docs/requirements.md)
- [GitHub and Insomnia setup](docs/setup.md)
- [Accepted proposal](COMS%206998%20Proposal.pdf)
- [Course requirements](HPML-Final-Project-Fall2026.pdf)

## Study

Zero-shot evaluation → one LoRA fine-tune → merged FP16 baseline → profile → 2–3 targeted optimizations → matched performance and quality evaluation.

Use one fixed NVIDIA GPU type and one batch size chosen on the baseline. Measure preprocessing, transfer, vision encoding, prefill, and decode. Report mean/p50/p95 latency, throughput, peak GPU memory, macro-F1, and per-class recall. The quality allowance is 0.02 absolute macro-F1 relative to baseline. The quantization claim is a hypothesis to test, not a promised speedup.

## Team

| Member | Initial responsibility |
| --- | --- |
| Owen Shi | Data, deduplication/splits, preprocessing/transfer, quality validation |
| Chengshun Tan | Benchmark harness, FP16 baseline, profiling, encoder/prefill |
| Ai Hsiao | LoRA/merge, quantization, memory analysis |
| Trung Kien Giang | Tracking, repo/environment/Slurm, decoding, report assembly |

Roles can evolve. Preserve individual commits and review teammates' pull requests.

## Reproduction status

Environment versions, exact GPU, dataset manifests, model revision, and commands will be frozen after the cluster smoke test. `requirements.in` is a dependency inventory, not a tested lockfile. Do not compare benchmark results across unpinned environments.

Keep data, weights, credentials, profiler traces, and generated run outputs outside Git; publish selected artifacts through the shared W&B project. See the setup guide for access and the next verification steps.
