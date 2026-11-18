"""Brain MRI classification with SmolVLM: data preparation, inference, training and experiment code.

Layout
  common.py, naming.py   shared helpers and the single naming scheme for folders and W&B
  environment.py         print the CUDA and package versions on a GPU node
  data/                  build, finalize and verify the frozen image splits
  inference/             evaluation, batched benchmarking, stage profiling and prediction checks
  training/              LoRA fine-tuning and FP16 merge
  experiments/<name>/    run, verify and publish one experiment

Run modules from the repository root, e.g. python -m mri_vlm.experiments.image_size.run --help
"""
