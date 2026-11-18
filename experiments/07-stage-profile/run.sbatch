#!/usr/bin/env bash
#SBATCH --job-name=mri-stage-profile
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/stage-profile-%j.out
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch on Insomnia}"
cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
export MRI_WORK="${MRI_WORK:-/insomnia001/depts/edu/users/$USER}"
module load anaconda/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MRI_CONDA_ENV:-$MRI_WORK/envs/smolmri}"
export CONDA_PKGS_DIRS="$MRI_WORK/cache/conda"
export PIP_CACHE_DIR="$MRI_WORK/cache/pip"
export HF_HOME="$MRI_WORK/cache/huggingface"
export KAGGLEHUB_CACHE="$MRI_WORK/cache/kaggle"
export TMPDIR="$MRI_WORK/tmp"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
export WANDB_CACHE_DIR="$MRI_WORK/cache/wandb/artifacts"
export WANDB_DATA_DIR="$MRI_WORK/cache/wandb/staging"
export WANDB_DIR="$PWD/wandb"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
run_dir="experiments/07-stage-profile/runs/$SLURM_JOB_ID"
mkdir -p experiments/07-stage-profile/runs
mkdir "$run_dir"
exec > >(tee "$run_dir/console.log") 2>&1
printf 'Job: %s\nHost: %s\n' "$SLURM_JOB_ID" "$(hostname)"
date -u '+Started UTC: %Y-%m-%dT%H:%M:%SZ'
git rev-parse HEAD > "$run_dir/base-commit.txt"
for name in $(git ls-files -- mri_vlm ':(glob)experiments/*/config.json' experiments/02-data-split/policy.json experiments/07-stage-profile/run.sbatch experiments/07-stage-profile/publish.sbatch); do
    mkdir -p "$run_dir/source/$(dirname "$name")"
    cp "$name" "$run_dir/source/$name"
done
python -m pip freeze > "$run_dir/pip-freeze-original.txt"
python -m pip list --format=freeze > "$run_dir/requirements-installed.txt"
python -m pip check | tee "$run_dir/pip-check.txt"
python -m mri_vlm.environment | tee "$run_dir/cuda-check.txt"
source_root="${MRI_SOURCE:-$(cat logs/source-root.txt)}"
python -m mri_vlm.data.verify --source "$source_root" --output "$run_dir/data-verification.json"
checkpoint_dir="${MRI_MERGED_MODEL:-$PWD/checkpoints/lora-finetune/13927137/merged-fp16}"
python -u -m mri_vlm.experiments.stage_profile.run --source "$source_root" --config experiments/07-stage-profile/config.json \
    --output "$run_dir" --checkpoint "$checkpoint_dir"
printf 'control-and-profiling\t0\n' > "$run_dir/exit-status.tsv"
python -m mri_vlm.experiments.stage_profile.verify --run-dir "$run_dir"
date -u '+Finished UTC: %Y-%m-%dT%H:%M:%SZ'
