#!/usr/bin/env bash
#SBATCH --job-name=smolmri-step6
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/step6-%j.out
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
run_dir="experiments/step6/runs/$SLURM_JOB_ID"
mkdir -p experiments/step6/runs
mkdir "$run_dir"
exec > >(tee "$run_dir/console.log") 2>&1
printf 'Job: %s\nHost: %s\n' "$SLURM_JOB_ID" "$(hostname)"
date -u '+Started UTC: %Y-%m-%dT%H:%M:%SZ'
git rev-parse HEAD > "$run_dir/base-commit.txt"
for name in scripts/*.py tests/*.py configs/*.json cluster/step6.sbatch cluster/publish_step6.sbatch requirements-cluster.txt; do
    mkdir -p "$run_dir/source/$(dirname "$name")"
    cp "$name" "$run_dir/source/$name"
done
python -m pip freeze > "$run_dir/pip-freeze-original.txt"
python -m pip list --format=freeze > "$run_dir/requirements-installed.txt"
python -m pip check | tee "$run_dir/pip-check.txt"
python -m unittest discover -s tests -v 2>&1 | tee "$run_dir/tests.txt"
python scripts/check_environment.py | tee "$run_dir/cuda-check.txt"
source_root="${MRI_SOURCE:-$(cat logs/source-root.txt)}"
python scripts/verify_data.py --source "$source_root" --output "$run_dir/data-verification.json"
checkpoint_dir="${MRI_MERGED_MODEL:-$PWD/checkpoints/step5/13927137/merged-fp16}"
python -u scripts/run_step6.py --source "$source_root" --config configs/step6.json \
    --output "$run_dir" --checkpoint "$checkpoint_dir"
printf 'benchmark-and-quality\t0\n' > "$run_dir/exit-status.tsv"
python scripts/verify_step6.py --run-dir "$run_dir"
date -u '+Finished UTC: %Y-%m-%dT%H:%M:%SZ'
