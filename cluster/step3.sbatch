#!/usr/bin/env bash
#SBATCH --job-name=smolmri-step3
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/step3-%j.out
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
export WANDB_DIR="$PWD/wandb"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$WANDB_DIR"
run_dir="experiments/step3/runs/$SLURM_JOB_ID"
mkdir -p experiments/step3/runs
mkdir "$run_dir"
exec > >(tee "$run_dir/console.log") 2>&1
printf 'Job: %s\nHost: %s\n' "$SLURM_JOB_ID" "$(hostname)"
date -u '+Started UTC: %Y-%m-%dT%H:%M:%SZ'
git rev-parse HEAD > "$run_dir/base-commit.txt"
python -m pip freeze > "$run_dir/pip-freeze-original.txt"
python -m pip list --format=freeze > "$run_dir/requirements-installed.txt"
python -m pip check | tee "$run_dir/pip-check.txt"
python -m unittest discover -s tests -v 2>&1 | tee "$run_dir/tests.txt"
python scripts/check_environment.py | tee "$run_dir/cuda-check.txt"
source_root="${MRI_SOURCE:-$(cat logs/source-root.txt)}"
python scripts/verify_data.py --source "$source_root" --output "$run_dir/data-verification.json"
status=0
for precision in fp16 int8 int4; do
    if python -u scripts/evaluate.py --source "$source_root" --precision "$precision" \
        --output "$run_dir/$precision" --log-wandb; then
        printf '%s\t0\n' "$precision" >> "$run_dir/exit-status.tsv"
    else
        code=$?
        printf '%s\t%s\n' "$precision" "$code" >> "$run_dir/exit-status.tsv"
        status=1
    fi
done
date -u '+Finished UTC: %Y-%m-%dT%H:%M:%SZ'
exit "$status"
