#!/usr/bin/env bash
#SBATCH --job-name=mri-lora-finetune-publish
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/lora-finetune-publish-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
: "${MRI_RUN_DIR:?Set MRI_RUN_DIR to a completed Step 5 run directory}"
export MRI_WORK="${MRI_WORK:-/insomnia001/depts/edu/users/$USER}"
module load anaconda/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MRI_CONDA_ENV:-$MRI_WORK/envs/smolmri}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
export TMPDIR="$MRI_WORK/tmp"
export WANDB_CACHE_DIR="$MRI_WORK/cache/wandb/artifacts"
export WANDB_DATA_DIR="$MRI_WORK/cache/wandb/staging"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$TMPDIR"
export WANDB_DIR="$PWD/wandb"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u -m mri_vlm.experiments.lora_finetune.publish --run-dir "$MRI_RUN_DIR" \
    --checkpoints "${MRI_CHECKPOINT_DIR:?Set MRI_CHECKPOINT_DIR to the matching checkpoint directory}"
