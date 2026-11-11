#!/bin/bash -l
#SBATCH --job-name=mri-data
#SBATCH --account=edu
#SBATCH --partition=edu1
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/data-%j.out
set -eo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MRI_CONDA_ENV:-$HOME/.conda/envs/coms6998}"
set -u
# Keep numerical libraries from starting additional worker pools per image worker.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-$PWD/data/cache}"
export WANDB_ENTITY="${WANDB_ENTITY:-coms6998-team32-fall2026}"
export WANDB_PROJECT="${WANDB_PROJECT:-brain-mri-vlm}"
python -u scripts/prepare_data.py --download --config configs/data.json \
    --output "${MRI_DATA_OUTPUT:-data/prepared-v2-phash4-ncc98}" \
    --workers "$SLURM_CPUS_PER_TASK" --log-wandb
