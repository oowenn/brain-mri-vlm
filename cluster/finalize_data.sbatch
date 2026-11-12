#!/bin/bash -l
#SBATCH --job-name=mri-finalize
#SBATCH --account=edu
#SBATCH --partition=edu1
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/finalize-%j.out
set -eo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MRI_CONDA_ENV:-$HOME/.conda/envs/coms6998}"
set -u
export WANDB_ENTITY="${WANDB_ENTITY:-coms6998-team32-fall2026}"
export WANDB_PROJECT="${WANDB_PROJECT:-brain-mri-vlm}"
python -u scripts/finalize_data.py --prepared "${MRI_AUDIT_DIR:-data/prepared-v2-phash4-ncc98}" \
    --mapping "${MRI_MAPPING_PATH:-data/figshare-recovery/results/patient_mapping.csv}" \
    --output "${MRI_DATA_OUTPUT:-data/prepared-v3-conservative-patients}" --log-wandb
