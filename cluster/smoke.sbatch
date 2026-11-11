#!/usr/bin/env bash
#SBATCH --job-name=smolmri-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke-%j.out
set -euo pipefail
# Supply verified --account, --partition, GPU request and optional constraint
# at submission time. No site-specific allocation values are assumed here.
: "${SLURM_JOB_ID:?Submit through sbatch on Insomnia}"
: "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"
cd "$SLURM_SUBMIT_DIR"
if [[ -z "${SMOLMRI_VENV:-}" || ! -f "$SMOLMRI_VENV/bin/activate" ]]; then
    echo 'Set SMOLMRI_VENV to the absolute path of the prepared Python venv.' >&2
    exit 1
fi
source "$SMOLMRI_VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
printf 'Job: %s\nHost: %s\n' "$SLURM_JOB_ID" "$(hostname)"
nvidia-smi
python scripts/check_environment.py
