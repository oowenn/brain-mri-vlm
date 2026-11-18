#!/usr/bin/env bash
#SBATCH --job-name=mri-baseline-benchmark-publish
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=logs/baseline-benchmark-publish-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
: "${MRI_RUN_DIR:?Set MRI_RUN_DIR to a completed Step 6 run directory}"
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
gpu_job="$(basename "$MRI_RUN_DIR")"
[[ "$gpu_job" =~ ^[0-9]+$ ]] || { echo 'Expected a numeric GPU job directory'; exit 1; }
sacct -j "$gpu_job" --parsable2 \
    --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,Start,End,NodeList,AllocCPUS,ReqMem,AllocTRES,MaxRSS \
    > "$MRI_RUN_DIR/slurm-accounting.tsv"
cp "logs/baseline-benchmark-$gpu_job.out" "$MRI_RUN_DIR/slurm-output.log"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
root=Path(os.environ['MRI_RUN_DIR'])
commit=(root/'base-commit.txt').read_text().strip()
verified={}
for path in sorted((root/'source').rglob('*')):
    if path.is_file():
        name=str(path.relative_to(root/'source'))
        original=subprocess.check_output(['git','show',f'{commit}:{name}'])
        if path.read_bytes()!=original:
            raise ValueError(f'Execution source differs from commit: {name}')
        verified[name]=hashlib.sha256(original).hexdigest()
(root/'source-provenance.json').write_text(json.dumps(dict(status='passed',commit=commit,files=verified),indent=2)+'\n')
PY
python -u -m mri_vlm.experiments.baseline_benchmark.publish --run-dir "$MRI_RUN_DIR"
