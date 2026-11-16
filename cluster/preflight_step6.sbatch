#!/usr/bin/env bash
#SBATCH --job-name=smolmri-step6-preflight
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/step6-preflight-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
export MRI_WORK="${MRI_WORK:-/insomnia001/depts/edu/users/$USER}"
module load anaconda/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MRI_CONDA_ENV:-$MRI_WORK/envs/smolmri}"
export TMPDIR="$MRI_WORK/tmp"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export MRI_PREFLIGHT_DIR="experiments/step6/preflight/$SLURM_JOB_ID"
mkdir -p experiments/step6/preflight
mkdir "$MRI_PREFLIGHT_DIR"
exec > >(tee "$MRI_PREFLIGHT_DIR/console.log") 2>&1
git rev-parse HEAD > "$MRI_PREFLIGHT_DIR/base-commit.txt"
cp cluster/preflight_step6.sbatch "$MRI_PREFLIGHT_DIR/launch.sbatch"
python -m pip check | tee "$MRI_PREFLIGHT_DIR/pip-check.txt"
python -m unittest discover -s tests -v 2>&1 | tee "$MRI_PREFLIGHT_DIR/tests.txt"
python -u - <<'PY'
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
import torch
from transformers import AutoProcessor
from benchmark import input_audit
from evaluate import select_sample, write_json
from run_step6 import verify_checkpoint
from verify_data import read_csv, validate_manifests
torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
cfg=json.loads(Path('configs/step6.json').read_text())
prepared=Path('data/prepared-v3-conservative-patients')
validate_manifests(prepared)
checkpoint=Path(os.environ.get('MRI_MERGED_MODEL','checkpoints/step5/13927137/merged-fp16'))
root=Path(os.environ['MRI_PREFLIGHT_DIR'])
write_json(root/'checkpoint-verification.json',verify_checkpoint(checkpoint,Path(cfg['reference_run'])))
source=Path(os.environ.get('MRI_SOURCE',Path('logs/source-root.txt').read_text().strip())).resolve()
rows=select_sample(read_csv(prepared/'validation.csv'),2,cfg['seed'])
processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
processor.tokenizer.padding_side=cfg['padding_side']
messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':cfg['prompt']}]}]
prompt=processor.apply_chat_template(messages,add_generation_prompt=True)
write_json(root/'input-audit.json',input_audit(processor,prompt,rows,source))
print('PASS: CPU tests, checkpoint hashes, and batched image/token audit; GPU execution remains pending')
PY
