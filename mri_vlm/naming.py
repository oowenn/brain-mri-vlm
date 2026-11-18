"""Names for every experiment: folders, configs, Slurm logs, checkpoints, and W&B groups and artifacts.

Each config keeps its roadmap step number in the ``stage`` field (``step3`` … ``step8``).
Archived runs and W&B configs record that value, so it never changes. Everything else
derives from the experiment's short name (its slug).

Runs executed before the November 18, 2026 reorganization recorded older paths
(``configs/step6.json``, ``experiments/step5/…``) and W&B artifact names
(``step5-13927137-checkpoints``). ``current_path``, ``current_artifact`` and ``upgrade_config``
translate those records to today's names, so archived evidence stays byte-for-byte unchanged
and can still be compared with current configs.
"""
from pathlib import Path

SLUGS = {
    'step2': 'data-split',
    'step3': 'precision-check',
    'step4': 'zero-shot',
    'step5': 'lora-finetune',
    'step6': 'baseline-benchmark',
    'step7': 'stage-profile',
    'step8': 'image-size',
}

# W&B display names. New runs append the Slurm job ID so repeated runs stay distinguishable.
TITLES = {
    ('step3', 'fp16'): 'Test FP16 inference on eight MRI images',
    ('step3', 'int8'): 'Test 8-bit inference on eight MRI images',
    ('step3', 'int4'): 'Test 4-bit NF4 inference on eight MRI images',
    ('step3', 'evidence'): 'Verify the three precision checks and save their predictions',
    ('step4', 'fp16'): 'Evaluate SmolVLM before fine-tuning',
    ('step4', 'evidence'): 'Verify the zero-shot predictions and scores',
    ('step5', 'run'): 'Fine-tune SmolVLM with LoRA and merge the weights',
    ('step5', 'evidence'): 'Back up and verify the fine-tuned model',
    ('step6', 'run'): 'Compare batch sizes and benchmark the FP16 model',
    ('step6', 'evidence'): 'Verify the FP16 batch timings and validation scores',
    ('step7', 'run'): 'Profile FP16 inference by stage',
    ('step7', 'recovery'): 'Recover and verify the FP16 stage timings',
    ('step7', 'evidence'): 'Archive the verified FP16 profiles and GPU traces',
    ('step8', 'run'): 'Compare 2,048 px and 1,024 px MRI inputs',
    ('step8', 'evidence'): 'Verify the image-size timings and predictions',
}


def slug(stage):
    return SLUGS[stage]


def folder(stage):
    """Repository-relative experiment folder, e.g. ``experiments/08-image-size``."""
    return f'experiments/{int(stage.removeprefix("step")):02d}-{SLUGS[stage]}'


def config_path(stage):
    return f'{folder(stage)}/config.json'


def group(stage, job):
    return f'{SLUGS[stage]}-{job}'


def artifact(stage, job, kind):
    """W&B artifact name: ``<slug>-<job>-<kind>``. Kinds: evidence, archive, checkpoints, fp16, int8, int4."""
    return f'{SLUGS[stage]}-{job}-{kind}'


def run_name(stage, role, job):
    return f'{TITLES[stage, role]} · {job}'


def checkpoint_dir(stage, job):
    """Where a restored model artifact lives beneath the repository on the cluster."""
    return f'checkpoints/{SLUGS[stage]}/{job}'


# Old path prefix → current path prefix. Exact files are listed before folders.
LEGACY_PATHS = [
    ('configs/data.json', 'experiments/02-data-split/config.json'),
    ('configs/data_policy.json', 'experiments/02-data-split/policy.json'),
    *[(f'configs/step{n}.json', config_path(f'step{n}')) for n in range(3, 9)],
    ('cluster/prepare_data.sbatch', 'experiments/02-data-split/prepare.sbatch'),
    ('cluster/finalize_data.sbatch', 'experiments/02-data-split/finalize.sbatch'),
    ('cluster/preflight_step6.sbatch', f'{folder("step6")}/preflight.sbatch'),
    ('cluster/recover_step7.sbatch', f'{folder("step7")}/recover.sbatch'),
    *[(f'cluster/step{n}.sbatch', f'{folder(f"step{n}")}/run.sbatch') for n in range(3, 9)],
    *[(f'cluster/publish_step{n}.sbatch', f'{folder(f"step{n}")}/publish.sbatch') for n in range(3, 9)],
    ('experiments/data_finalization.md', 'experiments/02-data-split/README.md'),
    ('experiments/similarity_calibration.md', 'experiments/02-data-split/similarity-calibration.md'),
    ('experiments/HISTORY.md', 'docs/history.md'),
    ('requirements-cluster.txt', 'requirements/cluster.txt'),
    ('requirements-data.txt', 'requirements/data.txt'),
    ('requirements.in', 'requirements/cluster.in'),
    ('checkpoints/step5/', f'{checkpoint_dir("step5", "")}'),
    *[(f'experiments/step{n}/', f'{folder(f"step{n}")}/') for n in range(3, 9)],
]

# Old W&B artifact collection → current name. A rename keeps every version, alias and digest.
LEGACY_ARTIFACTS = {
    'step3-13924504-fp16': 'precision-check-13924504-fp16',
    'step3-13924504-int8': 'precision-check-13924504-int8',
    'step3-13924504-int4': 'precision-check-13924504-int4',
    'step3-checkpoint-13924504': 'precision-check-13924504-archive',
    'step4-13925021-fp16': 'zero-shot-13925021-fp16',
    'step4-checkpoint-13925021': 'zero-shot-13925021-archive',
    'step5-13927137-training-evidence': 'lora-finetune-13927137-evidence',
    'step5-13927137-checkpoints': 'lora-finetune-13927137-checkpoints',
    'step5-checkpoint-13927137': 'lora-finetune-13927137-archive',
    'step6-13932242-baseline': 'baseline-benchmark-13932242-evidence',
    'step6-checkpoint-13932242': 'baseline-benchmark-13932242-archive',
    'step7-13933296-profile': 'stage-profile-13933296-evidence',
    'step7-checkpoint-13933296': 'stage-profile-13933296-archive',
    'step8-13935433-image-budget': 'image-size-13935433-evidence',
    'step8-checkpoint-13935433': 'image-size-13935433-archive',
}


def current_path(path):
    """Translate a path recorded by an older run into its current repository location."""
    text = str(path)
    for old, new in LEGACY_PATHS:
        if text == old or (old.endswith('/') and text.startswith(old)):
            return new + text[len(old):]
    return text


def legacy_path(path):
    """Inverse of ``current_path``: the name an older run used for a current file."""
    text = str(path)
    for old, new in LEGACY_PATHS:
        if text == new or (new.endswith('/') and text.startswith(new)):
            return old + text[len(new):]
    return text


def current_artifact(reference):
    """Translate ``entity/project/name:alias`` (or a bare name) to the current artifact name."""
    prefix, _, rest = reference.rpartition('/')
    name, colon, alias = rest.partition(':')
    name = LEGACY_ARTIFACTS.get(name, name)
    return f'{prefix}/{name}{colon}{alias}' if prefix else f'{name}{colon}{alias}'


def upgrade_config(value):
    """Rewrite only path- and artifact-valued strings; protocol values and prose are untouched."""
    if isinstance(value, dict):
        return {key: upgrade_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [upgrade_config(item) for item in value]
    if isinstance(value, str):
        return current_artifact(current_path(value)) if '/' in value else value
    return value


def snapshot(root, path):
    """A run's frozen copy of a repository file, under its current or its pre-reorganization name."""
    current = Path(root) / 'source' / current_path(path)
    return current if current.exists() else Path(root) / 'source' / legacy_path(current_path(path))
