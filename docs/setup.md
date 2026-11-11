# GitHub and Insomnia setup

Checked November 11, 2026. GitHub CLI authenticated successfully as `oowenn` when network access was allowed. Owen will create the remote repository. Insomnia became reachable after VPN connection; SSH reached the UNI password prompt, but authentication, Slurm authorization, and GPU execution are not yet verified.

## GitHub

Suggested name: **smolmri**. Alternatives: `mri-vlm` (more general), `vlm-prof` (emphasizes profiling). A private repository during development is a reasonable default; ensure instructors can access the final code and tracking record.

Create an empty repository without GitHub-generated README/license/gitignore so it can accept this folder's files. Choose a license with the team before public release; dataset and model licenses remain separate.

After local Git initialization and an initial commit, connect the actual remote (replace YOUR_REPO with the chosen name):

```bash
git remote add origin https://github.com/oowenn/YOUR_REPO.git
git push -u origin main
```

Invite the other three members using their confirmed GitHub handles. Use short feature branches and reviewed pull requests; retain individual author history for the contribution statement. Add setup/reproduction commands to the README as implementations land. Keep credentials, dataset images, checkpoints and run artifacts out of commits.

## Insomnia login

The official [getting-started guide](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145140/Insomnia%3A%2BGetting%2BStarted) requires approved access, Duo and an allowed secure campus network or Columbia/CUIMC VPN. CourseWorks instructions determine the course-specific allocation; do not substitute an arbitrary group account or assume free-tier GPU access.

In your own Terminal, run:

```bash
ssh as8254@insomnia.rcs.columbia.edu
```

Enter passwords and approve Duo locally; do not paste credentials into chat, files, or commands. The optional `cluster/ssh_config.example` provides an alias, but has not been installed into your personal SSH config. Each teammate must use their own UNI and approved account.

The cluster currently asks for a password with this machine's existing SSH credentials. To let automation reuse a manually authenticated connection for this setup session, optionally use a short-lived multiplexed session (private socket directory):

```bash
mkdir -p ~/.ssh/control
chmod 700 ~/.ssh/control
ssh -M -S ~/.ssh/control/insomnia-%r@%h:%p -o ControlPersist=10m as8254@insomnia.rcs.columbia.edu
```

Keep that session open, complete password/Duo in Terminal, and tell the assistant it is ready. A subsequent SSH client can use the same `-S` socket path without receiving your password. Close the master when done:

```bash
ssh -S ~/.ssh/control/insomnia-%r@%h:%p -O exit as8254@insomnia.rcs.columbia.edu
```

## Inspect access before allocating resources

Run these lightweight commands on the login node (or let the assistant run them through the authenticated connection):

```bash
id
sacctmgr -nP show assoc where user="$USER" format=Account,Partition,QOS
sinfo -o '%P %a %l %G %f'
module avail
pwd
df -h .
```

If accounting queries are restricted, use CourseWorks/the support team to identify the account and QOS. Confirm a project storage path, quota/retention policy, GPU type/GRES spelling, partition, permitted CPU/memory/walltime requests and whether the GPU is shared. These are not yet known. Use the live scheduler as the source of truth; documentation lists several GPU types, not a guarantee of course access.

Do not run training, inference, installs with substantial compilation, or dataset deduplication on the login node. Use Slurm compute allocations per the [job submission guide](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145134/Insomnia%2B-%2BSubmitting%2BJobs).

## Environment and storage

1. Pick a writable project location and create separate data, checkpoints, cache and output directories; confirm quota first. Allow headroom for dataset archives, expanded images, HF caches, LoRA checkpoints and traces. Measure actual use before setting a storage budget.
2. Clone the repo there. Find supported Python/CUDA modules using `module avail`; do not guess a module name. A compatible Python 3.11 environment is a provisional starting point.
3. In an appropriate allocation, create a venv and install a matched CUDA PyTorch/torchvision build using the [official PyTorch selector](https://pytorch.org/get-started/locally/) and the GPU driver's compatibility. Install the other dependencies from `requirements.in`. That inventory is not a lockfile.
4. Set `HF_HOME` and dataset/output paths to appropriate cluster storage. Authenticate W&B/Kaggle/HF locally only where needed. Never put tokens in Slurm scripts or job logs; offline W&B can be synced if compute-node internet is unavailable.
5. Verify model load/inference, one tiny LoRA update and merge consistency, then INT8/INT4 model load/inference. Only then freeze package versions (`python -m pip freeze > requirements.lock.txt`), Python version, modules, model revision and driver/CUDA metadata. Preserve the installation commands, including the PyTorch package source.

Hugging Face's [SmolVLM model card](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) and [bitsandbytes compatibility guide](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) are starting points; compatibility must be tested on the allocated GPU with the chosen package versions. No large model or dataset downloads have been started during preparation.

## GPU environment smoke job

`cluster/smoke.sbatch` is intentionally missing allocation-specific account/partition/GPU directives. It expects an existing venv. Its 4 CPUs / 32 GB RAM / 10 minutes are provisional smoke-test requests. Replace all uppercase allocation placeholders using verified values; GPU syntax may differ for the course.

From the repo on Insomnia, after loading the chosen modules:

```bash
mkdir -p logs
export SMOLMRI_VENV=/absolute/path/to/venv
sbatch --account=COURSE_ACCOUNT --partition=COURSE_PARTITION --gres=gpu:VERIFIED_TYPE:1 cluster/smoke.sbatch
squeue -u "$USER"
```

Choose the same GPU type for future matched experiments. `scripts/check_environment.py` prints versions/GPU properties and checks a small FP16 CUDA matrix multiplication. It does not download models, prove quantization support, or train. A successful smoke run is the first readiness gate; model and data checks follow.

## Pending

- [ ] Owen creates GitHub repo and supplies its URL; connect/push local project.
- [ ] Confirm collaborators and reviewer access.
- [ ] Complete password/Duo authentication locally.
- [ ] Verify Slurm account, partition, QOS, GPU type and storage.
- [ ] Resolve and freeze the environment after compatibility checks.
- [ ] Pass a GPU smoke job, then model/merge/quantization smoke checks.
- [ ] Create shared W&B project and dataset manifests.
