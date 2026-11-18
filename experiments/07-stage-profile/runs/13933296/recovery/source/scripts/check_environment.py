"""Run inside a Slurm GPU allocation after installing the candidate environment."""
import importlib.metadata
import json
import platform

import torch

packages = [
    'torch', 'torchvision', 'transformers', 'accelerate', 'peft',
    'bitsandbytes', 'Pillow', 'ImageHash', 'numpy', 'scikit-learn', 'wandb',
]
versions = {}
for name in packages:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None

print(json.dumps({
    'python': platform.python_version(),
    'packages': versions,
    'torch_cuda_build': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
}, indent=2))
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: check GPU allocation and PyTorch build.')
if any(value is None for value in versions.values()):
    raise SystemExit('Required packages are missing; see inventory above.')

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(json.dumps({
        'device': index,
        'name': props.name,
        'compute_capability': [props.major, props.minor],
        'total_memory_bytes': props.total_memory,
    }))

# This checks CUDA FP16 execution, not model/LoRA/quantization compatibility.
x = torch.ones((64, 64), device='cuda', dtype=torch.float16)
y = x @ x
assert torch.all(y == 64).item(), 'FP16 matrix multiplication failed'
torch.cuda.synchronize()
print('FP16 CUDA smoke check passed. Model, merge and quantization checks remain.')
