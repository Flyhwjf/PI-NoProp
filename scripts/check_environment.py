"""Fail early when PI-NoProp is launched from the wrong Python environment."""
import json
import os
import platform
import sys

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        'PyTorch is not installed in this interpreter. Use the maclearn environment:\n'
        'conda run -n maclearn python scripts/run_fast.py --help'
    ) from exc

report = {
    'python': sys.executable,
    'python_version': platform.python_version(),
    'torch': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_runtime': torch.version.cuda,
    'cpu_count': os.cpu_count(),
}
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    report.update(gpu=torch.cuda.get_device_name(0),
                  vram_gb=round(properties.total_memory / 1024**3, 2),
                  compute_capability=f'{properties.major}.{properties.minor}')
print(json.dumps(report, indent=2))
