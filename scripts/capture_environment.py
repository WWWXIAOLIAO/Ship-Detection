"""Record the runtime and installed packages for a new reproduction run."""

import json
import platform
import subprocess
import sys

from ship_detection.paths import ROOT


def main():
    import open_clip
    import torch
    import torchvision
    import ultralytics

    out = ROOT / "artifacts/environment"
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "open_clip": open_clip.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    (out / "runtime.json").write_text(json.dumps(info, indent=2))
    (out / "pip_freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    smi = subprocess.run(["nvidia-smi"], text=True, capture_output=True)
    (out / "nvidia_smi.txt").write_text(smi.stdout + smi.stderr)
    print(info)


if __name__ == "__main__":
    main()
