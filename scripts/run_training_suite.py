"""Run model/seed experiments in a queue, with one process per requested GPU."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ship_detection.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["0"], help="e.g. 0 or 0 1 2 3 4 5")
    parser.add_argument("--models", nargs="+", choices=["n", "s"], default=["n", "s"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices):
        parser.error("Each GPU must appear only once in --devices")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        parser.error("Models and seeds must not contain duplicates")
    import yaml
    from ultralytics.data.dataset import YOLODataset

    data = yaml.safe_load((ROOT / "configs/dataset.yaml").read_text())
    for split in ["train", "val"]:
        ds = YOLODataset(
            img_path=str(Path(data["path"]) / data[split]), data=data, imgsz=640, cache=False
        )
        print("Validated label cache", split, len(ds), flush=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    queue = [(kind, seed) for kind in args.models for seed in args.seeds]
    active = {}
    status = {}
    try:
        while queue or active:
            for device in args.devices:
                if device in active or not queue:
                    continue
                kind, seed = queue.pop(0)
                name = f"yolo11{kind}_seed{seed}"
                run = ROOT / "runs" / name
                if (run / "weights/best.pt").exists() and (run / "wall_time.json").exists():
                    status[name] = 0
                    print("Already completed", name, flush=True)
                    continue
                if run.exists():
                    raise RuntimeError(
                        f"Incomplete run: {run}. Resume it with train_detector.py --resume first."
                    )
                log = (ROOT / "logs" / f"{name}.log").open("w")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve().with_name("train_detector.py")),
                    "--model",
                    kind,
                    "--seed",
                    str(seed),
                    "--device",
                    device,
                    "--epochs",
                    str(args.epochs),
                    "--batch",
                    str(args.batch),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                )
                active[device] = (name, process, log)
                print(f"Started {name} on GPU {device}, pid={process.pid}", flush=True)
            for device, (name, process, log) in list(active.items()):
                if process.poll() is not None:
                    status[name] = process.returncode
                    log.close()
                    del active[device]
                    print("Finished", name, process.returncode, flush=True)
            (ROOT / "artifacts/training_status.json").write_text(json.dumps(status, indent=2))
            if active:
                time.sleep(5)
    finally:
        for _, process, log in active.values():
            process.terminate()
        for _, process, log in active.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
    if any(status.values()):
        raise SystemExit("Training failed; inspect logs and artifacts/training_status.json")


if __name__ == "__main__":
    main()
