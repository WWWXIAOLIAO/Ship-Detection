"""Train one reproducible image-only detector; never accesses test images."""

import argparse
import json
import os
import time

from ship_detection.paths import ROOT

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("WANDB_DISABLED", "true")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="n", choices=["n", "s"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    name = "smoke" if args.smoke else f"yolo11{args.model}_seed{args.seed}"
    run_dir = ROOT / "runs" / name
    if run_dir.exists() and not args.resume:
        p.error(f"Run already exists: {run_dir}. Use --resume or move the existing run.")
    started = time.time()
    if args.resume:
        checkpoint = run_dir / "weights/last.pt"
        if not checkpoint.exists():
            p.error(f"Resume checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True, device=args.device, workers=args.workers)
        (run_dir / "wall_time.json").write_text(
            json.dumps(
                {
                    "seconds": time.time() - started,
                    "gpu": args.device,
                    "seed": args.seed,
                    "resumed": True,
                }
            )
        )
        return
    model = YOLO(str(ROOT / "weights" / f"yolo11{args.model}.pt"))
    model.train(
        data=str(ROOT / "configs/dataset.yaml"),
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=False,
        epochs=args.epochs,
        patience=25,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        close_mosaic=10,
        cache="ram",
        amp=True,
        plots=True,
        save=True,
        save_period=-1,
        fraction=0.1 if args.smoke else 1.0,
    )
    (model.trainer.save_dir / "wall_time.json").write_text(
        json.dumps({"seconds": time.time() - started, "gpu": args.device, "seed": args.seed})
    )


if __name__ == "__main__":
    main()
