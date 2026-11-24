"""Synchronized batch-one timing including image decode, crop and inference."""

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from ship_detection.core import ROOT, ScorePredictor, TextScorer, dump, fuse_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0")
    p.add_argument("--text-device", default="cuda:0")
    p.add_argument("--images", type=int, default=30)
    args = p.parse_args()
    torch.set_num_threads(2)
    sel = json.loads((ROOT / "artifacts/frozen_selection.json").read_text())
    seed = sel.get("seeds", [42])[0]
    name = f"yolo11{sel['selected_size']}_seed{seed}"
    model = YOLO(str(ROOT / f"runs/{name}/weights/best.pt"))
    scorer = TextScorer(args.text_device)
    ann = json.loads((ROOT / "data/processed/annotations/instances_test.json").read_text())
    paths = [
        ROOT / "data/processed/images/test" / im["file_name"] for im in ann["images"][: args.images]
    ]
    results = {}
    for mode in ["baseline", "descriptions_fixed025"]:
        times = []
        objects = []
        for i, path in enumerate(paths[:5] + paths):
            torch.cuda.synchronize(int(args.device))
            torch.cuda.synchronize(scorer.device)
            start = time.perf_counter()
            # Decode is part of the measurement, model construction is not.
            image = Image.open(path).convert("RGB")
            r = model.predict(
                image,
                predictor=ScorePredictor,
                device=torch.device(f"cuda:{args.device}"),
                imgsz=640,
                rect=False,
                conf=0.25,
                iou=0.7,
                max_det=100,
                verbose=False,
            )[0]
            if mode != "baseline" and len(r.boxes):
                scores = scorer.score([scorer.crop(image, b) for b in r.boxes.xyxy.cpu().numpy()])
                data = {
                    "visual": r.class_scores.cpu().numpy(),
                    "classes": r.boxes.cls.cpu().numpy().astype(int),
                    **scores,
                }
                fuse_labels(data, "descriptions", 0.25)
            torch.cuda.synchronize(int(args.device))
            torch.cuda.synchronize(scorer.device)
            elapsed = (time.perf_counter() - start) * 1000
            if i >= 5:
                times.append(elapsed)
                objects.append(len(r.boxes))
        results[mode] = {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "images": len(times),
            "mean_predicted_objects": float(np.mean(objects)),
        }
    dump(
        {
            "run": name,
            "detector_gpu": args.device,
            "text_gpu": args.text_device,
            "batch": 1,
            "conf": 0.25,
            "warmup_images": 5,
            "includes": "JPEG decode, detector preprocess/forward/NMS, crop processing and CLIP encoding when enabled; excludes model loading, output rendering and disk saving; warm file-system cache; hardware depends on --device and --text-device; same GPU allowed",
            "results": results,
        },
        ROOT / "artifacts/benchmark.json",
    )
    print(results, flush=True)


if __name__ == "__main__":
    main()
