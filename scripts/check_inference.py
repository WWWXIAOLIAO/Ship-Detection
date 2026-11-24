"""Check custom score preservation against native Ultralytics prediction."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from ship_detection.core import ScorePredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device.startswith(("cpu", "cuda")) else f"cuda:{args.device}"
    )
    torch.set_num_threads(2)
    options = dict(
        device=device, imgsz=640, rect=False, conf=0.001, iou=0.7, max_det=100, verbose=False
    )
    native = YOLO(str(args.weights)).predict(str(args.image), **options)[0]
    custom = YOLO(str(args.weights)).predict(str(args.image), predictor=ScorePredictor, **options)[
        0
    ]
    torch.testing.assert_close(native.boxes.data, custom.boxes.data, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        custom.class_scores.max(1).values, custom.boxes.conf, atol=1e-5, rtol=1e-5
    )
    print("PASS: native predictions match; complete class scores retain the selected confidences")


if __name__ == "__main__":
    main()
