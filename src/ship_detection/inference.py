"""Predict on an arbitrary image without reading annotations or filename labels."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .core import ROOT, ScorePredictor, TextScorer, dump, fuse_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--weights", type=Path, required=True, help="Fine-tuned 17-class checkpoint")
    p.add_argument("--mode", choices=["baseline", "names", "descriptions"], default="baseline")
    p.add_argument("--config", type=Path, help="JSON containing validation-selected alpha values")
    p.add_argument("--knowledge", type=Path, default=ROOT / "data/raw/ship_knowledge.yaml")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--text-device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/predictions")
    args = p.parse_args()
    for path in [args.image, args.weights]:
        if not path.is_file():
            p.error(f"File not found: {path}")
    alpha = args.alpha
    if alpha is None and args.config:
        alpha = json.loads(args.config.read_text())["alpha"].get(args.mode)
    if args.mode == "baseline":
        alpha = 0.0
    elif alpha is None:
        p.error("Text modes require --alpha or a validation-selected --config")
    if not 0 <= alpha <= 1 or not 0 <= args.conf <= 1:
        p.error("--alpha and --conf must be in [0, 1]")
    if args.mode != "baseline" and alpha > 0 and not args.knowledge.is_file():
        p.error(f"Text knowledge not found: {args.knowledge}. Prepare the instructor data first.")
    device = torch.device(
        args.device if args.device.startswith(("cuda", "cpu")) else f"cuda:{args.device}"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; use --device cpu or install a compatible GPU PyTorch build")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = YOLO(str(args.weights))
    if len(model.names) != 17:
        p.error("Expected a fine-tuned 17-class checkpoint, not generic COCO weights")
    r = model.predict(
        str(args.image),
        predictor=ScorePredictor,
        device=device,
        conf=args.conf,
        iou=0.7,
        max_det=100,
        imgsz=640,
        rect=False,
        verbose=False,
    )[0]
    data = {
        "classes": r.boxes.cls.cpu().numpy().astype(int),
        "visual": r.class_scores.cpu().numpy(),
    }
    boxes = r.boxes.xyxy.cpu().numpy()
    image = Image.fromarray(r.orig_img[:, :, ::-1])
    if args.mode != "baseline" and alpha > 0 and len(boxes):
        scorer = TextScorer(args.text_device or str(device), knowledge_path=args.knowledge)
        scores = scorer.score([scorer.crop(image, b) for b in boxes])
        data.update(scores)
    cls = fuse_labels(data, args.mode, alpha)
    draw = ImageDraw.Draw(image)
    font_size = max(13, int(image.width / 75))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    predictions = []
    for box, c, confidence in zip(boxes, cls, r.boxes.conf.cpu().numpy()):
        box = list(map(float, box))
        name = r.names[int(c)]
        draw.rectangle(box, outline=(255, 80, 20), width=max(2, int(image.width / 500)))
        # Short readable name on the image; exact canonical name in JSON.
        text = f"{name.split('_')[-1]} {confidence:.2f}"
        x, y = box[:2]
        draw.text(
            (x, max(0, y - font_size - 5)),
            text,
            font=font,
            fill=(255, 80, 20),
            stroke_width=1,
            stroke_fill="white",
        )
        predictions.append(
            {"class_id": int(c), "class_name": name, "confidence": float(confidence), "xyxy": box}
        )
    image.save(args.output / (args.image.stem + "_prediction.jpg"))
    dump(
        {
            "image": str(args.image),
            "weights": str(args.weights),
            "mode": args.mode,
            "alpha": alpha,
            "predictions": predictions,
        },
        args.output / (args.image.stem + "_prediction.json"),
    )
    print(json.dumps(predictions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
