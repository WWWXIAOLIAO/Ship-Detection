"""Fusion and COCO evaluation, independent of model loading."""

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from .constants import SHORT_NAMES
from .paths import ROOT


def fuse_labels(data, mode="baseline", alpha=0.0):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if mode not in ["baseline", "names", "descriptions"]:
        raise ValueError(f"Unknown fusion mode: {mode}")
    classes = data["classes"].copy()
    if mode == "baseline" or alpha == 0:
        return classes
    assert mode in ["names", "descriptions"]
    eligible = classes != 16  # Preserve the detector's other_ship decision.
    visual = np.log(np.maximum(data["visual"][eligible, :16], 1e-10))
    text = data[mode][eligible]
    # Additive log-score fusion; subtracting row constants does not affect argmax.
    classes[eligible] = ((1 - alpha) * visual + alpha * text).argmax(axis=1)
    return classes


def to_coco(data, mode="baseline", alpha=0.0):
    cls = fuse_labels(data, mode, alpha)
    boxes = data["boxes"]
    return [
        {
            "image_id": int(im),
            "category_id": int(c) + 1,
            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
            "score": float(s),
        }
        for im, c, b, s in zip(data["image_id"], cls, boxes, data["confidence"])
    ]


def iou_matrix(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    area = np.maximum(right - left, 0).prod(-1)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    ba = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    return area / np.maximum(aa[:, None] + ba[None, :] - area, 1e-10)


def operating_metrics(ann, predictions, threshold=0.25):
    """IoU-first, class-agnostic one-to-one matching for classification diagnostics.

    Confusion rows=GT, columns=prediction, index17=background/missed.
    P/R require class correctness and IoU>=.5 at fixed confidence .25.
    """
    gt = {im["id"]: [] for im in ann["images"]}
    pred = {im["id"]: [] for im in ann["images"]}
    for a in ann["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append(([x, y, x + w, y + h], a["category_id"] - 1))
    for a in predictions:
        if a["score"] >= threshold:
            x, y, w, h = a["bbox"]
            pred[a["image_id"]].append(([x, y, x + w, y + h], a["category_id"] - 1))
    cm = np.zeros((18, 18), dtype=int)
    for im in gt:
        gg, pp = gt[im], pred[im]
        mat = iou_matrix([x[0] for x in gg], [x[0] for x in pp])
        candidates = sorted(
            [(float(mat[i, j]), i, j) for i, j in zip(*np.where(mat >= 0.5))], reverse=True
        )
        ug, up = set(), set()
        for _, i, j in candidates:
            if i not in ug and j not in up:
                ug.add(i)
                up.add(j)
                cm[gg[i][1], pp[j][1]] += 1
        for i in range(len(gg)):
            if i not in ug:
                cm[gg[i][1], 17] += 1
        for j in range(len(pp)):
            if j not in up:
                cm[17, pp[j][1]] += 1
    tp = int(np.trace(cm[:17, :17]))
    ng = int(cm[:17, :].sum())
    npred = int(cm[:, :17].sum())
    matched = int(cm[:17, :17].sum())
    return {
        "precision_at_conf025": tp / max(npred, 1),
        "recall_at_conf025": tp / max(ng, 1),
        "matched_class_accuracy": tp / max(matched, 1),
        "matched_objects": matched,
        "localization_recall": matched / max(ng, 1),
        "confusion_matrix": cm.tolist(),
        "operating_threshold": threshold,
    }


def evaluate(data, split, mode="baseline", alpha=0.0, details=False, annotations_path=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_path = (
        Path(annotations_path)
        if annotations_path is not None
        else ROOT / f"data/processed/annotations/instances_{split}.json"
    )
    predictions = to_coco(data, mode, alpha)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_path))
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {
                "images": gt.dataset["images"],
                "categories": gt.dataset["categories"],
                "annotations": [],
            }
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    result = {
        "map": float(ev.stats[0]),
        "map50": float(ev.stats[1]),
        "map75": float(ev.stats[2]),
        "ar100": float(ev.stats[8]),
        "mode": mode,
        "alpha": alpha,
    }
    if details:
        precision = ev.eval["precision"]
        perclass = []
        for c in range(17):
            p = precision[:, :, c, 0, -1]
            p50 = precision[0, :, c, 0, -1]
            perclass.append(
                {
                    "class_id": c,
                    "name": SHORT_NAMES[c],
                    "ap": float(p[p >= 0].mean()) if np.any(p >= 0) else None,
                    "ap50": float(p50[p50 >= 0].mean()) if np.any(p50 >= 0) else None,
                }
            )
        result["per_class"] = perclass
        result.update(operating_metrics(json.loads(ann_path.read_text()), predictions))
    return result
