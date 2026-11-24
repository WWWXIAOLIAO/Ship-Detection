"""Content-based signatures prevent silently reusing stale predictions."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_signature(weights, annotations, images, text_config):
    image_hashes = [[Path(p).name, file_sha256(p)] for p in images]
    content = {
        "schema": 1,
        "weights": file_sha256(weights),
        "annotations": file_sha256(annotations),
        "images": image_hashes,
        "text": text_config,
        "inference": {
            "imgsz": 640,
            "batch": 16,
            "rect": False,
            "half": False,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 100,
            "multi_label": False,
        },
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
