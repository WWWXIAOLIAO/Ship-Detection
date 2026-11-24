"""Shared image/text inference and evaluation. No ground truth enters inference."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .paths import ROOT

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops

from .cache import prediction_signature
from .constants import EN_NAMES
from .metrics import evaluate, fuse_labels, iou_matrix, operating_metrics, to_coco


def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class ScorePredictor(DetectionPredictor):
    """Keep the original 17 sigmoid class scores at each retained YOLO anchor."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        raw = preds[0] if isinstance(preds, (tuple, list)) else preds
        class_scores = raw[:, 4 : 4 + len(self.model.names), :].transpose(1, 2).clone()
        filtered, indices = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            return_idxs=True,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = self.construct_results(filtered, img, orig_imgs)
        for k, (result, idx) in enumerate(zip(results, indices)):
            result.class_scores = class_scores[k, idx.view(-1).long()]
            if len(result.boxes):
                assert torch.allclose(
                    result.class_scores.max(1).values.float(), result.boxes.conf.float(), atol=2e-4
                )
        return results


class TextScorer:
    def __init__(self, device="cuda:0", knowledge_path=None):
        import open_clip

        torch.set_num_threads(2)
        self.device = device
        weight = ROOT / "weights/clip/ViT-B-32.pt"
        if not weight.exists():
            from open_clip.pretrained import download_pretrained

            weight = Path(
                download_pretrained(
                    open_clip.get_pretrained_cfg("ViT-B-32", "openai"),
                    prefer_hf_hub=False,
                    cache_dir=str(weight.parent),
                )
            )
        # Official OpenAI files are TorchScript. Convert a hash-verified archive
        # once so current PyTorch can use its default weights-only loader.
        state_weight = weight.with_name("ViT-B-32-state.pt")
        if not state_weight.exists():
            expected = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
            assert hashlib.sha256(weight.read_bytes()).hexdigest() == expected
            state = torch.jit.load(str(weight), map_location="cpu").state_dict()
            for key in ["input_resolution", "context_length", "vocab_size"]:
                state.pop(key, None)
            torch.save(state, state_weight)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=str(state_weight),
            force_quick_gelu=True,
            device=device,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        knowledge_path = (
            Path(knowledge_path)
            if knowledge_path is not None
            else ROOT / "data/raw/ship_knowledge.yaml"
        )
        knowledge = yaml.safe_load(knowledge_path.read_text())
        if len(knowledge["classes"]) != 17:
            raise ValueError("Expected 17 classes in instructor class order")
        self.prompts = {
            "names": [f"a photo of a {n}." for n in EN_NAMES],
            "descriptions": [
                f"a photo of a {n}, {c['prompt']}."
                for n, c in zip(EN_NAMES, knowledge["classes"][:16])
            ],
        }
        # Never silently truncate prompt tokens.
        from open_clip.tokenizer import SimpleTokenizer

        tok = SimpleTokenizer()
        self.token_counts = {
            k: [len(tok.encode(t)) + 2 for t in texts] for k, texts in self.prompts.items()
        }
        assert max(max(v) for v in self.token_counts.values()) <= 77
        self.text = {}
        with torch.inference_mode():
            for key, texts in self.prompts.items():
                feat = self.model.encode_text(self.tokenizer(texts).to(device))
                self.text[key] = torch.nn.functional.normalize(feat.float(), dim=-1)
        self.scale = float(self.model.logit_scale.exp().item())
        dump(
            {
                "model": "OpenAI CLIP ViT-B/32",
                "weights": str(weight),
                "frozen": True,
                "logit_scale": self.scale,
                "prompts": self.prompts,
                "token_counts": self.token_counts,
                "crop": "5% context, gray square padding, official normalization",
            },
            ROOT / "artifacts/text_config.json",
        )

    def crop(self, image, box):
        x1, y1, x2, y2 = map(float, box)
        dx, dy = (x2 - x1) * 0.05, (y2 - y1) * 0.05
        x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
        x2, y2 = min(image.width, int(np.ceil(x2 + dx))), min(image.height, int(np.ceil(y2 + dy)))
        crop = image.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
        side = max(crop.size)
        square = Image.new("RGB", (side, side), (127, 127, 127))
        square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        return self.preprocess(square)

    def score(self, crops, batch=128):
        output = {key: [] for key in self.text}
        with torch.inference_mode():
            for start in range(0, len(crops), batch):
                x = torch.stack(crops[start : start + batch]).to(self.device)
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=torch.device(self.device).type == "cuda"
                ):
                    f = self.model.encode_image(x)
                f = torch.nn.functional.normalize(f.float(), dim=-1)
                for key, tf in self.text.items():
                    output[key].append((self.scale * f @ tf.T).cpu().numpy())
        return {
            key: np.concatenate(v) if v else np.empty((0, 16), dtype=np.float32)
            for key, v in output.items()
        }


def prediction_cache(weights, split, device, output, text_scorer=None):
    """Save one record per predicted box, including full visual scores and CLIP logits."""
    output = Path(output)
    ann = json.loads((ROOT / f"data/processed/annotations/instances_{split}.json").read_text())
    images = ann["images"]
    image_id = {im["file_name"]: im["id"] for im in images}
    paths = [str(ROOT / "data/processed/images" / split / im["file_name"]) for im in images]
    text_config = (
        None
        if text_scorer is None
        else {
            "prompts": text_scorer.prompts,
            "scale": text_scorer.scale,
            "crop_margin": 0.05,
            "model": "OpenAI-ViT-B-32-40d36571",
        }
    )
    signature = prediction_signature(
        weights, ROOT / f"data/processed/annotations/instances_{split}.json", paths, text_config
    )
    meta_path = output.with_suffix(".meta.json")
    if output.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("signature") == signature:
            with np.load(output, allow_pickle=False) as saved:
                return {key: saved[key] for key in saved.files}
        print(f"Rebuilding stale cache: {output.name}", flush=True)
    model = YOLO(str(weights))
    model.predictor = None
    ids, boxes, confidence, classes, visual = [], [], [], [], []
    names_logits, desc_logits = [], []
    detector_ms, clip_ms = [], []
    start = time.perf_counter()
    # A torch.device bypasses Ultralytics' string-based CUDA_VISIBLE_DEVICES
    # remapping, which is unsafe after a second model initialized CUDA already.
    detector_device = torch.device(
        device if str(device).startswith(("cuda", "cpu")) else f"cuda:{device}"
    )
    gen = model.predict(
        source=paths,
        predictor=ScorePredictor,
        stream=True,
        device=detector_device,
        imgsz=640,
        batch=16,
        rect=False,
        conf=0.001,
        iou=0.7,
        max_det=100,
        half=False,
        verbose=False,
        save=False,
    )
    # Keep crop batches bounded. The generator's image buffers are not retained.
    pending = []

    def flush():
        if pending and text_scorer is not None:
            if torch.device(text_scorer.device).type == "cuda":
                torch.cuda.synchronize(text_scorer.device)
            t = time.perf_counter()
            s = text_scorer.score(pending)

            if torch.device(text_scorer.device).type == "cuda":
                torch.cuda.synchronize(text_scorer.device)
            clip_ms.append((time.perf_counter() - t) * 1000)
            names_logits.append(s["names"])
            desc_logits.append(s["descriptions"])
            pending.clear()

    for index, r in enumerate(gen):
        bb = r.boxes.xyxy.cpu().numpy()
        count = len(bb)
        ids.extend([image_id[Path(r.path).name]] * count)
        boxes.extend(bb.tolist())
        confidence.extend(r.boxes.conf.cpu().numpy().tolist())
        classes.extend(r.boxes.cls.cpu().numpy().astype(int).tolist())
        visual.extend(r.class_scores.cpu().numpy().tolist())
        detector_ms.append(sum(r.speed.values()))
        if text_scorer is not None:
            image = Image.fromarray(r.orig_img[:, :, ::-1])
            for box in bb:
                pending.append(text_scorer.crop(image, box))
                if len(pending) >= 128:
                    flush()
        if (index + 1) % 50 == 0:
            print(
                f"{Path(weights).parents[1].name} {split}: {index + 1}/{len(images)} images, {len(ids)} boxes",
                flush=True,
            )
    flush()
    data = {
        "image_id": np.asarray(ids, np.int32),
        "boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
        "confidence": np.asarray(confidence, np.float32),
        "classes": np.asarray(classes, np.int32),
        "visual": np.asarray(visual, np.float32).reshape(-1, 17),
    }
    if text_scorer is not None:
        data["names"] = (
            np.concatenate(names_logits) if names_logits else np.empty((0, 16), np.float32)
        )
        data["descriptions"] = (
            np.concatenate(desc_logits) if desc_logits else np.empty((0, 16), np.float32)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    dump(
        {
            "signature": signature,
            "weights": str(weights),
            "split": split,
            "images": len(images),
            "boxes": len(ids),
            "detector_ms_per_image": float(np.mean(detector_ms)),
            "clip_encoder_ms_total": sum(clip_ms),
            "wall_seconds": time.perf_counter() - start,
            "device_requested": str(device),
            "device": str(model.predictor.device),
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 100,
            "timing_note": "Detector preprocessing+forward+NMS, batched; CLIP encoder timing excludes crop preparation; wall time includes startup, IO and crop processing.",
        },
        output.with_suffix(".meta.json"),
    )
    return data
