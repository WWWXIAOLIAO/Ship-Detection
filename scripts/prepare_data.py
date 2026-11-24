"""Extract and audit the instructor archive, then create a leakage-aware split.

Raw files are immutable. Canonical COCO and YOLO outputs share clipped boxes.
The instructor's val partition is reserved for final evaluation (test here).
"""

from __future__ import annotations

import argparse
import collections
import ctypes as C
import ctypes.util
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

from ship_detection.paths import ROOT


def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def extract(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".extraction_complete"
    if marker.exists():
        return
    library = ctypes.util.find_library("archive")
    if library is None:
        raise RuntimeError(
            "libarchive is required to extract the instructor RAR; install the system library first"
        )
    lib = C.CDLL(library)
    lib.archive_read_new.restype = C.c_void_p
    for name in [
        "archive_read_support_format_all",
        "archive_read_support_filter_all",
        "archive_read_free",
    ]:
        getattr(lib, name).argtypes = [C.c_void_p]
    lib.archive_read_open_filename.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
    lib.archive_read_next_header.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
    lib.archive_entry_pathname.argtypes = [C.c_void_p]
    lib.archive_entry_pathname.restype = C.c_char_p
    lib.archive_entry_filetype.argtypes = [C.c_void_p]
    lib.archive_entry_filetype.restype = C.c_uint
    lib.archive_read_data.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    lib.archive_read_data.restype = C.c_ssize_t
    lib.archive_error_string.argtypes = [C.c_void_p]
    lib.archive_error_string.restype = C.c_char_p
    a = lib.archive_read_new()
    lib.archive_read_support_format_all(a)
    lib.archive_read_support_filter_all(a)
    if lib.archive_read_open_filename(a, str(archive).encode(), 1024 * 1024) != 0:
        raise RuntimeError(lib.archive_error_string(a))
    entry = C.c_void_p()
    buf = C.create_string_buffer(1024 * 1024)
    count = 0
    while True:
        status = lib.archive_read_next_header(a, C.byref(entry))
        if status == 1:
            break
        if status < 0:
            raise RuntimeError(lib.archive_error_string(a))
        parts = Path(lib.archive_entry_pathname(entry).decode("utf-8")).parts
        if ".." in parts or not parts or Path(*parts).is_absolute():
            raise ValueError(parts)
        rel = Path(*parts[1:])
        dest = output / rel
        kind = lib.archive_entry_filetype(entry)
        if kind == 0o040000:
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if kind != 0o100000:
            raise ValueError((rel, kind))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            while True:
                n = lib.archive_read_data(a, buf, len(buf))
                if n == 0:
                    break
                if n < 0:
                    raise RuntimeError(lib.archive_error_string(a))
                f.write(buf.raw[:n])
        count += 1
    lib.archive_read_free(a)
    marker.write_text(str(count))
    print(f"Extracted {count} files", flush=True)


def image_hashes(path):
    with Image.open(path) as im:
        im.load()
        size = im.size
        gray = np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32)
        coeff = cv2.dct(gray)[:8, :8].ravel()[1:]
        ph = sum(int(x) << k for k, x in enumerate(coeff > np.median(coeff)))
        small = np.asarray(im.convert("L").resize((9, 8)))
        dh = sum(int(x) << k for k, x in enumerate((small[:, 1:] > small[:, :-1]).ravel()))
    return size, hashlib.sha256(path.read_bytes()).hexdigest(), ph, dh


def prepare(seed=42):
    raw = ROOT / "data/raw"
    processed = ROOT / "data/processed"
    artifacts = ROOT / "artifacts/data"
    artifacts.mkdir(parents=True, exist_ok=True)
    knowledge = yaml.safe_load((raw / "ship_knowledge.yaml").read_text())
    names = [c["name"] for c in knowledge["classes"]]
    records, corrections, mismatch = [], [], []
    for source in ["train", "val"]:
        coco = json.loads((raw / "annotations" / f"instances_{source}.json").read_text())
        assert [x["name"] for x in coco["categories"]] == names
        anns = collections.defaultdict(list)
        for ann in coco["annotations"]:
            assert 1 <= ann["category_id"] <= len(names)
            anns[ann["image_id"]].append(ann)
        assert set(anns) <= {x["id"] for x in coco["images"]}
        for im in coco["images"]:
            path = raw / "images" / source / im["file_name"]
            size, sha, ph, dh = image_hashes(path)
            assert size == (im["width"], im["height"]), (path, size, im)
            W, H = size
            boxes = []
            original_yolo = np.loadtxt(raw / "labels" / source / f"{path.stem}.txt", ndmin=2)
            expected = []
            for ann in anns[im["id"]]:
                x, y, w, h = map(float, ann["bbox"])
                expected.append(
                    [ann["category_id"] - 1, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H]
                )
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(W), x + w), min(float(H), y + h)
                assert x2 > x1 and y2 > y1, (path, ann)
                if min(x, y) < 0 or x + w > W or y + h > H:
                    corrections.append(
                        {
                            "file": im["file_name"],
                            "annotation_id": ann["id"],
                            "before": [x, y, w, h],
                            "after": [x1, y1, x2 - x1, y2 - y1],
                        }
                    )
                boxes.append([ann["category_id"] - 1, x1, y1, x2, y2])
            if original_yolo.shape != np.asarray(expected).shape or not np.allclose(
                original_yolo, expected, atol=2e-6
            ):
                mismatch.append(im["file_name"])
            records.append(
                {
                    "name": im["file_name"],
                    "source": source,
                    "width": W,
                    "height": H,
                    "sha256": sha,
                    "phash": ph,
                    "dhash": dh,
                    "boxes": boxes,
                }
            )
    assert not mismatch, mismatch
    assert len({r["name"] for r in records}) == len(records)
    # Conservative visual duplicate screen. Does not guarantee scene independence.
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i, r in enumerate(records):
        for j in range(i):
            s = records[j]
            exact = r["sha256"] == s["sha256"]
            pd = (r["phash"] ^ s["phash"]).bit_count()
            dd = (r["dhash"] ^ s["dhash"]).bit_count()
            aspect = abs(math.log((r["width"] / r["height"]) / (s["width"] / s["height"])))
            if exact or (pd <= 4 and dd <= 6 and aspect < 0.08):
                parent[find(i)] = find(j)
                pairs.append(
                    {
                        "a": r["name"],
                        "b": s["name"],
                        "a_source": r["source"],
                        "b_source": s["source"],
                        "exact": exact,
                        "phash_distance": pd,
                        "dhash_distance": dd,
                    }
                )
    groups = collections.defaultdict(list)
    for i, r in enumerate(records):
        r["group"] = find(i)
        groups[find(i)].append(i)
    excluded = []
    available_groups = []
    for gid, ids in groups.items():
        if any(records[i]["source"] == "val" for i in ids):
            for i in ids:
                r = records[i]
                r["split"] = "test" if r["source"] == "val" else "excluded"
                if r["split"] == "excluded":
                    excluded.append(r["name"])
        else:
            available_groups.append(gid)
    # Multi-label greedy group allocation with a minimum per-class training support.
    counts = collections.Counter(
        c for gid in available_groups for i in groups[gid] for c, *_ in records[i]["boxes"]
    )
    target = np.array([max(1, round(counts[c] * 0.15)) for c in range(len(names))])
    selected = np.zeros(len(names), dtype=int)
    remaining = np.array([counts[c] for c in range(len(names))])
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(available_groups))
    group_counts = {
        gid: np.bincount(
            [int(c) for i in groups[gid] for c, *_ in records[i]["boxes"]], minlength=len(names)
        )
        for gid in available_groups
    }
    chosen = set()
    while True:
        candidates = []
        for gid in order:
            if gid in chosen:
                continue
            gc = group_counts[gid]
            if np.any(
                remaining - gc < np.minimum(2, np.array([counts[c] for c in range(len(names))]))
            ):
                continue
            gain = (np.minimum(gc, np.maximum(target - selected, 0)) / np.maximum(target, 1)).sum()
            overshoot = (np.maximum(selected + gc - target, 0) / np.maximum(target, 1)).sum()
            if gain > 0:
                candidates.append((gain - 0.05 * overshoot, gid))
        if not candidates:
            break
        _, gid = max(candidates, key=lambda x: x[0])
        chosen.add(gid)
        selected += group_counts[gid]
        remaining -= group_counts[gid]
    for gid in available_groups:
        for i in groups[gid]:
            records[i]["split"] = "val" if gid in chosen else "train"
    # Remove only generated data, so a rerun with a different seed cannot leave
    # stale images in multiple partitions. Never run preparation during training.
    if processed.exists():
        shutil.rmtree(processed)
    for split in ["train", "val", "test"]:
        (processed / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed / "labels" / split).mkdir(parents=True, exist_ok=True)
        out = {
            "info": {
                "description": "Instructor ship data; canonical clipped annotations",
                "version": "1.0",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i + 1, "name": n, "supercategory": "ship"} for i, n in enumerate(names)
            ],
        }
        for r in records:
            if r["split"] != split:
                continue
            src = raw / "images" / r["source"] / r["name"]
            dest = processed / "images" / split / r["name"]
            if not dest.exists():
                dest.symlink_to(os.path.relpath(src, dest.parent))
            imid = len(out["images"]) + 1
            out["images"].append(
                {"id": imid, "file_name": r["name"], "width": r["width"], "height": r["height"]}
            )
            lines = []
            for c, x1, y1, x2, y2 in r["boxes"]:
                w, h = x2 - x1, y2 - y1
                out["annotations"].append(
                    {
                        "id": len(out["annotations"]) + 1,
                        "image_id": imid,
                        "category_id": int(c) + 1,
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                lines.append(
                    f"{int(c)} {(x1 + x2) / 2 / r['width']:.9f} {(y1 + y2) / 2 / r['height']:.9f} {w / r['width']:.9f} {h / r['height']:.9f}"
                )
            (processed / "labels" / split / (Path(r["name"]).stem + ".txt")).write_text(
                "\n".join(lines) + "\n"
            )
        dump(out, processed / "annotations" / f"instances_{split}.json")
    config = {
        "path": str(processed),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(names)),
    }
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs/dataset.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    )
    summary = {
        "seed": seed,
        "source_images": len(records),
        "source_boxes": sum(len(r["boxes"]) for r in records),
        "yolo_coco_mismatch": mismatch,
        "clipped_boxes": len(corrections),
        "duplicate_pairs": len(pairs),
        "cross_partition_pairs": sum(p["a_source"] != p["b_source"] for p in pairs),
        "excluded_train_images": excluded,
        "duplicate_screen": "SHA256 OR (63-bit pHash Hamming<=4 AND dHash Hamming<=6 AND abs(log aspect ratio difference)<0.08)",
        "splits": {},
        "classes": names,
    }
    for split in ["train", "val", "test", "excluded"]:
        rr = [r for r in records if r["split"] == split]
        cc = collections.Counter(int(c) for r in rr for c, *_ in r["boxes"])
        summary["splits"][split] = {
            "images": len(rr),
            "boxes": sum(len(r["boxes"]) for r in rr),
            "class_boxes": [cc[c] for c in range(len(names))],
        }
    dump(summary, artifacts / "audit_summary.json")
    dump(corrections, artifacts / "clipped_boxes.json")
    dump(pairs, artifacts / "duplicate_pairs.json")
    dump(records, artifacts / "manifest.json")
    # Visual evidence for manual review of the automatic duplicate screen.
    cross = [p for p in pairs if p["a_source"] != p["b_source"]]
    for start in range(0, min(len(cross), 100), 10):
        chunk = cross[start : start + 10]
        sheet = Image.new("RGB", (640, 180 * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, p in enumerate(chunk):
            for col, key in enumerate(["a", "b"]):
                src = raw / "images" / p[key + "_source"] / p[key]
                with Image.open(src) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (312, 150))
                    sheet.paste(thumb, (col * 320, row * 180))
                draw.text((col * 320, row * 180 + 150), p[key][-42:], fill="black")
        sheet.save(artifacts / f"duplicate_review_{start // 10:02d}.jpg")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.archive is None:
        archives = sorted(ROOT.glob("*.rar"))
        if not archives:
            parser.error("Place the instructor RAR in the project root, or pass --archive PATH")
        args.archive = archives[0]
    extract(args.archive, ROOT / "data/raw")
    prepare(args.seed)
