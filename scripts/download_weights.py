"""Retrieve pinned official YOLO11 pretrained weights, with resumable retries."""

import hashlib
import json
import time

import requests

from ship_detection.paths import ROOT

out = ROOT / "weights"
out.mkdir(exist_ok=True)
metadata = {}
for kind in ["n", "s"]:
    name = f"yolo11{kind}.pt"
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"
    dest = out / name
    if not dest.exists():
        for attempt in range(4):
            try:
                with requests.get(url, stream=True, timeout=(30, 120)) as r:
                    r.raise_for_status()
                    with dest.with_suffix(".part").open("wb") as f:
                        for chunk in r.iter_content(1024 * 1024):
                            f.write(chunk)
                dest.with_suffix(".part").rename(dest)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    metadata[name] = {
        "url": url,
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "bytes": dest.stat().st_size,
    }
    print(name, metadata[name], flush=True)
(out / "sources.json").write_text(json.dumps(metadata, indent=2))
