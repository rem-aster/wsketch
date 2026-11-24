"""Download Quick, Draw! raw drawings and keep the single-stroke ones.

Quick, Draw! dataset, Google, CC BY 4.0.
https://github.com/googlecreativelab/quickdraw-dataset

The raw files are streamed and never stored: for every category we keep only
recognized drawings made of exactly one stroke, as variable-length point lists.

    python -m wsketch.quickdraw            # everything
    python -m wsketch.quickdraw circle     # some categories
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

URL = "https://storage.googleapis.com/quickdraw_dataset/full/raw/{}.ndjson"
OUT = Path("data/qd")

# Categories that map onto model classes. Whole file is read.
SHAPES = [
    "line", "circle", "square", "triangle", "diamond", "hexagon", "octagon",
    "star", "cloud", "squiggle", "zigzag",
]
# Things that are not shapes when drawn in one stroke. Read until MAX_NEG kept.
NEGATIVES = [
    "lightning", "snake", "stairs", "moon", "ocean", "fish", "bowtie",
    "hourglass", "lollipop", "pants", "t-shirt", "crown", "sock",
    "hockey stick", "light bulb", "mushroom", "key", "snail", "hurricane",
    "bird", "candle", "boomerang", "elbow", "foot", "paper clip", "ear",
    "moustache", "wine glass", "ice cream", "golf club", "knee", "banana",
]
MAX_SHAPE = 150_000
MAX_NEG = 12_000


def fetch(cat: str, limit: int) -> str:
    out = OUT / f"{cat.replace(' ', '_')}.npz"
    if out.exists():
        return f"{cat}: cached"
    url = URL.format(urllib.parse.quote(cat))
    pts, lens, seen = [], [], 0
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r:
        for line in r:
            seen += 1
            # cheap pre-filter before the JSON parse
            if b'"recognized":true' not in line:
                continue
            d = json.loads(line)
            strokes = d["drawing"]
            if len(strokes) != 1:
                continue
            x, y, t = strokes[0]
            if len(x) < 4:
                continue
            pts.append(np.array([x, y, t], dtype=np.float32).T)
            lens.append(len(x))
            if len(lens) >= limit:
                break
    OUT.mkdir(parents=True, exist_ok=True)
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.savez_compressed(out, pts=np.concatenate(pts), offs=offs)
    return f"{cat}: kept {len(lens)} of {seen} read, {time.time() - t0:.0f}s"


def load(cat: str) -> list[np.ndarray]:
    """Strokes of one category as a list of (n, 3) arrays: x, y, t(ms)."""
    z = np.load(OUT / f"{cat.replace(' ', '_')}.npz")
    pts, offs = z["pts"], z["offs"]
    return [pts[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]


def main() -> None:
    want = sys.argv[1:]
    jobs = [(c, MAX_SHAPE) for c in SHAPES] + [(c, MAX_NEG) for c in NEGATIVES]
    if want:
        jobs = [j for j in jobs if j[0] in want]
    def run(job: tuple[str, int]) -> str:
        for attempt in range(3):
            try:
                return fetch(*job)
            except Exception as e:  # network hiccup: retry, then report
                err = e
        return f"{job[0]}: FAILED {err!r}"

    with ThreadPoolExecutor(6) as ex:
        for msg in ex.map(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
