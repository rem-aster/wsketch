"""Drop real strokes whose Quick, Draw! label a trained model confidently rejects.

Quick, Draw! labels are what the player was asked to draw. A first model,
anchored by the synthetic data, is right more often than those labels are
(a five-sided "hexagon" really is a pentagon), so strokes where it disagrees
with confidence are removed before the final training run.

    python -m wsketch.clean runs/v1/model.pt [class ...]   # data/real.npz -> data/real_clean.npz

Classes listed after the checkpoint are left untouched.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from . import CLASS_INDEX, CLASSES
from .export import load


PLAIN = ["line", "ellipse", "rectangle", "triangle"]


@torch.no_grad()
def main(path: str, conf: float = 0.6, trust: tuple[str, ...] = ()):
    dev = torch.device("cuda")
    net = load(path, dev)
    r = np.load("data/real.npz")
    X, Y = torch.from_numpy(r["X"]).to(dev), torch.from_numpy(r["Y"].astype(np.int64)).to(dev)
    probs = []
    for i in range(0, len(X), 4096):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs.append(torch.softmax(net(X[i:i + 4096].float())[0].float(), -1))
    p, pred = torch.cat(probs).max(-1)
    # A stroke labelled "none" that the model calls an arrow or a scribble is a hard negative
    # worth keeping. One it calls a plain primitive usually is that primitive (a "moon" drawn
    # as a circle, an "elbow" drawn as a straight line), so only those are removed.
    none = Y == CLASS_INDEX["none"]
    plain = torch.isin(pred, torch.tensor([CLASS_INDEX[c] for c in PLAIN], device=dev))
    drop = (~none & (pred != Y) & (p > conf)) | (none & plain & (p > 0.8))
    drop &= torch.from_numpy(r["cats"][r["S"]] != "user").to(dev)  # hand-labelled in the demo: always kept
    for name in trust:  # classes the checking model never saw real examples of
        drop &= Y != CLASS_INDEX[name]
    drop = drop.cpu().numpy()
    for c, name in enumerate(CLASSES):
        m = r["Y"] == c
        if m.any():
            print(f"{name:10s} {m.sum():7d} strokes, dropped {drop[m].mean() * 100:5.2f}%")
    np.savez("data/real_clean.npz", X=r["X"][~drop], Y=r["Y"][~drop], S=r["S"][~drop], cats=r["cats"])
    print(f"dropped {drop.sum()} of {len(drop)}, wrote data/real_clean.npz")


if __name__ == "__main__":
    main(sys.argv[1], trust=tuple(sys.argv[2:]))
