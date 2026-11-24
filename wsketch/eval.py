"""Evaluate a checkpoint on the held-out strokes.

    python -m wsketch.eval runs/v1/model.pt

Prints accuracy by class and source, the most common confusions, per-class
confidence thresholds, and renders the mistakes to runs/viz/errors_*.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import kind_to_class
from .export import load_checkpoint
from .train import evaluate, load_data
from .viz import sheet


@torch.no_grad()
def predict(net, X, idx, bs: int = 4096):
    out = []
    for i in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = net(X[idx[i:i + bs]].float())
        out.append(torch.softmax(logits.float(), -1))
    return torch.cat(out)


def thresholds(probs, y, classes, target: float = 0.99) -> dict[str, float]:
    """Lowest confidence per class at which its predictions are `target` precise."""
    conf, pred = probs.max(-1)
    out = {}
    for c, name in enumerate(classes):
        if name == "none":
            continue
        m = pred == c
        order = conf[m].argsort(descending=True)
        ok = (y[m][order] == c).float()
        prec = ok.cumsum(0) / torch.arange(1, len(ok) + 1, device=ok.device)
        good = torch.nonzero(prec >= target)
        t = conf[m][order][good[-1, 0]].item() if len(good) else 1.0
        kept = ((conf >= t) & m & (y == c)).float().sum().item() / max((y == c).sum().item(), 1)
        out[name] = {"conf": round(max(t, 0.4), 3), "recall": round(kept, 4)}
    return out


def main():
    path = sys.argv[1]
    real = sys.argv[2] if len(sys.argv) > 2 else "data/real.npz"
    dev = torch.device("cuda")
    net, CLASSES = load_checkpoint(path, dev)
    X, M, K, R, val = load_data(dev, real=real)
    Y = torch.tensor(kind_to_class(CLASSES), device=dev)[K]
    idx = torch.nonzero(val)[:, 0]
    ev = evaluate(net, X, M, Y, R, idx, CLASSES)
    print(f"accuracy {ev['acc']:.4f}  synthetic {ev['synth']:.4f}  real {ev['real']:.4f}  head IoU {ev['head_iou']:.3f}")
    print(f"{'class':10s} {'synth':>7s} {'real':>7s}")
    for c in CLASSES:
        s, r = ev["synth_by_class"].get(c), ev["real_by_class"].get(c)
        print(f"{c:10s} {s if s is not None else '':>7} {r if r is not None else '':>7}")

    conf = np.array(ev["confusion"])
    pairs = [(conf[i, j], CLASSES[i], CLASSES[j]) for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j]
    print("top confusions (true -> predicted):")
    for n, a, b in sorted(pairs, reverse=True)[:12]:
        print(f"  {a:10s} -> {b:10s} {n:6d}  ({n / max(conf[CLASSES.index(a)].sum(), 1) * 100:.2f}% of {a})")

    probs = predict(net, X, idx)
    y, r = Y[idx], R[idx]
    th = thresholds(probs, y, CLASSES)
    print("confidence thresholds for 99% precision:")
    for k, v in th.items():
        print(f"  {k:10s} conf >= {v['conf']:.3f}  keeps {v['recall'] * 100:.1f}% of true {k}")
    Path(path).with_name("thresholds.json").write_text(json.dumps(th, indent=2))

    pc, pred = probs.max(-1)
    for name, m in (("real", r), ("synth", ~r)):
        wrong = torch.nonzero((pred != y) & m)[:, 0]
        wrong = wrong[pc[wrong].argsort(descending=True)][:160]  # the most confident mistakes
        sheet(list(X[idx[wrong]].float().cpu().numpy()), f"runs/viz/errors_{name}.png",
              titles=[f"{CLASSES[y[i]]}>{CLASSES[pred[i]]} {pc[i]:.2f}" for i in wrong], cell=1.2)


if __name__ == "__main__":
    main()
