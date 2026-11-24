"""Train the stroke classifier.

    python -m wsketch.train --out runs/v1 --steps 20000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import CLASS_SETS, CLASSES, kind_to_class
from .aug import augment
from .model import Net


def load_data(dev, val_frac: float = 0.04, seed: int = 0, real: str = "data/real.npz"):
    s, r = np.load("data/synth.npz"), np.load(real)
    X = torch.from_numpy(np.concatenate([s["X"], r["X"]])).to(dev)                      # float16
    M = torch.from_numpy(np.concatenate([s["M"], np.zeros(r["X"].shape[:2], np.uint8)])).to(dev)
    Y = torch.from_numpy(np.concatenate([s["Y"], r["Y"]]).astype(np.int64)).to(dev)
    R = torch.cat([torch.zeros(len(s["Y"]), dtype=torch.bool), torch.ones(len(r["Y"]), dtype=torch.bool)]).to(dev)
    g = torch.Generator().manual_seed(seed)
    val = torch.zeros(len(Y), dtype=torch.bool)
    val[torch.randperm(len(Y), generator=g)[: int(len(Y) * val_frac)]] = True
    return X, M, Y, R, val.to(dev)


class Sampler:
    """Class-balanced batches, half real / half synthetic where a class has both."""

    def __init__(self, Y, R, train, classes=CLASSES, none_weight: float = 2.5, p_real: float = 0.5):
        dev = Y.device
        pools, starts, sizes, n = [], [], [], 0
        for c in range(len(classes)):
            for real in (False, True):
                idx = torch.nonzero(train & (Y == c) & (R == real))[:, 0]
                pools.append(idx), starts.append(n), sizes.append(len(idx))
                n += len(idx)
        self.pool = torch.cat(pools)
        self.start = torch.tensor(starts, device=dev)
        self.size = torch.tensor(sizes, device=dev)
        has = (self.size.view(-1, 2) > 0)
        self.p_real = torch.where(has[:, 0] & has[:, 1], p_real, has[:, 1].float())
        w = torch.ones(len(classes), device=dev)
        w[classes.index("none")] = none_weight
        self.w = w * has.any(dim=1)

    def __call__(self, B: int) -> torch.Tensor:
        c = torch.multinomial(self.w, B, replacement=True)
        p = c * 2 + (torch.rand(B, device=c.device) < self.p_real[c]).long()
        return self.pool[self.start[p] + (torch.rand(B, device=c.device) * self.size[p]).long()]


@torch.no_grad()
def evaluate(net, X, M, Y, R, idx, classes=CLASSES, bs: int = 4096):
    """Accuracy on the given indices, split by source and class; arrowhead IoU."""
    net.eval()
    pred, head_i, head_u = [], 0, 0
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, head = net(X[j].float())
        pred.append(logits.argmax(-1))
        h, m = head > 0, M[j] > 0
        head_i += (h & m).sum().item()
        head_u += (h | m).sum().item()
    pred, y, r = torch.cat(pred), Y[idx], R[idx]
    ok = pred == y
    out = {"acc": ok.float().mean().item(), "head_iou": head_i / max(head_u, 1)}
    for name, m in (("synth", ~r), ("real", r)):
        out[name] = ok[m].float().mean().item()
        out[name + "_by_class"] = {classes[c]: round(ok[m & (y == c)].float().mean().item(), 4)
                                   for c in range(len(classes)) if (m & (y == c)).any()}
    conf = torch.zeros(len(classes), len(classes), dtype=torch.long, device=y.device)
    conf.index_put_((y, pred), torch.ones_like(y), accumulate=True)
    out["confusion"] = conf.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--real", default="data/real.npz")
    ap.add_argument("--classes", default="full", choices=list(CLASS_SETS))
    ap.add_argument("--none-weight", type=float, default=2.5)
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    classes = CLASS_SETS[args.classes]
    X, M, K, R, val = load_data(dev, real=args.real)          # K: the 13 kinds the data is labelled with
    Y = torch.tensor(kind_to_class(classes), device=dev)[K]   # Y: what this model predicts
    sampler = Sampler(Y, R, ~val, classes, args.none_weight)
    val_idx = torch.nonzero(val)[:, 0]
    print(f"{len(Y)} strokes ({int(R.sum())} real), {len(val_idx)} held out")

    cfg = dict(d=args.d, depth=args.depth, heads=args.heads, n_classes=len(classes))
    net = Net(**cfg).to(dev)
    ema = copy.deepcopy(net).requires_grad_(False)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e3:.0f}k parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1, max(0, s - warm) / max(1, args.steps - warm)))))

    best, t0, log = 0.0, time.time(), open(out / "log.jsonl", "a")
    for step in range(1, args.steps + 1):
        net.train()
        j = sampler(args.bs)
        P, Mh = augment(X[j].float(), M[j].float(), K[j], R[j])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, head = net(P)
        loss_c = F.cross_entropy(logits.float(), Y[j], label_smoothing=0.05)
        loss_h = F.binary_cross_entropy_with_logits(head.float(), (Mh > 0.5).float())
        loss = loss_c + 0.5 * loss_h
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            decay = min(0.999, step / (step + 10))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - decay)

        if step % 100 == 0:
            print(f"\rstep {step} loss {loss_c.item():.4f} head {loss_h.item():.4f} "
                  f"{step / (time.time() - t0):.1f} it/s", end="", flush=True)
        if step % 1000 == 0 or step == args.steps:
            ev = evaluate(ema, X, M, Y, R, val_idx, classes)
            print(f"\nstep {step}: val acc {ev['acc']:.4f} synth {ev['synth']:.4f} real {ev['real']:.4f} "
                  f"head IoU {ev['head_iou']:.3f}")
            log.write(json.dumps({"step": step, **{k: v for k, v in ev.items() if k != "confusion"}}) + "\n")
            log.flush()
            if ev["acc"] >= best:
                best = ev["acc"]
                torch.save({"cfg": cfg, "classes": classes, "state": ema.state_dict(), "step": step, "eval": ev},
                           out / "model.pt")
    print(f"best val acc {best:.4f}, saved to {out / 'model.pt'}")


if __name__ == "__main__":
    main()
