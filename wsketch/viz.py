"""Contact sheets for eyeballing data.

    python -m wsketch.viz synth            # fresh synthetic samples, one row per class
    python -m wsketch.viz synth none 96    # 96 samples of one class
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import CLASSES


def sheet(strokes, path, cols: int = 16, titles=None, masks=None, cell: float = 1.0):
    """strokes: list of (n, 2) arrays in canvas coordinates (y down)."""
    if len(strokes) == 0:
        return
    rows = int(np.ceil(len(strokes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * cell), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, P in enumerate(strokes):
        ax = axes[i // cols][i % cols]
        ax.plot(P[:, 0], P[:, 1], "-", lw=0.9, color="#222")
        if masks is not None and masks[i].any():
            h = masks[i] > 0
            ax.plot(P[h, 0], P[h, 1], ".", ms=2.5, color="#d22")
        ax.plot(P[0, 0], P[0, 1], "o", ms=2.5, color="#2a2")
        ax.set_xlim(-0.6, 0.6), ax.set_ylim(0.6, -0.6), ax.set_aspect("equal")
        if titles is not None:
            ax.set_title(titles[i], fontsize=5, pad=1)
    fig.subplots_adjust(0, 0, 1, 0.98 if titles is not None else 1, 0.05, 0.25 if titles is not None else 0.05)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "synth"
    if what == "synth":
        from .synth import sample

        rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
        if len(sys.argv) > 2:
            name, n = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 96
            out = [sample(name, rng) for _ in range(n)]
            sheet([o[0] for o in out], f"runs/viz/synth_{name}.png", masks=[o[1] for o in out])
        else:
            out = [sample(name, rng) for name in CLASSES for _ in range(16)]
            sheet([o[0] for o in out], "runs/viz/synth.png", masks=[o[1] for o in out],
                  titles=[CLASSES[i // 16] if i % 16 == 0 else "" for i in range(len(out))])


if __name__ == "__main__":
    main()
