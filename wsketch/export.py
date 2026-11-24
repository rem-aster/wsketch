"""Export a trained checkpoint to ONNX plus a small JSON description.

    python -m wsketch.export runs/v1/model.pt models/shapes

Writes models/shapes.onnx, models/shapes.int8.onnx (8-bit weights, the one to ship)
and models/shapes.json, and checks that ONNX Runtime gives the same answers as PyTorch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import CLASSES, N_POINTS
from .model import Exported, Net


def load_checkpoint(path: str, dev: str = "cpu") -> tuple[Net, list[str]]:
    """The network and the classes it predicts (all 13 for checkpoints that do not say)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = Net(**ck["cfg"]).to(dev)
    net.load_state_dict(ck["state"])
    return net.eval(), ck.get("classes", CLASSES)


def load(path: str, dev: str = "cpu") -> Net:
    return load_checkpoint(path, dev)[0]


def main():
    src, dst = sys.argv[1], Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    net, classes = load_checkpoint(src)
    model = Exported(net).eval()
    x = torch.randn(2, N_POINTS, 2) * 0.3
    onnx_path = dst.with_suffix(".onnx")
    torch.onnx.export(
        model, (x,), str(onnx_path), input_names=["points"], output_names=["probs", "head"],
        dynamic_axes={"points": {0: "batch"}, "probs": {0: "batch"}, "head": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xs = (torch.rand(64, N_POINTS, 2) - 0.5).numpy()
    probs, head = sess.run(None, {"points": xs})
    with torch.no_grad():
        p_ref, h_ref = model(torch.from_numpy(xs))
    err = max(np.abs(probs - p_ref.numpy()).max(), np.abs(head - h_ref.numpy()).max())
    assert err < 1e-3, f"ONNX and PyTorch disagree: {err}"

    meta = {
        "classes": classes,
        "reject_classes": ["none"],
        "input": {
            "name": "points", "shape": [1, N_POINTS, 2], "n_points": N_POINTS,
            "contract": "arc-length resampling of the raw stroke to n_points (first and last kept), "
                        "then bounding-box centre to the origin and the longer side scaled to 1; "
                        "canvas coordinates, y down",
        },
        "outputs": {"probs": [1, len(classes)], "head": [1, N_POINTS]},
    }
    th = Path(src).with_name("thresholds.json")  # written by wsketch.eval
    if th.exists():
        meta["thresholds"] = {k: v["conf"] for k, v in json.loads(th.read_text()).items()}
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB), max difference vs PyTorch {err:.2e}")

    # 8-bit weights: a third of the size, same answers. This is the file to ship.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    small = dst.with_suffix(".int8.onnx")
    quantize_dynamic(str(onnx_path), str(small), weight_type=QuantType.QUInt8)
    strokes = np.load("data/synth.npz")["X"][::400].astype(np.float32)  # a few thousand, all classes
    ref = sess.run(["probs"], {"points": strokes})[0].argmax(1)
    q = ort.InferenceSession(str(small), providers=["CPUExecutionProvider"]).run(["probs"], {"points": strokes})[0]
    same = (q.argmax(1) == ref).mean()
    print(f"wrote {small} ({small.stat().st_size / 1024:.0f} KB), same answer as float on {same * 100:.2f}% of {len(strokes)} strokes")


if __name__ == "__main__":
    main()
