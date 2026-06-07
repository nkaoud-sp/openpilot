#!/usr/bin/env python3
"""
Compile a visual vehicle detector ONNX into a tinygrad .pkl runner.

Run this on the target device/architecture when possible, especially for QCOM:

  cd /data/openpilot
  python3 tools/nkaoud/compile_visual_vehicle_detector_tinygrad.py \
    --onnx selfdrive/modeld/models/visual_vehicle_detector.onnx \
    --out selfdrive/modeld/models/visual_vehicle_detector_tinygrad.pkl \
    --imgsz 320

Expected source model:
  - Ultralytics YOLOv8n/YOLO11n-style ONNX
  - input: [1, 3, H, W], RGB float32 0..1
  - output: raw [1, 84, N] or NMSed [N, 6]

This helper is intentionally isolated from controls. The detector daemon simply
loads the produced pkl if present and displays debug output in the UI.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _set_default_dev() -> None:
  if "DEV" in os.environ:
    return
  try:
    from openpilot.system.hardware import TICI
    os.environ["DEV"] = "QCOM" if TICI else "CPU"
  except Exception:
    os.environ["DEV"] = "CPU"


def _shape_dim(dim: Any, default: int) -> int:
  value = getattr(dim, "dim_value", 0)
  return int(value) if value else default


def _load_onnx_runner(onnx_path: Path):
  import onnx  # pylint: disable=import-error
  model = onnx.load(str(onnx_path))
  input_proto = model.graph.input[0]
  input_name = input_proto.name
  dims = input_proto.type.tensor_type.shape.dim
  n = _shape_dim(dims[0], 1) if len(dims) > 0 else 1
  c = _shape_dim(dims[1], 3) if len(dims) > 1 else 3
  h = _shape_dim(dims[2], 320) if len(dims) > 2 else 320
  w = _shape_dim(dims[3], 320) if len(dims) > 3 else 320

  try:
    from tinygrad.frontend.onnx import OnnxRunner  # pylint: disable=import-error
  except Exception as e:
    raise RuntimeError("tinygrad.frontend.onnx.OnnxRunner is not available in this tinygrad build") from e

  return OnnxRunner(model), input_name, (n, c, h, w)


def _extract_first_output(out):
  if isinstance(out, dict):
    return next(iter(out.values()))
  if isinstance(out, (list, tuple)):
    return out[0]
  return out


def _call_runner(runner, input_name: str, x):
  # tinygrad OnnxRunner call signatures have changed across versions. Try the
  # common variants used by openpilot/sunnypilot-era tinygrad builds.
  attempts = [
    lambda: runner({input_name: x}),
    lambda: runner(**{input_name: x}),
    lambda: runner(x),
  ]
  last_err = None
  for attempt in attempts:
    try:
      return _extract_first_output(attempt())
    except Exception as e:  # keep trying the next API style
      last_err = e
  raise RuntimeError(f"Unable to call tinygrad ONNX runner with known signatures: {last_err}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--onnx", default="selfdrive/modeld/models/visual_vehicle_detector.onnx")
  parser.add_argument("--out", default="selfdrive/modeld/models/visual_vehicle_detector_tinygrad.pkl")
  parser.add_argument("--metadata", default=None)
  parser.add_argument("--imgsz", type=int, default=320)
  parser.add_argument("--warmup", type=int, default=2)
  args = parser.parse_args()

  _set_default_dev()

  from tinygrad.tensor import Tensor  # pylint: disable=import-error
  try:
    from tinygrad.engine.jit import TinyJit  # pylint: disable=import-error
  except Exception:
    from tinygrad.jit import TinyJit  # type: ignore  # pylint: disable=import-error

  onnx_path = Path(args.onnx)
  out_path = Path(args.out)
  meta_path = Path(args.metadata) if args.metadata else out_path.with_suffix(".json")
  out_path.parent.mkdir(parents=True, exist_ok=True)

  runner, input_name, input_shape = _load_onnx_runner(onnx_path)
  # Override dynamic/unknown image dims with requested imgsz.
  input_shape = (1, 3, int(args.imgsz), int(args.imgsz))

  @TinyJit
  def model_run(**kwargs):
    return _call_runner(runner, input_name, kwargs[input_name])

  dummy = Tensor(np.zeros(input_shape, dtype=np.float32)).realize()
  for i in range(max(1, int(args.warmup))):
    y = model_run(**{input_name: dummy})
    if hasattr(y, "realize"):
      y.realize()
    print(f"warmup {i + 1}/{args.warmup} complete")

  with open(out_path, "wb") as f:
    pickle.dump(model_run, f)

  meta = {
    "source_onnx": str(onnx_path),
    "output_pkl": str(out_path),
    "input_name": input_name,
    "input_shape": list(input_shape),
    "dev": os.environ.get("DEV", ""),
    "format": "ultralytics_yolo_raw_or_nmsed",
  }
  meta_path.write_text(json.dumps(meta, indent=2))
  print(f"wrote {out_path}")
  print(f"wrote {meta_path}")
  print("Copy/keep both the .pkl and .json next to each other on the comma device.")


if __name__ == "__main__":
  main()
