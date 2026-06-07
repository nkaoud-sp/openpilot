#!/usr/bin/env python3
"""
Compile a visual vehicle detector ONNX into a tinygrad .pkl runner.

Run this on the target device/architecture when possible, especially for QCOM:

  cd /data/openpilot
  python3 tools/nkaoud/compile_visual_vehicle_detector_tinygrad.py \
    --onnx selfdrive/modeld/models/visual_vehicle_detector.onnx \
    --out selfdrive/modeld/models/visual_vehicle_detector_tinygrad.pkl

Expected source model:
  - Ultralytics YOLOv5n/YOLOv8n/YOLO11n-style ONNX
  - input: [1, 3, H, W], RGB float32 0..1
  - output: YOLOv5 raw [1, N, 85], YOLOv8 raw [1, 84, N], or NMSed [N, 6]

This helper is intentionally isolated from controls. The detector daemon simply
loads the produced pkl if present and displays debug output in the UI.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


def _set_default_dev() -> None:
  if "DEV" in os.environ:
    return
  try:
    from openpilot.system.hardware import TICI
    os.environ["DEV"] = "QCOM" if TICI else "CPU"
  except Exception:
    os.environ["DEV"] = "CPU"


def _load_onnx_runner(onnx_path: Path):
  try:
    from tinygrad.nn.onnx import OnnxRunner  # pylint: disable=import-error
  except Exception as e:
    raise RuntimeError("tinygrad.nn.onnx.OnnxRunner is not available in this tinygrad build") from e

  runner = OnnxRunner(onnx_path)
  if len(runner.graph_inputs) != 1:
    raise RuntimeError(f"Expected one ONNX input, found {list(runner.graph_inputs)}")

  input_name, input_spec = next(iter(runner.graph_inputs.items()))
  return runner, input_name, tuple(input_spec.shape)


def _resolve_input_shape(model_shape: tuple, imgsz: int | None) -> tuple[int, int, int, int]:
  if len(model_shape) != 4:
    raise RuntimeError(f"Expected a 4D NCHW input, found {model_shape}")

  resolved: list[int] = []
  for index, dim in enumerate(model_shape):
    if isinstance(dim, int) and dim > 0:
      resolved.append(dim)
    elif index == 0:
      resolved.append(1)
    elif index == 1:
      resolved.append(3)
    elif imgsz is not None:
      resolved.append(imgsz)
    else:
      raise RuntimeError(f"Dynamic image shape {model_shape} requires --imgsz")

  if resolved[0] != 1 or resolved[1] != 3:
    raise RuntimeError(f"Expected input shape [1, 3, H, W], found {tuple(resolved)}")

  if imgsz is not None and tuple(resolved[2:]) != (imgsz, imgsz):
    print(f"model input is fixed at {resolved[3]}x{resolved[2]}; ignoring --imgsz {imgsz}")

  return tuple(resolved)


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
  parser.add_argument("--imgsz", type=int, default=None, help="Image size for dynamic-shape ONNX models")
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

  runner, input_name, model_shape = _load_onnx_runner(onnx_path)
  input_shape = _resolve_input_shape(model_shape, args.imgsz)
  print(f"model input {input_name}: {input_shape}")

  model_run = TinyJit(
    lambda **kwargs: _call_runner(runner, input_name, kwargs[input_name]),
    prune=True,
  )

  dummy = Tensor(np.zeros(input_shape, dtype=np.float32)).realize()
  for i in range(max(1, int(args.warmup))):
    y = model_run(**{input_name: dummy})
    if hasattr(y, "realize"):
      y.realize()
    print(f"warmup {i + 1}/{args.warmup} complete")

  output_shape = tuple(int(dim) for dim in getattr(y, "shape", ()))
  if len(output_shape) not in (2, 3):
    raise RuntimeError(f"Unexpected detector output shape: {output_shape}")
  print(f"model output: {output_shape}")

  with open(out_path, "wb") as f:
    pickle.dump(model_run, f)

  with open(out_path, "rb") as f:
    reloaded_model = pickle.load(f)
  reloaded_output = reloaded_model(**{input_name: dummy})
  if hasattr(reloaded_output, "realize"):
    reloaded_output.realize()

  meta = {
    "source_onnx": str(onnx_path),
    "output_pkl": str(out_path),
    "input_name": input_name,
    "input_shape": list(input_shape),
    "output_shape": list(output_shape),
    "dev": os.environ.get("DEV", ""),
    "format": "ultralytics_yolo_raw_or_nmsed",
  }
  meta_path.write_text(json.dumps(meta, indent=2))
  print(f"wrote {out_path}")
  print(f"wrote {meta_path}")
  print("Copy/keep both the .pkl and .json next to each other on the comma device.")


if __name__ == "__main__":
  main()
