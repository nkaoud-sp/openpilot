"""
Minimal setup helpers for the standalone visual vehicle detector.

This mirrors the known-good vision BSM prototype pattern: the UI runs setup
work on a background thread and calls these helpers directly.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "selfdrive/modeld/models"
ONNX_PATH = MODEL_DIR / "visual_vehicle_detector.onnx"
PKL_PATH = MODEL_DIR / "visual_vehicle_detector_tinygrad.pkl"
META_PATH = MODEL_DIR / "visual_vehicle_detector_tinygrad.json"

DEFAULT_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"


def ensure_onnx(url: str = DEFAULT_MODEL_URL) -> None:
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  if not ONNX_PATH.exists():
    urllib.request.urlretrieve(url, ONNX_PATH)


def compile_pkl(imgsz: int | None = None, warmup: int = 2) -> None:
  from openpilot.tools.nkaoud.compile_visual_vehicle_detector_tinygrad import compile_model
  compile_model(str(ONNX_PATH), str(PKL_PATH), str(META_PATH), imgsz=imgsz, warmup=warmup)

