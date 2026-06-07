"""
Minimal setup helpers for the standalone visual vehicle detector.

This mirrors the known-good vision BSM prototype pattern: the UI runs setup
work on a background thread and calls these helpers directly.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "selfdrive/modeld/models"
ONNX_PATH = MODEL_DIR / "visual_vehicle_detector.onnx"
PKL_PATH = MODEL_DIR / "visual_vehicle_detector_tinygrad.pkl"
META_PATH = MODEL_DIR / "visual_vehicle_detector_tinygrad.json"
STATUS_PATH = MODEL_DIR / "visual_vehicle_detector_setup_status.json"

DEFAULT_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"


def _size_mb(path: Path) -> float:
  try:
    return round(path.stat().st_size / (1024 * 1024), 2)
  except Exception:
    return 0.0


def read_status() -> dict[str, Any]:
  try:
    return json.loads(STATUS_PATH.read_text())
  except Exception:
    return {}


def write_status(state: str, message: str, **extra: Any) -> dict[str, Any]:
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  status = {
    "state": state,
    "message": message,
    "updated_at": time.time(),
    "onnx_exists": ONNX_PATH.exists(),
    "pkl_exists": PKL_PATH.exists(),
    "meta_exists": META_PATH.exists(),
    "onnx_size_mb": _size_mb(ONNX_PATH),
    "pkl_size_mb": _size_mb(PKL_PATH),
  }
  status.update(extra)
  STATUS_PATH.write_text(json.dumps(status, separators=(",", ":")))
  return status


def ensure_onnx(url: str = DEFAULT_MODEL_URL) -> None:
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  write_status("downloading", "Downloading ONNX...", url=url)
  try:
    if ONNX_PATH.exists():
      ONNX_PATH.unlink()
    urllib.request.urlretrieve(url, ONNX_PATH)
    write_status("downloaded", "ONNX download complete.", url=url)
  except Exception as e:
    write_status("error", f"download failed: {e}", url=url)
    raise


def compile_pkl(imgsz: int | None = None, warmup: int = 2) -> None:
  write_status("compiling", "Compiling PKL...")
  from openpilot.tools.nkaoud.compile_visual_vehicle_detector_tinygrad import compile_model
  try:
    meta = compile_model(str(ONNX_PATH), str(PKL_PATH), str(META_PATH), imgsz=imgsz, warmup=warmup)
    shape = meta.get("input_shape", []) if isinstance(meta, dict) else []
    if isinstance(shape, list) and len(shape) >= 4:
      size_text = f" ({shape[3]}x{shape[2]})"
    else:
      size_text = ""
    write_status("compiled", f"PKL compile complete{size_text}.", input_shape=shape)
  except Exception as e:
    write_status("error", f"compile failed: {e}")
    raise
