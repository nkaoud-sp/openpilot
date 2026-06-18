"""
Minimal setup helpers for the standalone visual vehicle detector.

This mirrors the known-good vision BSM prototype pattern: the UI runs setup
work on a background thread and calls these helpers directly.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_MODEL_DIR = REPO_ROOT / "selfdrive/modeld/models"
MODEL_DIR = Path("/data/visual_vehicle_detector")
ONNX_PATH = MODEL_DIR / "visual_vehicle_detector.onnx"
PKL_PATH = MODEL_DIR / "visual_vehicle_detector_tinygrad.pkl"
META_PATH = MODEL_DIR / "visual_vehicle_detector_tinygrad.json"
# Separate model for the driver camera (typically a square export, e.g. 320x320,
# to avoid letterbox waste on the square driver-window crop).
DRIVER_PKL_PATH = MODEL_DIR / "visual_vehicle_detector_driver_tinygrad.pkl"
DRIVER_META_PATH = MODEL_DIR / "visual_vehicle_detector_driver_tinygrad.json"
STATUS_PATH = MODEL_DIR / "visual_vehicle_detector_setup_status.json"

# Driver-camera occupancy classifier (MobileNetV3-Small, 320x320, 2-class).
# Separate from the YOLO detector model above; the detector loads this only for
# the driver camera. ONNX runs directly (with AllowOnnx); the pkl is the
# preferred on-device runtime. NOTE: host a self-contained ONNX (weights
# embedded, no external .onnx.data sidecar) or the single-file download/compile
# will fail with missing weights.
CLASSIFIER_ONNX_PATH = MODEL_DIR / "visual_vehicle_classifier_driver.onnx"
CLASSIFIER_PKL_PATH = MODEL_DIR / "visual_vehicle_classifier_driver_tinygrad.pkl"
CLASSIFIER_META_PATH = MODEL_DIR / "visual_vehicle_classifier_driver_tinygrad.json"
DEFAULT_CLASSIFIER_URL = "https://github.com/nkaoud-sp/resources/raw/refs/heads/main/mobilenet_v3_dm_320x320.onnx"

# Wide-camera car classifier (MobileNetV3-Small, 320x128, single-zone). Replaces
# YOLO on the wide cam. Self-contained ONNX; the detector loads its preprocessing
# recipe from MODEL_CONFIG (models["wide"]).
WIDE_CLASSIFIER_ONNX_PATH = MODEL_DIR / "visual_vehicle_classifier_wide.onnx"
WIDE_CLASSIFIER_PKL_PATH = MODEL_DIR / "visual_vehicle_classifier_wide_tinygrad.pkl"
WIDE_CLASSIFIER_META_PATH = MODEL_DIR / "visual_vehicle_classifier_wide_tinygrad.json"
DEFAULT_WIDE_CLASSIFIER_URL = "https://github.com/nkaoud-sp/resources/raw/refs/heads/main/mobilenet_v3_wide_320x128.onnx"

# Combined per-camera preprocessing config (models[<camera>]); the detector reads
# it from here.
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
DEFAULT_MODEL_CONFIG_URL = "https://github.com/nkaoud-sp/resources/raw/refs/heads/main/model_config.json"

DEFAULT_MODEL_640_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
DEFAULT_MODEL_480_URL = "https://github.com/nkaoud-sp/resources/raw/refs/heads/main/yolov5n_480x480_v7.0.onnx"
DEFAULT_MODEL_480X224_URL = "https://github.com/nkaoud-sp/resources/raw/refs/heads/main/yolov5n_480x224_v7.0.onnx"
DEFAULT_MODEL_320_URL = "https://github.com/nkaoud-sp/resources/raw/refs/heads/main/yolov5n_320x320_v7.0.onnx"
DEFAULT_MODEL_256_URL = "https://github.com/nkaoud-sp/resources/raw/refs/heads/main/yolov5n_256x256_v7.0.onnx"

LEGACY_ONNX_PATH = LEGACY_MODEL_DIR / ONNX_PATH.name
LEGACY_PKL_PATH = LEGACY_MODEL_DIR / PKL_PATH.name
LEGACY_META_PATH = LEGACY_MODEL_DIR / META_PATH.name
LEGACY_STATUS_PATH = LEGACY_MODEL_DIR / STATUS_PATH.name


def _size_mb(path: Path) -> float:
  try:
    return round(path.stat().st_size / (1024 * 1024), 2)
  except Exception:
    return 0.0


def read_status() -> dict[str, Any]:
  migrate_legacy_artifacts()
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


def migrate_legacy_artifacts() -> None:
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  for old_path, new_path in (
    (LEGACY_ONNX_PATH, ONNX_PATH),
    (LEGACY_PKL_PATH, PKL_PATH),
    (LEGACY_META_PATH, META_PATH),
    (LEGACY_STATUS_PATH, STATUS_PATH),
  ):
    try:
      if new_path.exists() or not old_path.exists():
        continue
      shutil.move(str(old_path), str(new_path))
    except Exception:
      # Best-effort migration only; setup/compile can still recreate artifacts.
      pass


def ensure_onnx(url: str = DEFAULT_MODEL_640_URL) -> None:
  migrate_legacy_artifacts()
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


def ensure_onnx_640() -> None:
  ensure_onnx(DEFAULT_MODEL_640_URL)


def ensure_onnx_480() -> None:
  if not DEFAULT_MODEL_480_URL:
    write_status("error", "No 480x480 ONNX URL configured. Export a 480 ONNX and place it at /data/visual_vehicle_detector/visual_vehicle_detector.onnx, then tap Compile PKL.")
    raise RuntimeError("No 480x480 ONNX URL configured")
  ensure_onnx(DEFAULT_MODEL_480_URL)


def ensure_onnx_480x224() -> None:
  if not DEFAULT_MODEL_480X224_URL:
    write_status("error", "No 480x224 ONNX URL configured. Export a 480x224 ONNX and place it at /data/visual_vehicle_detector/visual_vehicle_detector.onnx, then tap Compile PKL.")
    raise RuntimeError("No 480x224 ONNX URL configured")
  ensure_onnx(DEFAULT_MODEL_480X224_URL)


def ensure_onnx_320() -> None:
  if not DEFAULT_MODEL_320_URL:
    write_status("error", "No 320x320 ONNX URL configured. Export a 320 ONNX and place it at /data/visual_vehicle_detector/visual_vehicle_detector.onnx, then tap Compile PKL.")
    raise RuntimeError("No 320x320 ONNX URL configured")
  ensure_onnx(DEFAULT_MODEL_320_URL)


def ensure_onnx_256() -> None:
  if not DEFAULT_MODEL_256_URL:
    write_status("error", "No 256x256 ONNX URL configured.")
    raise RuntimeError("No 256x256 ONNX URL configured")
  ensure_onnx(DEFAULT_MODEL_256_URL)


def ensure_classifier_onnx(url: str = DEFAULT_CLASSIFIER_URL) -> None:
  """Download the hosted driver-cam classifier ONNX to the path the detector reads."""
  migrate_legacy_artifacts()
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  if not url:
    write_status("error", "No DM classifier URL configured.")
    raise RuntimeError("No DM classifier URL configured")
  write_status("downloading", "Downloading DM classifier ONNX...", url=url)
  try:
    if CLASSIFIER_ONNX_PATH.exists():
      CLASSIFIER_ONNX_PATH.unlink()
    urllib.request.urlretrieve(url, CLASSIFIER_ONNX_PATH)
    write_status("downloaded", "DM classifier ONNX download complete.", url=url,
                 classifier_onnx_mb=_size_mb(CLASSIFIER_ONNX_PATH))
  except Exception as e:
    write_status("error", f"classifier download failed: {e}", url=url)
    raise


def compile_pkl(imgsz: int | None = None, warmup: int = 2,
                onnx_path: Path = ONNX_PATH,
                pkl_path: Path = PKL_PATH, meta_path: Path = META_PATH, label: str = "PKL") -> None:
  migrate_legacy_artifacts()
  write_status("compiling", f"Compiling {label}...")
  from openpilot.tools.nkaoud.compile_visual_vehicle_detector_tinygrad import compile_model
  try:
    meta = compile_model(str(onnx_path), str(pkl_path), str(meta_path), imgsz=imgsz, warmup=warmup)
    shape = meta.get("input_shape", []) if isinstance(meta, dict) else []
    if isinstance(shape, list) and len(shape) >= 4:
      size_text = f" ({shape[3]}x{shape[2]})"
    else:
      size_text = ""
    write_status("compiled", f"{label} compile complete{size_text}.", input_shape=shape)
  except Exception as e:
    write_status("error", f"compile failed: {e}")
    raise


def compile_pkl_driver(imgsz: int | None = None, warmup: int = 2) -> None:
  """Compile the current ONNX into the separate driver-camera PKL."""
  compile_pkl(imgsz=imgsz, warmup=warmup, pkl_path=DRIVER_PKL_PATH, meta_path=DRIVER_META_PATH, label="DM PKL")


def compile_classifier_pkl(warmup: int = 2) -> None:
  """Compile the downloaded DM classifier ONNX into its tinygrad PKL. The model
  has a fixed 320x320 input, so imgsz is left to the model's own shape."""
  compile_pkl(imgsz=None, warmup=warmup, onnx_path=CLASSIFIER_ONNX_PATH,
              pkl_path=CLASSIFIER_PKL_PATH, meta_path=CLASSIFIER_META_PATH, label="DM Classifier PKL")


def ensure_model_config(url: str = DEFAULT_MODEL_CONFIG_URL) -> None:
  """Download the combined per-camera preprocessing config (models[<camera>])."""
  migrate_legacy_artifacts()
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  if not url:
    write_status("error", "No model_config URL configured.")
    raise RuntimeError("No model_config URL configured")
  write_status("downloading", "Downloading model_config.json...", url=url)
  try:
    if MODEL_CONFIG_PATH.exists():
      MODEL_CONFIG_PATH.unlink()
    urllib.request.urlretrieve(url, MODEL_CONFIG_PATH)
    write_status("downloaded", "model_config.json download complete.", url=url)
  except Exception as e:
    write_status("error", f"model_config download failed: {e}", url=url)
    raise


def ensure_wide_classifier_onnx(url: str = DEFAULT_WIDE_CLASSIFIER_URL) -> None:
  """Download the hosted wide-cam classifier ONNX, plus the combined config it
  needs (best-effort)."""
  migrate_legacy_artifacts()
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  if not url:
    write_status("error", "No wide classifier URL configured.")
    raise RuntimeError("No wide classifier URL configured")
  write_status("downloading", "Downloading wide classifier ONNX...", url=url)
  try:
    if WIDE_CLASSIFIER_ONNX_PATH.exists():
      WIDE_CLASSIFIER_ONNX_PATH.unlink()
    urllib.request.urlretrieve(url, WIDE_CLASSIFIER_ONNX_PATH)
    write_status("downloaded", "Wide classifier ONNX download complete.", url=url,
                 wide_classifier_onnx_mb=_size_mb(WIDE_CLASSIFIER_ONNX_PATH))
  except Exception as e:
    write_status("error", f"wide classifier download failed: {e}", url=url)
    raise
  try:
    ensure_model_config()
  except Exception:
    pass  # config is best-effort; the detector falls back to baked-in defaults


def compile_wide_classifier_pkl(warmup: int = 2) -> None:
  """Compile the downloaded wide classifier ONNX into its tinygrad PKL (the
  model's own fixed 320x128 input is used)."""
  compile_pkl(imgsz=None, warmup=warmup, onnx_path=WIDE_CLASSIFIER_ONNX_PATH,
              pkl_path=WIDE_CLASSIFIER_PKL_PATH, meta_path=WIDE_CLASSIFIER_META_PATH,
              label="Wide Classifier PKL")
