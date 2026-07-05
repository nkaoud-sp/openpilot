#!/usr/bin/env python3
"""
Standalone visual adjacent-vehicle detector for UI/debug validation.

This daemon is independent from navigation and controls:
  - It does not publish a desire.
  - It does not block lane changes.
  - It publishes visualVehicleDetectorStateSP for the UI and other services.

Runtime order:
  1. Prefer compiled tinygrad pkl:
       /data/openpilot/selfdrive/modeld/models/visual_vehicle_detector_tinygrad.pkl
  2. Optional ONNX fallback only if VisualVehicleDetectorAllowOnnx is enabled.

The pkl path is preferred for comma3x because it follows the same style as the
existing tinygrad model runners and avoids a separate ONNX Runtime process.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cereal import messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import MODEL_DIR as ARTIFACT_DIR, migrate_legacy_artifacts

# Live preview: the UI dialog/web server touches the request file while open,
# the detector writes the latest letterboxed RGB tensor to the PNG path.
PREVIEW_REQUEST_PATH = "/tmp/nkaoud_vvd_preview.request"
PREVIEW_PNG_PATH = "/tmp/nkaoud_vvd_preview.png"               # production (full range)
PREVIEW_PNG_PATH_FULL = "/tmp/nkaoud_vvd_preview_full.png"
PREVIEW_PNG_PATH_LIMITED = "/tmp/nkaoud_vvd_preview_limited.png"
PREVIEW_RAW_Y_PATH = "/tmp/nkaoud_vvd_raw_y.png"
PREVIEW_RAW_U_PATH = "/tmp/nkaoud_vvd_raw_u.png"
PREVIEW_RAW_V_PATH = "/tmp/nkaoud_vvd_raw_v.png"
# Extra detector previews used by the cropped-inference path.
# These are only written while PREVIEW_REQUEST_PATH exists.
PREVIEW_FULL_FRAME_CROP_PATH = "/tmp/nkaoud_vvd_preview_full_frame_crop.png"
PREVIEW_DETECTOR_CROP_PATH = "/tmp/nkaoud_vvd_preview_detector_crop.png"
PREVIEW_MODEL_INPUT_PATH = "/tmp/nkaoud_vvd_preview_model_input.png"
BUF_GEOMETRY_PATH = "/tmp/nkaoud_vvd_buf_geometry.json"
DEFAULT_PKL_PATH = str(ARTIFACT_DIR / "visual_vehicle_detector_tinygrad.pkl")
DEFAULT_DRIVER_PKL_PATH = str(ARTIFACT_DIR / "visual_vehicle_detector_driver_tinygrad.pkl")
DEFAULT_ONNX_PATH = str(ARTIFACT_DIR / "visual_vehicle_detector.onnx")
STATE_PUBLISH_HZ = 5.0
STATE_PUBLISH_INTERVAL = 1.0 / STATE_PUBLISH_HZ

# Dataset capture: while the capture portal/dialog is open (sentinel present)
# and the car is onroad, the detector saves the selected camera's crop as JPEG
# for future training. Capped to protect device storage.
CAPTURE_DIR = ARTIFACT_DIR / "captures"
CAPTURE_REQUEST_PATH = "/tmp/nkaoud_vvd_capture.request"
DEFAULT_CAPTURE_HZ = 2.0
CAPTURE_MAX_FILES = 8000
CAPTURE_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB safety cap

# Driver-camera occupancy classifier. The driver camera REPLACES the YOLO
# detection path with a whole-crop 2-class classifier (the current model is a
# MobileNetV3-Small car classifier). It takes the per-camera driver crop (the
# same crop the capture portal saves as cap_driver_*.jpg), cuts a fixed inner
# ROI, stretches it to img_size, normalizes (see CLASSIFIER_NORM below), and
# emits softmax logits over 2 classes; CLASSIFIER_POS_INDEX picks the
# "blocked"/positive (car-present) logit.
DEFAULT_CLASSIFIER_PKL_PATH = str(ARTIFACT_DIR / "visual_vehicle_classifier_driver_tinygrad.pkl")
DEFAULT_CLASSIFIER_ONNX_PATH = str(ARTIFACT_DIR / "visual_vehicle_classifier_driver.onnx")
CLASSIFIER_IMG_SIZE = 320            # fixed model input (the ONNX is 1x3x320x320)
CLASSIFIER_SRC_W = 543               # the cap_driver_*.jpg geometry training was cut from
CLASSIFIER_SRC_H = 530
CLASSIFIER_ROI_ABS = (30, 55, 535, 425)  # x1,y1,x2,y2 ROI inside that 543x530 crop
# Applied as fractions of the live driver crop so it stays correct if the crop
# size changes proportionally (exact when the crop is 543x530, as at capture).
CLASSIFIER_ROI_FRAC = (
  CLASSIFIER_ROI_ABS[0] / CLASSIFIER_SRC_W, CLASSIFIER_ROI_ABS[1] / CLASSIFIER_SRC_H,
  CLASSIFIER_ROI_ABS[2] / CLASSIFIER_SRC_W, CLASSIFIER_ROI_ABS[3] / CLASSIFIER_SRC_H,
)
CLASSIFIER_CAMERA = "driver"
DEFAULT_BLOCKED_THRESHOLD = 0.5

# Classifier input normalization. MobileNetV3-Small (torchvision/timm) trains
# with ImageNet mean/std, so that is the default. If the model was trained
# differently, switch on-device without recompiling via the env var:
#   imagenet -> (RGB/255 - mean) / std        (default, MobileNetV3 standard)
#   signed   -> RGB/127.5 - 1                  (the previous TinyCNN's [-1, 1])
#   unit     -> RGB/255                         (plain [0, 1])
CLASSIFIER_NORM = os.getenv("NKAOUD_VISUAL_VEHICLE_CLASSIFIER_NORM", "imagenet").strip().lower()
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Which softmax index is the positive ("blocked"/car-present) class. The current
# MobileNetV3 DM model uses class 0 == car, so car-present is softmax[0] -> 0.
# (The old TinyCNN was [clear, blocked] -> 1.) Override if your label order
# differs: NKAOUD_VISUAL_VEHICLE_CLASSIFIER_POS_INDEX=1.
CLASSIFIER_POS_INDEX = int(os.getenv("NKAOUD_VISUAL_VEHICLE_CLASSIFIER_POS_INDEX", "0"))

# Per-camera classifier models. Each camera that uses a classifier (instead of
# YOLO) carries its own model + preprocessing recipe. The recipe is read from a
# combined model_config.json shipped alongside the models (models[<camera>]),
# falling back to the baked-in defaults below if the file is absent.
CLASSIFIER_CAMERAS = ("driver", "wide")
MODEL_CONFIG_PATH = str(ARTIFACT_DIR / "model_config.json")
DEFAULT_WIDE_CLASSIFIER_PKL_PATH = str(ARTIFACT_DIR / "visual_vehicle_classifier_wide_tinygrad.pkl")
DEFAULT_WIDE_CLASSIFIER_ONNX_PATH = str(ARTIFACT_DIR / "visual_vehicle_classifier_wide.onnx")


@dataclass
class SideRecipe:
  """Per-side preprocessing overrides. When a ClassifierRecipe carries a
  side_crops mapping, the detector reads the side's entry to pick the crop and
  mirror for that side -- overriding the top-level recipe fields. Missing keys
  in the JSON inherit from the top-level recipe at parse time."""
  crop_mode: str
  roi_frac: tuple[float, float, float, float]
  left_crop_px: int
  mirror: bool


@dataclass
class ClassifierRecipe:
  """Per-model preprocessing recipe. crop_mode is either 'roi_inset' (cut a fixed
  inner ROI, as the driver model wants) or 'remove_left_keep_rest' (chop N px off
  the left, as the wide model wants). sides drives left/right alternation
  ('left','right') vs a single forward bit ('center'). side_crops, when present,
  overrides crop_mode/roi_frac/left_crop_px/mirror per side."""
  res_w: int
  res_h: int
  crop_mode: str
  roi_frac: tuple[float, float, float, float]
  left_crop_px: int
  src_w: int
  src_h: int
  sides: tuple[str, ...]
  mirror_left: bool
  mean: np.ndarray
  std: np.ndarray
  pos_index: int
  side_crops: dict[str, SideRecipe] | None = None


DEFAULT_RECIPES = {
  "driver": ClassifierRecipe(
    res_w=CLASSIFIER_IMG_SIZE, res_h=CLASSIFIER_IMG_SIZE, crop_mode="roi_inset",
    roi_frac=CLASSIFIER_ROI_FRAC, left_crop_px=0, src_w=CLASSIFIER_SRC_W, src_h=CLASSIFIER_SRC_H,
    sides=("left", "right"), mirror_left=True, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    pos_index=CLASSIFIER_POS_INDEX),
  "wide": ClassifierRecipe(
    res_w=320, res_h=128, crop_mode="remove_left_keep_rest",
    roi_frac=(0.0, 0.0, 1.0, 1.0), left_crop_px=90, src_w=854, src_h=280,
    sides=("center",), mirror_left=False, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    pos_index=CLASSIFIER_POS_INDEX),
}


def load_classifier_recipe(camera: str) -> ClassifierRecipe:
  """Recipe for `camera`: baked-in default overlaid with models[camera] from the
  combined model_config.json when present."""
  base = DEFAULT_RECIPES.get(camera, DEFAULT_RECIPES["driver"])
  try:
    models = (json.loads(Path(MODEL_CONFIG_PATH).read_text()).get("models", {}) or {})
    m = models.get(camera)
  except Exception:
    m = None
  if not isinstance(m, dict):
    return base
  res_w = int(m.get("res_width", base.res_w))
  res_h = int(m.get("res_height", base.res_h))
  src_w = int(m.get("source_width", base.src_w))
  src_h = int(m.get("source_height", base.src_h))
  roi = m.get("roi_abs")
  if isinstance(roi, (list, tuple)) and len(roi) == 4 and src_w > 0 and src_h > 0:
    roi_frac = (roi[0] / src_w, roi[1] / src_h, roi[2] / src_w, roi[3] / src_h)
  else:
    roi_frac = base.roi_frac
  sides = tuple(str(s) for s in m.get("sides", base.sides)) or base.sides
  car_idx = (m.get("class_to_idx", {}) or {}).get("car")
  crop_mode = str(m.get("crop_mode", base.crop_mode))
  left_crop_px = int(m.get("left_crop_px", base.left_crop_px))
  raw_side_crops = m.get("side_crops")
  side_crops: dict[str, SideRecipe] | None = None
  if isinstance(raw_side_crops, dict):
    side_crops = {}
    for s, sm in raw_side_crops.items():
      if not isinstance(sm, dict):
        continue
      s_roi = sm.get("roi_abs")
      if isinstance(s_roi, (list, tuple)) and len(s_roi) == 4 and src_w > 0 and src_h > 0:
        s_roi_frac = (s_roi[0] / src_w, s_roi[1] / src_h, s_roi[2] / src_w, s_roi[3] / src_h)
      else:
        s_roi_frac = roi_frac
      side_crops[str(s)] = SideRecipe(
        crop_mode=str(sm.get("crop_mode", crop_mode)),
        roi_frac=s_roi_frac,
        left_crop_px=int(sm.get("left_crop_px", left_crop_px)),
        mirror=bool(sm.get("mirror", False)),
      )
  return ClassifierRecipe(
    res_w=res_w, res_h=res_h, crop_mode=crop_mode,
    roi_frac=roi_frac, left_crop_px=left_crop_px,
    src_w=src_w, src_h=src_h, sides=sides, mirror_left=bool(m.get("mirror_left", base.mirror_left)),
    mean=np.array(m.get("normalization_mean", base.mean.tolist()), dtype=np.float32),
    std=np.array(m.get("normalization_std", base.std.tolist()), dtype=np.float32),
    pos_index=int(car_idx) if car_idx is not None else base.pos_index,
    side_crops=side_crops)


def side_is_augmented(recipe: ClassifierRecipe, side: str | None) -> bool:
  """True when `side` is the mirrored (augmented) side relative to the model's
  reference orientation: its per-side mirror flag (new side_crops schema) or the
  legacy mirror_left (LEFT-only) toggle. The reference side returns False."""
  if recipe.side_crops and side and side in recipe.side_crops:
    return bool(recipe.side_crops[side].mirror)
  return bool(side == "left" and recipe.mirror_left)


def reference_roi_frac(recipe: ClassifierRecipe) -> tuple[float, float, float, float]:
  """The single ROI (fractions of the crop) the portal tunes, taken in the
  model's reference (un-augmented, mirror==False) orientation. The augmented
  side's ROI is derived from this as its horizontal mirror at inference time."""
  def _from(crop_mode: str, roi_frac: tuple[float, float, float, float], left_crop_px: int):
    if crop_mode == "remove_left_keep_rest":
      return (left_crop_px / recipe.src_w if recipe.src_w else 0.0, 0.0, 1.0, 1.0)
    return roi_frac
  if recipe.side_crops:
    for cfg in recipe.side_crops.values():
      if not cfg.mirror:
        return _from(cfg.crop_mode, cfg.roi_frac, cfg.left_crop_px)
  return _from(recipe.crop_mode, recipe.roi_frac, recipe.left_crop_px)

# COCO class IDs from Ultralytics YOLO exports.
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

# Detector crop and ROI layout.
#
# We crop this fixed 928x416 region from the original wide-road RGB frame:
#   x=854..1782, y=425..841  (Python slicing: end coordinate is exclusive)
# Then YOLO runs only on that cropped image. The compiled model should be
# 480x224, matching this wide crop closely.
#
# ROI sizes below are expressed in the 480x224 model coordinate system:
#   - left ROI:  x=0..32,    y=0..224
#   - right ROI: x=32..480,  y=0..224
# They are converted to normalized coordinates so the same ROI logic works
# after YOLO boxes are mapped back into the 928x416 crop coordinates.
DETECT_CROP_X = 854
DETECT_CROP_Y = 425
DETECT_CROP_W = 928
DETECT_CROP_H = 416
ROI_MODEL_W = 480
ROI_MODEL_H = 224


def _env_float(name: str, default: float) -> float:
  raw = os.getenv(name, "")
  if raw == "":
    return default
  try:
    return float(raw)
  except ValueError:
    cloudlog.warning("visual vehicle detector invalid %s=%r; using %s", name, raw, default)
    return default


# Normalized crop ROIs: x1, y1, x2, y2 (fractions of the detector crop).
# The left ROI is a fixed thin reference sliver; the right ROI is the
# adjacent-lane band and is live-tunable (env seeds the default, the tuning
# file/web portal can override it at runtime).
LEFT_ROI = (0.0 / ROI_MODEL_W, 0.0 / ROI_MODEL_H, 32.0 / ROI_MODEL_W, 224.0 / ROI_MODEL_H)

# Defaults (env-seeded). These become the starting values for the live tuning
# file; the apparent-size / position gate rejects far-lane detections (a car in
# the adjacent lane is large and low; cars two or three lanes over are small and
# sit near the horizon).
DEFAULT_RIGHT_X1 = _env_float("NKAOUD_VVD_RIGHT_X1", 32.0 / ROI_MODEL_W)
DEFAULT_RIGHT_Y1 = _env_float("NKAOUD_VVD_RIGHT_Y1", 0.0 / ROI_MODEL_H)
DEFAULT_RIGHT_X2 = _env_float("NKAOUD_VVD_RIGHT_X2", 480.0 / ROI_MODEL_W)
DEFAULT_RIGHT_Y2 = _env_float("NKAOUD_VVD_RIGHT_Y2", 224.0 / ROI_MODEL_H)
DEFAULT_MIN_BOX_W = _env_float("NKAOUD_VVD_MIN_BOX_W", 0.08)
DEFAULT_MIN_BOX_H = _env_float("NKAOUD_VVD_MIN_BOX_H", 0.20)
DEFAULT_MIN_BOTTOM_Y = _env_float("NKAOUD_VVD_MIN_BOTTOM_Y", 0.35)
DEFAULT_CONF = _env_float("NKAOUD_VISUAL_VEHICLE_CONF", 0.35)

# Crop box (pixels in the full wide-road frame) and detector rate. Also live-
# tunable; the crop is clamped to the actual frame in _crop_detector_region.
DEFAULT_CROP_X = _env_float("NKAOUD_VISUAL_VEHICLE_CROP_X", float(DETECT_CROP_X))
DEFAULT_CROP_Y = _env_float("NKAOUD_VISUAL_VEHICLE_CROP_Y", float(DETECT_CROP_Y))
DEFAULT_CROP_W = _env_float("NKAOUD_VISUAL_VEHICLE_CROP_W", float(DETECT_CROP_W))
DEFAULT_CROP_H = _env_float("NKAOUD_VISUAL_VEHICLE_CROP_H", float(DETECT_CROP_H))
# Upper bound for the detector rate. The detector is a nice(15) background
# process that shares the GPU with modeld (20 Hz, safety-critical), so this is
# the contention ceiling -- raising it trades responsiveness for the risk of
# starving the driving model. Default rate stays 1 Hz; the slider/env can go up
# to this cap.
MAX_DETECTOR_HZ = 20
DEFAULT_HZ = float(max(1, min(MAX_DETECTOR_HZ, int(_env_float("NKAOUD_VISUAL_VEHICLE_HZ", 1.0)))))

CAMERA_CROP_DEFAULT_OVERRIDES: dict[str, dict[str, float]] = {
  "driver": {
    "crop_x": 1365.0,
    "crop_y": 231.0,
    "crop_w": 543.0,
    "crop_h": 530.0,
    "hz": 5.0,
    "blocked_threshold": 0.90,
  },
  "wide": {
    "crop_x": 903.0,
    "crop_y": 509.0,
    "crop_w": 855.0,
    "crop_h": 281.0,
    "hz": 5.0,
    "blocked_threshold": 0.90,
  },
}

# Live tuning: a small JSON the web portal writes and the detector re-reads
# (mtime-gated) every frame, so ROI/gate/confidence can be adjusted on-road
# without a restart. Persisted under /data so it survives reboots. The file is
# keyed by camera ("road"/"wide"/"driver") since each has its own FOV/geometry.
TUNING_PATH = str(ARTIFACT_DIR / "vvd_tuning.json")

# Selectable camera sources (index matches the VisualVehicleDetectorCamera param
# and the UI selector order).
CAMERAS = ["road", "wide", "driver", "wide+driver"]
CAMERA_PARAM = "VisualVehicleDetectorCamera"

# "wide+driver" runs both classifier cameras concurrently, cycling one
# (camera, side) per frame so it stays at one inference/frame (GPU unchanged);
# each zone refreshes at ~1/4 the loop rate, each camera's L/R at ~1/2.
DUAL_CAMERA = "wide+driver"
DUAL_CAMERAS = ("wide", "driver")
STATE_SERVICE = "visualVehicleDetectorStateSP"
_frame_info_sm = None


def active_camera(params: Params | None = None) -> str:
  p = params if params is not None else Params()
  try:
    idx = int(p.get(CAMERA_PARAM, return_default=True) or 0)
  except (TypeError, ValueError):
    idx = 0
  return CAMERAS[idx] if 0 <= idx < len(CAMERAS) else CAMERAS[0]


def frame_info() -> dict[str, Any]:
  """Latest camera/frame dimensions from the detector service, so the
  tuning portal can auto-range the crop sliders to the live stream."""
  global _frame_info_sm
  if _frame_info_sm is None:
    _frame_info_sm = messaging.SubMaster([STATE_SERVICE])
  _frame_info_sm.update(0)
  if _frame_info_sm.recv_frame.get(STATE_SERVICE, 0) <= 0 or not _frame_info_sm.valid.get(STATE_SERVICE, False):
    return {}
  msg = _frame_info_sm[STATE_SERVICE]
  return {"camera": str(msg.camera), "frame_w": int(msg.crop.frameW), "frame_h": int(msg.crop.frameH)}


# ---- Dataset capture helpers (shared with the capture web portal) ----

def capture_set_request(enabled: bool, hz: float = DEFAULT_CAPTURE_HZ) -> None:
  """Create/remove the capture sentinel. Its content is the capture rate (Hz)."""
  if enabled:
    try:
      with open(CAPTURE_REQUEST_PATH, "w") as f:
        f.write(str(float(hz)))
    except OSError:
      cloudlog.exception("visual vehicle detector failed to set capture request")
  else:
    try:
      os.remove(CAPTURE_REQUEST_PATH)
    except FileNotFoundError:
      pass
    except OSError:
      cloudlog.exception("visual vehicle detector failed to clear capture request")


def capture_requested() -> bool:
  return os.path.exists(CAPTURE_REQUEST_PATH)


def capture_hz() -> float:
  try:
    return max(0.2, min(5.0, float(Path(CAPTURE_REQUEST_PATH).read_text().strip())))
  except Exception:
    return DEFAULT_CAPTURE_HZ


def capture_stats() -> dict[str, int]:
  count = 0
  total = 0
  try:
    with os.scandir(CAPTURE_DIR) as it:
      for entry in it:
        if entry.is_file() and entry.name.endswith(".jpg"):
          count += 1
          try:
            total += entry.stat().st_size
          except OSError:
            pass
  except FileNotFoundError:
    pass
  return {"count": count, "bytes": total}


def capture_files() -> list[str]:
  try:
    return sorted(str(e.path) for e in os.scandir(CAPTURE_DIR) if e.is_file() and e.name.endswith(".jpg"))
  except FileNotFoundError:
    return []


def capture_delete_all() -> int:
  removed = 0
  for path in capture_files():
    try:
      os.remove(path)
      removed += 1
    except OSError:
      pass
  return removed

# key -> (min, max) clamp range. Crop values are pixels; _crop_detector_region
# clamps them again to the actual frame size.
TUNING_KEYS: dict[str, tuple[float, float]] = {
  "right_x1": (0.0, 1.0),
  "right_y1": (0.0, 1.0),
  "right_x2": (0.0, 1.0),
  "right_y2": (0.0, 1.0),
  # Classifier reference ROI (fractions of the crop, in the right/un-augmented
  # orientation). The augmented side is auto-derived as its horizontal mirror,
  # so only this single ROI is tuned; it overrides model_config.json side_crops.
  "roi_x1": (0.0, 1.0),
  "roi_y1": (0.0, 1.0),
  "roi_x2": (0.0, 1.0),
  "roi_y2": (0.0, 1.0),
  "min_box_w": (0.0, 1.0),
  "min_box_h": (0.0, 1.0),
  "min_bottom_y": (0.0, 1.0),
  "confidence": (0.01, 0.99),
  "crop_x": (0.0, 4096.0),
  "crop_y": (0.0, 4096.0),
  "crop_w": (16.0, 4096.0),
  "crop_h": (16.0, 4096.0),
  "hz": (1.0, float(MAX_DETECTOR_HZ)),
  "blocked_threshold": (0.05, 0.95),
}

TUNING_DEFAULTS: dict[str, float] = {
  "right_x1": DEFAULT_RIGHT_X1,
  "right_y1": DEFAULT_RIGHT_Y1,
  "right_x2": DEFAULT_RIGHT_X2,
  "right_y2": DEFAULT_RIGHT_Y2,
  # Base ROI fallback (whole crop). Classifier cameras override these in
  # tuning_defaults() with the model's reference-side ROI.
  "roi_x1": 0.0,
  "roi_y1": 0.0,
  "roi_x2": 1.0,
  "roi_y2": 1.0,
  "min_box_w": DEFAULT_MIN_BOX_W,
  "min_box_h": DEFAULT_MIN_BOX_H,
  "min_bottom_y": DEFAULT_MIN_BOTTOM_Y,
  "confidence": DEFAULT_CONF,
  "crop_x": DEFAULT_CROP_X,
  "crop_y": DEFAULT_CROP_Y,
  "crop_w": DEFAULT_CROP_W,
  "crop_h": DEFAULT_CROP_H,
  "hz": DEFAULT_HZ,
  "blocked_threshold": DEFAULT_BLOCKED_THRESHOLD,
}


def tuning_defaults(camera: str | None = None) -> dict[str, float]:
  cam = camera or active_camera()
  values = dict(TUNING_DEFAULTS)
  values.update(CAMERA_CROP_DEFAULT_OVERRIDES.get(cam, {}))
  # Seed the classifier ROI defaults from the model's reference-side ROI so the
  # portal opens on the shipped ROI and "Reset" restores it. The augmented side
  # is auto-mirrored from this at inference time.
  if cam in CLASSIFIER_CAMERAS:
    rx1, ry1, rx2, ry2 = reference_roi_frac(load_classifier_recipe(cam))
    values.update({"roi_x1": rx1, "roi_y1": ry1, "roi_x2": rx2, "roi_y2": ry2})
  return values


def _load_all_tuning() -> dict[str, Any]:
  try:
    data = json.loads(Path(TUNING_PATH).read_text())
    return data if isinstance(data, dict) else {}
  except Exception:
    return {}


def load_tuning(camera: str | None = None) -> dict[str, float]:
  """Tuning values for `camera` (or the active one), defaults filled in."""
  cam = camera or active_camera()
  values = tuning_defaults(cam)
  data = _load_all_tuning().get(cam, {})
  if isinstance(data, dict):
    for key in TUNING_KEYS:
      if key in data:
        try:
          values[key] = float(data[key])
        except (TypeError, ValueError):
          continue
  return values


def save_tuning(updates: dict[str, Any], camera: str | None = None) -> dict[str, float]:
  """Merge `updates` into `camera`'s tuning, clamp to valid ranges, persist."""
  cam = camera or active_camera()
  all_data = _load_all_tuning()
  values = load_tuning(cam)
  for key, raw in updates.items():
    if key not in TUNING_KEYS:
      continue
    lo, hi = TUNING_KEYS[key]
    try:
      values[key] = min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
      continue
  all_data[cam] = values
  try:
    path = Path(TUNING_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_data, separators=(",", ":")))
  except OSError:
    cloudlog.exception("visual vehicle detector failed to write tuning file")
  return values


@dataclass
class Detection:
  xyxy: tuple[float, float, float, float]
  confidence: float
  class_id: int


class DebouncedFlag:
  def __init__(self, threshold: int = 2, maximum: int = 3) -> None:
    self.threshold = threshold
    self.maximum = maximum
    self.score = 0

  def update(self, raw: bool) -> bool:
    if raw:
      self.score = min(self.maximum, self.score + 1)
    else:
      self.score = max(0, self.score - 1)
    return self.score >= self.threshold

  def value(self) -> bool:
    """Current debounced state without consuming a sample (for the side that
    wasn't evaluated this frame)."""
    return self.score >= self.threshold


class VisualVehicleDetector:
  def __init__(self) -> None:
    migrate_legacy_artifacts()
    self.params = Params()
    self.pkl_path = os.getenv("NKAOUD_VISUAL_VEHICLE_PKL", DEFAULT_PKL_PATH)
    # Base (road/wide) model and a separate driver-camera model; the detector
    # switches between them with the camera. Falls back to the base model if the
    # driver model isn't compiled yet.
    self.base_pkl_path = self.pkl_path
    self.driver_pkl_path = os.getenv("NKAOUD_VISUAL_VEHICLE_PKL_DRIVER", DEFAULT_DRIVER_PKL_PATH)
    self._loaded_pkl: str | None = None
    self._loaded_pkl_mtime = -1.0
    self._cold_inference = False
    self.last_timing: dict[str, Any] = {}
    self._last_state_write_t = 0.0
    self._pending_state: tuple[bool, bool, float, dict[str, Any]] | None = None
    self.pm = messaging.PubMaster([STATE_SERVICE])

    # Dataset capture state.
    self._last_capture_t = 0.0
    self._capture_scan_t = 0.0
    self._capture_count = 0
    self._capture_bytes = 0
    self._capture_seq = 0
    self.onnx_path = os.getenv("NKAOUD_VISUAL_VEHICLE_ONNX", DEFAULT_ONNX_PATH)
    self.confidence = float(os.getenv("NKAOUD_VISUAL_VEHICLE_CONF", "0.35"))
    # Default the detector well below camera/modeld cadence on comma3x; the
    # slider/env can raise it up to MAX_DETECTOR_HZ.
    self.detector_hz = max(1, min(MAX_DETECTOR_HZ, int(os.getenv("NKAOUD_VISUAL_VEHICLE_HZ", "1"))))
    self.log_debug = False
    self.runtime = "none"

    # tinygrad pkl runtime fields
    self.Tensor = None
    self.pkl_model_run = None
    self.pkl_input_name = "images"
    self.pkl_input_dtype = None
    self.pkl_input_device = None
    self.expected_output_shape: tuple[int, ...] = ()

    # ONNX fallback runtime fields
    self.onnx_session = None
    self.onnx_input_name = ""

    # Per-camera classifier runtime (separate from the YOLO runtime above). Each
    # classifier camera has its own model file; the active one is loaded on
    # demand (swapped on camera switch) by _ensure_classifier.
    self.classifier_pkl_paths = {
      "driver": os.getenv("NKAOUD_VISUAL_VEHICLE_CLASSIFIER_PKL", DEFAULT_CLASSIFIER_PKL_PATH),
      "wide": os.getenv("NKAOUD_VISUAL_VEHICLE_CLASSIFIER_PKL_WIDE", DEFAULT_WIDE_CLASSIFIER_PKL_PATH),
    }
    self.classifier_onnx_paths = {
      "driver": os.getenv("NKAOUD_VISUAL_VEHICLE_CLASSIFIER_ONNX", DEFAULT_CLASSIFIER_ONNX_PATH),
      "wide": os.getenv("NKAOUD_VISUAL_VEHICLE_CLASSIFIER_ONNX_WIDE", DEFAULT_WIDE_CLASSIFIER_ONNX_PATH),
    }
    self.classifier_pkl_path = self.classifier_pkl_paths["driver"]
    self.classifier_onnx_path = self.classifier_onnx_paths["driver"]
    self.classifier_runtime = "none"
    self.classifier_pkl_run = None
    self.classifier_session = None
    self.classifier_input_name = "input_img"
    self.classifier_input_dtype = None
    self.classifier_input_device = None
    self._classifier_loaded: str | None = None
    self._classifier_mtime = -1.0
    # Resident per-camera classifier runtimes so wide+driver can both stay loaded
    # (no reload on the rotation). camera -> captured runtime slot.
    self._cls_cache: dict[str, dict[str, Any]] = {}
    self.blocked_threshold = float(os.getenv("NKAOUD_VISUAL_VEHICLE_BLOCKED_THRESHOLD", str(DEFAULT_BLOCKED_THRESHOLD)))

    # Throughput / latency measurement. _proc_ts holds the wall-clock time of
    # recent processed (new) frames so we can report the real sustained rate
    # (independent of the slider setpoint); _infer_ms is the last model inference
    # time in ms. Both are surfaced in the debug readout.
    self._proc_ts: deque[float] = deque(maxlen=20)
    self._measured_hz = 0.0
    self._infer_ms = 0.0

    self.input_shape: tuple[int, int] = (320, 320)  # width, height
    # Classifier input buffers, one per model input shape (reused across frames;
    # keyed by shape so driver 320x320 and wide 320x128 don't thrash each other).
    self._classifier_input_bufs: dict[tuple[int, ...], np.ndarray] = {}
    self.left_flag = DebouncedFlag()
    self.right_flag = DebouncedFlag()
    # Per-camera, per-zone classifier state (zones come from the model recipe:
    # left/right for driver and wide, a single 'center' for a single-zone model).
    # Snappy debounce (threshold=1, maximum=2): each per-zone sample is already
    # ~half/quarter-rate, so 1 sample asserts; maximum=2 keeps the clear quick.
    self._cls_flags: dict[str, dict[str, DebouncedFlag]] = {}
    self._cls_probs: dict[str, dict[str, float | None]] = {}
    self._recipes: dict[str, ClassifierRecipe] = {}
    self._recipe_mtime = -2.0
    self.startup_debug: dict[str, Any] = {"reason": "not_started", "runtime": self.runtime}
    self.last_model_debug: dict[str, Any] = {}
    self._preproc_dumped = False
    self._logged_buf_geometry = False

    # Live-tunable ROI / gate state, refreshed from TUNING_PATH each frame.
    self.right_roi: tuple[float, float, float, float] = (
      DEFAULT_RIGHT_X1, DEFAULT_RIGHT_Y1, DEFAULT_RIGHT_X2, DEFAULT_RIGHT_Y2,
    )
    # Classifier reference ROI (right/un-augmented orientation), live-tuned. The
    # augmented side is auto-mirrored from this in _preprocess_classifier.
    self.roi_frac_tuned: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    self.min_box_w_frac = DEFAULT_MIN_BOX_W
    self.min_box_h_frac = DEFAULT_MIN_BOX_H
    self.min_bottom_y_frac = DEFAULT_MIN_BOTTOM_Y
    self.crop_x = DEFAULT_CROP_X
    self.crop_y = DEFAULT_CROP_Y
    self.crop_w = DEFAULT_CROP_W
    self.crop_h = DEFAULT_CROP_H
    self.camera = active_camera(self.params)
    self._tuning_mtime = -2.0
    self._refresh_tuning()

  def _refresh_tuning(self, camera: str | None = None) -> None:
    """Re-read tuning for `camera` (or the active selection) if file/selection
    changed. The dual rotation passes the camera being processed this tick."""
    cam = camera if camera is not None else active_camera(self.params)
    try:
      mtime = os.path.getmtime(TUNING_PATH)
    except OSError:
      mtime = -1.0
    if cam == self.camera and mtime == self._tuning_mtime:
      return
    self.camera = cam
    self._tuning_mtime = mtime
    t = load_tuning(cam)
    self.right_roi = (t["right_x1"], t["right_y1"], t["right_x2"], t["right_y2"])
    self.roi_frac_tuned = (t["roi_x1"], t["roi_y1"], t["roi_x2"], t["roi_y2"])
    self.min_box_w_frac = t["min_box_w"]
    self.min_box_h_frac = t["min_box_h"]
    self.min_bottom_y_frac = t["min_bottom_y"]
    self.confidence = t["confidence"]
    self.crop_x = t["crop_x"]
    self.crop_y = t["crop_y"]
    self.crop_w = t["crop_w"]
    self.crop_h = t["crop_h"]
    self.detector_hz = max(1, min(MAX_DETECTOR_HZ, int(round(t["hz"]))))
    self.blocked_threshold = t["blocked_threshold"]

  @staticmethod
  def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
      return float(value)
    except (TypeError, ValueError):
      return default

  @staticmethod
  def _safe_int(value: Any, default: int = 0) -> int:
    try:
      return int(value)
    except (TypeError, ValueError):
      return default

  @staticmethod
  def _set_zone_list(zone_list, zones: list[dict[str, Any]]) -> None:
    for i, zone in enumerate(zones):
      zone_list[i].name = str(zone.get("name", ""))
      zone_list[i].blocked = bool(zone.get("blocked", False))
      prob = zone.get("p")
      if prob is not None:
        zone_list[i].probability = float(prob)
        zone_list[i].hasProbability = True

  def _build_state_message(self, left: bool, right: bool, monotonic_time: float, debug: dict[str, Any]):
    msg = messaging.new_message(STATE_SERVICE, valid=True)
    state = msg.visualVehicleDetectorStateSP
    state.leftBlocked = bool(left)
    state.rightBlocked = bool(right)
    state.monotonicTime = float(monotonic_time)
    state.reason = str(debug.get("reason", ""))
    state.runtime = str(debug.get("runtime", ""))
    state.camera = str(debug.get("camera", ""))
    state.side = str(debug.get("side", ""))
    state.hz = self._safe_float(debug.get("hz"))
    state.frameId = self._safe_int(debug.get("frame_id"), -1)
    state.dual = bool(debug.get("dual", False))
    state.inputShape = [int(v) for v in debug.get("input_shape", [])]
    state.detections = self._safe_int(debug.get("detections"))
    state.bestConf = self._safe_float(debug.get("best_conf"))
    state.leftScore = self._safe_int(debug.get("left_score"))
    state.rightScore = self._safe_int(debug.get("right_score"))
    state.parser = str(debug.get("parser", ""))
    state.outputShape = [int(v) for v in debug.get("output_shape", [])]
    state.rawBestObj = self._safe_float(debug.get("raw_best_obj"))
    state.rawBestCls = self._safe_float(debug.get("raw_best_cls"))
    state.rawBestConf = self._safe_float(debug.get("raw_best_conf"))
    state.rawBestClassId = self._safe_int(debug.get("raw_best_class_id"), -1)
    state.rawBestVehicle = bool(debug.get("raw_best_vehicle", False))
    state.rawBestLeftRoi = bool(debug.get("raw_best_left_roi", False))
    state.rawBestRightRoi = bool(debug.get("raw_best_right_roi", False))
    state.rawBestBox = [float(v) for v in debug.get("raw_best_box", [])]
    if "raw_best_center_x" in debug and "raw_best_center_y" in debug:
      state.rawBestCenterX = self._safe_float(debug.get("raw_best_center_x"))
      state.rawBestCenterY = self._safe_float(debug.get("raw_best_center_y"))
      state.rawBestCenterValid = True
    state.pklPath = str(debug.get("pkl_path", ""))
    state.onnxPath = str(debug.get("onnx_path", ""))
    state.pklExists = bool(debug.get("pkl_exists", False))
    state.onnxExists = bool(debug.get("onnx_exists", False))
    state.modelName = str((debug.get("timing", {}) or {}).get("model", ""))
    state.error = str(debug.get("error", ""))

    timing = debug.get("timing", {}) or {}
    state.timing.cropRgbMs = self._safe_float(timing.get("crop_rgb_ms"))
    state.timing.preprocessMs = self._safe_float(timing.get("preprocess_ms"))
    state.timing.inferMs = self._safe_float(timing.get("infer_ms"))
    state.timing.stateWriteMs = self._safe_float(timing.get("state_write_ms"))
    state.timing.measuredHz = self._safe_float(timing.get("measured_hz"))
    state.timing.modelLoadMs = self._safe_float(timing.get("model_load_ms"))
    state.timing.firstInfMs = self._safe_float(timing.get("first_inf_ms"))
    state.timing.camConnectMs = self._safe_float(timing.get("cam_connect_ms"))

    crop = debug.get("crop", {}) or {}
    state.crop.cropX = self._safe_int(crop.get("crop_x"))
    state.crop.cropY = self._safe_int(crop.get("crop_y"))
    state.crop.cropW = self._safe_int(crop.get("crop_w"))
    state.crop.cropH = self._safe_int(crop.get("crop_h"))
    state.crop.frameW = self._safe_int(crop.get("frame_w"))
    state.crop.frameH = self._safe_int(crop.get("frame_h"))

    capture = debug.get("capture", {}) or {}
    state.capture.on = bool(capture.get("on", False))
    state.capture.saved = self._safe_int(capture.get("saved"))

    classifier = debug.get("classifier", {}) or {}
    state.classifier.active = bool(classifier.get("active", False))
    state.classifier.side = str(classifier.get("side", ""))
    state.classifier.threshold = self._safe_float(classifier.get("threshold"))
    state.classifier.leftBlocked = bool(classifier.get("left_blocked", False))
    state.classifier.rightBlocked = bool(classifier.get("right_blocked", False))
    zones = classifier.get("zones", []) or []
    state.classifier.init("zones", len(zones))
    self._set_zone_list(state.classifier.zones, zones)

    wide_zones = ((debug.get("cameras", {}) or {}).get("wide", {}) or {}).get("zones", []) or []
    driver_zones = ((debug.get("cameras", {}) or {}).get("driver", {}) or {}).get("zones", []) or []
    state.init("wideZones", len(wide_zones))
    self._set_zone_list(state.wideZones, wide_zones)
    state.init("driverZones", len(driver_zones))
    self._set_zone_list(state.driverZones, driver_zones)
    return msg

  def _write_state(self, left: bool, right: bool, debug: dict[str, Any] | None = None, force: bool = False) -> None:
    monotonic_time = time.monotonic()
    debug_payload = debug or {}
    self._pending_state = (bool(left), bool(right), monotonic_time, debug_payload)
    if not force and monotonic_time - self._last_state_write_t < STATE_PUBLISH_INTERVAL:
      return

    pending = self._pending_state
    if pending is None:
      return

    t0 = time.monotonic()
    try:
      send_msg = self._build_state_message(*pending)
      self.pm.send(STATE_SERVICE, send_msg)
      self._last_state_write_t = pending[2]
      self._pending_state = None
      self.last_timing["state_write_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
    except Exception:
      cloudlog.exception("visual vehicle detector failed to publish state")

  def _set_startup_debug(self, **debug: Any) -> None:
    payload = {
      "runtime": self.runtime,
      "pkl_path": self.pkl_path,
      "onnx_path": self.onnx_path,
      "pkl_exists": os.path.exists(self.pkl_path),
      "onnx_exists": os.path.exists(self.onnx_path),
      "allow_onnx": self.params.get_bool("VisualVehicleDetectorAllowOnnx"),
    }
    payload.update(debug)
    self.startup_debug = payload
    self._write_state(False, False, payload, force=True)

  def _set_tinygrad_device_env(self) -> None:
    # Match openpilot's tinygrad modeld convention when possible.
    try:
      from openpilot.selfdrive.modeld.tinygrad_helpers import set_tinygrad_backend_from_compiled_flags
      set_tinygrad_backend_from_compiled_flags()
    except Exception:
      pass

    if "DEV" not in os.environ:
      try:
        from openpilot.system.hardware import TICI
        os.environ["DEV"] = "QCOM" if TICI else "CPU"
      except Exception:
        os.environ["DEV"] = "CPU"

  def _load_runtime(self) -> bool:
    if os.path.exists(self.pkl_path) and self._load_tinygrad_pkl():
      return True

    allow_onnx = self.params.get_bool("VisualVehicleDetectorAllowOnnx")
    if allow_onnx and os.path.exists(self.onnx_path) and self._load_onnx_runtime():
      return True

    reason = "pkl_missing"
    if os.path.exists(self.pkl_path):
      reason = "pkl_load_failed"
    elif allow_onnx and not os.path.exists(self.onnx_path):
      reason = "pkl_and_onnx_missing"

    self._set_startup_debug(reason=reason, allow_onnx=allow_onnx)
    return False

  def _model_path_for(self, camera: str) -> str:
    if camera == "driver" and os.path.exists(self.driver_pkl_path):
      return self.driver_pkl_path
    return self.base_pkl_path

  def _ensure_model_for(self, camera: str) -> None:
    """Load the model matching the camera, reloading (and timing) on change.
    Also hot-reloads when the model file itself changes (e.g. recompiled),
    so no camera toggle is needed. Only applies to the tinygrad runtime."""
    if camera in CLASSIFIER_CAMERAS:
      return  # classifier cameras don't use a YOLO pkl
    if self.runtime == "onnx_cpu":
      return
    desired = self._model_path_for(camera)
    try:
      mtime = os.path.getmtime(desired)
    except OSError:
      mtime = -1.0
    if desired == self._loaded_pkl and self.pkl_model_run is not None and mtime == self._loaded_pkl_mtime:
      return
    self.pkl_path = desired
    self._load_tinygrad_pkl()

  def _load_tinygrad_pkl(self) -> bool:
    load_t0 = time.monotonic()
    try:
      self._set_tinygrad_device_env()
      from tinygrad.tensor import Tensor  # pylint: disable=import-error
      self.Tensor = Tensor

      with open(self.pkl_path, "rb") as f:
        self.pkl_model_run = pickle.load(f)

      meta_path = Path(self.pkl_path).with_suffix(".json")
      if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        self.pkl_input_name = str(meta.get("input_name", self.pkl_input_name))
        shape = meta.get("input_shape", [1, 3, 320, 320])
        if len(shape) >= 4:
          self.input_shape = (int(shape[3]), int(shape[2]))
        out_shape = meta.get("output_shape", [])
        if isinstance(out_shape, list) and out_shape:
          self.expected_output_shape = tuple(int(v) for v in out_shape)

      captured = getattr(self.pkl_model_run, "captured", None)
      if captured is not None:
        names = list(getattr(captured, "expected_names", []) or [])
        if names:
          self.pkl_input_name = names[0]
        infos = list(getattr(captured, "expected_input_info", []) or [])
        if infos:
          info = infos[0]
          if len(info) > 2:
            self.pkl_input_dtype = info[2]
          if len(info) > 3:
            self.pkl_input_device = info[3]

      self.runtime = "tinygrad_pkl"
      self._loaded_pkl = self.pkl_path
      try:
        self._loaded_pkl_mtime = os.path.getmtime(self.pkl_path)
      except OSError:
        self._loaded_pkl_mtime = -1.0
      self._cold_inference = True  # next inference pays the GPU warmup cost
      load_ms = round((time.monotonic() - load_t0) * 1000.0, 1)
      self.last_timing["model"] = os.path.basename(self.pkl_path)
      self.last_timing["model_load_ms"] = load_ms
      cloudlog.warning("visual vehicle detector loaded tinygrad pkl %s input=%s shape=%s load_ms=%.1f",
                       self.pkl_path, self.pkl_input_name, self.input_shape, load_ms)
      return True
    except Exception as e:
      self.runtime = "tinygrad_pkl_failed"
      self._set_startup_debug(reason="pkl_load_failed", error=str(e))
      cloudlog.exception("visual vehicle detector failed to load tinygrad pkl")
      return False

  def _load_onnx_runtime(self) -> bool:
    try:
      import onnxruntime as ort  # pylint: disable=import-error
      self.onnx_session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
      model_input = self.onnx_session.get_inputs()[0]
      self.onnx_input_name = model_input.name
      shape = list(model_input.shape)
      h = shape[2] if len(shape) >= 4 and isinstance(shape[2], int) else 320
      w = shape[3] if len(shape) >= 4 and isinstance(shape[3], int) else 320
      self.input_shape = (int(w), int(h))
      self.runtime = "onnx_cpu"
      cloudlog.warning("visual vehicle detector loaded ONNX fallback %s input=%s", self.onnx_path, self.input_shape)
      return True
    except Exception as e:
      self.runtime = "onnx_failed"
      self._set_startup_debug(reason="onnx_load_failed", error=str(e))
      cloudlog.exception("visual vehicle detector failed to initialize ONNX fallback")
      return False

  # ---- Camera occupancy classifier (replaces YOLO for classifier cameras) ----

  def classifier_available(self) -> bool:
    return self.classifier_runtime in ("tinygrad_pkl", "onnx_cpu")

  def _recipe(self, camera: str) -> ClassifierRecipe:
    """Per-camera preprocessing recipe, cached and reloaded when the combined
    model_config.json changes on disk."""
    try:
      mtime = os.path.getmtime(MODEL_CONFIG_PATH)
    except OSError:
      mtime = -1.0
    if mtime != self._recipe_mtime:
      self._recipe_mtime = mtime
      self._recipes = {}
    if camera not in self._recipes:
      self._recipes[camera] = load_classifier_recipe(camera)
    return self._recipes[camera]

  def _classifier_buf(self, recipe: ClassifierRecipe) -> np.ndarray:
    shape = (1, 3, recipe.res_h, recipe.res_w)
    buf = self._classifier_input_bufs.get(shape)
    if buf is None:
      buf = np.empty(shape, dtype=np.float32)
      self._classifier_input_bufs[shape] = buf
    return buf

  def _ensure_classifier(self, camera: str) -> None:
    """Load (and hot-reload on file change / camera switch) the classifier model
    for `camera`. Prefers the compiled tinygrad pkl; falls back to ONNX when
    VisualVehicleDetectorAllowOnnx is set, mirroring the YOLO runtime selection."""
    self.classifier_pkl_path = pkl = self.classifier_pkl_paths.get(camera, self.classifier_pkl_paths["driver"])
    self.classifier_onnx_path = onnx = self.classifier_onnx_paths.get(camera, self.classifier_onnx_paths["driver"])
    allow_onnx = self.params.get_bool("VisualVehicleDetectorAllowOnnx")
    if os.path.exists(pkl):
      desired, mtime = pkl, os.path.getmtime(pkl)
    elif allow_onnx and os.path.exists(onnx):
      desired, mtime = onnx, os.path.getmtime(onnx)
    else:
      desired, mtime = None, -1.0

    # If this camera's runtime is already resident and current, just re-activate
    # it (cheap) -- no reload. This is what lets wide+driver both stay loaded.
    slot = self._cls_cache.get(camera)
    if slot is not None and slot.get("loaded") == desired and slot.get("mtime") == mtime \
       and (slot.get("pkl_run") is not None or slot.get("session") is not None):
      self._restore_cls_slot(slot)
      return

    if desired == pkl:
      if self._load_classifier_pkl(pkl):
        self._classifier_loaded, self._classifier_mtime = pkl, mtime
      self._cls_cache[camera] = self._capture_cls_slot()
      return
    if desired == onnx:
      if self._load_classifier_onnx(onnx):
        self._classifier_loaded, self._classifier_mtime = onnx, mtime
      self._cls_cache[camera] = self._capture_cls_slot()
      return

    # Nothing loadable -- drop any stale runtime so the readout shows "missing".
    self.classifier_runtime = "none"
    self.classifier_pkl_run = None
    self.classifier_session = None
    self._classifier_loaded = None
    self._cls_cache.pop(camera, None)

  def _capture_cls_slot(self) -> dict[str, Any]:
    return {"pkl_run": self.classifier_pkl_run, "session": self.classifier_session,
            "input_name": self.classifier_input_name, "input_dtype": self.classifier_input_dtype,
            "input_device": self.classifier_input_device, "runtime": self.classifier_runtime,
            "loaded": self._classifier_loaded, "mtime": self._classifier_mtime}

  def _restore_cls_slot(self, slot: dict[str, Any]) -> None:
    self.classifier_pkl_run = slot["pkl_run"]
    self.classifier_session = slot["session"]
    self.classifier_input_name = slot["input_name"]
    self.classifier_input_dtype = slot["input_dtype"]
    self.classifier_input_device = slot["input_device"]
    self.classifier_runtime = slot["runtime"]
    self._classifier_loaded = slot["loaded"]
    self._classifier_mtime = slot["mtime"]

  def _load_classifier_pkl(self, path: str) -> bool:
    try:
      self._set_tinygrad_device_env()
      from tinygrad.tensor import Tensor  # pylint: disable=import-error
      self.Tensor = Tensor
      with open(path, "rb") as f:
        self.classifier_pkl_run = pickle.load(f)
      captured = getattr(self.classifier_pkl_run, "captured", None)
      if captured is not None:
        names = list(getattr(captured, "expected_names", []) or [])
        if names:
          self.classifier_input_name = names[0]
        infos = list(getattr(captured, "expected_input_info", []) or [])
        if infos and len(infos[0]) > 2:
          self.classifier_input_dtype = infos[0][2]
          if len(infos[0]) > 3:
            self.classifier_input_device = infos[0][3]
      self.classifier_session = None
      self.classifier_runtime = "tinygrad_pkl"
      cloudlog.warning("visual vehicle classifier loaded tinygrad pkl %s input=%s", path, self.classifier_input_name)
      return True
    except Exception:
      self.classifier_runtime = "tinygrad_pkl_failed"
      cloudlog.exception("visual vehicle classifier failed to load tinygrad pkl")
      return False

  def _load_classifier_onnx(self, path: str) -> bool:
    try:
      import onnxruntime as ort  # pylint: disable=import-error
      self.classifier_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
      self.classifier_input_name = self.classifier_session.get_inputs()[0].name
      self.classifier_pkl_run = None
      self.classifier_runtime = "onnx_cpu"
      cloudlog.warning("visual vehicle classifier loaded ONNX %s input=%s", path, self.classifier_input_name)
      return True
    except Exception:
      self.classifier_runtime = "onnx_failed"
      cloudlog.exception("visual vehicle classifier failed to load ONNX")
      return False

  def _preprocess_classifier(self, detector_rgb: np.ndarray,
                             recipe: ClassifierRecipe,
                             side: str | None = None) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Reproduce the model's training preprocessing: cut the ROI region,
    bilinear-resize to res_w x res_h, normalize, NCHW. Verified byte-identical
    to the per-model training preprocess for both models.

    The live-tuned reference ROI (self.roi_frac_tuned) drives the crop: it is
    applied directly on the reference side and horizontally mirrored on the
    augmented side, so one tuned ROI covers both. This overrides any side_crops
    from model_config.json (which is now just the shipped default seed)."""
    h, w = detector_rgb.shape[:2]
    # One reference ROI, auto-mirrored for the augmented side.
    rx1, ry1, rx2, ry2 = self.roi_frac_tuned
    fx1, fy1, fx2, fy2 = (1.0 - rx2, ry1, 1.0 - rx1, ry2) if side_is_augmented(recipe, side) \
      else (rx1, ry1, rx2, ry2)
    x1 = int(np.clip(round(fx1 * w), 0, w - 1))
    y1 = int(np.clip(round(fy1 * h), 0, h - 1))
    x2 = int(np.clip(round(fx2 * w), x1 + 1, w))
    y2 = int(np.clip(round(fy2 * h), y1 + 1, h))
    roi = detector_rgb[y1:y2, x1:x2]
    # Bilinear stretch to match the training resize (PIL is already a dep here).
    from PIL import Image
    resized = np.asarray(
      Image.fromarray(roi, "RGB").resize((recipe.res_w, recipe.res_h), Image.BILINEAR)
    )
    x = self._classifier_buf(recipe)
    np.copyto(x[0], resized.transpose(2, 0, 1), casting="unsafe")
    if CLASSIFIER_NORM == "signed":
      x[0] /= 127.5
      x[0] -= 1.0                                           # [-1, 1] (old TinyCNN)
    elif CLASSIFIER_NORM == "unit":
      x[0] /= 255.0                                         # [0, 1]
    else:  # "imagenet" (default) -- per-model mean/std from the recipe
      x[0] /= 255.0
      x[0] -= recipe.mean[:, None, None]
      x[0] /= recipe.std[:, None, None]
    return x, {"roi_x1": x1, "roi_y1": y1, "roi_x2": x2, "roi_y2": y2}, resized

  def _run_classifier(self, inp: np.ndarray, pos_index: int) -> float:
    """Return p_car = softmax(logits)[pos_index]."""
    if self.classifier_runtime == "tinygrad_pkl" and self.classifier_pkl_run is not None and self.Tensor is not None:
      tensor = self.Tensor(inp, device=self.classifier_input_device).realize() if self.classifier_input_device \
        else self.Tensor(inp).realize()
      if self.classifier_input_dtype is not None:
        tensor = tensor.cast(self.classifier_input_dtype)
      out = self.classifier_pkl_run(**{self.classifier_input_name: tensor})
      logits = self._tensor_to_numpy(out)
    elif self.classifier_runtime == "onnx_cpu" and self.classifier_session is not None:
      logits = self.classifier_session.run(None, {self.classifier_input_name: inp})[0]
    else:
      raise RuntimeError("classifier runtime not loaded")
    l = np.asarray(logits, dtype=np.float32).reshape(-1)[:2]  # 2-class logits
    e = np.exp(l - l.max())
    return float((e / e.sum())[pos_index])

  def _ensure_cls_zones(self, camera: str, sides: tuple[str, ...]) -> None:
    """(Re)build the per-camera per-zone debounce flags when the zone set changes
    (e.g. a left/right model vs a single-zone model). Per-camera so wide and
    driver keep independent left/right state in the dual rotation."""
    if tuple(self._cls_flags.get(camera, {})) != tuple(sides):
      self._cls_flags[camera] = {s: DebouncedFlag(threshold=1, maximum=2) for s in sides}
      self._cls_probs[camera] = dict.fromkeys(sides)

  def _classifier_zones(self, camera: str) -> list[dict[str, Any]]:
    flags = self._cls_flags.get(camera, {})
    probs = self._cls_probs.get(camera, {})
    return [{"name": s, "blocked": bool(f.value()),
             "p": round(probs[s], 4) if probs.get(s) is not None else None}
            for s, f in flags.items()]

  def _update_classifier(self, rgb_full: np.ndarray, detector_rgb: np.ndarray,
                         crop_debug: dict[str, int | str], side: str | None,
                         recipe: ClassifierRecipe, camera: str) -> tuple[bool, bool, dict[str, Any]]:
    """Classifier path: whole-crop car/no-car. Per-frame it evaluates one `side`
    (zone) of `camera`; the others hold their last debounced state. Zones come
    from the model recipe -- left/right or a single 'center'. Returns the
    (left, right) top-level flags mapped from the zone states."""
    self._ensure_cls_zones(camera, recipe.sides)
    side = side if side in self._cls_flags.get(camera, {}) else recipe.sides[0]

    def _topline() -> tuple[bool, bool]:
      blk = {z["name"]: z["blocked"] for z in self._classifier_zones(camera)}
      left = blk.get("left", blk.get("center", False))
      right = blk.get("right", blk.get("center", False))
      return left, right

    base = {"runtime": self.classifier_runtime, "crop": crop_debug, "side": side,
            "input_shape": [1, 3, recipe.res_h, recipe.res_w],
            "hz": self.detector_hz, "timing": dict(self.last_timing),
            "capture": {"on": capture_requested(), "saved": self._capture_count}}
    if not self.classifier_available():
      return *_topline(), {**base, "reason": "classifier_missing",
                           "classifier": {"active": False, "model": self.classifier_pkl_path,
                                          "allow_onnx": self.params.get_bool("VisualVehicleDetectorAllowOnnx")}}
    try:
      t0 = time.monotonic()
      inp, roi, model_input = self._preprocess_classifier(detector_rgb, recipe, side)
      self.last_timing["preprocess_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
      t0 = time.monotonic()
      p_car = self._run_classifier(inp, recipe.pos_index)
      self._infer_ms = (time.monotonic() - t0) * 1000.0
      self.last_timing["infer_ms"] = round(self._infer_ms, 1)
      base["timing"] = dict(self.last_timing)  # refresh: base snapshotted pre-inference
    except Exception as e:
      cloudlog.exception("visual vehicle classifier inference failed")
      return *_topline(), {**base, "reason": "classifier_error", "error": str(e)}

    # Live preview (Crop & Rate / classifier input), only while a portal holds
    # the sentinel. rgb_full is None on the hot path (no portal).
    if rgb_full is not None and os.path.exists(PREVIEW_REQUEST_PATH):
      try:
        self._write_classifier_previews(rgb_full, detector_rgb, crop_debug, roi, model_input)
      except Exception:
        cloudlog.exception("visual vehicle classifier preview write failed")

    # Update only the zone evaluated this frame; the others persist.
    self._cls_flags[camera][side].update(p_car >= self.blocked_threshold)
    self._cls_probs[camera][side] = p_car
    left_blocked, right_blocked = _topline()
    base["classifier"] = {
      "active": True,
      "side": side,
      "zones": self._classifier_zones(camera),
      "left_blocked": bool(left_blocked),
      "right_blocked": bool(right_blocked),
      "threshold": round(self.blocked_threshold, 3),
      "roi": [roi["roi_x1"], roi["roi_y1"], roi["roi_x2"], roi["roi_y2"]],
      "src_crop": [int(detector_rgb.shape[1]), int(detector_rgb.shape[0])],
      "expected_src": [recipe.src_w, recipe.src_h],
      "crop_mode": "roi_inset",  # tuned reference ROI, auto-mirrored per side
      "roi_ref": [round(v, 4) for v in self.roi_frac_tuned],
      "augmented": side_is_augmented(recipe, side),
      "norm": CLASSIFIER_NORM,
      "pos_index": recipe.pos_index,
    }
    base["reason"] = "ok"
    return left_blocked, right_blocked, base

  def _write_classifier_previews(self, rgb_full: np.ndarray, detector_rgb: np.ndarray,
                                 crop_debug: dict[str, int | str], roi: dict[str, int],
                                 model_input: np.ndarray) -> None:
    """Driver-cam equivalent of the YOLO stage previews, feeding the same portal
    image routes: full frame + crop box, the crop with the classifier ROI box,
    and the exact 320x320 tensor (pre-normalization) the model sees."""
    from PIL import Image, ImageDraw

    full_img = Image.fromarray(rgb_full, "RGB")
    fd = ImageDraw.Draw(full_img)
    cx, cy = int(crop_debug.get("crop_x", 0)), int(crop_debug.get("crop_y", 0))
    cw = int(crop_debug.get("crop_w", detector_rgb.shape[1]))
    ch = int(crop_debug.get("crop_h", detector_rgb.shape[0]))
    fd.rectangle((cx, cy, cx + cw - 1, cy + ch - 1), outline=(255, 255, 0), width=4)
    self._draw_text(fd, (cx + 6, cy + 6), f"driver crop {cw}x{ch}", (255, 255, 0))
    full_img.save(PREVIEW_FULL_FRAME_CROP_PATH)

    crop_img = Image.fromarray(detector_rgb, "RGB")
    cd = ImageDraw.Draw(crop_img)
    cd.rectangle((roi["roi_x1"], roi["roi_y1"], roi["roi_x2"] - 1, roi["roi_y2"] - 1),
                 outline=(0, 200, 200), width=3)
    self._draw_text(cd, (roi["roi_x1"] + 4, roi["roi_y1"] + 4), "classifier ROI", (0, 200, 200))
    crop_img.save(PREVIEW_DETECTOR_CROP_PATH)

    # The 320x320 the model classifies (this is what blocked/clear is decided on).
    Image.fromarray(model_input, "RGB").save(PREVIEW_MODEL_INPUT_PATH)

  def _buf_geometry(self, buf: VisionBuf) -> tuple[int, int, int, int, int, int]:
    """NV12 buffer geometry: (width, height, stride, uv_offset, uv_height,
    uv_plane_size). Logs the layout once for debugging (shared by the full-frame
    and crop paths)."""
    width = int(buf.width)
    height = int(buf.height)
    stride = int(buf.stride)
    uv_offset = int(buf.uv_offset)
    # uv_height calculation copied from snapshot.py.
    uv_height = ((height // 2) + 15) // 16 * 16
    uv_plane_size = stride * uv_height
    if not self._logged_buf_geometry:
      try:
        dlen = len(buf.data)
      except Exception:
        dlen = -1
      cloudlog.warning(
        "visual vehicle detector NV12 buf geometry: "
        "width=%d height=%d stride=%d uv_offset=%d uv_height=%d "
        "uv_plane_size=%d data_len=%d",
        width, height, stride, uv_offset, uv_height, uv_plane_size, dlen,
      )
      try:
        import json as _json
        with open(BUF_GEOMETRY_PATH, "w") as f:
          _json.dump({
            "width": width, "height": height, "stride": stride,
            "uv_offset": uv_offset, "uv_height": uv_height,
            "uv_plane_size": uv_plane_size, "data_len": dlen,
            "uv_offset_matches_y_plane": uv_offset == stride * height,
            "uv_offset_matches_y_plane_aligned": uv_offset == stride * (((height + 31) // 32) * 32),
          }, f)
      except OSError:
        cloudlog.exception("visual vehicle detector failed to write geometry json")
      self._logged_buf_geometry = True
    return width, height, stride, uv_offset, uv_height, uv_plane_size

  def _vipc_crop_to_rgb(self, buf: VisionBuf, rect: tuple[int, int, int, int], mirror: bool) -> np.ndarray | None:
    """Hot path: read and convert ONLY the detector crop from the NV12 buffer.

    This avoids full-frame materialization and also avoids a few large temporary
    arrays from the generic YUV path (`repeat`, `stack`, extra int32 copies) by
    expanding U/V directly into full-res working buffers for the crop only.
    """
    try:
      width, _height, stride, uv_offset, uv_height, uv_plane_size = self._buf_geometry(buf)
      crop_x, crop_y, crop_w, crop_h = rect
      sx = (width - crop_x - crop_w) if mirror else crop_x
      bx = 2 * (sx // 2)  # even byte column where the interleaved U/V crop starts
      flat = np.frombuffer(buf.data, dtype=np.uint8)

      # Y: a (rows x stride) view; slice only the crop and widen once.
      y2d = flat[:uv_offset].reshape((-1, stride))
      y = y2d[crop_y:crop_y + crop_h, sx:sx + crop_w].astype(np.int32)

      # UV: crop the row band and byte columns that cover the crop, then expand
      # the half-res chroma directly to full-res int32 planes for the crop.
      uv2d = flat[uv_offset:uv_offset + uv_plane_size].reshape((uv_height, stride))
      block = uv2d[crop_y // 2:(crop_y + crop_h) // 2, bx:bx + crop_w]
      u_half = block[:, 0::2]
      v_half = block[:, 1::2]

      u = np.empty((crop_h, crop_w), dtype=np.int32)
      v = np.empty((crop_h, crop_w), dtype=np.int32)
      u[0::2, 0::2] = u_half
      u[0::2, 1::2] = u_half
      u[1::2, 0::2] = u_half
      u[1::2, 1::2] = u_half
      v[0::2, 0::2] = v_half
      v[0::2, 1::2] = v_half
      v[1::2, 0::2] = v_half
      v[1::2, 1::2] = v_half

      d = u - 128
      e = v - 128
      rgb = np.empty((crop_h, crop_w, 3), dtype=np.uint8)
      rgb[..., 0] = np.clip((256 * y + 359 * e + 128) >> 8, 0, 255)
      rgb[..., 1] = np.clip((256 * y - 88 * d - 183 * e + 128) >> 8, 0, 255)
      rgb[..., 2] = np.clip((256 * y + 454 * d + 128) >> 8, 0, 255)
      if mirror:
        return np.ascontiguousarray(rgb[:, ::-1])
      return rgb
    except Exception:
      cloudlog.exception("visual vehicle detector failed crop plane extraction")
      return None

  def _vipc_to_yuv_planes(self, buf: VisionBuf) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Full-frame (y, u, v) int32 planes from the NV12 buffer, or None. y is
    full-resolution; u and v are quarter-resolution. Used for the preview path;
    the hot loop uses _vipc_crop_to_rgb to avoid touching the whole frame."""
    try:
      width, height, stride, uv_offset, _uv_height, uv_plane_size = self._buf_geometry(buf)
      # Slice buf.data as a memoryview FIRST, then wrap with np.array. int32 (not
      # int16) is required: 256*Y overflows int16 for any Y >= 128, wraps
      # negative, and after >> 8 biases the BT.601 sums (dark / magenta).
      y = np.array(buf.data[:uv_offset], dtype=np.uint8) \
            .reshape((-1, stride))[:height, :width].astype(np.int32)
      uv_data = buf.data[uv_offset:uv_offset + uv_plane_size]
      # Standard NV12 chroma interleave: even bytes are U (Cb), odd bytes are V (Cr).
      u = np.array(uv_data[::2], dtype=np.uint8) \
            .reshape((-1, stride // 2))[:height // 2, :width // 2].astype(np.int32)
      v = np.array(uv_data[1::2], dtype=np.uint8) \
            .reshape((-1, stride // 2))[:height // 2, :width // 2].astype(np.int32)
      return y, u, v
    except Exception:
      cloudlog.exception("visual vehicle detector failed frame plane extraction")
      return None

  @staticmethod
  def _yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray, full_range: bool) -> np.ndarray:
    # u, v are quarter-resolution; upsample to match y.
    u_full = u.repeat(2, axis=0).repeat(2, axis=1)
    v_full = v.repeat(2, axis=0).repeat(2, axis=1)
    # Trim to y's shape in case y has an odd dimension.
    h, w = y.shape[:2]
    u_full = u_full[:h, :w]
    v_full = v_full[:h, :w]
    d = u_full - 128
    e = v_full - 128
    if full_range:
      # BT.601 full range (camerad emits CAM_COLOR_SPACE_BT601_FULL; see spectra.cc).
      r = (256 * y + 359 * e + 128) >> 8
      g = (256 * y -  88 * d - 183 * e + 128) >> 8
      b = (256 * y + 454 * d + 128) >> 8
    else:
      # BT.601 limited range (Y in 16..235): used here only for visual comparison.
      c = np.clip(y - 16, 0, None)
      r = (298 * c + 409 * e + 128) >> 8
      g = (298 * c - 100 * d - 208 * e + 128) >> 8
      b = (298 * c + 516 * d + 128) >> 8
    return np.clip(np.stack((r, g, b), axis=-1), 0, 255).astype(np.uint8)

  def _vipc_to_rgb(self, buf: VisionBuf, full_range: bool = True) -> np.ndarray | None:
    planes = self._vipc_to_yuv_planes(buf)
    if planes is None:
      return None
    try:
      return self._yuv_to_rgb(*planes, full_range=full_range)
    except Exception:
      cloudlog.exception("visual vehicle detector failed YUV->RGB conversion")
      return None

  @staticmethod
  def _resize_nn(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    src_h, src_w = rgb.shape[:2]
    if src_h == out_h and src_w == out_w:
      return rgb
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, out_w)).astype(np.int32), 0, src_w - 1)
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, out_h)).astype(np.int32), 0, src_h - 1)
    return rgb[y_idx][:, x_idx]

  def _write_preview_pair(self, rgb_full: np.ndarray, rgb_limited: np.ndarray) -> None:
    from PIL import Image
    canvas_full, *_ = self._letterbox(rgb_full)
    canvas_limited, *_ = self._letterbox(rgb_limited)
    Image.fromarray(canvas_full, "RGB").save(PREVIEW_PNG_PATH_FULL)
    Image.fromarray(canvas_limited, "RGB").save(PREVIEW_PNG_PATH_LIMITED)
    # Keep the legacy single-image path pointing at the production (full-range)
    # view so any older client still works.
    Image.fromarray(canvas_full, "RGB").save(PREVIEW_PNG_PATH)

  def _write_raw_planes(self, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    """Dump Y, U, V as grayscale PNGs for visual diagnosis of stride / row
    alignment / range issues that aren't visible in the composited RGB."""
    from PIL import Image
    y8 = np.clip(y, 0, 255).astype(np.uint8)
    u8 = np.clip(u, 0, 255).astype(np.uint8)
    v8 = np.clip(v, 0, 255).astype(np.uint8)
    Image.fromarray(y8, "L").save(PREVIEW_RAW_Y_PATH)
    Image.fromarray(u8, "L").save(PREVIEW_RAW_U_PATH)
    Image.fromarray(v8, "L").save(PREVIEW_RAW_V_PATH)

  @staticmethod
  def _draw_text(draw: Any, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    # PIL text drawing should never break the detector loop due to font/encoding issues.
    try:
      draw.text(xy, text, fill=fill)
    except Exception:
      pass

  def _write_detector_previews(self, rgb_full: np.ndarray, detector_rgb: np.ndarray,
                               crop_debug: dict[str, int | str], detections: list[Detection]) -> None:
    """Write both views needed for debugging cropped inference:

    1. Full camera frame with the fixed 928x416 detector crop rectangle drawn.
    2. Raw detector crop with the left/right ROI split drawn in crop coordinates.
    3. Exact model input image after letterbox/resize, with ROI and detection boxes.

    This is intentionally separate from _write_preview_pair(), which still writes
    the full-frame color-conversion A/B preview for YUV debugging.
    """
    from PIL import Image, ImageDraw

    # 1) Full frame + crop rectangle: confirms we are cutting the intended area
    # from the original wide-road frame.
    full_img = Image.fromarray(rgb_full, "RGB")
    full_draw = ImageDraw.Draw(full_img)
    crop_x = int(crop_debug.get("crop_x", 0))
    crop_y = int(crop_debug.get("crop_y", 0))
    crop_w = int(crop_debug.get("crop_w", detector_rgb.shape[1]))
    crop_h = int(crop_debug.get("crop_h", detector_rgb.shape[0]))
    full_rect = (crop_x, crop_y, crop_x + crop_w - 1, crop_y + crop_h - 1)
    full_draw.rectangle(full_rect, outline=(255, 255, 0), width=4)
    self._draw_text(full_draw, (crop_x + 6, crop_y + 6), f"detector crop {crop_w}x{crop_h}", (255, 255, 0))
    full_img.save(PREVIEW_FULL_FRAME_CROP_PATH)

    # Pass/fail per detection: green = trips the right flag (inside the live
    # right ROI AND passes the size/position gate), gray = ignored. Computed in
    # crop coordinates with the live thresholds so it tracks the tuning portal.
    cw_full, ch_full = detector_rgb.shape[1], detector_rgb.shape[0]
    passes = [self._box_in_roi(det.xyxy, self.right_roi, cw_full, ch_full) for det in detections]
    pass_color = (0, 255, 0)
    fail_color = (150, 150, 150)

    # 2) Raw detector crop + live right ROI band in crop coordinates.
    crop_img = Image.fromarray(detector_rgb, "RGB")
    crop_draw = ImageDraw.Draw(crop_img)
    cw, ch = crop_img.size
    left_x2 = int(round(LEFT_ROI[2] * cw))
    crop_draw.rectangle((0, 0, max(0, left_x2 - 1), ch - 1), outline=(255, 0, 0), width=2)
    self._draw_roi_band(crop_draw, self.right_roi, cw, ch)
    self._draw_gate_lines(crop_draw, cw, ch)
    self._draw_text(crop_draw, (4, 4), "LEFT", (255, 0, 0))

    for det, ok in zip(detections, passes):
      x1, y1, x2, y2 = det.xyxy
      rect = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
      color = pass_color if ok else fail_color
      crop_draw.rectangle(rect, outline=color, width=3 if ok else 1)
      self._draw_text(crop_draw, (rect[0], max(0, rect[1] - 12)), f"{det.class_id}:{det.confidence:.2f}", color)
    crop_img.save(PREVIEW_DETECTOR_CROP_PATH)

    # 3) Exact tensor image going to YOLO after letterbox/resize. This should be
    # 480x224 when the compiled PKL metadata reports input_shape [1,3,224,480].
    canvas, pad_x, pad_y, scale = self._letterbox(detector_rgb)
    model_img = Image.fromarray(canvas, "RGB")
    model_draw = ImageDraw.Draw(model_img)
    mw, mh = model_img.size

    def _to_model(rx: float, ry: float) -> tuple[int, int]:
      # ROI fraction is of the crop; map crop px -> letterboxed model px.
      return int(round(rx * cw_full * scale + pad_x)), int(round(ry * ch_full * scale + pad_y))

    lx2 = int(round(LEFT_ROI[2] * cw_full * scale + pad_x))
    model_draw.rectangle((int(pad_x), int(pad_y), max(0, lx2 - 1), mh - 1), outline=(255, 0, 0), width=1)
    rb_x1, rb_y1 = _to_model(self.right_roi[0], self.right_roi[1])
    rb_x2, rb_y2 = _to_model(self.right_roi[2], self.right_roi[3])
    model_draw.rectangle((rb_x1, rb_y1, rb_x2, rb_y2), outline=(0, 200, 200), width=2)

    for det, ok in zip(detections, passes):
      x1, y1, x2, y2 = det.xyxy
      mx1 = int(round(x1 * scale + pad_x))
      my1 = int(round(y1 * scale + pad_y))
      mx2 = int(round(x2 * scale + pad_x))
      my2 = int(round(y2 * scale + pad_y))
      color = pass_color if ok else fail_color
      model_draw.rectangle((mx1, my1, mx2, my2), outline=color, width=2 if ok else 1)
      self._draw_text(model_draw, (mx1, max(0, my1 - 12)), f"{det.class_id}:{det.confidence:.2f}", color)
    model_img.save(PREVIEW_MODEL_INPUT_PATH)

  def _draw_roi_band(self, draw: Any, roi: tuple[float, float, float, float], w: int, h: int) -> None:
    rx1 = int(round(roi[0] * w))
    ry1 = int(round(roi[1] * h))
    rx2 = int(round(roi[2] * w))
    ry2 = int(round(roi[3] * h))
    draw.rectangle((rx1, ry1, max(rx1, rx2 - 1), max(ry1, ry2 - 1)), outline=(0, 200, 200), width=3)
    self._draw_text(draw, (rx1 + 4, ry1 + 4), "RIGHT ROI", (0, 200, 200))

  def _draw_gate_lines(self, draw: Any, w: int, h: int) -> None:
    # Horizontal line marking the min-bottom-y gate: a box whose bottom is above
    # this line is rejected as "too far".
    y = int(round(self.min_bottom_y_frac * h))
    draw.line((0, y, w - 1, y), fill=(255, 255, 0), width=1)
    self._draw_text(draw, (4, min(h - 14, y + 2)), "min bottom-y", (255, 255, 0))

  def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, int, int, float]:
    w, h = self.input_shape
    src_h, src_w = rgb.shape[:2]
    scale = min(w / float(src_w), h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = self._resize_nn(rgb, new_w, new_h)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    pad_x = (w - new_w) // 2
    pad_y = (h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, pad_x, pad_y, scale

  def _preprocess(self, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    src_h, src_w = rgb.shape[:2]
    canvas, pad_x, pad_y, scale = self._letterbox(rgb)
    # Optional one-shot dump (env-controlled).
    dump = os.getenv("NKAOUD_VISUAL_VEHICLE_DUMP_PREPROC", "")
    if dump and not self._preproc_dumped:
      try:
        from PIL import Image
        Image.fromarray(canvas, "RGB").save(dump)
        self._preproc_dumped = True
        cloudlog.warning("visual vehicle detector wrote preproc dump to %s", dump)
      except Exception:
        cloudlog.exception("visual vehicle detector preproc dump failed")
        self._preproc_dumped = True
    x = canvas.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    prep = {
      "scale": scale,
      "pad_x": pad_x,
      "pad_y": pad_y,
      "src_w": src_w,
      "src_h": src_h,
    }
    return np.ascontiguousarray(x), prep

  def _tensor_to_numpy(self, out: Any) -> np.ndarray:
    if isinstance(out, dict):
      out = next(iter(out.values()))
    if isinstance(out, (list, tuple)):
      out = out[0]
    if hasattr(out, "numpy"):
      arr = out.realize().numpy() if hasattr(out, "realize") else out.numpy()
    elif hasattr(out, "contiguous"):
      arr = out.contiguous().realize().uop.base.buffer.numpy()
    else:
      arr = np.asarray(out)
    arr = np.asarray(arr)
    if arr.ndim == 1 and self.expected_output_shape:
      expected_size = int(np.prod(self.expected_output_shape))
      if arr.size == expected_size:
        arr = arr.reshape(self.expected_output_shape)
    return arr

  def _run_model(self, rgb: np.ndarray) -> list[Detection]:
    inp, prep = self._preprocess(rgb)
    self.last_model_debug = {}
    # Time the first inference after a (re)load: this pays the GPU warmup cost.
    cold = self._cold_inference
    t0 = time.monotonic()
    dets = self._infer(inp, prep)
    self._infer_ms = (time.monotonic() - t0) * 1000.0
    self.last_timing["infer_ms"] = round(self._infer_ms, 1)
    if cold and self.pkl_model_run is not None:
      self._cold_inference = False
      self.last_timing["first_inf_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
      cloudlog.warning("visual vehicle detector first_inference model=%s ms=%.1f",
                       self.last_timing.get("model"), self.last_timing["first_inf_ms"])
    return dets

  def _infer(self, inp: np.ndarray, prep: dict[str, float | int]) -> list[Detection]:
    if self.runtime == "tinygrad_pkl":
      if self.pkl_model_run is None or self.Tensor is None:
        return []
      if self.pkl_input_device:
        tensor = self.Tensor(inp, device=self.pkl_input_device).realize()
      else:
        tensor = self.Tensor(inp).realize()
      if self.pkl_input_dtype is not None:
        tensor = tensor.cast(self.pkl_input_dtype)
      output = self.pkl_model_run(**{self.pkl_input_name: tensor})
      arr = self._tensor_to_numpy(output)
      self.last_model_debug["output_shape"] = list(np.asarray(arr).shape)
      return self._parse_yolo_output(arr, prep)

    if self.runtime == "onnx_cpu" and self.onnx_session is not None:
      outputs = self.onnx_session.run(None, {self.onnx_input_name: inp})
      self.last_model_debug["output_shape"] = list(np.asarray(outputs[0]).shape)
      return self._parse_yolo_output(outputs[0], prep)

    return []

  def _set_best_raw_debug(self, cls_i: int, score_f: float, mapped: tuple[float, float, float, float],
                          image_w: int, image_h: int) -> None:
    raw_vehicle = cls_i in VEHICLE_CLASS_IDS
    raw_left_roi = self._box_in_roi(mapped, LEFT_ROI, image_w, image_h)
    raw_right_roi = self._box_in_roi(mapped, self.right_roi, image_w, image_h)
    x1, y1, x2, y2 = mapped
    self.last_model_debug.update({
      "raw_best_class_id": cls_i,
      "raw_best_conf": round(score_f, 5),
      "raw_best_box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
      "raw_best_vehicle": raw_vehicle,
      "raw_best_left_roi": raw_left_roi,
      "raw_best_right_roi": raw_right_roi,
      "raw_best_center_x": round((x1 + x2) * 0.5 / max(1.0, image_w), 5),
      "raw_best_center_y": round((y1 + y2) * 0.5 / max(1.0, image_h), 5),
    })

  def _map_box_to_source(self, box: tuple[float, float, float, float], prep: dict[str, float | int]) -> tuple[float, float, float, float]:
    scale = float(prep["scale"])
    pad_x = float(prep["pad_x"])
    pad_y = float(prep["pad_y"])
    src_w = float(prep["src_w"])
    src_h = float(prep["src_h"])
    x1, y1, x2, y2 = box
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    return (
      float(np.clip(x1, 0.0, src_w - 1.0)),
      float(np.clip(y1, 0.0, src_h - 1.0)),
      float(np.clip(x2, 0.0, src_w - 1.0)),
      float(np.clip(y2, 0.0, src_h - 1.0)),
    )

  def _parse_yolo_output(self, output: np.ndarray, prep: dict[str, float | int]) -> list[Detection]:
    arr = np.squeeze(np.asarray(output))
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1] and arr.shape[0] >= 6:
      arr = arr.T

    detections: list[Detection] = []
    if arr.ndim != 2:
      return detections
    image_w = int(prep["src_w"])
    image_h = int(prep["src_h"])

    # Case A: already NMSed [x1, y1, x2, y2, score, class].
    if arr.shape[1] == 6:
      if arr.shape[0]:
        best_idx = int(np.argmax(arr[:, 4]))
        best_row = arr[best_idx]
        best_box = self._map_box_to_source(tuple(float(v) for v in best_row[:4]), prep)
        self._set_best_raw_debug(int(best_row[5]), float(best_row[4]), best_box, image_w, image_h)
      for row in arr:
        score = float(row[4])
        cls = int(row[5])
        if score < self.confidence or cls not in VEHICLE_CLASS_IDS:
          continue
        x1, y1, x2, y2 = [float(v) for v in row[:4]]
        detections.append(Detection(self._map_box_to_source((x1, y1, x2, y2), prep), score, cls))
      return detections

    # Case B1: YOLOv5 raw output [cx, cy, w, h, objectness, class_scores...].
    # The default download is yolov5n.onnx, so we must multiply objectness by
    # the best class score. If we treat objectness as class 0, vehicle detection
    # becomes wrong or unstable.
    if arr.shape[1] == 85:
      boxes = arr[:, :4]
      obj = arr[:, 4]
      class_scores = arr[:, 5:]
      cls_ids = np.argmax(class_scores, axis=1)
      cls_confs = class_scores[np.arange(class_scores.shape[0]), cls_ids]
      confs = obj * cls_confs
      self.last_model_debug.update({
        "parser": "yolov5_raw",
        "raw_best_obj": round(float(np.max(obj)), 5) if obj.size else 0.0,
        "raw_best_cls": round(float(np.max(cls_confs)), 5) if cls_confs.size else 0.0,
        "raw_best_conf": round(float(np.max(confs)), 5) if confs.size else 0.0,
      })
      if confs.size:
        best_idx = int(np.argmax(confs))
        best_cls_i = int(cls_ids[best_idx])
        cx, cy, bw, bh = [float(v) for v in boxes[best_idx]]
        best_mapped = self._map_box_to_source((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), prep)
        self._set_best_raw_debug(best_cls_i, float(confs[best_idx]), best_mapped, image_w, image_h)

      for box, cls, score in zip(boxes, cls_ids, confs):
        cls_i = int(cls)
        score_f = float(score)
        if score_f < self.confidence or cls_i not in VEHICLE_CLASS_IDS:
          continue
        cx, cy, bw, bh = [float(v) for v in box]
        mapped = self._map_box_to_source((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), prep)
        detections.append(Detection(mapped, score_f, cls_i))
      return detections

    # Case B2: YOLOv8 / YOLO11 raw output [cx, cy, w, h, class_scores...].
    # Ultralytics v8/v11 exports commonly produce [1, 84, N] for COCO.
    if arr.shape[1] >= 4 + max(VEHICLE_CLASS_IDS) + 1:
      boxes = arr[:, :4]
      scores = arr[:, 4:]
      cls_ids = np.argmax(scores, axis=1)
      confs = scores[np.arange(scores.shape[0]), cls_ids]
      self.last_model_debug.update({
        "parser": "yolov8_raw",
        "raw_best_conf": round(float(np.max(confs)), 5) if confs.size else 0.0,
      })
      if confs.size:
        best_idx = int(np.argmax(confs))
        best_cls_i = int(cls_ids[best_idx])
        cx, cy, bw, bh = [float(v) for v in boxes[best_idx]]
        best_mapped = self._map_box_to_source((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), prep)
        self._set_best_raw_debug(best_cls_i, float(confs[best_idx]), best_mapped, image_w, image_h)

      for box, cls, score in zip(boxes, cls_ids, confs):
        cls_i = int(cls)
        score_f = float(score)
        if score_f < self.confidence or cls_i not in VEHICLE_CLASS_IDS:
          continue
        cx, cy, bw, bh = [float(v) for v in box]
        mapped = self._map_box_to_source((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), prep)
        detections.append(Detection(mapped, score_f, cls_i))
    return detections
  

  def _crop_rect(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Live-tunable detector crop box (pixels), clamped to the frame and
    even-aligned so a YUV420 crop-before-convert is pixel-exact (chroma is
    half-resolution). Seeded by NKAOUD_VISUAL_VEHICLE_CROP_X/Y/W/H."""
    crop_w = max(2, min(int(round(self.crop_w)), frame_w))
    crop_h = max(2, min(int(round(self.crop_h)), frame_h))
    crop_x = max(0, min(int(round(self.crop_x)), frame_w - crop_w))
    crop_y = max(0, min(int(round(self.crop_y)), frame_h - crop_h))
    crop_x -= crop_x % 2
    crop_y -= crop_y % 2
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    crop_w = max(2, min(crop_w, frame_w - crop_x))
    crop_h = max(2, min(crop_h, frame_h - crop_y))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    return crop_x, crop_y, crop_w, crop_h

  def _crop_debug(self, frame_w: int, frame_h: int, rect: tuple[int, int, int, int]) -> dict[str, int | str]:
    crop_x, crop_y, crop_w, crop_h = rect
    return {
      "crop_mode": "tunable",
      "crop_x": int(crop_x),
      "crop_y": int(crop_y),
      "crop_w": int(crop_w),
      "crop_h": int(crop_h),
      "crop_x2_exclusive": int(crop_x + crop_w),
      "crop_y2_exclusive": int(crop_y + crop_h),
      "frame_w": int(frame_w),
      "frame_h": int(frame_h),
      "roi_model_w": int(ROI_MODEL_W),
      "roi_model_h": int(ROI_MODEL_H),
    }

  def _box_in_roi(self, box: tuple[float, float, float, float], roi: tuple[float, float, float, float], image_w: int, image_h: int) -> bool:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    rx1 *= image_w
    rx2 *= image_w
    ry1 *= image_h
    ry2 *= image_h
    cx = 0.5 * (x1 + x2)
    bottom = y2
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    # Distance gate: adjacent-lane cars are large and low in the frame; far-lane
    # cars (two or three lanes over) are small and high. Reject anything too
    # small or whose bottom edge sits too high up toward the horizon.
    if width < image_w * self.min_box_w_frac or height < image_h * self.min_box_h_frac:
      return False
    if bottom < image_h * self.min_bottom_y_frac:
      return False
    return rx1 <= cx <= rx2 and ry1 <= bottom <= ry2

  def _orient(self, rgb: np.ndarray) -> np.ndarray:
    # The driver camera is mirrored vs. the world (selfie view). Un-mirror it so
    # image-right == world-right, matching the road/wide cameras. Done here, on
    # the full RGB frame, so the crop, ROI, detections and previews all operate
    # on the corrected image.
    if self.camera == "driver":
      return np.ascontiguousarray(rgb[:, ::-1])
    return rgb

  def _maybe_capture(self, detector_rgb: np.ndarray, side: str | None = None) -> None:
    """Save the camera crop as JPEG for training while the capture portal is
    open and the car is onroad. Throttled to the requested Hz and capped to
    protect device storage."""
    if not capture_requested():
      return
    if self.params.get_bool("IsOffroad"):
      return  # onroad only

    now = time.monotonic()
    if now - self._last_capture_t < 1.0 / capture_hz():
      return

    # Re-scan periodically so the cap reflects external deletes and respects the
    # storage limit without statting the directory every frame.
    if now - self._capture_scan_t > 2.0:
      stats = capture_stats()
      self._capture_count = stats["count"]
      self._capture_bytes = stats["bytes"]
      self._capture_scan_t = now

    if self._capture_count >= CAPTURE_MAX_FILES or self._capture_bytes >= CAPTURE_MAX_BYTES:
      return  # cap reached; the portal status surfaces this

    try:
      from PIL import Image
      CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
      self._capture_seq += 1
      # Tag the side so alternating driver-cam crops (left frames are mirrored
      # to look like right) stay distinguishable in a future dataset.
      side_tag = f"_{side}" if side else ""
      fname = f"cap_{self.camera}{side_tag}_{int(time.time() * 1000)}_{self._capture_seq}.jpg"
      path = CAPTURE_DIR / fname
      Image.fromarray(detector_rgb, "RGB").save(path, "JPEG", quality=90)
      self._last_capture_t = now
      self._capture_count += 1
      self._capture_bytes += path.stat().st_size
    except Exception:
      cloudlog.exception("visual vehicle detector capture failed")

  def _record_proc(self) -> None:
    """Record this processed (new) frame and update the measured throughput.
    update() is called once per fresh camera frame, so the spacing of these
    timestamps is the real sustained rate, independent of the slider setpoint."""
    self._proc_ts.append(time.monotonic())
    if len(self._proc_ts) >= 2:
      span = self._proc_ts[-1] - self._proc_ts[0]
      self._measured_hz = (len(self._proc_ts) - 1) / span if span > 0 else 0.0
    self.last_timing["measured_hz"] = round(self._measured_hz, 1)
    self.last_timing["infer_ms"] = round(self._infer_ms, 1)

  def update(self, buf: VisionBuf, camera: str, side: str | None) -> tuple[bool, bool, dict[str, Any]]:
    """Process one (camera, side) rotation step. `side` is the zone for classifier
    cameras (left/right/center, driven by the run loop) or None for YOLO road."""
    self._refresh_tuning(camera)
    self._record_proc()
    # Classifier cameras (driver, wide) run the car classifier; road runs YOLO.
    # Re-activate the camera's resident model (cheap) / hot-reload on file change.
    recipe = None
    if camera in CLASSIFIER_CAMERAS:
      self._ensure_classifier(camera)
      recipe = self._recipe(camera)
    else:
      self._ensure_model_for(camera)
    frame_w, frame_h = int(buf.width), int(buf.height)
    rect = self._crop_rect(frame_w, frame_h)
    crop_debug = self._crop_debug(frame_w, frame_h, rect)
    # The driver cam is a selfie, so base mirror un-mirrors it. Per-side mirror
    # then XORs on top: side_crops[side].mirror when present (new schema), else
    # the legacy mirror_left flag (toggles only on the LEFT zone).
    mirror = camera == "driver"
    if recipe is not None and side_is_augmented(recipe, side):
      mirror = not mirror
    # Extracting + converting the whole ~2MP frame every tick is the bottleneck,
    # and the model only sees the crop, so read just the crop's bytes from the
    # NV12 buffer on the hot path. A portal preview needs the whole oriented
    # frame, so fall back to full extraction only when one is open.
    preview_on = os.path.exists(PREVIEW_REQUEST_PATH)
    planes = None
    try:
      t0 = time.monotonic()
      if preview_on:
        planes = self._vipc_to_yuv_planes(buf)
        if planes is None:
          raise RuntimeError("plane extraction failed")
        # Match the hot path's per-side orientation (mirror toggles by side) so
        # the preview crop is the same image the classifier sees.
        rgb = self._yuv_to_rgb(*planes, full_range=True)
        if mirror:
          rgb = np.ascontiguousarray(rgb[:, ::-1])
        crop_x, crop_y, crop_w, crop_h = rect
        detector_rgb = np.ascontiguousarray(rgb[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w])
      else:
        rgb = None
        detector_rgb = self._vipc_crop_to_rgb(buf, rect, mirror)
        if detector_rgb is None:
          raise RuntimeError("crop extraction failed")
      self.last_timing["crop_rgb_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
    except Exception:
      cloudlog.exception("visual vehicle detector failed YUV->RGB conversion")
      left = self.left_flag.update(False)
      right = self.right_flag.update(False)
      return left, right, {"reason": "frame_convert_failed", "runtime": self.runtime}

    self._maybe_capture(detector_rgb, side)

    # Classifier cameras: whole-crop car/no-car, not YOLO boxes.
    if recipe is not None:
      return self._update_classifier(rgb, detector_rgb, crop_debug, side, recipe, camera)

    # Run YOLO on the fixed crop only, not on the full camera frame.
    detections = self._run_model(detector_rgb)
    image_h, image_w = detector_rgb.shape[:2]

    # Live preview: keep the original full-frame color-conversion previews and
    # add cropped-inference previews. Only active while the dialog/web server
    # holds the request sentinel.
    if os.path.exists(PREVIEW_REQUEST_PATH):
      try:
        if planes is None:  # preview turned on this tick; build the full frame now
          planes = self._vipc_to_yuv_planes(buf)
        if rgb is None:
          rgb = self._orient(self._yuv_to_rgb(*planes, full_range=True))
        rgb_limited = self._orient(self._yuv_to_rgb(*planes, full_range=False))
        self._write_preview_pair(rgb, rgb_limited)
        self._write_raw_planes(*planes)
        self._write_detector_previews(rgb, detector_rgb, crop_debug, detections)
      except Exception:
        cloudlog.exception("visual vehicle detector live preview write failed")

    # LEFT_ROI is a fixed reference sliver; self.right_roi is the live-tunable
    # adjacent-lane band. Both are normalized fractions of the crop that
    # _run_model returns, and the size/position gate is applied in _box_in_roi.
    raw_left = any(self._box_in_roi(det.xyxy, LEFT_ROI, image_w, image_h) for det in detections)
    raw_right = any(self._box_in_roi(det.xyxy, self.right_roi, image_w, image_h) for det in detections)
    left = self.left_flag.update(raw_left)
    right = self.right_flag.update(raw_right)

    best_conf = max((d.confidence for d in detections), default=0.0)
    debug = {
      "reason": "ok",
      "runtime": self.runtime,
      "pkl_path": self.pkl_path,
      "onnx_path": self.onnx_path,
      "input_shape": list(self.input_shape),
      "crop": crop_debug,
      "preview_paths": {
        "full_frame_crop": PREVIEW_FULL_FRAME_CROP_PATH,
        "detector_crop": PREVIEW_DETECTOR_CROP_PATH,
        "model_input": PREVIEW_MODEL_INPUT_PATH,
      },
      "left_roi_norm": [round(float(v), 5) for v in LEFT_ROI],
      "right_roi_norm": [round(float(v), 5) for v in self.right_roi],
      "gate": {
        "min_box_w": round(self.min_box_w_frac, 5),
        "min_box_h": round(self.min_box_h_frac, 5),
        "min_bottom_y": round(self.min_bottom_y_frac, 5),
        "confidence": round(self.confidence, 5),
      },
      "raw_left": raw_left,
      "raw_right": raw_right,
      "left_score": self.left_flag.score,
      "right_score": self.right_flag.score,
      "detections": len(detections),
      "best_conf": round(float(best_conf), 3),
      "hz": self.detector_hz,
      "timing": dict(self.last_timing),
      "capture": {"on": capture_requested(), "saved": self._capture_count},
    }
    debug.update(self.last_model_debug)
    return left, right, debug

  @staticmethod
  def _stream_for(camera: str) -> Any:
    return {
      "road": VisionStreamType.VISION_STREAM_ROAD,
      "wide": VisionStreamType.VISION_STREAM_WIDE_ROAD,
      "driver": VisionStreamType.VISION_STREAM_DRIVER,
    }.get(camera, VisionStreamType.VISION_STREAM_ROAD)

  def _cameras_for_mode(self, mode: str) -> list[str]:
    return list(DUAL_CAMERAS) if mode == DUAL_CAMERA else [mode]

  def _rotation_for(self, cams: list[str]) -> list[tuple[str, str | None]]:
    """Ordered (camera, side) steps. Classifier cams contribute one step per zone
    (left/right, or a single center); the YOLO road cam contributes one step with
    side=None. wide+driver -> wide-L, wide-R, driver-L, driver-R."""
    rot: list[tuple[str, str | None]] = []
    for c in cams:
      if c in CLASSIFIER_CAMERAS:
        rot.extend((c, s) for s in self._recipe(c).sides)
      else:
        rot.append((c, None))
    return rot

  def _write_dual_state(self, cam_debug: dict[str, dict], active_cam: str, active_side: str | None) -> None:
    """Combined state for wide+driver: each camera's zones under debug['cameras'],
    with top-level left/right = OR across cameras."""
    cams: dict[str, Any] = {}
    any_left = any_right = False
    for c in DUAL_CAMERAS:
      cls = (cam_debug.get(c, {}) or {}).get("classifier", {}) or {}
      cams[c] = cls
      any_left = any_left or bool(cls.get("left_blocked"))
      any_right = any_right or bool(cls.get("right_blocked"))
    ad = cam_debug.get(active_cam, {}) or {}
    debug = {
      "dual": True,
      "reason": ad.get("reason", "ok"),
      "runtime": self.classifier_runtime,
      "cameras": cams,
      "camera": active_cam,
      "side": f"{active_cam}-{active_side}" if active_side else active_cam,
      "hz": MAX_DETECTOR_HZ,
      "timing": ad.get("timing", dict(self.last_timing)),
      "capture": ad.get("capture", {}),
      "frame_id": ad.get("frame_id"),
    }
    self._write_state(any_left, any_right, debug)

  def run(self) -> None:
    self.log_debug = self.params.get_bool("VisualVehicleDetectorLogDebug")
    runtime_ok = self._load_runtime()
    for c in self._cameras_for_mode(active_camera(self.params)):
      if c in CLASSIFIER_CAMERAS:
        self._ensure_classifier(c)  # classifier cameras run without a YOLO model
    if not runtime_ok and not self.classifier_available():
      rk = Ratekeeper(1.0)
      while True:
        # Keep the real startup failure visible instead of overwriting it with
        # a generic inactive heartbeat.
        self._write_state(False, False, self.startup_debug)
        rk.keep_time()

    # Multi-camera rotation: one (camera, side) per tick, so the GPU still does
    # one inference per frame. Single modes have a 1-2 step rotation; "wide+driver"
    # cycles wide-L, wide-R, driver-L, driver-R. Both classifier models stay
    # resident (see the _ensure_classifier cache), so the rotation never reloads.
    clients: dict[str, Any] = {}
    last_frame_id: dict[tuple[str, str | None], int] = {}
    cam_debug: dict[str, dict] = {}
    rotation: list[tuple[str, str | None]] = []
    rot_idx = 0
    active_mode = None
    rk = Ratekeeper(float(self.detector_hz))
    current_hz = self.detector_hz

    while True:
      mode = active_camera(self.params)
      cams = self._cameras_for_mode(mode)
      if mode != active_mode:
        active_mode = mode
        for c in list(clients):
          if c not in cams:
            clients.pop(c, None)
        last_frame_id = {}
        cam_debug = {}
        rotation = self._rotation_for(cams)
        rot_idx = 0
        self.left_flag = DebouncedFlag()
        self.right_flag = DebouncedFlag()
        self._cls_flags = {}
        self._cls_probs = {}
        self._proc_ts.clear()  # don't carry the switch gap into the measured rate

      # Connect any camera in the current mode that isn't connected yet, and
      # preload+cache its model so the rotation never pays a reload.
      waiting = False
      for c in cams:
        if c in clients:
          continue
        stream = self._stream_for(c)
        if stream not in VisionIpcClient.available_streams("camerad", block=False):
          self._write_state(False, False, {"reason": "camera_unavailable", "camera": c, "runtime": self.runtime})
          waiting = True
          break
        conn_t0 = time.monotonic()
        vc = VisionIpcClient("camerad", stream, True)
        if not vc.connect(False):
          self._write_state(False, False, {"reason": "waiting_for_vipc", "camera": c, "runtime": self.runtime})
          waiting = True
          break
        clients[c] = vc
        self.last_timing["cam_connect_ms"] = round((time.monotonic() - conn_t0) * 1000.0, 1)
        if c in CLASSIFIER_CAMERAS:
          self._ensure_classifier(c)
        else:
          self._ensure_model_for(c)
        cloudlog.warning("visual vehicle detector connect camera=%s stream=%s size=%sx%s runtime=%s",
                         c, stream, vc.width, vc.height, self.runtime)
      if waiting or not rotation:
        time.sleep(0.1)
        continue

      # Dual rotation runs at a fixed cap so the loop rate doesn't oscillate
      # between the two cameras' per-camera Hz; single modes follow their slider.
      target_hz = MAX_DETECTOR_HZ if mode == DUAL_CAMERA else self.detector_hz
      if target_hz != current_hz:
        current_hz = target_hz
        rk = Ratekeeper(float(current_hz))
      new_log_debug = self.params.get_bool("VisualVehicleDetectorLogDebug")
      if new_log_debug != self.log_debug:
        self.log_debug = new_log_debug

      # Advance the rotation and process that (camera, side). Dedup per step so a
      # camera's left and right can share one frame (different crops) but a stale
      # frame isn't re-run for the same step.
      cam, side = rotation[rot_idx]
      rot_idx = (rot_idx + 1) % len(rotation)
      vc = clients.get(cam)
      if vc is None:
        rk.keep_time()
        continue
      buf = vc.recv()
      if buf is None or vc.frame_id == last_frame_id.get((cam, side)):
        rk.keep_time()
        continue
      last_frame_id[(cam, side)] = vc.frame_id
      try:
        left, right, debug = self.update(buf, cam, side)
        debug["frame_id"] = int(vc.frame_id)
        debug["camera"] = cam
        cam_debug[cam] = debug
        if len(cams) > 1:
          self._write_dual_state(cam_debug, cam, side)
        else:
          self._write_state(left, right, debug)
        if self.log_debug:
          cloudlog.info("visual_vehicle_detector cam=%s side=%s left=%s right=%s", cam, side, left, right)
      except Exception as e:
        cloudlog.exception("visual vehicle detector update failed")
        self._write_state(False, False, {"reason": "exception", "runtime": self.runtime, "error": str(e)})
      rk.keep_time()


def main() -> None:
  try:
    os.nice(15)
  except Exception:
    pass
  VisualVehicleDetector().run()


if __name__ == "__main__":
  main()
