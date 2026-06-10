#!/usr/bin/env python3
"""
Standalone visual adjacent-vehicle detector for UI/debug validation.

This daemon is independent from navigation and controls:
  - It does not publish a desire.
  - It does not block lane changes.
  - It only writes /tmp/nkaoud_visual_vehicle_detector.json for the UI readout.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import MODEL_DIR as ARTIFACT_DIR, migrate_legacy_artifacts

STATE_PATH = Path("/tmp/nkaoud_visual_vehicle_detector.json")
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
DEFAULT_ONNX_PATH = str(ARTIFACT_DIR / "visual_vehicle_detector.onnx")

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

# Normalized crop ROIs: x1, y1, x2, y2.
LEFT_ROI = (0.0 / ROI_MODEL_W, 0.0 / ROI_MODEL_H, 32.0 / ROI_MODEL_W, 224.0 / ROI_MODEL_H)
RIGHT_ROI = (32.0 / ROI_MODEL_W, 0.0 / ROI_MODEL_H, 480.0 / ROI_MODEL_W, 224.0 / ROI_MODEL_H)


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


class VisualVehicleDetector:
  def __init__(self) -> None:
    migrate_legacy_artifacts()
    self.params = Params()
    self.pkl_path = os.getenv("NKAOUD_VISUAL_VEHICLE_PKL", DEFAULT_PKL_PATH)
    self.onnx_path = os.getenv("NKAOUD_VISUAL_VEHICLE_ONNX", DEFAULT_ONNX_PATH)
    self.confidence = float(os.getenv("NKAOUD_VISUAL_VEHICLE_CONF", "0.35"))
    # Keep the debug detector well below camera/modeld cadence on comma3x.
    self.detector_hz = max(1, min(5, int(os.getenv("NKAOUD_VISUAL_VEHICLE_HZ", "1"))))
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

    self.input_shape: tuple[int, int] = (320, 320)  # width, height
    self.left_flag = DebouncedFlag()
    self.right_flag = DebouncedFlag()
    self.startup_debug: dict[str, Any] = {"reason": "not_started", "runtime": self.runtime}
    self.last_model_debug: dict[str, Any] = {}
    self._preproc_dumped = False
    self._logged_buf_geometry = False

  def _write_state(self, left: bool, right: bool, debug: dict[str, Any] | None = None) -> None:
    state = {
      "left": bool(left),
      "right": bool(right),
      "monotonic_time": time.monotonic(),
      "debug": debug or {},
    }
    tmp_path = STATE_PATH.with_suffix(".tmp")
    try:
      tmp_path.write_text(json.dumps(state, separators=(",", ":")))
      os.replace(tmp_path, STATE_PATH)
    except Exception:
      cloudlog.exception("visual vehicle detector failed to write state")

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
    self._write_state(False, False, payload)

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

  def _load_tinygrad_pkl(self) -> bool:
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
      cloudlog.warning("visual vehicle detector loaded tinygrad pkl %s input=%s shape=%s",
                       self.pkl_path, self.pkl_input_name, self.input_shape)
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

  def _vipc_to_yuv_planes(self, buf: VisionBuf) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns (y, u, v) as int16 arrays from the NV12 VisionIPC buffer, or None.
    y is full-resolution (height, width); u and v are quarter-resolution
    (height/2, width/2). Mirrors system/camerad/snapshot.py exactly to avoid
    any shape/dtype surprises from buf.data."""
    try:
      width = int(buf.width)
      height = int(buf.height)
      stride = int(buf.stride)
      uv_offset = int(buf.uv_offset)
      # uv_height calculation copied from snapshot.py.
      uv_height = ((height // 2) + 15) // 16 * 16
      uv_plane_size = stride * uv_height

      # One-time debug log so we can verify the buffer layout values reported
      # by VisionBuf rather than guessing -- if the splatter persists, these
      # numbers are the first thing to check.
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
        # Also persist for the web preview page so the user can read these
        # values without SSH access.
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

      # Slice buf.data as a memoryview FIRST, then wrap with np.array. This is
      # what snapshot.py does and it sidesteps any shape buf.data might carry.
      # int32 (not int16) is required: subsequent 256*Y overflows int16 for any
      # Y >= 128 (256*128 = 32768 > int16 max 32767), wraps negative, and after
      # >> 8 ends up as a large negative bias in the BT.601 sums -- which is
      # exactly what was making bright regions render as dark / magenta.
      y = np.array(buf.data[:uv_offset], dtype=np.uint8) \
            .reshape((-1, stride))[:height, :width].astype(np.int32)
      uv_data = buf.data[uv_offset:uv_offset + uv_plane_size]
      # Standard NV12 chroma interleave: even bytes are U (Cb), odd bytes are V
      # (Cr). This matches system/camerad/snapshot.py and spectra.cc, which
      # configures CAM_FORMAT_NV12. An earlier NV21 (V,U) assumption here fed
      # the V/Cr signal into the blue channel, so red tail lights rendered blue.
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
    self._draw_text(full_draw, (crop_x + 6, crop_y + 6), "detector crop 928x416", (255, 255, 0))
    full_img.save(PREVIEW_FULL_FRAME_CROP_PATH)

    # 2) Raw detector crop + ROI split in crop coordinates: confirms the left/right
    # classification zones after detections are mapped back to the crop.
    crop_img = Image.fromarray(detector_rgb, "RGB")
    crop_draw = ImageDraw.Draw(crop_img)
    cw, ch = crop_img.size
    left_x2 = int(round(LEFT_ROI[2] * cw))
    crop_draw.rectangle((0, 0, max(0, left_x2 - 1), ch - 1), outline=(255, 0, 0), width=3)
    crop_draw.rectangle((left_x2, 0, cw - 1, ch - 1), outline=(0, 128, 255), width=3)
    crop_draw.line((left_x2, 0, left_x2, ch - 1), fill=(255, 255, 0), width=2)
    self._draw_text(crop_draw, (4, 4), "LEFT ROI", (255, 0, 0))
    self._draw_text(crop_draw, (left_x2 + 6, 4), "RIGHT ROI", (0, 128, 255))

    for det in detections:
      x1, y1, x2, y2 = det.xyxy
      rect = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
      crop_draw.rectangle(rect, outline=(0, 255, 0), width=2)
      self._draw_text(crop_draw, (rect[0], max(0, rect[1] - 12)), f"{det.class_id}:{det.confidence:.2f}", (0, 255, 0))
    crop_img.save(PREVIEW_DETECTOR_CROP_PATH)

    # 3) Exact tensor image going to YOLO after letterbox/resize. This should be
    # 480x224 when the compiled PKL metadata reports input_shape [1,3,224,480].
    canvas, pad_x, pad_y, scale = self._letterbox(detector_rgb)
    model_img = Image.fromarray(canvas, "RGB")
    model_draw = ImageDraw.Draw(model_img)
    mw, mh = model_img.size
    lx2 = int(round(LEFT_ROI[2] * mw))
    model_draw.rectangle((0, 0, max(0, lx2 - 1), mh - 1), outline=(255, 0, 0), width=2)
    model_draw.rectangle((lx2, 0, mw - 1, mh - 1), outline=(0, 128, 255), width=2)
    model_draw.line((lx2, 0, lx2, mh - 1), fill=(255, 255, 0), width=1)
    self._draw_text(model_draw, (3, 3), "LEFT", (255, 0, 0))
    self._draw_text(model_draw, (lx2 + 4, 3), "RIGHT", (0, 128, 255))

    for det in detections:
      x1, y1, x2, y2 = det.xyxy
      mx1 = int(round(x1 * scale + pad_x))
      my1 = int(round(y1 * scale + pad_y))
      mx2 = int(round(x2 * scale + pad_x))
      my2 = int(round(y2 * scale + pad_y))
      model_draw.rectangle((mx1, my1, mx2, my2), outline=(0, 255, 0), width=2)
      self._draw_text(model_draw, (mx1, max(0, my1 - 12)), f"{det.class_id}:{det.confidence:.2f}", (0, 255, 0))
    model_img.save(PREVIEW_MODEL_INPUT_PATH)

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
    raw_right_roi = self._box_in_roi(mapped, RIGHT_ROI, image_w, image_h)
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
  

  def _crop_detector_region(self, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, int | str]]:
    """Crop the fixed detector region from the original RGB camera frame.

    Defaults:
      - x/y: 854,425
      - width/height: 928x416
      - region: x=854..1782, y=425..841

    You can still tune the crop without editing code by setting:
      NKAOUD_VISUAL_VEHICLE_CROP_X
      NKAOUD_VISUAL_VEHICLE_CROP_Y
    """
    image_h, image_w = rgb.shape[:2]

    crop_w = min(DETECT_CROP_W, image_w)
    crop_h = min(DETECT_CROP_H, image_h)

    default_x = DETECT_CROP_X
    default_y = DETECT_CROP_Y

    def _get_int_env(name: str, default: int) -> int:
      raw = os.getenv(name, "")
      if raw == "":
        return default
      try:
        return int(raw)
      except Exception:
        cloudlog.warning("visual vehicle detector invalid %s=%r; using %d", name, raw, default)
        return default

    crop_x = _get_int_env("NKAOUD_VISUAL_VEHICLE_CROP_X", default_x)
    crop_y = _get_int_env("NKAOUD_VISUAL_VEHICLE_CROP_Y", default_y)

    crop_x = max(0, min(crop_x, image_w - crop_w))
    crop_y = max(0, min(crop_y, image_h - crop_h))

    crop = rgb[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    debug = {
      "crop_mode": "fixed_928x416",
      "crop_x": int(crop_x),
      "crop_y": int(crop_y),
      "crop_w": int(crop_w),
      "crop_h": int(crop_h),
      "crop_x2_exclusive": int(crop_x + crop_w),
      "crop_y2_exclusive": int(crop_y + crop_h),
      "frame_w": int(image_w),
      "frame_h": int(image_h),
      "roi_model_w": int(ROI_MODEL_W),
      "roi_model_h": int(ROI_MODEL_H),
      "left_roi_model_px": "x=0..32,y=0..224",
      "right_roi_model_px": "x=32..480,y=0..224",
    }
    return crop, debug

  @staticmethod
  def _box_in_roi(box: tuple[float, float, float, float], roi: tuple[float, float, float, float], image_w: int, image_h: int) -> bool:
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
    if width < image_w * 0.025 or height < image_h * 0.04:
      return False
    return rx1 <= cx <= rx2 and ry1 <= bottom <= ry2

  def update(self, buf: VisionBuf) -> tuple[bool, bool, dict[str, Any]]:
    planes = self._vipc_to_yuv_planes(buf)
    if planes is None:
      left = self.left_flag.update(False)
      right = self.right_flag.update(False)
      return left, right, {"reason": "frame_convert_failed", "runtime": self.runtime}
    try:
      rgb = self._yuv_to_rgb(*planes, full_range=True)
    except Exception:
      cloudlog.exception("visual vehicle detector failed YUV->RGB conversion")
      left = self.left_flag.update(False)
      right = self.right_flag.update(False)
      return left, right, {"reason": "frame_convert_failed", "runtime": self.runtime}

    # Run YOLO on the fixed crop only, not on the full camera frame.
    detector_rgb, crop_debug = self._crop_detector_region(rgb)
    detections = self._run_model(detector_rgb)
    image_h, image_w = detector_rgb.shape[:2]

    # Live preview: keep the original full-frame color-conversion previews and
    # add cropped-inference previews. Only active while the dialog/web server
    # holds the request sentinel.
    if os.path.exists(PREVIEW_REQUEST_PATH):
      try:
        rgb_limited = self._yuv_to_rgb(*planes, full_range=False)
        self._write_preview_pair(rgb, rgb_limited)
        self._write_raw_planes(*planes)
        self._write_detector_previews(rgb, detector_rgb, crop_debug, detections)
      except Exception:
        cloudlog.exception("visual vehicle detector live preview write failed")

    # LEFT_ROI and RIGHT_ROI are normalized from the intended 480x224 model
    # layout, then applied to the crop coordinates that _run_model returns.
    # Current layout:
    #   left:  x=0..32   of model input
    #   right: x=32..480 of model input
    raw_left = any(self._box_in_roi(det.xyxy, LEFT_ROI, image_w, image_h) for det in detections)
    raw_right = any(self._box_in_roi(det.xyxy, RIGHT_ROI, image_w, image_h) for det in detections)
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
      "right_roi_norm": [round(float(v), 5) for v in RIGHT_ROI],
      "raw_left": raw_left,
      "raw_right": raw_right,
      "left_score": self.left_flag.score,
      "right_score": self.right_flag.score,
      "detections": len(detections),
      "best_conf": round(float(best_conf), 3),
      "hz": self.detector_hz,
    }
    debug.update(self.last_model_debug)
    return left, right, debug

  def run(self) -> None:
    self.log_debug = self.params.get_bool("VisualVehicleDetectorLogDebug")
    if not self._load_runtime():
      rk = Ratekeeper(1.0)
      while True:
        # Keep the real startup failure visible instead of overwriting it with
        # a generic inactive heartbeat.
        self._write_state(False, False, self.startup_debug)
        rk.keep_time()

    while True:
      streams = VisionIpcClient.available_streams("camerad", block=False)
      if VisionStreamType.VISION_STREAM_WIDE_ROAD in streams:
        stream = VisionStreamType.VISION_STREAM_WIDE_ROAD
        break
      if VisionStreamType.VISION_STREAM_ROAD in streams:
        stream = VisionStreamType.VISION_STREAM_ROAD
        break
      self._write_state(False, False, {"reason": "waiting_for_camera", "runtime": self.runtime})
      time.sleep(0.2)

    # Always consume only the freshest frame; this detector should drop old
    # frames instead of competing with modeld by trying to catch up.
    vipc_client = VisionIpcClient("camerad", stream, True)
    while not vipc_client.connect(False):
      self._write_state(False, False, {"reason": "waiting_for_vipc", "runtime": self.runtime})
      time.sleep(0.1)

    cloudlog.warning("visual vehicle detector connected stream=%s size=%sx%s runtime=%s",
                     stream, vipc_client.width, vipc_client.height, self.runtime)
    rk = Ratekeeper(float(self.detector_hz))
    last_frame_id = -1
    while True:
      new_log_debug = self.params.get_bool("VisualVehicleDetectorLogDebug")
      if new_log_debug != self.log_debug:
        self.log_debug = new_log_debug

      buf = vipc_client.recv()
      if buf is None:
        self._write_state(False, False, {"reason": "no_frame", "runtime": self.runtime})
        rk.keep_time()
        continue
      if vipc_client.frame_id == last_frame_id:
        rk.keep_time()
        continue
      last_frame_id = vipc_client.frame_id
      try:
        left, right, debug = self.update(buf)
        debug["frame_id"] = int(vipc_client.frame_id)
        debug["stream"] = str(stream)
        self._write_state(left, right, debug)
        if self.log_debug:
          cloudlog.info("visual_vehicle_detector left=%s right=%s debug=%s", left, right, debug)
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
