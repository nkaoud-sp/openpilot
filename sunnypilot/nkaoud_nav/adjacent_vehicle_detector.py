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
from openpilot.common.realtime import Ratekeeper, config_realtime_process
from openpilot.common.swaglog import cloudlog

STATE_PATH = Path("/tmp/nkaoud_visual_vehicle_detector.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "selfdrive/modeld/models"
DEFAULT_PKL_PATH = str(MODEL_DIR / "visual_vehicle_detector_tinygrad.pkl")
DEFAULT_ONNX_PATH = str(MODEL_DIR / "visual_vehicle_detector.onnx")

# COCO class IDs from Ultralytics YOLO exports.
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

# Normalized side-lane ROIs: x1, y1, x2, y2.
LEFT_ROI = (0.00, 0.35, 0.42, 1.00)
RIGHT_ROI = (0.58, 0.35, 1.00, 1.00)


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
    self.params = Params()
    self.pkl_path = os.getenv("NKAOUD_VISUAL_VEHICLE_PKL", DEFAULT_PKL_PATH)
    self.onnx_path = os.getenv("NKAOUD_VISUAL_VEHICLE_ONNX", DEFAULT_ONNX_PATH)
    self.confidence = float(os.getenv("NKAOUD_VISUAL_VEHICLE_CONF", "0.35"))
    self.detector_hz = max(1, min(10, int(os.getenv("NKAOUD_VISUAL_VEHICLE_HZ", "5"))))
    self.log_debug = False
    self.runtime = "none"

    self.cv2 = None

    # tinygrad pkl runtime fields
    self.Tensor = None
    self.pkl_model_run = None
    self.pkl_input_name = "images"
    self.pkl_input_dtype = None
    self.pkl_input_device = None

    # ONNX fallback runtime fields
    self.onnx_session = None
    self.onnx_input_name = ""

    self.input_shape: tuple[int, int] = (320, 320)  # width, height
    self.left_flag = DebouncedFlag()
    self.right_flag = DebouncedFlag()
    self.startup_debug: dict[str, Any] = {"reason": "not_started", "runtime": self.runtime}

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

  def _load_cv2(self) -> bool:
    try:
      import cv2  # pylint: disable=import-error
      self.cv2 = cv2
      return True
    except Exception as e:
      self._set_startup_debug(reason="missing_cv2", error=str(e))
      cloudlog.warning("visual vehicle detector missing cv2: %s", e)
      return False

  def _vipc_to_bgr(self, buf: VisionBuf) -> np.ndarray | None:
    if self.cv2 is None:
      return None
    try:
      width, height = int(buf.width), int(buf.height)
      data = np.asarray(buf.data, dtype=np.uint8).ravel()
      expected = width * height * 3 // 2
      if data.size < expected:
        return None
      nv12 = data[:expected].reshape((height * 3 // 2, width))
      return self.cv2.cvtColor(nv12, self.cv2.COLOR_YUV2BGR_NV12)
    except Exception:
      cloudlog.exception("visual vehicle detector failed frame conversion")
      return None

  def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
    assert self.cv2 is not None
    w, h = self.input_shape
    resized = self.cv2.resize(bgr, (w, h), interpolation=self.cv2.INTER_LINEAR)
    rgb = self.cv2.cvtColor(resized, self.cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    return np.ascontiguousarray(x)

  @staticmethod
  def _tensor_to_numpy(out: Any) -> np.ndarray:
    if isinstance(out, dict):
      out = next(iter(out.values()))
    if isinstance(out, (list, tuple)):
      out = out[0]
    if hasattr(out, "contiguous"):
      return out.contiguous().realize().uop.base.buffer.numpy()
    return np.asarray(out)

  def _run_model(self, bgr: np.ndarray) -> list[Detection]:
    inp = self._preprocess(bgr)
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
      return self._parse_yolo_output(arr, bgr.shape[1], bgr.shape[0])

    if self.runtime == "onnx_cpu" and self.onnx_session is not None:
      outputs = self.onnx_session.run(None, {self.onnx_input_name: inp})
      return self._parse_yolo_output(outputs[0], bgr.shape[1], bgr.shape[0])

    return []

  def _parse_yolo_output(self, output: np.ndarray, image_w: int, image_h: int) -> list[Detection]:
    arr = np.squeeze(np.asarray(output))
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1] and arr.shape[0] >= 6:
      arr = arr.T

    detections: list[Detection] = []
    if arr.ndim != 2:
      return detections

    in_w, in_h = self.input_shape
    sx, sy = image_w / float(in_w), image_h / float(in_h)

    # Case A: already NMSed [x1, y1, x2, y2, score, class].
    if arr.shape[1] == 6:
      for row in arr:
        score = float(row[4])
        cls = int(row[5])
        if score < self.confidence or cls not in VEHICLE_CLASS_IDS:
          continue
        x1, y1, x2, y2 = [float(v) for v in row[:4]]
        detections.append(Detection((x1, y1, x2, y2), score, cls))
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

      for box, cls, score in zip(boxes, cls_ids, confs):
        cls_i = int(cls)
        score_f = float(score)
        if score_f < self.confidence or cls_i not in VEHICLE_CLASS_IDS:
          continue
        cx, cy, bw, bh = [float(v) for v in box]
        x1 = (cx - bw / 2.0) * sx
        y1 = (cy - bh / 2.0) * sy
        x2 = (cx + bw / 2.0) * sx
        y2 = (cy + bh / 2.0) * sy
        detections.append(Detection((x1, y1, x2, y2), score_f, cls_i))
      return detections

    # Case B2: YOLOv8 / YOLO11 raw output [cx, cy, w, h, class_scores...].
    # Ultralytics v8/v11 exports commonly produce [1, 84, N] for COCO.
    if arr.shape[1] >= 4 + max(VEHICLE_CLASS_IDS) + 1:
      boxes = arr[:, :4]
      scores = arr[:, 4:]
      cls_ids = np.argmax(scores, axis=1)
      confs = scores[np.arange(scores.shape[0]), cls_ids]

      for box, cls, score in zip(boxes, cls_ids, confs):
        cls_i = int(cls)
        score_f = float(score)
        if score_f < self.confidence or cls_i not in VEHICLE_CLASS_IDS:
          continue
        cx, cy, bw, bh = [float(v) for v in box]
        x1 = (cx - bw / 2.0) * sx
        y1 = (cy - bh / 2.0) * sy
        x2 = (cx + bw / 2.0) * sx
        y2 = (cy + bh / 2.0) * sy
        detections.append(Detection((x1, y1, x2, y2), score_f, cls_i))
    return detections
  

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
    bgr = self._vipc_to_bgr(buf)
    if bgr is None:
      left = self.left_flag.update(False)
      right = self.right_flag.update(False)
      return left, right, {"reason": "frame_convert_failed", "runtime": self.runtime}

    detections = self._run_model(bgr)
    image_h, image_w = bgr.shape[:2]
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
      "raw_left": raw_left,
      "raw_right": raw_right,
      "left_score": self.left_flag.score,
      "right_score": self.right_flag.score,
      "detections": len(detections),
      "best_conf": round(float(best_conf), 3),
      "hz": self.detector_hz,
    }
    return left, right, debug

  def run(self) -> None:
    self.log_debug = self.params.get_bool("VisualVehicleDetectorLogDebug")
    if not self._load_cv2() or not self._load_runtime():
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

    vipc_client = VisionIpcClient("camerad", stream, False)
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
  config_realtime_process(2, 5)
  VisualVehicleDetector().run()


if __name__ == "__main__":
  main()
