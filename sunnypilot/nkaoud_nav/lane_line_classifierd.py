#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

lane_line_classifierd: standalone daemon that classifies the two ego lane lines
as solid (not crossable) or broken (crossable) and publishes the result on
`laneLineClassificationSP` for the on-road Lane Line Visualizer readout.

It reuses the pure-numpy classifier in
sunnypilot.selfdrive.controls.lib.lane_line_classifier. The classifier needs
camera pixels, so this runs as a process (like the visual vehicle detector)
rather than in the UI: it reads the road camera via VisionIPC, builds the same
device->image transform the UI uses from liveCalibration, samples luminance
along each projected lane line, and classifies from the marking's duty cycle
and periodicity. UI/debug only for now; it does not gate lane changes.
"""
from __future__ import annotations

import time

import numpy as np

from cereal import messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler

from sunnypilot.selfdrive.controls.lib.lane_line_classifier import (
  LaneLineClassifier, LaneLineClassifierConfig, DEFAULT_CONFIG,
)

SERVICE = "laneLineClassificationSP"
CALIBRATED = 1  # cereal LiveCalibrationData.Status.calibrated
PUBLISH_HZ = 5.0
DEFAULT_CAMERA = ("tici", "ar0231")


def build_transform(rpy_calib, intrinsics) -> np.ndarray:
  """device/calib frame -> image pixel 3x3, matching the UI's calib_transform."""
  device_from_calib = rot_from_euler(np.asarray(rpy_calib, dtype=np.float64))
  view_from_calib = view_frame_from_device_frame @ device_from_calib
  return intrinsics @ view_from_calib


def nv12_y_plane(buf) -> np.ndarray:
  """Extract the full-res luminance (Y) plane from an NV12 VisionBuf."""
  width, height, stride = int(buf.width), int(buf.height), int(buf.stride)
  flat = np.frombuffer(buf.data, dtype=np.uint8)
  y = flat[:stride * height].reshape(height, stride)
  return y[:, :width]


class LaneLineClassifierD:
  def __init__(self):
    self.params = Params()
    self.sm = messaging.SubMaster(["modelV2", "liveCalibration", "roadCameraState", "deviceState"])
    self.pm = messaging.PubMaster([SERVICE])
    self.clf = LaneLineClassifier()
    self.rpy_calib: list[float] | None = None
    self.intrinsics = DEVICE_CAMERAS[DEFAULT_CAMERA].fcam.intrinsics
    self._cam_resolved = False
    self._last_pub_t = 0.0
    self._pub_count = 0
    self._pub_window_t = time.monotonic()
    self._hz = 0.0
    self._cfg = DEFAULT_CONFIG
    self._cfg_t = 0.0

  def _get_int(self, key: str, default: int) -> int:
    try:
      v = self.params.get(key, return_default=True)
      return int(v) if v is not None else default
    except Exception:
      return default

  def _refresh_config(self):
    # Cheap to read, but only refresh ~1 Hz so tuning applies live without
    # hammering the params store every frame.
    now = time.monotonic()
    if now - self._cfg_t < 1.0:
      return
    self._cfg_t = now
    self._cfg = LaneLineClassifierConfig(
      sample_x_max=float(self._get_int("LaneLineSampleMaxM", 60)),
      min_contrast=float(self._get_int("LaneLineMinContrast", 18)),
      solid_duty=self._get_int("LaneLineSolidDuty", 80) / 100.0,
      min_period_m=float(self._get_int("LaneLineMinPeriodM", 3)),
      max_period_m=float(self._get_int("LaneLineMaxPeriodM", 30)),
      min_autocorr=self._get_int("LaneLineMinAutocorr", 30) / 100.0,
    )

  def _resolve_camera(self):
    # Pick the real device camera once both messages have been seen, else keep
    # the tici/ar0231 default. Mirrors augmented_road_view.
    if self._cam_resolved:
      return
    if self.sm.seen["roadCameraState"] and self.sm.seen["deviceState"]:
      key = (str(self.sm["deviceState"].deviceType), str(self.sm["roadCameraState"].sensor))
      cam = DEVICE_CAMERAS.get(key)
      if cam is not None:
        self.intrinsics = cam.fcam.intrinsics
        self._cam_resolved = True
        cloudlog.warning("lane_line_classifierd using camera %s", key)

  def _camera_offset(self) -> float:
    try:
      return float(self.params.get("CameraOffset", return_default=True) or 0.0)
    except Exception:
      return 0.0

  def _publish(self, gate, frame_id: int, reason: str, valid: bool, camera_offset: float):
    msg = messaging.new_message(SERVICE, valid=True)
    st = getattr(msg, SERVICE)
    st.monotonicTime = time.monotonic()
    st.frameId = int(frame_id) & 0xFFFFFFFF
    st.valid = valid
    st.reason = reason
    st.hz = float(self._hz)
    st.cameraOffset = float(camera_offset)
    if gate is not None:
      for dst, src in ((st.left, gate.left), (st.right, gate.right)):
        dst.lineType = int(src.line_type)
        dst.confidence = float(src.confidence)
        dst.duty = float(src.duty)
        dst.periodM = float(src.period_m)
        dst.validFrac = float(src.valid_frac)
        dst.nSamples = int(min(src.n_samples, 0xFFFF))
      st.leftCrossable = bool(gate.left_crossable)
      st.rightCrossable = bool(gate.right_crossable)
    self.pm.send(SERVICE, msg)

    self._pub_count += 1
    now = time.monotonic()
    if now - self._pub_window_t >= 1.0:
      self._hz = self._pub_count / (now - self._pub_window_t)
      self._pub_count = 0
      self._pub_window_t = now

  def run(self):
    rk = Ratekeeper(PUBLISH_HZ)
    stream = VisionStreamType.VISION_STREAM_ROAD
    vc: VisionIpcClient | None = None
    while True:
      self.sm.update(0)
      self._resolve_camera()
      self._refresh_config()

      if self.sm.updated["liveCalibration"]:
        lc = self.sm["liveCalibration"]
        if lc.calStatus == CALIBRATED and len(lc.rpyCalib) == 3:
          self.rpy_calib = list(lc.rpyCalib)

      if vc is None or not vc.is_connected():
        if stream not in VisionIpcClient.available_streams("camerad", block=False):
          self._publish(None, 0, "camera_unavailable", False, 0.0)
          rk.keep_time()
          continue
        vc = VisionIpcClient("camerad", stream, True)
        if not vc.connect(False):
          self._publish(None, 0, "waiting_for_vipc", False, 0.0)
          rk.keep_time()
          continue

      buf = vc.recv(timeout_ms=100)
      if buf is None:
        rk.keep_time()
        continue

      if self.rpy_calib is None:
        self._publish(None, int(vc.frame_id), "waiting_for_calib", False, 0.0)
        rk.keep_time()
        continue
      if not self.sm.seen["modelV2"]:
        self._publish(None, int(vc.frame_id), "waiting_for_model", False, 0.0)
        rk.keep_time()
        continue

      try:
        frame_y = nv12_y_plane(buf)
        transform = build_transform(self.rpy_calib, self.intrinsics)
        camera_offset = self._camera_offset()
        gate = self.clf.update(frame_y, self.sm["modelV2"], transform, camera_offset, self._cfg)
        self._publish(gate, int(vc.frame_id), "ok", True, camera_offset)
      except Exception as e:
        cloudlog.exception("lane_line_classifierd update failed")
        self._publish(None, int(vc.frame_id), f"exception: {e}", False, 0.0)
      rk.keep_time()


def main() -> None:
  import os
  try:
    os.nice(15)
  except Exception:
    pass
  LaneLineClassifierD().run()


if __name__ == "__main__":
  main()
