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
from collections import OrderedDict

import numpy as np

from cereal import messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler

from sunnypilot.nkaoud_nav.lane_line_logger import LaneLineSessionLogger, finalize_orphan_sessions, LABELS
from sunnypilot.selfdrive.controls.lib.lane_line_classifier import (
  LaneLineClassifier, LaneLineClassifierConfig, DEFAULT_CONFIG, SAMPLE_X_MIN,
  LEFT_EGO_LINE, RIGHT_EGO_LINE, CONTRAST_METHOD_COUNT, scan_geometry_uv,
)

SERVICE = "laneLineClassificationSP"
CALIBRATED = 1  # cereal LiveCalibrationData.Status.calibrated
LOOP_HZ = 20.0        # camera drain rate; must keep up with camerad
PUBLISH_HZ = 5.0      # classification/publish rate
FRAME_CACHE = 8       # recent Y-planes kept for frameId matching (~0.4 s)
FRAME_ID_SLOP = 2     # accept a cached frame within this many ids of the model's
DEFAULT_CAMERA = ("tici", "ar0231")


def build_transform(rpy_calib, intrinsics) -> np.ndarray:
  """device/calib frame -> image pixel 3x3, matching the UI's calib_transform."""
  device_from_calib = rot_from_euler(np.asarray(rpy_calib, dtype=np.float64))
  view_from_calib = view_frame_from_device_frame @ device_from_calib
  return intrinsics @ view_from_calib


def nv12_y_plane(buf) -> np.ndarray:
  """Extract the full-res luminance (Y) plane from an NV12 VisionBuf.

  Returns a copy: the underlying VisionIPC buffer is recycled by camerad, so a
  view would be silently overwritten while we're still sampling it.
  """
  width, height, stride = int(buf.width), int(buf.height), int(buf.stride)
  flat = np.frombuffer(buf.data, dtype=np.uint8)
  y = flat[:stride * height].reshape(height, stride)
  return y[:, :width].copy()


class LaneLineClassifierD:
  def __init__(self):
    self.params = Params()
    # Keep only the cereal streams required for classification and the
    # onroad gate. Logging intentionally does not subscribe to the extra
    # selfdriveState/carState streams in this resource-isolation experiment.
    self.sm = messaging.SubMaster(["modelV2", "liveCalibration", "roadCameraState", "deviceState"])
    self.pm = messaging.PubMaster([SERVICE])
    self.clf = LaneLineClassifier()
    self.logger = LaneLineSessionLogger()
    self._log_enabled = False
    self._log_active = False
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
      contrast_method=int(np.clip(self._get_int("LaneLineContrastMethod", 0),
                                 0, CONTRAST_METHOD_COUNT - 1)),
    )
    try:
      self._log_enabled = self.params.get_bool("LaneLineVisualizerLogging")
      self._log_active = self.params.get_bool("LaneLineVisualizerLogActive")
    except Exception:
      self._log_enabled = False
      self._log_active = False

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
    # actual scan geometry, so the UI can outline the assessed corridor
    st.scanHalfM = float(self._cfg.scan_half_m)
    st.sampleXMinM = float(SAMPLE_X_MIN)
    st.sampleXMaxM = float(self._cfg.sample_x_max)
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

  def _publish_throttled(self, gate, frame_id: int, reason: str, valid: bool, camera_offset: float):
    """Publish, but never faster than PUBLISH_HZ (the loop runs at LOOP_HZ)."""
    now = time.monotonic()
    if now - self._last_pub_t < 1.0 / PUBLISH_HZ - 0.005:
      return
    self._last_pub_t = now
    self._publish(gate, frame_id, reason, valid, camera_offset)

  def _logging_wanted(self) -> bool:
    """Capture while onroad when logging is enabled and LANE is active.

    This branch intentionally uses deviceState.started instead of subscribing
    to selfdriveState, so logging load can be tested independently from the
    extra cereal subscription. The LANE button controls the capture session.
    """
    return (self._log_enabled and self._log_active and self.sm.seen["deviceState"]
            and bool(self.sm["deviceState"].started))

  def _update_logging(self, gate, model, frame_y, transform, camera_offset: float):
    """Record this assessment to the per-engagement troubleshooting session."""
    if not self.logger.is_active:
      from dataclasses import asdict
      self.logger.start({
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "camera_offset": camera_offset,
        "sample_x_min": SAMPLE_X_MIN,
        "config": asdict(self._cfg),
      })

    snapshots = []
    lines = list(model.laneLines)
    for side, res, idx in (("L", gate.left, LEFT_EGO_LINE), ("R", gate.right, RIGHT_EGO_LINE)):
      label = LABELS.get(int(res.line_type), "unknown")
      if idx < len(lines) and self.logger.snapshot_due(side, label):
        ll = lines[idx]
        geometry = scan_geometry_uv(ll.x, ll.y, ll.z, transform, camera_offset, self._cfg,
                                    res.lateral_offset_m)
        name = self.logger.save_snapshot(frame_y, side, label, res.duty, geometry, res.present)
        if name:
          snapshots.append(name)

    # carState is deliberately not subscribed to in this isolation test.
    # Keep the CSV shape stable while marking speed as unavailable.
    v_ego = None
    self.logger.log_row(time.monotonic(), int(model.frameId), v_ego, gate, snapshots)

  def run(self):
    # the loop runs at camera rate so the non-conflating VisionIPC queue never
    # overflows; classification/publishing is throttled down to PUBLISH_HZ
    rk = Ratekeeper(LOOP_HZ)
    stream = VisionStreamType.VISION_STREAM_ROAD
    vc: VisionIpcClient | None = None
    # recent Y-planes keyed by frame_id, so each modelV2 is classified against
    # the exact frame it was computed from. Pairing the newest camera frame
    # with the newest model output (the old behaviour) misaligns the polyline
    # by a few frames: metres of forward travel, and on curves enough lateral
    # error to pull the far-field scan off the paint entirely.
    frames: OrderedDict[int, np.ndarray] = OrderedDict()
    finalize_orphan_sessions()
    while True:
      self.sm.update(0)
      self._resolve_camera()
      self._refresh_config()

      # disengaged (or logging switched off): flush the session so the zip is
      # queued for email right away
      if self.logger.is_active and not self._logging_wanted():
        self.logger.end()

      if self.sm.updated["liveCalibration"]:
        lc = self.sm["liveCalibration"]
        if lc.calStatus == CALIBRATED and len(lc.rpyCalib) == 3:
          self.rpy_calib = list(lc.rpyCalib)

      if vc is None or not vc.is_connected():
        frames.clear()
        if stream not in VisionIpcClient.available_streams("camerad", block=False):
          self._publish_throttled(None, 0, "camera_unavailable", False, 0.0)
          rk.keep_time()
          continue
        vc = VisionIpcClient("camerad", stream, False)
        if not vc.connect(False):
          self._publish_throttled(None, 0, "waiting_for_vipc", False, 0.0)
          rk.keep_time()
          continue

      # drain everything queued since the last tick
      timeout_ms = 20
      while (buf := vc.recv(timeout_ms=timeout_ms)) is not None:
        timeout_ms = 0
        frames[int(vc.frame_id)] = nv12_y_plane(buf)
        while len(frames) > FRAME_CACHE:
          frames.popitem(last=False)

      if not frames:
        rk.keep_time()
        continue
      newest_fid = next(reversed(frames))

      if self.rpy_calib is None:
        self._publish_throttled(None, newest_fid, "waiting_for_calib", False, 0.0)
        rk.keep_time()
        continue
      if not self.sm.seen["modelV2"]:
        self._publish_throttled(None, newest_fid, "waiting_for_model", False, 0.0)
        rk.keep_time()
        continue

      # classify at PUBLISH_HZ, on the frame the current model output describes
      if time.monotonic() - self._last_pub_t < 1.0 / PUBLISH_HZ - 0.005:
        rk.keep_time()
        continue

      model = self.sm["modelV2"]
      model_fid = int(model.frameId)
      frame_y = frames.get(model_fid)
      if frame_y is None:
        near = [fid for fid in frames if abs(fid - model_fid) <= FRAME_ID_SLOP]
        if near:
          frame_y = frames[max(near)]
      if frame_y is None:
        self._publish_throttled(None, newest_fid, "waiting_for_frame", False, 0.0)
        rk.keep_time()
        continue

      try:
        transform = build_transform(self.rpy_calib, self.intrinsics)
        camera_offset = self._camera_offset()
        gate = self.clf.update(frame_y, model, transform, camera_offset, self._cfg)
        self._last_pub_t = time.monotonic()
        self._publish(gate, model_fid, "ok", True, camera_offset)
        if self._logging_wanted():
          self._update_logging(gate, model, frame_y, transform, camera_offset)
      except Exception as e:
        cloudlog.exception("lane_line_classifierd update failed")
        self._last_pub_t = time.monotonic()
        self._publish(None, model_fid, f"exception: {e}", False, 0.0)
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
