#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Offline harness to test the solid-vs-broken lane-line classifier on a real
route. It reads modelV2 + liveCalibration from the logs and the road-camera
frames, builds the same device->image transform the UI uses (calib_transform),
and prints the classification for the two ego lane lines per frame.

Usage (in an openpilot env, with the tools deps available):

    python -m sunnypilot.selfdrive.controls.lib.lane_line_classifier_replay \
        "a2ce8c8f6b7f9b1a|2024-01-01--12-00-00" --stride 5

The route identifier is anything tools.lib.route.Route accepts. Camera
intrinsics default to the tici fcam; pass --wide for the wide camera.

Alignment note: modelV2 and the road camera both run at ~20 Hz, so this harness
joins them by sequence order within the route. That is exact when no frames are
dropped and good enough for tuning; a production join would use the encode
index. ``build_transform`` is import-safe and dependency-light, so it can be
reused directly on-device where you already have the live frame + modelV2.
"""
from __future__ import annotations

import argparse

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler

from sunnypilot.selfdrive.controls.lib.lane_line_classifier import LaneLineClassifier, LaneLineType

CALIBRATED = 1  # cereal LiveCalibrationData.Status.calibrated
_NAME = {LaneLineType.SOLID: "SOLID ", LaneLineType.BROKEN: "broken",
         LaneLineType.UNKNOWN: "  ?   ", LaneLineType.DOUBLE: "DOUBLE"}


def build_transform(rpy_calib, intrinsics) -> np.ndarray:
  """device/calib frame -> image pixel 3x3, matching the UI's calib_transform.

  This mirrors augmented_road_view.py: view_from_calib = view_frame_from_device
  @ device_from_calib, then intrinsics @ view_from_calib. Reuse this on-device.
  """
  device_from_calib = rot_from_euler(np.asarray(rpy_calib, dtype=np.float64))
  view_from_calib = view_frame_from_device_frame @ device_from_calib
  return intrinsics @ view_from_calib


def iter_model_calib(route_id: str):
  """Yield (modelV2_reader, rpyCalib) in time order, carrying latest calibration."""
  from openpilot.tools.lib.logreader import LogReader
  latest_rpy = None
  for msg in LogReader(route_id):
    w = msg.which()
    if w == "liveCalibration":
      lc = msg.liveCalibration
      if lc.calStatus == CALIBRATED and len(lc.rpyCalib) == 3:
        latest_rpy = tuple(lc.rpyCalib)
    elif w == "modelV2" and latest_rpy is not None:
      yield msg.modelV2, latest_rpy


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("route", help="route/segment identifier")
  ap.add_argument("--wide", action="store_true", help="use the wide camera")
  ap.add_argument("--stride", type=int, default=5, help="classify every Nth model frame")
  ap.add_argument("--camera-offset", type=float, default=0.0)
  ap.add_argument("--device", default="tici")
  ap.add_argument("--sensor", default="ar0231")
  ap.add_argument("--limit", type=int, default=0, help="stop after N classified frames (0 = all)")
  args = ap.parse_args()

  from openpilot.tools.lib.framereader import FrameReader
  from openpilot.tools.lib.route import Route

  cam = DEVICE_CAMERAS[(args.device, args.sensor)]
  intrinsics = cam.ecam.intrinsics if args.wide else cam.fcam.intrinsics

  route = Route(args.route)
  seg_paths = route.ecamera_paths() if args.wide else route.camera_paths()

  # Build a lazily-opened, route-global frame index from per-segment videos.
  readers: dict[int, FrameReader] = {}
  seg_counts: list[int] = []
  for i, p in enumerate(seg_paths):
    if p is None:
      seg_counts.append(0)
      continue
    fr = FrameReader(p, pix_fmt="gray")
    readers[i] = fr
    seg_counts.append(fr.frame_count)
  seg_starts = np.cumsum([0] + seg_counts)

  def get_frame(global_idx: int):
    seg = int(np.searchsorted(seg_starts, global_idx, side="right") - 1)
    if seg not in readers:
      return None
    local = global_idx - seg_starts[seg]
    if local >= seg_counts[seg]:
      return None
    f = readers[seg].get(int(local))
    if f is None:
      return None
    return f[:, :, 0] if f.ndim == 3 else f

  clf = LaneLineClassifier()
  processed = 0
  for global_idx, (model, rpy) in enumerate(iter_model_calib(args.route)):
    if global_idx % args.stride:
      continue
    frame_y = get_frame(global_idx)
    if frame_y is None:
      continue
    transform = build_transform(rpy, intrinsics)
    gate = clf.update(frame_y, model, transform, args.camera_offset)
    print(f"f{global_idx:6d}  "
          f"L[{_NAME[gate.left.line_type]} d={gate.left.duty:.2f} p={gate.left.period_m:4.1f} "
          f"c={gate.left.confidence:.2f}] cross={int(gate.left_crossable)}   "
          f"R[{_NAME[gate.right.line_type]} d={gate.right.duty:.2f} p={gate.right.period_m:4.1f} "
          f"c={gate.right.confidence:.2f}] cross={int(gate.right_crossable)}")
    processed += 1
    if args.limit and processed >= args.limit:
      break
  print(f"\nprocessed {processed} frames")


if __name__ == "__main__":
  main()
