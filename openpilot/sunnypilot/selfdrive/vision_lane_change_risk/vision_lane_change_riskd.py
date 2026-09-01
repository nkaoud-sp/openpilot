#!/usr/bin/env python3
from __future__ import annotations

import os
import time

import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper, config_realtime_process, Priority
from openpilot.common.swaglog import cloudlog
from msgq.visionipc import VisionIpcClient, VisionBuf

from openpilot.sunnypilot.selfdrive.vision_lane_change_risk.common_frame_tracker import (
  CommonFrameMotionTracker,
  CAMERA_GRID_H,
  CAMERA_GRID_W,
  compose_common_frame,
  write_debug_png,
)


LaneChangeDirection = log.LaneChangeDirection
LaneChangeState = log.LaneChangeState
Direction = custom.VisionLaneChangeRisk.Direction
Source = custom.VisionLaneChangeRisk.Source

MODEL_RATE = 20
DEBUG_DUMP_INTERVAL = 1.0
DEBUG_DUMP_DIR = "/data/media/0/vision_lane_change_risk_debug"
SOURCE_NAMES = {
  Source.none: "none",
  Source.wideRoad: "wideRoad",
  Source.narrowRoad: "narrowRoad",
  Source.commonFrame: "commonFrame",
}
STREAM_CONFIGS = {
  "wide": VisionStreamType.VISION_STREAM_WIDE_ROAD,
  "narrow": VisionStreamType.VISION_STREAM_NARROW_ROAD,
  "cabin": VisionStreamType.VISION_STREAM_CABIN,
}


def y_plane_to_grid(buf: VisionBuf) -> np.ndarray:
  y = np.frombuffer(buf.data, dtype=np.uint8, count=buf.uv_offset)
  y = y.reshape((-1, buf.stride))[:buf.height, :buf.width]

  # Keep the full camera image for the fisheye projection, but shrink it before
  # CPU stitching so the daemon stays light enough to run live.
  ys = np.linspace(0, y.shape[0] - 1, CAMERA_GRID_H).astype(np.int32)
  xs = np.linspace(0, y.shape[1] - 1, CAMERA_GRID_W).astype(np.int32)
  return y[np.ix_(ys, xs)]


def intended_direction(model_v2) -> int:
  if model_v2.meta.laneChangeState == LaneChangeState.preLaneChange:
    if model_v2.meta.laneChangeDirection == LaneChangeDirection.left:
      return Direction.left
    if model_v2.meta.laneChangeDirection == LaneChangeDirection.right:
      return Direction.right
  return Direction.none


def connect_cameras() -> dict[str, VisionIpcClient]:
  while True:
    streams = VisionIpcClient.available_streams("camerad", block=False)
    if any(stream in streams for stream in STREAM_CONFIGS.values()):
      break
    time.sleep(0.1)

  clients: dict[str, VisionIpcClient] = {}
  for name, stream in STREAM_CONFIGS.items():
    if stream not in streams:
      continue

    client = VisionIpcClient("camerad", stream, True)
    while not client.connect(False):
      time.sleep(0.1)
    clients[name] = client
    cloudlog.warning(
      f"vision_lane_change_riskd connected {name} "
      f"({client.width}x{client.height}, stride={client.stride})"
    )

  return clients


def main_camera_name(clients: dict[str, VisionIpcClient]) -> str:
  for name in ("wide", "narrow", "cabin"):
    if name in clients:
      return name
  raise RuntimeError("vision_lane_change_riskd has no connected camera clients")


def read_common_frame(clients: dict[str, VisionIpcClient]) -> tuple[np.ndarray | None, int, int]:
  main_name = main_camera_name(clients)
  bufs: dict[str, VisionBuf] = {}
  bufs[main_name] = clients[main_name].recv()

  main_sof = clients[main_name].timestamp_sof
  for name, client in clients.items():
    if name == main_name:
      continue
    buf = client.recv()
    if buf is not None and abs(client.timestamp_sof - main_sof) < 50_000_000:
      bufs[name] = buf

  frames = {
    name: y_plane_to_grid(buf)
    for name, buf in bufs.items()
    if buf is not None
  }
  return compose_common_frame(frames), clients[main_name].frame_id, main_sof


def build_reason(tracker: CommonFrameMotionTracker, direction: int) -> str:
  if direction == Direction.left and tracker.left.risk:
    return "left conflict zone persistent motion"
  if direction == Direction.right and tracker.right.risk:
    return "right conflict zone persistent motion"
  if tracker.left.risk or tracker.right.risk:
    return "side conflict zone persistent motion"
  return ""


def main() -> None:
  config_realtime_process(4, Priority.CTRL_LOW)
  params = Params()
  pm = messaging.PubMaster(["visionLaneChangeRisk"])
  sm = messaging.SubMaster(["carState", "modelV2"])
  clients = connect_cameras()
  tracker = CommonFrameMotionTracker()
  last_debug_dump_t = 0.0
  rk = Ratekeeper(MODEL_RATE, print_delay_threshold=None)

  while True:
    frame, frame_id, timestamp_sof = read_common_frame(clients)
    sm.update(0)

    msg = messaging.new_message("visionLaneChangeRisk")
    risk = msg.visionLaneChangeRisk
    risk.valid = False
    risk.source = Source.commonFrame

    if frame is not None:
      tracker.update(frame)
      direction = intended_direction(sm["modelV2"])
      intended_risk = ((direction == Direction.left and tracker.left.risk) or
                       (direction == Direction.right and tracker.right.risk))

      risk.valid = sm.valid["modelV2"]
      msg.valid = risk.valid
      risk.leftRisk = tracker.left.risk
      risk.rightRisk = tracker.right.risk
      risk.intendedRisk = bool(intended_risk)
      risk.leftConfidence = tracker.left.confidence
      risk.rightConfidence = tracker.right.confidence
      risk.leftTrackAge = tracker.left.track_age
      risk.rightTrackAge = tracker.right.track_age
      risk.frameId = frame_id
      risk.timestampSof = timestamp_sof
      risk.intendedDirection = direction
      risk.reason = build_reason(tracker, direction)

      debug_enabled = (
        os.getenv("VLCR_DEBUG_PNGS") == "1" or
        params.get_bool("VisionLaneChangeRiskDebug")
      )
      now = time.monotonic()
      if debug_enabled and now - last_debug_dump_t >= DEBUG_DUMP_INTERVAL:
        filename = f"vlcr_{frame_id:08d}_{timestamp_sof}.png"
        path = os.path.join(DEBUG_DUMP_DIR, filename)
        write_debug_png(
          path,
          frame,
          tracker.left.risk,
          tracker.right.risk,
          tracker.left.confidence,
          tracker.right.confidence,
        )
        last_debug_dump_t = now

    pm.send("visionLaneChangeRisk", msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
