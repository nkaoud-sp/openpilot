#!/usr/bin/env python3
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
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
  GRID_H,
  GRID_W,
  compose_tuned_frame,
  debug_frame_rgb,
  model_lead_detections,
  rgb_to_yuv420,
  write_debug_png,
)


LaneChangeDirection = log.LaneChangeDirection
LaneChangeState = log.LaneChangeState
Direction = custom.VisionLaneChangeRisk.Direction
Source = custom.VisionLaneChangeRisk.Source

MODEL_RATE = 20
DEBUG_DUMP_INTERVAL = 1.0
DEBUG_DUMP_DIR = "/data/media/0/vision_lane_change_risk_debug"
VIDEO_DUMP_FPS = 10
VIDEO_DUMP_SECONDS = 180
STREAM_SERVER_NAME = "vision_lane_change_riskd"
STREAM_TYPE = VisionStreamType.VISION_STREAM_MAP
STREAM_CONFIGS = {
  "wide": VisionStreamType.VISION_STREAM_WIDE_ROAD,
  "cabin": VisionStreamType.VISION_STREAM_CABIN,
}


def y_plane_to_grid(buf: VisionBuf) -> np.ndarray:
  y = np.frombuffer(buf.data, dtype=np.uint8, count=buf.uv_offset)
  return y.reshape((-1, buf.stride))[:buf.height, :buf.width]


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
  for name in ("wide", "cabin"):
    if name in clients:
      return name
  raise RuntimeError("vision_lane_change_riskd has no connected camera clients")


def read_tuned_frame(clients: dict[str, VisionIpcClient]) -> tuple[np.ndarray | None, int, int]:
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
  return compose_tuned_frame(frames), clients[main_name].frame_id, main_sof


class ProcessedFrameStreamer:
  def __init__(self) -> None:
    from msgq.visionipc import VisionIpcServer

    self.server = VisionIpcServer(STREAM_SERVER_NAME)
    self.server.create_buffers(STREAM_TYPE, 4, GRID_W, GRID_H)
    self.server.start_listener()
    cloudlog.warning(f"vision_lane_change_riskd streaming processed frames on {STREAM_SERVER_NAME}:{STREAM_TYPE}")

  def send(self, rgb: np.ndarray, frame_id: int, timestamp_sof: int) -> None:
    self.server.send(STREAM_TYPE, rgb_to_yuv420(rgb), frame_id, timestamp_sof, int(time.monotonic() * 1e9))


class DebugVideoRecorder:
  def __init__(self) -> None:
    self.proc: subprocess.Popen | None = None
    self.output_path = ""
    self.started_t = 0.0
    self.last_frame_t = 0.0
    self.unavailable = False

  def update(self, rgb: np.ndarray, now: float) -> None:
    if self.unavailable:
      return
    if self.proc is not None and VIDEO_DUMP_SECONDS > 0 and now - self.started_t > VIDEO_DUMP_SECONDS:
      self.close()
      return
    if now - self.last_frame_t < 1.0 / VIDEO_DUMP_FPS:
      return
    if self.proc is None and not self._start():
      return

    try:
      assert self.proc is not None
      assert self.proc.stdin is not None
      self.proc.stdin.write(rgb.tobytes())
      self.last_frame_t = now
    except (BrokenPipeError, OSError):
      cloudlog.exception("vision_lane_change_riskd debug video writer failed")
      self.close()
      self.unavailable = True

  def _start(self) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
      cloudlog.warning("vision_lane_change_riskd debug video disabled: ffmpeg not found")
      self.unavailable = True
      return False

    os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
    self.output_path = os.path.join(DEBUG_DUMP_DIR, f"vlcr_tracks_{int(time.time())}.mp4")
    cmd = [
      ffmpeg,
      "-y",
      "-f", "rawvideo",
      "-pixel_format", "rgb24",
      "-video_size", f"{GRID_W}x{GRID_H}",
      "-framerate", str(VIDEO_DUMP_FPS),
      "-i", "-",
      "-an",
      "-vcodec", "libx264",
      "-preset", "ultrafast",
      "-pix_fmt", "yuv420p",
      self.output_path,
    ]
    try:
      self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      self.started_t = time.monotonic()
      cloudlog.warning(f"vision_lane_change_riskd debug video recording {self.output_path}")
      return True
    except OSError:
      cloudlog.exception("vision_lane_change_riskd debug video failed to start")
      self.unavailable = True
      return False

  def close(self) -> None:
    if self.proc is None:
      return
    try:
      if self.proc.stdin is not None:
        self.proc.stdin.close()
      self.proc.wait(timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
      self.proc.kill()
    finally:
      self.proc = None


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
  streamer = ProcessedFrameStreamer()
  video_recorder = DebugVideoRecorder()
  atexit.register(video_recorder.close)
  last_debug_dump_t = 0.0
  rk = Ratekeeper(MODEL_RATE, print_delay_threshold=None)

  while True:
    now = time.monotonic()
    frame, frame_id, timestamp_sof = read_tuned_frame(clients)
    sm.update(0)

    msg = messaging.new_message("visionLaneChangeRisk")
    risk = msg.visionLaneChangeRisk
    risk.valid = False
    risk.source = Source.commonFrame

    if frame is not None:
      tracker.update(frame, model_lead_detections(sm["modelV2"]))
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
      overlay = debug_frame_rgb(
        frame,
        tracker.left.risk,
        tracker.right.risk,
        tracker.left.confidence,
        tracker.right.confidence,
        tracker.tracks,
      )
      streamer.send(overlay, frame_id, timestamp_sof)

      debug_enabled = (
        os.getenv("VLCR_DEBUG_PNGS") == "1" or
        params.get_bool("VisionLaneChangeRiskDebug")
      )
      if debug_enabled:
        video_recorder.update(overlay, now)
      if debug_enabled and now - last_debug_dump_t >= DEBUG_DUMP_INTERVAL:
        filename = f"vlcr_{frame_id:08d}_{timestamp_sof}_processed.png"
        path = os.path.join(DEBUG_DUMP_DIR, filename)
        write_debug_png(
          path,
          frame,
          tracker.left.risk,
          tracker.right.risk,
          tracker.left.confidence,
          tracker.right.confidence,
          tracker.tracks,
        )
        last_debug_dump_t = now

    pm.send("visionLaneChangeRisk", msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
