import numpy as np

from openpilot.sunnypilot.selfdrive.vision_lane_change_risk.common_frame_tracker import (
  CommonFrameMotionTracker,
  GRID_H,
  GRID_W,
  compose_common_frame,
  write_debug_png,
)


def test_persistent_left_motion_sets_left_risk_only():
  tracker = CommonFrameMotionTracker()
  frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)
  tracker.update(frame)

  for i in range(8):
    frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)
    frame[18:32, 3 + i:27 + i] = 130
    tracker.update(frame)

  assert tracker.left.risk
  assert not tracker.right.risk


def test_global_brightness_change_does_not_create_risk():
  tracker = CommonFrameMotionTracker()
  tracker.update(np.full((GRID_H, GRID_W), 80, dtype=np.uint8))

  for val in (90, 70, 92, 74, 88, 80):
    tracker.update(np.full((GRID_H, GRID_W), val, dtype=np.uint8))

  assert not tracker.left.risk
  assert not tracker.right.risk


def test_common_frame_uses_wide_narrow_and_cabin_regions():
  wide = np.full((GRID_H, GRID_W), 20, dtype=np.uint8)
  narrow = np.full((GRID_H, GRID_W), 100, dtype=np.uint8)
  cabin = np.full((GRID_H, GRID_W), 200, dtype=np.uint8)

  common = compose_common_frame({"wide": wide, "narrow": narrow, "cabin": cabin})

  assert common is not None
  assert common.shape == (GRID_H, GRID_W)
  assert common[5, 5] == 20
  assert common[5, GRID_W // 2] == 100
  assert 100 < common[-2, 5] < 200


def test_write_debug_png(tmp_path):
  path = tmp_path / "debug.png"
  frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)

  write_debug_png(str(path), frame, True, False, 0.75, 0.25)

  data = path.read_bytes()
  assert data.startswith(b"\x89PNG\r\n\x1a\n")
  assert b"IHDR" in data
  assert b"IDAT" in data
