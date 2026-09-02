import numpy as np

from openpilot.sunnypilot.selfdrive.vision_lane_change_risk.common_frame_tracker import (
  CommonFrameMotionTracker,
  GRID_H,
  GRID_W,
  LEFT_CONFLICT,
  compose_raw_strip,
  compose_tuned_frame,
  compose_tuned_frame_from_raw,
  debug_frame_rgb,
  region_pixels,
  rgb_to_yuv420,
  write_debug_png,
)


def test_persistent_left_motion_sets_left_risk_only():
  tracker = CommonFrameMotionTracker()
  frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)
  tracker.update(frame)

  x0, y0, x1, y1 = region_pixels(LEFT_CONFLICT, GRID_W, GRID_H)
  for i in range(8):
    frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)
    frame[y0:y1, x0 + i:x1] = 130
    tracker.update(frame)

  assert tracker.left.risk
  assert not tracker.right.risk
  assert len(tracker.tracks) == 1
  assert tracker.tracks[0].side == "left"


def test_global_brightness_change_does_not_create_risk():
  tracker = CommonFrameMotionTracker()
  tracker.update(np.full((GRID_H, GRID_W), 80, dtype=np.uint8))

  for val in (90, 70, 92, 74, 88, 80):
    tracker.update(np.full((GRID_H, GRID_W), val, dtype=np.uint8))

  assert not tracker.left.risk
  assert not tracker.right.risk
  assert not tracker.tracks


def test_track_bbox_moves_with_side_motion():
  tracker = CommonFrameMotionTracker()
  tracker.update(np.full((GRID_H, GRID_W), 80, dtype=np.uint8))

  x0, y0, x1, y1 = region_pixels(LEFT_CONFLICT, GRID_W, GRID_H)
  for offset in (0, 8, 16, 24):
    frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)
    frame[y0 + 40:y0 + 100, x0 + 50 + offset:x0 + 170 + offset] = 140
    tracker.update(frame)

  tracks = tracker.tracks
  assert len(tracks) == 1
  assert tracks[0].age >= 1
  assert tracks[0].x0 >= x0 + 50
  assert tracks[0].vx > 0.0


def test_track_id_persists_from_center_to_side_zone():
  tracker = CommonFrameMotionTracker()
  tracker.update(np.full((GRID_H, GRID_W), 80, dtype=np.uint8))

  track_ids = []
  for x in (1180, 1280, 1390, 1510):
    frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)
    frame[250:330, x:x + 160] = 145
    tracker.update(frame)
    tracks = tracker.tracks
    assert tracks
    track_ids.append(tracks[0].track_id)

  assert len(set(track_ids)) == 1
  assert tracker.tracks[0].side == "right"


def test_tuned_frame_uses_wide_and_cabin_regions():
  cabin = np.zeros((4, 8), dtype=np.uint8)
  cabin[:, :4] = 40
  cabin[:, 4:] = 80
  wide = np.full((4, 8), 160, dtype=np.uint8)

  frame = compose_tuned_frame({"wide": wide, "cabin": cabin})

  assert frame is not None
  assert frame.shape == (GRID_H, GRID_W)
  assert frame[256, 1024] == 160
  assert np.any(frame[:, :512] == 80)
  assert np.any(frame[:, 1536:] == 40)


def test_write_debug_png(tmp_path):
  path = tmp_path / "debug.png"
  frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)

  write_debug_png(str(path), frame, True, False, 0.75, 0.25)

  data = path.read_bytes()
  assert data.startswith(b"\x89PNG\r\n\x1a\n")
  assert b"IHDR" in data
  assert b"IDAT" in data


def test_debug_rgb_and_yuv420_include_track_overlay():
  frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)
  tracker = CommonFrameMotionTracker()
  tracker.update(frame)
  x0, y0, _, _ = region_pixels(LEFT_CONFLICT, GRID_W, GRID_H)
  for _ in range(4):
    frame = np.full((GRID_H, GRID_W), 80, dtype=np.uint8)
    frame[y0 + 30:y0 + 90, x0 + 40:x0 + 150] = 145
    tracker.update(frame)

  rgb = debug_frame_rgb(frame, tracker.left.risk, tracker.right.risk, 0.5, 0.0, tracker.tracks)
  payload = rgb_to_yuv420(rgb)

  assert rgb.shape == (GRID_H, GRID_W, 3)
  assert len(payload) == GRID_W * GRID_H * 3 // 2
  assert np.any(np.all(rgb == np.array([255, 96, 64], dtype=np.uint8), axis=2))


def test_compose_raw_strip_uses_left_dm_wide_right_dm():
  cabin = np.zeros((4, 8), dtype=np.uint8)
  cabin[:, :4] = 40
  cabin[:, 4:] = 80
  wide = np.full((4, 8), 160, dtype=np.uint8)

  strip = compose_raw_strip({"cabin": cabin, "wide": wide})

  assert strip is not None
  assert strip.shape == (512, 2048)
  assert strip[256, 128] == 40
  assert strip[256, 768] == 160
  assert strip[256, 1920] == 80


def test_compose_tuned_frame_from_raw_swaps_rotated_dm_panels():
  cabin = np.zeros((4, 8), dtype=np.uint8)
  cabin[:, :4] = 40
  cabin[:, 4:] = 80
  wide = np.full((4, 8), 160, dtype=np.uint8)

  strip = compose_raw_strip({"cabin": cabin, "wide": wide})
  assert strip is not None

  v2 = compose_tuned_frame_from_raw(strip)

  assert v2.shape == (512, 2048)
  assert v2[256, 1024] == 160
  assert np.any(v2[:, :512] == 80)
  assert np.any(v2[:, 1536:] == 40)
  assert v2[0, 0] == 255
