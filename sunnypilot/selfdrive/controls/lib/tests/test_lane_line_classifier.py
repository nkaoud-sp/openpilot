"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Synthetic tests for the solid-vs-broken lane-line classifier. These render a
known 3D line into a fake luminance frame using a pinhole transform, so the full
pipeline (project -> perpendicular sample -> classify) is exercised with no
camera or route required.
"""
import numpy as np

from sunnypilot.selfdrive.controls.lib.lane_line_classifier import (
  LaneLineType, classify_line, LaneLineClassifier,
  SAMPLE_X_MIN, SAMPLE_X_MAX,
)

# --- synthetic camera --------------------------------------------------------
# Device frame: x fwd, y left, z up.  View frame: x right, y down, z fwd.
_R = np.array([[0.0, -1.0, 0.0],
               [0.0, 0.0, -1.0],
               [1.0, 0.0, 0.0]])
_F, _CX, _CY = 910.0, 964.0, 604.0
_K = np.array([[_F, 0.0, _CX], [0.0, _F, _CY], [0.0, 0.0, 1.0]])
TRANSFORM = _K @ _R
FRAME_H, FRAME_W = 1208, 1928
CAM_Z = -1.2          # ground plane height relative to camera (m)
ROAD_LUMA = 70
PAINT_LUMA = 235


def _project(p):
  q = TRANSFORM @ np.asarray(p, dtype=np.float64)
  if q[2] <= 1e-6:
    return None
  return q[0] / q[2], q[1] / q[2]


def _render(line_y=1.85, solid=True, paint_m=3.0, period_m=12.0, noise=6):
  """Render a frame with a line at lateral offset line_y, plus its (x,y,z) grid."""
  rng = np.random.default_rng(0)
  frame = (ROAD_LUMA + rng.normal(0, noise, size=(FRAME_H, FRAME_W))).clip(0, 255).astype(np.uint8)

  # paint finely so the perpendicular scan always finds the marking
  for x in np.arange(SAMPLE_X_MIN - 1.0, SAMPLE_X_MAX + 5.0, 0.1):
    if not solid and (x % period_m) >= paint_m:
      continue
    uv = _project([x, line_y, CAM_Z])
    if uv is None:
      continue
    cx, cy = int(round(uv[0])), int(round(uv[1]))
    # a small paint blob; wider near-field, thinner far-field ~ constant world width
    r = max(1, int(round(3 * (10.0 / max(x, 5.0)))))
    y0, y1 = max(0, cy - r), min(FRAME_H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(FRAME_W, cx + r + 1)
    frame[y0:y1, x0:x1] = PAINT_LUMA

  xs = np.linspace(0.0, SAMPLE_X_MAX + 10.0, 40)
  ys = np.full_like(xs, line_y)
  zs = np.full_like(xs, CAM_Z)
  return frame, xs, ys, zs


def test_solid_line_detected():
  frame, x, y, z = _render(solid=True)
  res = classify_line(frame, x, y, z, TRANSFORM)
  assert res.line_type == LaneLineType.SOLID, res
  assert res.duty >= 0.8
  assert res.confidence > 0.5


def test_broken_line_detected():
  frame, x, y, z = _render(solid=False, paint_m=3.0, period_m=12.0)
  res = classify_line(frame, x, y, z, TRANSFORM)
  assert res.line_type == LaneLineType.BROKEN, res
  # recovered period should be near the true 12 m dash cycle
  assert 8.0 <= res.period_m <= 16.0, res.period_m
  assert 0.1 <= res.duty <= 0.75, res.duty


def test_blank_frame_is_unknown():
  rng = np.random.default_rng(1)
  frame = (ROAD_LUMA + rng.normal(0, 6, size=(FRAME_H, FRAME_W))).clip(0, 255).astype(np.uint8)
  x = np.linspace(0, SAMPLE_X_MAX + 10, 40)
  res = classify_line(frame, x, np.full_like(x, 1.85), np.full_like(x, CAM_Z), TRANSFORM)
  assert res.line_type == LaneLineType.UNKNOWN, res
  assert not res.crossable


def test_short_line_is_unknown():
  frame, _, _, _ = _render(solid=True)
  res = classify_line(frame, [1.0, 2.0], [1.85, 1.85], [CAM_Z, CAM_Z], TRANSFORM)
  assert res.line_type == LaneLineType.UNKNOWN


class _FakeLine:
  def __init__(self, x, y, z):
    self.x, self.y, self.z = x, y, z


class _FakeModel:
  def __init__(self, lines):
    self.laneLines = lines


def test_classifier_debounce_latches_crossable():
  frame_solid, x, y, z = _render(solid=True)
  frame_broken, *_ = _render(solid=False)
  far = _FakeLine([0, 100], [6, 6], [CAM_Z, CAM_Z])  # dummy outer lines
  # left ego line broken, right ego line solid
  bl, bx, by, bz = frame_broken, x, y, z
  broken_line = _FakeLine(bx, by, bz)
  solid_line = _FakeLine(x, np.full_like(np.asarray(x, float), -1.85), z)

  clf = LaneLineClassifier()
  model = _FakeModel([far, broken_line, solid_line, far])
  gate = None
  for _ in range(4):  # needs COUNTER_ON frames to latch
    gate = clf.update(frame_broken, model, TRANSFORM)
  assert gate.left.line_type == LaneLineType.BROKEN
  assert gate.left_crossable is True
  assert gate.right_crossable is False


if __name__ == "__main__":
  import sys
  import pytest
  sys.exit(pytest.main([__file__, "-v"]))
