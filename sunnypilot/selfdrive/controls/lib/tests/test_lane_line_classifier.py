"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Synthetic tests for the solid-vs-broken lane-line classifier. These render a
known 3D line into a fake luminance frame using a pinhole transform, so the full
pipeline (project -> lateral world-space sample -> classify) is exercised with
no camera or route required.

The synthetic camera mirrors the real tici road camera (focal 2648 px,
1928x1208) and the paint is rendered at a realistic 13 cm world width. That
matters: with the real focal length a marking is ~30 px wide at 10 m and ~5 px
at 60 m, which is exactly the geometry that broke the old fixed-pixel scan.
"""
import numpy as np

from sunnypilot.selfdrive.controls.lib.lane_line_classifier import (
  LaneLineType, classify_line, LaneLineClassifier, LaneLineClassifierConfig,
  SAMPLE_X_MAX,
)

# --- synthetic camera: real tici fcam geometry -------------------------------
# Device frame: x fwd, y left, z up.  View frame: x right, y down, z fwd.
_R = np.array([[0.0, -1.0, 0.0],
               [0.0, 0.0, -1.0],
               [1.0, 0.0, 0.0]])
_F, _CX, _CY = 2648.0, 964.0, 604.0
_K = np.array([[_F, 0.0, _CX], [0.0, _F, _CY], [0.0, 0.0, 1.0]])
TRANSFORM = _K @ _R
FRAME_H, FRAME_W = 1208, 1928
CAM_Z = -1.2          # ground plane height relative to camera (m)
ROAD_LUMA = 70
PAINT_LUMA = 190
PAINT_W = 0.13        # world width of the marking (m), realistic US/EU paint


def _project(p):
  q = TRANSFORM @ np.asarray(p, dtype=np.float64)
  if q[2] <= 1e-6:
    return None
  return q[0] / q[2], q[1] / q[2]


def _paint_strip(frame, x, dx, line_y, luma):
  """Paint the marking between forward distances x and x+dx with true world width.

  Both the lateral (paint width) and longitudinal (dx) extents are projected,
  so far-field paint is as thin vertically as the real camera would see it.
  """
  a = _project([x, line_y + PAINT_W / 2, CAM_Z])
  b = _project([x, line_y - PAINT_W / 2, CAM_Z])
  c = _project([x + dx, line_y, CAM_Z])
  if a is None or b is None or c is None:
    return
  u0, u1 = sorted((a[0], b[0]))
  v0, v1 = sorted(((a[1] + b[1]) / 2, c[1]))
  x0, x1 = max(0, int(np.floor(u0))), min(FRAME_W, int(np.ceil(u1)) + 1)
  y0, y1 = max(0, int(np.floor(v0))), min(FRAME_H, max(int(np.ceil(v1)), int(np.floor(v0)) + 1))
  if x1 > x0 and y1 > y0:
    frame[y0:y1, x0:x1] = luma


def _render(line_y=1.85, solid=True, paint_m=3.0, period_m=12.0, noise=6,
            road=ROAD_LUMA, paint=PAINT_LUMA, seed=0, dash_spans=None):
  """Render a frame with a line at lateral offset line_y, plus its (x,y,z) grid.

  ``dash_spans``: explicit [(x0, x1), ...] painted intervals; overrides
  solid/paint_m/period_m when given (for irregular-dash tests).
  """
  rng = np.random.default_rng(seed)
  frame = (road + rng.normal(0, noise, size=(FRAME_H, FRAME_W))).clip(0, 255).astype(np.uint8)

  step = 0.05
  for x in np.arange(2.0, SAMPLE_X_MAX + 10.0, step):
    if dash_spans is not None:
      if not any(x0 <= x < x1 for x0, x1 in dash_spans):
        continue
    elif not solid and (x % period_m) >= paint_m:
      continue
    _paint_strip(frame, x, step, line_y, paint)

  xs = np.linspace(0.0, SAMPLE_X_MAX + 10.0, 40)
  ys = np.full_like(xs, line_y)
  zs = np.full_like(xs, CAM_Z)
  return frame, xs, ys, zs


def _occlude_span(frame, x_start, x_end, line_y=1.85, pad_m=0.3, value=ROAD_LUMA):
  """Blank the marking between two forward distances (e.g. car shadow, dirt).

  Padding is in world metres so the occlusion depth doesn't balloon with
  distance the way a fixed pixel pad would.
  """
  uv0 = _project([x_start - pad_m, line_y, CAM_Z])
  uv1 = _project([x_end + pad_m, line_y, CAM_Z])
  ua = _project([x_start - pad_m, line_y + 0.3, CAM_Z])
  ub = _project([x_start - pad_m, line_y - 0.3, CAM_Z])
  assert uv0 is not None and uv1 is not None and ua is not None and ub is not None
  x0 = int(min(ua[0], ub[0]))
  x1 = int(max(ua[0], ub[0])) + 1
  y0 = int(min(uv0[1], uv1[1])) - 1
  y1 = int(max(uv0[1], uv1[1])) + 2
  frame[max(0, y0):min(FRAME_H, y1), max(0, x0):min(FRAME_W, x1)] = value


def test_solid_line_detected():
  # regression: with real focal length + realistic paint width, the old
  # fixed-pixel perpendicular scan read contrast ~0 for a perfectly projected
  # solid line everywhere nearer than ~35 m and misclassified it as BROKEN
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


def test_short_dash_detected():
  frame, x, y, z = _render(solid=False, paint_m=1.0, period_m=4.0)
  res = classify_line(frame, x, y, z, TRANSFORM)
  assert res.line_type == LaneLineType.BROKEN, res
  assert 2.0 <= res.period_m <= 6.0, res.period_m


def test_projection_offset_still_detects():
  # model/calibration error moves the projected line off the paint; the
  # world-space scan half-width is the tolerance budget
  for offset_m in (0.10, 0.20, 0.30):
    frame, x, y, z = _render(solid=True)
    res = classify_line(frame, x, y + offset_m, z, TRANSFORM)
    assert res.line_type == LaneLineType.SOLID, (offset_m, res)
    frame, x, y, z = _render(solid=False)
    res = classify_line(frame, x, y + offset_m, z, TRANSFORM)
    assert res.line_type == LaneLineType.BROKEN, (offset_m, res)


def test_low_contrast_night_paint():
  # dim but clean paint (night / washed out): fails the absolute contrast
  # floor but passes the double-SNR path
  frame, x, y, z = _render(solid=True, road=60, paint=74, noise=1, seed=3)
  res = classify_line(frame, x, y, z, TRANSFORM)
  assert res.line_type == LaneLineType.SOLID, res
  frame, x, y, z = _render(solid=False, road=60, paint=74, noise=1, seed=4)
  res = classify_line(frame, x, y, z, TRANSFORM)
  assert res.line_type == LaneLineType.BROKEN, res


def test_heavy_road_texture_not_mistaken_for_paint():
  # noisy road surface produces contrast spikes; the SNR gate must reject them
  rng = np.random.default_rng(11)
  frame = (ROAD_LUMA + rng.normal(0, 16, size=(FRAME_H, FRAME_W))).clip(0, 255).astype(np.uint8)
  x = np.linspace(0, SAMPLE_X_MAX + 10, 40)
  res = classify_line(frame, x, np.full_like(x, 1.85), np.full_like(x, CAM_Z), TRANSFORM)
  assert res.line_type == LaneLineType.UNKNOWN, res


def test_short_occlusion_does_not_break_solid_line():
  frame, x, y, z = _render(solid=True)
  _occlude_span(frame, 20.0, 20.8)
  res = classify_line(frame, x, y, z, TRANSFORM)
  assert res.line_type == LaneLineType.SOLID, res


def test_fragmented_solid_line_prefers_solid():
  # a few 1.5 m dropouts (dirt, cracks) exceed the gap-repair budget but the
  # continuity bias should still land on SOLID, not BROKEN/UNKNOWN
  frame, x, y, z = _render(solid=True)
  for x_start, x_end in ((16.0, 17.5), (26.0, 27.5), (38.0, 39.5)):
    _occlude_span(frame, x_start, x_end)
  cfg = LaneLineClassifierConfig(solid_duty=0.95)
  res = classify_line(frame, x, y, z, TRANSFORM, cfg=cfg)
  assert res.line_type == LaneLineType.SOLID, res
  assert res.duty < cfg.solid_duty, res.duty


def test_broken_line_not_misclassified_as_continuous_solid():
  frame, x, y, z = _render(solid=False, paint_m=5.0, period_m=12.0)
  cfg = LaneLineClassifierConfig(solid_duty=0.90)
  res = classify_line(frame, x, y, z, TRANSFORM, cfg=cfg)
  assert res.line_type == LaneLineType.BROKEN, res


def test_irregular_dashes_classified_by_run_shape():
  # real dashes are often too irregular for a clean autocorrelation peak
  # (worn paint, merges, resurfacing joints); the run-length fallback should
  # still call them BROKEN
  spans = [(6.0, 9.5), (13.0, 15.0), (24.0, 28.0), (39.0, 41.5), (52.0, 55.0)]
  frame, x, y, z = _render(dash_spans=spans)
  # force the autocorrelation path off so the fallback is what's under test
  cfg = LaneLineClassifierConfig(min_autocorr=0.99)
  res = classify_line(frame, x, y, z, TRANSFORM, cfg=cfg)
  assert res.line_type == LaneLineType.BROKEN, res
  assert res.confidence <= 0.65  # fallback is deliberately less confident


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


def test_config_overrides_are_honored():
  frame, x, y, z = _render(solid=False, paint_m=3.0, period_m=12.0)
  # default config -> BROKEN
  assert classify_line(frame, x, y, z, TRANSFORM).line_type == LaneLineType.BROKEN
  # an unreachable contrast floor -> nothing detected -> UNKNOWN
  # (the dim-paint path floors at 0.4x min_contrast, so use 400 to clear both)
  no_contrast = LaneLineClassifierConfig(min_contrast=400.0)
  assert classify_line(frame, x, y, z, TRANSFORM, cfg=no_contrast).line_type == LaneLineType.UNKNOWN
  # a solid line with an unreachable solid-duty threshold is no longer SOLID
  fs, sx, sy, sz = _render(solid=True)
  assert classify_line(fs, sx, sy, sz, TRANSFORM).line_type == LaneLineType.SOLID
  never_solid = LaneLineClassifierConfig(solid_duty=1.01, solid_contig_frac=2.0, solid_top2_frac=2.0)
  assert classify_line(fs, sx, sy, sz, TRANSFORM, cfg=never_solid).line_type != LaneLineType.SOLID


class _FakeLine:
  def __init__(self, x, y, z):
    self.x, self.y, self.z = x, y, z


class _FakeModel:
  def __init__(self, lines, probs=None):
    self.laneLines = lines
    if probs is not None:
      self.laneLineProbs = probs


def test_classifier_debounce_latches_crossable():
  frame_broken, x, y, z = _render(solid=False)
  far = _FakeLine([0, 100], [6, 6], [CAM_Z, CAM_Z])  # dummy outer lines
  # left ego line broken, right ego line solid (rendered off-frame right side)
  broken_line = _FakeLine(x, y, z)
  solid_line = _FakeLine(x, np.full_like(np.asarray(x, float), -1.85), z)

  clf = LaneLineClassifier()
  model = _FakeModel([far, broken_line, solid_line, far])
  gate = None
  for _ in range(4):  # needs COUNTER_ON frames to latch
    gate = clf.update(frame_broken, model, TRANSFORM)
  assert gate.left.line_type == LaneLineType.BROKEN
  assert gate.left_crossable is True
  assert gate.right_crossable is False


def test_low_model_prob_line_is_unknown():
  # when modelV2 barely sees a line, don't classify whatever texture happens
  # to be at its polyline: fail safe to UNKNOWN
  frame, x, y, z = _render(solid=True)
  far = _FakeLine([0, 100], [6, 6], [CAM_Z, CAM_Z])
  line = _FakeLine(x, y, z)
  clf = LaneLineClassifier()
  model = _FakeModel([far, line, far, far], probs=[0.0, 0.05, 0.0, 0.0])
  gate = clf.update(frame, model, TRANSFORM)
  assert gate.left.line_type == LaneLineType.UNKNOWN
  # same geometry with a confident model -> classified
  model = _FakeModel([far, line, far, far], probs=[0.0, 0.9, 0.0, 0.0])
  gate = LaneLineClassifier().update(frame, model, TRANSFORM)
  assert gate.left.line_type == LaneLineType.SOLID


def test_temporal_filter_holds_confident_label_for_one_bad_frame():
  frame_solid, x, y, z = _render(solid=True)
  rng = np.random.default_rng(22)
  blank = (ROAD_LUMA + rng.normal(0, 6, size=(FRAME_H, FRAME_W))).clip(0, 255).astype(np.uint8)
  far = _FakeLine([0, 100], [6, 6], [CAM_Z, CAM_Z])
  solid_line = _FakeLine(x, y, z)

  clf = LaneLineClassifier()
  model = _FakeModel([far, solid_line, far, far])
  gate = clf.update(frame_solid, model, TRANSFORM)
  assert gate.left.line_type == LaneLineType.SOLID

  gate = clf.update(blank, model, TRANSFORM)
  assert gate.left.line_type == LaneLineType.SOLID, gate.left
