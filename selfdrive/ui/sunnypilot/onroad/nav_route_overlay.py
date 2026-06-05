"""
Projects the active Mapbox route polyline (navRoute) onto the driving view.

Supports several rendering styles via NkaoudNavPolylineStyle:
  0 SOLID     -- original sharp blue stroke with black outline.
  1 SMOOTH    -- Catmull-Rom interpolated curve, width tapers wider near the
                 car, alpha fades toward the horizon. Calm + polished.
  2 GLOW      -- smooth curve with a multi-pass halo, neon look.
  3 CHEVRONS  -- animated forward-flowing chevrons over a faint base line.
  4 RIBBON    -- filled ribbon (quad strip) with width that tapers; a soft
                 lane-shaped "this is the route" swath rather than a stroke.
  5 DASHED    -- dashes at constant world-space spacing, walked along the
                 path; reads like a navigation-app waypoint dotted line.
  6 SMOKE     -- diffuse soft trail; 5 stacked passes of varying width
                 and slight horizontal jitter, looks like a thick aurora.
  7 COMPOSITE -- SMOOTH base with CHEVRONS overlaid for direction + body.

Gated on NkaoudNavEnabled + NkaoudNavShowPolyline.

Device frame convention (per common/transformations/camera.py): x=forward,
y=right, z=down. Road surface = z = +live_calibration.height.
"""
from __future__ import annotations

import math
import time
from enum import IntEnum

import numpy as np
import pyray as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app


MAX_RENDER_DISTANCE_M = 250.0
MIN_FORWARD_M = 3.0
MIN_PROJ_Z = 0.1
PARAM_REFRESH_S = 0.5
METERS_PER_DEG_LAT = 111320.0
Z_GROUND_OFFSET_M = 0.10
# Tilt the line forward so the far end lifts off the road and the near end
# stays on it. Subtracted from z (z = down in device frame), so a positive
# value here raises the far end UP in world space. The lift scales linearly
# with forward distance: 0 at the camera, LIFT_FAR_M at MAX_RENDER_DISTANCE_M.
LIFT_FAR_M = 1.0

# Smoothing time constants.
RC_POS = 0.10
RC_YAW = 0.20

# Style-specific constants.
BLUE = (0, 191, 255)

# SOLID
SOLID_THICKNESS = 10.0
SOLID_OUTLINE_EXTRA = 4.0
SOLID_ALPHA = 220

# SMOOTH (default)
SMOOTH_THICKNESS_NEAR = 16.0
SMOOTH_THICKNESS_FAR = 4.0
SMOOTH_ALPHA_NEAR = 230
SMOOTH_ALPHA_FAR = 60
SMOOTH_SUBSAMPLES = 8

# GLOW (smooth + halo)
GLOW_LAYERS = (
  (26.0, 30),    # widest, faintest
  (18.0, 55),
  (12.0, 90),
  (6.0, 220),    # core
)

# CHEVRONS
CHEVRON_SPACING_M = 22.0
CHEVRON_FLOW_SPEED_MS = 6.0
CHEVRON_SIZE_NEAR = 42.0
CHEVRON_SIZE_FAR = 12.0
CHEVRON_THICKNESS_NEAR = 8.0
CHEVRON_THICKNESS_FAR = 3.0
CHEVRON_BASELINE_ALPHA = 60     # faint base line under the chevrons

# RIBBON (screen-space perpendicular offset; tapers with forward distance)
RIBBON_HALFWIDTH_NEAR = 52.0
RIBBON_HALFWIDTH_FAR = 10.0
RIBBON_FILL_ALPHA_NEAR = 150
RIBBON_FILL_ALPHA_FAR = 35
RIBBON_EDGE_ALPHA_NEAR = 230
RIBBON_EDGE_ALPHA_FAR = 70

# DASHED (constant world-space step)
DASH_ON_M = 4.0
DASH_OFF_M = 3.0
DASH_THICKNESS_NEAR = 14.0
DASH_THICKNESS_FAR = 4.0
DASH_ALPHA_NEAR = 240
DASH_ALPHA_FAR = 60

# SMOKE (multi-pass diffuse trail)
SMOKE_LAYERS = (
  # (width, base_alpha, x_jitter_px)
  (28.0, 22, -4),
  (22.0, 35, +3),
  (16.0, 55, -2),
  (10.0, 90, +1),
  (4.0,  220, 0),     # bright core
)


class NavPolylineStyle(IntEnum):
  SOLID = 0
  SMOOTH = 1
  GLOW = 2
  CHEVRONS = 3
  RIBBON = 4
  DASHED = 5
  SMOKE = 6
  COMPOSITE = 7


def _clip01(t: float) -> float:
  return max(0.0, min(1.0, t))


def _lerp(a: float, b: float, t: float) -> float:
  return a + (b - a) * t


def _rgba(color: tuple[int, int, int], alpha: int) -> rl.Color:
  return rl.Color(color[0], color[1], color[2], max(0, min(255, int(alpha))))


def _catmull_rom(points: list[rl.Vector2], forwards: list[float],
                 samples: int) -> tuple[list[rl.Vector2], list[float]]:
  """Catmull-Rom-interpolate a polyline in screen space. Forwards are
  linearly interpolated alongside so downstream code can still know each
  sample's world-forward distance (for width/alpha tapering)."""
  n = len(points)
  if n < 2:
    return points[:], forwards[:]
  out_pts: list[rl.Vector2] = [points[0]]
  out_fwd: list[float] = [forwards[0]]
  for i in range(n - 1):
    p0 = points[max(0, i - 1)]
    p1 = points[i]
    p2 = points[i + 1]
    p3 = points[min(n - 1, i + 2)]
    f1 = forwards[i]
    f2 = forwards[i + 1]
    for j in range(1, samples + 1):
      t = j / samples
      t2 = t * t
      t3 = t2 * t
      x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t
                 + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2
                 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3)
      y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t
                 + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2
                 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3)
      out_pts.append(rl.Vector2(float(x), float(y)))
      out_fwd.append(f1 + (f2 - f1) * t)
  return out_pts, out_fwd


class NavRouteOverlay:
  def __init__(self) -> None:
    self._transform = np.eye(3, dtype=np.float32)
    self._enabled = False
    self._show_polyline = False
    self._style = NavPolylineStyle.SMOOTH
    self._next_param_check = 0.0
    dt = 1.0 / max(1, gui_app.target_fps)
    self._lat_filter = FirstOrderFilter(0.0, RC_POS, dt)
    self._lon_filter = FirstOrderFilter(0.0, RC_POS, dt)
    self._yaw_sin_filter = FirstOrderFilter(0.0, RC_YAW, dt)
    self._yaw_cos_filter = FirstOrderFilter(1.0, RC_YAW, dt)
    self._filters_seeded = False

  def set_transform(self, m: np.ndarray) -> None:
    self._transform = m.astype(np.float32)

  def _refresh_params(self) -> None:
    now = time.monotonic()
    if now < self._next_param_check:
      return
    self._next_param_check = now + PARAM_REFRESH_S
    self._enabled = ui_state.params.get_bool("NkaoudNavEnabled")
    self._show_polyline = ui_state.params.get_bool("NkaoudNavShowPolyline")
    style_raw = ui_state.params.get("NkaoudNavPolylineStyle", return_default=True) or 1
    try:
      self._style = NavPolylineStyle(int(style_raw))
    except (ValueError, TypeError):
      self._style = NavPolylineStyle.SMOOTH

  def _smoothed_pose(self, lat: float, lon: float, yaw_rad: float) -> tuple[float, float, float]:
    if not self._filters_seeded:
      self._lat_filter.x = lat
      self._lon_filter.x = lon
      self._yaw_sin_filter.x = math.sin(yaw_rad)
      self._yaw_cos_filter.x = math.cos(yaw_rad)
      self._filters_seeded = True
      return lat, lon, yaw_rad
    self._lat_filter.update(lat)
    self._lon_filter.update(lon)
    self._yaw_sin_filter.update(math.sin(yaw_rad))
    self._yaw_cos_filter.update(math.cos(yaw_rad))
    smoothed_yaw = math.atan2(self._yaw_sin_filter.x, self._yaw_cos_filter.x)
    return self._lat_filter.x, self._lon_filter.x, smoothed_yaw

  def render(self, rect: rl.Rectangle) -> None:
    self._refresh_params()
    if not (self._enabled and self._show_polyline):
      return

    sm = ui_state.sm
    llk = sm['liveLocationKalman']
    if not llk.gpsOK:
      return
    geo = llk.positionGeodetic
    if not geo.valid or len(geo.value) < 2:
      return
    ori = llk.calibratedOrientationNED
    if not ori.valid or len(ori.value) != 3:
      return

    coords = sm['navRoute'].coordinates
    if len(coords) < 2:
      return

    lat0, lon0, yaw = self._smoothed_pose(geo.value[0], geo.value[1], ori.value[2])
    cos_lat0 = math.cos(math.radians(lat0))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    me_per_deg_lon = METERS_PER_DEG_LAT * cos_lat0

    calib = sm['liveCalibration']
    z_ground = (float(calib.height[0]) if (calib.height and len(calib.height))
                else float(HEIGHT_INIT[0])) + Z_GROUND_OFFSET_M

    # World-frame segments. Drop points outside [MIN_FORWARD_M, MAX_RENDER_DISTANCE_M].
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for c in coords:
      dn = (c.latitude - lat0) * METERS_PER_DEG_LAT
      de = (c.longitude - lon0) * me_per_deg_lon
      forward = de * sin_yaw + dn * cos_yaw
      right = de * cos_yaw - dn * sin_yaw
      in_window = (MIN_FORWARD_M <= forward <= MAX_RENDER_DISTANCE_M)
      if in_window:
        current.append((forward, right))
      elif current:
        segments.append(current)
        current = []
    if current:
      segments.append(current)

    for seg in segments:
      for sub_pts, sub_fwds in self._project_segment(seg, z_ground):
        self._draw_style(sub_pts, sub_fwds)

  # ---------- projection ----------
  def _project_segment(self, seg: list[tuple[float, float]], z_ground: float
                       ) -> list[tuple[list[rl.Vector2], list[float]]]:
    """Project a world-frame segment to a list of (screen_pts, forwards) sub-segments,
    breaking the segment any time a point projects to the wrong side of the image plane.

    z is lifted linearly with forward distance (LIFT_FAR_M at MAX_RENDER_DISTANCE_M)
    so the line tilts up toward the horizon instead of lying perfectly flat."""
    sub_segments: list[tuple[list[rl.Vector2], list[float]]] = []
    cur_pts: list[rl.Vector2] = []
    cur_fwd: list[float] = []
    for forward, right in seg:
      lift = LIFT_FAR_M * min(1.0, max(0.0, forward / MAX_RENDER_DISTANCE_M))
      z = z_ground - lift
      p = self._transform @ np.array([forward, right, z], dtype=np.float32)
      if p[2] < MIN_PROJ_Z:
        if len(cur_pts) >= 2:
          sub_segments.append((cur_pts, cur_fwd))
        cur_pts, cur_fwd = [], []
        continue
      cur_pts.append(rl.Vector2(float(p[0] / p[2]), float(p[1] / p[2])))
      cur_fwd.append(forward)
    if len(cur_pts) >= 2:
      sub_segments.append((cur_pts, cur_fwd))
    return sub_segments

  # ---------- style dispatcher ----------
  def _draw_style(self, pts: list[rl.Vector2], fwds: list[float]) -> None:
    if len(pts) < 2:
      return
    style = self._style
    if style == NavPolylineStyle.SOLID:
      self._draw_solid(pts)
    elif style == NavPolylineStyle.SMOOTH:
      self._draw_smooth(pts, fwds)
    elif style == NavPolylineStyle.GLOW:
      self._draw_glow(pts, fwds)
    elif style == NavPolylineStyle.CHEVRONS:
      self._draw_chevrons(pts, fwds)
    elif style == NavPolylineStyle.RIBBON:
      self._draw_ribbon(pts, fwds)
    elif style == NavPolylineStyle.DASHED:
      self._draw_dashed(pts, fwds)
    elif style == NavPolylineStyle.SMOKE:
      self._draw_smoke(pts, fwds)
    elif style == NavPolylineStyle.COMPOSITE:
      self._draw_smooth(pts, fwds)
      self._draw_chevrons(pts, fwds)

  # ---------- styles ----------
  def _draw_solid(self, pts: list[rl.Vector2]) -> None:
    outline_color = rl.Color(0, 0, 0, 140)
    primary = _rgba(BLUE, SOLID_ALPHA)
    for i in range(len(pts) - 1):
      rl.draw_line_ex(pts[i], pts[i + 1], SOLID_THICKNESS + SOLID_OUTLINE_EXTRA, outline_color)
    for i in range(len(pts) - 1):
      rl.draw_line_ex(pts[i], pts[i + 1], SOLID_THICKNESS, primary)

  def _draw_smooth(self, pts: list[rl.Vector2], fwds: list[float]) -> None:
    spts, sfwds = _catmull_rom(pts, fwds, SMOOTH_SUBSAMPLES)
    self._draw_tapered(spts, sfwds, SMOOTH_THICKNESS_NEAR, SMOOTH_THICKNESS_FAR,
                       SMOOTH_ALPHA_NEAR, SMOOTH_ALPHA_FAR, with_outline=True)

  def _draw_glow(self, pts: list[rl.Vector2], fwds: list[float]) -> None:
    spts, sfwds = _catmull_rom(pts, fwds, SMOOTH_SUBSAMPLES)
    for width_add, base_alpha in GLOW_LAYERS:
      for i in range(len(spts) - 1):
        a = self._alpha_at(sfwds[i], base_alpha, base_alpha // 4)
        rl.draw_line_ex(spts[i], spts[i + 1], width_add, _rgba(BLUE, a))

  def _draw_chevrons(self, pts: list[rl.Vector2], fwds: list[float]) -> None:
    # Faint base line so the path is still visible between chevrons.
    spts, sfwds = _catmull_rom(pts, fwds, SMOOTH_SUBSAMPLES)
    for i in range(len(spts) - 1):
      a = self._alpha_at(sfwds[i], CHEVRON_BASELINE_ALPHA, CHEVRON_BASELINE_ALPHA // 3)
      rl.draw_line_ex(spts[i], spts[i + 1], 4.0, _rgba(BLUE, a))

    # Flowing chevrons along the path. Each chevron is a "<" pointed forward.
    if len(spts) < 2:
      return
    min_f, max_f = sfwds[0], sfwds[-1]
    span = max_f - min_f
    if span < CHEVRON_SPACING_M:
      return
    flow_offset = (time.monotonic() * CHEVRON_FLOW_SPEED_MS) % CHEVRON_SPACING_M
    d = min_f + flow_offset
    seg_idx = 0
    while d < max_f:
      # advance segment pointer until sfwds[seg_idx+1] >= d
      while seg_idx < len(sfwds) - 2 and sfwds[seg_idx + 1] < d:
        seg_idx += 1
      f0, f1 = sfwds[seg_idx], sfwds[seg_idx + 1]
      if f1 - f0 > 1e-3:
        t = (d - f0) / (f1 - f0)
        cx = spts[seg_idx].x + t * (spts[seg_idx + 1].x - spts[seg_idx].x)
        cy = spts[seg_idx].y + t * (spts[seg_idx + 1].y - spts[seg_idx].y)
        dx = spts[seg_idx + 1].x - spts[seg_idx].x
        dy = spts[seg_idx + 1].y - spts[seg_idx].y
        n = math.hypot(dx, dy)
        if n > 1e-3:
          ux, uy = dx / n, dy / n
          t_far = _clip01(d / MAX_RENDER_DISTANCE_M)
          size = _lerp(CHEVRON_SIZE_NEAR, CHEVRON_SIZE_FAR, t_far)
          thick = _lerp(CHEVRON_THICKNESS_NEAR, CHEVRON_THICKNESS_FAR, t_far)
          a = self._alpha_at(d, 240, 90)
          color = _rgba(BLUE, a)
          tip = rl.Vector2(cx, cy)
          back_l = rl.Vector2(cx - ux * size - uy * size * 0.7,
                              cy - uy * size + ux * size * 0.7)
          back_r = rl.Vector2(cx - ux * size + uy * size * 0.7,
                              cy - uy * size - ux * size * 0.7)
          rl.draw_line_ex(back_l, tip, thick, color)
          rl.draw_line_ex(back_r, tip, thick, color)
      d += CHEVRON_SPACING_M

  def _draw_ribbon(self, pts: list[rl.Vector2], fwds: list[float]) -> None:
    spts, sfwds = _catmull_rom(pts, fwds, SMOOTH_SUBSAMPLES)
    if len(spts) < 2:
      return
    # Build left/right offset polylines in screen space. Width tapers with forward.
    left: list[rl.Vector2] = []
    right: list[rl.Vector2] = []
    for i in range(len(spts)):
      # Use the local tangent (segment to next, or previous if at the end).
      j = i + 1 if i + 1 < len(spts) else i - 1
      dx, dy = spts[j].x - spts[i].x, spts[j].y - spts[i].y
      n = math.hypot(dx, dy)
      if n < 1e-3:
        left.append(spts[i])
        right.append(spts[i])
        continue
      # Perpendicular (right-handed rotate 90° clockwise on screen: (dx, dy) -> (-dy, dx))
      px = -dy / n
      py = dx / n
      t_far = _clip01(sfwds[i] / MAX_RENDER_DISTANCE_M)
      half = _lerp(RIBBON_HALFWIDTH_NEAR, RIBBON_HALFWIDTH_FAR, t_far)
      left.append(rl.Vector2(spts[i].x + px * half, spts[i].y + py * half))
      right.append(rl.Vector2(spts[i].x - px * half, spts[i].y - py * half))

    # Quad strip filled. Two triangles per quad.
    for i in range(len(spts) - 1):
      t_far = _clip01(0.5 * (sfwds[i] + sfwds[i + 1]) / MAX_RENDER_DISTANCE_M)
      alpha = int(_lerp(RIBBON_FILL_ALPHA_NEAR, RIBBON_FILL_ALPHA_FAR, t_far))
      fill = _rgba(BLUE, alpha)
      # raylib expects vertices in counter-clockwise order for the visible face.
      rl.draw_triangle(left[i], right[i], right[i + 1], fill)
      rl.draw_triangle(left[i], right[i + 1], left[i + 1], fill)

    # Bright edges along the ribbon for definition.
    for i in range(len(spts) - 1):
      t_far = _clip01(0.5 * (sfwds[i] + sfwds[i + 1]) / MAX_RENDER_DISTANCE_M)
      alpha = int(_lerp(RIBBON_EDGE_ALPHA_NEAR, RIBBON_EDGE_ALPHA_FAR, t_far))
      edge = _rgba(BLUE, alpha)
      rl.draw_line_ex(left[i], left[i + 1], 3.0, edge)
      rl.draw_line_ex(right[i], right[i + 1], 3.0, edge)

  def _draw_dashed(self, pts: list[rl.Vector2], fwds: list[float]) -> None:
    spts, sfwds = _catmull_rom(pts, fwds, SMOOTH_SUBSAMPLES)
    if len(spts) < 2:
      return
    min_f, max_f = sfwds[0], sfwds[-1]
    if max_f - min_f < DASH_ON_M:
      return
    period = DASH_ON_M + DASH_OFF_M

    def screen_at(d: float, hint_idx: int) -> tuple[rl.Vector2 | None, int]:
      i = hint_idx
      while i < len(sfwds) - 1 and sfwds[i + 1] < d:
        i += 1
      if i >= len(sfwds) - 1:
        return None, i
      f0, f1 = sfwds[i], sfwds[i + 1]
      if f1 - f0 < 1e-3:
        return spts[i], i
      t = (d - f0) / (f1 - f0)
      return rl.Vector2(
        spts[i].x + t * (spts[i + 1].x - spts[i].x),
        spts[i].y + t * (spts[i + 1].y - spts[i].y),
      ), i

    d = min_f
    hint = 0
    while d + DASH_ON_M <= max_f:
      p1, hint = screen_at(d, hint)
      p2, hint = screen_at(d + DASH_ON_M, hint)
      if p1 is not None and p2 is not None:
        t = _clip01(d / MAX_RENDER_DISTANCE_M)
        w = _lerp(DASH_THICKNESS_NEAR, DASH_THICKNESS_FAR, t)
        a = int(_lerp(DASH_ALPHA_NEAR, DASH_ALPHA_FAR, t))
        rl.draw_line_ex(p1, p2, w + 3.0, rl.Color(0, 0, 0, max(0, a // 3)))
        rl.draw_line_ex(p1, p2, w, _rgba(BLUE, a))
      d += period

  def _draw_smoke(self, pts: list[rl.Vector2], fwds: list[float]) -> None:
    spts, sfwds = _catmull_rom(pts, fwds, SMOOTH_SUBSAMPLES)
    for width, base_alpha, jitter_x in SMOKE_LAYERS:
      for i in range(len(spts) - 1):
        t_far = _clip01(0.5 * (sfwds[i] + sfwds[i + 1]) / MAX_RENDER_DISTANCE_M)
        a = int(_lerp(base_alpha, max(8, base_alpha // 4), t_far))
        p1 = rl.Vector2(spts[i].x + jitter_x, spts[i].y)
        p2 = rl.Vector2(spts[i + 1].x + jitter_x, spts[i + 1].y)
        rl.draw_line_ex(p1, p2, width, _rgba(BLUE, a))

  # ---------- shared helpers ----------
  def _alpha_at(self, forward: float, near: int, far: int) -> int:
    t = _clip01(forward / MAX_RENDER_DISTANCE_M)
    return int(_lerp(near, far, t))

  def _draw_tapered(self, pts: list[rl.Vector2], fwds: list[float],
                    w_near: float, w_far: float, a_near: int, a_far: int,
                    with_outline: bool) -> None:
    """Per-segment width + alpha taper. Draws each tiny segment with its own
    width/alpha based on the forward distance at its midpoint."""
    for i in range(len(pts) - 1):
      f_mid = 0.5 * (fwds[i] + fwds[i + 1])
      t = _clip01(f_mid / MAX_RENDER_DISTANCE_M)
      w = _lerp(w_near, w_far, t)
      a = int(_lerp(a_near, a_far, t))
      if with_outline:
        rl.draw_line_ex(pts[i], pts[i + 1], w + 3.0,
                        rl.Color(0, 0, 0, max(0, a // 3)))
      rl.draw_line_ex(pts[i], pts[i + 1], w, _rgba(BLUE, a))
