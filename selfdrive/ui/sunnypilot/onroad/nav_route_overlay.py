"""
Projects the active Mapbox route polyline (navRoute) onto the driving view.

Reuses the same car-space-to-screen transform that ModelRenderer uses, so the
polyline follows the road. Gated on NkaoudNavEnabled + NkaoudNavShowPolyline.

Device frame convention (per common/transformations/camera.py): x=forward,
y=right, z=down. Camera is mounted ~1.22 m above the road, so the road surface
sits at z = +live_calibration.height (positive = downward). Passing z = 0
would place the polyline at the camera's own height -- i.e. flat on the
horizon plane.

Smoothing:
  liveLocationKalman publishes at ~20 Hz, the UI renders at 60 Hz. Without
  filtering, the polyline snaps on every GPS tick. We low-pass the vehicle's
  (lat, lon, yaw) at render rate so each frame eases toward the latest GPS
  sample. Yaw is filtered via its sin/cos pair to handle wrap.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pyray as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app


MAX_RENDER_DISTANCE_M = 250.0
BEHIND_SLACK_M = 5.0
LINE_THICKNESS = 10.0
LINE_COLOR_PRIMARY = rl.Color(0, 191, 255, 220)   # deep sky blue
LINE_COLOR_OUTLINE = rl.Color(0, 0, 0, 140)
PARAM_REFRESH_S = 0.5
METERS_PER_DEG_LAT = 111320.0

# Small downward offset (m) so the line visually rests on the road instead
# of skimming a fingernail's width above it. Positive = further down in the
# device frame (z = down).
Z_GROUND_OFFSET_M = 0.10

# Smoothing time constants.
RC_POS = 0.10        # s, lat/lon low-pass (light smoothing -- car actually moves)
RC_YAW = 0.20        # s, heading low-pass (heavier -- yaw is the jitter culprit)


class NavRouteOverlay:
  def __init__(self) -> None:
    self._transform = np.eye(3, dtype=np.float32)
    self._enabled = False
    self._show_polyline = False
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

  def _smoothed_pose(self, lat: float, lon: float, yaw_rad: float) -> tuple[float, float, float]:
    if not self._filters_seeded:
      # snap to first sample so we don't slew from (0, 0) across the equator
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

    # Road surface is below the camera by the calibrated height. In device
    # frame z = down, so this is positive. Add a small offset so the line
    # rests on the road instead of floating a few cm above it.
    calib = sm['liveCalibration']
    z_ground = (float(calib.height[0]) if (calib.height and len(calib.height))
                else float(HEIGHT_INIT[0])) + Z_GROUND_OFFSET_M

    # Build a contiguous polyline of (forward, right) points, dropping segments
    # that are entirely behind the vehicle or beyond the render distance.
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for c in coords:
      dn = (c.latitude - lat0) * METERS_PER_DEG_LAT
      de = (c.longitude - lon0) * me_per_deg_lon
      forward = de * sin_yaw + dn * cos_yaw
      right = de * cos_yaw - dn * sin_yaw

      in_window = (-BEHIND_SLACK_M <= forward <= MAX_RENDER_DISTANCE_M)
      if in_window:
        current.append((forward, right))
      elif current:
        segments.append(current)
        current = []
    if current:
      segments.append(current)

    for seg in segments:
      self._draw_segment(seg, z_ground)

  def _draw_segment(self, seg: list[tuple[float, float]], z_ground: float) -> None:
    if len(seg) < 2:
      return
    pts: list[rl.Vector2] = []
    for forward, right in seg:
      p = self._transform @ np.array([forward, right, z_ground], dtype=np.float32)
      if abs(p[2]) < 1e-6:
        continue
      pts.append(rl.Vector2(float(p[0] / p[2]), float(p[1] / p[2])))
    if len(pts) < 2:
      return

    # Outline first (thicker) for contrast, then the bright line on top.
    for i in range(len(pts) - 1):
      rl.draw_line_ex(pts[i], pts[i + 1], LINE_THICKNESS + 4.0, LINE_COLOR_OUTLINE)
    for i in range(len(pts) - 1):
      rl.draw_line_ex(pts[i], pts[i + 1], LINE_THICKNESS, LINE_COLOR_PRIMARY)
