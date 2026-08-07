"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Debug overlay: draws the device-frame route path that NavPathAssist (mode B)
steers toward -- the exact navPathX/Y published by navd -- so you can see what
the assist is consuming and blending against. Deliberately a different colour
from the blue navRoute polyline: this is "what the assist sees / steers toward",
not "the route line on the map".

Unlike NavRouteOverlay, no GPS projection is needed -- navPath is already in the
device frame (x forward, y left) at the pose navd used. We only flip y (navd's
y-left vs this renderer's y-right convention) and project through the same
device->screen transform.

Gated on NkaoudNavEnabled + NkaoudNavShowPath, and only drawn while navPathValid.
"""
from __future__ import annotations

import time

import numpy as np
import pyray as rl
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.sunnypilot.onroad.nav_route_overlay import _clip01, _lerp, _rgba

MIN_FORWARD_M = 3.0
MAX_RENDER_DISTANCE_M = 120.0
MIN_PROJ_Z = 0.1
PARAM_REFRESH_S = 0.5
Z_GROUND_OFFSET_M = 0.10
LIFT_FAR_M = 1.0

# Amber, to contrast with the blue navRoute polyline.
AMBER = (255, 176, 0)
THICKNESS_NEAR = 14.0
THICKNESS_FAR = 4.0
ALPHA_NEAR = 235
ALPHA_FAR = 70
DOT_RADIUS = 5.0


class NavPathOverlay:
  def __init__(self) -> None:
    self._transform = np.eye(3, dtype=np.float32)
    self._enabled = False
    self._show = False
    self._next_param_check = 0.0

  def set_transform(self, m: np.ndarray) -> None:
    self._transform = m.astype(np.float32)

  def _refresh_params(self) -> None:
    now = time.monotonic()
    if now < self._next_param_check:
      return
    self._next_param_check = now + PARAM_REFRESH_S
    self._enabled = ui_state.params.get_bool("NkaoudNavEnabled")
    self._show = ui_state.params.get_bool("NkaoudNavShowPath")

  def render(self, rect: rl.Rectangle) -> None:
    self._refresh_params()
    if not (self._enabled and self._show):
      return

    sm = ui_state.sm
    nav = sm['nkaoudNavigationSP']
    if not nav.navPathValid:
      return
    xs = nav.navPathX
    ys = nav.navPathY
    n = min(len(xs), len(ys))
    if n < 2:
      return

    calib = sm['liveCalibration']
    z_ground = (float(calib.height[0]) if (calib.height and len(calib.height))
                else float(HEIGHT_INIT[0])) + Z_GROUND_OFFSET_M

    # Project device-frame points (x forward, y left -> right = -y) to screen.
    screen: list[rl.Vector2] = []
    fwds: list[float] = []
    for i in range(n):
      forward = float(xs[i])
      right = -float(ys[i])
      if not (MIN_FORWARD_M <= forward <= MAX_RENDER_DISTANCE_M):
        # break the line on out-of-window points
        self._draw(screen, fwds)
        screen, fwds = [], []
        continue
      lift = LIFT_FAR_M * _clip01(forward / MAX_RENDER_DISTANCE_M)
      p = self._transform @ np.array([forward, right, z_ground - lift], dtype=np.float32)
      if p[2] < MIN_PROJ_Z:
        self._draw(screen, fwds)
        screen, fwds = [], []
        continue
      screen.append(rl.Vector2(float(p[0] / p[2]), float(p[1] / p[2])))
      fwds.append(forward)
    self._draw(screen, fwds)

  def _draw(self, pts: list[rl.Vector2], fwds: list[float]) -> None:
    if len(pts) < 2:
      return
    for i in range(len(pts) - 1):
      t = _clip01(0.5 * (fwds[i] + fwds[i + 1]) / MAX_RENDER_DISTANCE_M)
      w = _lerp(THICKNESS_NEAR, THICKNESS_FAR, t)
      a = int(_lerp(ALPHA_NEAR, ALPHA_FAR, t))
      rl.draw_line_ex(pts[i], pts[i + 1], w + 3.0, rl.Color(0, 0, 0, max(0, a // 3)))
      rl.draw_line_ex(pts[i], pts[i + 1], w, _rgba(AMBER, a))
    for i in range(len(pts)):
      t = _clip01(fwds[i] / MAX_RENDER_DISTANCE_M)
      rl.draw_circle_v(pts[i], _lerp(DOT_RADIUS, 2.0, t), _rgba(AMBER, int(_lerp(ALPHA_NEAR, ALPHA_FAR, t))))
