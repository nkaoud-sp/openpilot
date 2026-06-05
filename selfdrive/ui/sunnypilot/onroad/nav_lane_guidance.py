"""
Always-visible lane guidance hint for nkaoud_nav.

Reads nkaoudNavigationSP.recommendedLaneSide + laneKeepDistance and shows a
small pill that says e.g. "USE RIGHT LANES -- 1.2 km" whenever the route
prefers a specific half of the road and we're still inside the lane-keep
distance window. Unlike NavManeuverBanner this isn't tied to the immediate
turn -- it's the preparation cue while approaching the maneuver.

Hidden when:
  - master toggle is off
  - nkaoudNavigationSP.active is false
  - recommendedLaneSide is "none"
  - distance to maneuver > laneKeepDistance (out of range)
"""
from __future__ import annotations

import time

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached


PILL_WIDTH = 540
PILL_HEIGHT = 84
TOP_OFFSET = 200             # below the maneuver banner
RADIUS = 0.28
BG_COLOR = rl.Color(0, 0, 0, 170)
BORDER_COLOR = rl.Color(255, 255, 255, 70)
TEXT_COLOR = rl.Color(255, 255, 255, 235)
DISTANCE_COLOR = rl.Color(128, 216, 166, 255)
PARAM_REFRESH_S = 0.5

LABEL_FONT_SIZE = 36
DISTANCE_FONT_SIZE = 42


def _format_distance(d_m: float, is_metric: bool) -> str:
  if is_metric:
    rounded = int(round(d_m / 10) * 10)
    if rounded < 1000:
      return f"{rounded} m"
    return f"{d_m / 1000.0:.1f} km"
  ft = d_m * 3.28084
  ft_rounded = int(round(ft / 10) * 10)
  if ft_rounded < 1000:
    return f"{ft_rounded} ft"
  mi = d_m * 0.000621371
  return f"{mi:.1f} mi"


class NavLaneGuidance:
  def __init__(self) -> None:
    self._enabled = False
    self._show_banner = False
    self._next_param_check = 0.0
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_medium = gui_app.font(FontWeight.MEDIUM)

  def _refresh_params(self) -> None:
    now = time.monotonic()
    if now < self._next_param_check:
      return
    self._next_param_check = now + PARAM_REFRESH_S
    self._enabled = ui_state.params.get_bool("NkaoudNavEnabled")
    # Reuse the banner toggle for now -- this widget is part of the same
    # navigation banner family.
    self._show_banner = ui_state.params.get_bool("NkaoudNavShowBanner")

  def render(self, rect: rl.Rectangle) -> None:
    self._refresh_params()
    if not (self._enabled and self._show_banner):
      return

    sm = ui_state.sm
    nav_sp = sm['nkaoudNavigationSP']
    if not nav_sp.active:
      return

    side = str(nav_sp.recommendedLaneSide)
    if side not in ("left", "right"):
      return
    dist_to_maneuver = float(nav_sp.distanceToManeuver)
    lane_keep_dist = float(nav_sp.laneKeepDistance)
    if dist_to_maneuver <= 0 or dist_to_maneuver > lane_keep_dist:
      return

    label = "USE LEFT LANES" if side == "left" else "USE RIGHT LANES"
    distance_str = _format_distance(dist_to_maneuver, ui_state.is_metric)

    bx = rect.x + (rect.width - PILL_WIDTH) / 2
    by = rect.y + TOP_OFFSET
    pill = rl.Rectangle(bx, by, PILL_WIDTH, PILL_HEIGHT)
    rl.draw_rectangle_rounded(pill, RADIUS, 10, BG_COLOR)
    rl.draw_rectangle_rounded_lines_ex(pill, RADIUS, 10, 2, BORDER_COLOR)

    cy = by + PILL_HEIGHT / 2
    cur_x = bx + 24
    label_size = measure_text_cached(self._font_bold, label, LABEL_FONT_SIZE)
    rl.draw_text_ex(self._font_bold, label,
                    rl.Vector2(cur_x, cy - label_size.y / 2),
                    LABEL_FONT_SIZE, 0, TEXT_COLOR)

    dist_size = measure_text_cached(self._font_bold, distance_str, DISTANCE_FONT_SIZE)
    rl.draw_text_ex(self._font_bold, distance_str,
                    rl.Vector2(bx + PILL_WIDTH - 24 - dist_size.x, cy - dist_size.y / 2),
                    DISTANCE_FONT_SIZE, 0, DISTANCE_COLOR)
