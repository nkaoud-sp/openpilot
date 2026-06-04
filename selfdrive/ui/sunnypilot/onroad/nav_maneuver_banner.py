"""
Onroad maneuver banner for nkaoud_nav.

Reads navInstruction (published by nkaoud_navd) and draws a small pill at the
top-center of the driving view showing the upcoming maneuver.

Layout: [arrow]   [distance]   [primary text]

Gated on:
- NkaoudNavEnabled  (master toggle)
- NkaoudNavShowBanner (per-feature toggle)
- nkaoudNavigationSP.active (navd has a fetched route)
"""
from __future__ import annotations

import time

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached


# Map cereal maneuverModifier values (strings) -> Unicode arrow glyphs.
# Fallback "↑" covers unknown / straight / arrive / continue.
ARROWS = {
  "left":        "←",   # ←
  "right":       "→",   # →
  "straight":    "↑",   # ↑
  "slightLeft":  "↖",   # ↖
  "slightRight": "↗",   # ↗
  "uturn":       "↶",   # ↶
  "sharpLeft":   "↙",   # ↙
  "sharpRight":  "↘",   # ↘
}

BANNER_WIDTH = 900
BANNER_HEIGHT = 110
BANNER_TOP_OFFSET = 60     # px below the top of the camera content rect
BANNER_RADIUS = 0.25
BG_COLOR = rl.Color(0, 0, 0, 180)
BORDER_COLOR = rl.Color(255, 255, 255, 80)
TEXT_COLOR = rl.Color(255, 255, 255, 240)
DISTANCE_COLOR = rl.Color(128, 216, 166, 255)  # green like ENGAGED
PARAM_REFRESH_S = 0.5

ARROW_FONT_SIZE = 80
DISTANCE_FONT_SIZE = 56
PRIMARY_FONT_SIZE = 46


def _format_distance(d_m: float, is_metric: bool) -> str:
  if is_metric:
    rounded = int(round(d_m / 10) * 10)
    if rounded < 1000:
      return f"{rounded} m"
    return f"{d_m / 1000.0:.1f} km"
  # imperial
  ft = d_m * 3.28084
  ft_rounded = int(round(ft / 10) * 10)
  if ft_rounded < 1000:
    return f"{ft_rounded} ft"
  mi = d_m * 0.000621371
  return f"{mi:.1f} mi"


class NavManeuverBanner:
  def __init__(self) -> None:
    self._enabled = False
    self._show_banner = False
    self._next_param_check = 0.0
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)

  def _refresh_params(self) -> None:
    now = time.monotonic()
    if now < self._next_param_check:
      return
    self._next_param_check = now + PARAM_REFRESH_S
    self._enabled = ui_state.params.get_bool("NkaoudNavEnabled")
    self._show_banner = ui_state.params.get_bool("NkaoudNavShowBanner")

  def render(self, rect: rl.Rectangle) -> None:
    self._refresh_params()
    if not (self._enabled and self._show_banner):
      return

    sm = ui_state.sm
    nav_sp = sm['nkaoudNavigationSP']
    if not nav_sp.active:
      return

    inst = sm['navInstruction']
    primary = inst.maneuverPrimaryText
    modifier = inst.maneuverModifier
    dist_m = inst.maneuverDistance
    if not primary and dist_m <= 0.0:
      return  # nothing meaningful to show

    arrow = ARROWS.get(modifier, ARROWS["straight"])
    distance_str = _format_distance(dist_m, ui_state.is_metric) if dist_m > 0 else ""

    bx = rect.x + (rect.width - BANNER_WIDTH) / 2
    by = rect.y + BANNER_TOP_OFFSET
    banner_rect = rl.Rectangle(bx, by, BANNER_WIDTH, BANNER_HEIGHT)
    rl.draw_rectangle_rounded(banner_rect, BANNER_RADIUS, 12, BG_COLOR)
    rl.draw_rectangle_rounded_lines_ex(banner_rect, BANNER_RADIUS, 12, 3, BORDER_COLOR)

    cur_x = bx + 30
    cy = by + BANNER_HEIGHT / 2

    # Arrow
    arrow_size = measure_text_cached(self._font_bold, arrow, ARROW_FONT_SIZE)
    rl.draw_text_ex(self._font_bold, arrow,
                    rl.Vector2(cur_x, cy - arrow_size.y / 2),
                    ARROW_FONT_SIZE, 0, TEXT_COLOR)
    cur_x += arrow_size.x + 30

    # Distance
    if distance_str:
      dist_size = measure_text_cached(self._font_bold, distance_str, DISTANCE_FONT_SIZE)
      rl.draw_text_ex(self._font_bold, distance_str,
                      rl.Vector2(cur_x, cy - dist_size.y / 2),
                      DISTANCE_FONT_SIZE, 0, DISTANCE_COLOR)
      cur_x += dist_size.x + 30

    # Primary text (elide to fit remaining width)
    if primary:
      max_text_w = banner_rect.x + banner_rect.width - 30 - cur_x
      text = primary
      text_size = measure_text_cached(self._font_semi, text, PRIMARY_FONT_SIZE)
      if text_size.x > max_text_w and len(text) > 3:
        while len(text) > 3 and measure_text_cached(self._font_semi, text + "...", PRIMARY_FONT_SIZE).x > max_text_w:
          text = text[:-1]
        text = text + "..."
        text_size = measure_text_cached(self._font_semi, text, PRIMARY_FONT_SIZE)
      rl.draw_text_ex(self._font_semi, text,
                      rl.Vector2(cur_x, cy - text_size.y / 2),
                      PRIMARY_FONT_SIZE, 0, TEXT_COLOR)
