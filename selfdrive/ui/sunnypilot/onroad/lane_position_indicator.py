"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.controls.lib.lane_position import LanePositionEstimator

# Square size / spacing for the [□□■□] indicator
_SQ = 27           # filled square side (px)
_SQ_GAP = 6        # gap between squares
_BORDER = 3        # outer border thickness
_INNER_PAD = 9     # padding between squares and border
# Sit just above the follow readout (which anchors its top edge at 0.75 * rect.height)
_BOTTOM_ANCHOR = 0.75
_BOTTOM_MARGIN = 16  # gap (px) between this widget's bottom and the follow readout's top

_BG = rl.Color(0, 0, 0, 130)
_FILL_FG = rl.Color(255, 255, 255, 255)
_EMPTY_FG = rl.Color(255, 255, 255, 100)
_BLOCK_FG = rl.Color(255, 110, 70, 235)  # diagonal strike on filter-blocked edge lanes

# Border colour by confidence
_CONF_COLORS = {
  "high": rl.Color(0, 200, 90, 255),    # green
  "medium": rl.Color(255, 180, 0, 255), # amber
  "low": rl.Color(255, 70, 70, 255),    # red
  "unknown": rl.Color(150, 150, 150, 255),
}


class LanePositionIndicator:
  """Small overlay button showing total lanes detected (boxes) with the
  current lane filled. Border colour reflects confidence."""

  def __init__(self):
    self._alpha: float = 0.0
    self._estimator = LanePositionEstimator()

  def _update_alpha(self, visible: bool):
    if visible:
      self._alpha = min(1.0, self._alpha + 0.1)
    else:
      self._alpha = max(0.0, self._alpha - 0.05)

  def draw(self, sm, rect: rl.Rectangle):
    if not ui_state.lane_position_indicator:
      self._alpha = 0.0
      return

    filter_mode = int(ui_state.lane_edge_filter_mode or 0)
    current, total, conf = self._estimator.update(sm['modelV2'], filter_mode=filter_mode)
    self._update_alpha(total > 0)
    if self._alpha <= 0.0 or total <= 0:
      return

    debug = self._estimator.debug
    # Render the raw lane grid so a demoted edge stays visible (struck through);
    # when no edge is blocked, raw_* == usable_* and the widget is identical to base.
    display_total = max(debug.raw_total_lanes, total)
    display_current = debug.raw_current_lane if debug.raw_current_lane > 0 else current
    self._render(rect, display_current, display_total, conf,
                 debug.blocked_left, debug.blocked_right)

  def _render(self, rect: rl.Rectangle, current: int, total: int, conf: str,
              blocked_left: bool = False, blocked_right: bool = False):
    a = self._alpha

    def fade(c: rl.Color) -> rl.Color:
      return rl.Color(c.r, c.g, c.b, int(c.a * a))

    squares_w = total * _SQ + (total - 1) * _SQ_GAP
    panel_w = squares_w + 2 * (_INNER_PAD + _BORDER)
    panel_h = _SQ + 2 * (_INNER_PAD + _BORDER)

    x = rect.x + (rect.width - panel_w) / 2
    y = rect.y + rect.height * _BOTTOM_ANCHOR - _BOTTOM_MARGIN - panel_h

    # Background
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, panel_w, panel_h), 0.25, 8, fade(_BG))

    # Border (confidence colour) — drawn as a rounded outline
    border_color = _CONF_COLORS.get(conf, _CONF_COLORS["unknown"])
    rl.draw_rectangle_rounded_lines_ex(
      rl.Rectangle(x, y, panel_w, panel_h), 0.25, 8, _BORDER, fade(border_color)
    )

    # Squares
    sq_y = y + _BORDER + _INNER_PAD
    sq_x0 = x + _BORDER + _INNER_PAD
    for i in range(total):
      sx = sq_x0 + i * (_SQ + _SQ_GAP)
      is_blocked = (i == 0 and blocked_left) or (i == total - 1 and blocked_right)
      if (i + 1) == current:
        # filled square
        rl.draw_rectangle_rounded(rl.Rectangle(sx, sq_y, _SQ, _SQ), 0.2, 6, fade(_FILL_FG))
      else:
        # empty square (outline only)
        outline = _BLOCK_FG if is_blocked else _EMPTY_FG
        rl.draw_rectangle_rounded_lines_ex(
          rl.Rectangle(sx, sq_y, _SQ, _SQ), 0.2, 6, 2, fade(outline)
        )
      if is_blocked:
        rl.draw_line_ex(
          rl.Vector2(sx + 4, sq_y + _SQ - 4),
          rl.Vector2(sx + _SQ - 4, sq_y + 4),
          3, fade(_BLOCK_FG),
        )
