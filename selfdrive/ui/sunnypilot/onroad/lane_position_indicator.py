"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.controls.lib.lane_position import LanePositionEstimator

# Square size / spacing for the [□□■□] indicator
_SQ = 36           # filled square side (px)
_SQ_GAP = 8        # gap between squares
_BORDER = 4        # outer border thickness
_INNER_PAD = 12    # padding between squares and border
_TOP_OFFSET = 30   # distance from top of the rect

_BG = rl.Color(0, 0, 0, 130)
_FILL_FG = rl.Color(255, 255, 255, 255)
_EMPTY_FG = rl.Color(255, 255, 255, 100)

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

    current, total, conf = self._estimator.update(sm['modelV2'])
    self._update_alpha(total > 0)
    if self._alpha <= 0.0 or total <= 0:
      return

    self._render(rect, current, total, conf)

  def _render(self, rect: rl.Rectangle, current: int, total: int, conf: str):
    a = self._alpha

    def fade(c: rl.Color) -> rl.Color:
      return rl.Color(c.r, c.g, c.b, int(c.a * a))

    squares_w = total * _SQ + (total - 1) * _SQ_GAP
    panel_w = squares_w + 2 * (_INNER_PAD + _BORDER)
    panel_h = _SQ + 2 * (_INNER_PAD + _BORDER)

    x = rect.x + (rect.width - panel_w) / 2
    y = rect.y + _TOP_OFFSET

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
      if (i + 1) == current:
        # filled square
        rl.draw_rectangle_rounded(rl.Rectangle(sx, sq_y, _SQ, _SQ), 0.2, 6, fade(_FILL_FG))
      else:
        # empty square (outline only)
        rl.draw_rectangle_rounded_lines_ex(
          rl.Rectangle(sx, sq_y, _SQ, _SQ), 0.2, 6, 2, fade(_EMPTY_FG)
        )
