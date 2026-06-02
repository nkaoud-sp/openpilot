"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Tolerance (s) around the desired follow time within which the actual gap is
# considered "on target" and drawn green. Closer than this -> amber/red.
_ON_TARGET_BAND = 0.15

_GREEN = rl.Color(0, 200, 90, 255)
_AMBER = rl.Color(255, 180, 0, 255)
_RED = rl.Color(255, 70, 70, 255)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(180, 180, 180, 255)


class FollowReadout:
  """On-screen readout comparing the planner's desired follow time with the
  actual (measured) gap to the lead vehicle."""

  def __init__(self):
    self._alpha: float = 0.0
    self._font = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_bold = gui_app.font(FontWeight.BOLD)

  def _update_alpha(self, visible: bool):
    if visible:
      self._alpha = min(1.0, self._alpha + 0.1)
    else:
      self._alpha = max(0.0, self._alpha - 0.05)

  @staticmethod
  def _actual_color(actual: float, desired: float) -> rl.Color:
    if actual >= desired - _ON_TARGET_BAND:
      return _GREEN
    if actual >= desired - 2 * _ON_TARGET_BAND:
      return _AMBER
    return _RED

  def draw(self, sm, radar_state, rect: rl.Rectangle):
    if not ui_state.follow_readout:
      self._alpha = 0.0
      return

    lead = radar_state.leadOne if radar_state else None
    has_lead = bool(lead.status) if lead else False
    self._update_alpha(has_lead)
    if self._alpha <= 0.0 or not has_lead:
      return

    v_ego = sm['carState'].vEgo
    desired_t = float(sm['longitudinalPlanSP'].tFollow)
    actual_t = (lead.dRel / v_ego) if v_ego > 0.5 else 0.0

    desired_str = f"{desired_t:.2f}s"
    actual_str = f"{actual_t:.2f}s" if v_ego > 0.5 else "--"
    actual_color = self._actual_color(actual_t, desired_t) if v_ego > 0.5 else _DIM

    self._render(rect, desired_str, actual_str, actual_color)

  def _render(self, rect: rl.Rectangle, desired_str: str, actual_str: str, actual_color: rl.Color):
    a = self._alpha
    label_size = 30
    value_size = 44
    pad = 18
    row_h = 52

    rows = [
      ("SET", desired_str, _WHITE),
      ("NOW", actual_str, actual_color),
    ]

    # Widest content determines the panel width
    label_w = max(measure_text_cached(self._font, lbl, label_size, 0).x for lbl, _, _ in rows)
    value_w = max(measure_text_cached(self._font_bold, val, value_size, 0).x for _, val, _ in rows)
    panel_w = pad + label_w + 16 + value_w + pad
    panel_h = pad + len(rows) * row_h

    # Horizontally centred, top edge at the top of the bottom quarter of the screen
    x = rect.x + (rect.width - panel_w) / 2
    y = rect.y + rect.height * 0.75

    bg = rl.Color(0, 0, 0, int(110 * a))
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, panel_w, panel_h), 0.25, 10, bg)

    for i, (label, value, color) in enumerate(rows):
      ry = y + pad + i * row_h
      lbl_color = rl.Color(_DIM.r, _DIM.g, _DIM.b, int(255 * a))
      val_color = rl.Color(color.r, color.g, color.b, int(255 * a))

      lbl_y = ry + (value_size - label_size) / 2
      rl.draw_text_ex(self._font, label, rl.Vector2(int(x + pad), int(lbl_y)), label_size, 0, lbl_color)
      vx = x + pad + label_w + 16
      rl.draw_text_ex(self._font_bold, value, rl.Vector2(int(vx), int(ry)), value_size, 0, val_color)
