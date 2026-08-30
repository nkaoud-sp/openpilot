"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.common.params import Params
from openpilot.system.ui.widgets import Widget

# Momentary params the desire controllers read every cycle.
# Each stores 0 = none, 1 = left, 2 = right.
LANE_TURN_BUTTON_PARAM = "LaneTurnButtonDirection"
LANE_CHANGE_BUTTON_PARAM = "LaneChangeButtonDirection"

# Direction values written by the buttons (shared by both maneuver params).
DIR_NONE = 0
DIR_LEFT = 1
DIR_RIGHT = 2


class DesireButton(Widget):
  """On-road button that manually requests a maneuver (turn or lane change) while held.

  On press it writes ``direction`` to ``param_key``; on release it writes 0. The
  matching controller reads the param every cycle and feeds the desire to the model.
  """

  def __init__(self, param_key: str, direction: int, chevron_color: rl.Color, button_size: int):
    super().__init__()
    self._params = Params()
    self._param_key = param_key
    self._direction = direction
    self._points_left = direction == DIR_LEFT
    self._chevron_color = chevron_color

    self._black_bg: rl.Color = rl.Color(0, 0, 0, 166)
    self._pressed_bg: rl.Color = rl.Color(chevron_color.r, chevron_color.g, chevron_color.b, 120)
    self._rect = rl.Rectangle(0, 0, button_size, button_size)

  def _write_direction(self, direction: int) -> None:
    self._params.put(self._param_key, int(direction))

  def _handle_mouse_press(self, mouse_pos) -> None:
    self._write_direction(self._direction)

  def _handle_mouse_release(self, mouse_pos) -> None:
    super()._handle_mouse_release(mouse_pos)
    self._write_direction(DIR_NONE)

  def _render(self, rect: rl.Rectangle) -> None:
    center_x = int(rect.x + rect.width // 2)
    center_y = int(rect.y + rect.height // 2)
    radius = rect.width / 2

    rl.draw_circle(center_x, center_y, radius, self._pressed_bg if self.is_pressed else self._black_bg)

    a = radius * 0.42
    thickness = max(radius * 0.16, 4.0)

    # Chevron pointing in the button's direction (no triangle winding to worry about).
    tip_x = center_x - a if self._points_left else center_x + a
    base_x = center_x + a if self._points_left else center_x - a
    tip = rl.Vector2(tip_x, center_y)
    top = rl.Vector2(base_x, center_y - a)
    bottom = rl.Vector2(base_x, center_y + a)

    rl.draw_line_ex(top, tip, thickness, self._chevron_color)
    rl.draw_line_ex(tip, bottom, thickness, self._chevron_color)
