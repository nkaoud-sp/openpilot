"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.system.ui.widgets import Widget

TurnDirection = custom.ModelDataV2SP.TurnDirection

# Momentary param the LaneTurnController reads every cycle.
# 0 = none, 1 = turnLeft, 2 = turnRight (matches custom.ModelDataV2SP.TurnDirection).
LANE_TURN_BUTTON_PARAM = "LaneTurnButtonDirection"


class LaneTurnButton(Widget):
  """On-road button that manually requests a turn desire in a single direction while held."""

  def __init__(self, direction: int, button_size: int):
    super().__init__()
    self._params = Params()
    self._direction = direction
    self._points_left = direction == TurnDirection.turnLeft

    self._black_bg: rl.Color = rl.Color(0, 0, 0, 166)
    self._white: rl.Color = rl.Color(255, 255, 255, 255)
    self._active: rl.Color = rl.Color(0x2c, 0xb7, 0xf7, 255)  # blue while pressed
    self._rect = rl.Rectangle(0, 0, button_size, button_size)

  def _write_direction(self, direction: int) -> None:
    self._params.put(LANE_TURN_BUTTON_PARAM, int(direction))

  def _handle_mouse_press(self, mouse_pos) -> None:
    self._write_direction(self._direction)

  def _handle_mouse_release(self, mouse_pos) -> None:
    super()._handle_mouse_release(mouse_pos)
    self._write_direction(TurnDirection.none)

  def _render(self, rect: rl.Rectangle) -> None:
    center_x = int(rect.x + rect.width // 2)
    center_y = int(rect.y + rect.height // 2)
    radius = rect.width / 2

    rl.draw_circle(center_x, center_y, radius, self._black_bg)

    color = self._active if self.is_pressed else self._white
    a = radius * 0.42
    thickness = max(radius * 0.16, 4.0)

    # Chevron pointing in the button's direction (no triangle winding to worry about).
    tip_x = center_x - a if self._points_left else center_x + a
    base_x = center_x + a if self._points_left else center_x - a
    tip = rl.Vector2(tip_x, center_y)
    top = rl.Vector2(base_x, center_y - a)
    bottom = rl.Vector2(base_x, center_y + a)

    rl.draw_line_ex(top, tip, thickness, color)
    rl.draw_line_ex(tip, bottom, thickness, color)
