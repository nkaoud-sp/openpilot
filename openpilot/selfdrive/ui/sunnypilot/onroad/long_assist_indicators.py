"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.onroad.hud_renderer import COLORS
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

# Launch Assist only fires for the moment the car pulls away, so hold the pill
# long enough to be noticed. Halt Assist is steady but can chatter at its fade edge.
LAUNCH_HOLD_TIME = 2.0
HALT_HOLD_TIME = 0.5
LAUNCH_COLOR = rl.Color(0, 200, 255, 255)
HALT_COLOR = rl.Color(255, 170, 0, 255)

# Mirrors the SCC pills on the right side, clear of the right turn signal
X_OFFSET = 320
LAUNCH_Y_OFFSET = -100
HALT_Y_OFFSET = -40
FONT_SIZE = 36
PADDING_V = 5
MIN_BOX_WIDTH = 160


class AssistPill:
  def __init__(self, label: str, color: rl.Color, hold_time: float, y_offset: int):
    self.label = label
    self.color = color
    self.hold_time = hold_time
    self.y_offset = y_offset
    self._last_active = float('-inf')
    self._alpha = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

  def update(self, active: bool, now: float):
    if active:
      self._last_active = now
    self._alpha.update(1.0 if now - self._last_active <= self.hold_time else 0.0)

  @property
  def alpha(self) -> float:
    return self._alpha.x


class LongAssistIndicatorsRenderer(Widget):
  def __init__(self):
    super().__init__()
    self.long_override = False
    self._launch = AssistPill("LAUNCH", LAUNCH_COLOR, LAUNCH_HOLD_TIME, LAUNCH_Y_OFFSET)
    self._halt = AssistPill("HALT", HALT_COLOR, HALT_HOLD_TIME, HALT_Y_OFFSET)
    self.font = gui_app.font(FontWeight.BOLD)

  def update(self):
    sm = ui_state.sm
    lp_sp = sm["longitudinalPlanSP"]
    if sm.updated["carControl"]:
      self.long_override = sm["carControl"].cruiseControl.override

    now = rl.get_time()
    self._launch.update(lp_sp.launchAssistActive, now)
    self._halt.update(lp_sp.haltAssistActive, now)

  def _draw_pill(self, rect: rl.Rectangle, pill: AssistPill):
    alpha = pill.alpha
    if alpha <= 0.01:
      return

    sz = measure_text_cached(self.font, pill.label, FONT_SIZE)
    box_width = max(MIN_BOX_WIDTH, sz.x + 30)
    box_height = sz.y + PADDING_V * 2

    base = COLORS.OVERRIDE if self.long_override else pill.color
    box_color = rl.Color(base.r, base.g, base.b, int(alpha * 255))
    text_color = rl.Color(0, 0, 0, int(alpha * 255))

    box_x = rect.x + rect.width / 2 + X_OFFSET - box_width / 2
    box_y = rect.height / 4 + pill.y_offset - box_height / 2

    rl.draw_rectangle_rounded(rl.Rectangle(box_x, box_y, box_width, box_height), 0.2, 10, box_color)
    rl.draw_text_ex(self.font, pill.label,
                    rl.Vector2(box_x + (box_width - sz.x) / 2, box_y + (box_height - sz.y) / 2),
                    FONT_SIZE, 0, text_color)

  def _render(self, rect: rl.Rectangle):
    self._draw_pill(rect, self._launch)
    self._draw_pill(rect, self._halt)
