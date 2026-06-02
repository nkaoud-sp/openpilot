"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Eagerness -> required lead speed (m/s), mirrors longitudinal_planner LAUNCH_VLEAD_*
_EAGER_MIN, _EAGER_MAX = 1, 10
_VLEAD_AT_MIN, _VLEAD_AT_MAX = 1.5, 0.2
_SHOW_BELOW_SPEED = 2.0  # m/s; only show the readout near a stop / during launch

_GO_C = rl.Color(0, 200, 90, 255)
_ARMED_C = rl.Color(255, 180, 0, 255)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(180, 180, 180, 255)


def _trigger_speed(eagerness: float) -> float:
  e = max(_EAGER_MIN, min(_EAGER_MAX, eagerness))
  frac = (e - _EAGER_MIN) / (_EAGER_MAX - _EAGER_MIN)
  return _VLEAD_AT_MIN + frac * (_VLEAD_AT_MAX - _VLEAD_AT_MIN)


class LaunchReadout:
  """On-screen readout for the lead-departure launch assist: whether it is armed
  / firing, the lead's speed, and the speed the lead must reach to trigger."""

  def __init__(self):
    self._alpha: float = 0.0
    self._cap_font = gui_app.font(FontWeight.SEMI_BOLD)
    self._val_font = gui_app.font(FontWeight.BOLD)

  def _update_alpha(self, visible: bool):
    if visible:
      self._alpha = min(1.0, self._alpha + 0.1)
    else:
      self._alpha = max(0.0, self._alpha - 0.05)

  def draw(self, sm, rect: rl.Rectangle):
    v_ego = sm['carState'].vEgo
    active = bool(sm['longitudinalPlanSP'].launchAssistActive)
    visible = ui_state.launch_readout and ui_state.has_longitudinal_control and (active or v_ego < _SHOW_BELOW_SPEED)
    self._update_alpha(visible)
    if self._alpha <= 0.0 or not visible:
      return

    lead = sm['radarState'].leadOne
    has_lead = bool(lead.status)
    v_lead = lead.vLead if has_lead else 0.0
    stopped = v_ego < 0.5

    if not ui_state.launch_assist:
      state, state_c = "OFF", _DIM
    elif active:
      state, state_c = "GO", _GO_C
    elif stopped and has_lead:
      state, state_c = "ARMED", _ARMED_C
    else:
      state, state_c = "WAIT", _DIM

    thresh = _trigger_speed(ui_state.launch_eagerness)
    cells = [
      ("STATE", state, state_c),
      ("LEAD m/s", f"{v_lead:+.2f}" if has_lead else "--", _WHITE),
      ("TRIG m/s", f"{thresh:.2f}", _DIM),
    ]
    self._render(rect, cells)

  def _render(self, rect: rl.Rectangle, cells):
    a = self._alpha
    cap_size = 24
    val_size = 40
    pad = 20
    cell_gap = 26
    cap_val_gap = 4

    def fade(c: rl.Color) -> rl.Color:
      return rl.Color(c.r, c.g, c.b, int(255 * a))

    cell_w = 0.0
    for cap, val, _c in cells:
      cw = max(measure_text_cached(self._cap_font, cap, cap_size, 0).x,
               measure_text_cached(self._val_font, val, val_size, 0).x)
      cell_w = max(cell_w, cw)

    row_h = cap_size + cap_val_gap + val_size
    content_w = len(cells) * cell_w + (len(cells) - 1) * cell_gap
    panel_w = pad + content_w + pad
    panel_h = pad + row_h + pad

    # Right-aligned, in the bottom-quarter band (jerk readout sits on the left)
    x = rect.x + rect.width - panel_w - 60
    y = rect.y + rect.height * 0.75

    rl.draw_rectangle_rounded(rl.Rectangle(x, y, panel_w, panel_h), 0.18, 10, rl.Color(0, 0, 0, int(120 * a)))

    row_y = y + pad
    for c, (cap, val, color) in enumerate(cells):
      cx = x + pad + c * (cell_w + cell_gap)
      cap_w = measure_text_cached(self._cap_font, cap, cap_size, 0).x
      rl.draw_text_ex(self._cap_font, cap, rl.Vector2(int(cx + (cell_w - cap_w) / 2), int(row_y)),
                      cap_size, 0, fade(_DIM))
      val_w = measure_text_cached(self._val_font, val, val_size, 0).x
      rl.draw_text_ex(self._val_font, val, rl.Vector2(int(cx + (cell_w - val_w) / 2), int(row_y + cap_size + cap_val_gap)),
                      val_size, 0, fade(color))
