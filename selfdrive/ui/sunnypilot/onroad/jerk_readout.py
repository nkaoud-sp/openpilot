"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

_ACCEL_DEADBAND = 0.05  # m/s^2 around zero treated as neither accel nor decel

_ACCEL_C = rl.Color(60, 200, 255, 255)   # accelerating
_DECEL_C = rl.Color(255, 150, 60, 255)   # braking
_SMOOTH_C = rl.Color(0, 200, 90, 255)    # factor > 1 (gentler)
_SNAPPY_C = rl.Color(255, 180, 0, 255)   # factor < 1 (snappier)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(180, 180, 180, 255)


class JerkReadout:
  """On-screen readout of the asymmetric accel/decel jerk smoothing: the current
  mode, commanded vs actual acceleration, the planned jerk (which the smoothness
  factor shapes), and the active factor."""

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
    visible = ui_state.jerk_readout and ui_state.has_longitudinal_control
    self._update_alpha(visible)
    if self._alpha <= 0.0 or not visible:
      return

    a_ego = sm['carState'].aEgo
    lp = sm['longitudinalPlan']
    a_target = float(lp.aTarget)
    jerks = lp.jerks
    jerk = float(jerks[0]) if len(jerks) else 0.0

    # Mode / active factor follow the planner's commanded acceleration (aTarget),
    # which is what the MPC's per-stage jerk weighting is gated on.
    decel = a_target < -_ACCEL_DEADBAND
    accel = a_target > _ACCEL_DEADBAND
    if decel:
      mode, mode_c = "DECEL", _DECEL_C
    elif accel:
      mode, mode_c = "ACCEL", _ACCEL_C
    else:
      mode, mode_c = "HOLD", _DIM

    if ui_state.asymmetric_jerk:
      factor = ui_state.jerk_factor_decel if decel else ui_state.jerk_factor_accel
    else:
      factor = 1.0
    factor_c = _SMOOTH_C if factor > 1.0 else (_SNAPPY_C if factor < 1.0 else _WHITE)

    cells = [
      ("MODE", mode, mode_c),
      ("CMD m/s²", f"{a_target:+.2f}", _WHITE),
      ("RATE m/s²", f"{a_ego:+.2f}", _DIM),
      ("JERK m/s³", f"{jerk:+.2f}", mode_c),
      ("SMOOTH", f"{factor:.2f}x", factor_c),
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

    # Left-aligned, in the bottom-quarter band (follow readout sits centred)
    x = rect.x + 60
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
