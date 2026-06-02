"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Fixed standstill buffer baked into the planner's safe-distance equation
# (selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py:STOP_DISTANCE). The
# follow-time param does not include it, so the true target gap is
# tFollow + STOP_DISTANCE / v_ego. Kept as a local constant to avoid importing
# the (acados-heavy) long_mpc module into the UI process.
STOP_DISTANCE = 6.0
M_TO_FT = 3.28084

# Tolerance (s) around the true target follow time within which the actual gap
# is considered "on target" and drawn green. Closer than this -> amber/red.
_ON_TARGET_BAND = 0.15
_MIN_SPEED = 0.5  # m/s below which time gaps are undefined (division blows up)

_GREEN = rl.Color(0, 200, 90, 255)
_AMBER = rl.Color(255, 180, 0, 255)
_RED = rl.Color(255, 70, 70, 255)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(180, 180, 180, 255)


class FollowReadout:
  """On-screen readout comparing the planner's desired follow gap with the
  actual (measured) gap to the lead vehicle, in both time and distance."""

  def __init__(self):
    self._alpha: float = 0.0
    self._cap_font = gui_app.font(FontWeight.SEMI_BOLD)
    self._val_font = gui_app.font(FontWeight.BOLD)

  def _update_alpha(self, visible: bool):
    if visible:
      self._alpha = min(1.0, self._alpha + 0.1)
    else:
      self._alpha = max(0.0, self._alpha - 0.05)

  @staticmethod
  def _actual_color(actual: float, target: float) -> rl.Color:
    if actual >= target - _ON_TARGET_BAND:
      return _GREEN
    if actual >= target - 2 * _ON_TARGET_BAND:
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
    t_follow = float(sm['longitudinalPlanSP'].tFollow)
    d_rel = lead.dRel
    has_speed = v_ego > _MIN_SPEED

    # Time (s): raw param, true target (incl. stop buffer), and actual gap
    set_t = t_follow
    target_t = t_follow + STOP_DISTANCE / v_ego if has_speed else 0.0
    now_t = d_rel / v_ego if has_speed else 0.0

    # Distance: true target gap vs actual measured gap
    set_d = t_follow * v_ego + STOP_DISTANCE
    now_d = d_rel

    color = self._actual_color(now_t, target_t) if has_speed else _DIM

    metric = ui_state.is_metric
    du = "m" if metric else "ft"
    df = 1.0 if metric else M_TO_FT

    def ts(v):
      return f"{v:.2f}s" if has_speed else "--"

    # Two groups of cells: (group label, [(caption, value, color), ...])
    groups = [
      ("TIME", [
        ("SET", f"{set_t:.2f}s", _DIM),
        ("TARGET", ts(target_t), _WHITE),
        ("NOW", ts(now_t), color),
      ]),
      ("DIST", [
        ("SET", f"{set_d * df:.0f}{du}", _WHITE),
        ("NOW", f"{now_d * df:.0f}{du}", color),
      ]),
    ]
    self._render(rect, groups)

  def _render(self, rect: rl.Rectangle, groups):
    a = self._alpha
    cap_size = 24
    val_size = 40
    gl_size = 28          # group label
    pad = 20
    cell_gap = 26
    row_gap = 18
    cap_val_gap = 4

    def fade(c: rl.Color) -> rl.Color:
      return rl.Color(c.r, c.g, c.b, int(255 * a))

    # Uniform cell width across all cells for tidy columns
    cell_w = 0.0
    for _, cells in groups:
      for cap, val, _c in cells:
        cw = max(measure_text_cached(self._cap_font, cap, cap_size, 0).x,
                 measure_text_cached(self._val_font, val, val_size, 0).x)
        cell_w = max(cell_w, cw)

    gl_w = max(measure_text_cached(self._val_font, gl, gl_size, 0).x for gl, _ in groups)
    cell_h = cap_size + cap_val_gap + val_size
    row_h = cell_h

    max_cells = max(len(cells) for _, cells in groups)
    content_w = gl_w + cell_gap + max_cells * cell_w + (max_cells - 1) * cell_gap
    panel_w = pad + content_w + pad
    panel_h = pad + len(groups) * row_h + (len(groups) - 1) * row_gap + pad

    # Horizontally centred, top edge at the top of the bottom quarter of the screen
    x = rect.x + (rect.width - panel_w) / 2
    y = rect.y + rect.height * 0.75

    rl.draw_rectangle_rounded(rl.Rectangle(x, y, panel_w, panel_h), 0.18, 10, rl.Color(0, 0, 0, int(120 * a)))

    for r, (gl, cells) in enumerate(groups):
      row_y = y + pad + r * (row_h + row_gap)

      # group label, vertically centred in the row
      gl_y = row_y + (row_h - gl_size) / 2
      rl.draw_text_ex(self._val_font, gl, rl.Vector2(int(x + pad), int(gl_y)), gl_size, 0, fade(_DIM))

      cells_x = x + pad + gl_w + cell_gap
      for c, (cap, val, color) in enumerate(cells):
        cx = cells_x + c * (cell_w + cell_gap)
        # caption centred over the value
        cap_w = measure_text_cached(self._cap_font, cap, cap_size, 0).x
        rl.draw_text_ex(self._cap_font, cap, rl.Vector2(int(cx + (cell_w - cap_w) / 2), int(row_y)),
                        cap_size, 0, fade(_DIM))
        val_w = measure_text_cached(self._val_font, val, val_size, 0).x
        rl.draw_text_ex(self._val_font, val, rl.Vector2(int(cx + (cell_w - val_w) / 2), int(row_y + cap_size + cap_val_gap)),
                        val_size, 0, fade(color))
